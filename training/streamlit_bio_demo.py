"""
Streamlit Demo for BIO Tagger - Subject-Predicate-Object Extraction

Interactive demo for testing the trained multi-class BERT BIO tagger.
Allows users to input text and see predicted S-P-O entities.
Includes eval set iteration with Box-Cox resampling results.
"""

import streamlit as st
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import json
import msgpack
from pathlib import Path
import numpy as np
from sklearn.metrics import classification_report, precision_recall_fscore_support
from itertools import product as cartesian_product
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
import io
from collections import defaultdict

matplotlib.use('Agg')  # non-interactive backend for Streamlit

# WordNet for synset/hypernym resolution
try:
    from nltk.corpus import wordnet as wn
    _WN_AVAILABLE = True
except ImportError:
    _WN_AVAILABLE = False

# Import stopword cleaning from inference module
try:
    from inference_bio_tagger import clean_span_tokens, SPAN_STOPWORDS
except ImportError:
    SPAN_STOPWORDS = {
        'a', 'an', 'the', 'to', 'of', 'in', 'on', 'at', 'by', 'for', 'with',
        'from', 'into', 'through', 'and', 'or', 'but', 'nor',
        'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'that', 'which', 'who', 'this', 'these', 'those', 'it', 'its',
    }
    def clean_span_tokens(tokens, label_type):
        cleaned = [t for t in tokens if t.lower() not in SPAN_STOPWORDS]
        return cleaned

# BIO label mapping
# When I-PRED is removed, labels are RE-INDEXED to 0-4
LABEL_MAP = {
    0: 'B-SUBJ',
    1: 'I-SUBJ',
    2: 'B-PRED',
    3: 'B-OBJ',   # Re-indexed from 4
    4: 'I-OBJ'    # Re-indexed from 5
}

# Full 6-class version (if I-PRED exists)
LABEL_MAP_FULL = {
    0: 'B-SUBJ',
    1: 'I-SUBJ',
    2: 'B-PRED',
    3: 'I-PRED',
    4: 'B-OBJ',
    5: 'I-OBJ'
}

class BIOTaggerMultiClass(nn.Module):
    """Multi-label BIO tagger with independent binary classifiers (matches tune_bio_tagger.py)"""
    
    def __init__(self, num_labels=5, dropout=0.1, unfreeze_layers=2):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        hidden_size = self.bert.config.hidden_size
        total_layers = len(self.bert.encoder.layer)
        self.num_labels = num_labels
        
        # Freeze all BERT layers initially
        for param in self.bert.parameters():
            param.requires_grad = False
        
        # Unfreeze top N layers if specified
        if unfreeze_layers == -1 or unfreeze_layers >= total_layers:
            # Unfreeze ALL layers
            for param in self.bert.parameters():
                param.requires_grad = True
        elif unfreeze_layers > 0:
            # Unfreeze top N layers
            for layer in self.bert.encoder.layer[-unfreeze_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
        
        self.dropout = nn.Dropout(dropout)
        
        # Independent binary classifiers (one per label)
        self.classifiers = nn.ModuleList([
            nn.Linear(hidden_size, 1) for _ in range(num_labels)
        ])
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        
        logits = []
        for classifier in self.classifiers:
            logit = classifier(sequence_output).squeeze(-1)
            logits.append(logit)
        
        logits = torch.stack(logits, dim=-1)
        probs = self.sigmoid(logits)
        
        return probs


def extract_entities(tokens, labels):
    """Extract Subject, Predicate, Object entities from BIO labels.
    
    Applies post-inference stopword cleaning to each span.
    Returns dict with 'SUBJ', 'PRED', 'OBJ' keys, each a list of span strings.
    Also stores individual cleaned tokens for cartesian expansion.
    """
    entities = {'SUBJ': [], 'PRED': [], 'OBJ': []}
    # Also store individual cleaned word lists for cartesian product
    entities_words = {'SUBJ': [], 'PRED': [], 'OBJ': []}
    
    current_entity = None
    current_type = None
    
    def _save_span(span_tokens, span_type):
        """Clean and save a completed span."""
        cleaned = clean_span_tokens(span_tokens, span_type)
        if cleaned:
            entities[span_type].append(' '.join(cleaned))
            entities_words[span_type].append(cleaned)
    
    for token, label in zip(tokens, labels):
        if token in ['[CLS]', '[SEP]', '[PAD]']:
            continue
        
        if label.startswith('B-'):
            # Save previous entity
            if current_entity:
                _save_span(current_entity, current_type)
            
            # Start new entity
            current_type = label.split('-')[1]
            current_entity = [token]
        
        elif label.startswith('I-'):
            entity_type = label.split('-')[1]
            if current_entity and current_type == entity_type:
                current_entity.append(token)
            else:
                # Orphaned I- tag, treat as new entity
                if current_entity:
                    _save_span(current_entity, current_type)
                current_type = entity_type
                current_entity = [token]
        
        else:  # 'O' label
            if current_entity:
                _save_span(current_entity, current_type)
                current_entity = None
                current_type = None
    
    # Don't forget last entity
    if current_entity:
        _save_span(current_entity, current_type)
    
    return entities, entities_words


def expand_cartesian_triplets(entities_words):
    """Expand cleaned span words into cartesian-product triplets.
    
    Given: SUBJ=[['humans']], PRED=[['possess']], OBJ=[['extraordinary','ability','create','utilize','tools']]
    Produces: [(s, p, o) for each combo of individual words across all spans]
    """
    subj_words = sorted({w for span in entities_words.get('SUBJ', []) for w in span})
    pred_words = sorted({w for span in entities_words.get('PRED', []) for w in span})
    obj_words = sorted({w for span in entities_words.get('OBJ', []) for w in span})
    
    if not subj_words or not pred_words or not obj_words:
        return [], subj_words, pred_words, obj_words
    
    triplets = list(cartesian_product(subj_words, pred_words, obj_words))
    return triplets, subj_words, pred_words, obj_words


def render_triplet_graph(subj_words, pred_words, obj_words):
    """Render a networkx graph showing SUBJ -[PRED]-> OBJ relations.
    
    Subjects and objects are nodes; predicates are edge labels.
    Returns a matplotlib figure.
    """
    G = nx.DiGraph()
    
    # Add nodes with types
    for s in subj_words:
        G.add_node(s, node_type='SUBJ')
    for o in obj_words:
        G.add_node(o, node_type='OBJ')
    
    # Add edges: every subject connects to every object via every predicate
    for s, p, o in cartesian_product(subj_words, pred_words, obj_words):
        if G.has_edge(s, o):
            # Append predicate label if multiple predicates
            existing = G[s][o].get('label', '')
            if p not in existing:
                G[s][o]['label'] = f"{existing}, {p}"
        else:
            G.add_edge(s, o, label=p)
    
    if len(G.nodes) == 0:
        return None
    
    # Layout
    fig, ax = plt.subplots(1, 1, figsize=(max(8, len(G.nodes) * 1.2), max(5, len(G.nodes) * 0.8)))
    
    # Use shell layout: subjects on left, objects on right
    subj_nodes = [n for n in G.nodes if G.nodes[n].get('node_type') == 'SUBJ']
    obj_nodes = [n for n in G.nodes if G.nodes[n].get('node_type') == 'OBJ']
    
    pos = {}
    # Place subjects on left column
    for i, s in enumerate(subj_nodes):
        pos[s] = (-1.5, -i * 1.2)
    # Place objects on right column
    for i, o in enumerate(obj_nodes):
        pos[o] = (1.5, -i * 1.2)
    
    # Center vertically
    all_y = [v[1] for v in pos.values()]
    if all_y:
        y_mid = (min(all_y) + max(all_y)) / 2
        pos = {k: (v[0], v[1] - y_mid) for k, v in pos.items()}
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, nodelist=subj_nodes,
                           node_color='#ff9999', node_size=2000,
                           node_shape='o', alpha=0.9, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=obj_nodes,
                           node_color='#9999ff', node_size=2000,
                           node_shape='o', alpha=0.9, ax=ax)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color='#66cc66', width=2,
                           arrows=True, arrowsize=20,
                           connectionstyle='arc3,rad=0.05', ax=ax)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=11, font_weight='bold', ax=ax)
    
    # Edge labels (predicate names)
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                  font_size=10, font_color='#228B22',
                                  label_pos=0.5, ax=ax)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#ff9999', label='Subject'),
        Patch(facecolor='#66cc66', label='Predicate (edge)'),
        Patch(facecolor='#9999ff', label='Object'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    ax.set_title('SPO Relation Graph (Cartesian Expansion)', fontsize=13, fontweight='bold')
    ax.axis('off')
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
#  Synset / Hypernym Resolution & Collapsed Graph
# ---------------------------------------------------------------------------

def _wn_pos_hint(role):
    """Map SPO role to likely WordNet POS."""
    return {'SUBJ': wn.NOUN, 'PRED': wn.VERB, 'OBJ': wn.NOUN}.get(role)


def resolve_synset_hypernym(word, role='OBJ'):
    """Resolve a word to its first WordNet synset and one-level hypernym.

    Returns:
        (synset_name, hypernym_name, synset_lemmas)
        e.g. ('tool.n.01', 'implement.n.01', ['tool'])
        If no hypernym exists the word is a root: hypernym_name == synset_name.
    """
    if not _WN_AVAILABLE:
        return (word, word, [word])

    pos_hint = _wn_pos_hint(role)
    synsets = wn.synsets(word, pos=pos_hint) if pos_hint else []
    if not synsets:
        synsets = wn.synsets(word)  # fallback: any POS
    if not synsets:
        return (word, word, [word])  # unknown word

    ss = synsets[0]
    hypers = ss.hypernyms()
    hyper_name = hypers[0].name() if hypers else ss.name()  # root if no parent
    return (ss.name(), hyper_name, ss.lemma_names())


def _resolve_word(word, role='OBJ'):
    """Resolve a surface word through 3 abstraction layers.

    Returns dict with keys:
        lemma        : str   – morphy lemma (or word itself)
        synset       : str   – first WordNet synset name (or lemma)
        synset_lemmas: list   – lemma_names of that synset
        hypernym     : str   – one-level-up hypernym name (or synset)
    """
    pos_hint = _wn_pos_hint(role)
    # Layer 1: lemma via morphy
    lemma = (wn.morphy(word, pos_hint) if pos_hint else None) or wn.morphy(word) or word
    # Layer 2: first synset of the lemma
    synsets = (wn.synsets(lemma, pos_hint) if pos_hint else []) or wn.synsets(lemma)
    if not synsets:
        return dict(lemma=lemma, synset=lemma, synset_lemmas=[lemma], hypernym=lemma)
    ss = synsets[0]
    ss_name = ss.name()
    # Layer 3: one-level hypernym
    hypers = ss.hypernyms()
    hyp_name = hypers[0].name() if hypers else ss_name
    return dict(lemma=lemma, synset=ss_name, synset_lemmas=ss.lemma_names(), hypernym=hyp_name)


def render_layered_graph(subj_words, pred_words, obj_words):
    """Render a 4-layer ANN-style knowledge graph.

    Layer 0 (bottom) : surface words  – SPO edges (SUBJ → OBJ via PRED)
    Layer 1           : lemmas        – morphy-reduced, deduplicated
    Layer 2           : synsets       – first WordNet synset, deduplicated
    Layer 3 (top)     : hypernyms     – one level up, deduplicated

    Vertical dashed edges connect each node upward to its parent.
    Multiple words sharing the same lemma/synset/hypernym converge.

    Returns (fig, details_table) or (None, []).
    """
    if not _WN_AVAILABLE:
        return None, []

    # ── resolve every surface word ──────────────────────────────────
    word_info = {}          # (word, role) -> resolution dict
    all_surface = []        # ordered list of (word, role)
    for role, words in [('SUBJ', subj_words), ('PRED', pred_words), ('OBJ', obj_words)]:
        for w in words:
            key = (w, role)
            if key not in word_info:
                word_info[key] = _resolve_word(w, role)
                all_surface.append(key)

    # ── collect unique nodes at each layer ──────────────────────────
    # Layer keys are the string identifier shown as the node label.
    # Use sets to deduplicate.
    lemma_nodes  = {}   # lemma_str  -> set of (word, role) that map here
    synset_nodes = {}   # synset_str -> set of lemma_str that map here
    hyper_nodes  = {}   # hyper_str  -> set of synset_str that map here

    for (w, role), info in word_info.items():
        lem = info['lemma']
        ss  = info['synset']
        hyp = info['hypernym']
        lemma_nodes.setdefault(lem, set()).add((w, role))
        synset_nodes.setdefault(ss, set()).add(lem)
        hyper_nodes.setdefault(hyp, set()).add(ss)

    # ── build the graph ─────────────────────────────────────────────
    G = nx.DiGraph()

    # Prefixed node IDs to avoid collisions across layers
    def _nid(layer, label):
        return f"L{layer}:{label}"

    # Layer 0 – surface words
    for w, role in all_surface:
        nid = _nid(0, f"{w}_{role}")
        G.add_node(nid, label=w, layer=0, role=role)

    # SPO edges at Layer 0
    for s in subj_words:
        for o in obj_words:
            for p in pred_words:
                sid = _nid(0, f"{s}_SUBJ")
                oid = _nid(0, f"{o}_OBJ")
                if G.has_edge(sid, oid):
                    ex = G[sid][oid].get('label', '')
                    if p not in ex:
                        G[sid][oid]['label'] = f"{ex}, {p}"
                else:
                    G.add_edge(sid, oid, label=p, edge_type='spo')

    # Layer 1 – lemmas (deduplicated)
    for lem in lemma_nodes:
        nid = _nid(1, lem)
        G.add_node(nid, label=lem, layer=1, role='LEMMA')
    # vertical edges: L0 word -> L1 lemma
    for (w, role), info in word_info.items():
        G.add_edge(_nid(0, f"{w}_{role}"), _nid(1, info['lemma']), edge_type='vertical')

    # Layer 2 – synsets (deduplicated)
    for ss in synset_nodes:
        short = ss.split('.')[0]  # human-friendly label
        G.add_node(_nid(2, ss), label=short, layer=2, role='SYNSET')
    # vertical edges: L1 lemma -> L2 synset
    for (w, role), info in word_info.items():
        G.add_edge(_nid(1, info['lemma']), _nid(2, info['synset']), edge_type='vertical')

    # Layer 3 – hypernyms (deduplicated)
    for hyp in hyper_nodes:
        short = hyp.split('.')[0]
        G.add_node(_nid(3, hyp), label=short, layer=3, role='HYPERNYM')
    # vertical edges: L2 synset -> L3 hypernym
    for (w, role), info in word_info.items():
        G.add_edge(_nid(2, info['synset']), _nid(3, info['hypernym']), edge_type='vertical')

    if len(G.nodes) == 0:
        return None, []

    # ── layout: columns by SPO role, rows by layer ──────────────────
    layer_spacing_y = 2.5
    pos = {}

    # Layer 0: SUBJ on left, PRED in center, OBJ on right
    role_x = {'SUBJ': -3.0, 'PRED': 0.0, 'OBJ': 3.0}
    l0_counts = {'SUBJ': 0, 'PRED': 0, 'OBJ': 0}
    for w, role in all_surface:
        nid = _nid(0, f"{w}_{role}")
        x_base = role_x[role]
        idx = l0_counts[role]
        pos[nid] = (x_base + idx * 1.4, 0)
        l0_counts[role] += 1

    # Layers 1-3: spread nodes evenly across the x-axis
    for layer_i, node_set in [(1, lemma_nodes), (2, synset_nodes), (3, hyper_nodes)]:
        keys = sorted(node_set.keys())
        n = len(keys)
        x_span = max(8, n * 1.8)
        x_start = -x_span / 2
        for i, k in enumerate(keys):
            nid = _nid(layer_i, k)
            pos[nid] = (x_start + i * (x_span / max(n - 1, 1)), layer_i * layer_spacing_y)

    # ── drawing ─────────────────────────────────────────────────────
    n_total = len(G.nodes)
    fig_w = max(12, n_total * 0.9)
    fig_h = max(8, 3.5 * layer_spacing_y)
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))

    # Separate edges by type
    spo_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('edge_type') == 'spo']
    vert_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('edge_type') == 'vertical']
    # Deduplicate vertical edges (same visual connection drawn once)
    vert_edges = list(set(vert_edges))

    # Layer colours & shapes
    layer_style = {
        0: {'color_SUBJ': '#ff9999', 'color_PRED': '#99ff99', 'color_OBJ': '#9999ff', 'size': 1800, 'shape': 'o'},
        1: {'color': '#d4d4d4', 'size': 1400, 'shape': 's'},
        2: {'color': '#ffe066', 'size': 1400, 'shape': 's'},
        3: {'color': '#d9b3ff', 'size': 1600, 'shape': 'D'},
    }

    # Draw L0 nodes per role colour
    for sub_role, col_key in [('SUBJ', 'color_SUBJ'), ('PRED', 'color_PRED'), ('OBJ', 'color_OBJ')]:
        nlist = [n for n in G.nodes if G.nodes[n].get('layer') == 0 and G.nodes[n].get('role') == sub_role]
        if nlist:
            nx.draw_networkx_nodes(G, pos, nodelist=nlist,
                                   node_color=layer_style[0][col_key],
                                   node_size=layer_style[0]['size'],
                                   node_shape=layer_style[0]['shape'],
                                   alpha=0.9, ax=ax)

    # Draw L1-L3 nodes
    for li in [1, 2, 3]:
        nlist = [n for n in G.nodes if G.nodes[n].get('layer') == li]
        if nlist:
            nx.draw_networkx_nodes(G, pos, nodelist=nlist,
                                   node_color=layer_style[li]['color'],
                                   node_size=layer_style[li]['size'],
                                   node_shape=layer_style[li]['shape'],
                                   alpha=0.85, ax=ax)

    # Draw SPO edges (solid green)
    if spo_edges:
        nx.draw_networkx_edges(G, pos, edgelist=spo_edges,
                               edge_color='#66cc66', width=2.5,
                               arrows=True, arrowsize=18,
                               connectionstyle='arc3,rad=0.08', ax=ax)
    # SPO edge labels
    spo_edge_labels = {(u, v): G[u][v].get('label', '') for u, v in spo_edges}
    if spo_edge_labels:
        nx.draw_networkx_edge_labels(G, pos, edge_labels=spo_edge_labels,
                                      font_size=9, font_color='#228B22',
                                      label_pos=0.5, ax=ax)

    # Draw vertical edges (dashed gray)
    if vert_edges:
        nx.draw_networkx_edges(G, pos, edgelist=vert_edges,
                               edge_color='#999999', width=1.0,
                               style='dashed', arrows=True, arrowsize=12,
                               ax=ax)

    # Node labels (use the 'label' attribute, not the full nid)
    nlabels = {n: G.nodes[n].get('label', n) for n in G.nodes}
    nx.draw_networkx_labels(G, pos, labels=nlabels, font_size=9, font_weight='bold', ax=ax)

    # Layer annotations on the left
    layer_names = {0: 'Surface Words', 1: 'Lemmas', 2: 'Synsets', 3: 'Hypernyms'}
    x_min = min(v[0] for v in pos.values()) - 2.0
    for li, name in layer_names.items():
        ax.text(x_min, li * layer_spacing_y, f'L{li}: {name}',
                fontsize=10, fontstyle='italic', color='#555555',
                ha='right', va='center')

    # Legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor='#ff9999', label='Subject'),
        Patch(facecolor='#99ff99', label='Predicate'),
        Patch(facecolor='#9999ff', label='Object'),
        Patch(facecolor='#d4d4d4', label='Lemma'),
        Patch(facecolor='#ffe066', label='Synset'),
        Patch(facecolor='#d9b3ff', label='Hypernym'),
        Line2D([0], [0], color='#66cc66', lw=2, label='SPO edge'),
        Line2D([0], [0], color='#999999', lw=1, linestyle='dashed', label='Abstraction link'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8, ncol=2)

    ax.set_title('Layered Knowledge Graph  (Words → Lemmas → Synsets → Hypernyms)',
                 fontsize=13, fontweight='bold')
    ax.axis('off')
    fig.tight_layout()

    # ── details table ───────────────────────────────────────────────
    details = []
    for (w, role), info in word_info.items():
        details.append({
            'Role': role,
            'Word': w,
            'Lemma': info['lemma'],
            'Synset': info['synset'],
            'Synset Lemmas': ', '.join(info['synset_lemmas'][:4]),
            'Hypernym': info['hypernym'],
        })
    return fig, details


@st.cache_resource
def load_model():
    """Load trained model (cached)."""
    model_path = Path('bio_tagger_atomic.pt')
    if not model_path.exists():
        st.error(f"Model not found: {model_path}")
        st.info("Run tune_bio_tagger.py first to train the model")
        return None, None, None
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # Checkpoint with metadata
        state_dict = checkpoint['model_state_dict']
        num_labels = checkpoint.get('num_labels', 5)
    else:
        # Direct state_dict - detect num_labels from classifiers
        state_dict = checkpoint
        # Count classifiers (classifiers.0.weight, classifiers.1.weight, etc.)
        classifier_keys = [k for k in state_dict.keys() if k.startswith('classifiers.') and k.endswith('.weight')]
        num_labels = len(classifier_keys) if classifier_keys else 5
    
    model = BIOTaggerMultiClass(num_labels=num_labels, dropout=0.1, unfreeze_layers=12)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Build label map from num_labels
    if num_labels == 5:
        # I-PRED removed (conditional)
        label_map = LABEL_MAP
    else:
        # Full 6 classes
        label_map = LABEL_MAP_FULL
    
    return model, tokenizer, label_map


@st.cache_data
def load_eval_data():
    """Load eval set from msgpack."""
    data_path = Path('data/bio_training_250chunks_complete_FIXED.msgpack')
    if not data_path.exists():
        st.error(f"Data not found: {data_path}")
        return None
    
    with open(data_path, 'rb') as f:
        data = msgpack.unpackb(f.read(), raw=False)
    
    return data


@st.cache_data
def load_holdout_predictions():
    """Load holdout predictions (cached)."""
    predictions_path = Path('holdout_predictions.json')
    if not predictions_path.exists():
        return None
    
    with open(predictions_path, 'r') as f:
        return json.load(f)


def predict_text(text, model, tokenizer, label_map, threshold=0.3):
    """Run BIO prediction on input text using multi-label approach."""
    device = next(model.parameters()).device
    
    # Tokenize
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Predict (returns probabilities from sigmoid)
    with torch.no_grad():
        probs = model(input_ids, attention_mask)
    
    # Decode tokens
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    valid_length = attention_mask[0].sum().item()
    tokens = tokens[:valid_length]
    
    # Convert probabilities to labels using threshold
    probs = probs[0, :valid_length].cpu().numpy()  # [seq_len, num_labels]
    predicted_labels = []
    
    for token_probs in probs:
        # Find labels above threshold
        active_labels = [i for i, p in enumerate(token_probs) if p > threshold]
        
        if not active_labels:
            # No labels above threshold - default to 'O' (outside entity)
            predicted_labels.append('O')
        else:
            # Take highest probability label
            best_label_idx = max(active_labels, key=lambda i: token_probs[i])
            predicted_labels.append(label_map[best_label_idx])
    
    return tokens, predicted_labels


def predict_eval_example(example, model, tokenizer, label_map, threshold=0.3):
    """Run prediction on eval example using multi-label approach."""
    device = next(model.parameters()).device
    
    input_ids = torch.tensor(example['input_ids']).unsqueeze(0).to(device)
    attention_mask = torch.tensor(example['attention_mask']).unsqueeze(0).to(device)
    true_labels_list = example['labels']  # List of label indices
    
    # Predict (returns probabilities from sigmoid)
    with torch.no_grad():
        probs = model(input_ids, attention_mask)
    
    # Decode tokens
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    valid_length = attention_mask[0].sum().item()
    tokens = tokens[:valid_length]
    
    # Convert probabilities to label indices
    probs = probs[0, :valid_length].cpu().numpy()  # [seq_len, num_labels]
    predicted_labels_ids = []
    
    for token_probs in probs:
        # Find labels above threshold
        active_labels = [i for i, p in enumerate(token_probs) if p > threshold]
        
        if not active_labels:
            # Default to first label if none active (shouldn't happen with proper training)
            predicted_labels_ids.append(0)
        else:
            # Take highest probability label
            best_label_idx = max(active_labels, key=lambda i: token_probs[i])
            predicted_labels_ids.append(best_label_idx)
    
    # Filter to valid length
    true_labels_ids = true_labels_list[:valid_length]
    
    # Convert to label names
    predicted_labels = [label_map[idx] for idx in predicted_labels_ids]
    true_labels_names = [label_map[idx] for idx in true_labels_ids]
    
    return tokens, predicted_labels, true_labels_names, predicted_labels_ids, true_labels_ids


def main():
    st.set_page_config(
        page_title="BIO Tagger Demo",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 Subject-Predicate-Object Extraction Demo")
    st.markdown("**Interactive BIO Tagger** | Box-Cox Resampling | Trained with Optuna")
    
    # Load model
    with st.spinner("Loading model..."):
        model, tokenizer, label_map = load_model()
    
    if model is None:
        st.error("Failed to load model. Run tune_bio_tagger.py first.")
        return
    
    st.success(f"✅ Model loaded successfully! ({len(label_map)} classes)")
    
    # Load eval data
    with st.spinner("Loading eval data..."):
        data = load_eval_data()
    
    # Tabs for different modes
    tab1, tab2, tab3 = st.tabs(["📝 Interactive Testing", "📊 Eval Set Browser", "📈 Eval Metrics"])
    
    # TAB 1: Interactive Testing
    with tab1:
        st.subheader("📝 Enter Text for Analysis")
        
        user_text = st.text_area(
            "Input text:",
            value="Humans possess an extraordinary ability to create and utilize tools.",
            height=150,
            help="Enter a sentence to extract Subject-Predicate-Object relationships"
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            predict_button = st.button("🚀 Extract Entities", type="primary")
        
        if predict_button and user_text.strip():
            with st.spinner("Analyzing text..."):
                tokens, predicted_labels = predict_text(user_text, model, tokenizer, label_map)
                entities, entities_words = extract_entities(tokens, predicted_labels)
            
            st.markdown("---")
            
            # Display results in columns
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🎯 Extracted Entities")
                
                for entity_type in ['SUBJ', 'PRED', 'OBJ']:
                    entity_list = entities.get(entity_type, [])
                    if entity_list:
                        st.markdown(f"**{entity_type}:**")
                        for entity in entity_list:
                            st.markdown(f"- `{entity}`")
                    else:
                        st.markdown(f"**{entity_type}:** *(none found)*")
                
                # Summary
                st.markdown("---")
                st.metric("Total Entities", 
                         sum(len(entities.get(k, [])) for k in ['SUBJ', 'PRED', 'OBJ']))
            
            with col2:
                st.subheader("🏷️ Token-Level Predictions")
                
                # Create a dataframe for better display
                import pandas as pd
                
                df = pd.DataFrame({
                    'Token': tokens,
                    'BIO Label': predicted_labels
                })
                
                # Color code by label type
                def color_label(label):
                    if 'SUBJ' in label:
                        return 'background-color: #ffcccc'
                    elif 'PRED' in label:
                        return 'background-color: #ccffcc'
                    elif 'OBJ' in label:
                        return 'background-color: #ccccff'
                    else:
                        return ''
                
                st.dataframe(
                    df.style.applymap(color_label, subset=['BIO Label']),
                    height=400
                )
                
                st.markdown("""
                **Legend:**
                - 🔴 Red: Subject
                - 🟢 Green: Predicate  
                - 🔵 Blue: Object
                """)
            
            # --- Cartesian Expansion & Graph ---
            st.markdown("---")
            st.subheader("🔗 Cartesian SPO Expansion")
            
            triplets, s_words, p_words, o_words = expand_cartesian_triplets(entities_words)
            
            if triplets:
                # Show the word decomposition
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.markdown("**SUBJ words:**")
                    for w in s_words:
                        st.markdown(f"- `{w}`")
                with col_b:
                    st.markdown("**PRED words:**")
                    for w in p_words:
                        st.markdown(f"- `{w}`")
                with col_c:
                    st.markdown("**OBJ words:**")
                    for w in o_words:
                        st.markdown(f"- `{w}`")
                
                st.markdown(f"**Cartesian triplets:** {len(triplets)}")
                
                # Show triplets as a table
                import pandas as pd
                triplets_df = pd.DataFrame(triplets, columns=['Subject', 'Predicate', 'Object'])
                st.dataframe(triplets_df, use_container_width=True, height=min(400, 35 * (len(triplets) + 1)))
                
                # Render the atomic graph
                st.subheader("🕸️ Relation Graph")
                fig = render_triplet_graph(s_words, p_words, o_words)
                if fig is not None:
                    st.pyplot(fig)
                    plt.close(fig)
                
                # --- Layered Knowledge Graph ---
                if _WN_AVAILABLE:
                    st.markdown("---")
                    st.subheader("🧠 Layered Knowledge Graph")
                    st.caption(
                        "Each word is connected upward through layers of "
                        "increasing abstraction: **Lemma → Synset → Hypernym**.  "
                        "Nodes are deduplicated — if two words share the same "
                        "lemma (or synset, or hypernym), they converge to a single node."
                    )
                    
                    fig_syn, details = render_layered_graph(s_words, p_words, o_words)
                    if fig_syn is not None:
                        st.pyplot(fig_syn)
                        plt.close(fig_syn)
                        
                        # Show the resolution table
                        if details:
                            import pandas as pd
                            st.markdown("**Resolution Details  (Word → Lemma → Synset → Hypernym):**")
                            st.dataframe(
                                pd.DataFrame(details),
                                use_container_width=True,
                                height=min(300, 35 * (len(details) + 1))
                            )
                    else:
                        st.info("WordNet could not resolve any synsets for these words.")
            else:
                st.info("Need at least one Subject, Predicate, and Object to build the graph.")
    
    # TAB 2: Eval Set Browser
    with tab2:
        if data is None:
            st.warning("Eval data not loaded")
        else:
            st.subheader("📊 Browse Eval Set Predictions")
            
            # Get eval indices (Box-Cox selected)
            eval_data = data.get('eval_data', [])
            
            if not eval_data:
                st.warning("No eval data found in msgpack file")
            else:
                st.info(f"Total eval examples: {len(eval_data)}")
                
                # Example selector
                example_idx = st.selectbox(
                    "Select example:",
                    range(len(eval_data)),
                    format_func=lambda i: f"Example {i+1} / {len(eval_data)}"
                )
                
                example = eval_data[example_idx]
                
                # Run prediction
                tokens, pred_labels, true_labels, pred_ids, true_ids = predict_eval_example(
                    example, model, tokenizer, label_map
                )
                
                # Display
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**🎯 Predicted Entities:**")
                    pred_entities, _ = extract_entities(tokens, pred_labels)
                    for entity_type in ['SUBJ', 'PRED', 'OBJ']:
                        entities = pred_entities.get(entity_type, [])
                        if entities:
                            st.markdown(f"**{entity_type}:** {', '.join(entities)}")
                        else:
                            st.markdown(f"**{entity_type}:** *(none)*")
                
                with col2:
                    st.markdown("**✅ True Entities:**")
                    true_entities, _ = extract_entities(tokens, true_labels)
                    for entity_type in ['SUBJ', 'PRED', 'OBJ']:
                        entities = true_entities.get(entity_type, [])
                        if entities:
                            st.markdown(f"**{entity_type}:** {', '.join(entities)}")
                        else:
                            st.markdown(f"**{entity_type}:** *(none)*")
                
                # Token-level comparison
                st.markdown("---")
                st.markdown("**🏷️ Token-Level Comparison:**")
                
                import pandas as pd
                
                # Create comparison dataframe
                df = pd.DataFrame({
                    'Token': tokens,
                    'Predicted': pred_labels,
                    'True': true_labels,
                    'Match': ['✅' if p == t else '❌' for p, t in zip(pred_labels, true_labels)]
                })
                
                # Color mismatches
                def highlight_mismatch(row):
                    if row['Match'] == '❌':
                        return ['background-color: #ffcccc'] * len(row)
                    else:
                        return [''] * len(row)
                
                st.dataframe(
                    df.style.apply(highlight_mismatch, axis=1),
                    height=400
                )
                
                # Example-level metrics
                correct = sum(1 for p, t in zip(pred_ids, true_ids) if p == t)
                total = len(pred_ids)
                accuracy = correct / total if total > 0 else 0
                
                st.metric("Example Accuracy", f"{accuracy:.2%}", f"{correct}/{total} tokens")
    
    # TAB 3: Eval Metrics
    with tab3:
        if data is None:
            st.warning("Eval data not loaded")
        else:
            st.subheader("📈 Overall Eval Set Metrics")
            
            eval_data = data.get('eval_data', [])
            
            if not eval_data:
                st.warning("No eval data found")
            else:
                with st.spinner(f"Running predictions on {len(eval_data)} examples..."):
                    all_preds = []
                    all_trues = []
                    
                    progress_bar = st.progress(0)
                    
                    for idx, example in enumerate(eval_data):
                        _, _, _, pred_ids, true_ids = predict_eval_example(
                            example, model, tokenizer, label_map
                        )
                        all_preds.extend(pred_ids)
                        all_trues.extend(true_ids)
                        
                        progress_bar.progress((idx + 1) / len(eval_data))
                    
                    progress_bar.empty()
                
                # Compute metrics
                labels = sorted(label_map.keys())
                label_names = [label_map[i] for i in labels]
                
                precision, recall, f1, support = precision_recall_fscore_support(
                    all_trues, all_preds, labels=labels, average=None, zero_division=0
                )
                
                # Overall metrics
                macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
                    all_trues, all_preds, labels=labels, average='macro', zero_division=0
                )
                
                weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
                    all_trues, all_preds, labels=labels, average='weighted', zero_division=0
                )
                
                # Display overall metrics
                col1, col2, col3 = st.columns(3)
                col1.metric("Macro F1", f"{macro_f1:.4f}")
                col2.metric("Weighted F1", f"{weighted_f1:.4f}")
                col3.metric("Total Tokens", len(all_preds))
                
                st.markdown("---")
                
                # Per-class metrics table
                st.markdown("**Per-Class Metrics:**")
                
                import pandas as pd
                
                metrics_df = pd.DataFrame({
                    'Label': label_names,
                    'Precision': [f"{p:.4f}" for p in precision],
                    'Recall': [f"{r:.4f}" for r in recall],
                    'F1': [f"{f:.4f}" for f in f1],
                    'Support': support
                })
                
                st.dataframe(metrics_df, use_container_width=True)
                
                # Classification report
                st.markdown("---")
                st.markdown("**Full Classification Report:**")
                
                report = classification_report(
                    all_trues, all_preds,
                    labels=labels,
                    target_names=label_names,
                    zero_division=0
                )
                
                st.text(report)
    
    # Footer with model info
    st.markdown("---")
    st.markdown(f"""
    **Model Details:**
    - Architecture: BERT-base-uncased with unfrozen top 12 layers
    - Training: Box-Cox resampling with seed rotation
    - Classes: {', '.join(label_map.values())}
    - Labels: {len(label_map)} (I-PRED conditionally removed if nonexistent)
    """)


if __name__ == "__main__":
    main()
