"""
BIO-Tagged Training Data Extraction for Triplet Distillation

ARCHITECTURE OVERVIEW
===================

Core Thesis:
  OpenIE already solved the hard problem (extracting triplets with POS rules).
  We just need to learn to reproduce OpenIE's output with a fast neural model.
  
  This is KNOWLEDGE DISTILLATION:
    Slow Teacher (OpenIE) → Training labels → Fast Student (BERT)

Two-Component System:
  1. Training: Use OpenIE to extract triplets → Convert to BIO tags
  2. Inference: BERT predicts BIO tags → Simple rules reconstruct triplets

Why BIO Tagging (Not Seq2seq):
  - Triplet elements are SPANS in the original text (not generated)
  - Token classification is simpler than sequence generation
  - Parallel predictions (all tokens at once) vs sequential
  - Much faster training and inference

Handling Overlapping Triplets:
  Text: "The model learns patterns and predicts outcomes"
  
  Triplet 1: P(learns, model, patterns)
  Triplet 2: P(predicts, model, outcomes)
  
  Problem: "model" is subject in BOTH triplets
  Solution: Multi-hot labels (token can have multiple roles)
  
  Standard BIO: Each token gets ONE label (mutually exclusive)
  Multi-hot BIO: Each token gets 6 INDEPENDENT binary labels
    - B-SUBJ, I-SUBJ, B-PRED, I-PRED, B-OBJ, I-OBJ
    - "model" can be B-SUBJ=1 in multiple triplets

Model Architecture:
  BERT → 6 independent binary classifiers (NOT 7-way softmax)
  
  class TripletBIOTagger(nn.Module):
      def __init__(self):
          self.bert = BERT
          self.subj_begin = Linear(768, 1)  # Independent
          self.subj_inside = Linear(768, 1) # classifiers
          self.pred_begin = Linear(768, 1)  # per
          self.pred_inside = Linear(768, 1) # BIO
          self.obj_begin = Linear(768, 1)   # label
          self.obj_inside = Linear(768, 1)
      
      def forward(self, input_ids):
          hidden = self.bert(input_ids)
          return {
              'B-SUBJ': sigmoid(self.subj_begin(hidden)),
              'I-SUBJ': sigmoid(self.subj_inside(hidden)),
              # ... etc (6 independent probabilities per token)
          }
  
  Training: Binary cross-entropy for EACH label independently
  Inference: threshold > 0.5 for each label

Post-Processing (Reconstruction):
  After tagging tokens, use simple position-based heuristics:
    1. Extract spans: Group consecutive B/I tags
    2. Match predicates with subjects/objects by proximity:
       - Subject: Nearest B-SUBJ before predicate
       - Object: Nearest B-OBJ after predicate
    
  This mimics OpenIE's behavior without re-learning from scratch.

Training Data Format:
  {
    'chunk_id': 123,
    'tokens': ['The', 'model', 'learns', 'patterns'],
    'labels': {
      'B-SUBJ': [0, 1, 0, 0],  # Multi-hot: can have multiple 1s
      'I-SUBJ': [0, 0, 0, 0],
      'B-PRED': [0, 0, 1, 0],
      'I-PRED': [0, 0, 0, 0],
      'B-OBJ': [0, 0, 0, 1],
      'I-OBJ': [0, 0, 0, 0]
    },
    'triplets': [  # Original OpenIE output (for validation)
      {'subject': 'model', 'predicate': 'learns', 'object': 'patterns'}
    ]
  }

Performance Expectations:
  Training time: ~2-3 hours (3000 chunks with OpenIE extraction)
  Inference time: ~10ms per chunk (vs 6s for OpenIE = 600x faster)
  Quality: Should match OpenIE 80-90% (knowledge distillation typical)

Integration with Synset Vocabulary:
  After extracting triplets, apply synset reduction to spans:
    "neural network" → synset_reducer.reduce() → ["nervous", "network"]
  
  This is done POST-TAGGING (not during model training).
  Model learns OpenIE's behavior, synsets applied at inference.
"""

import msgpack
import psycopg2
import numpy as np
from tqdm import tqdm
import argparse
from collections import defaultdict
from transformers import AutoTokenizer
import re
import stanza  # Add at top for better performance

# Database configuration (matches all working scripts)
DB_CONFIG = {
    'dbname': 'langchain',
    'user': 'langchain',
    'password': 'langchain',
    'host': 'localhost',
    'port': 5432
}


def get_subtree_text(head_word, sent) -> str:
    """Get all text tokens in the subtree rooted at head_word (keeps multi-word phrases intact)."""
    subtree_words = [head_word]
    
    # Recursively collect all descendants
    def collect_descendants(word_id):
        for w in sent.words:
            if w.head == word_id:
                subtree_words.append(w)
                collect_descendants(w.id)
    
    collect_descendants(head_word.id)
    
    # Sort by position and join
    sorted_words = sorted(subtree_words, key=lambda x: x.id)
    return ' '.join([w.text for w in sorted_words])


def extract_spo_from_sentence(sent) -> list:
    """Extract Subject-Predicate-Object triplets from Stanza sentence using dependency parse.
    
    CRITICAL: Keeps multi-word phrases intact (e.g., "deep learning models" as one subject).
    This ensures BIO labels will have proper B-I continuous spans.
    
    Handles two construction types:
        Regular verbs:     subject→nsubj of VERB ROOT, object→obj/obl of VERB ROOT
        Copula (UD style): NOUN/ADJ is ROOT, 'is/are' is cop child, nsubj→subject
                           predicate = the nominal head itself (not the copula verb)
                           e.g. "The bible is the word of God"
                                ROOT=word, cop=is, nsubj=bible, nmod=God
                                → (bible, word, god)
    
    Args:
        sent: Stanza Sentence object with dependency parse
    
    Returns:
        List of triplets as dicts: {'subject': str, 'predicate': str, 'object': str}
    """
    triplets = []

    # Find main verb(s) — VERB/AUX ROOT/conj, but skip bare copulas (deprel=cop)
    # Copulas are handled separately below as copula constructions
    verbs = []
    for word in sent.words:
        if word.upos in ['VERB', 'AUX'] and (word.head == 0 or word.deprel in ['ROOT', 'conj']):
            if word.deprel != 'cop':
                verbs.append(word)

    # Copula constructions (UD style): NOUN/PROPN/ADJ is ROOT, 'is/are/was' is cop child
    # e.g. "The bible is the word of God"
    #   → ROOT=word(NOUN), cop=is, nsubj=bible, nmod=God
    #   → predicate=word, subject=bible, object=god
    copular_predicates = []
    for word in sent.words:
        if word.upos in ['NOUN', 'PROPN', 'ADJ'] and (word.head == 0 or word.deprel in ['ROOT', 'conj']):
            has_cop = any(w.head == word.id and w.deprel == 'cop' for w in sent.words)
            if has_cop:
                copular_predicates.append(word)

    if not verbs and not copular_predicates:
        return []

    # Process regular verbs
    for verb in verbs:
        subject_head = None
        object_head = None
        for word in sent.words:
            if word.head == verb.id:
                if word.deprel in ['nsubj', 'nsubj:pass', 'csubj']:
                    subject_head = word
                elif word.deprel in ['obj', 'dobj', 'iobj', 'obl']:
                    object_head = word
        if verb:
            subject_text = get_subtree_text(subject_head, sent) if subject_head else '?'
            predicate_text = verb.text
            object_text = get_subtree_text(object_head, sent) if object_head else '?'
            if predicate_text and (subject_text != '?' or object_text != '?'):
                triplets.append({
                    'subject': subject_text if subject_text != '?' else '',
                    'predicate': predicate_text,
                    'object': object_text if object_text != '?' else ''
                })

    # Process copula constructions — nominal/adjectival predicate as PRED
    for pred_nominal in copular_predicates:
        subject_head = None
        object_head = None
        for word in sent.words:
            if word.head == pred_nominal.id:
                if word.deprel in ['nsubj', 'nsubj:pass', 'csubj']:
                    subject_head = word
                elif word.deprel in ['nmod', 'obl', 'obj']:
                    object_head = word
        subject_text = get_subtree_text(subject_head, sent) if subject_head else '?'
        predicate_text = pred_nominal.text  # just the head word — not full subtree
        object_text = get_subtree_text(object_head, sent) if object_head else '?'
        if subject_text != '?' or object_text != '?':
            triplets.append({
                'subject': subject_text if subject_text != '?' else '',
                'predicate': predicate_text,
                'object': object_text if object_text != '?' else ''
            })

    return triplets


def extract_triplets_openie(chunk_id, content):
    """
    Extract triplets using Stanza dependency parsing.
    
    KEY FIX: Uses get_subtree_text to preserve multi-word phrases.
    Example: "deep learning models" stays intact, not split into 3 separate entities.
    
    This ensures BIO labels will have:
      [B-SUBJ, I-SUBJ, I-SUBJ] for "deep learning models"
    NOT:
      [B-SUBJ, B-SUBJ, B-SUBJ] (which violates proper BIO tagging)
    
    Returns:
        List of triplets: [{'subject': str, 'predicate': str, 'object': str}, ...]
    """
    # Initialize Stanza (first time only - it's cached)
    try:
        nlp = stanza.Pipeline('en', processors='tokenize,pos,lemma,depparse', use_gpu=False, download_method=None)
    except:
        print("  WARNING: Stanza not initialized, downloading models...")
        stanza.download('en')
        nlp = stanza.Pipeline('en', processors='tokenize,pos,lemma,depparse', use_gpu=False)
    
    # Parse content
    doc = nlp(content)
    
    # Extract triplets from each sentence
    all_triplets = []
    for sent in doc.sentences:
        sentence_triplets = extract_spo_from_sentence(sent)
        all_triplets.extend(sentence_triplets)
    
    return all_triplets


def find_span_in_tokens(tokens, text):
    """
    Find span of text in token list.
    
    Args:
        tokens: List of tokenized strings
        text: Text to find
    
    Returns:
        (start_idx, end_idx) or (None, None) if not found
    
    Example:
        tokens = ['The', 'neural', 'network', 'learns']
        text = 'neural network'
        → (1, 3)  # tokens[1:3] = ['neural', 'network']
    """
    if not text or not text.strip():
        return None, None
    
    # Normalize text
    text_lower = text.lower().strip()
    text_tokens = text_lower.split()
    
    # Try to find contiguous match
    for i in range(len(tokens)):
        # Check if tokens[i:i+len(text_tokens)] matches
        match = True
        for j, text_tok in enumerate(text_tokens):
            if i + j >= len(tokens):
                match = False
                break
            if tokens[i + j].lower() != text_tok:
                match = False
                break
        
        if match:
            return i, i + len(text_tokens)
    
    # Fallback: Try partial match (first word)
    if text_tokens:
        first_word = text_tokens[0]
        for i, tok in enumerate(tokens):
            if tok.lower() == first_word:
                return i, i + 1
    
    return None, None


def create_bio_labels(tokens, triplets):
    """
    Convert OpenIE triplets to multi-hot BIO labels.
    
    Args:
        tokens: List of tokens from tokenizer
        triplets: List of dicts with subject/predicate/object
    
    Returns:
        Dict of {label_name: [0/1 list]} where 1 indicates active label
    
    Key Design:
        Multi-hot encoding allows overlapping triplets.
        Token can be B-SUBJ in multiple triplets simultaneously.
    """
    # Initialize all labels to 0
    labels = {
        'B-SUBJ': [0] * len(tokens),
        'I-SUBJ': [0] * len(tokens),
        'B-PRED': [0] * len(tokens),
        'I-PRED': [0] * len(tokens),
        'B-OBJ': [0] * len(tokens),
        'I-OBJ': [0] * len(tokens)
    }
    
    # Mark spans for each triplet
    for triplet in triplets:
        # Subject
        if triplet.get('subject'):
            subj_start, subj_end = find_span_in_tokens(tokens, triplet['subject'])
            if subj_start is not None:
                labels['B-SUBJ'][subj_start] = 1
                for i in range(subj_start + 1, subj_end):
                    labels['I-SUBJ'][i] = 1
        
        # Predicate
        if triplet.get('predicate'):
            pred_start, pred_end = find_span_in_tokens(tokens, triplet['predicate'])
            if pred_start is not None:
                labels['B-PRED'][pred_start] = 1
                for i in range(pred_start + 1, pred_end):
                    labels['I-PRED'][i] = 1
        
        # Object (optional)
        if triplet.get('object'):
            obj_start, obj_end = find_span_in_tokens(tokens, triplet['object'])
            if obj_start is not None:
                labels['B-OBJ'][obj_start] = 1
                for i in range(obj_start + 1, obj_end):
                    labels['I-OBJ'][i] = 1
    
    return labels


def main():
    parser = argparse.ArgumentParser(description='Extract BIO-tagged training data from OpenIE')
    parser.add_argument('--chunks', type=int, default=1000, help='Number of chunks to process (start with 1000, evaluate, add more if needed)')
    parser.add_argument('--output', type=str, default='bio_training_data.msgpack',
                       help='Output msgpack file')
    args = parser.parse_args()
    
    print("="*70)
    print("BIO-TAGGED TRAINING DATA EXTRACTION")
    print("="*70)
    print(f"\nKnowledge Distillation: OpenIE (teacher) -> BERT (student)")
    print(f"Target chunks: {args.chunks:,}")
    print(f"Output: {args.output}")
    
    # Initialize tokenizer
    print("\n[1/5] Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    # Database connection
    print("\n[2/5] Connecting to database...")
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    # Get random chunk IDs
    print("\n[3/5] Fetching chunk IDs...")
    cur.execute("""
        SELECT chunk_id 
        FROM arxiv_chunks 
        ORDER BY RANDOM() 
        LIMIT %s
    """, (args.chunks,))
    chunk_ids = [row[0] for row in cur.fetchall()]
    print(f"  Fetched {len(chunk_ids):,} chunk IDs")
    
    # Extract triplets with Stanza + convert to BIO tags
    print("\n[4/5] Extracting triplets with Stanza (multi-word phrase preservation)...")
    print("  Note: Using Stanza dependency parsing instead of OpenIE")
    print("  This preserves multi-word phrases -> proper B-I continuous spans")
    
    # Initialize Stanza once
    try:
        nlp_stanza = stanza.Pipeline('en', processors='tokenize,pos,lemma,depparse', use_gpu=False, download_method=None)
        print("  [OK] Stanza loaded\n")
    except:
        print("  Downloading Stanza models (one-time only)...")
        stanza.download('en')
        nlp_stanza = stanza.Pipeline('en', processors='tokenize,pos,lemma,depparse', use_gpu=False)
        print("  [OK] Stanza loaded\n")
    
    training_data = []
    stats = {
        'total_chunks': 0,
        'chunks_with_triplets': 0,
        'total_triplets': 0,
        'total_tokens': 0,
        'avg_triplets_per_chunk': 0,
        'label_distribution': defaultdict(int)
    }
    
    for chunk_id in tqdm(chunk_ids, desc="  Processing chunks"):
        # Get chunk content
        cur.execute("SELECT content FROM arxiv_chunks WHERE chunk_id = %s", (chunk_id,))
        row = cur.fetchone()
        if not row:
            continue
        
        content = row[0]
        
        # Parse with Stanza and extract triplets (per sentence)
        try:
            doc_stanza = nlp_stanza(content)
            
            # Process each sentence separately
            for sent in doc_stanza.sentences:
                # Extract triplets using dependency parse
                triplets = extract_spo_from_sentence(sent)
                
                if not triplets:
                    continue
                
                # Get sentence text
                sentence = sent.text
                
                # Tokenize sentence with BERT tokenizer
                tokens = tokenizer.tokenize(sentence)
                
                # Skip if too long
                if len(tokens) > 510:
                    tokens = tokens[:510]
                
                # Convert triplets to BIO labels
                bio_labels = create_bio_labels(tokens, triplets)
                
                # Track statistics
                stats['total_chunks'] += 1
                stats['chunks_with_triplets'] += 1
                stats['total_triplets'] += len(triplets)
                stats['total_tokens'] += len(tokens)
                
                for label_name, label_values in bio_labels.items():
                    stats['label_distribution'][label_name] += sum(label_values)
                
                # Store training example (sentence-level)
                training_data.append({
                    'chunk_id': chunk_id,
                    'sentence': sentence,
                    'tokens': tokens,
                    'labels': bio_labels,
                    'triplets': triplets  # Keep for validation
                })
                
        except Exception as e:
            continue
    
    # Calculate statistics
    if stats['chunks_with_triplets'] > 0:
        stats['avg_triplets_per_chunk'] = stats['total_triplets'] / stats['chunks_with_triplets']
    
    print(f"\n  Processed: {stats['total_chunks']:,} chunks")
    print(f"  Chunks with triplets: {stats['chunks_with_triplets']:,}")
    print(f"  Total triplets extracted: {stats['total_triplets']:,}")
    print(f"  Avg triplets per chunk: {stats['avg_triplets_per_chunk']:.2f}")
    print(f"\n  Label distribution:")
    for label in ['B-SUBJ', 'I-SUBJ', 'B-PRED', 'I-PRED', 'B-OBJ', 'I-OBJ']:
        count = stats['label_distribution'][label]
        pct = 100 * count / stats['total_tokens'] if stats['total_tokens'] > 0 else 0
        print(f"    {label}: {count:,} tokens ({pct:.2f}%)")
    
    # Save to msgpack
    print(f"\n[5/5] Saving to {args.output}...")
    save_data = {
        'training_data': training_data,
        'stats': dict(stats),
        'tokenizer_name': 'bert-base-uncased',
        'label_names': ['B-SUBJ', 'I-SUBJ', 'B-PRED', 'I-PRED', 'B-OBJ', 'I-OBJ'],
        'architecture': 'multi-hot BIO tagging (6 independent binary classifiers)'
    }
    
    with open(args.output, 'wb') as f:
        msgpack.pack(save_data, f)
    
    print(f"  Saved {len(training_data):,} training examples")
    
    # Close connection
    cur.close()
    conn.close()
    
    print("\n" + "="*70)
    print("COMPLETE - Training data ready for model training")
    print("="*70)
    print(f"\nTraining examples: {len(training_data):,}")
    print(f"Total tokens: {stats['total_tokens']:,}")
    print(f"Total triplets: {stats['total_triplets']:,}")
    
    print("\nArchitecture Summary:")
    print("  Model: BERT + 6 independent binary classifiers")
    print("  Labels: Multi-hot BIO (handles overlapping triplets)")
    print("  Loss: Binary cross-entropy per label")
    print("  Output: Token-level probabilities [0-1] for each BIO tag")
    
    print("\nNext Steps:")
    print("  1. Train model: python train_bio_tagger.py")
    print("  2. Inference: BERT -> BIO tags -> Simple rules -> Triplets")
    print("  3. Apply synset reduction at inference time")
    
    print("\nExpected Performance:")
    print("  Training: ~2-3 hours on GPU")
    print("  Inference: ~10ms per chunk (600x faster than OpenIE)")
    print("  Quality: 80-90% of OpenIE accuracy (typical distillation)")


if __name__ == '__main__':
    main()
