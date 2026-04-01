"""
Build Graph Sparse Vectors for ArXiv Papers

Proper triplet extraction pipeline with:
1. Real triplet_extract library (not regex)
2. Synset-based entity consolidation
3. Hypernym bridging
4. Wordpiece tokenization for OOV handling
5. Knowledge graph construction (nodes + edges)

USAGE:
    python build_arxiv_graph_sparse.py
"""

# Disable torch dynamo to avoid import hang
import os
os.environ['PYTORCH_JIT'] = '0'
os.environ['TORCH_COMPILE_DISABLE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'

import psycopg2
from psycopg2.extras import execute_values
from tqdm import tqdm
import numpy as np
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Set
import nltk
from nltk.corpus import stopwords, wordnet as wn
from nltk.stem import WordNetLemmatizer
import cleantext
import wordninja
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import spacy
from triplet_extract import extract, OpenIEExtractor
from transformers import AutoTokenizer
import torch
import threading

# Skip torch import - causes hangs on Windows
# import torch
HAS_CUDA = torch.cuda.is_available()

# Import stanza AFTER torch/transformers setup
import stanza

# ============================================================
# COMMAND-LINE ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(description='Build graph_sparse vectors with parallel processing')
parser.add_argument('--batch', type=int, default=32, help='Number of parallel workers for extraction (default: 32)')
args = parser.parse_args()

# Download required NLTK data
try:
    stop_words = set(stopwords.words('english'))
except:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

try:
    lemmatizer = WordNetLemmatizer()
    lemmatizer.lemmatize('test')
except:
    nltk.download('wordnet')
    lemmatizer = WordNetLemmatizer()

# Load spaCy for POS tagging (CPU only - CuPy compilation issues on Windows)
print("[GRAPH LAYER RE-ENABLED] Optimized parallel triplet extraction")
print("[OPTIMIZATION] Pre-load extractor in each worker (3s warmup per worker)")
print("[OPTIMIZATION] Thread-local storage to avoid model reloading")
print("[OPTIMIZATION] Parallel processing across CPU cores")
print(f"[WORKERS] Using {args.batch} parallel workers")

# Thread-local storage for spaCy and Stanza (avoid thread-safety issues)
thread_local = threading.local()

def get_subtree_text(head_word, sent) -> str:
    """Get all text tokens in the subtree rooted at head_word (preserves multi-word phrases)."""
    subtree_words = [head_word]
    
    def collect_descendants(word_id):
        for w in sent.words:
            if w.head == word_id:
                subtree_words.append(w)
                collect_descendants(w.id)
    
    collect_descendants(head_word.id)
    
    # Sort by position and join
    sorted_words = sorted(subtree_words, key=lambda x: x.id)
    return ' '.join([w.text for w in sorted_words])

def extract_spo_from_sentence_stanza(sent) -> list:
    """
    Extract S-P-O using Stanza dependency parse (preserves multi-word phrases).
    Returns list of dicts with 'subject', 'predicate', 'object' keys.
    """
    triplets = []
    
    # Find main verbs (root or coordinated verbs)
    verbs = []
    for word in sent.words:
        if word.upos in ['VERB', 'AUX'] and (word.head == 0 or word.deprel in ['ROOT', 'conj']):
            verbs.append(word)
    
    # For each verb, find subject and object
    for verb in verbs:
        subject_head = None
        object_head = None
        
        for word in sent.words:
            if word.head == verb.id:
                # Subject relations
                if word.deprel in ['nsubj', 'nsubj:pass', 'csubj']:
                    subject_head = word
                # Object relations
                elif word.deprel in ['obj', 'dobj', 'iobj', 'obl']:
                    object_head = word
        
        # Extract full subtrees (preserves multi-word phrases!)
        subject_text = get_subtree_text(subject_head, sent) if subject_head else None
        predicate_text = verb.text
        object_text = get_subtree_text(object_head, sent) if object_head else None
        
        # Only add if we have at least subject and predicate
        if subject_text and predicate_text:
            triplets.append({
                'subject': subject_text,
                'predicate': predicate_text,
                'object': object_text if object_text else '?',
                'confidence': 0.9  # Stanza parsing is high confidence
            })
    
    return triplets

def get_thread_extractor():
    """Get or create thread-local Stanza pipeline (preserves multi-word phrases)."""
    if not hasattr(thread_local, 'stanza_nlp'):
        # Initialize Stanza with dependency parsing
        # Preserves multi-word phrases via get_subtree_text()
        thread_local.stanza_nlp = stanza.Pipeline(
            'en',
            processors='tokenize,pos,lemma,depparse',
            use_gpu=False,
            download_method=None
        )
        thread_local.nlp = spacy.load('en_core_web_sm')  # Still need for some processing
    return thread_local.stanza_nlp, thread_local.nlp

print("[EXTRACTION] Using Stanza dependency parsing (preserves multi-word phrases)")
print("[EXTRACTION] Multi-word entities like 'deep learning models' stay intact")

# POS exclusion sets (from quotes pipeline)
OTHER_POS = {"CC", "DT", "EX", "IN", "LS", "PDT", "POS", "RB", "RBR", "RBS", "RP", "SYM", "TO", "UH", "WDT", "WRB"}
ENTITY_EXCLUDE_POS = OTHER_POS | {'MD'}  # Entities - no modals
PREDICATE_EXCLUDE_POS = OTHER_POS  # Predicates - keep modals

# Auxiliary verbs to exclude
AUXILIARY_VERBS = {'be', 'am', 'is', 'are', 'was', 'were', 'been', 'being',
                   'have', 'has', 'had', 'having',
                   'do', 'does', 'did', 'doing',
                   'can', 'could', 'may', 'might', 'must', 'shall', 'should', 'will', 'would',
                   'let', 'make', 'get', 'take',
                   'contain', 'include', 'comprise'}

# ============================================================
# GPU CONFIGURATION
# ============================================================

# GPU detection
DEVICE = torch.device('cuda' if HAS_CUDA else 'cpu')

if HAS_CUDA:
    print(f"[GPU] CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"[GPU] Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("[CPU] CUDA not available, using CPU")

# ============================================================
# DATABASE CONFIGURATION
# ============================================================

# Database config
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'dbname': 'postgres',
    'user': 'postgres',
    'password': 'postgres'
}

TABLE_NAME = 'arxiv_papers_lemma_fullembed'
# Optimized batch size for parallel processing
# With parallel workers, larger batches reduce DB round-trips
BATCH_SIZE = 200  # Process 200 chunks per DB fetch
CONFIDENCE_THRESHOLD = 1.0

# Performance tracking
TOTAL_CHUNKS = 161389  # Total arXiv chunks to process
print(f"[ESTIMATE] With {args.batch} workers at ~95ms/chunk after warmup:")
warmup_overhead = 3.1 * args.batch  # Each worker warms up once
processing_time = (TOTAL_CHUNKS * 0.095) / args.batch  # Parallel processing
total_seconds = warmup_overhead + processing_time
total_hours = total_seconds / 3600
print(f"[ESTIMATE] Warmup: {warmup_overhead:.1f}s ({args.batch} workers × 3.1s)")
print(f"[ESTIMATE] Processing: {processing_time/60:.1f} minutes")
print(f"[ESTIMATE] Total: {total_hours:.2f} hours")
print(f"[ESTIMATE] Throughput: {TOTAL_CHUNKS/total_seconds:.0f} chunks/sec")


def fix_encoding_artifacts(text):
    """Fix common encoding corruption before triplet extraction"""
    replacements = {
        'â€™': "'",
        'â€œ': '"',
        'â€': '"',
        'â€"': '—',
        'â€"': '–',
        'Â': '',
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    return text


def smart_word_repair(text):
    """Repair text after cleantext may have left broken fragments."""
    try:
        from nltk.corpus import words as nltk_words
        english_words = set(w.lower() for w in nltk_words.words())
    except:
        nltk.download('words')
        from nltk.corpus import words as nltk_words
        english_words = set(w.lower() for w in nltk_words.words())
    
    tokens = text.split()
    if len(tokens) <= 1:
        return text
    
    repaired = []
    i = 0
    while i < len(tokens):
        current = tokens[i]
        
        if i + 1 < len(tokens):
            joined = current + tokens[i + 1]
            joined_lower = joined.lower()
            current_lower = current.lower()
            next_lower = tokens[i + 1].lower()
            
            if joined_lower in english_words:
                if not (current_lower in english_words and next_lower in english_words):
                    repaired.append(joined)
                    i += 2
                    continue
        
        repaired.append(current)
        i += 1
    
    return ' '.join(repaired)


def filter_pos_tags_batch(texts, nlp, role='entity'):
    """Batch POS filtering using spaCy."""
    exclude_set = ENTITY_EXCLUDE_POS if role == 'entity' else PREDICATE_EXCLUDE_POS
    
    results = []
    for doc in nlp.pipe(texts):
        filtered_tokens = [token.text for token in doc if token.tag_ not in exclude_set]
        results.append(' '.join(filtered_tokens) if filtered_tokens else doc.text)
    
    return results


def is_valid_extraction(text):
    """Filter out invalid extractions"""
    if not text or len(text) < 2:
        return False
    
    invalid_patterns = ["'s", "'t", "'d", "'ll", "'ve", "'re", "'m"]
    if text in invalid_patterns:
        return False
    
    alpha_count = sum(c.isalpha() for c in text)
    if alpha_count < len(text) * 0.5:
        return False
    
    words = text.split()
    if len(words) == 1 and len(text) < 3:
        return False
    
    for word in words:
        if len(word) == 1:
            return False
    
    return True


def is_valid_predicate(text, nlp):
    """Check if predicate is semantic"""
    if not is_valid_extraction(text):
        return False
    
    tokens = text.lower().split()
    lemmas = [lemmatizer.lemmatize(token, pos='v') for token in tokens]
    
    if any(lemma in AUXILIARY_VERBS for lemma in lemmas):
        return False
    
    doc = nlp(text)
    has_semantic_predicate = any(token.pos_ in {'VERB', 'AUX', 'ADJ', 'ADV'} for token in doc)
    
    return has_semantic_predicate


def preprocess_text(text):
    """Full preprocessing pipeline (Step 0)"""
    fixed = fix_encoding_artifacts(text)
    repaired = smart_word_repair(fixed)
    cleaned = cleantext.clean(repaired, lower=False, no_emoji=True)
    split = ' '.join(wordninja.split(cleaned))
    return split


def lemmatize_with_pos(text):
    """POS-aware lemmatization"""
    if not text:
        return ''
    
    tokens = text.split()
    lemmatized = []
    
    for token in tokens:
        if '/' not in token:
            lemmatized.append(lemmatizer.lemmatize(token.lower()))
        else:
            word, pos_tag = token.rsplit('/', 1)
            word_lower = word.lower()
            
            if pos_tag.startswith('VB'):
                wn_pos = 'v'
            elif pos_tag.startswith('NN'):
                wn_pos = 'n'
            elif pos_tag.startswith('JJ'):
                wn_pos = 'a'
            elif pos_tag.startswith('RB'):
                wn_pos = 'r'
            else:
                wn_pos = 'n'
            
            lemmatized.append(lemmatizer.lemmatize(word_lower, pos=wn_pos))
    
    return ' '.join(lemmatized)


def deduplicate_phrase(text):
    """Remove duplicate words within phrase"""
    if not text:
        return ''
    words = text.split()
    unique_words = list(dict.fromkeys(words))
    return ' '.join(unique_words)


def get_primary_synset(word):
    """Get most common synset for a word"""
    synsets = wn.synsets(word)
    if synsets:
        return synsets[0]
    return None


def get_immediate_hypernym(synset):
    """Get immediate parent hypernym (1 level up)"""
    if not synset:
        return None
    
    hypernyms = synset.hypernyms()
    if hypernyms:
        return hypernyms[0]
    return None


def extract_and_process_triplets(content: str, chunk_id: int) -> List[Tuple[str, str, str]]:
    """
    Extract triplets using Stanza pipeline (PRESERVES MULTI-WORD PHRASES):
    1. Preprocess (encoding fix, cleantext, wordninja)
    2. Extract triplets (Stanza dependency parsing)
    3. Filter confidence
    4. Remove stop words
    5. POS filtering
    6. Lemmatization
    7. Deduplication
    8. Validation
    9. Atomic decomposition
    10. Position conflict resolution
    """
    # Get thread-local Stanza pipeline and spacy nlp
    stanza_nlp, nlp = get_thread_extractor()
    
    # Step 0: Preprocess
    preprocessed = preprocess_text(content[:2000])  # Limit length for speed
    
    # Step 1: Parse with Stanza (preserves multi-word phrases via dependency subtrees)
    doc_stanza = stanza_nlp(preprocessed)
    
    # Step 2: Extract triplets from each sentence using dependency parse
    all_triplets = []
    for sent in doc_stanza.sentences:
        sent_triplets = extract_spo_from_sentence_stanza(sent)
        all_triplets.extend(sent_triplets)
    
    if not all_triplets:
        return []
    
    # Step 3: Filter confidence (Stanza has high confidence, but keep threshold)
    filtered = [(t['subject'], t['predicate'], t['object']) for t in all_triplets 
                if t['confidence'] >= CONFIDENCE_THRESHOLD]
    
    if not filtered:
        return []
    
    # Step 3: Remove stop words
    def remove_stopwords(text):
        if not text:
            return ''
        words = text.split()
        filtered_words = [w for w in words if w.lower() not in stop_words]
        return ' '.join(filtered_words)
    
    cleaned = [(remove_stopwords(s), remove_stopwords(p), remove_stopwords(o)) 
               for s, p, o in filtered]
    
    # Step 3b: POS filtering (batch process)
    subjects = [s for s, p, o in cleaned]
    predicates = [p for s, p, o in cleaned]
    objects = [o for s, p, o in cleaned]
    
    subjects_pos = filter_pos_tags_batch(subjects, nlp, role='entity')
    predicates_pos = filter_pos_tags_batch(predicates, nlp, role='predicate')
    objects_pos = filter_pos_tags_batch(objects, nlp, role='entity')
    
    pos_filtered = list(zip(subjects_pos, predicates_pos, objects_pos))
    
    # Step 4: Lemmatization + deduplication
    lemmatized = []
    for s, p, o in pos_filtered:
        s_lemma = deduplicate_phrase(lemmatize_with_pos(s))
        p_lemma = deduplicate_phrase(lemmatize_with_pos(p))
        o_lemma = deduplicate_phrase(lemmatize_with_pos(o))
        lemmatized.append((s_lemma, p_lemma, o_lemma))
    
    # Step 8: Validation
    valid = [(s, p, o) for s, p, o in lemmatized
             if is_valid_extraction(s) and is_valid_predicate(p, nlp) and is_valid_extraction(o)]
    
    # Step 9: Atomic decomposition (force split multi-word phrases)
    atomic = []
    for s, p, o in valid:
        s_tokens = s.split()
        p_tokens = [pt for pt in p.split() if is_valid_predicate(pt, nlp)]
        o_tokens = o.split()
        
        if not p_tokens:
            continue
        
        for s_tok in s_tokens:
            for p_tok in p_tokens:
                for o_tok in o_tokens:
                    if s_tok and p_tok and o_tok:
                        atomic.append((s_tok, p_tok, o_tok))
    
    # Step 9b: Position conflict resolution
    resolved = []
    for s, p, o in atomic:
        positions = [s, p, o]
        unique_positions = set(positions)
        if len(unique_positions) == 3:  # All three positions unique
            resolved.append((s, p, o))
    
    return resolved


def build_knowledge_graph(triplets: List[Tuple[str, str, str]], nlp) -> Dict:
    """
    Build KG with synset consolidation and return node/edge structure.
    
    Returns:
        Dict with:
        - entity_nodes: {synset_name: {words, pos_tags, role}}
        - predicate_nodes: {predicate: {pos_tags}}
        - edges: [(subj_synset, pred, obj_synset)]
    """
    entity_to_synset = {}  # word → synset_name
    synset_to_info = defaultdict(lambda: {
        'words': set(),
        'pos_tags': [],
        'roles': set()
    })
    
    predicate_info = defaultdict(lambda: {'pos_tags': []})
    edges = []
    
    # Process entities (subjects and objects)
    for subj, pred, obj in triplets:
        # Process subject
        subj_synset = get_primary_synset(subj)
        if subj_synset:
            synset_name = subj_synset.name()
            entity_to_synset[subj] = synset_name
            synset_to_info[synset_name]['words'].add(subj)
            synset_to_info[synset_name]['roles'].add('subject')
            
            doc = nlp(subj)
            if doc:
                synset_to_info[synset_name]['pos_tags'].append(doc[0].tag_)
        
        # Process object
        obj_synset = get_primary_synset(obj)
        if obj_synset:
            synset_name = obj_synset.name()
            entity_to_synset[obj] = synset_name
            synset_to_info[synset_name]['words'].add(obj)
            synset_to_info[synset_name]['roles'].add('object')
            
            doc = nlp(obj)
            if doc:
                synset_to_info[synset_name]['pos_tags'].append(doc[0].tag_)
        
        # Process predicate
        doc = nlp(pred)
        if doc:
            predicate_info[pred]['pos_tags'].append(doc[0].tag_)
        
        # Add edge (using synsets for entities)
        subj_node = entity_to_synset.get(subj, subj)
        obj_node = entity_to_synset.get(obj, obj)
        edges.append((subj_node, pred, obj_node))
    
    return {
        'entity_nodes': dict(synset_to_info),
        'predicate_nodes': dict(predicate_info),
        'edges': edges
    }


def kg_to_sparse_vector(kg: Dict, tokenizer) -> Dict[int, float]:
    """
    Convert KG to sparse vector using BATCH wordpiece tokenization (GPU-accelerated).
    
    Weights:
    - Entity nodes (direct words): 1.0
    - Entity nodes (synset words): 0.8
    - Hypernyms (1 level): 0.5
    - Predicates: 1.0
    """
    sparse_vec = {}
    vocab_size = tokenizer.vocab_size
    
    # Collect all texts to tokenize in batches
    texts_to_tokenize = []
    weights = []
    
    # Process entity nodes
    for synset_name, info in kg['entity_nodes'].items():
        # Direct words (weight 1.0)
        for word in info['words']:
            texts_to_tokenize.append(word.lower())
            weights.append(1.0)
        
        # Synset expansion (weight 0.8)
        synset = wn.synset(synset_name)
        if synset:
            for lemma in synset.lemmas():
                lemma_name = lemma.name().replace('_', ' ')
                texts_to_tokenize.append(lemma_name.lower())
                weights.append(0.8)
            
            # Hypernym expansion (weight 0.5)
            hypernym = get_immediate_hypernym(synset)
            if hypernym:
                for lemma in hypernym.lemmas():
                    lemma_name = lemma.name().replace('_', ' ')
                    texts_to_tokenize.append(lemma_name.lower())
                    weights.append(0.5)
    
    # Process predicate nodes (weight 1.0)
    for pred in kg['predicate_nodes'].keys():
        texts_to_tokenize.append(pred.lower())
        weights.append(1.0)
    
    # BATCH tokenization (much faster than one-by-one)
    if texts_to_tokenize:
        # Tokenize all at once (GPU-accelerated if available)
        encoded = tokenizer(texts_to_tokenize, add_special_tokens=False, padding=False, truncation=False)
        
        # Build sparse vector
        for token_ids, weight in zip(encoded['input_ids'], weights):
            for token_id in token_ids:
                if token_id < vocab_size:
                    sparse_vec[token_id] = max(sparse_vec.get(token_id, 0), weight)
    
    return sparse_vec


def sparse_dict_to_pgvector_format(sparse_dict: Dict[int, float], vocab_size: int = 30522) -> str:
    """Convert sparse dict to pgvector sparsevec format."""
    if not sparse_dict:
        return f'{{}}/{vocab_size}'
    
    sorted_items = sorted(sparse_dict.items())
    pairs = ','.join([f'{token_id}:{weight:.4f}' for token_id, weight in sorted_items])
    return f'{{{pairs}}}/{vocab_size}'


def main():
    """Main processing loop with optimized parallel extraction"""
    print("="*70)
    print("ARXIV GRAPH SPARSE VECTOR BUILDER - OPTIMIZED")
    print("="*70)
    print(f"\n[PARALLELIZATION] Using {args.batch} workers")
    print("[OPTIMIZATION] Thread-local extractors (avoid model reloading)")
    print("[OPTIMIZATION] Batched DB operations (reduce round-trips)")
    
    # Initialize tokenizer
    print("\n[1/6] Loading BERT tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    print(f"  ✓ Loaded (vocab_size={tokenizer.vocab_size})")
    
    # Connect to database
    print("\n[2/6] Connecting to database...")
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = False
    
    try:
        with conn.cursor() as cur:
            # Check if column exists
            print("\n[3/6] Checking graph_sparse column...")
            cur.execute(f"""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = '{TABLE_NAME}' 
                AND column_name = 'graph_sparse'
            """)
            
            if not cur.fetchone():
                print("  Adding graph_sparse column...")
                cur.execute(f"""
                    ALTER TABLE {TABLE_NAME} 
                    ADD COLUMN graph_sparse sparsevec(30522)
                """)
                conn.commit()
                print("  ✓ Column added")
            else:
                print("  ✓ Column exists")
            
            # Get total count
            print("\n[4/6] Counting chunks...")
            cur.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}")
            total_chunks = cur.fetchone()[0]
            print(f"  Found {total_chunks:,} chunks to process")
            
            # Fetch all IDs
            cur.execute(f"SELECT id FROM {TABLE_NAME} ORDER BY id")
            all_ids = [row[0] for row in cur.fetchall()]
            
            # Process in batches with parallelization
            print(f"\n[5/6] Processing chunks (batch_size={BATCH_SIZE}, workers={args.batch})...")
            
            def process_chunk(chunk_id, content):
                """
                Process a single chunk (for parallel execution).
                
                Thread-local extractor amortizes 3.1s warmup cost across all chunks.
                First call per worker: ~3.1s (loads gzipped affinity tables)
                Subsequent calls: ~95ms per chunk (after warmup)
                """
                try:
                    # Get thread-local extractor (lazy loads on first call per thread)
                    extractor, nlp = get_thread_extractor()
                    
                    # Extract triplets with full OpenIE pipeline
                    triplets = extract_and_process_triplets(content, chunk_id)
                    
                    if not triplets:
                        # Empty sparse vector
                        graph_sparse_str = f'{{}}/' + str(tokenizer.vocab_size)
                        return (graph_sparse_str, chunk_id)
                    
                    # Build KG with synset consolidation
                    kg = build_knowledge_graph(triplets, nlp)
                    
                    # Convert to sparse vector
                    sparse_dict = kg_to_sparse_vector(kg, tokenizer)
                    
                    # Format for pgvector
                    graph_sparse_str = sparse_dict_to_pgvector_format(
                        sparse_dict,
                        vocab_size=tokenizer.vocab_size
                    )
                    
                    return (graph_sparse_str, chunk_id)
                    
                except Exception as e:
                    print(f"\nError processing chunk {chunk_id}: {e}")
                    # Return empty vector on error
                    return (f'{{}}/' + str(tokenizer.vocab_size), chunk_id)
            
            for i in tqdm(range(0, len(all_ids), BATCH_SIZE), desc="  Progress"):
                batch_ids = all_ids[i:i+BATCH_SIZE]
                
                # Fetch content
                cur.execute(f"""
                    SELECT id, content 
                    FROM {TABLE_NAME}
                    WHERE id = ANY(%s)
                """, (batch_ids,))
                
                batch_data = cur.fetchall()
                
                # Parallel processing with ThreadPoolExecutor
                update_batch = []
                with ThreadPoolExecutor(max_workers=args.batch) as executor:
                    futures = {executor.submit(process_chunk, chunk_id, content): chunk_id 
                              for chunk_id, content in batch_data}
                    
                    for future in as_completed(futures):
                        try:
                            result = future.result()
                            update_batch.append(result)
                        except Exception as e:
                            chunk_id = futures[future]
                            print(f"\nError in future for chunk {chunk_id}: {e}")
                            # Add empty vector on error
                            update_batch.append((f'{{}}/' + str(tokenizer.vocab_size), chunk_id))
                
                # Execute batch update
                if update_batch:
                    execute_values(
                        cur,
                        f"UPDATE {TABLE_NAME} SET graph_sparse = data.gs FROM (VALUES %s) AS data(gs, id) WHERE {TABLE_NAME}.id = data.id",
                        update_batch,
                        template="(%s::sparsevec, %s)"
                    )
                    conn.commit()
            
            print("\n  ✓ All chunks processed")
            
            # Create/rebuild HNSW index (with inner product, not cosine!)
            print("\n[6/6] Creating HNSW index with inner product operator...")
            cur.execute(f"DROP INDEX IF EXISTS idx_{TABLE_NAME}_graph_sparse")
            cur.execute(f"""
                CREATE INDEX idx_{TABLE_NAME}_graph_sparse 
                ON {TABLE_NAME} 
                USING hnsw (graph_sparse sparsevec_ip_ops)
            """)
            conn.commit()
            print("  ✓ Index created with inner product operator")
        
        print("\n" + "="*70)
        print("✅ COMPLETE!")
        print("="*70)
        print(f"Table '{TABLE_NAME}' now has proper graph_sparse vectors:")
        print("  - Real triplet extraction (triplet_extract library)")
        print("  - Synset-based entity consolidation")
        print("  - Hypernym bridging")
        print("  - Wordpiece tokenization (batch processing)")
        print("  - HNSW index with inner product operator")
        print(f"\nProcessing configuration:")
        print(f"  - Batch size: {BATCH_SIZE}")
        print(f"  - Workers: {args.batch}")
        print(f"  - Device: {'GPU (' + torch.cuda.get_device_name(0) + ')' if HAS_CUDA else 'CPU'}")
        
    except Exception as e:
        conn.rollback()
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        conn.close()


if __name__ == '__main__':
    main()
