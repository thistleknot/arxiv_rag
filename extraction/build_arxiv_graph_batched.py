"""Build Graph Sparse Vectors - BATCHED sentence extraction for GPU efficiency"""

# Disable torch dynamo to avoid import hang
import os
os.environ['PYTORCH_JIT'] = '0'
os.environ['TORCH_COMPILE_DISABLE'] = '1'

import psycopg2
from psycopg2.extras import execute_batch
import re
import string
import argparse
import time
from collections import defaultdict
from typing import List, Tuple, Dict
from tqdm import tqdm
import spacy
from triplet_extract import OpenIEExtractor
import nltk
from nltk.corpus import stopwords, wordnet as wn
from nltk.stem import WordNetLemmatizer
from transformers import AutoTokenizer
import torch

HAS_CUDA = torch.cuda.is_available()

# Load resources
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

# GPU detection
if HAS_CUDA:
    print(f"[GPU] CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"[GPU] Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("[GPU] Deep search mode with BATCHED sentence extraction")
else:
    print("[CPU] No GPU detected")

# Load spaCy
print("\n[1/8] Loading spaCy...")
nlp = spacy.load('en_core_web_sm')

# Load extractor (will be used for batched extraction)
print("[2/8] Loading extractor...")
extractor = OpenIEExtractor(
    nlp=nlp,
    enable_clause_split=True,
    enable_entailment=True,
    min_confidence=0.3,
    fast=False,
    speed_preset="balanced",
    high_quality=True,
    deep_search=True  # GPU mode
)

# Load tokenizer
print("[3/8] Loading BERT tokenizer...")
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
print(f"  ✓ Vocab size: {tokenizer.vocab_size}")

# Database config
DB_CONFIG = {
    'dbname': 'postgres',
    'user': 'postgres',
    'password': 'postgres',
    'host': 'localhost',
    'port': 5432
}

# Constants
CONFIDENCE_THRESHOLD = 0.3
ENTITY_EXCLUDE_POS = {'DT', 'IN', 'TO', 'CC', 'PRP', 'PRP$', 'WP', 'WP$', 'WDT', 'WRB'}
PREDICATE_EXCLUDE_POS = {'DT', 'PRP', 'PRP$', 'WP', 'WP$'}
AUXILIARY_VERBS = {'be', 'is', 'are', 'was', 'were', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did'}

# Validation patterns
VALID_PATTERN = re.compile(r'^[a-zA-Z0-9\s\-]+$')
def is_valid_extraction(text: str) -> bool:
    if not text or len(text.strip()) < 2:
        return False
    if not VALID_PATTERN.match(text):
        return False
    tokens = text.split()
    if len(tokens) > 5:
        return False
    return True

def preprocess_text(text: str) -> str:
    """Minimal preprocessing"""
    text = text.encode('utf-8', 'ignore').decode('utf-8')
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def filter_pos_tags_batch(texts, nlp, role='entity'):
    """Batch POS filtering"""
    exclude_set = ENTITY_EXCLUDE_POS if role == 'entity' else PREDICATE_EXCLUDE_POS
    results = []
    for doc in nlp.pipe(texts):
        filtered_tokens = [token.text for token in doc if token.tag_ not in exclude_set]
        results.append(' '.join(filtered_tokens) if filtered_tokens else doc.text)
    return results

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

def build_knowledge_graph(triplets: List[Tuple[str, str, str]], nlp) -> Dict:
    """Build KG with synset consolidation"""
    entity_to_synset = {}
    synset_to_info = defaultdict(lambda: {'entities': set(), 'count': 0, 'pos_tags': []})
    
    for subj, pred, obj in triplets:
        for entity in [subj, obj]:
            if entity not in entity_to_synset:
                doc = nlp(entity)
                if doc:
                    synsets = wn.synsets(entity.replace(' ', '_'))
                    if synsets:
                        synset_name = synsets[0].name()
                        entity_to_synset[entity] = synset_name
                        synset_to_info[synset_name]['entities'].add(entity)
                        synset_to_info[synset_name]['pos_tags'].append(doc[0].tag_)
                    else:
                        entity_to_synset[entity] = entity
                        synset_to_info[entity]['entities'].add(entity)
                else:
                    entity_to_synset[entity] = entity
                    synset_to_info[entity]['entities'].add(entity)
    
    graph = defaultdict(lambda: defaultdict(set))
    for subj, pred, obj in triplets:
        subj_synset = entity_to_synset.get(subj, subj)
        obj_synset = entity_to_synset.get(obj, obj)
        doc = nlp(pred)
        if doc:
            synsets = wn.synsets(pred.replace(' ', '_'), pos=wn.VERB)
            if synsets:
                pred_synset = synsets[0].name()
            else:
                pred_synset = pred
        else:
            pred_synset = pred
        graph[subj_synset][pred_synset].add(obj_synset)
    
    return dict(graph)

def kg_to_sparse_vector(kg: Dict, tokenizer) -> Dict[int, int]:
    """Convert KG to sparse vector"""
    sparse_dict = {}
    for subj, pred_dict in kg.items():
        subj_tokens = tokenizer.encode(subj, add_special_tokens=False)
        for token_id in subj_tokens:
            sparse_dict[token_id] = sparse_dict.get(token_id, 0) + 1
        for pred, objs in pred_dict.items():
            pred_tokens = tokenizer.encode(pred, add_special_tokens=False)
            for token_id in pred_tokens:
                sparse_dict[token_id] = sparse_dict.get(token_id, 0) + 1
            for obj in objs:
                obj_tokens = tokenizer.encode(obj, add_special_tokens=False)
                for token_id in obj_tokens:
                    sparse_dict[token_id] = sparse_dict.get(token_id, 0) + 1
    return sparse_dict

def sparse_dict_to_pgvector_format(sparse_dict: Dict[int, int], vocab_size: int) -> str:
    """Convert to pgvector sparse format"""
    if not sparse_dict:
        return f'{{}}/' + str(vocab_size)
    indices = sorted(sparse_dict.keys())
    values = [sparse_dict[idx] for idx in indices]
    indices_str = ','.join(map(str, indices))
    values_str = ','.join(map(str, values))
    return f'{{{indices_str}}}/{{{values_str}}}/' + str(vocab_size)

def process_chunks_batched(chunk_data: List[Tuple[int, str]], batch_size: int = 64) -> Dict[int, str]:
    """
    Process chunks with batched sentence extraction
    
    Args:
        chunk_data: List of (chunk_id, content) tuples
        batch_size: Max sentences per GPU batch
    
    Returns:
        Dict mapping chunk_id -> graph_sparse_str
    """
    print(f"\n[4/8] Preprocessing {len(chunk_data)} chunks...")
    
    # Step 1: Extract all sentences and build inverted index
    sentence_to_chunks = []  # List of (sentence, chunk_id)
    chunk_to_sentence_indices = defaultdict(list)  # chunk_id -> list of sentence indices
    
    for chunk_id, content in chunk_data:
        preprocessed = preprocess_text(content[:2000])
        doc = nlp(preprocessed)
        sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 20]
        
        for sent in sentences:
            sent_idx = len(sentence_to_chunks)
            sentence_to_chunks.append((sent, chunk_id))
            chunk_to_sentence_indices[chunk_id].append(sent_idx)
    
    print(f"  ✓ Extracted {len(sentence_to_chunks)} sentences from {len(chunk_data)} chunks")
    print(f"  ✓ Average: {len(sentence_to_chunks) / len(chunk_data):.1f} sentences/chunk")
    
    # Step 2: Batch extract triplets from ALL sentences
    print(f"\n[5/8] Extracting triplets (batch_size={batch_size})...")
    sentence_triplets = []  # List of triplets per sentence
    
    sentences_only = [sent for sent, _ in sentence_to_chunks]
    
    start_time = time.time()
    for i in tqdm(range(0, len(sentences_only), batch_size), desc="  Batches"):
        batch = sentences_only[i:i+batch_size]
        
        # Extract per sentence in batch
        for sent in batch:
            triplets = extractor.extract_triplet_objects(sent)
            filtered = [(t.subject, t.relation, t.object) for t in triplets 
                       if t.confidence >= CONFIDENCE_THRESHOLD]
            sentence_triplets.append(filtered)
    
    extract_time = time.time() - start_time
    print(f"  ✓ Extracted in {extract_time:.1f}s ({extract_time / len(sentences_only):.3f}s per sentence)")
    
    # Step 3: Process triplets (stopwords, POS, lemma, validation)
    print(f"\n[6/8] Processing triplets...")
    
    def remove_stopwords(text):
        if not text:
            return ''
        words = text.split()
        filtered_words = [w for w in words if w.lower() not in stop_words]
        return ' '.join(filtered_words)
    
    # Collect all unique terms for batch processing
    all_subjects = set()
    all_predicates = set()
    all_objects = set()
    
    for triplets in sentence_triplets:
        for s, p, o in triplets:
            all_subjects.add(remove_stopwords(s))
            all_predicates.add(remove_stopwords(p))
            all_objects.add(remove_stopwords(o))
    
    # Batch POS filtering
    subjects_pos_map = dict(zip(all_subjects, filter_pos_tags_batch(list(all_subjects), nlp, role='entity')))
    predicates_pos_map = dict(zip(all_predicates, filter_pos_tags_batch(list(all_predicates), nlp, role='predicate')))
    objects_pos_map = dict(zip(all_objects, filter_pos_tags_batch(list(all_objects), nlp, role='entity')))
    
    # Process each sentence's triplets
    processed_sentence_triplets = []
    for triplets in sentence_triplets:
        if not triplets:
            processed_sentence_triplets.append([])
            continue
        
        # Apply POS filtering
        pos_filtered = []
        for s, p, o in triplets:
            s_clean = remove_stopwords(s)
            p_clean = remove_stopwords(p)
            o_clean = remove_stopwords(o)
            
            s_pos = subjects_pos_map.get(s_clean, s_clean)
            p_pos = predicates_pos_map.get(p_clean, p_clean)
            o_pos = objects_pos_map.get(o_clean, o_clean)
            
            pos_filtered.append((s_pos, p_pos, o_pos))
        
        # Lemmatization
        lemmatized = []
        for s, p, o in pos_filtered:
            s_lemma = ' '.join([lemmatizer.lemmatize(w.lower()) for w in s.split()])
            p_lemma = ' '.join([lemmatizer.lemmatize(w.lower(), pos='v') for w in p.split()])
            o_lemma = ' '.join([lemmatizer.lemmatize(w.lower()) for w in o.split()])
            lemmatized.append((s_lemma, p_lemma, o_lemma))
        
        # Validation
        valid = [(s, p, o) for s, p, o in lemmatized
                 if is_valid_extraction(s) and is_valid_predicate(p, nlp) and is_valid_extraction(o)]
        
        # Deduplication
        unique = list(set(valid))
        processed_sentence_triplets.append(unique)
    
    print(f"  ✓ Processed {sum(len(t) for t in processed_sentence_triplets)} total triplets")
    
    # Step 4: Aggregate triplets per chunk using inverted index
    print(f"\n[7/8] Aggregating triplets per chunk...")
    chunk_results = {}
    
    for chunk_id in chunk_to_sentence_indices.keys():
        sent_indices = chunk_to_sentence_indices[chunk_id]
        
        # Collect all triplets for this chunk
        chunk_triplets = []
        for sent_idx in sent_indices:
            chunk_triplets.extend(processed_sentence_triplets[sent_idx])
        
        # Deduplicate at chunk level
        chunk_triplets = list(set(chunk_triplets))
        
        if not chunk_triplets:
            graph_sparse_str = f'{{}}/' + str(tokenizer.vocab_size)
            chunk_results[chunk_id] = graph_sparse_str
            continue
        
        # Build KG
        kg = build_knowledge_graph(chunk_triplets, nlp)
        
        # Convert to sparse vector
        sparse_dict = kg_to_sparse_vector(kg, tokenizer)
        graph_sparse_str = sparse_dict_to_pgvector_format(sparse_dict, tokenizer.vocab_size)
        
        chunk_results[chunk_id] = graph_sparse_str
    
    print(f"  ✓ Generated {len(chunk_results)} chunk graphs")
    
    return chunk_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=256, help='Number of chunks to process')
    parser.add_argument('--batch', type=int, default=64, help='Sentence batch size for GPU')
    args = parser.parse_args()
    
    print("=" * 70)
    print("ARXIV GRAPH SPARSE VECTOR BUILDER - BATCHED MODE")
    print("=" * 70)
    
    # Connect to database
    print("\n[8/8] Connecting to database...")
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    # Get chunks
    cur.execute(f"""
        SELECT id, content 
        FROM arxiv_papers_lemma_fullembed 
        WHERE content IS NOT NULL 
        LIMIT {args.limit}
    """)
    chunk_data = cur.fetchall()
    
    print(f"  ✓ Found {len(chunk_data)} chunks to process")
    
    # Process chunks with batched extraction
    print(f"\n{'='*70}")
    print(f"PROCESSING WITH BATCHED SENTENCE EXTRACTION")
    print(f"  Chunks: {len(chunk_data)}")
    print(f"  Sentence batch size: {args.batch}")
    print(f"{'='*70}")
    
    overall_start = time.time()
    chunk_results = process_chunks_batched(chunk_data, batch_size=args.batch)
    overall_time = time.time() - overall_start
    
    print(f"\n{'='*70}")
    print(f"PERFORMANCE SUMMARY")
    print(f"{'='*70}")
    print(f"  Total time: {overall_time:.1f}s")
    print(f"  Chunks processed: {len(chunk_results)}")
    print(f"  Average per chunk: {overall_time / len(chunk_results):.2f}s")
    print(f"  Estimated for 161,389 chunks: {(overall_time / len(chunk_results)) * 161389 / 3600:.1f}h")
    
    # Update database
    print(f"\n[SKIP] Database update (test mode)")
    
    cur.close()
    conn.close()
    
    print(f"\n✅ COMPLETE!")

if __name__ == '__main__':
    main()
