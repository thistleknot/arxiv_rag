"""
Extract Graph Sparse Vectors to msgpack

Standalone extraction that saves to msgpack file instead of PostgreSQL.
Creates both forward index (chunk_id → sparse_vector) and inverse index
(token_id → chunk_ids) based on BERT wordpiece tokenizer.

Usage:
    python extract_graph_to_msgpack.py --batch 32 --limit 1000  # Test on 1000 chunks
    python extract_graph_to_msgpack.py --batch 32                # Full extraction (161k chunks)
    python extract_graph_to_msgpack.py --batch 32 --output custom_name.msgpack

Output Structure:
    {
        'metadata': {
            'vocab_size': 30522,
            'total_chunks': 161389,
            'created': '2026-02-02T...',
            'tokenizer': 'bert-base-uncased'
        },
        'forward_index': {
            'chunk_id_1': {
                'sparse_vector': {token_id: weight, ...},
                'triplet_count': 5,
                'kg_nodes': 12
            },
            ...
        },
        'inverse_index': {
            token_id_1: [chunk_id_1, chunk_id_3, ...],
            token_id_2: [chunk_id_2, chunk_id_5, ...],
            ...
        }
    }
"""

# Disable torch dynamo to avoid import hang
import os
os.environ['PYTORCH_JIT'] = '0'
os.environ['TORCH_COMPILE_DISABLE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'

import argparse
import psycopg2
import threading
import msgpack
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from transformers import AutoTokenizer

# Import core extraction functions
from triplet_extract import extract
import spacy
from triplet_extract.extractor import OpenIEExtractor

# Database configuration (read-only for fetching chunks)
DB_CONFIG = {
    'dbname': 'postgres',
    'user': 'postgres',
    'password': 'postgres',
    'host': 'localhost',
    'port': 5432
}

TABLE_NAME = 'arxiv_papers_lemma_fullembed'
BATCH_SIZE = 200  # Fetch 200 chunks per DB query

# Thread-local storage for extractors
thread_local = threading.local()


def get_thread_extractor():
    """Get or create thread-local OpenIEExtractor instance (amortizes 3.1s warmup)"""
    if not hasattr(thread_local, 'extractor'):
        nlp = spacy.load('en_core_web_sm')
        thread_local.extractor = OpenIEExtractor(
            nlp=nlp,
            enable_clause_split=True,
            enable_entailment=True,
            min_confidence=0.3,
            fast=True,
            speed_preset="fast",
            high_quality=False,
            deep_search=False
        )
        thread_local.nlp = nlp
    return thread_local.extractor, thread_local.nlp


def extract_and_process_triplets(text, chunk_id):
    """Extract triplets from text using OpenIE (adapted from build_arxiv_graph_sparse.py)"""
    try:
        triplet_objects = extract(text)
        
        if not triplet_objects:
            return []
        
        triplets = []
        for tobj in triplet_objects:
            try:
                subj = str(tobj.subject).strip() if tobj.subject else None
                pred = str(tobj.relation).strip() if tobj.relation else None
                obj = str(tobj.object).strip() if tobj.object else None
                
                if subj and pred and obj:
                    # Basic cleanup
                    if len(subj) > 100:
                        subj = subj[:100]
                    if len(pred) > 50:
                        pred = pred[:50]
                    if len(obj) > 100:
                        obj = obj[:100]
                    
                    triplets.append((subj, pred, obj))
            except Exception:
                continue
        
        return triplets
    except Exception as e:
        print(f"\nError extracting triplets from chunk {chunk_id}: {e}")
        return []


def build_knowledge_graph(triplets, nlp):
    """Build knowledge graph from triplets with synset consolidation"""
    from collections import defaultdict
    
    kg = {
        'nodes': set(),
        'edges': []
    }
    
    synsets = defaultdict(set)
    
    for subj, pred, obj in triplets:
        # Add nodes
        kg['nodes'].add(subj)
        kg['nodes'].add(obj)
        
        # Add edge
        kg['edges'].append((subj, pred, obj))
        
        # Build synset map (entities that appear together)
        synsets[subj].add(obj)
        synsets[obj].add(subj)
    
    return kg


def kg_to_sparse_vector(kg, tokenizer):
    """Convert KG to sparse vector using BERT tokenizer"""
    sparse_dict = defaultdict(float)
    
    # Weight nodes
    for node in kg['nodes']:
        tokens = tokenizer.encode(node, add_special_tokens=False)
        for token_id in tokens:
            sparse_dict[token_id] += 1.0
    
    # Weight edges (predicates)
    for subj, pred, obj in kg['edges']:
        pred_tokens = tokenizer.encode(pred, add_special_tokens=False)
        for token_id in pred_tokens:
            sparse_dict[token_id] += 0.5  # Lower weight for predicates
    
    # Normalize
    if sparse_dict:
        max_weight = max(sparse_dict.values())
        if max_weight > 0:
            for token_id in sparse_dict:
                sparse_dict[token_id] /= max_weight
    
    return dict(sparse_dict)


def process_chunk(chunk_id, content, tokenizer):
    """
    Process a single chunk: extract triplets → KG → sparse vector
    
    Returns: (chunk_id, result_dict) or (chunk_id, None) on error
    """
    try:
        import time
        start = time.time()
        
        # Get thread-local extractor (lazy loads on first call per thread)
        extractor, nlp = get_thread_extractor()
        
        # Count sentences in chunk
        sentences = [s for s in content.split('.') if len(s.strip()) > 10]
        
        # Extract triplets
        triplets = extract_and_process_triplets(content, chunk_id)
        
        elapsed = time.time() - start
        print(f"[Chunk {chunk_id}] {len(sentences)} sentences, {len(triplets)} triplets, {elapsed:.2f}s")
        
        if not triplets:
            return (chunk_id, {
                'sparse_vector': {},
                'triplet_count': 0,
                'kg_nodes': 0
            })
        
        # Build KG
        kg = build_knowledge_graph(triplets, nlp)
        
        # Convert to sparse vector
        sparse_dict = kg_to_sparse_vector(kg, tokenizer)
        
        return (chunk_id, {
            'sparse_vector': sparse_dict,
            'triplet_count': len(triplets),
            'kg_nodes': len(kg['nodes'])
        })
        
    except Exception as e:
        print(f"\nError processing chunk {chunk_id}: {e}")
        return (chunk_id, None)


def build_inverse_index(forward_index):
    """
    Build inverse index: token_id → list of chunk_ids
    
    Enables fast lookup: "Which chunks contain this token?"
    """
    inverse_index = defaultdict(list)
    
    for chunk_id, data in forward_index.items():
        if data and 'sparse_vector' in data:
            for token_id in data['sparse_vector'].keys():
                inverse_index[token_id].append(chunk_id)
    
    # Convert to regular dict (msgpack can't serialize defaultdict)
    return {k: v for k, v in inverse_index.items()}


def main():
    # Command-line arguments
    parser = argparse.ArgumentParser(description='Extract graph sparse vectors to msgpack')
    parser.add_argument('--batch', type=int, default=32, 
                       help='Number of parallel workers (default: 32)')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of chunks to process (for testing)')
    parser.add_argument('--output', type=str, default='arxiv_graph_sparse.msgpack',
                       help='Output msgpack file path')
    args = parser.parse_args()
    
    print("="*70)
    print("ARXIV GRAPH EXTRACTION TO MSGPACK")
    print("="*70)
    print(f"\n[CONFIG] Workers: {args.batch}")
    print(f"[CONFIG] Output: {args.output}")
    print(f"[CONFIG] Limit: {args.limit or 'ALL'}")
    
    # Load tokenizer
    print("\n[1/5] Loading BERT tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    print(f"  ✓ Loaded (vocab_size={tokenizer.vocab_size})")
    
    # Connect to database (read-only)
    print("\n[2/5] Connecting to database...")
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    print("  ✓ Connected")
    
    # Get total count
    print("\n[3/5] Counting chunks...")
    cur.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}")
    total_count = cur.fetchone()[0]
    
    if args.limit:
        chunk_limit = min(args.limit, total_count)
    else:
        chunk_limit = total_count
    
    print(f"  ✓ Will process {chunk_limit:,} chunks (of {total_count:,} total)")
    
    # Performance estimates
    print("\n[4/5] Performance estimates:")
    warmup_time = args.batch * 3.1
    processing_time = (chunk_limit * 0.095) / args.batch
    total_time = warmup_time + processing_time
    print(f"  Warmup: {warmup_time:.1f}s ({args.batch} workers × 3.1s)")
    print(f"  Processing: {processing_time/60:.1f} minutes")
    print(f"  Total: {total_time/60:.1f} minutes")
    print(f"  Throughput: {chunk_limit/total_time:.0f} chunks/sec")
    
    # Fetch chunk IDs
    print("\n[5/5] Extracting graph sparse vectors...")
    cur.execute(f"SELECT id FROM {TABLE_NAME} ORDER BY id LIMIT %s", (chunk_limit,))
    all_ids = [row[0] for row in cur.fetchall()]
    
    # Process in batches
    forward_index = {}
    
    # Use single progress bar for all chunks
    with tqdm(total=len(all_ids), desc="  Extracting", unit="chunks") as pbar:
        for i in range(0, len(all_ids), BATCH_SIZE):
            batch_ids = all_ids[i:i+BATCH_SIZE]
            
            # Fetch content from DB
            cur.execute(f"""
                SELECT id, content 
                FROM {TABLE_NAME}
                WHERE id = ANY(%s)
            """, (batch_ids,))
            batch_data = cur.fetchall()
            
            # Parallel processing
            with ThreadPoolExecutor(max_workers=args.batch) as executor:
                futures = {
                    executor.submit(process_chunk, chunk_id, content, tokenizer): chunk_id 
                    for chunk_id, content in batch_data
                }
                
                for future in as_completed(futures):
                    try:
                        chunk_id, result = future.result()
                        if result is not None:
                            forward_index[str(chunk_id)] = result
                        pbar.update(1)  # Update per chunk
                    except Exception as e:
                        chunk_id = futures[future]
                        print(f"\nError in future for chunk {chunk_id}: {e}")
                        pbar.update(1)
    
    cur.close()
    conn.close()
    
    # Build inverse index
    print("\n[6/6] Building inverse index...")
    inverse_index = build_inverse_index(forward_index)
    print(f"  ✓ Inverse index: {len(inverse_index):,} unique tokens")
    
    # Create output data structure
    output_data = {
        'metadata': {
            'vocab_size': tokenizer.vocab_size,
            'total_chunks': len(forward_index),
            'created': datetime.now().isoformat(),
            'tokenizer': 'bert-base-uncased',
            'workers': args.batch,
            'extraction_time_minutes': total_time / 60
        },
        'forward_index': forward_index,
        'inverse_index': inverse_index
    }
    
    # Save to msgpack
    print(f"\n[7/7] Saving to {args.output}...")
    with open(args.output, 'wb') as f:
        msgpack.pack(output_data, f)
    
    # File size
    file_size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"  ✓ Saved ({file_size_mb:.1f} MB)")
    
    # Summary
    print("\n" + "="*70)
    print("EXTRACTION COMPLETE")
    print("="*70)
    print(f"Forward index: {len(forward_index):,} chunks")
    print(f"Inverse index: {len(inverse_index):,} tokens")
    print(f"File: {args.output} ({file_size_mb:.1f} MB)")
    
    # Stats
    non_empty = sum(1 for d in forward_index.values() if d['triplet_count'] > 0)
    print(f"\nChunks with triplets: {non_empty:,} ({100*non_empty/len(forward_index):.1f}%)")


if __name__ == "__main__":
    main()
