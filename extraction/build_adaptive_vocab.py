"""
Adaptive Vocabulary Building for Fast Graph Extraction

Strategy: Iteratively grow vocabulary by extracting more chunks until 
similarity metric plateaus (diminishing returns).

Process:
1. Extract N chunks with OpenIE → build vocab → test similarity
2. Extract +N more chunks → rebuild vocab → test similarity
3. Repeat until similarity improvement < threshold
4. Save final vocabulary for fast extraction on all 161k chunks
"""

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import msgpack
import numpy as np
import psycopg2
from collections import defaultdict
from transformers import AutoTokenizer
from tqdm import tqdm
import time
import argparse

# Import from existing scripts
import sys
sys.path.insert(0, '.')
from extract_graph_to_msgpack import extract_and_process_triplets, build_knowledge_graph, kg_to_sparse_vector, get_thread_extractor
from fast_graph_extraction import fast_extract_sparse, cosine_similarity

DB_CONFIG = {
    'dbname': 'postgres',
    'user': 'postgres',
    'password': 'postgres',
    'host': 'localhost',
    'port': 5432
}
TABLE_NAME = 'arxiv_papers_lemma_fullembed'

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=100, help='Chunks to extract per iteration')
parser.add_argument('--test-size', type=int, default=50, help='Held-out test set size')
parser.add_argument('--threshold', type=float, default=0.01, help='Min improvement to continue')
parser.add_argument('--max-iters', type=int, default=10, help='Max iterations')
args = parser.parse_args()


def extract_batch_with_openie(chunk_ids, tokenizer):
    """Extract triplets from batch of chunks using OpenIE"""
    print(f"  Extracting {len(chunk_ids)} chunks with OpenIE...")
    
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    cur.execute(f"""
        SELECT id, content 
        FROM {TABLE_NAME}
        WHERE id = ANY(%s)
    """, (chunk_ids,))
    
    chunks = cur.fetchall()
    cur.close()
    conn.close()
    
    results = {}
    for chunk_id, content in tqdm(chunks, desc="    Processing"):
        try:
            # Get thread-local extractor
            extractor, nlp = get_thread_extractor()
            
            # Extract triplets
            triplets = extract_and_process_triplets(content, chunk_id)
            
            if not triplets:
                results[chunk_id] = {'sparse_vector': {}, 'triplet_count': 0}
                continue
            
            # Build KG
            kg = build_knowledge_graph(triplets, nlp)
            
            # Convert to sparse vector
            sparse_dict = kg_to_sparse_vector(kg, tokenizer)
            
            results[chunk_id] = {
                'sparse_vector': sparse_dict,
                'triplet_count': len(triplets)
            }
        except Exception as e:
            print(f"\n    Error on chunk {chunk_id}: {e}")
            results[chunk_id] = {'sparse_vector': {}, 'triplet_count': 0}
    
    return results


def build_vocabulary_from_extractions(extractions):
    """Build vocabulary from extracted sparse vectors"""
    token_freq = defaultdict(int)
    
    for data in extractions.values():
        sparse = data.get('sparse_vector', {})
        for token_id in sparse.keys():
            tid = int(token_id) if isinstance(token_id, str) else token_id
            token_freq[tid] += 1
    
    # Filter to tokens appearing in multiple chunks
    important = {tid for tid, freq in token_freq.items() if freq >= 2}
    
    return important, token_freq


def evaluate_vocab_on_test(extractions, important_tokens, tokenizer):
    """Evaluate vocabulary by comparing fast vs slow extraction on test set"""
    similarities = []
    
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    for chunk_id, slow_data in extractions.items():
        # Get content
        cur.execute(f"SELECT content FROM {TABLE_NAME} WHERE id = %s", (chunk_id,))
        row = cur.fetchone()
        if not row:
            continue
        
        content = row[0]
        slow_sparse = slow_data.get('sparse_vector', {})
        
        # Fast extraction with current vocab
        fast_sparse = fast_extract_sparse(content, tokenizer, important_tokens)
        
        # Compare
        sim = cosine_similarity(fast_sparse, slow_sparse)
        similarities.append(sim)
    
    cur.close()
    conn.close()
    
    return np.mean(similarities) if similarities else 0.0


def main():
    print("="*70)
    print("ADAPTIVE VOCABULARY BUILDING")
    print("="*70)
    print(f"Batch size: {args.batch_size} chunks/iteration")
    print(f"Test size: {args.test_size} chunks")
    print(f"Improvement threshold: {args.threshold}")
    print(f"Max iterations: {args.max_iters}")
    
    # Load tokenizer
    print("\n[1/3] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    print(f"  ✓ Vocab size: {tokenizer.vocab_size}")
    
    # Get chunk IDs from database
    print("\n[2/3] Fetching chunk IDs...")
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute(f"SELECT id FROM {TABLE_NAME} ORDER BY id")
    all_ids = [row[0] for row in cur.fetchall()]
    cur.close()
    conn.close()
    print(f"  ✓ Total chunks in DB: {len(all_ids):,}")
    
    # Split into train/test
    np.random.seed(42)
    shuffled = np.random.permutation(all_ids)
    test_ids = shuffled[:args.test_size].tolist()
    train_pool = shuffled[args.test_size:].tolist()
    
    print(f"  ✓ Test set: {len(test_ids)} chunks")
    print(f"  ✓ Train pool: {len(train_pool):,} chunks")
    
    # Extract test set once (for evaluation)
    print("\n[3/3] Extracting test set with OpenIE (one-time)...")
    test_extractions = extract_batch_with_openie(test_ids, tokenizer)
    
    # Training loop
    print("\n" + "="*70)
    print("TRAINING LOOP")
    print("="*70)
    
    train_extractions = {}
    best_similarity = 0.0
    history = []
    
    for iteration in range(args.max_iters):
        print(f"\n--- Iteration {iteration + 1}/{args.max_iters} ---")
        
        # Get next batch
        start_idx = iteration * args.batch_size
        end_idx = start_idx + args.batch_size
        
        if start_idx >= len(train_pool):
            print("  Exhausted training pool!")
            break
        
        batch_ids = train_pool[start_idx:end_idx]
        print(f"  Train set size: {len(train_extractions)} → {len(train_extractions) + len(batch_ids)}")
        
        # Extract new batch
        new_extractions = extract_batch_with_openie(batch_ids, tokenizer)
        train_extractions.update(new_extractions)
        
        # Build vocabulary
        print(f"\n  Building vocabulary from {len(train_extractions)} chunks...")
        important_tokens, token_freq = build_vocabulary_from_extractions(train_extractions)
        print(f"    Important tokens: {len(important_tokens)}")
        
        # Evaluate on test set
        print(f"  Evaluating on {len(test_extractions)} test chunks...")
        similarity = evaluate_vocab_on_test(test_extractions, important_tokens, tokenizer)
        print(f"    Average similarity: {similarity:.4f}")
        
        # Check improvement
        improvement = similarity - best_similarity
        print(f"    Improvement: {improvement:+.4f}")
        
        history.append({
            'iteration': iteration + 1,
            'train_size': len(train_extractions),
            'vocab_size': len(important_tokens),
            'similarity': similarity,
            'improvement': improvement
        })
        
        if improvement < args.threshold and iteration > 0:
            print(f"\n  ✓ Converged! Improvement {improvement:.4f} < {args.threshold}")
            break
        
        best_similarity = max(best_similarity, similarity)
    
    # Summary
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    print("\nIter | Train Size | Vocab Size | Similarity | Improvement")
    print("-" * 65)
    for h in history:
        print(f"{h['iteration']:4} | {h['train_size']:10} | {h['vocab_size']:10} | {h['similarity']:10.4f} | {h['improvement']:+11.4f}")
    
    print(f"\nFinal vocabulary size: {len(important_tokens)}")
    print(f"Final similarity: {best_similarity:.4f}")
    
    # Save vocabulary
    vocab_path = 'graph_vocab_adaptive.msgpack'
    print(f"\nSaving vocabulary to {vocab_path}...")
    vocab_data = {
        'important_tokens': list(important_tokens),
        'token_freq': {int(k): int(v) for k, v in token_freq.items()},
        'train_size': len(train_extractions),
        'test_similarity': best_similarity,
        'history': history
    }
    
    with open(vocab_path, 'wb') as f:
        msgpack.pack(vocab_data, f)
    
    print(f"  ✓ Saved")
    
    print("\nNext steps:")
    print(f"  1. Run fast extraction on all 161k chunks using {vocab_path}")
    print(f"  2. Expected time: ~3 minutes")
    print(f"  3. Expected quality: {best_similarity:.1%} of OpenIE baseline")


if __name__ == "__main__":
    main()
