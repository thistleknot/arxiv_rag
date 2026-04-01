"""
Fast Graph Extraction via Token Vocabulary

Strategy: Use 100-chunk sample to identify "important" graph tokens,
then fast-match on all chunks using just BERT tokenization.

No OpenIE, no ML training - just vocabulary lookup.
Expected: ~5ms per chunk vs 5000ms with OpenIE (1000x speedup)
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

DB_CONFIG = {
    'dbname': 'postgres',
    'user': 'postgres', 
    'password': 'postgres',
    'host': 'localhost',
    'port': 5432
}
TABLE_NAME = 'arxiv_papers_lemma_fullembed'


def load_sample_extraction(path='arxiv_graph_sparse.msgpack'):
    """Load extracted graph data from msgpack"""
    print(f"Loading {path}...")
    with open(path, 'rb') as f:
        data = msgpack.unpack(f, raw=False, strict_map_key=False)
    
    forward_index = data.get('forward_index', {})
    print(f"  Loaded {len(forward_index)} chunks")
    return forward_index


def build_vocabulary(forward_index):
    """Build vocabulary from sample: token_id → frequency"""
    print("\nBuilding vocabulary from sample...")
    
    token_freq = defaultdict(int)
    
    for chunk_data in forward_index.values():
        sparse = chunk_data.get('sparse_vector', {})
        for token_id in sparse.keys():
            tid = int(token_id) if isinstance(token_id, str) else token_id
            token_freq[tid] += 1
    
    # Filter to tokens appearing in multiple chunks (reduce noise)
    important = {tid for tid, freq in token_freq.items() if freq >= 2}
    
    print(f"  Total unique tokens: {len(token_freq)}")
    print(f"  Important tokens (freq≥2): {len(important)}")
    
    return important, token_freq


def fast_extract_sparse(text, tokenizer, important_tokens):
    """
    Fast extraction: BERT tokenize + filter to vocabulary.
    
    NO OpenIE, NO tree walking - just tokenization.
    """
    # Tokenize
    tokens = tokenizer.encode(text, add_special_tokens=False, max_length=512, truncation=True)
    
    # Filter to important tokens and count
    sparse = defaultdict(float)
    for tid in tokens:
        if tid in important_tokens:
            sparse[tid] += 1.0
    
    # Normalize
    if sparse:
        max_val = max(sparse.values())
        for tid in sparse:
            sparse[tid] /= max_val
    
    return dict(sparse)


def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two sparse vectors"""
    if not vec1 or not vec2:
        return 0.0
    
    # Get intersection
    common = set(vec1.keys()) & set(vec2.keys())
    if not common:
        return 0.0
    
    # Dot product
    dot = sum(vec1[k] * vec2[k] for k in common)
    
    # Magnitudes
    mag1 = sum(v**2 for v in vec1.values()) ** 0.5
    mag2 = sum(v**2 for v in vec2.values()) ** 0.5
    
    return dot / (mag1 * mag2) if mag1 > 0 and mag2 > 0 else 0.0


def test_fast_vs_slow(forward_index, tokenizer, important_tokens):
    """Compare fast extraction vs OpenIE extraction"""
    print("\n" + "="*70)
    print("TESTING: Fast Extraction vs OpenIE Baseline")
    print("="*70)
    
    # Connect to DB
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    # Get sample chunk IDs - just use first 20 from DB regardless of extraction sample
    cur.execute(f"SELECT id, content FROM {TABLE_NAME} LIMIT 20")
    chunks = cur.fetchall()
    cur.close()
    conn.close()
    
    # Get a few extracted chunks to compare against
    extracted_ids = list(forward_index.keys())[:5]
    print(f"\nExtracted sample IDs: {extracted_ids[:5]}")
    print(f"\nTesting fast extraction on {len(chunks)} random chunks...")
    print("\nChunk ID | Fast(ms) | Tokens | Sample")
    print("-" * 60)
    
    times = []
    
    for chunk_id, content in chunks:
        # Fast extraction
        start = time.time()
        fast_sparse = fast_extract_sparse(content, tokenizer, important_tokens)
        elapsed_ms = (time.time() - start) * 1000
        times.append(elapsed_ms)
        
        # Check if this chunk was in our extraction sample
        in_sample = str(chunk_id) in forward_index
        sample_mark = "✓" if in_sample else " "
        
        print(f"{chunk_id:8} | {elapsed_ms:8.2f} | {len(fast_sparse):6} | {sample_mark}")
    
    # Now compare the ones that were actually extracted
    print(f"\n\nComparing similarity on {len(extracted_ids)} extracted chunks...")
    print("\nChunk ID | FastTokens | SlowTokens | Similarity")
    print("-" * 60)
    
    similarities = []
    for chunk_id_str in extracted_ids:
        chunk_id = int(chunk_id_str)
        
        cur = conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute(f"SELECT content FROM {TABLE_NAME} WHERE id = %s", (chunk_id,))
        row = cur.fetchone()
        cur.close()
        conn.close()
        
        if not row:
            continue
            
        content = row[0]
        slow_sparse = forward_index[chunk_id_str].get('sparse_vector', {})
        
        # Fast extraction
        fast_sparse = fast_extract_sparse(content, tokenizer, important_tokens)
        
        # Compare
        sim = cosine_similarity(fast_sparse, slow_sparse)
        similarities.append(sim)
        
        print(f"{chunk_id:8} | {len(fast_sparse):10} | {len(slow_sparse):10} | {sim:10.3f}")
    
    # Summary
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Average fast extraction time: {np.mean(times):.2f}ms")
    print(f"Average similarity to OpenIE: {np.mean(similarities):.3f}")
    print(f"\nEstimated time for 161k chunks:")
    print(f"  Fast method: {161389 * np.mean(times) / 1000 / 60:.1f} minutes")
    print(f"  OpenIE method: ~39 hours")
    print(f"  Speedup: {39 * 60 / (161389 * np.mean(times) / 1000 / 60):.0f}x")


def main():
    print("="*70)
    print("FAST GRAPH EXTRACTION - VOCABULARY MATCHING")
    print("="*70)
    
    # Load tokenizer
    print("\n[1/4] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    print(f"  ✓ Vocab size: {tokenizer.vocab_size}")
    
    # Load sample extraction
    print("\n[2/4] Loading sample...")
    forward_index = load_sample_extraction()
    
    if len(forward_index) < 10:
        print("  ✗ Need more sample data!")
        print("  Run: python extract_graph_to_msgpack.py --batch 8 --limit 100")
        return
    
    # Build vocabulary
    print("\n[3/4] Building vocabulary...")
    important_tokens, token_freq = build_vocabulary(forward_index)
    
    # Test fast vs slow
    print("\n[4/4] Testing...")
    test_fast_vs_slow(forward_index, tokenizer, important_tokens)


if __name__ == "__main__":
    main()
