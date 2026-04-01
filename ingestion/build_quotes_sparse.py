"""
Update Quotes Table with BM25 and Graph Sparse Vectors

Populates the bm25_sparse and graph_sparse columns in the quotes table.
"""

import psycopg2
import numpy as np
from typing import List, Dict
from collections import Counter, defaultdict
from tqdm import tqdm
from pathlib import Path
import msgpack

# Import from pgvector_retriever
import sys
sys.path.append('.')
from pgvector_retriever import (
    TextPreprocessor,
    extract_triplets,
    create_graph_sparse_vector,
    sparse_dict_to_pgvector_format,
    get_bert_tokenizer
)

PG_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'postgres',
    'user': 'postgres',
    'password': 'postgres'
}

BM25_K1 = 1.5
BM25_B = 0.75


def build_bm25_vectors(conn):
    """Build BM25 sparse vectors for all quotes."""
    print("\n[1/2] Building BM25 sparse vectors...")
    
    # Fetch all quotes
    with conn.cursor() as cur:
        cur.execute("SELECT chunk_id, content FROM quotes ORDER BY chunk_id")
        rows = cur.fetchall()
    
    chunk_ids = [r[0] for r in rows]
    contents = [r[1] for r in rows]
    
    print(f"  Loaded {len(contents)} quotes")
    
    # Build vocabulary
    print("  Building vocabulary...")
    vocab = TextPreprocessor.build_vocab_from_corpus(contents, min_df=2)
    preprocessor = TextPreprocessor(vocab)
    
    # Compute IDF
    print("  Computing IDF...")
    N = len(contents)
    df = defaultdict(int)
    for text in tqdm(contents, desc="  Computing DF", leave=False):
        token_ids = preprocessor.preprocess(text)
        for tid in set(token_ids):
            if tid >= 0:
                df[tid] += 1
    
    idf = {tid: np.log((N - doc_freq + 0.5) / (doc_freq + 0.5) + 1)
           for tid, doc_freq in df.items()}
    
    # Compute document lengths
    doc_lengths = [len(preprocessor.preprocess(text)) for text in contents]
    avgdl = np.mean(doc_lengths)
    
    # Build sparse vectors
    print("  Building sparse vectors...")
    updates = []
    for i, text in enumerate(tqdm(contents, desc="  Vectorizing", leave=False)):
        token_ids = preprocessor.preprocess(text)
        tf = Counter(token_ids)
        
        indices, values = [], []
        for tid, freq in tf.items():
            if tid < 0 or tid not in idf:
                continue
            
            # BM25 weight
            numerator = freq * (BM25_K1 + 1)
            denominator = freq + BM25_K1 * (
                1 - BM25_B + BM25_B * doc_lengths[i] / avgdl
            )
            weight = idf[tid] * (numerator / denominator)
            
            if weight > 0:
                indices.append(tid)
                values.append(weight)
        
        if indices:
            pairs = [f"{idx}:{val:.4f}" for idx, val in zip(indices, values)]
            sparse_str = '{' + ','.join(pairs) + '}/' + str(len(vocab))
        else:
            sparse_str = '{}/' + str(len(vocab))
        
        updates.append((sparse_str, chunk_ids[i]))
    
    # Update database
    print("  Updating database...")
    with conn.cursor() as cur:
        for sparse_str, chunk_id in tqdm(updates, desc="  Updating"):
            cur.execute(
                "UPDATE quotes SET bm25_sparse = %s WHERE chunk_id = %s",
                (sparse_str, chunk_id)
            )
    conn.commit()
    
    # Save vocab for later queries
    print("  Saving vocabulary cache...")
    vocab_path = Path("data/quotes_bm25_vocab.msgpack")
    with open(vocab_path, 'wb') as f:
        msgpack.dump({
            'vocab': vocab,
            'vocab_size': len(vocab),
            'idf': idf,
            'avgdl': avgdl,
            'N': N
        }, f)
    
    print(f"  ✓ BM25 vectors built (vocab size: {len(vocab)})")


def build_graph_sparse_vectors(conn):
    """Build graph sparse vectors for all quotes."""
    print("\n[2/2] Building graph sparse vectors...")
    
    # Fetch all quotes
    with conn.cursor() as cur:
        cur.execute("SELECT chunk_id, content FROM quotes ORDER BY chunk_id")
        rows = cur.fetchall()
    
    chunk_ids = [r[0] for r in rows]
    contents = [r[1] for r in rows]
    
    print(f"  Loaded {len(contents)} quotes")
    
    # Load tokenizer
    tokenizer = get_bert_tokenizer()
    
    # Build sparse vectors
    print("  Building graph sparse vectors...")
    updates = []
    for content in tqdm(contents, desc="  Processing"):
        # Extract triplets
        triplets = extract_triplets(content)
        
        if not triplets:
            # Fallback: create pseudo-triplets from words
            words = [w.strip(',.!?;:()[]{}"\'-').lower() 
                    for w in content.split() 
                    if len(w.strip(',.!?;:()[]{}"\'-')) > 2]
            if words:
                triplets = [(words[i], 'relates', words[min(i+1, len(words)-1)]) 
                           for i in range(len(words))]
        
        # Create sparse vector
        if triplets:
            sparse_dict = create_graph_sparse_vector(triplets, tokenizer)
            sparse_str = sparse_dict_to_pgvector_format(sparse_dict, vocab_size=30522)
        else:
            sparse_str = '{}/30522'
        
        updates.append((sparse_str, chunk_ids[len(updates)]))
    
    # Update database
    print("  Updating database...")
    with conn.cursor() as cur:
        for sparse_str, chunk_id in tqdm(updates, desc="  Updating"):
            cur.execute(
                "UPDATE quotes SET graph_sparse = %s WHERE chunk_id = %s",
                (sparse_str, chunk_id)
            )
    conn.commit()
    
    print("  ✓ Graph sparse vectors built")


def main():
    print("Connecting to PostgreSQL...")
    conn = psycopg2.connect(**PG_CONFIG)
    print("✓ Connected")
    
    try:
        build_bm25_vectors(conn)
        build_graph_sparse_vectors(conn)
        
        print("\n✓ Quotes table update complete!")
        print("\nNext step:")
        print("  Create query_quotes.py and test GraphRAG retrieval")
        
    finally:
        conn.close()


if __name__ == '__main__':
    main()
