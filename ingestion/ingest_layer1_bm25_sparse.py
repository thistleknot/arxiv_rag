"""
Ingest Layer 1 BM25 Sparse Vectors to pgvector

This script:
1. Loads wordpiece_bm25_index.msgpack (Layer 1 lemmatized BM25 index)
2. Rebuilds BM25Okapi from saved doc_tokens
3. Computes BM25 sparse scores for each chunk
4. Converts sparse vectors to JSONB format: {term_id: weight}
5. Creates layer1_bm25_sparse table in PostgreSQL
6. Populates table with JSONB sparse vectors
7. Creates GIN index for fast sparse vector lookup

Note: pgvector 0.8.1 doesn't have sparsevec extension yet, so we use JSONB.

Usage:
    python ingest_layer1_bm25_sparse.py

Database:
    - Table: layer1_bm25_sparse
    - Schema: (chunk_id TEXT PRIMARY KEY, bm25_vector JSONB)
    - Index: GIN on bm25_vector for key lookup

Dependencies:
    - wordpiece_bm25_index.msgpack (from build_wordpiece_bm25.py)
    - chunks.msgpack
"""

import msgpack
import numpy as np
import psycopg2
from psycopg2.extras import Json, execute_values
from pathlib import Path
from tqdm import tqdm
import json
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer
from collections import defaultdict


def tokenize_triplet_text(tokenizer, text):
    """WordPiece tokenize text, removing special tokens."""
    tokens = tokenizer.tokenize(text)
    tokens = [t for t in tokens if t not in ['[CLS]', '[SEP]', '[PAD]', '[UNK]']]
    return tokens


def compute_chunk_bm25_sparse(chunk_text, bm25, tokenizer, chunk_to_triplets_map=None, triplets_list=None):
    """
    Compute BM25 sparse vector for a chunk.
    
    For Layer 1: use chunk text directly (lemmatized content).
    If chunk_to_triplets_map provided: use concatenated triplet texts.
    
    Returns: dict {term_id: weight} where term_id is WordPiece token ID
    """
    # Get text to tokenize
    if chunk_to_triplets_map and triplets_list:
        # Use triplets if provided (Layer 1 uses triplet-based lemmatization)
        triplet_indices = chunk_to_triplets_map.get(chunk_text, [])
        if triplet_indices:
            triplet_texts = [triplets_list[i]['text'] for i in triplet_indices if i < len(triplets_list)]
            text = " ".join(triplet_texts)
        else:
            text = chunk_text
    else:
        text = chunk_text
    
    # Tokenize with WordPiece
    tokens = tokenize_triplet_text(tokenizer, text)
    
    if not tokens:
        return {}
    
    # Get BM25 scores
    scores = bm25.get_scores(tokens)
    
    # Build sparse dict: {term_id: weight}
    # term_id is the position in BM25's vocabulary (doc index)
    sparse_dict = {}
    
    # BM25 scores are per-document, we need per-token weights
    # Use token frequency × IDF approach
    token_counts = defaultdict(int)
    for token in tokens:
        token_counts[token] += 1
    
    # Convert token strings to term IDs (use tokenizer vocab)
    vocab = tokenizer.get_vocab()
    
    for token, count in token_counts.items():
        if token in vocab:
            term_id = vocab[token]
            # Simple TF-IDF style weight (could use BM25 formula if needed)
            weight = float(count)  # Simplified; BM25 scoring happens at query time
            sparse_dict[term_id] = weight
    
    return sparse_dict


def main():
    # ========================================================================
    # CONFIGURATION
    # ========================================================================
    
    chunks_path = Path("checkpoints/chunks.msgpack")
    wordpiece_bm25_path = Path("checkpoints/chunk_bm25_sparse.msgpack")
    enriched_triplets_path = Path("checkpoints/enriched_triplets.msgpack")
    chunk_to_triplets_path = Path("checkpoints/chunk_to_triplets.msgpack")
    
    db_config = {
        'host': 'localhost',
        'port': 5432,
        'dbname': 'langchain',
        'user': 'langchain',
        'password': 'langchain'
    }
    
    table_name = 'layer1_bm25_sparse'
    batch_size = 1000
    
    # ========================================================================
    # LOAD DATA
    # ========================================================================
    
    print("\n" + "="*70)
    print("LAYER 1 BM25 SPARSE VECTORS INGESTION")
    print("="*70)
    
    # Load chunks
    print(f"\nLoading chunks from {chunks_path}...")
    with open(chunks_path, 'rb') as f:
        chunks_data = msgpack.unpackb(f.read(), strict_map_key=False)
    
    if isinstance(chunks_data, dict) and 'chunks' in chunks_data:
        chunks = chunks_data['chunks']
    elif isinstance(chunks_data, list):
        chunks = chunks_data
    else:
        raise ValueError(f"Unexpected chunks format: {type(chunks_data)}")
    
    print(f"  ✓ Loaded {len(chunks):,} chunks")
    
    # Load WordPiece BM25 index
    print(f"\nLoading WordPiece BM25 index from {wordpiece_bm25_path}...")
    with open(wordpiece_bm25_path, 'rb') as f:
        bm25_data = msgpack.unpackb(f.read(), strict_map_key=False)
    
    doc_tokens = bm25_data['doc_tokens']
    tokenizer_name = bm25_data.get('tokenizer_name', 'bert-base-uncased')
    vocab_size = bm25_data.get('vocab_size', len(doc_tokens))
    
    print(f"  ✓ Loaded {len(doc_tokens):,} documents")
    print(f"  ✓ Tokenizer: {tokenizer_name}")
    print(f"  ✓ Vocab size: {vocab_size:,}")
    
    # Rebuild BM25Okapi from doc_tokens
    print(f"\nRebuilding BM25Okapi from doc_tokens...")
    bm25 = BM25Okapi(doc_tokens)
    print(f"  ✓ BM25 built: {len(bm25.doc_len)} documents")
    
    # Load tokenizer
    print(f"\nLoading WordPiece tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    print(f"  ✓ Tokenizer loaded: {len(tokenizer)} tokens")
    
    # Load triplets and mappings (optional for triplet-based lemmatization)
    triplets_list = None
    chunk_to_triplets_map = None
    
    if enriched_triplets_path.exists() and chunk_to_triplets_path.exists():
        print(f"\nLoading triplets for lemmatization...")
        with open(enriched_triplets_path, 'rb') as f:
            triplets_data = msgpack.unpackb(f.read(), strict_map_key=False)
        triplets_list = triplets_data.get('triplets', triplets_data)
        
        with open(chunk_to_triplets_path, 'rb') as f:
            chunk_to_triplets_map = msgpack.unpackb(f.read(), strict_map_key=False)
        
        print(f"  ✓ Loaded {len(triplets_list):,} triplets")
        print(f"  ✓ Loaded {len(chunk_to_triplets_map):,} chunk→triplet mappings")
    else:
        print(f"\n⚠ Triplet files not found, using chunk text directly")
    
    # ========================================================================
    # CONNECT TO POSTGRESQL
    # ========================================================================
    
    print(f"\nConnecting to PostgreSQL {db_config['host']}:{db_config['port']}/{db_config['dbname']}...")
    conn = psycopg2.connect(**db_config)
    conn.autocommit = False
    cur = conn.cursor()
    print("  ✓ Connected")
    
    # ========================================================================
    # CREATE TABLE
    # ========================================================================
    
    print(f"\nCreating table {table_name}...")
    
    cur.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE")
    
    cur.execute(f"""
    CREATE TABLE {table_name} (
        chunk_id TEXT PRIMARY KEY,
        bm25_vector JSONB NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    conn.commit()
    print(f"  ✓ Table created: {table_name}")
    
    # ========================================================================
    # COMPUTE AND INSERT SPARSE VECTORS
    # ========================================================================
    
    print(f"\nComputing and inserting BM25 sparse vectors (batch_size={batch_size})...")
    
    total_batches = (len(chunks) + batch_size - 1) // batch_size
    total_nonzero = 0
    
    for i in tqdm(range(0, len(chunks), batch_size), total=total_batches, desc="  Processing batches"):
        batch_chunks = chunks[i:i+batch_size]
        
        insert_data = []
        for chunk in batch_chunks:
            chunk_id = chunk.get('doc_id') or chunk.get('chunk_id')
            chunk_text = chunk.get('text', '')
            
            if chunk_id is None:
                raise ValueError(f"Chunk missing doc_id/chunk_id: {chunk}")
            
            # Compute sparse vector
            sparse_dict = compute_chunk_bm25_sparse(
                chunk_text, 
                bm25, 
                tokenizer,
                chunk_to_triplets_map,
                triplets_list
            )
            
            total_nonzero += len(sparse_dict)
            
            # Convert to JSONB (convert int keys to strings for JSON)
            sparse_json = {str(k): v for k, v in sparse_dict.items()}
            
            insert_data.append((chunk_id, Json(sparse_json)))
        
        # Batch insert
        execute_values(
            cur,
            f"INSERT INTO {table_name} (chunk_id, bm25_vector) VALUES %s",
            insert_data,
            template="(%s, %s)"
        )
        
        conn.commit()
    
    avg_nonzero = total_nonzero / len(chunks)
    print(f"  ✓ Inserted {len(chunks):,} rows")
    print(f"  ✓ Average nonzero terms per chunk: {avg_nonzero:.1f}")
    
    # ========================================================================
    # VERIFY INSERTION
    # ========================================================================
    
    print(f"\nVerifying insertion...")
    cur.execute(f"SELECT COUNT(*) FROM {table_name}")
    count = cur.fetchone()[0]
    
    print(f"  ✓ Row count: {count:,}")
    
    # Sample check
    cur.execute(f"SELECT chunk_id, bm25_vector FROM {table_name} LIMIT 1")
    sample_id, sample_vec = cur.fetchone()
    
    print(f"  ✓ Sample chunk_id: {sample_id}")
    print(f"  ✓ Sample nonzero terms: {len(sample_vec)}")
    print(f"  ✓ Sample terms: {list(sample_vec.keys())[:5]}...")
    
    # ========================================================================
    # CREATE GIN INDEX
    # ========================================================================
    
    print(f"\nCreating GIN index on bm25_vector (JSONB)...")
    print("  (This enables fast key lookups for sparse vector dot products)")
    
    cur.execute(f"""
    CREATE INDEX idx_{table_name}_gin ON {table_name}
    USING gin (bm25_vector jsonb_path_ops)
    """)
    
    conn.commit()
    print(f"  ✓ GIN index created: idx_{table_name}_gin")
    
    # ========================================================================
    # FINAL STATISTICS
    # ========================================================================
    
    print(f"\nFinal statistics:")
    cur.execute(f"""
    SELECT 
        pg_size_pretty(pg_total_relation_size('{table_name}')) as total_size,
        pg_size_pretty(pg_relation_size('{table_name}')) as table_size,
        pg_size_pretty(pg_indexes_size('{table_name}')) as indexes_size
    """)
    total_size, table_size, indexes_size = cur.fetchone()
    
    print(f"  Table size: {table_size}")
    print(f"  Index size: {indexes_size}")
    print(f"  Total size: {total_size}")
    
    cur.close()
    conn.close()
    
    # ========================================================================
    # SUCCESS SUMMARY
    # ========================================================================
    
    print("\n" + "="*70)
    print("LAYER 1 BM25 SPARSE VECTORS INGESTION COMPLETE")
    print("="*70)
    print(f"✓ Table created: {table_name}")
    print(f"✓ Rows inserted: {count:,}")
    print(f"✓ GIN index created: idx_{table_name}_gin")
    print(f"✓ Average nonzero terms: {avg_nonzero:.1f}")
    print(f"✓ Vocab size: {vocab_size:,}")
    print("\nNext steps:")
    print("  1. Run ingest_layer2_triplet_bm25.py for Layer 2 BM25 triplet vectors")
    print("  2. Run update_postgres_256d.py for Layer 2 dense embeddings (256d)")
    print("="*70)


if __name__ == '__main__':
    main()
