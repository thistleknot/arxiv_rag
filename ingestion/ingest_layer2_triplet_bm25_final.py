"""
Layer 2 Triplet BM25 Ingestion (ACTUALLY CORRECT)

Key insight: Use existing chunk_bm25_sparse vocabulary, build BM25 on aggregated
triplet texts, extract sparse vectors directly (NO per-chunk scoring).

Runtime: ~30 seconds total (not per chunk!)
"""

import psycopg2
import msgpack
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from scipy.sparse import csr_matrix

# Configuration
DB_CONFIG = {
    'dbname': 'langchain',
    'user': 'langchain',
    'password': 'langchain',
    'host': 'localhost',
    'port': 5432
}

CHECKPOINT_DIR = Path('checkpoints')
CHUNKS_FILE = CHECKPOINT_DIR / 'chunks.msgpack'
CHUNK_TO_TRIPLETS = CHECKPOINT_DIR / 'chunk_to_triplets.msgpack'
TRIPLET_INDEX = CHECKPOINT_DIR / 'triplet_bm25_index.msgpack'
CHUNK_BM25_SPARSE = CHECKPOINT_DIR / 'chunk_bm25_sparse.msgpack'


def load_chunk_bm25_vocab():
    """Load existing vocabulary from chunk_bm25_sparse.msgpack."""
    print("\nLoading existing chunk BM25 vocabulary...")
    with open(CHUNK_BM25_SPARSE, 'rb') as f:
        data = msgpack.unpackb(f.read(), strict_map_key=False)
    
    vocab = data['vocab']  # dict: token -> id
    print(f"  ✓ Vocabulary: {len(vocab):,} tokens")
    
    # Create reverse mapping (id -> token)
    id_to_token = {v: k for k, v in vocab.items()}
    
    return vocab, id_to_token


def load_data():
    """Load chunks, mappings, and triplet texts."""
    print("\nLoading data...")
    
    # Chunks
    with open(CHUNKS_FILE, 'rb') as f:
        chunks = msgpack.unpackb(f.read(), strict_map_key=False)
    print(f"  ✓ Loaded {len(chunks):,} chunks")
    
    # Chunk→triplets mapping
    with open(CHUNK_TO_TRIPLETS, 'rb') as f:
        chunk_to_triplets = msgpack.unpackb(f.read(), strict_map_key=False)
    print(f"  ✓ Loaded {len(chunk_to_triplets):,} chunk→triplet mappings")
    
    # Triplet texts
    with open(TRIPLET_INDEX, 'rb') as f:
        triplet_data = msgpack.unpackb(f.read(), strict_map_key=False)
    triplet_texts = triplet_data['triplet_texts']
    print(f"  ✓ Loaded {len(triplet_texts):,} triplet texts")
    
    return chunks, chunk_to_triplets, triplet_texts


def aggregate_chunk_triplets(chunks, chunk_to_triplets, triplet_texts):
    """
    Aggregate triplets to chunk level.
    
    Returns:
        list of (chunk_id, aggregated_text)
    """
    print(f"\nAggregating triplets to chunk level...")
    
    chunk_aggregated = []
    skipped = 0
    
    for chunk in tqdm(chunks, desc="  Processing"):
        # Extract chunk_id
        chunk_id = chunk.get('doc_id') or chunk.get('chunk_id')
        section_idx = chunk.get('section_idx', 0)
        chunk_idx = chunk.get('chunk_idx', 0)
        
        # Construct mapping key (with duplicated suffix)
        mapping_key = f"{chunk_id}_s{section_idx}_c{chunk_idx}"
        
        # Get triplet indices
        triplet_indices = chunk_to_triplets.get(mapping_key, [])
        
        if not triplet_indices:
            skipped += 1
            continue
        
        # Aggregate triplet texts for this chunk
        chunk_triplet_texts = [
            triplet_texts[idx] 
            for idx in triplet_indices 
            if idx < len(triplet_texts)
        ]
        
        if not chunk_triplet_texts:
            skipped += 1
            continue
        
        # Concatenate into single text blob
        aggregated_text = ' '.join(chunk_triplet_texts)
        
        chunk_aggregated.append((chunk_id, aggregated_text))
    
    print(f"  ✓ Aggregated: {len(chunk_aggregated):,} chunks")
    print(f"  ✓ Skipped (no triplets): {skipped:,}")
    
    return chunk_aggregated


def compute_sparse_vectors_fast(chunk_aggregated, vocab):
    """
    Compute BM25 sparse vectors efficiently using existing vocabulary.
    
    This builds BM25 ONCE on all aggregated texts, then extracts vectors directly.
    NOT scoring each chunk against corpus (that's O(n²)).
    """
    print(f"\nComputing BM25 sparse vectors (FAST)...")
    
    # Tokenize using existing vocabulary
    print("  Tokenizing with existing vocabulary...")
    tokenized_corpus = []
    for _, text in tqdm(chunk_aggregated, desc="    Tokenizing"):
        tokens = [t.lower() for t in text.split() if t.lower() in vocab]
        tokenized_corpus.append(tokens)
    
    # Build BM25 index on aggregated texts
    print("  Building BM25 index...")
    bm25 = BM25Okapi(tokenized_corpus)
    
    # Extract sparse vectors directly from BM25 internal structures
    print("  Extracting sparse vectors...")
    sparse_vectors = []
    
    for i, (chunk_id, _) in enumerate(tqdm(chunk_aggregated, desc="    Extracting")):
        # Get document from BM25 internal structure
        doc = bm25.doc_freqs[i]
        
        # Convert to sparse dict using vocab IDs
        sparse_dict = {}
        for token, freq in doc.items():
            if token in vocab:
                token_id = vocab[token]
                # Use BM25 score for this token in this doc
                score = bm25.idf.get(token, 0) * freq
                sparse_dict[str(token_id)] = float(score)
        
        sparse_vectors.append((chunk_id, sparse_dict))
    
    print(f"  ✓ Computed {len(sparse_vectors):,} sparse vectors")
    
    return sparse_vectors


def bulk_insert_sparse_vectors(sparse_vectors, conn):
    """Bulk insert sparse vectors into layer2_triplet_bm25."""
    print(f"\nInserting {len(sparse_vectors):,} sparse vectors...")
    
    cursor = conn.cursor()
    batch_size = 1000
    batch = []
    
    for chunk_id, sparse_dict in tqdm(sparse_vectors, desc="  Inserting"):
        sparse_json = json.dumps(sparse_dict)
        batch.append((chunk_id, sparse_json))
        
        if len(batch) >= batch_size:
            cursor.executemany(
                """
                INSERT INTO layer2_triplet_bm25 (chunk_id, triplet_bm25_vector)
                VALUES (%s, %s::jsonb)
                ON CONFLICT (chunk_id) DO UPDATE
                SET triplet_bm25_vector = EXCLUDED.triplet_bm25_vector
                """,
                batch
            )
            conn.commit()
            batch = []
    
    # Insert remaining
    if batch:
        cursor.executemany(
            """
            INSERT INTO layer2_triplet_bm25 (chunk_id, triplet_bm25_vector)
            VALUES (%s, %s::jsonb)
            ON CONFLICT (chunk_id) DO UPDATE
            SET triplet_bm25_vector = EXCLUDED.triplet_bm25_vector
            """,
            batch
        )
        conn.commit()
    
    cursor.close()
    print(f"  ✓ Inserted {len(sparse_vectors):,} rows")


def verify_ingestion(conn):
    """Verify ingestion."""
    cursor = conn.cursor()
    
    # Row count
    cursor.execute("SELECT COUNT(*) FROM layer2_triplet_bm25")
    row_count = cursor.fetchone()[0]
    
    # Non-null
    cursor.execute("SELECT COUNT(*) FROM layer2_triplet_bm25 WHERE triplet_bm25_vector IS NOT NULL")
    non_null = cursor.fetchone()[0]
    
    # Table size
    cursor.execute("SELECT pg_size_pretty(pg_total_relation_size('layer2_triplet_bm25'))")
    size = cursor.fetchone()[0]
    
    cursor.close()
    
    print("\n" + "="*60)
    print("VERIFICATION")
    print("="*60)
    print(f"Rows: {row_count:,}")
    print(f"Non-null triplet_bm25_vector: {non_null:,} ({100*non_null/row_count:.1f}%)")
    print(f"Table size: {size}")


def main():
    print("="*60)
    print("LAYER 2: TRIPLET BM25 (ACTUALLY CORRECT)")
    print("="*60)
    
    # Load existing vocabulary
    vocab, id_to_token = load_chunk_bm25_vocab()
    
    # Load data
    chunks, chunk_to_triplets, triplet_texts = load_data()
    
    # Aggregate triplets to chunk level
    chunk_aggregated = aggregate_chunk_triplets(chunks, chunk_to_triplets, triplet_texts)
    
    # Compute BM25 sparse vectors efficiently
    sparse_vectors = compute_sparse_vectors_fast(chunk_aggregated, vocab)
    
    # Connect to database
    print("\nConnecting to PostgreSQL...")
    conn = psycopg2.connect(**DB_CONFIG)
    print("  ✓ Connected")
    
    try:
        # Bulk insert
        bulk_insert_sparse_vectors(sparse_vectors, conn)
        
        # Verify
        verify_ingestion(conn)
        
        print("\n" + "="*60)
        print("✓ LAYER 2 COMPLETE")
        print("="*60)
        
    finally:
        conn.close()


if __name__ == '__main__':
    main()
