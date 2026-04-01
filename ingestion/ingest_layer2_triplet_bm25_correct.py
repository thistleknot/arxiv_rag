"""
Layer 2 Triplet BM25 Ingestion (CORRECT ARCHITECTURE)

Aggregates triplets to chunk level, computes BM25 on aggregated triplet text.
Same complexity as layer1_bm25_sparse (~30 seconds).

Steps:
1. Load chunks + chunk_to_triplets mapping + triplet texts
2. For each chunk: aggregate its triplets into one text blob
3. Build BM25 index on aggregated texts (same as layer1 did for chunk content)
4. Store sparse vectors in layer2_triplet_bm25

Runtime: ~30 seconds (same as layer1_bm25_sparse)
"""

import psycopg2
import msgpack
import json
from pathlib import Path
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from collections import defaultdict

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
        
        # Construct mapping key (with duplicated suffix to match chunk_to_triplets.msgpack)
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


def compute_bm25_sparse_vectors(chunk_aggregated):
    """
    Compute BM25 sparse vectors for aggregated triplet texts.
    Same as layer1_bm25_sparse did for chunk content.
    """
    print(f"\nComputing BM25 sparse vectors...")
    
    # Tokenize
    print("  Tokenizing...")
    tokenized_corpus = [text.lower().split() for _, text in chunk_aggregated]
    
    # Build BM25 index
    print("  Building BM25 index...")
    bm25 = BM25Okapi(tokenized_corpus)
    
    # Build vocabulary (unique tokens)
    print("  Building vocabulary...")
    vocab = {}
    for tokens in tokenized_corpus:
        for token in tokens:
            if token not in vocab:
                vocab[token] = len(vocab)
    
    print(f"  ✓ Vocabulary size: {len(vocab):,} tokens")
    
    # Compute sparse vectors
    print("  Computing sparse vectors...")
    sparse_vectors = []
    
    for i, (chunk_id, _) in enumerate(tqdm(chunk_aggregated, desc="    Processing")):
        tokens = tokenized_corpus[i]
        
        # Get BM25 scores
        scores = bm25.get_scores(tokens)
        
        # Convert to sparse dict (only non-zero)
        non_zero = {}
        for j, score in enumerate(scores):
            if score > 0:
                non_zero[str(j)] = float(score)
        
        sparse_vectors.append((chunk_id, non_zero))
    
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
                INSERT INTO layer2_triplet_bm25 (chunk_id, sparse_vector)
                VALUES (%s, %s::jsonb)
                ON CONFLICT (chunk_id) DO UPDATE
                SET sparse_vector = EXCLUDED.sparse_vector
                """,
                batch
            )
            conn.commit()
            batch = []
    
    # Insert remaining
    if batch:
        cursor.executemany(
            """
            INSERT INTO layer2_triplet_bm25 (chunk_id, sparse_vector)
            VALUES (%s, %s::jsonb)
            ON CONFLICT (chunk_id) DO UPDATE
            SET sparse_vector = EXCLUDED.sparse_vector
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
    cursor.execute("SELECT COUNT(*) FROM layer2_triplet_bm25 WHERE sparse_vector IS NOT NULL")
    non_null = cursor.fetchone()[0]
    
    # Table size
    cursor.execute("SELECT pg_size_pretty(pg_total_relation_size('layer2_triplet_bm25'))")
    size = cursor.fetchone()[0]
    
    cursor.close()
    
    print("\n" + "="*60)
    print("VERIFICATION")
    print("="*60)
    print(f"Rows: {row_count:,}")
    print(f"Non-null sparse_vector: {non_null:,} ({100*non_null/row_count:.1f}%)")
    print(f"Table size: {size}")


def main():
    print("="*60)
    print("LAYER 2: TRIPLET BM25 INGESTION (CORRECT)")
    print("="*60)
    
    # Load
    chunks, chunk_to_triplets, triplet_texts = load_data()
    
    # Aggregate triplets to chunk level
    chunk_aggregated = aggregate_chunk_triplets(chunks, chunk_to_triplets, triplet_texts)
    
    # Compute BM25 sparse vectors (same as layer1_bm25_sparse)
    sparse_vectors = compute_bm25_sparse_vectors(chunk_aggregated)
    
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
