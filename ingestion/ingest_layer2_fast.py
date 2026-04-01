"""
Fast Layer 2 Triplet BM25 Ingestion (Using Pre-computed Lookup)

Uses pre-computed triplet BM25 lookup for fast aggregation.
Runtime: ~30 seconds (vs infinite hang with on-the-fly BM25)

Requires: checkpoints/triplet_bm25_lookup.msgpack (from precompute_triplet_bm25_lookup.py)
"""

import psycopg2
import msgpack
import json
from pathlib import Path
from tqdm import tqdm
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
TRIPLET_LOOKUP = CHECKPOINT_DIR / 'triplet_bm25_lookup.msgpack'

def load_chunks():
    """Load chunks metadata."""
    print(f"\nLoading chunks from {CHUNKS_FILE}...")
    with open(CHUNKS_FILE, 'rb') as f:
        chunks = msgpack.unpackb(f.read(), strict_map_key=False)
    print(f"  ✓ Loaded {len(chunks):,} chunks")
    return chunks


def load_chunk_to_triplets():
    """Load chunk→triplets mapping."""
    print(f"\nLoading chunk→triplets mapping from {CHUNK_TO_TRIPLETS}...")
    with open(CHUNK_TO_TRIPLETS, 'rb') as f:
        mapping = msgpack.unpackb(f.read(), strict_map_key=False)
    print(f"  ✓ Loaded {len(mapping):,} mappings")
    return mapping


def load_triplet_lookup():
    """Load pre-computed triplet BM25 lookup."""
    print(f"\nLoading pre-computed triplet lookup from {TRIPLET_LOOKUP}...")
    with open(TRIPLET_LOOKUP, 'rb') as f:
        lookup = msgpack.unpackb(f.read(), strict_map_key=False)
    
    # Convert string keys back to int
    lookup = {int(k): v for k, v in lookup.items()}
    
    non_empty = sum(1 for v in lookup.values() if v)
    print(f"  ✓ Loaded {len(lookup):,} triplet vectors")
    print(f"  ✓ Non-empty: {non_empty:,} ({100*non_empty/len(lookup):.1f}%)")
    
    return lookup


def aggregate_triplet_bm25_fast(chunk_id, triplet_indices, triplet_lookup):
    """
    Fast aggregation using pre-computed lookup (dictionary lookups only).
    
    Args:
        chunk_id: Chunk identifier
        triplet_indices: List of triplet IDs linked to this chunk
        triplet_lookup: Pre-computed dict mapping triplet_id → sparse_vector_dict
    
    Returns:
        dict: Aggregated sparse vector {term_id: score}
    """
    aggregated = defaultdict(float)
    
    for triplet_idx in triplet_indices:
        # Fast dictionary lookup (no BM25 computation!)
        triplet_sparse = triplet_lookup.get(triplet_idx, {})
        
        # Aggregate scores
        for term_id, score in triplet_sparse.items():
            aggregated[term_id] += score
    
    # Convert back to regular dict
    return dict(aggregated)


def bulk_insert_triplet_bm25(chunks, chunk_to_triplets, triplet_lookup, conn):
    """Bulk insert aggregated triplet BM25 vectors."""
    print(f"\nAggregating and inserting {len(chunks):,} chunk triplet vectors...")
    
    cursor = conn.cursor()
    batch_size = 1000
    batch = []
    
    inserted_count = 0
    skipped_count = 0
    
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
            skipped_count += 1
            continue
        
        # Fast aggregation using pre-computed lookup
        aggregated_sparse = aggregate_triplet_bm25_fast(chunk_id, triplet_indices, triplet_lookup)
        
        if not aggregated_sparse:
            skipped_count += 1
            continue
        
        # Prepare for insert
        sparse_json = json.dumps(aggregated_sparse)
        batch.append((chunk_id, sparse_json))
        
        # Bulk insert when batch is full
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
            inserted_count += len(batch)
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
        inserted_count += len(batch)
    
    cursor.close()
    
    print(f"\n  ✓ Inserted: {inserted_count:,} rows")
    print(f"  ✓ Skipped (no triplets): {skipped_count:,}")
    
    return inserted_count


def verify_ingestion(conn):
    """Verify Layer 2 ingestion."""
    cursor = conn.cursor()
    
    # Row count
    cursor.execute("SELECT COUNT(*) FROM layer2_triplet_bm25")
    row_count = cursor.fetchone()[0]
    
    # Non-null sparse vectors
    cursor.execute("SELECT COUNT(*) FROM layer2_triplet_bm25 WHERE sparse_vector IS NOT NULL")
    non_null_count = cursor.fetchone()[0]
    
    # Sample data
    cursor.execute("""
        SELECT chunk_id, 
               jsonb_object_keys(sparse_vector)::int AS term_count
        FROM layer2_triplet_bm25
        LIMIT 5
    """)
    samples = cursor.fetchall()
    
    # Table size
    cursor.execute("""
        SELECT pg_size_pretty(pg_total_relation_size('layer2_triplet_bm25'))
    """)
    table_size = cursor.fetchone()[0]
    
    cursor.close()
    
    print("\n" + "="*60)
    print("VERIFICATION")
    print("="*60)
    print(f"Rows: {row_count:,}")
    print(f"Non-null sparse_vector: {non_null_count:,} ({100*non_null_count/row_count:.1f}%)")
    print(f"Table size: {table_size}")
    print(f"\nSample data:")
    for chunk_id, term_count in samples:
        print(f"  {chunk_id}: {term_count} terms")


def main():
    print("="*60)
    print("LAYER 2: TRIPLET BM25 INGESTION (FAST)")
    print("="*60)
    
    # Load data
    chunks = load_chunks()
    chunk_to_triplets = load_chunk_to_triplets()
    triplet_lookup = load_triplet_lookup()
    
    # Connect to database
    print("\nConnecting to PostgreSQL...")
    conn = psycopg2.connect(**DB_CONFIG)
    print("  ✓ Connected")
    
    try:
        # Bulk insert
        inserted_count = bulk_insert_triplet_bm25(chunks, chunk_to_triplets, triplet_lookup, conn)
        
        # Verify
        verify_ingestion(conn)
        
        print("\n" + "="*60)
        print("✓ LAYER 2 INGESTION COMPLETE")
        print("="*60)
        
    finally:
        conn.close()
        print("\n✓ Database connection closed")


if __name__ == '__main__':
    main()
