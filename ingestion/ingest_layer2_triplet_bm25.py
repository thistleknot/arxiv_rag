"""
Ingest Layer 2 Triplet BM25 Sparse Vectors to pgvector

This script:
1. Loads triplet_bm25_index.msgpack (Layer 2 triplet-based BM25 index)
2. Loads chunk↔triplet mappings
3. Rebuilds BM25Okapi from saved triplet tokens
4. For each chunk: aggregates BM25 scores across associated triplets
5. Converts aggregated sparse vectors to JSONB format: {triplet_term_id: weight}
6. Creates layer2_triplet_bm25 table in PostgreSQL
7. Populates table with JSONB sparse vectors
8. Creates GIN index for fast sparse vector lookup

Layer 2 uses triplet-level BM25 scoring, which captures graph structure and
semantic relationships extracted from the text.

Usage:
    python ingest_layer2_triplet_bm25.py

Database:
    - Table: layer2_triplet_bm25
    - Schema: (chunk_id TEXT PRIMARY KEY, triplet_bm25_vector JSONB)
    - Index: GIN on triplet_bm25_vector for key lookup

Dependencies:
    - triplet_bm25_index.msgpack (from build_triplet_bm25.py)
    - chunk_to_triplets.msgpack
    - enriched_triplets.msgpack
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
from collections import defaultdict, Counter


def aggregate_triplet_bm25(chunk_id, chunk_to_triplets_map, triplet_bm25, triplets_list):
    """
    Aggregate BM25 scores for all triplets associated with a chunk.
    
    Args:
        chunk_id: The chunk identifier
        chunk_to_triplets_map: dict mapping chunk IDs to list of triplet indices
        triplet_bm25: BM25Okapi instance built from triplet tokens
        triplets_list: list of triplet dicts with 'tokens' field
    
    Returns:
        dict {triplet_term_id: aggregated_weight}
    """
    # Get triplet indices for this chunk
    triplet_indices = chunk_to_triplets_map.get(chunk_id, [])
    
    if not triplet_indices:
        return {}
    
    # Aggregate BM25 scores across triplets
    # Use max pooling: for each term, take max score across all triplets
    aggregated_scores = defaultdict(float)
    
    for triplet_idx in triplet_indices:
        if triplet_idx >= len(triplets_list):
            continue
        
        triplet = triplets_list[triplet_idx]
        
        # Get tokens for this triplet
        triplet_tokens = triplet.get('tokens', [])
        if not triplet_tokens:
            # Fallback: tokenize text if tokens not stored
            triplet_text = triplet.get('text', '')
            if triplet_text:
                triplet_tokens = triplet_text.lower().split()
        
        if not triplet_tokens:
            continue
        
        # Get BM25 scores for this triplet (query against entire corpus)
        # Note: BM25 scores are typically computed per-query, but we want
        # per-triplet weights, so we use the triplet as a query
        scores = triplet_bm25.get_scores(triplet_tokens)
        
        # For each unique token in this triplet, accumulate score
        token_counts = Counter(triplet_tokens)
        
        for token, count in token_counts.items():
            # Use triplet_idx as term_id (maps to triplet in vocabulary)
            # This creates a triplet-level sparse vector
            # Weight by token frequency × BM25 score
            weight = count * scores[triplet_idx] if triplet_idx < len(scores) else count
            
            # Max pooling: keep highest score for this token across triplets
            aggregated_scores[triplet_idx] = max(aggregated_scores[triplet_idx], weight)
    
    return dict(aggregated_scores)


def main():
    # ========================================================================
    # CONFIGURATION
    # ========================================================================
    
    chunks_path = Path("checkpoints/chunks.msgpack")
    triplet_bm25_path = Path("checkpoints/triplet_bm25_index.msgpack")
    chunk_to_triplets_path = Path("checkpoints/chunk_to_triplets.msgpack")
    enriched_triplets_path = Path("checkpoints/enriched_triplets.msgpack")
    
    db_config = {
        'host': 'localhost',
        'port': 5432,
        'dbname': 'langchain',
        'user': 'langchain',
        'password': 'langchain'
    }
    
    table_name = 'layer2_triplet_bm25'
    batch_size = 1000
    
    # ========================================================================
    # LOAD DATA
    # ========================================================================
    
    print("\n" + "="*70)
    print("LAYER 2 TRIPLET BM25 SPARSE VECTORS INGESTION")
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
    
    # Load triplet BM25 index
    print(f"\nLoading triplet BM25 index from {triplet_bm25_path}...")
    with open(triplet_bm25_path, 'rb') as f:
        bm25_data = msgpack.unpackb(f.read(), strict_map_key=False)
    
    triplet_tokens = bm25_data.get('triplet_tokens') or bm25_data.get('doc_tokens', [])
    n_triplets = len(triplet_tokens)
    
    print(f"  ✓ Loaded {n_triplets:,} triplet token sequences")
    
    # Rebuild BM25Okapi from triplet_tokens
    print(f"\nRebuilding BM25Okapi from triplet tokens...")
    triplet_bm25 = BM25Okapi(triplet_tokens)
    print(f"  ✓ BM25 built: {len(triplet_bm25.doc_len)} documents")
    
    # Load chunk↔triplet mappings
    print(f"\nLoading chunk↔triplet mappings from {chunk_to_triplets_path}...")
    with open(chunk_to_triplets_path, 'rb') as f:
        chunk_to_triplets_map = msgpack.unpackb(f.read(), strict_map_key=False)
    
    print(f"  ✓ Loaded {len(chunk_to_triplets_map):,} chunk→triplet mappings")
    
    # Load triplet tokens from triplet_bm25_index.msgpack (which already has tokens)
    # Note: enriched_triplets.msgpack doesn't exist, but triplet_bm25_index has 'triplet_tokens'
    print(f"\nConverting triplet tokens to triplet dicts...")
    loaded_data = msgpack.unpackb(open(Path("checkpoints/triplet_bm25_index.msgpack"), 'rb').read(), strict_map_key=False)
    triplet_tokens_list = loaded_data['triplet_tokens']
    
    # Convert raw token lists to dict format expected by aggregate function
    triplets_list = [{'tokens': tokens} for tokens in triplet_tokens_list]
    print(f"  ✓ Converted {len(triplets_list):,} triplet token sequences to dict format")
    
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
        triplet_bm25_vector JSONB NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    conn.commit()
    print(f"  ✓ Table created: {table_name}")
    
    # ========================================================================
    # COMPUTE AND INSERT TRIPLET SPARSE VECTORS
    # ========================================================================
    
    print(f"\nComputing and inserting triplet BM25 sparse vectors (batch_size={batch_size})...")
    
    total_batches = (len(chunks) + batch_size - 1) // batch_size
    total_nonzero = 0
    chunks_with_triplets = 0
    
    for i in tqdm(range(0, len(chunks), batch_size), total=total_batches, desc="  Processing batches"):
        batch_chunks = chunks[i:i+batch_size]
        
        insert_data = []
        for chunk in batch_chunks:
            chunk_id = chunk.get('doc_id') or chunk.get('chunk_id')
            
            if chunk_id is None:
                raise ValueError(f"Chunk missing doc_id/chunk_id: {chunk}")
            
            # Build mapping key: chunk_to_triplets uses duplicated suffix format
            # e.g., doc_id='1301_3781_s0_c0' but mapping key='1301_3781_s0_c0_s0_c0'
            section_idx = chunk.get('section_idx', 0)
            chunk_idx = chunk.get('chunk_idx', 0)
            mapping_key = f"{chunk_id}_s{section_idx}_c{chunk_idx}"
            
            # Aggregate triplet BM25 scores for this chunk
            sparse_dict = aggregate_triplet_bm25(
                mapping_key,
                chunk_to_triplets_map,
                triplet_bm25,
                triplets_list
            )
            
            if sparse_dict:
                chunks_with_triplets += 1
                total_nonzero += len(sparse_dict)
            
            # Convert to JSONB (convert int keys to strings for JSON)
            sparse_json = {str(k): v for k, v in sparse_dict.items()}
            
            insert_data.append((chunk_id, Json(sparse_json)))
        
        # Batch insert
        execute_values(
            cur,
            f"INSERT INTO {table_name} (chunk_id, triplet_bm25_vector) VALUES %s",
            insert_data,
            template="(%s, %s)"
        )
        
        conn.commit()
    
    avg_nonzero = total_nonzero / chunks_with_triplets if chunks_with_triplets > 0 else 0
    coverage = chunks_with_triplets / len(chunks) * 100
    
    print(f"  ✓ Inserted {len(chunks):,} rows")
    print(f"  ✓ Chunks with triplets: {chunks_with_triplets:,} ({coverage:.1f}%)")
    print(f"  ✓ Average nonzero triplets per chunk: {avg_nonzero:.1f}")
    
    # ========================================================================
    # VERIFY INSERTION
    # ========================================================================
    
    print(f"\nVerifying insertion...")
    cur.execute(f"SELECT COUNT(*) FROM {table_name}")
    count = cur.fetchone()[0]
    
    print(f"  ✓ Row count: {count:,}")
    
    # Sample check (find a chunk with non-empty vector)
    cur.execute(f"""
    SELECT chunk_id, triplet_bm25_vector 
    FROM {table_name} 
    WHERE jsonb_typeof(triplet_bm25_vector) = 'object' 
      AND triplet_bm25_vector != '{{}}'::jsonb
    LIMIT 1
    """)
    result = cur.fetchone()
    
    if result:
        sample_id, sample_vec = result
        print(f"  ✓ Sample chunk_id: {sample_id}")
        print(f"  ✓ Sample nonzero triplets: {len(sample_vec)}")
        print(f"  ✓ Sample triplet IDs: {list(sample_vec.keys())[:5]}...")
    else:
        print(f"  ⚠ No chunks with non-empty triplet vectors found")
    
    # ========================================================================
    # CREATE GIN INDEX
    # ========================================================================
    
    print(f"\nCreating GIN index on triplet_bm25_vector (JSONB)...")
    print("  (This enables fast key lookups for sparse vector dot products)")
    
    cur.execute(f"""
    CREATE INDEX idx_{table_name}_gin ON {table_name}
    USING gin (triplet_bm25_vector jsonb_path_ops)
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
    print("LAYER 2 TRIPLET BM25 SPARSE VECTORS INGESTION COMPLETE")
    print("="*70)
    print(f"✓ Table created: {table_name}")
    print(f"✓ Rows inserted: {count:,}")
    print(f"✓ GIN index created: idx_{table_name}_gin")
    print(f"✓ Chunks with triplets: {chunks_with_triplets:,} ({coverage:.1f}%)")
    print(f"✓ Average nonzero triplets: {avg_nonzero:.1f}")
    print(f"✓ Total triplets in corpus: {n_triplets:,}")
    print("\nNext steps:")
    print("  1. Run update_postgres_256d.py for Layer 2 dense embeddings (256d)")
    print("  2. Verify all 4 collections are populated")
    print("  3. Update three_layer_gist_retriever.py for pgvector support")
    print("="*70)


if __name__ == '__main__':
    main()
