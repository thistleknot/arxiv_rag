"""
Ingest Layer 1 BM25 Sparse Vectors to pgvector (Simplified for CSR format)

This script:
1. Loads chunk_bm25_sparse.msgpack (pre-computed CSR sparse matrix)
2. Extracts sparse vectors for each chunk
3. Converts sparse vectors to JSONB format: {term_id: weight}
4. Creates layer1_bm25_sparse table in PostgreSQL
5. Populates table with JSONB sparse vectors
6. Creates GIN index for fast sparse vector lookup

Usage:
    python ingest_layer1_bm25_sparse_csr.py

Database:
    - Table: layer1_bm25_sparse
    - Schema: (chunk_id TEXT PRIMARY KEY, sparse_vector JSONB)
    - Index: GIN on sparse_vector for key lookup
"""

import msgpack
import numpy as np
import psycopg2
from psycopg2.extras import Json, execute_values
from pathlib import Path
from tqdm import tqdm
from scipy.sparse import csr_matrix


def extract_sparse_row(csr, row_idx):
    """
    Extract a single row from CSR matrix as {term_id: weight} dict.
    
    Args:
        csr: scipy.sparse.csr_matrix
        row_idx: int, row index
        
    Returns:
        dict {term_id: weight} with non-zero entries only
    """
    start = csr.indptr[row_idx]
    end = csr.indptr[row_idx + 1]
    
    sparse_dict = {}
    for i in range(start, end):
        term_id = int(csr.indices[i])
        weight = float(csr.data[i])
        if weight > 0:  # Only store positive weights
            sparse_dict[term_id] = weight
    
    return sparse_dict


def main():
    # ========================================================================
    # CONFIGURATION
    # ========================================================================
    
    chunks_path = Path("checkpoints/chunks.msgpack")
    bm25_csr_path = Path("checkpoints/chunk_bm25_sparse.msgpack")
    
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
    print("LAYER 1 BM25 SPARSE VECTORS INGESTION (CSR FORMAT)")
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
    
    # Load BM25 CSR matrix
    print(f"\nLoading BM25 CSR sparse matrix from {bm25_csr_path}...")
    with open(bm25_csr_path, 'rb') as f:
        bm25_data = msgpack.unpackb(f.read(), strict_map_key=False)
    
    # Extract CSR components
    data = np.array(bm25_data['data'], dtype=np.float32)
    indices = np.array(bm25_data['indices'], dtype=np.int32)
    indptr = np.array(bm25_data['indptr'], dtype=np.int32)
    shape = tuple(bm25_data['shape'])
    vocab = bm25_data['vocab']
    
    print(f"  ✓ Matrix shape: {shape[0]:,} chunks × {shape[1]:,} terms")
    print(f"  ✓ Vocab size: {len(vocab):,}")
    print(f"  ✓ Non-zero entries: {len(data):,}")
    print(f"  ✓ Sparsity: {100 * (1 - len(data) / (shape[0] * shape[1])):.2f}%")
    
    # Build CSR matrix
    print(f"\nBuilding CSR matrix...")
    csr = csr_matrix((data, indices, indptr), shape=shape)
    print(f"  ✓ CSR matrix created")
    
    # Verify chunk count matches
    if len(chunks) != shape[0]:
        raise ValueError(f"Chunk count mismatch: {len(chunks)} chunks vs {shape[0]} matrix rows")
    
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
    
    create_table_sql = f"""
    CREATE TABLE {table_name} (
        chunk_id TEXT PRIMARY KEY,
        sparse_vector JSONB NOT NULL
    )
    """
    
    cur.execute(create_table_sql)
    conn.commit()
    print(f"  ✓ Table created: {table_name}")
    
    # ========================================================================
    # INSERT SPARSE VECTORS
    # ========================================================================
    
    print(f"\nInserting {len(chunks):,} sparse vectors (batch_size={batch_size})...")
    
    rows = []
    for i, chunk in enumerate(tqdm(chunks, desc="  Extracting sparse vectors")):
        chunk_id = chunk['doc_id']  # Use 'doc_id' key from chunks
        
        # Extract sparse vector for this chunk
        sparse_dict = extract_sparse_row(csr, i)
        
        # Convert to strings for JSONB (keys must be strings, not ints)
        sparse_dict_str = {str(k): v for k, v in sparse_dict.items()}
        
        rows.append((chunk_id, Json(sparse_dict_str)))
        
        # Insert batch
        if len(rows) >= batch_size:
            execute_values(
                cur,
                f"INSERT INTO {table_name} (chunk_id, sparse_vector) VALUES %s",
                rows,
                template="(%s, %s)"
            )
            conn.commit()
            rows = []
    
    # Insert remaining rows
    if rows:
        execute_values(
            cur,
            f"INSERT INTO {table_name} (chunk_id, sparse_vector) VALUES %s",
            rows,
            template="(%s, %s)"
        )
        conn.commit()
    
    print(f"  ✓ Inserted {len(chunks):,} rows")
    
    # ========================================================================
    # VERIFY INSERTION
    # ========================================================================
    
    print(f"\nVerifying insertion...")
    
    cur.execute(f"SELECT COUNT(*) FROM {table_name}")
    count = cur.fetchone()[0]
    print(f"  ✓ Row count: {count:,}")
    
    cur.execute(f"SELECT chunk_id, sparse_vector FROM {table_name} LIMIT 1")
    sample_id, sample_vec = cur.fetchone()
    print(f"  ✓ Sample chunk_id: {sample_id}")
    print(f"  ✓ Sample sparse vector: {len(sample_vec)} non-zero entries")
    print(f"  ✓ Sample (first 3): {dict(list(sample_vec.items())[:3])}")
    
    # ========================================================================
    # CREATE GIN INDEX
    # ========================================================================
    
    print(f"\nCreating GIN index on sparse_vector...")
    print(f"  (This may take a few minutes for {count:,} vectors)")
    
    cur.execute(f"CREATE INDEX idx_{table_name}_gin ON {table_name} USING GIN (sparse_vector)")
    conn.commit()
    print(f"  ✓ GIN index created: idx_{table_name}_gin")
    
    # ========================================================================
    # FINAL STATISTICS
    # ========================================================================
    
    print(f"\nFinal statistics:")
    
    cur.execute(f"SELECT pg_size_pretty(pg_total_relation_size('{table_name}'))")
    total_size = cur.fetchone()[0]
    print(f"  Total size: {total_size}")
    
    cur.execute(f"SELECT pg_size_pretty(pg_relation_size('{table_name}'))")
    table_size = cur.fetchone()[0]
    print(f"  Table size: {table_size}")
    
    cur.execute(f"SELECT pg_size_pretty(pg_indexes_size('{table_name}'))")
    index_size = cur.fetchone()[0]
    print(f"  Index size: {index_size}")
    
    # ========================================================================
    # CLEANUP
    # ========================================================================
    
    cur.close()
    conn.close()
    
    print("\n" + "="*70)
    print("LAYER 1 BM25 SPARSE VECTORS INGESTION COMPLETE")
    print("="*70)
    print(f"✓ Table created: {table_name}")
    print(f"✓ Rows inserted: {count:,}")
    print(f"✓ GIN index created: idx_{table_name}_gin")
    print(f"✓ Total size: {total_size}")
    print("\nNext steps:")
    print("  1. Run ingest_layer2_triplet_bm25.py for Layer 2 BM25 triplet vectors")
    print("  2. Run update_postgres_256d.py for Layer 2 dense embeddings")
    print("="*70)
    print()


if __name__ == '__main__':
    import time
    start_time = time.time()
    
    try:
        main()
        elapsed = time.time() - start_time
        print(f"✓ ingest_layer1_bm25_sparse_csr.py completed successfully ({elapsed:.1f}s)")
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n✗ Error after {elapsed:.1f}s:")
        print(f"  {type(e).__name__}: {e}")
        raise
