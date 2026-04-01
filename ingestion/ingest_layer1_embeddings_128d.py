"""
Ingest Layer 1 Embeddings (128d) to pgvector

CRITICAL: This script:
1. Loads Qwen3 256d embeddings from msgpack
2. Fits PCA: 256d → 128d and SAVES the PCA model to disk
3. Creates layer1_embeddings_128d table in PostgreSQL
4. Populates table with PCA-reduced 128d embeddings
5. Creates HNSW index for fast cosine similarity search

The saved PCA model (checkpoints/pca_256to128.pkl) is REQUIRED for query-time
embedding reduction to ensure consistency between ingestion and retrieval.

Usage:
    python ingest_layer1_embeddings_128d.py

Database:
    - Table: layer1_embeddings_128d
    - Schema: (chunk_id TEXT PRIMARY KEY, embedding vector(128))
    - Index: HNSW on embedding for cosine similarity

Output:
    - PostgreSQL table: layer1_embeddings_128d (populated)
    - PCA model: checkpoints/pca_256to128.pkl (saved)
"""

import msgpack
import numpy as np
import psycopg2
from psycopg2.extras import execute_values
from pathlib import Path
from tqdm import tqdm
import pickle
from sklearn.decomposition import PCA


def main():
    # ========================================================================
    # CONFIGURATION
    # ========================================================================
    
    chunks_path = Path("checkpoints/chunks.msgpack")
    embeddings_path = Path("checkpoints/chunk_embeddings_qwen3.msgpack")
    pca_model_path = Path("checkpoints/pca_256to128.pkl")
    
    db_config = {
        'host': 'localhost',
        'port': 5432,
        'dbname': 'langchain',
        'user': 'langchain',
        'password': 'langchain'
    }
    
    table_name = 'layer1_embeddings_128d'
    batch_size = 1000
    
    # ========================================================================
    # LOAD DATA
    # ========================================================================
    
    print("\n" + "="*70)
    print("LAYER 1 EMBEDDINGS (128d) INGESTION")
    print("="*70)
    
    # Load chunks for chunk_id mapping
    print(f"\nLoading chunks from {chunks_path}...")
    with open(chunks_path, 'rb') as f:
        chunks_data = msgpack.unpackb(f.read(), strict_map_key=False)
    
    # Handle both dict and list formats
    if isinstance(chunks_data, dict) and 'chunks' in chunks_data:
        chunks = chunks_data['chunks']
    elif isinstance(chunks_data, list):
        chunks = chunks_data
    else:
        raise ValueError(f"Unexpected chunks format: {type(chunks_data)}")
    
    print(f"  ✓ Loaded {len(chunks):,} chunks")
    
    # Load Qwen3 256d embeddings
    print(f"\nLoading Qwen3 256d embeddings from {embeddings_path}...")
    with open(embeddings_path, 'rb') as f:
        embed_data = msgpack.unpackb(f.read(), strict_map_key=False)
    
    embeddings_256d = np.array(embed_data['embeddings'], dtype=np.float32)
    expected_shape = embed_data.get('shape', embeddings_256d.shape)
    
    if embeddings_256d.ndim == 1:
        embeddings_256d = embeddings_256d.reshape(expected_shape)
    
    print(f"  ✓ Loaded {embeddings_256d.shape} ({embeddings_256d.nbytes / 1024 / 1024:.1f} MB)")
    print(f"  ✓ Model: {embed_data['metadata']['model']}")
    
    # Validate dimensions match
    assert len(chunks) == embeddings_256d.shape[0], \
        f"Mismatch: {len(chunks)} chunks vs {embeddings_256d.shape[0]} embeddings"
    assert embeddings_256d.shape[1] == 256, \
        f"Expected 256d embeddings, got {embeddings_256d.shape[1]}d"
    
    # ========================================================================
    # FIT PCA: 256d → 128d
    # ========================================================================
    
    print(f"\nFitting PCA: 256d → 128d...")
    pca = PCA(n_components=128, random_state=42)
    embeddings_128d = pca.fit_transform(embeddings_256d)
    
    # Report variance retained
    explained_variance = pca.explained_variance_ratio_.sum() * 100
    print(f"  ✓ PCA fitted: 128 components")
    print(f"  ✓ Variance retained: {explained_variance:.2f}%")
    print(f"  ✓ Output shape: {embeddings_128d.shape}")
    
    # ========================================================================
    # SAVE PCA MODEL (CRITICAL!)
    # ========================================================================
    
    print(f"\nSaving PCA model to {pca_model_path}...")
    pca_model_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(pca_model_path, 'wb') as f:
        pickle.dump(pca, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Verify saved model
    with open(pca_model_path, 'rb') as f:
        loaded_pca = pickle.load(f)
    
    # Test roundtrip
    test_embedding = embeddings_256d[0:1]
    test_reduced_original = embeddings_128d[0]
    test_reduced_loaded = loaded_pca.transform(test_embedding)[0]
    
    diff = np.abs(test_reduced_original - test_reduced_loaded).max()
    print(f"  ✓ PCA model saved ({pca_model_path.stat().st_size / 1024:.1f} KB)")
    print(f"  ✓ Roundtrip test: max diff = {diff:.2e} (should be ~0)")
    
    if diff > 1e-5:
        raise ValueError(f"PCA roundtrip failed: max diff {diff} > 1e-5")
    
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
    
    # Drop existing table (clean slate)
    cur.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE")
    
    # Create new table with vector(128)
    cur.execute(f"""
    CREATE TABLE {table_name} (
        chunk_id TEXT PRIMARY KEY,
        embedding vector(128) NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    conn.commit()
    print(f"  ✓ Table created: {table_name}")
    
    # ========================================================================
    # INSERT EMBEDDINGS
    # ========================================================================
    
    print(f"\nInserting {len(chunks):,} embeddings (batch_size={batch_size})...")
    
    total_batches = (len(chunks) + batch_size - 1) // batch_size
    
    for i in tqdm(range(0, len(chunks), batch_size), total=total_batches, desc="  Inserting batches"):
        batch_chunks = chunks[i:i+batch_size]
        batch_embeddings = embeddings_128d[i:i+batch_size]
        
        # Prepare data: [(chunk_id, embedding_list), ...]
        insert_data = []
        for chunk, embedding in zip(batch_chunks, batch_embeddings):
            chunk_id = chunk.get('doc_id') or chunk.get('chunk_id')
            if chunk_id is None:
                raise ValueError(f"Chunk missing doc_id/chunk_id: {chunk}")
            
            embedding_list = embedding.tolist()
            insert_data.append((chunk_id, embedding_list))
        
        # Batch insert with execute_values
        execute_values(
            cur,
            f"INSERT INTO {table_name} (chunk_id, embedding) VALUES %s",
            insert_data,
            template="(%s, %s::vector(128))"
        )
        
        conn.commit()
    
    print(f"  ✓ Inserted {len(chunks):,} rows")
    
    # ========================================================================
    # VERIFY INSERTION
    # ========================================================================
    
    print(f"\nVerifying insertion...")
    cur.execute(f"SELECT COUNT(*) FROM {table_name}")
    count = cur.fetchone()[0]
    
    cur.execute(f"SELECT pg_column_size(embedding) FROM {table_name} LIMIT 1")
    emb_size = cur.fetchone()[0]
    
    print(f"  ✓ Row count: {count:,}")
    print(f"  ✓ Embedding storage size: {emb_size} bytes (expected: ~512 for vector(128))")
    
    # Sample check
    cur.execute(f"SELECT chunk_id, embedding::text FROM {table_name} LIMIT 1")
    sample_id, sample_emb_str = cur.fetchone()
    
    # Parse embedding from PostgreSQL text format: '[0.1,0.2,...]'
    sample_emb_list = eval(sample_emb_str)  # Safe here since it's from our own database
    sample_emb_array = np.array(sample_emb_list, dtype=np.float32)
    
    print(f"  ✓ Sample chunk_id: {sample_id}")
    print(f"  ✓ Sample embedding dim: {len(sample_emb_array)}")
    print(f"  ✓ Sample embedding norm: {np.linalg.norm(sample_emb_array):.4f}")
    
    # ========================================================================
    # CREATE HNSW INDEX
    # ========================================================================
    
    print(f"\nCreating HNSW index on embedding (cosine distance)...")
    print("  (This may take a few minutes for 161k vectors)")
    
    # HNSW parameters:
    # - m=16: number of bi-directional links (higher = better recall, more memory)
    # - ef_construction=64: size of dynamic candidate list (higher = better quality, slower build)
    cur.execute(f"""
    CREATE INDEX idx_{table_name}_hnsw ON {table_name}
    USING hnsw (embedding vector_cosine_ops)
    WITH (m=16, ef_construction=64)
    """)
    
    conn.commit()
    print(f"  ✓ HNSW index created: idx_{table_name}_hnsw")
    
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
    print("LAYER 1 EMBEDDINGS (128d) INGESTION COMPLETE")
    print("="*70)
    print(f"✓ PCA model saved: {pca_model_path}")
    print(f"✓ Table created: {table_name}")
    print(f"✓ Rows inserted: {count:,}")
    print(f"✓ HNSW index created: idx_{table_name}_hnsw")
    print(f"✓ Variance retained: {explained_variance:.2f}%")
    print("\nNext steps:")
    print("  1. Run ingest_layer1_bm25_sparse.py for Layer 1 BM25 sparse vectors")
    print("  2. Run ingest_layer2_triplet_bm25.py for Layer 2 BM25 triplet vectors")
    print("  3. Run update_postgres_256d.py for Layer 2 dense embeddings (256d)")
    print("="*70)


if __name__ == '__main__':
    main()
