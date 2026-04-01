"""
Master Orchestration Script for pgvector Migration

This script orchestrates the complete migration from msgpack to pgvector:

1. Creates base arxiv_chunks table (metadata + content, no embeddings)
2. Populates arxiv_chunks from chunks.msgpack
3. Runs ingest_layer1_embeddings_128d.py (PCA + 128d embeddings + HNSW index)
4. Runs ingest_layer1_bm25_sparse.py (lemmatized BM25 + GIN index)
5. Runs ingest_layer2_triplet_bm25.py (triplet BM25 + GIN index)
6. Updates arxiv_chunks with 256d embeddings (reuses update_postgres_256d.py logic)
7. Verifies data integrity across all 4 collections
8. Prints comprehensive summary statistics

After completion, the pgvector database will be ready for production queries.

Usage:
    python ingest_master.py

Database Tables Created:
    - arxiv_chunks (base metadata + content, includes 256d embedding)
    - layer1_embeddings_128d (PCA-reduced Qwen3 128d)
    - layer1_bm25_sparse (lemmatized WordPiece BM25)
    - layer2_triplet_bm25 (triplet-aggregated BM25)

Note: This script imports and runs the other ingestion scripts as modules.
Ensure all dependencies are installed before running.
"""

import msgpack
import numpy as np
import psycopg2
from psycopg2.extras import execute_values
from pathlib import Path
from tqdm import tqdm
import subprocess
import sys
import time
from datetime import datetime


def create_base_arxiv_chunks_table(conn, chunks):
    """
    Create and populate base arxiv_chunks table with metadata and content.
    
    This table stores the core chunk data without embeddings initially.
    The 256d embedding column will be added later by update_postgres_256d.py.
    """
    print("\n" + "="*70)
    print("CREATING BASE ARXIV_CHUNKS TABLE")
    print("="*70)
    
    cur = conn.cursor()
    
    # Drop existing table
    print("\nDropping existing arxiv_chunks table (if exists)...")
    cur.execute("DROP TABLE IF EXISTS arxiv_chunks CASCADE")
    conn.commit()
    print("  ✓ Dropped")
    
    # Create base table (no embeddings yet)
    print("\nCreating base arxiv_chunks table...")
    cur.execute("""
    CREATE TABLE arxiv_chunks (
        chunk_id TEXT PRIMARY KEY,
        paper_id TEXT NOT NULL,
        section_idx INTEGER,
        chunk_idx INTEGER,
        content TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    conn.commit()
    print("  ✓ Table created")
    
    # Insert chunks in batches
    print(f"\nInserting {len(chunks):,} chunks (batch_size=1000)...")
    batch_size = 1000
    total_batches = (len(chunks) + batch_size - 1) // batch_size
    
    for i in tqdm(range(0, len(chunks), batch_size), total=total_batches, desc="  Inserting batches"):
        batch = chunks[i:i+batch_size]
        
        insert_data = []
        for chunk in batch:
            chunk_id = chunk.get('doc_id') or chunk.get('chunk_id')
            paper_id = chunk.get('paper_id', 'unknown')
            section_idx = chunk.get('section_idx')
            chunk_idx = chunk.get('chunk_idx')
            content = chunk.get('text', '')
            
            if chunk_id is None:
                raise ValueError(f"Chunk missing doc_id/chunk_id: {chunk}")
            
            insert_data.append((chunk_id, paper_id, section_idx, chunk_idx, content))
        
        execute_values(
            cur,
            "INSERT INTO arxiv_chunks (chunk_id, paper_id, section_idx, chunk_idx, content) VALUES %s",
            insert_data
        )
        conn.commit()
    
    print(f"  ✓ Inserted {len(chunks):,} rows")
    
    # Create index on paper_id
    print("\nCreating index on paper_id...")
    cur.execute("CREATE INDEX idx_arxiv_chunks_paper_id ON arxiv_chunks(paper_id)")
    conn.commit()
    print("  ✓ Index created")
    
    # Verify
    cur.execute("SELECT COUNT(*) FROM arxiv_chunks")
    count = cur.fetchone()[0]
    print(f"\n✓ Verification: {count:,} rows in arxiv_chunks")
    
    cur.close()
    return count


def run_subprocess(script_name, description):
    """Run a Python script as subprocess and report status."""
    print("\n" + "="*70)
    print(f"{description}")
    print("="*70)
    print(f"\nRunning: python {script_name}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            capture_output=False,  # Show live output
            text=True
        )
        
        elapsed = time.time() - start_time
        print(f"\n✓ {script_name} completed successfully ({elapsed:.1f}s)")
        return True
        
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n✗ {script_name} failed (exit code {e.returncode}) ({elapsed:.1f}s)")
        print(f"Error: {e}")
        return False


def verify_data_integrity(conn):
    """Verify all tables are populated correctly and data is consistent."""
    print("\n" + "="*70)
    print("VERIFYING DATA INTEGRITY")
    print("="*70)
    
    cur = conn.cursor()
    
    # Check all table counts
    tables = [
        'arxiv_chunks',
        'layer1_embeddings_128d',
        'layer1_bm25_sparse',
        'layer2_triplet_bm25'
    ]
    
    counts = {}
    for table in tables:
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        counts[table] = cur.fetchone()[0]
        print(f"  {table:30s}: {counts[table]:>10,} rows")
    
    # Check if arxiv_chunks has embedding column (added by update_postgres_256d.py)
    cur.execute("""
    SELECT column_name, data_type 
    FROM information_schema.columns 
    WHERE table_name = 'arxiv_chunks' AND column_name = 'embedding'
    """)
    embedding_col = cur.fetchone()
    
    if embedding_col:
        print(f"\n  ✓ arxiv_chunks.embedding column exists: {embedding_col[1]}")
        
        # Check embedding dimension
        cur.execute("SELECT pg_column_size(embedding) FROM arxiv_chunks LIMIT 1")
        emb_size = cur.fetchone()[0]
        expected_size_256d = 256 * 4 + 8  # 4 bytes per float32, 8 bytes overhead
        print(f"  ✓ Embedding storage size: {emb_size} bytes (expected ~{expected_size_256d} for vector(256))")
    else:
        print(f"\n  ⚠ arxiv_chunks.embedding column NOT found (update_postgres_256d.py may have failed)")
    
    # Validate counts are consistent
    base_count = counts['arxiv_chunks']
    all_match = all(count == base_count for count in counts.values())
    
    if all_match:
        print(f"\n✓ All tables have matching row counts: {base_count:,}")
    else:
        print(f"\n⚠ Row count mismatch detected:")
        for table, count in counts.items():
            if count != base_count:
                diff = count - base_count
                print(f"     {table}: {count:,} ({diff:+,} difference)")
    
    # Sample spot checks
    print("\n" + "-"*70)
    print("SPOT CHECKS")
    print("-"*70)
    
    # Check Layer 1 embedding dimensions
    cur.execute("SELECT chunk_id, embedding FROM layer1_embeddings_128d LIMIT 1")
    sample_id, sample_emb = cur.fetchone()
    sample_emb_array = np.array(sample_emb, dtype=np.float32)
    print(f"  Layer 1 (128d): chunk_id={sample_id}, dim={len(sample_emb_array)}, norm={np.linalg.norm(sample_emb_array):.4f}")
    
    # Check Layer 1 BM25 sparsity
    cur.execute("SELECT chunk_id, bm25_vector FROM layer1_bm25_sparse LIMIT 1")
    sample_id, sample_vec = cur.fetchone()
    print(f"  Layer 1 BM25: chunk_id={sample_id}, nonzero_terms={len(sample_vec)}")
    
    # Check Layer 2 triplet BM25 sparsity
    cur.execute("""
    SELECT chunk_id, triplet_bm25_vector 
    FROM layer2_triplet_bm25 
    WHERE jsonb_typeof(triplet_bm25_vector) = 'object' AND triplet_bm25_vector != '{{}}'::jsonb
    LIMIT 1
    """)
    result = cur.fetchone()
    if result:
        sample_id, sample_vec = result
        print(f"  Layer 2 triplet BM25: chunk_id={sample_id}, nonzero_triplets={len(sample_vec)}")
    else:
        print(f"  Layer 2 triplet BM25: No non-empty vectors found")
    
    # Check Layer 2 embedding if exists
    if embedding_col:
        cur.execute("SELECT chunk_id, embedding FROM arxiv_chunks WHERE embedding IS NOT NULL LIMIT 1")
        result = cur.fetchone()
        if result:
            sample_id, sample_emb = result
            sample_emb_array = np.array(sample_emb, dtype=np.float32)
            print(f"  Layer 2 (256d): chunk_id={sample_id}, dim={len(sample_emb_array)}, norm={np.linalg.norm(sample_emb_array):.4f}")
    
    cur.close()
    
    return all_match


def print_final_summary(conn, start_time):
    """Print comprehensive summary of migration."""
    elapsed = time.time() - start_time
    
    print("\n" + "="*70)
    print("PGVECTOR MIGRATION COMPLETE")
    print("="*70)
    
    cur = conn.cursor()
    
    # Database sizes
    cur.execute("""
    SELECT 
        schemaname,
        tablename,
        pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS total_size,
        pg_size_pretty(pg_relation_size(schemaname||'.'||tablename)) AS table_size,
        pg_size_pretty(pg_indexes_size(schemaname||'.'||tablename)) AS indexes_size
    FROM pg_tables
    WHERE schemaname = 'public' AND tablename IN (
        'arxiv_chunks',
        'layer1_embeddings_128d',
        'layer1_bm25_sparse',
        'layer2_triplet_bm25'
    )
    ORDER BY tablename
    """)
    
    print("\nDatabase Size Summary:")
    print("  " + "-"*66)
    print(f"  {'Table':<30s} {'Total':<12s} {'Table':<12s} {'Indexes':<12s}")
    print("  " + "-"*66)
    
    for schema, table, total, table_size, indexes in cur.fetchall():
        print(f"  {table:<30s} {total:<12s} {table_size:<12s} {indexes:<12s}")
    
    print("  " + "-"*66)
    
    # Total database size
    cur.execute("SELECT pg_size_pretty(pg_database_size(current_database()))")
    total_db_size = cur.fetchone()[0]
    print(f"  Total database size: {total_db_size}")
    
    # Indexes created
    cur.execute("""
    SELECT tablename, indexname, indexdef
    FROM pg_indexes
    WHERE schemaname = 'public' AND tablename IN (
        'arxiv_chunks',
        'layer1_embeddings_128d',
        'layer1_bm25_sparse',
        'layer2_triplet_bm25'
    )
    ORDER BY tablename, indexname
    """)
    
    print("\nIndexes Created:")
    for table, index, indexdef in cur.fetchall():
        index_type = 'HNSW' if 'hnsw' in indexdef.lower() else 'GIN' if 'gin' in indexdef.lower() else 'BTREE'
        print(f"  [{index_type:6s}] {table}.{index}")
    
    cur.close()
    
    print(f"\n✓ Total migration time: {elapsed/60:.1f} minutes")
    print("\nNext steps:")
    print("  1. Update three_layer_gist_retriever.py to add pgvector query support")
    print("  2. Update query_three_layer.py to add --use-pgvector flag")
    print("  3. Test queries: python query_three_layer.py 'your query' --use-pgvector")
    print("="*70)


def main():
    start_time = time.time()
    
    print("\n" + "="*70)
    print("PGVECTOR MIGRATION MASTER ORCHESTRATOR")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # ========================================================================
    # CONFIGURATION
    # ========================================================================
    
    chunks_path = Path("checkpoints/chunks.msgpack")
    
    db_config = {
        'host': 'localhost',
        'port': 5432,
        'dbname': 'langchain',
        'user': 'langchain',
        'password': 'langchain'
    }
    
    # ========================================================================
    # LOAD CHUNKS
    # ========================================================================
    
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
    
    # ========================================================================
    # CONNECT TO POSTGRESQL
    # ========================================================================
    
    print(f"\nConnecting to PostgreSQL {db_config['host']}:{db_config['port']}/{db_config['dbname']}...")
    conn = psycopg2.connect(**db_config)
    conn.autocommit = False
    print("  ✓ Connected")
    
    # ========================================================================
    # PHASE 1: CREATE BASE TABLE
    # ========================================================================
    
    base_count = create_base_arxiv_chunks_table(conn, chunks)
    
    if base_count != len(chunks):
        raise ValueError(f"Base table row count mismatch: {base_count} != {len(chunks)}")
    
    # ========================================================================
    # PHASE 2: INGEST LAYER 1 EMBEDDINGS (128d with PCA)
    # ========================================================================
    
    success = run_subprocess(
        'ingest_layer1_embeddings_128d.py',
        'PHASE 2: LAYER 1 EMBEDDINGS (128d) INGESTION'
    )
    
    if not success:
        print("\n✗ Migration failed at Phase 2 (Layer 1 embeddings)")
        conn.close()
        sys.exit(1)
    
    # ========================================================================
    # PHASE 3: INGEST LAYER 1 BM25 SPARSE
    # ========================================================================
    
    success = run_subprocess(
        'ingest_layer1_bm25_sparse_csr.py',
        'PHASE 3: LAYER 1 BM25 SPARSE VECTORS INGESTION'
    )
    
    if not success:
        print("\n✗ Migration failed at Phase 3 (Layer 1 BM25)")
        conn.close()
        sys.exit(1)
    
    # ========================================================================
    # PHASE 4: INGEST LAYER 2 TRIPLET BM25
    # ========================================================================
    
    success = run_subprocess(
        'ingest_layer2_triplet_bm25.py',
        'PHASE 4: LAYER 2 TRIPLET BM25 SPARSE VECTORS INGESTION'
    )
    
    if not success:
        print("\n✗ Migration failed at Phase 4 (Layer 2 triplet BM25)")
        conn.close()
        sys.exit(1)
    
    # ========================================================================
    # PHASE 5: UPDATE ARXIV_CHUNKS WITH 256d EMBEDDINGS
    # ========================================================================
    
    success = run_subprocess(
        'update_postgres_256d.py',
        'PHASE 5: LAYER 2 EMBEDDINGS (256d) UPDATE'
    )
    
    if not success:
        print("\n✗ Migration failed at Phase 5 (Layer 2 embeddings)")
        conn.close()
        sys.exit(1)
    
    # ========================================================================
    # PHASE 6: VERIFY DATA INTEGRITY
    # ========================================================================
    
    all_match = verify_data_integrity(conn)
    
    if not all_match:
        print("\n⚠ Data integrity check revealed mismatches (see above)")
        print("  Migration completed but with potential issues")
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    
    print_final_summary(conn, start_time)
    
    conn.close()


if __name__ == '__main__':
    main()
