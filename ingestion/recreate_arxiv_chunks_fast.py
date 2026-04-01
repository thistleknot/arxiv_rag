"""
Recreate arxiv_chunks with embedding vector(256) - FAST approach
Drop table and recreate with embedding column already defined, then bulk INSERT
"""
import psycopg2
from psycopg2.extras import execute_values
import msgpack
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Config
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'dbname': 'langchain',
    'user': 'langchain',
    'password': 'langchain'
}

def main():
    print("\n" + "="*70)
    print("RECREATING arxiv_chunks WITH 256d EMBEDDINGS (FAST METHOD)")
    print("="*70)
    
    # Load data
    print("\n[1/5] Loading chunks from checkpoints/chunks.msgpack...")
    with open(Path("checkpoints/chunks.msgpack"), 'rb') as f:
        chunks = msgpack.unpackb(f.read(), strict_map_key=False)
    print(f"  ✓ Loaded {len(chunks):,} chunks")
    
    print("\n[2/5] Loading 256d embeddings from checkpoints/chunk_embeddings_qwen3.msgpack...")
    with open(Path("checkpoints/chunk_embeddings_qwen3.msgpack"), 'rb') as f:
        embeddings_data = msgpack.unpackb(f.read(), raw=False)
    
    # Handle dict or array format
    if isinstance(embeddings_data, dict):
        embeddings = np.array(embeddings_data['embeddings'])
    else:
        embeddings = np.array(embeddings_data)
    print(f"  ✓ Loaded embeddings: {embeddings.shape}")
    
    # Connect to DB
    print("\n[3/5] Connecting to PostgreSQL...")
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    print("  ✓ Connected")
    
    # Drop and recreate table
    print("\n[4/5] Dropping and recreating arxiv_chunks table...")
    cur.execute("DROP TABLE IF EXISTS arxiv_chunks CASCADE")
    cur.execute("""
        CREATE TABLE arxiv_chunks (
            chunk_id TEXT PRIMARY KEY,
            paper_id TEXT,
            section_idx INTEGER,
            chunk_idx INTEGER,
            content TEXT,
            embedding vector(256),
            created_at TIMESTAMP DEFAULT NOW()
        )
    """)
    conn.commit()
    print("  ✓ Table recreated with embedding vector(256) column")
    
    # Prepare data for bulk insert
    print("\n[5/5] Bulk inserting chunks with embeddings (batch_size=1000)...")
    rows = []
    for i, chunk in enumerate(chunks):
        chunk_id = chunk.get('doc_id') or chunk.get('chunk_id')
        paper_id = chunk.get('paper_id', '')
        section_idx = chunk.get('section_idx', 0)
        chunk_idx = chunk.get('chunk_idx', 0)
        content = chunk.get('text', '')
        embedding_vec = embeddings[i].tolist()
        
        rows.append((chunk_id, paper_id, section_idx, chunk_idx, content, embedding_vec))
    
    # Bulk insert with execute_values (fast!)
    batch_size = 1000
    for i in tqdm(range(0, len(rows), batch_size), desc="  Inserting batches"):
        batch = rows[i:i+batch_size]
        execute_values(
            cur,
            "INSERT INTO arxiv_chunks (chunk_id, paper_id, section_idx, chunk_idx, content, embedding) VALUES %s",
            batch,
            page_size=batch_size
        )
    conn.commit()
    print(f"  ✓ Inserted {len(rows):,} rows with embeddings")
    
    # Verify
    print("\nVerifying insertion...")
    cur.execute("SELECT COUNT(*) FROM arxiv_chunks")
    count = cur.fetchone()[0]
    print(f"  ✓ Row count: {count:,}")
    
    cur.execute("SELECT COUNT(*) FROM arxiv_chunks WHERE embedding IS NOT NULL")
    emb_count = cur.fetchone()[0]
    print(f"  ✓ Non-null embeddings: {emb_count:,} ({emb_count/count*100:.1f}%)")
    
    cur.execute("SELECT chunk_id FROM arxiv_chunks LIMIT 1")
    sample = cur.fetchone()
    print(f"  ✓ Sample chunk_id: {sample[0]}")
    print(f"  ✓ Embedding dimensions: 256 (as defined in schema)")
    
    # Create HNSW index
    print("\nCreating HNSW index on embedding...")
    print("  (This may take a few minutes for 161k vectors)")
    cur.execute("""
        CREATE INDEX idx_arxiv_chunks_embedding_hnsw 
        ON arxiv_chunks 
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
    """)
    conn.commit()
    print("  ✓ HNSW index created")
    
    # Final stats
    print("\nFinal statistics:")
    cur.execute("SELECT pg_size_pretty(pg_table_size('arxiv_chunks'))")
    table_size = cur.fetchone()[0]
    print(f"  Table size: {table_size}")
    
    cur.execute("SELECT pg_size_pretty(pg_indexes_size('arxiv_chunks'))")
    index_size = cur.fetchone()[0]
    print(f"  Index size: {index_size}")
    
    cur.execute("SELECT pg_size_pretty(pg_total_relation_size('arxiv_chunks'))")
    total_size = cur.fetchone()[0]
    print(f"  Total size: {total_size}")
    
    cur.close()
    conn.close()
    
    print("\n" + "="*70)
    print("✓ arxiv_chunks RECREATED WITH 256d EMBEDDINGS")
    print("="*70)
    print(f"✓ Table recreated: arxiv_chunks")
    print(f"✓ Rows inserted: {count:,}")
    print(f"✓ Embeddings: {emb_count:,} (256d)")
    print(f"✓ HNSW index: idx_arxiv_chunks_embedding_hnsw")
    print(f"✓ Total size: {total_size}")
    print("="*70)

if __name__ == '__main__':
    main()
