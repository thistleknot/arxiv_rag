"""
Update PostgreSQL arxiv_chunks table to use 256d Model2Vec Qwen3 embeddings

This script:
1. Loads pre-computed 256d Qwen3 embeddings from checkpoints/chunk_embeddings_qwen3.msgpack
2. Updates the PostgreSQL table schema (if needed) to vector(256)
3. Updates all chunk embeddings with the 256d vectors

Note: The Qwen3 embeddings are already computed. This just updates PostgreSQL.
"""

import msgpack
import numpy as np
from pathlib import Path
from tqdm import tqdm
import psycopg2
from psycopg2.extras import execute_values
import sys

# Force UTF-8 output encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def main():
    print("=" * 70)
    print("UPDATE POSTGRESQL WITH 256D MODEL2VEC QWEN3 EMBEDDINGS")
    print("=" * 70)
    
    # Load chunks
    print("\n[1/5] Loading chunks...")
    with open("checkpoints/chunks.msgpack", "rb") as f:
        chunks = msgpack.unpackb(f.read(), raw=False)
    print(f"✓ Loaded {len(chunks):,} chunks")
    
    # Load Qwen3 256d embeddings
    print("\n[2/5] Loading Qwen3 256d embeddings...")
    with open("checkpoints/chunk_embeddings_qwen3.msgpack", "rb") as f:
        qwen3_data = msgpack.unpackb(f.read(), raw=False)
    
    embeddings = np.array(qwen3_data['embeddings']).reshape(qwen3_data['shape'])
    print(f"✓ Loaded embeddings: shape={embeddings.shape}, dtype={embeddings.dtype}")
    
    if embeddings.shape[0] != len(chunks):
        raise ValueError(f"Embedding count mismatch: {embeddings.shape[0]} vs {len(chunks)}")
    
    if embeddings.shape[1] != 256:
        raise ValueError(f"Expected 256d embeddings, got {embeddings.shape[1]}d")
    
    # Connect to PostgreSQL
    print("\n[3/5] Connecting to PostgreSQL...")
    conn = psycopg2.connect(
        host='localhost',
        port=5432,
        dbname='langchain',
        user='langchain',
        password='langchain'
    )
    cur = conn.cursor()
    print("✓ Connected to PostgreSQL")
    
    # Check current embedding dimension
    print("\n[4/5] Checking current schema...")
    cur.execute("""
        SELECT column_name, data_type, udt_name 
        FROM information_schema.columns 
        WHERE table_name = 'arxiv_chunks' AND column_name = 'embedding'
    """)
    result = cur.fetchone()
    
    if result:
        print(f"  Current column: {result[0]} | type: {result[1]} | udt: {result[2]}")
        
        # Check dimension
        cur.execute("SELECT COUNT(*) FROM arxiv_chunks WHERE embedding IS NOT NULL LIMIT 1")
        if cur.fetchone()[0] > 0:
            cur.execute("SELECT array_length(embedding::real[], 1) FROM arxiv_chunks WHERE embedding IS NOT NULL LIMIT 1")
            current_dim = cur.fetchone()[0]
            print(f"  Current dimension: {current_dim}d")
            
            if current_dim != 256:
                print(f"  ⚠️ Dimension mismatch: {current_dim}d → 256d")
                print("  Altering column to vector(256)...")
                
                # Drop existing column and recreate
                cur.execute("ALTER TABLE arxiv_chunks DROP COLUMN IF EXISTS embedding CASCADE")
                cur.execute("ALTER TABLE arxiv_chunks ADD COLUMN embedding vector(256)")
                conn.commit()
                print("  ✓ Column altered to vector(256)")
            else:
                print("  ✓ Dimension already 256d")
    else:
        print("  Creating embedding column as vector(256)...")
        cur.execute("ALTER TABLE arxiv_chunks ADD COLUMN embedding vector(256)")
        conn.commit()
        print("  ✓ Column created")
    
    # Update embeddings in batches
    print("\n[5/5] Updating embeddings...")
    batch_size = 1000
    
    for i in tqdm(range(0, len(chunks), batch_size), desc="  Updating batches"):
        batch_chunks = chunks[i:i+batch_size]
        batch_embeddings = embeddings[i:i+batch_size]
        
        # Prepare update data: (embedding_list, chunk_id)
        # Note: chunks['doc_id'] maps to PostgreSQL's 'chunk_id' column
        update_data = []
        for chunk, embedding in zip(batch_chunks, batch_embeddings):
            chunk_id = chunk['doc_id']  # doc_id in chunks.msgpack = chunk_id in PostgreSQL
            embedding_list = embedding.tolist()
            update_data.append((embedding_list, chunk_id))
        
        # Execute batch update
        execute_values(
            cur,
            """
            UPDATE arxiv_chunks SET embedding = data.emb
            FROM (VALUES %s) AS data(emb, chunk_id)
            WHERE arxiv_chunks.chunk_id = data.chunk_id
            """,
            update_data,
            template="(%s::vector(256), %s)"
        )
    
    conn.commit()
    print(f"✓ Updated {len(chunks):,} embeddings")
    
    # Verify update
    print("\n[Verification] Checking updated embeddings...")
    cur.execute("SELECT COUNT(*) FROM arxiv_chunks WHERE embedding IS NOT NULL")
    count = cur.fetchone()[0]
    print(f"  Non-null embeddings: {count:,}")
    
    cur.execute("SELECT array_length(embedding::real[], 1) FROM arxiv_chunks WHERE embedding IS NOT NULL LIMIT 1")
    dim = cur.fetchone()[0]
    print(f"  Dimension: {dim}d")
    
    # Check if index exists
    cur.execute("""
        SELECT indexname FROM pg_indexes 
        WHERE tablename = 'arxiv_chunks' AND indexname LIKE '%embedding%'
    """)
    indexes = cur.fetchall()
    if indexes:
        print(f"  Existing indexes: {[idx[0] for idx in indexes]}")
        print("  ⚠️ You may need to rebuild indexes for optimal performance")
        print("     Run: DROP INDEX IF EXISTS arxiv_chunks_embedding_idx;")
        print("          CREATE INDEX arxiv_chunks_embedding_idx ON arxiv_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);")
    else:
        print("  No embedding indexes found")
    
    cur.close()
    conn.close()
    
    print("\n" + "=" * 70)
    print("✅ COMPLETE: PostgreSQL updated with 256d embeddings")
    print("=" * 70)

if __name__ == '__main__':
    main()
