"""
Populate PostgreSQL with chunks, embeddings, and BM25 sparse vectors.
Uses EXISTING bm25_vocab.msgpack to ensure consistency with retrieval.

JSONB Storage (not sparsevec):
- pgvector 0.8.1-pg18 doesn't have sparsevec extension
- Store as JSONB: {index: value} dictionary
- Retrieval uses manual dot product in SQL
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import msgpack
from scipy.sparse import csr_matrix, lil_matrix
from tqdm import tqdm
import psycopg2
from psycopg2.extras import execute_values

# UTF-8 encoding fix for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def main():
    # Configuration
    db_config = {
        'host': 'localhost',
        'port': 5432,
        'dbname': 'langchain',
        'user': 'langchain',
        'password': 'langchain'
    }
    table_name = 'arxiv_chunks'
    
    # ========================================================================
    # LOAD DATA FILES
    # ========================================================================
    
    print("\n" + "="*70)
    print("POSTGRESQL POPULATION (WITH EXISTING BM25 VOCAB)")
    print("="*70)
    
    # Load chunks
    chunks_path = Path("checkpoints/chunks.msgpack")
    print(f"\nLoading chunks from {chunks_path}...")
    with open(chunks_path, 'rb') as f:
        chunks_data = msgpack.unpackb(f.read(), strict_map_key=False)
    
    # Extract chunks list (handle both dict and list formats)
    if isinstance(chunks_data, dict) and 'chunks' in chunks_data:
        chunks = chunks_data['chunks']
    elif isinstance(chunks_data, list):
        chunks = chunks_data
    else:
        raise ValueError(f"Unexpected chunks format: {type(chunks_data)}")
    
    print(f"  [OK] {len(chunks):,} chunks")
    
    # Load 64d embeddings from checkpoint
    embeddings_checkpoint = Path("checkpoints/chunk_embeddings_64d.msgpack")
    print(f"\nLoading 64d embeddings from {embeddings_checkpoint}...")
    with open(embeddings_checkpoint, 'rb') as f:
        embed_data = msgpack.unpackb(f.read(), strict_map_key=False)
    
    embeddings_64d = np.array(embed_data['embeddings']).reshape(embed_data['shape'])
    print(f"  [OK] {embeddings_64d.shape} ({embeddings_64d.nbytes / 1024 / 1024:.1f} MB)")
    
    # Load EXISTING BM25 vocab (proper format with idf, avgdl, etc.)
    bm25_vocab_path = Path("bm25_vocab.msgpack")
    print(f"\nLoading EXISTING BM25 vocab from {bm25_vocab_path}...")
    with open(bm25_vocab_path, 'rb') as f:
        bm25_data = msgpack.unpackb(f.read(), strict_map_key=False)
    
    vocab = bm25_data['vocab']
    vocab_size = bm25_data['vocab_size']
    idf = {int(k): v for k, v in bm25_data['idf'].items()}
    avgdl = bm25_data['avgdl']
    k1 = bm25_data.get('k1', 1.5)
    b = bm25_data.get('b', 0.75)
    
    print(f"  [OK] vocab: {vocab_size:,}, avgdl: {avgdl:.2f}, k1: {k1}, b: {b}")
    
    # ========================================================================
    # BUILD BM25 SPARSE MATRIX FROM EXISTING VOCAB
    # ========================================================================
    
    print("\nBuilding BM25 sparse vectors using existing vocab...")
    
    def tokenize(text):
        """Simple regex tokenizer - must match BM25 vocab builder"""
        import re
        return re.findall(r'\b[a-z0-9]{2,}\b', text.lower())
    
    # Build sparse matrix
    sparse_matrix = lil_matrix((len(chunks), vocab_size), dtype=np.float32)
    
    for i, chunk in enumerate(tqdm(chunks, desc="  Computing BM25 scores")):
        tokens = tokenize(chunk['text'])
        token_counts = defaultdict(int)
        for token in tokens:
            if token in vocab:
                token_counts[vocab[token]] += 1
        
        # BM25 scoring
        doc_len = len(tokens)
        for term_id, freq in token_counts.items():
            if term_id in idf:
                numerator = freq * (k1 + 1)
                denominator = freq + k1 * (1 - b + b * doc_len / avgdl)
                score = idf[term_id] * (numerator / denominator)
                sparse_matrix[i, term_id] = score
    
    sparse_matrix = sparse_matrix.tocsr()
    print(f"  [OK] shape={sparse_matrix.shape}, nnz={sparse_matrix.nnz:,}")
    
    # ========================================================================
    # POPULATE POSTGRESQL
    # ========================================================================
    
    print(f"\nConnecting to PostgreSQL {db_config['host']}:{db_config['port']}/{db_config['dbname']}...")
    conn = psycopg2.connect(**db_config)
    conn.autocommit = False
    cur = conn.cursor()
    print("  [OK]")
    
    # Drop and recreate table
    print(f"\nCreating table {table_name}...")
    cur.execute(f"DROP TABLE IF EXISTS {table_name}")
    
    cur.execute(f"""
    CREATE TABLE {table_name} (
        id SERIAL PRIMARY KEY,
        chunk_id TEXT UNIQUE NOT NULL,
        paper_id TEXT NOT NULL,
        section_idx INTEGER NOT NULL,
        chunk_idx INTEGER NOT NULL,
        content TEXT NOT NULL,
        embedding vector(64) NOT NULL,
        bm25_sparse JSONB NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    conn.commit()
    print("  [OK]")
    
    # Insert data in batches
    print(f"\nInserting {len(chunks):,} chunks...")
    batch_size = 1000
    
    for batch_start in tqdm(range(0, len(chunks), batch_size), desc="  Batches"):
        batch_end = min(batch_start + batch_size, len(chunks))
        values = []
        
        for local_idx in range(batch_end - batch_start):
            global_idx = batch_start + local_idx
            chunk = chunks[global_idx]
            
            # Build chunk_id
            chunk_id = chunk['doc_id']
            paper_id = chunk['paper_id']
            section_idx = chunk['section_idx']
            chunk_idx = chunk['chunk_idx']
            content = chunk['text']
            
            # Embedding
            embedding_vec = embeddings_64d[global_idx].tolist()
            
            # BM25 sparse as JSONB {index: value}
            row = sparse_matrix.getrow(global_idx)
            sparse_dict = {int(idx): float(val) for idx, val in zip(row.indices, row.data)}
            sparse_json = json.dumps(sparse_dict)
            
            values.append((chunk_id, paper_id, section_idx, chunk_idx, content, embedding_vec, sparse_json))
        
        # Batch insert
        execute_values(cur, f"""
            INSERT INTO {table_name} 
            (chunk_id, paper_id, section_idx, chunk_idx, content, embedding, bm25_sparse)
            VALUES %s
        """, values)
        conn.commit()
    
    print("  [OK]")
    
    # Create indexes
    print("\nCreating indexes...")
    
    # HNSW index for vector similarity (cosine distance)
    print("  1. HNSW index on embedding...")
    cur.execute(f"""
    CREATE INDEX ON {table_name} 
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64)
    """)
    conn.commit()
    print("     [OK]")
    
    # GIN index for JSONB BM25 sparse vectors
    print("  2. GIN index on bm25_sparse (JSONB)...")
    cur.execute(f"""
    CREATE INDEX ON {table_name} 
    USING gin (bm25_sparse jsonb_path_ops)
    """)
    conn.commit()
    print("     [OK]")
    
    # Text search index on content
    print("  3. Text search index on content...")
    cur.execute(f"""
    CREATE INDEX ON {table_name} 
    USING gin (to_tsvector('english', content))
    """)
    conn.commit()
    print("     [OK]")
    
    # B-tree index on paper_id for fast filtering
    print("  4. B-tree index on paper_id...")
    cur.execute(f"CREATE INDEX ON {table_name} (paper_id)")
    conn.commit()
    print("     [OK]")
    
    # Verify count
    cur.execute(f"SELECT COUNT(*) FROM {table_name}")
    final_count = cur.fetchone()[0]
    print(f"\nFinal count: {final_count:,} rows")
    
    cur.close()
    conn.close()
    
    print("\n" + "="*70)
    print("POPULATION COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
