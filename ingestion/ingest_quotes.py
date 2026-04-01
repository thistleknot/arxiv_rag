"""
Ingest Quotes Dataset into PostgreSQL with pgvector

Creates quotes table with same structure as arxiv:
- chunk_id, quote_id, chunk_idx, content
- embedding (GIST), bm25_sparse, graph_sparse

For quotes, we don't split into chunks - each quote is a single "chunk"
with chunk_idx=0.
"""

import json
import psycopg2
import numpy as np
from typing import List, Dict, Any
import struct
from tqdm import tqdm

# Same config as arxiv
PG_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'langchain',
    'user': 'langchain',
    'password': 'langchain'
}

MODEL_PATH = 'model2vec_jina'  # GIST model


def load_gist_model():
    """Load Model2Vec GIST model for embeddings with PCA reduction."""
    try:
        from model2vec import StaticModel
        from sklearn.decomposition import PCA
        
        model = StaticModel.from_pretrained(MODEL_PATH)
        print(f"✓ Loaded GIST model: {MODEL_PATH} (dim={model.dim})")
        
        # We need PCA to reduce from 512 → 64 dims
        # Load quotes and fit PCA
        print("Fitting PCA for 512 → 64 dimension reduction...")
        with open('quotes_dataset.json', 'r', encoding='utf-8') as f:
            quotes = json.load(f)
        
        # Sample for PCA fitting (use all quotes, they're small)
        texts = [f"{q['text']} - {q.get('author', 'Unknown')}" for q in quotes[:1000]]
        embeddings = model.encode(texts)
        
        pca = PCA(n_components=128)  # 25% of 512d (consistent with Qwen3 strategy)
        pca.fit(embeddings)
        
        explained = sum(pca.explained_variance_ratio_) * 100
        print(f"✓ PCA fitted: 512d→128d, {explained:.1f}% variance retained")
        
        return model, pca
        
    except Exception as e:
        print(f"✗ Failed to load GIST model: {e}")
        return None, None


def create_quotes_table(conn):
    """Create quotes table with same schema as arxiv."""
    with conn.cursor() as cur:
        # Drop existing table
        cur.execute("DROP TABLE IF EXISTS quotes CASCADE")
        
        # Create table
        cur.execute("""
            CREATE TABLE quotes (
                chunk_id TEXT PRIMARY KEY,
                quote_id TEXT NOT NULL,
                chunk_idx INTEGER NOT NULL,
                content TEXT NOT NULL,
                embedding vector(128),
                bm25_sparse sparsevec(30522),
                graph_sparse sparsevec(30522)
            )
        """)
        
        # Create indexes
        print("Creating indexes...")
        
        # IVFFlat for embeddings (fast approximate search)
        cur.execute("""
            CREATE INDEX quotes_embedding_idx 
            ON quotes 
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100)
        """)
        
        # HNSW for graph_sparse (fast graph search)
        cur.execute("""
            CREATE INDEX quotes_graph_sparse_idx 
            ON quotes 
            USING hnsw (graph_sparse sparsevec_cosine_ops)
            WITH (m = 16, ef_construction = 64)
        """)
        
        # Regular indexes for filtering
        cur.execute("CREATE INDEX quotes_quote_id_idx ON quotes (quote_id)")
        cur.execute("CREATE INDEX quotes_chunk_idx_idx ON quotes (chunk_idx)")
        
        conn.commit()
        print("✓ Created quotes table with indexes")


def encode_sparse_vector(indices: List[int], values: List[float], dim: int = 30522) -> str:
    """
    Encode sparse vector for PostgreSQL sparsevec type.
    Format: '{i1:v1,i2:v2,...}/dim'
    """
    if not indices:
        return '{}/' + str(dim)
    
    pairs = [f'{i}:{v}' for i, v in zip(indices, values)]
    return '{' + ','.join(pairs) + '}/' + str(dim)


def ingest_quotes(conn, quotes_data: List[Dict], gist_model, pca):
    """Ingest quotes into PostgreSQL."""
    print(f"Ingesting {len(quotes_data)} quotes...")
    
    batch_size = 100
    batches = [quotes_data[i:i+batch_size] for i in range(0, len(quotes_data), batch_size)]
    
    with conn.cursor() as cur:
        for batch in tqdm(batches, desc="Inserting batches"):
            for quote in batch:
                quote_id = quote['quote_id']
                text = quote['text']
                author = quote.get('author', 'Unknown')
                
                # For quotes, use single chunk (chunk_idx=0)
                chunk_id = f"{quote_id}_0"
                chunk_idx = 0
                
                # Add author to content
                content = f"{text} - {author}"
                
                # Generate GIST embedding
                if gist_model and pca:
                    embedding_512 = gist_model.encode([content])[0]
                    embedding = pca.transform([embedding_512])[0]  # Reduce to 128 (25%)
                    embedding_list = embedding.tolist()
                else:
                    embedding_list = None
                
                # Initialize empty sparse vectors (will be populated by separate scripts)
                empty_sparse = '{}/30522'
                
                # Insert
                cur.execute("""
                    INSERT INTO quotes (
                        chunk_id, quote_id, chunk_idx, content,
                        embedding, bm25_sparse, graph_sparse
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (
                    chunk_id, quote_id, chunk_idx, content,
                    embedding_list, empty_sparse, empty_sparse
                ))
            
            conn.commit()
    
    # Get final count
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM quotes")
        count = cur.fetchone()[0]
    
    print(f"✓ Ingested {count} quotes into database")


def main():
    # Load quotes dataset
    print("Loading quotes dataset...")
    with open('quotes_dataset.json', 'r', encoding='utf-8') as f:
        quotes_data = json.load(f)
    print(f"✓ Loaded {len(quotes_data)} quotes")
    
    # Load GIST model with PCA
    gist_model, pca = load_gist_model()
    
    # Connect to database
    print("Connecting to PostgreSQL...")
    conn = psycopg2.connect(**PG_CONFIG)
    print("✓ Connected")
    
    try:
        # Create table
        create_quotes_table(conn)
        
        # Ingest quotes
        ingest_quotes(conn, quotes_data, gist_model, pca)
        
        print("\n✓ Quotes ingestion complete!")
        print("\nNext steps:")
        print("1. Run BM25 vocabulary builder on quotes table")
        print("2. Run graph sparse encoder on quotes table")
        print("3. Test with query_quotes.py")
        
    finally:
        conn.close()


if __name__ == '__main__':
    main()
