"""
Populate PostgreSQL with arxiv chunks using pre-computed Qwen3 embeddings

Uses:
- Pre-computed Qwen3 embeddings from checkpoints/chunk_embeddings_qwen3.msgpack
- Simple tokenizer for BM25 (regex-based for consistency)

Estimated time: ~7 minutes for 161,389 chunks
"""

import msgpack
import numpy as np
from pathlib import Path
from tqdm import tqdm
import psycopg2
from psycopg2.extras import execute_values
from collections import Counter, defaultdict
from scipy.sparse import lil_matrix, csr_matrix
import sys

# Force UTF-8 output encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Load chunks
print("Loading chunks from checkpoints/chunks.msgpack...")
with open("checkpoints/chunks.msgpack", "rb") as f:
    chunks = msgpack.unpackb(f.read(), raw=False)
print(f"[OK] Loaded {len(chunks):,} chunks")

# Load or compute 64d embeddings
embeddings_checkpoint = Path("checkpoints/chunk_embeddings_64d.msgpack")

if embeddings_checkpoint.exists():
    print("\n[OK] Loading pre-computed 64d embeddings from checkpoint...")
    with open(embeddings_checkpoint, 'rb') as f:
        embed_data = msgpack.unpackb(f.read(), strict_map_key=False)
    embeddings_64d = np.array(embed_data['embeddings']).reshape(embed_data['shape'])
    print(f"  Shape: {embeddings_64d.shape}")
else:
    # Load Qwen3 embeddings (256d) - we'll use PCA to reduce to 64d
    print("\nLoading Qwen3 embeddings from checkpoints/chunk_embeddings_qwen3.msgpack...")
    with open("checkpoints/chunk_embeddings_qwen3.msgpack", "rb") as f:
        qwen3_data = msgpack.unpackb(f.read(), raw=False)
    
    # Extract embeddings array
    embeddings = np.array(qwen3_data['embeddings']).reshape(qwen3_data['shape'])
    print(f"[OK] Loaded embeddings: shape={embeddings.shape}")
    
    # PCA reduction: 256d -> 64d
    print("\nReducing embeddings: 256d -> 64d...")
    from sklearn.decomposition import PCA
    pca = PCA(n_components=64)
    embeddings_64d = pca.fit_transform(embeddings)
    print(f"[OK] Reduced to: shape={embeddings_64d.shape}, explained_variance={pca.explained_variance_ratio_.sum():.3f}")
    
    # Save checkpoint
    print("  Saving 64d embeddings checkpoint...")
    with open(embeddings_checkpoint, 'wb') as f:
        f.write(msgpack.packb({
            'embeddings': embeddings_64d.tolist(),
            'shape': embeddings_64d.shape,
            'dtype': str(embeddings_64d.dtype)
        }))

# Build or load BM25 index
bm25_checkpoint = Path("checkpoints/chunk_bm25_sparse.msgpack")

if bm25_checkpoint.exists():
    print("\n[OK] Loading pre-computed BM25 index from checkpoint...")
    with open(bm25_checkpoint, 'rb') as f:
        bm25_data = msgpack.unpackb(f.read(), strict_map_key=False)
    vocab = bm25_data['vocab']
    sparse_matrix = csr_matrix(
        (bm25_data['data'], bm25_data['indices'], bm25_data['indptr']),
        shape=bm25_data['shape']
    )
    print(f"  Vocab size: {len(vocab):,}")
    print(f"  Sparse matrix: shape={sparse_matrix.shape}, nnz={sparse_matrix.nnz:,}")
else:
    print("\nBuilding BM25 index...")
    def tokenize(text):
        """Simple tokenization - lowercase alphanumeric words"""
        import re
        return re.findall(r'\b[a-z0-9]{2,}\b', text.lower())
    
    # Build vocabulary from all chunks with fixed indices
    print("  Building vocabulary...")
    vocab_counter = Counter()
    for chunk in tqdm(chunks, desc="    Tokenizing", leave=False):
        tokens = tokenize(chunk['text'])
        vocab_counter.update(set(tokens))
    
    # Filter min_df=2 and create vocab with sorted words for consistent indices
    filtered_words = sorted([word for word, count in vocab_counter.items() if count >= 2])
    vocab = {word: idx for idx, word in enumerate(filtered_words)}
    print(f"  [OK] Vocab size: {len(vocab):,}")
    
    # Compute IDF
    print("  Computing IDF...")
    N = len(chunks)
    df = defaultdict(int)
    for chunk in tqdm(chunks, desc="    DF", leave=False):
        tokens = tokenize(chunk['text'])
        for token in set(tokens):
            if token in vocab:
                df[vocab[token]] += 1
    
    idf = {tid: np.log((N - doc_freq + 0.5) / (doc_freq + 0.5) + 1)
           for tid, doc_freq in df.items()}
    
    # Build sparse BM25 matrix
    print("  Building sparse matrix...")
    sparse_matrix = lil_matrix((len(chunks), len(vocab)), dtype=np.float32)
    k1 = 1.5
    b = 0.75
    
    # Compute average doc length
    avg_dl = np.mean([len(tokenize(chunk['text'])) for chunk in chunks])
    
    for i, chunk in enumerate(tqdm(chunks, desc="    BM25", leave=False)):
        tokens = tokenize(chunk['text'])
        dl = len(tokens)
        tf = Counter()
        for token in tokens:
            if token in vocab:
                tf[vocab[token]] += 1
        
        for tid, freq in tf.items():
            idf_val = idf.get(tid, 0)
            score = idf_val * (freq * (k1 + 1)) / (freq + k1 * (1 - b + b * dl / avg_dl))
            sparse_matrix[i, tid] = score
    
    sparse_matrix = sparse_matrix.tocsr()
    print(f"  [OK] Sparse matrix: shape={sparse_matrix.shape}, nnz={sparse_matrix.nnz:,}")
    
    # Save checkpoint
    print("  Saving BM25 checkpoint...")
    with open(bm25_checkpoint, 'wb') as f:
        f.write(msgpack.packb({
            'vocab': vocab,
            'data': sparse_matrix.data.tolist(),
            'indices': sparse_matrix.indices.tolist(),
            'indptr': sparse_matrix.indptr.tolist(),
            'shape': sparse_matrix.shape
        }))

# Connect to PostgreSQL
print("\nConnecting to PostgreSQL...")
conn = psycopg2.connect(
    host="localhost",
    database="langchain",
    user="langchain",
    password="langchain",
    port=5432
)
cur = conn.cursor()

# Create table with pgvector extension
table_name = "arxiv_chunks"
print(f"Creating table {table_name}...")

# Drop if exists
cur.execute(f"DROP TABLE IF EXISTS {table_name}")

# Create pgvector extension
cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

# Create table with JSONB for sparse BM25
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
print(f"[OK] Table created")

# Insert chunks in batches
print(f"\nInserting {len(chunks):,} chunks...")
batch_size = 1000
total_batches = (len(chunks) + batch_size - 1) // batch_size

for batch_idx in tqdm(range(0, len(chunks), batch_size), total=total_batches, desc="  Batches"):
    batch_chunks = chunks[batch_idx:batch_idx + batch_size]
    batch_embeddings = embeddings_64d[batch_idx:batch_idx + batch_size]
    
    # Prepare data
    values = []
    for i, chunk in enumerate(batch_chunks):
        global_idx = batch_idx + i
        row = sparse_matrix.getrow(global_idx)
        
        # Format sparse vector as JSONB {index: value}
        import json
        sparse_dict = {int(idx): float(val) for idx, val in zip(row.indices, row.data)}
        sparse_json = json.dumps(sparse_dict)
        
        values.append((
            chunk['doc_id'],
            chunk['paper_id'],
            chunk['section_idx'],
            chunk['chunk_idx'],
            chunk['text'],
            batch_embeddings[i].tolist(),
            sparse_json
        ))
    
    # Bulk insert
    execute_values(
        cur,
        f"""
        INSERT INTO {table_name} 
        (chunk_id, paper_id, section_idx, chunk_idx, content, embedding, bm25_sparse)
        VALUES %s
        """,
        values
    )
    conn.commit()

print(f"[OK] Inserted {len(chunks):,} chunks")

# Create indexes
print("\nCreating indexes...")

# Vector index using HNSW
cur.execute(f"""
    CREATE INDEX ON {table_name} 
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64)
""")

# JSONB index for sparse BM25
cur.execute(f"""
    CREATE INDEX ON {table_name} 
    USING gin (bm25_sparse jsonb_path_ops)
""")

# Text search index
cur.execute(f"""
    CREATE INDEX ON {table_name} 
    USING gin (to_tsvector('english', content))
""")

# Paper ID index
cur.execute(f"CREATE INDEX ON {table_name} (paper_id)")

conn.commit()
print(f"[OK] Indexes created")

# Verify
cur.execute(f"SELECT COUNT(*) FROM {table_name}")
count = cur.fetchone()[0]
print(f"\n[OK] Final verification: {count:,} rows in {table_name}")

conn.close()
print("\n[OK] PostgreSQL population complete!")
