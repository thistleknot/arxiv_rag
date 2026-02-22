"""
Ingest Layer 2 Dense Embeddings (256d)

Core Thesis:
    Creates and populates layer2_embeddings_256d table from Qwen3 model2vec
    256d embeddings over original chunks. This is the L2 dense parallel path
    alongside layer2_triplet_bm25 (sparsevec over semantic triplets).

Schema:
    layer2_embeddings_256d (
        chunk_id TEXT PRIMARY KEY,
        embedding vector(256)
    )
    + HNSW cosine index

Source:
    checkpoints/chunk_embeddings_qwen3.msgpack → shape [161389, 256]
    checkpoints/chunks.msgpack → chunk_id = chunk['doc_id']
"""

import os
import sys
import msgpack
import numpy as np
import psycopg2
from psycopg2.extras import execute_values
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHUNKS_PATH = os.path.join(BASE_DIR, "checkpoints", "chunks.msgpack")
EMBEDDINGS_PATH = os.path.join(BASE_DIR, "checkpoints", "chunk_embeddings_qwen3.msgpack")

# ── DB ────────────────────────────────────────────────────────────────────────
DB_PARAMS = dict(
    host="localhost", port=5432,
    dbname="langchain", user="langchain", password="langchain"
)

BATCH_SIZE = 1000


def load_data():
    print("[1/4] Loading chunks...")
    with open(CHUNKS_PATH, "rb") as f:
        chunks = msgpack.unpack(f, raw=False)
    print(f"  ✓ {len(chunks):,} chunks")

    print("[2/4] Loading Qwen3 256d embeddings...")
    with open(EMBEDDINGS_PATH, "rb") as f:
        data = msgpack.unpack(f, raw=False)

    shape = data["shape"]
    embeddings = np.array(data["embeddings"], dtype=np.float32).reshape(shape)
    print(f"  ✓ embeddings shape: {embeddings.shape}")

    assert embeddings.shape[1] == 256, f"Expected 256d, got {embeddings.shape[1]}d"
    assert len(chunks) == embeddings.shape[0], (
        f"Count mismatch: {len(chunks)} chunks vs {embeddings.shape[0]} embeddings"
    )

    return chunks, embeddings


def create_table(cur, conn):
    print("[3/4] Creating layer2_embeddings_256d table + HNSW index...")

    cur.execute("""
        CREATE TABLE IF NOT EXISTS layer2_embeddings_256d (
            chunk_id TEXT PRIMARY KEY,
            embedding vector(256)
        )
    """)
    conn.commit()
    print("  ✓ Table created (or already exists)")

    # HNSW for fast cosine ANN
    cur.execute("""
        CREATE INDEX IF NOT EXISTS layer2_emb_256d_hnsw
            ON layer2_embeddings_256d
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64)
    """)
    conn.commit()
    print("  ✓ HNSW index created (or already exists)")


def ingest(chunks, embeddings, cur, conn):
    print("[4/4] Inserting embeddings...")

    # Check existing row count to support resume
    cur.execute("SELECT COUNT(*) FROM layer2_embeddings_256d")
    existing = cur.fetchone()[0]
    if existing > 0:
        print(f"  ⚠️  Table already has {existing:,} rows — using INSERT ON CONFLICT DO NOTHING")

    for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="  Batches"):
        batch_chunks = chunks[i:i + BATCH_SIZE]
        batch_embeddings = embeddings[i:i + BATCH_SIZE]

        rows = [
            (chunk["doc_id"], emb.tolist())
            for chunk, emb in zip(batch_chunks, batch_embeddings)
        ]

        execute_values(
            cur,
            """
            INSERT INTO layer2_embeddings_256d (chunk_id, embedding)
            VALUES %s
            ON CONFLICT (chunk_id) DO NOTHING
            """,
            rows,
            template="(%s, %s::vector(256))",
        )

    conn.commit()
    print(f"  ✓ Done")


def verify(cur):
    cur.execute("SELECT COUNT(*) FROM layer2_embeddings_256d")
    count = cur.fetchone()[0]

    cur.execute("""
        SELECT array_length(embedding::real[], 1)
        FROM layer2_embeddings_256d LIMIT 1
    """)
    dim_row = cur.fetchone()
    dim = dim_row[0] if dim_row else "N/A"

    cur.execute("""
        SELECT indexname FROM pg_indexes
        WHERE tablename = 'layer2_embeddings_256d'
    """)
    indexes = [r[0] for r in cur.fetchall()]

    print("\n── Verification ──────────────────────────────────────────")
    print(f"  Rows    : {count:,}")
    print(f"  Dim     : {dim}d")
    print(f"  Indexes : {indexes}")
    return count


def main():
    print("=" * 60)
    print("Ingest layer2_embeddings_256d (Qwen3 model2vec 256d)")
    print("=" * 60)

    chunks, embeddings = load_data()

    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()

    try:
        create_table(cur, conn)
        ingest(chunks, embeddings, cur, conn)
        count = verify(cur)
    finally:
        cur.close()
        conn.close()

    expected = len(chunks)
    if count >= expected:
        print(f"\n✅ COMPLETE — {count:,} rows in layer2_embeddings_256d")
    else:
        print(f"\n⚠️  Only {count:,}/{expected:,} rows inserted — check for errors")
        sys.exit(1)


if __name__ == "__main__":
    main()
