"""
_migrate_bm25_sparsevec.py
==========================

Migrate BM25 sparse vectors from JSONB to sparsevec(16000) signed hashing.

Two corpora, two independent sets of stat tables:
  Layer 1 (arxiv_chunks.content):
    - bm25_global_stats  / bm25_term_global      (layer1 IDF)
    - arxiv_chunks.bm25_sparse  sparsevec(16000)

  Layer 2 (stage6_with_hypernyms.msgpack triplet enriched_terms):
    - bm25_l2_stats      / bm25_l2_term_df        (layer2 IDF)
    - layer2_triplet_bm25.triplet_bm25_vector  sparsevec(16000)

Two-pass approach:
  Pass 1: compute corpus statistics in Python
  Pass 2: vectorize each doc with finalized IDF

Run:
  c:\\users\\user\\py310\\scripts\\python.exe _migrate_bm25_sparsevec.py
  c:\\users\\user\\py310\\scripts\\python.exe _migrate_bm25_sparsevec.py --layer2-only
  c:\\users\\user\\py310\\scripts\\python.exe _migrate_bm25_sparsevec.py --layer1-only
"""

import sys
import math
import argparse
from collections import defaultdict
from typing import Dict, List, Iterator, Tuple

import mmh3
import msgpack
import psycopg2
import numpy as np
from pgvector.psycopg2 import register_vector, SparseVector
from _refresh_bm25_stats import refresh_layer1, refresh_layer2

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DB_CONFIG = dict(
    host="localhost",
    port=5432,
    dbname="langchain",
    user="langchain",
    password="langchain",
)

MSGPACK_PATH = "triplet_checkpoints_full/stage6_with_hypernyms.msgpack"

N_BUCKETS = 16_000      # sparsevec(16000)
BM25_K1   = 1.5
BM25_B    = 0.75
BATCH     = 2_000       # rows per DB batch write

# ---------------------------------------------------------------------------
# Signed hashing (identical to query_modules/pgvector_retriever.py)
# ---------------------------------------------------------------------------

def hash_token(term: str) -> tuple:
    bucket = mmh3.hash(term, seed=0, signed=False) % N_BUCKETS
    sign   = 1.0 if mmh3.hash(term, seed=1, signed=True) >= 0 else -1.0
    return bucket, sign


def tokenize(text: str) -> List[str]:
    return text.lower().split()


# ---------------------------------------------------------------------------
# Stat table schemas (layer-independent names passed as params)
# ---------------------------------------------------------------------------

def create_stat_tables(cur, global_table: str, term_table: str) -> None:
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {global_table} (
            id           SMALLINT PRIMARY KEY DEFAULT 1,
            n_docs       BIGINT   NOT NULL DEFAULT 0,
            total_tokens BIGINT   NOT NULL DEFAULT 0
        )
    """)
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {term_table} (
            bucket   INT    PRIMARY KEY,
            doc_freq BIGINT NOT NULL DEFAULT 0
        )
    """)


def write_stats(
    cur,
    global_table: str,
    term_table: str,
    n_docs: int,
    total_tokens: int,
    bucket_df: Dict[int, int],
) -> None:
    # global stats (upsert single row)
    cur.execute(f"""
        INSERT INTO {global_table} (id, n_docs, total_tokens)
        VALUES (1, %s, %s)
        ON CONFLICT (id) DO UPDATE
            SET n_docs = EXCLUDED.n_docs,
                total_tokens = EXCLUDED.total_tokens
    """, (n_docs, total_tokens))

    # per-bucket doc_freq (batch upsert)
    rows = list(bucket_df.items())
    for start in range(0, len(rows), 5_000):
        chunk = rows[start:start + 5_000]
        args  = [(b, df) for b, df in chunk]
        cur.execute(f"""
            INSERT INTO {term_table} (bucket, doc_freq)
            SELECT v.bucket, v.doc_freq
            FROM (VALUES {','.join(['(%s,%s)'] * len(args))}) AS v(bucket, doc_freq)
            ON CONFLICT (bucket) DO UPDATE
                SET doc_freq = EXCLUDED.doc_freq
        """, [x for pair in args for x in pair])


# ---------------------------------------------------------------------------
# BM25 vectorization (uses pre-committed stats, NO staging table)
# ---------------------------------------------------------------------------

def compute_doc_vector(
    tokens: List[str],
    dl: int,
    avgdl: float,
    n: int,
    bucket_df: Dict[int, int],
) -> SparseVector:
    """
    Full Okapi BM25 TF*IDF sparsevec.  Signed hashing preserves dot-product
    expectation (Weinberger et al. 2009).
    """
    # raw signed TF per bucket
    bucket_raw: Dict[int, float] = defaultdict(float)
    for term in tokens:
        bucket, sign = hash_token(term)
        bucket_raw[bucket] += sign

    indices, values = [], []
    for bucket, signed_tf in bucket_raw.items():
        raw_tf = abs(signed_tf)
        tf_sat = (raw_tf * (BM25_K1 + 1)) / (
            raw_tf + BM25_K1 * (1 - BM25_B + BM25_B * dl / max(avgdl, 1))
        )
        df  = bucket_df.get(bucket, 0)
        idf = math.log((n - df + 0.5) / (df + 0.5) + 1)
        w   = tf_sat * idf
        if w > 0:
            indices.append(bucket)
            values.append(math.copysign(w, signed_tf))

    return SparseVector(dict(zip(indices, values)), N_BUCKETS)


# ---------------------------------------------------------------------------
# Layer 1: arxiv_chunks.content -> bm25_sparse sparsevec(16000)
# ---------------------------------------------------------------------------

def migrate_layer1(conn) -> None:
    print("\n" + "=" * 60)
    print("LAYER 1: arxiv_chunks.bm25_sparse -> sparsevec(16000)")
    print("=" * 60)

    cur = conn.cursor()

    # ---- schema: stat tables ------------------------------------------------
    print("Creating stat tables...")
    create_stat_tables(cur, "bm25_global_stats", "bm25_term_global")
    conn.commit()

    # ---- schema: bm25_sparse column -----------------------------------------
    print("Dropping old bm25_sparse column and re-adding as sparsevec(16000)...")
    cur.execute("ALTER TABLE arxiv_chunks DROP COLUMN IF EXISTS bm25_sparse")
    cur.execute("ALTER TABLE arxiv_chunks ADD COLUMN bm25_sparse sparsevec(16000)")
    conn.commit()
    print("  Column dropped and re-added (old JSONB data cleared)")

    # ---- Pass 1: corpus stats -----------------------------------------------
    print("\nPass 1: computing corpus statistics from arxiv_chunks.content...")
    cur.execute("SELECT COUNT(*) FROM arxiv_chunks WHERE content IS NOT NULL AND content != ''")
    total_rows = cur.fetchone()[0]
    print(f"  Found {total_rows:,} rows with content")

    bucket_df: Dict[int, int] = defaultdict(int)
    total_tokens = 0
    n_docs = 0

    cur.execute("""
        SELECT chunk_id, content FROM arxiv_chunks
        WHERE content IS NOT NULL AND content != ''
        ORDER BY chunk_id
    """)

    rows_processed = 0
    while True:
        batch = cur.fetchmany(BATCH * 2)
        if not batch:
            break
        for chunk_id, content in batch:
            tokens = tokenize(content)
            if not tokens:
                continue
            total_tokens += len(tokens)
            n_docs        += 1
            seen_buckets: set = set()
            for term in tokens:
                b, _ = hash_token(term)
                if b not in seen_buckets:
                    seen_buckets.add(b)
                    bucket_df[b] += 1
        rows_processed += len(batch)
        if rows_processed % 20_000 == 0:
            print(f"  Stats pass: {rows_processed:,}/{total_rows:,}")

    print(f"  Stats pass complete: n_docs={n_docs:,}, total_tokens={total_tokens:,}, "
          f"unique_buckets={len(bucket_df):,}")
    avgdl = total_tokens / max(n_docs, 1)

    # ---- Write stats to DB --------------------------------------------------
    print("Writing stats to DB...")
    write_stats(cur, "bm25_global_stats", "bm25_term_global",
                n_docs, total_tokens, dict(bucket_df))
    conn.commit()
    print("  Stats committed")

    # ---- Pass 2: vectorize & write ------------------------------------------
    print("\nPass 2: vectorizing and writing sparsevec(16000) to arxiv_chunks...")
    snapshot_df = dict(bucket_df)  # frozen corpus stats for all docs

    cur_read  = conn.cursor()
    cur_write = conn.cursor()
    cur_read.execute("""
        SELECT chunk_id, content FROM arxiv_chunks
        WHERE content IS NOT NULL AND content != ''
        ORDER BY chunk_id
    """)

    batch_buf: List[Tuple] = []
    rows_written = 0

    while True:
        rows = cur_read.fetchmany(BATCH)
        if not rows:
            break
        for chunk_id, content in rows:
            tokens = tokenize(content)
            if not tokens:
                continue
            vec = compute_doc_vector(tokens, len(tokens), avgdl, n_docs, snapshot_df)
            batch_buf.append((vec, chunk_id))

        if len(batch_buf) >= BATCH:
            cur_write.executemany(
                "UPDATE arxiv_chunks SET bm25_sparse = %s WHERE chunk_id = %s",
                batch_buf,
            )
            conn.commit()
            rows_written += len(batch_buf)
            print(f"  Written {rows_written:,}...")
            batch_buf = []

    if batch_buf:
        cur_write.executemany(
            "UPDATE arxiv_chunks SET bm25_sparse = %s WHERE chunk_id = %s",
            batch_buf,
        )
        conn.commit()
        rows_written += len(batch_buf)

    print(f"  Vectorization complete: {rows_written:,} rows written")

    # ---- HNSW index -------------------------------------------------------
    print("\nCreating HNSW index on arxiv_chunks.bm25_sparse...")
    try:
        cur.execute("DROP INDEX IF EXISTS arxiv_chunks_bm25_sparse_idx")
        cur.execute("""
            CREATE INDEX arxiv_chunks_bm25_sparse_idx
            ON arxiv_chunks USING hnsw (bm25_sparse sparsevec_ip_ops)
            WITH (m = 32, ef_construction = 100)
        """)
        conn.commit()
        print("  HNSW index created")
    except Exception as e:
        conn.rollback()
        print(f"  WARNING: Index creation skipped ({e})")
        print("  Data is written. Run CREATE INDEX manually after confirming pgvector version supports sparsevec HNSW.")
    # Refresh stats from the stored sparsevec (authoritative post-write pass)
    print("\nRefreshing stats from stored sparsevec...")
    refresh_layer1(conn)
    print("Layer 1 migration COMPLETE")


# ---------------------------------------------------------------------------
# Layer 2: stage6 enriched_terms -> layer2_triplet_bm25.triplet_bm25_vector
# ---------------------------------------------------------------------------

def load_enriched_terms(path: str) -> Iterator[Tuple[str, List[str]]]:
    """
    Yield (chunk_id, [term, ...]) from stage6_with_hypernyms.msgpack.
    Aggregates enriched_terms from all triplets per chunk.
    """
    with open(path, "rb") as fh:
        data = msgpack.unpack(fh, raw=False)

    for record in data:
        chunk_id = record.get("chunk_id") or record.get("id")
        if chunk_id is None:
            continue
        terms: List[str] = []
        for triplet in record.get("triplets", []):
            et = triplet.get("enriched_terms", [])
            if isinstance(et, list):
                terms.extend(str(t) for t in et if t)
        if terms:
            yield str(chunk_id), terms


def migrate_layer2(conn) -> None:
    print("\n" + "=" * 60)
    print("LAYER 2: layer2_triplet_bm25.triplet_bm25_vector -> sparsevec(16000)")
    print("=" * 60)

    cur = conn.cursor()

    # ---- schema: stat tables ------------------------------------------------
    print("Creating layer2 stat tables (bm25_l2_stats, bm25_l2_term_df)...")
    create_stat_tables(cur, "bm25_l2_stats", "bm25_l2_term_df")
    conn.commit()

    # ---- schema: triplet_bm25_vector column ---------------------------------
    print("Dropping old triplet_bm25_vector column and re-adding as sparsevec(16000)...")
    cur.execute("ALTER TABLE layer2_triplet_bm25 DROP COLUMN IF EXISTS triplet_bm25_vector")
    cur.execute("ALTER TABLE layer2_triplet_bm25 ADD COLUMN triplet_bm25_vector sparsevec(16000)")
    conn.commit()
    print("  Column dropped and re-added (old JSONB data cleared)")

    # ---- Load msgpack --------------------------------------------------------
    print(f"\nLoading {MSGPACK_PATH} ...")
    all_docs: List[Tuple[str, List[str]]] = list(load_enriched_terms(MSGPACK_PATH))
    print(f"  Loaded {len(all_docs):,} chunks with triplet terms")

    # ---- Pass 1: corpus stats -----------------------------------------------
    print("\nPass 1: computing corpus stats from enriched_terms...")
    bucket_df: Dict[int, int] = defaultdict(int)
    total_tokens = 0
    n_docs = 0

    for chunk_id, terms in all_docs:
        total_tokens += len(terms)
        n_docs        += 1
        seen_buckets: set = set()
        for term in terms:
            for tok in tokenize(term):
                b, _ = hash_token(tok)
                if b not in seen_buckets:
                    seen_buckets.add(b)
                    bucket_df[b] += 1

    print(f"  Stats: n_docs={n_docs:,}, total_tokens={total_tokens:,}, "
          f"unique_buckets={len(bucket_df):,}")
    avgdl = total_tokens / max(n_docs, 1)

    # ---- Write stats to DB --------------------------------------------------
    print("Writing layer2 stats to DB...")
    write_stats(cur, "bm25_l2_stats", "bm25_l2_term_df",
                n_docs, total_tokens, dict(bucket_df))
    conn.commit()
    print("  Stats committed")

    # ---- Pass 2: vectorize & write ------------------------------------------
    print("\nPass 2: vectorizing and writing sparsevec(16000) to layer2_triplet_bm25...")
    snapshot_df = dict(bucket_df)
    cur_write   = conn.cursor()
    batch_buf: List[Tuple] = []
    rows_written = 0

    for chunk_id, terms in all_docs:
        # Flatten enriched_terms -> tokenized strings
        tokens = []
        for term in terms:
            tokens.extend(tokenize(term))
        if not tokens:
            continue

        vec = compute_doc_vector(tokens, len(tokens), avgdl, n_docs, snapshot_df)
        batch_buf.append((vec, chunk_id))

        if len(batch_buf) >= BATCH:
            cur_write.executemany(
                "UPDATE layer2_triplet_bm25 SET triplet_bm25_vector = %s WHERE chunk_id = %s",
                batch_buf,
            )
            conn.commit()
            rows_written += len(batch_buf)
            print(f"  Written {rows_written:,}...")
            batch_buf = []

    if batch_buf:
        cur_write.executemany(
            "UPDATE layer2_triplet_bm25 SET triplet_bm25_vector = %s WHERE chunk_id = %s",
            batch_buf,
        )
        conn.commit()
        rows_written += len(batch_buf)

    print(f"  Vectorization complete: {rows_written:,} rows written")

    # ---- HNSW index -------------------------------------------------------
    print("\nCreating HNSW index on layer2_triplet_bm25.triplet_bm25_vector...")
    try:
        cur.execute("DROP INDEX IF EXISTS layer2_triplet_bm25_vector_idx")
        cur.execute("""
            CREATE INDEX layer2_triplet_bm25_vector_idx
            ON layer2_triplet_bm25 USING hnsw (triplet_bm25_vector sparsevec_ip_ops)
            WITH (m = 32, ef_construction = 100)
        """)
        conn.commit()
        print("  HNSW index created")
    except Exception as e:
        conn.rollback()
        print(f"  WARNING: Index creation skipped ({e})")
        print("  Data is written. Run CREATE INDEX manually after confirming pgvector version supports sparsevec HNSW.")
    # Refresh stats from the stored sparsevec (authoritative post-write pass)
    print("\nRefreshing stats from stored sparsevec...")
    refresh_layer2(conn)
    print("Layer 2 migration COMPLETE")


# ---------------------------------------------------------------------------
# Smoke test: verify counts + a quick query
# ---------------------------------------------------------------------------

def smoke_test(conn) -> None:
    print("\n" + "=" * 60)
    print("SMOKE TEST")
    print("=" * 60)
    cur = conn.cursor()

    # Layer 1 count
    cur.execute("SELECT COUNT(*) FROM arxiv_chunks WHERE bm25_sparse IS NOT NULL")
    l1_count = cur.fetchone()[0]
    print(f"Layer 1 rows with sparsevec: {l1_count:,}")

    # Layer 2 count
    cur.execute("SELECT COUNT(*) FROM layer2_triplet_bm25 WHERE triplet_bm25_vector IS NOT NULL")
    l2_count = cur.fetchone()[0]
    print(f"Layer 2 rows with sparsevec: {l2_count:,}")

    # Stat tables
    cur.execute("SELECT n_docs, total_tokens FROM bm25_global_stats WHERE id = 1")
    row = cur.fetchone()
    if row:
        print(f"L1 stats: n_docs={row[0]:,}, total_tokens={row[1]:,}, avgdl={row[1]/max(row[0],1):.1f}")

    cur.execute("SELECT n_docs, total_tokens FROM bm25_l2_stats WHERE id = 1")
    row = cur.fetchone()
    if row:
        print(f"L2 stats: n_docs={row[0]:,}, total_tokens={row[1]:,}, avgdl={row[1]/max(row[0],1):.1f}")

    # Quick inner-product test (build a query vector and retrieve top-5)
    print("\nQuick query test: 'contrastive learning'")
    from query_modules.pgvector_retriever import PGVectorRetriever, PGVectorConfig
    cfg = PGVectorConfig()
    with PGVectorRetriever(cfg) as r:
        l1 = r.query_layer1_bm25("contrastive learning self-supervised", top_k=3)
        l2 = r.query_layer2_triplet_bm25("contrastive learning self-supervised", top_k=3)
        print(f"  L1 BM25 top-3: {[(x['chunk_id'], round(x['score'], 4)) for x in l1]}")
        print(f"  L2 BM25 top-3: {[(x['chunk_id'], round(x['score'], 4)) for x in l2]}")

    print("Smoke test DONE")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate BM25 vectors to sparsevec(16000)")
    parser.add_argument("--layer1-only", action="store_true")
    parser.add_argument("--layer2-only", action="store_true")
    parser.add_argument("--smoke-test-only", action="store_true")
    args = parser.parse_args()

    run_l1 = not args.layer2_only and not args.smoke_test_only
    run_l2 = not args.layer1_only and not args.smoke_test_only

    print("Connecting to PostgreSQL...")
    conn = psycopg2.connect(**DB_CONFIG)
    register_vector(conn)
    print("  Connected")

    try:
        if run_l1:
            migrate_layer1(conn)
        if run_l2:
            migrate_layer2(conn)
        smoke_test(conn)
    except Exception as e:
        conn.rollback()
        print(f"\nERROR: {e}")
        raise
    finally:
        conn.close()
        print("\nConnection closed")


if __name__ == "__main__":
    main()
