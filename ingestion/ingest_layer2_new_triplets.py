"""
Ingest Layer 2 Triplet BM25 as sparsevec(16000) with signed feature hashing.

Input:  triplet_checkpoints_full/stage6_with_hypernyms.msgpack
        format: [{chunk_id, text, triplets: [{enriched_terms: [str,...]},...]}]
Output: layer2_triplet_bm25.triplet_bm25_vector = sparsevec(16000)
Stats:  bm25_l2_stats (n_docs, total_tokens) + bm25_l2_term_df (bucket, doc_freq)

Hashing: mmh3 signed trick (Weinberger et al. 2009)
  bucket = mmh3.hash(term, seed=0, signed=False) % 16_000
  sign   = +1 if mmh3.hash(term, seed=1, signed=True) >= 0 else -1
  Same hash_token as retrieval/pgvector_retriever.py -- query/ingest are aligned.
"""

import math
import mmh3
import msgpack
import psycopg2
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from pgvector.psycopg2 import register_vector
from pgvector.psycopg2 import SparseVector
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

N_BUCKETS    = 16_000
BM25_K1      = 1.5
BM25_B       = 0.75
BATCH        = 2_000
MSGPACK_PATH = Path("triplet_checkpoints_full/stage6_with_hypernyms.msgpack")

DB_CONFIG = dict(
    host="localhost", port=5432,
    dbname="langchain", user="langchain", password="langchain",
)

# ---------------------------------------------------------------------------
# Signed hashing -- MUST match retrieval/pgvector_retriever.py hash_token()
# ---------------------------------------------------------------------------

def hash_token(term: str) -> Tuple[int, float]:
    bucket = mmh3.hash(term, seed=0, signed=False) % N_BUCKETS
    sign   = 1.0 if mmh3.hash(term, seed=1, signed=True) >= 0 else -1.0
    return bucket, sign


# ---------------------------------------------------------------------------
# Stat tables
# ---------------------------------------------------------------------------

def create_stat_tables(cur) -> None:
    cur.execute("""
        CREATE TABLE IF NOT EXISTS bm25_l2_stats (
            id           SMALLINT PRIMARY KEY DEFAULT 1,
            n_docs       BIGINT   NOT NULL DEFAULT 0,
            total_tokens BIGINT   NOT NULL DEFAULT 0
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS bm25_l2_term_df (
            bucket   INT    PRIMARY KEY,
            doc_freq BIGINT NOT NULL DEFAULT 0
        )
    """)


def write_stats(cur, n_docs: int, total_tokens: int, bucket_df: Dict[int, int]) -> None:
    cur.execute("""
        INSERT INTO bm25_l2_stats (id, n_docs, total_tokens)
        VALUES (1, %s, %s)
        ON CONFLICT (id) DO UPDATE
            SET n_docs = EXCLUDED.n_docs,
                total_tokens = EXCLUDED.total_tokens
    """, (n_docs, total_tokens))

    rows = list(bucket_df.items())
    for start in range(0, len(rows), 5_000):
        chunk = rows[start:start + 5_000]
        args  = [(b, df) for b, df in chunk]
        placeholders = ",".join(["(%s,%s)"] * len(args))
        cur.execute("""
            INSERT INTO bm25_l2_term_df (bucket, doc_freq)
            SELECT v.bucket, v.doc_freq
            FROM (VALUES {ph}) AS v(bucket, doc_freq)
            ON CONFLICT (bucket) DO UPDATE
                SET doc_freq = EXCLUDED.doc_freq
        """.format(ph=placeholders),
            [x for pair in args for x in pair],
        )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_stage6() -> List[Tuple[str, List[str]]]:
    """Load stage6 msgpack; return list of (chunk_id_str, [enriched_term,...])."""
    path = MSGPACK_PATH
    print(f"Loading {path}  ({path.stat().st_size / 1e6:.1f} MB)...")
    with open(path, "rb") as fh:
        data = msgpack.unpack(fh, raw=False)

    result = []
    skipped = 0
    for record in data:
        chunk_id = record.get("chunk_id") or record.get("id")
        if chunk_id is None:
            skipped += 1
            continue
        terms: List[str] = []
        for triplet in record.get("triplets", []):
            et = triplet.get("enriched_terms", [])
            if isinstance(et, list):
                terms.extend(str(t) for t in et if t)
        if terms:
            result.append((str(chunk_id), terms))
        else:
            skipped += 1

    print(f"  Loaded {len(result):,} chunks with terms  ({skipped:,} empty/skipped)")
    return result


# ---------------------------------------------------------------------------
# BM25 vectorization (signed hashing)
# ---------------------------------------------------------------------------

def compute_doc_vector(
    terms: List[str],
    dl: int,
    avgdl: float,
    n: int,
    bucket_df: Dict[int, int],
) -> SparseVector:
    """Okapi BM25 sparsevec with signed hashing (Weinberger et al. 2009)."""
    bucket_raw: Dict[int, float] = defaultdict(float)
    for term in terms:
        b, sign = hash_token(term)
        bucket_raw[b] += sign

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
# Ingestion
# ---------------------------------------------------------------------------

def ingest(all_docs: List[Tuple[str, List[str]]]) -> None:
    print("\nConnecting to PostgreSQL...")
    conn = psycopg2.connect(**DB_CONFIG)
    register_vector(conn)
    cur = conn.cursor()
    print("  Connected")

    # ---- stat tables -------------------------------------------------------
    print("Creating stat tables (bm25_l2_stats, bm25_l2_term_df)...")
    create_stat_tables(cur)
    conn.commit()

    # ---- Pass 1: corpus stats ----------------------------------------------
    print("\nPass 1: computing corpus stats from enriched_terms...")
    bucket_df: Dict[int, int] = defaultdict(int)
    total_tokens = 0
    n_docs = 0

    for _, terms in tqdm(all_docs, desc="  Stats pass"):
        total_tokens += len(terms)
        n_docs       += 1
        seen: set = set()
        for term in terms:
            b, _ = hash_token(term)
            if b not in seen:
                seen.add(b)
                bucket_df[b] += 1

    avgdl = total_tokens / max(n_docs, 1)
    print(f"  Stats: n_docs={n_docs:,}, total_tokens={total_tokens:,}, "
          f"unique_buckets={len(bucket_df):,}, avgdl={avgdl:.1f}")

    print("Writing stats to bm25_l2_stats + bm25_l2_term_df...")
    write_stats(cur, n_docs, total_tokens, dict(bucket_df))
    conn.commit()
    print("  Stats committed")

    # ---- Truncate stale data -----------------------------------------------
    print("\nTruncating stale layer2_triplet_bm25 rows...")
    cur.execute("DELETE FROM layer2_triplet_bm25")
    conn.commit()
    print("  Old rows removed")

    # ---- Pass 2: vectorize & insert ----------------------------------------
    print(f"\nPass 2: vectorizing + inserting {len(all_docs):,} rows (batch={BATCH})...")
    snapshot_df = dict(bucket_df)
    batch_buf: List[Tuple] = []
    rows_written = 0

    INSERT_SQL = """
        INSERT INTO layer2_triplet_bm25 (chunk_id, triplet_bm25_vector)
        VALUES (%s, %s)
        ON CONFLICT (chunk_id) DO UPDATE
            SET triplet_bm25_vector = EXCLUDED.triplet_bm25_vector,
                created_at = now()
    """

    for chunk_id, terms in tqdm(all_docs, desc="  Vectorizing"):
        if not terms:
            continue
        vec = compute_doc_vector(terms, len(terms), avgdl, n_docs, snapshot_df)
        batch_buf.append((chunk_id, vec))

        if len(batch_buf) >= BATCH:
            cur.executemany(INSERT_SQL, batch_buf)
            conn.commit()
            rows_written += len(batch_buf)
            batch_buf = []

    if batch_buf:
        cur.executemany(INSERT_SQL, batch_buf)
        conn.commit()
        rows_written += len(batch_buf)

    print(f"  Inserted {rows_written:,} rows")
    # Note: no HNSW index — BM25 sparsevec uses sequential scan (<#> inner product).
    # HNSW is for dense ANN; sparse BM25 at 118k rows is fast exact via seq scan.

    cur.close()
    conn.close()


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify() -> None:
    conn = psycopg2.connect(**DB_CONFIG)
    register_vector(conn)
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*), MAX(created_at) FROM layer2_triplet_bm25")
    cnt, ts = cur.fetchone()

    cur.execute("""
        SELECT COUNT(*) FROM layer2_triplet_bm25
        WHERE triplet_bm25_vector IS NOT NULL
    """)
    non_null = cur.fetchone()[0]

    cur.execute("SELECT n_docs, total_tokens FROM bm25_l2_stats WHERE id = 1")
    stats = cur.fetchone()

    cur.execute("SELECT COUNT(*) FROM bm25_l2_term_df")
    n_buckets = cur.fetchone()[0]

    cur.execute("SELECT pg_size_pretty(pg_total_relation_size('layer2_triplet_bm25'))")
    size = cur.fetchone()[0]

    cur.close()
    conn.close()

    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    print(f"Rows: {cnt:,}   Non-null sparsevec: {non_null:,}")
    print(f"Max created_at: {ts}")
    print(f"Table size: {size}")
    if stats:
        print(f"bm25_l2_stats: n_docs={stats[0]:,}, total_tokens={stats[1]:,}")
    print(f"bm25_l2_term_df buckets: {n_buckets:,}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("LAYER 2 TRIPLET BM25 INGESTION -- sparsevec(16000) signed-hash")
    print("=" * 60)

    all_docs = load_stage6()
    ingest(all_docs)
    verify()

    print("\nLAYER 2 INGESTION COMPLETE")


if __name__ == "__main__":
    main()