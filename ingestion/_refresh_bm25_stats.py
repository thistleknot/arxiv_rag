"""
_refresh_bm25_stats.py
======================
Derive BM25 global stats from stored sparsevec(16000) columns.

After ingestion the stored vectors are the source of truth. This script
scans them to re-derive:
    n_docs      = COUNT of non-null rows
    bucket_df   = per-bucket document frequency (rows with nonzero at index b)
    total_nnz   = sum of NNZ per doc  (proxy for raw token count; avgdl approx)

Stats are upserted into the appropriate stat tables:
    Layer 1: bm25_global_stats  / bm25_term_global
    Layer 2: bm25_l2_stats      / bm25_l2_term_df

Usage:
    c:\\users\\user\\py310\\scripts\\python.exe _refresh_bm25_stats.py          # both
    c:\\users\\user\\py310\\scripts\\python.exe _refresh_bm25_stats.py --layer1
    c:\\users\\user\\py310\\scripts\\python.exe _refresh_bm25_stats.py --layer2
"""

import argparse
from collections import defaultdict
from typing import Dict, Tuple

import psycopg2
from pgvector.psycopg2 import register_vector

# ---------------------------------------------------------------------------
# DB config (identical to _migrate_bm25_sparsevec.py)
# ---------------------------------------------------------------------------

DB_CONFIG = dict(
    host="localhost",
    port=5432,
    dbname="langchain",
    user="langchain",
    password="langchain",
)

BATCH = 2_000   # rows per server-side cursor fetch


# ---------------------------------------------------------------------------
# Stat table helpers (mirrors _migrate_bm25_sparsevec.write_stats)
# ---------------------------------------------------------------------------

def _ensure_stat_tables(cur, global_table: str, term_table: str) -> None:
    """CREATE IF NOT EXISTS for both stat tables."""
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


def _upsert_stats(
    cur,
    global_table: str,
    term_table: str,
    n_docs: int,
    total_nnz: int,
    bucket_df: Dict[int, int],
) -> None:
    """Upsert stat tables from scanned values."""
    # single-row global stats
    cur.execute(f"""
        INSERT INTO {global_table} (id, n_docs, total_tokens)
        VALUES (1, %s, %s)
        ON CONFLICT (id) DO UPDATE
            SET n_docs       = EXCLUDED.n_docs,
                total_tokens = EXCLUDED.total_tokens
    """, (n_docs, total_nnz))

    # per-bucket doc_freq — batch in 5k chunks
    rows = list(bucket_df.items())
    for start in range(0, len(rows), 5_000):
        chunk = rows[start : start + 5_000]
        args  = [(b, df) for b, df in chunk]
        cur.execute(f"""
            INSERT INTO {term_table} (bucket, doc_freq)
            SELECT v.bucket, v.doc_freq
            FROM (VALUES {','.join(['(%s,%s)'] * len(args))}) AS v(bucket, doc_freq)
            ON CONFLICT (bucket) DO UPDATE
                SET doc_freq = EXCLUDED.doc_freq
        """, [x for pair in args for x in pair])


# ---------------------------------------------------------------------------
# Core scan function (layer-agnostic)
# ---------------------------------------------------------------------------

def refresh_layer_stats(
    conn,
    global_table: str,
    term_table: str,
    src_table: str,
    src_col: str,
    id_col: str = "chunk_id",
    batch: int = BATCH,
) -> Tuple[int, int, Dict[int, int]]:
    """
    Stream stored sparsevec from src_table.src_col and recompute stats.

    Tallies .indices on each SparseVector to build:
        n_docs    = number of non-null rows
        bucket_df = {bucket_index: doc_count}
        total_nnz = sum of unique-bucket hits per doc (avgdl proxy)

    Upserts results into global_table and term_table.
    Returns (n_docs, total_nnz, bucket_df).
    """
    print(f"\n{'=' * 60}")
    print(f"BM25 Stats Refresh: {global_table} / {term_table}")
    print(f"  Source : {src_table}.{src_col}")
    print(f"{'=' * 60}")

    bucket_df: Dict[int, int] = defaultdict(int)
    n_docs    = 0
    total_nnz = 0

    # Use a named (server-side) cursor for streaming — avoids loading all rows
    with conn.cursor("_refresh_stream") as cur_r:
        cur_r.itersize = batch
        cur_r.execute(f"""
            SELECT {id_col}, {src_col}
            FROM {src_table}
            WHERE {src_col} IS NOT NULL
            ORDER BY {id_col}
        """)

        while True:
            rows = cur_r.fetchmany(batch)
            if not rows:
                break
            for _row_id, sv in rows:
                if sv is None:
                    continue
                indices = sv.indices()  # method call — returns array of bucket positions
                n_docs    += 1
                total_nnz += len(indices)
                for b in indices:
                    bucket_df[b] += 1
            if n_docs % 20_000 == 0:
                print(f"  ... scanned {n_docs:,} docs")

    print(f"  Scan done  : n_docs={n_docs:,}, total_nnz={total_nnz:,}, "
          f"unique_buckets={len(bucket_df):,}")
    if n_docs:
        print(f"  Avg NNZ    : {total_nnz / n_docs:.1f} buckets/doc")

    # upsert
    cur_w = conn.cursor()
    _ensure_stat_tables(cur_w, global_table, term_table)
    _upsert_stats(cur_w, global_table, term_table, n_docs, total_nnz, dict(bucket_df))
    conn.commit()
    print(f"  Committed  : {global_table} + {term_table}")

    return n_docs, total_nnz, dict(bucket_df)


# ---------------------------------------------------------------------------
# Per-layer wrappers
# ---------------------------------------------------------------------------

def refresh_layer1(conn) -> None:
    refresh_layer_stats(
        conn,
        global_table="bm25_global_stats",
        term_table="bm25_term_global",
        src_table="arxiv_chunks",
        src_col="bm25_sparse",
        id_col="chunk_id",
    )


def refresh_layer2(conn) -> None:
    refresh_layer_stats(
        conn,
        global_table="bm25_l2_stats",
        term_table="bm25_l2_term_df",
        src_table="layer2_triplet_bm25",
        src_col="triplet_bm25_vector",
        id_col="chunk_id",
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refresh BM25 stat tables from stored sparsevec columns."
    )
    parser.add_argument("--layer1", action="store_true", help="Layer 1 only")
    parser.add_argument("--layer2", action="store_true", help="Layer 2 only")
    args = parser.parse_args()

    # default: both layers
    run_l1 = args.layer1 or (not args.layer1 and not args.layer2)
    run_l2 = args.layer2 or (not args.layer1 and not args.layer2)

    print("Connecting to PostgreSQL...")
    conn = psycopg2.connect(**DB_CONFIG)
    register_vector(conn)
    print("  Connected")

    try:
        if run_l1:
            refresh_layer1(conn)
        if run_l2:
            refresh_layer2(conn)
        print("\nAll requested layers refreshed.")
    except Exception as e:
        conn.rollback()
        print(f"\nERROR: {e}")
        raise
    finally:
        conn.close()
        print("Connection closed")


if __name__ == "__main__":
    main()
