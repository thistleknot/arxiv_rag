"""
Offline KG cache warmup — pre-extract KNWLER SPO triplets from utility strings.

Core Thesis:
    Stage 2 of the syllogism retriever uses KNWLER SPO triplets as structured
    semantic premises.  Those triplets must exist in graph/kg_cache/ before
    querying.  Run this script once (or resume it) to populate the cache for
    all papers in the CSV.

Workflow:
    1. Load arxiv_ids + utility strings from CSV
    2. Skip papers already cached (graph/kg_cache/<id>.json exists)
    3. For each uncached paper, call GraphRetriever._extract_paper_graph()
       (Ollama KNWLER extraction → disk cache write)
    4. Print progress + ETA

Usage:
    python warm_kg_cache.py                        # process all uncached papers
    python warm_kg_cache.py --limit 50             # process at most 50 papers
    python warm_kg_cache.py --start_from 1504.04788  # resume from a specific id
    python warm_kg_cache.py --dry_run              # show what would be processed

Notes:
    • Extraction takes ~165 s per paper (Ollama qwen3.5:2b with think blocks).
    • The script is safely restartable — cached papers are always skipped.
    • Run in a background terminal; the main retriever remains usable meanwhile.
"""

import argparse
import csv
import ast
import json
import sys
import time
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
_ROOT    = Path(__file__).resolve().parent.parent
_CSV     = _ROOT / "papers" / "post_processed" / "arxiv_data_with_analysis_cleaned.csv"
_KG_DIR  = _ROOT / "graph" / "kg_cache"
sys.path.insert(0, str(_ROOT))  # ensure arxiv_id_lists/ on path for graph.*


# ── Utility coercion (mirrors syllogism_retriever._coerce_utility) ────────────
def _coerce_utility(v: str) -> str:
    """Convert a stringified JSON list into plain text."""
    if v and v.startswith("["):
        try:
            items = ast.literal_eval(v)
            if isinstance(items, list):
                return ". ".join(str(x).strip().rstrip(".") for x in items if x) + "."
        except (ValueError, SyntaxError):
            pass
    return v


# ── CSV loader ────────────────────────────────────────────────────────────────
def _load_csv() -> list[dict]:
    rows = []
    with open(_CSV, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            aid     = row.get("arxiv_id", "").strip().strip('"')
            utility = row.get("utility", "").strip()
            if aid and utility:
                rows.append({"arxiv_id": aid, "utility": _coerce_utility(utility)})
    return rows


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="Pre-warm KG cache from utility strings")
    ap.add_argument("--limit",      type=int,  default=0,
                    help="Process at most N uncached papers (0 = all)")
    ap.add_argument("--start_from", type=str,  default="",
                    help="Skip papers before this arxiv_id in file order")
    ap.add_argument("--dry_run",    action="store_true",
                    help="Print plan only — do no extraction")
    args = ap.parse_args()

    _KG_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load CSV ──────────────────────────────────────────────────────────────
    print(f"Loading CSV: {_CSV}")
    all_papers = _load_csv()
    print(f"  {len(all_papers)} papers with utility strings")

    # ── Filter: skip already cached ──────────────────────────────────────────
    def _cache_path(arxiv_id: str) -> Path:
        return _KG_DIR / f"{arxiv_id.replace('/', '_')}.json"

    uncached = [p for p in all_papers if not _cache_path(p["arxiv_id"]).exists()]
    print(f"  {len(all_papers) - len(uncached)} already cached, "
          f"{len(uncached)} remaining")

    # ── start_from offset ────────────────────────────────────────────────────
    if args.start_from:
        ids = [p["arxiv_id"] for p in uncached]
        try:
            idx = ids.index(args.start_from)
            uncached = uncached[idx:]
            print(f"  Resuming from index {idx} (arxiv_id={args.start_from})")
        except ValueError:
            print(f"  Warning: start_from id '{args.start_from}' not found in uncached list")

    # ── optional limit ────────────────────────────────────────────────────────
    if args.limit > 0:
        uncached = uncached[:args.limit]

    if not uncached:
        print("Nothing to do — all papers are cached.")
        return

    eta_s = 165 * len(uncached)
    eta_h = eta_s / 3600
    print(f"\n  Will process: {len(uncached)} papers")
    print(f"  Estimated time: {eta_h:.1f} h  ({eta_s/60:.0f} min)  @ ~165 s/paper")

    if args.dry_run:
        print("\nDry run — first 10 papers that would be processed:")
        for p in uncached[:10]:
            print(f"  {p['arxiv_id']}  utility[:80]={p['utility'][:80]!r}")
        return

    # ── Import GraphRetriever (deferred so --dry_run works without Ollama) ───
    sys.path.insert(0, str(_ROOT))
    from graph.graph_retriever import GraphRetriever

    gr = GraphRetriever(
        ollama_host="http://127.0.0.1:11434",
        model="qwen3.5:2b",
        cache_dir=str(_KG_DIR),
        # embed_model defaults to all-MiniLM-L6-v2
    )

    # ── Extraction loop ───────────────────────────────────────────────────────
    n_total  = len(uncached)
    n_done   = 0
    n_errors = 0
    t_start  = time.time()

    print(f"\nStarting extraction at {time.strftime('%H:%M:%S')}")
    print("-" * 60)

    for paper in uncached:
        aid     = paper["arxiv_id"]
        utility = paper["utility"]
        t0      = time.time()

        try:
            pg = gr._extract_paper_graph(aid, utility)
            elapsed = time.time() - t0
            n_done += 1
            n_triples = len(pg.triplets)

            # ETA
            avg_s      = (time.time() - t_start) / n_done
            remaining  = n_total - n_done
            eta_remain = avg_s * remaining / 60

            print(
                f"  [{n_done:>5}/{n_total}] {aid:<20}  "
                f"{n_triples:>3} triplets  {elapsed:>5.0f}s  "
                f"ETA {eta_remain:.0f} min"
            )
        except Exception as exc:
            n_errors += 1
            print(f"  [{n_done:>5}/{n_total}] {aid:<20}  ERROR: {exc}")

    # ── Summary ───────────────────────────────────────────────────────────────
    total_elapsed = time.time() - t_start
    print("\n" + "=" * 60)
    print(f"Done.  Processed={n_done}  Errors={n_errors}  "
          f"Total time={total_elapsed/60:.1f} min")
    cached_now = sum(1 for p in all_papers if _cache_path(p["arxiv_id"]).exists())
    print(f"Cache coverage: {cached_now}/{len(all_papers)} papers  "
          f"({100*cached_now/len(all_papers):.1f}%)")


if __name__ == "__main__":
    main()
