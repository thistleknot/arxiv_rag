"""
Single entry point for the arXiv syllogism retrieval pipeline.

Core Thesis:
    Orchestrates two sequential stages in one command:
      1. KG cache warmup  — extract KNWLER SPO triplets for uncached papers
      2. Retrieval report — run the 9-stage syllogism retriever and write markdown

Usage:
    python run.py "your query here"
    python run.py "transformer attention mechanisms" --top_k 13 --output report.md
    python run.py "your query" --warmup_limit 100   # warm at most 100 papers first
    python run.py "your query" --skip_warmup         # skip warmup entirely

Flags:
    query           Retrieval query (required unless --dry_run is used)
    --top_k         Number of papers to return (default: 13)
    --n_papers      Candidate pool size — 0 = all (default: 0)
    --output        Path for the markdown report (default: _report.md)
    --warmup_limit  Max uncached papers to process before retrieval (default: 0 = all)
    --skip_warmup   Skip the KG cache warmup stage entirely
    --dry_run       Show warmup plan only; do not warm or retrieve
"""

import argparse
import csv
import ast
import json
import os
import sys
import time
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT   = Path(__file__).resolve().parent.parent
_CSV    = _ROOT / "papers" / "post_processed" / "arxiv_data_with_analysis_cleaned.csv"
_KG_DIR = _ROOT / "graph" / "kg_cache"
sys.path.insert(0, str(_ROOT))  # ensure arxiv_id_lists/ on path for graph.*, reasoning.*, arxiv_pipeline.*


# ── Helpers (mirrored from warm_kg_cache.py) ──────────────────────────────────
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


def _load_csv_rows() -> list:
    rows = []
    with open(_CSV, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            aid     = row.get("arxiv_id", "").strip().strip('"')
            utility = row.get("utility", "").strip()
            if aid and utility:
                rows.append({"arxiv_id": aid, "utility": _coerce_utility(utility)})
    return rows


def _cache_path(arxiv_id: str) -> Path:
    return _KG_DIR / f"{arxiv_id.replace('/', '_')}.json"


# ── Stage 1: KG cache warmup ──────────────────────────────────────────────────
def run_warmup(limit: int = 0, dry_run: bool = False) -> None:
    """Pre-extract KNWLER triplets for all uncached papers."""
    _KG_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("STAGE 1 — KG cache warmup")
    print("=" * 60)
    print(f"Loading CSV: {_CSV}")
    all_papers = _load_csv_rows()
    print(f"  {len(all_papers)} papers with utility strings")

    uncached = [p for p in all_papers if not _cache_path(p["arxiv_id"]).exists()]
    print(f"  {len(all_papers) - len(uncached)} already cached, "
          f"{len(uncached)} remaining")

    if limit > 0:
        uncached = uncached[:limit]

    if not uncached:
        print("  Nothing to do — all papers are cached.\n")
        return

    eta_s = 165 * len(uncached)
    print(f"\n  Will process: {len(uncached)} papers")
    print(f"  Estimated time: {eta_s/3600:.1f} h  ({eta_s/60:.0f} min)  @ ~165 s/paper")

    if dry_run:
        print("\n  Dry run — first 10 papers that would be processed:")
        for p in uncached[:10]:
            print(f"    {p['arxiv_id']}  utility[:80]={p['utility'][:80]!r}")
        return

    from graph.graph_retriever import GraphRetriever

    gr = GraphRetriever(
        ollama_host="http://127.0.0.1:11434",
        model="qwen3.5:2b",
        cache_dir=str(_KG_DIR),
    )

    n_total  = len(uncached)
    n_done   = 0
    n_errors = 0
    t_start  = time.time()

    print(f"\n  Starting at {time.strftime('%H:%M:%S')}")
    print("  " + "-" * 56)

    for paper in uncached:
        aid     = paper["arxiv_id"]
        utility = paper["utility"]
        t0      = time.time()
        try:
            pg = gr._extract_paper_graph(aid, utility)
            elapsed = time.time() - t0
            n_done += 1
            avg_s   = (time.time() - t_start) / n_done
            remain  = avg_s * (n_total - n_done) / 60
            print(
                f"  [{n_done:>5}/{n_total}] {aid:<20}  "
                f"{len(pg.triplets):>3} triplets  {elapsed:>5.0f}s  "
                f"ETA {remain:.0f} min"
            )
        except Exception as exc:
            n_errors += 1
            print(f"  [{n_done:>5}/{n_total}] {aid:<20}  ERROR: {exc}")

    total_elapsed = time.time() - t_start
    cached_now = sum(1 for p in all_papers if _cache_path(p["arxiv_id"]).exists())
    print(f"\n  Done.  Processed={n_done}  Errors={n_errors}  "
          f"Time={total_elapsed/60:.1f} min")
    print(f"  Cache coverage: {cached_now}/{len(all_papers)} papers "
          f"({100*cached_now/len(all_papers):.1f}%)\n")


# ── Stage 2: Retrieval report ─────────────────────────────────────────────────
def run_retrieval(query: str, n_papers: int, top_k: int, output: str) -> None:
    """Run the 9-stage syllogism retriever and write the markdown report."""
    print("\n" + "=" * 60)
    print("STAGE 2 — Syllogism retrieval")
    print("=" * 60)
    print(f"  Query    : {query}")
    print(f"  n_papers : {'all' if n_papers == 0 else n_papers}")
    print(f"  top_k    : {top_k}")
    print(f"  output   : {output}\n")

    from arxiv_pipeline.syllogism_retriever import SyllogismRetriever

    retriever = SyllogismRetriever()
    result    = retriever.retrieve(query, n_papers=n_papers, top_k=top_k)

    md = result.to_markdown()
    out_path = Path(output)
    out_path.write_text(md, encoding="utf-8")
    print(f"\n  Report written → {out_path.resolve()}")
    print(f"  Papers retrieved: {len(result.papers)}")


# ── Entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(
        description="arXiv syllogism pipeline — warm cache then retrieve",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("query",          nargs="?",  default="",
                    help="Retrieval query (required unless --dry_run)")
    ap.add_argument("--top_k",        type=int,   default=13,
                    help="Papers to return (default: 13)")
    ap.add_argument("--n_papers",     type=int,   default=0,
                    help="Candidate pool size; 0 = all (default: 0)")
    ap.add_argument("--output",       type=str,   default="_report.md",
                    help="Output markdown file (default: _report.md)")
    ap.add_argument("--warmup_limit", type=int,   default=0,
                    help="Max uncached papers to warm before retrieval (0 = all)")
    ap.add_argument("--warmup",       action="store_true",
                    help="Run KG cache warmup before retrieval (optional, slow)")
    ap.add_argument("--dry_run",      action="store_true",
                    help="Show warmup plan only; do not warm or retrieve")
    args = ap.parse_args()

    if args.dry_run:
        run_warmup(limit=args.warmup_limit, dry_run=True)
        return

    if not args.query:
        try:
            args.query = input("Query: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit(0)
        if not args.query:
            ap.error("query is required (unless --dry_run is used)")

    if args.warmup:
        run_warmup(limit=args.warmup_limit, dry_run=False)

    run_retrieval(
        query    = args.query,
        n_papers = args.n_papers,
        top_k    = args.top_k,
        output   = args.output,
    )


if __name__ == "__main__":
    main()
