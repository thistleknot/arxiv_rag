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

    # Combine per-query reports into a single final _report.md:
    python run.py --combine _report_asd.md _report_ags.md _report_dsf.md

Flags:
    query           Retrieval query (required unless --dry_run or --combine)
    --top_k         Number of papers to return (default: 13)
    --n_papers      Standalone fallback candidate pool size (default: 13).
                    Only used when the 3-layer pgvector retriever is unavailable.
    --output        Path for the markdown report (default: _report.md)
    --combine       Combine Synthesis sections from listed report files into --output
    --warmup_limit  Max uncached papers to process before retrieval (default: 0 = all)
    --skip_warmup   Skip the KG cache warmup stage entirely
    --extract       Run tiered on-demand methods extraction after retrieval:
                      • top-3 cached  → eager full-pipeline on top 5
                      • some missing  → eager full-pipeline on top 3
                      • floor: always attempt ≥2 fully-enriched results
                      • remainder of top_k: text-only Phase-5 extraction
                        (injected into report, NOT written to disk)
"""

import argparse
import csv
import ast
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT        = Path(__file__).resolve().parent.parent
_CSV         = _ROOT / "papers" / "post_processed" / "arxiv_data_with_analysis_cleaned.csv"
_KG_DIR      = _ROOT / "graph" / "kg_cache"
_POSTPROC    = _ROOT / "papers" / "post_processed"
_PAPERS_DIR  = _ROOT / "papers"
_PIPELINE_BAT = _ROOT / "run_pipeline.bat"

EXTRACT_EAGER_BASE     = 3  # full-pipeline papers when any top-3 are missing
EXTRACT_EAGER_EXTENDED = 5  # full-pipeline papers when all top-3 already done
EXTRACT_MIN_FULL       = 2  # minimum papers with disk-persisted full methods

sys.path.insert(0, str(_ROOT))  # ensure arxiv_id_lists/ on path for graph.*, reasoning.*, arxiv_pipeline.*

from constants import AGENT_TOP_K  # noqa: E402  (import after sys.path fixup)


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
        cache_dir=_KG_DIR,
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


# ── Report combiner ───────────────────────────────────────────────────────────
def _extract_query(text: str) -> str:
    """Pull the query string from a report's header line."""
    for line in text.splitlines():
        if line.startswith("> **Query**:"):
            return line.replace("> **Query**:", "").strip()
    return "(unknown query)"


def _extract_section(text: str, heading: str) -> str:
    """Extract content between a ## heading and the next ## heading (or EOF)."""
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.strip() == heading:
            start = i + 1
            break
    if start is None:
        return ""
    out = []
    for line in lines[start:]:
        if line.startswith("## "):
            break
        out.append(line)
    return "\n".join(out).strip()


def combine_reports(input_paths: list, output: str) -> None:
    """Concatenate Synthesis sections from multiple per-query reports into one file."""
    parts = []
    for p in input_paths:
        path = Path(p)
        if not path.exists():
            print(f"  [combine] Skipping missing file: {p}")
            continue
        text  = path.read_text(encoding="utf-8")
        query = _extract_query(text)
        synth = _extract_section(text, "## Synthesis")
        if not synth:
            print(f"  [combine] No ## Synthesis section found in {p} — skipping")
            continue
        block = f"# Query: {query}\n\n## Synthesis\n\n{synth}"
        parts.append(block)
        print(f"  [combine] {path.name}  →  query: {query!r}")

    if not parts:
        print("  [combine] Nothing to combine — no valid input files.")
        return

    combined = "\n\n---\n\n".join(parts)
    out_path = Path(output)
    out_path.write_text(combined, encoding="utf-8")
    print(f"\n  Combined report written → {out_path.resolve()}")
    print(f"  Sections combined: {len(parts)}")



def _pgvec_id_to_csv_id(pid: str) -> str:
    """Normalize a pgvector paper_id to the CSV arxiv_id format.

    pgvector stores IDs as YYMM_NNNNN (underscore separator, optional vN suffix).
    The utility CSV uses YYMM.NNNNN (dot separator, no version suffix).

    Examples:
        "2502_12110"    → "2502.12110"
        "2403_19889v1"  → "2403.19889"
        "ISLP"          → "ISLP"  (non-standard IDs pass through unchanged)
    """
    pid = re.sub(r'v\d+$', '', pid)                     # strip version suffix
    pid = re.sub(r'^(\d{4})_(\d)', r'\1.\2', pid)       # YYMM_N → YYMM.N
    return pid


def _norm_to_underscore(pid: str) -> str:
    """Normalize arxiv_id to filesystem underscore form (YYMM_NNNNN)."""
    pid = re.sub(r'v\d+$', '', pid)
    pid = re.sub(r'^(\d{4})\.(\d)', r'\1_\2', pid)
    return pid


def _extract_methods_text_only(paper_id: str) -> str | None:
    """Run Phase 5 extraction inline without writing _methods.md to disk.

    Used for papers outside the eager-extraction tier. Produces a 'bastardized'
    methods extraction: text body only, no VLM image descriptions. The result is
    injected into the report but never persisted so it cannot be mistaken for a
    fully-enriched (Phase 3+4+5) extraction.

    Require: .md exists in post_processed for paper_id.
    Guarantee: _methods.md is never created; returns content string or None.
    Failure modes: missing .md, proxy unreachable, empty response → returns None.
    """
    uid = _norm_to_underscore(paper_id)
    md_path = _POSTPROC / f"{uid}.md"
    if not md_path.is_file():
        return None
    try:
        from extract_methods import (  # noqa: PLC0415
            call_proxy, load_system_prompt, strip_post_references,
        )
    except ImportError as exc:
        print(f"  [extract] cannot import extract_methods: {exc}")
        return None
    try:
        system_prompt = load_system_prompt()
        body, _ = strip_post_references(md_path.read_text(encoding="utf-8"))
        content = call_proxy(body, system_prompt, "gpt-4.1", "localhost", 8069)
        if not content.strip():
            return None
        return (
            "> ⚠️ **Text-only extraction** — image descriptions not yet available "
            "for this paper.\n\n" + content
        )
    except Exception as exc:
        print(f"  [extract] text-only failed for {uid}: {exc}")
        return None


def _ensure_methods(paper_id: str) -> tuple[str | None, str]:
    """Return (_methods.md content, source_label) for paper_id, running extraction if needed.

    _methods.md is the canonical signal that the paper has been through the full
    new pipeline (docling → Phase 2 base64 strip → Phase 3 VLM descriptions →
    Phase 4 reinsert → Phase 5 methods). Its absence always triggers a full re-run,
    even if a .md exists from an older pipeline run without image descriptions.

    Resolution order:
      1. _methods.md in post_processed → "cached"
      2. .pdf in papers/             → copy to post_processed, run full pipeline → "full_pipeline"
      3. download from arxiv.org     → run full pipeline → "downloaded"

    Require: paper_id is a valid arxiv ID in CSV (dot) or path (underscore) form.
    Guarantee: if returned content is not None, _methods.md exists in post_processed.
    Failure modes: subprocess errors, network errors — logged, returns (None, "error").
    """
    uid = _norm_to_underscore(paper_id)
    dot_id = _pgvec_id_to_csv_id(uid)

    methods_path = _POSTPROC / f"{uid}_methods.md"

    # ── Level 1: cache hit ────────────────────────────────────────────────────
    if methods_path.exists():
        return methods_path.read_text(encoding="utf-8"), "cached"

    # ── Level 2 & 3: full pipeline needed (PDF source → post_processed) ──────
    # Prefer existing local PDF; fall back to arxiv download.
    pdf_src = _PAPERS_DIR / f"{uid}.pdf"
    pdf_dest = _POSTPROC / f"{uid}.pdf"

    def _run_pipeline_on_pdf(label: str) -> tuple[str | None, str]:
        try:
            result = subprocess.run(
                [str(_PIPELINE_BAT), str(pdf_dest)],
                capture_output=True, text=True, timeout=600, shell=True,
            )
            pdf_dest.unlink(missing_ok=True)
            if result.returncode == 0 and methods_path.exists():
                return methods_path.read_text(encoding="utf-8"), label
            print(f"  [extract] Pipeline failed (rc={result.returncode}): {result.stderr[:300]}")
        except subprocess.TimeoutExpired:
            pdf_dest.unlink(missing_ok=True)
            print(f"  [extract] Pipeline timed out for {uid}")
        except Exception as exc:
            pdf_dest.unlink(missing_ok=True)
            print(f"  [extract] Pipeline error for {uid}: {exc}")
        return None, "error"

    if pdf_src.exists():
        print(f"  [extract] {uid}: copying PDF → running full pipeline...")
        try:
            import shutil
            shutil.copy2(pdf_src, pdf_dest)
        except Exception as exc:
            print(f"  [extract] PDF copy error for {uid}: {exc}")
            return None, "error"
        return _run_pipeline_on_pdf("full_pipeline")

    arxiv_url = f"https://arxiv.org/pdf/{dot_id}.pdf"
    print(f"  [extract] {uid}: downloading PDF from {arxiv_url}...")
    try:
        _POSTPROC.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(arxiv_url, pdf_dest)
    except Exception as exc:
        pdf_dest.unlink(missing_ok=True)
        print(f"  [extract] Download error for {uid}: {exc}")
        return None, "error"
    return _run_pipeline_on_pdf("downloaded")


def _run_on_demand_extraction(result) -> None:
    """Populate result.methods_content using a tiered extraction strategy.

    Tier 1 — Eager full-pipeline (writes _methods.md to disk):
      - Counts how many of the top-3 papers already have _methods.md.
      - If all 3 are cached: extends eager processing to EXTRACT_EAGER_EXTENDED (5).
      - Otherwise:          eager window is EXTRACT_EAGER_BASE (3).
      - After the eager pass, if <EXTRACT_MIN_FULL full results were obtained,
        continues trying papers beyond the eager window until the floor is met.

    Tier 2 — Text-only bastardized extraction (never writes to disk):
      - Every remaining top-k paper that didn't receive Tier 1 treatment.
      - Runs Phase 5 directly on the existing .md (no VLM image descriptions).
      - Output is injected into the report labelled as text-only.

    Modifies result in-place.
    Require: result.papers is a sorted list of RetrievedDoc with metadata["paper_id"].
    Guarantee: result.methods_content[pid] set for each paper where any extraction
               succeeded; _methods.md is only written for Tier 1 papers.
    """
    papers = result.papers
    if not papers:
        return

    # ── Determine eager-extraction limit ──────────────────────────────────────
    n_top3 = min(3, len(papers))
    top3_done = sum(
        1 for doc in papers[:n_top3]
        if (_POSTPROC / f"{_norm_to_underscore(doc.metadata.get('paper_id', doc.doc_id))}_methods.md").exists()
    )
    eager_limit = min(
        EXTRACT_EAGER_EXTENDED if top3_done == n_top3 else EXTRACT_EAGER_BASE,
        len(papers),
    )
    print(
        f"\n  [extract] Tier config: top3_cached={top3_done}/{n_top3}  "
        f"eager_limit={eager_limit}  min_full={EXTRACT_MIN_FULL}"
    )

    t0 = time.time()

    # ── Tier 1a: Eager full-pipeline ──────────────────────────────────────────
    n_full = 0
    for doc in papers[:eager_limit]:
        pid = doc.metadata.get("paper_id", doc.doc_id)
        content, source = _ensure_methods(pid)
        if content:
            result.methods_content[pid] = content
            n_full += 1
            print(f"  [extract] {pid}: ✓ full ({source})")
        else:
            print(f"  [extract] {pid}: ✗ full pipeline failed")

    # ── Tier 1b: Min-full guarantee — extend past eager_limit if needed ───────
    if n_full < EXTRACT_MIN_FULL:
        for doc in papers[eager_limit:]:
            if n_full >= EXTRACT_MIN_FULL:
                break
            pid = doc.metadata.get("paper_id", doc.doc_id)
            if pid in result.methods_content:
                continue
            content, source = _ensure_methods(pid)
            if content:
                result.methods_content[pid] = content
                n_full += 1
                print(f"  [extract] {pid}: ✓ full ({source}, min-full pass)")
            else:
                print(f"  [extract] {pid}: ✗ full pipeline failed (min-full pass)")

    # ── Tier 2: Text-only bastardized extraction for remainder ────────────────
    n_bastard = 0
    for doc in papers:
        pid = doc.metadata.get("paper_id", doc.doc_id)
        if pid in result.methods_content:
            continue
        content = _extract_methods_text_only(pid)
        if content:
            result.methods_content[pid] = content
            n_bastard += 1
            print(f"  [extract] {pid}: ~ text-only (not persisted)")

    elapsed = time.time() - t0
    print(
        f"\n  [extract] Done: {n_full} full + {n_bastard} text-only "
        f"({len(result.methods_content)} total) in {elapsed:.0f}s\n"
    )


def _try_pgvector_retrieval(query: str, top_k: int):
    """Attempt to retrieve top_k papers via the 3-layer φ-scaled pgvector pipeline.

    top_k here is the candidate pool size for the pgvector retriever — intentionally
    larger than the final AGENT_TOP_K to give the syllogism pipeline enough candidates
    to filter from.

    Returns list of (arxiv_id, score) tuples if successful, None if unavailable.
    Only falls back on availability errors; unexpected exceptions are re-raised.
    """
    try:
        sys.path.insert(0, str(_ROOT / "retrieval"))
        from pgvector_retriever import PGVectorConfig  # type: ignore
        from arxiv_retriever import ArxivRetriever     # type: ignore
        config = PGVectorConfig(
            db_host="localhost",
            db_port=5432,
            db_name="langchain",
            db_user="langchain",
            db_password="langchain",
            table_name="arxiv_chunks",
        )
        retriever = ArxivRetriever(config)
        results = retriever.search(query, top_k=top_k)
        return [
            (_pgvec_id_to_csv_id(r.doc_id), float(getattr(r, "final_score", None) or 0.5))
            for r in results
        ]
    except (ImportError, ModuleNotFoundError) as e:
        print(f"  [L1-L3] Modules unavailable — using standalone fallback: {e}")
        return None
    except Exception as e:
        err_type = type(e).__name__
        _availability = {"OperationalError", "InterfaceError", "DatabaseError", "ProgrammingError"}
        if err_type in _availability or "connect" in str(e).lower() or "refused" in str(e).lower():
            print(f"  [L1-L3] Database unavailable — using standalone fallback: {e}")
            return None
        raise


_BEST_PARAMS_PATH = _ROOT / "best_retriever_params.json"


def _load_best_params() -> dict:
    """Load tuned retriever params if available; return empty dict on miss."""
    if _BEST_PARAMS_PATH.exists():
        try:
            with open(_BEST_PARAMS_PATH, encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            pass
    return {}


def _bridge_and_inject(query: str, top_n_derive: int = 5) -> list:
    """
    Run arXiv bridge search, append novel papers to CSV, return pre_selected list.

    Preconditions:
        arxiv_bridge module importable from _ROOT.
        copilot-proxy running (for LLM utility derivation of novel papers).
    Postconditions:
        Novel papers with LLM-derived utility are appended to the master CSV so
        SyllogismRetriever._load_csv_rows() can find them on the next init.
        Returns list of (arxiv_id, relevance_score) for ALL bridge results.
    Failure modes:
        bridge_search returns [] → returns []; caller falls back to pgvector/standalone.
        CSV append fails → logged, retrieval continues with papers already in CSV.
    """
    import json as _json
    import csv as _csv_mod

    sys.path.insert(0, str(_ROOT))
    from arxiv_bridge import bridge_search  # noqa: E402

    print("\n  [bridge] Searching upstream arXiv sources...")
    results = bridge_search(query, limit=20, top_n_derive=top_n_derive)
    if not results:
        print("  [bridge] No upstream results — using local sources only.")
        return []

    # Append novel+complete papers to CSV so the retriever can see them
    novel_written = 0
    existing_ids: set[str] = set()
    try:
        with open(_CSV, encoding="utf-8", newline="") as fh:
            for row in _csv_mod.DictReader(fh):
                existing_ids.add(row.get("arxiv_id", "").strip().strip('"'))
    except Exception:
        pass

    to_append = [r for r in results
                 if not r.local and r.is_complete and r.arxiv_id not in existing_ids]
    if to_append:
        try:
            with open(_CSV, "a", encoding="utf-8", newline="") as fh:
                writer = _csv_mod.writer(fh)
                for r in to_append:
                    writer.writerow([
                        r.arxiv_id,
                        r.title,
                        r.abstract,
                        _json.dumps(r.utility),
                        _json.dumps(r.barriers),
                        r.thesis,
                        True,
                    ])
            novel_written = len(to_append)
            print(f"  [bridge] Appended {novel_written} novel paper(s) to corpus CSV.")
        except Exception as exc:
            print(f"  [bridge] WARNING: CSV append failed ({exc}); novel papers skipped.")

    local_count   = sum(1 for r in results if r.local)
    novel_count   = len(results) - local_count
    print(f"  [bridge] {len(results)} results: {local_count} local, "
          f"{novel_count} novel ({novel_written} added to CSV, "
          f"{novel_count - novel_written} text-only or cached)")

    return [(r.arxiv_id, r.relevance_score) for r in results]


def run_retrieval(
    query: str,
    n_papers: int,
    top_k: int,
    output: str,
    extract: bool = False,
    bridge: bool = False,
    bridge_derive: int = 5,
) -> None:
    """Run the 9-stage syllogism retriever and write the markdown report.

    If bridge=True, upstream arXiv/S2 results are fetched first, novel papers
    are appended to the corpus CSV, and all results feed into pre_selected so
    the retriever can rank them alongside local papers.
    """
    print("\n" + "=" * 60)
    print("STAGE 2 — Syllogism retrieval")
    print("=" * 60)
    print(f"  Query    : {query}")
    print(f"  top_k    : {top_k}")
    print(f"  output   : {output}\n")

    # Load tuned blend weights if available
    best = _load_best_params()
    blend_weights = best.get("blend_weights") or None
    if blend_weights:
        print(f"  Blend    : {blend_weights}  (from best_retriever_params.json)")
        if "n_papers" in best and n_papers == AGENT_TOP_K:
            n_papers = best["n_papers"]
    else:
        print("  Blend    : default (best_retriever_params.json not found)")

    # Optional: upstream arXiv bridge (must run BEFORE SyllogismRetriever init
    # so appended CSV rows are visible when _load_csv_rows() fires in __init__)
    bridge_pre_selected: list = []
    if bridge:
        bridge_pre_selected = _bridge_and_inject(query, top_n_derive=bridge_derive)

    # pgvector candidate pool — use the tuned top_k from best_retriever_params (or 13).
    # This is deliberately larger than the final top_k to give the syllogism pipeline
    # enough candidates to rerank from.
    pgvector_top_k = best.get("top_k", 13)
    pgvec_pre_selected = _try_pgvector_retrieval(query, pgvector_top_k)

    # Merge: pgvector takes priority (higher signal), bridge fills in the rest
    if bridge_pre_selected or pgvec_pre_selected:
        seen: set[str] = set()
        pre_selected: list = []
        for aid, score in (pgvec_pre_selected or []):
            if aid not in seen:
                seen.add(aid)
                pre_selected.append((aid, score))
        for aid, score in bridge_pre_selected:
            if aid not in seen:
                seen.add(aid)
                pre_selected.append((aid, score))
        if pgvec_pre_selected:
            print(f"  Source   : pgvector ({len(pgvec_pre_selected)}) + "
                  f"bridge ({len(bridge_pre_selected)}) → "
                  f"{len(pre_selected)} merged candidates → top {top_k}")
        else:
            print(f"  Source   : bridge only ({len(pre_selected)} candidates → top {top_k})")
    else:
        pre_selected = None
        print(f"  Source   : standalone cosine search (n_papers={n_papers})")

    from arxiv_pipeline.syllogism_retriever import SyllogismRetriever

    retriever = SyllogismRetriever(blend_weights=blend_weights)
    result    = retriever.retrieve(query, n_papers=n_papers, top_k=top_k,
                                   pre_selected=pre_selected)

    if extract:
        _run_on_demand_extraction(result)

    md = result.to_markdown()
    out_path = Path(output)
    out_path.write_text(md, encoding="utf-8")
    print(f"\n  Report written → {out_path.resolve()}")
    print(f"  Papers retrieved: {len(result.papers)}")
    if result.methods_content:
        print(f"  Methods extracted: {len(result.methods_content)} paper(s)")


# ── Entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(
        description="arXiv syllogism pipeline — warm cache then retrieve",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("query",          nargs="?",  default="",
                    help="Retrieval query (required unless --dry_run)")
    ap.add_argument("--top_k",        type=int,   default=AGENT_TOP_K,
                    help=f"Final papers to return from syllogism pipeline (default: {AGENT_TOP_K})")
    ap.add_argument("--n_papers",     type=int,   default=AGENT_TOP_K,
                    help="Standalone fallback candidate pool size (default: AGENT_TOP_K). "
                         "Ignored when pgvector 3-layer retriever is available.")
    ap.add_argument("--output",       type=str,   default="_report.md",
                    help="Output markdown file (default: _report.md)")
    ap.add_argument("--combine",      nargs="+",  metavar="FILE",
                    help="Combine Synthesis sections from these report files into --output")
    ap.add_argument("--warmup_limit", type=int,   default=0,
                    help="Max uncached papers to warm before retrieval (0 = all)")
    ap.add_argument("--warmup",       action="store_true",
                    help="Run KG cache warmup before retrieval (optional, slow)")
    ap.add_argument("--dry_run",      action="store_true",
                    help="Show warmup plan only; do not warm or retrieve")
    ap.add_argument("--extract",      action="store_true",
                    help=f"Run on-demand PDF extraction for top papers "
                         f"(Phase 5 ~20s if MD cached; full pipeline ~6 min if not)")
    ap.add_argument("--bridge",       action="store_true",
                    help="Augment with upstream arXiv/S2 search; novel papers are LLM-derived "
                         "and appended to corpus before retrieval")
    ap.add_argument("--bridge_derive", type=int, default=5,
                    help="Max novel papers to run LLM utility derivation on (default: 5)")
    args = ap.parse_args()

    if args.combine:
        combine_reports(args.combine, args.output)
        return

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
        query        = args.query,
        n_papers     = args.n_papers,
        top_k        = args.top_k,
        output       = args.output,
        extract      = args.extract,
        bridge       = args.bridge,
        bridge_derive= args.bridge_derive,
    )


if __name__ == "__main__":
    main()
