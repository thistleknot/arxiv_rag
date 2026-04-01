"""
docling_pipeline.py — Download arXiv PDFs and convert to markdown via docling.

Processes only papers whose .md file is missing from papers/post_processed/.
Output filename convention: {arxiv_id.replace('.', '_')}.md  (matches ingestion pipeline).

Pipeline per paper:
  1. Check if .md already exists → skip
  2. Download PDF from https://arxiv.org/pdf/{arxiv_id}
  3. Run docling DocumentConverter → export_to_markdown()
  4. Save to papers/post_processed/{stem}.md
  5. Delete temp PDF

Usage:
  python docling_pipeline.py                  # all gaps
  python docling_pipeline.py --limit 10       # first 10
  python docling_pipeline.py --delay 5        # seconds between downloads (default 3)
  python docling_pipeline.py --workers 2      # parallel docling workers
"""

import argparse
import csv
import os
import pathlib
import sys
import time
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import requests

_ROOT      = pathlib.Path(__file__).resolve().parent.parent
_POST      = _ROOT / "papers" / "post_processed"
_PDFS_DIR  = _ROOT / "papers" / "pdfs"
_FAIL_LOG  = _ROOT / "_docling_failures.csv"

_POST.mkdir(parents=True, exist_ok=True)
_PDFS_DIR.mkdir(parents=True, exist_ok=True)

ARXIV_PDF_URL = "https://arxiv.org/pdf/{arxiv_id}"

# ── helpers ──────────────────────────────────────────────────────────────────

def arxiv_stem(arxiv_id: str) -> str:
    """2310.02025 → 2310_02025"""
    return arxiv_id.strip().strip('"').replace(".", "_")


def md_path(arxiv_id: str) -> pathlib.Path:
    return _POST / f"{arxiv_stem(arxiv_id)}.md"


def load_gap_ids() -> list[str]:
    """Return arxiv_ids present in the cleaned CSV but missing a .md file."""
    import pandas as pd
    csv_path = _POST / "arxiv_data_with_analysis_cleaned.csv"
    df = pd.read_csv(csv_path, encoding_errors="replace")
    all_ids = df["arxiv_id"].astype(str).str.strip().str.strip('"').tolist()
    missing = [aid for aid in all_ids if not md_path(aid).exists()]
    return missing


# ── download ─────────────────────────────────────────────────────────────────

def download_pdf(arxiv_id: str, dest: pathlib.Path, timeout: int = 60) -> bool:
    """Download PDF to dest.  Returns True on success."""
    urls = [
        ARXIV_PDF_URL.format(arxiv_id=arxiv_id),
        ARXIV_PDF_URL.format(arxiv_id=arxiv_id) + "v1",
    ]
    for url in urls:
        try:
            resp = requests.get(url, timeout=timeout, headers={"User-Agent": "arxiv-research-tool/1.0"})
            if resp.status_code == 200 and len(resp.content) > 1_000:
                dest.write_bytes(resp.content)
                return True
        except Exception as e:
            print(f"  [{arxiv_id}] Download error: {e}")
    print(f"  [{arxiv_id}] All URLs failed – skipping")
    return False


# ── docling ──────────────────────────────────────────────────────────────────

_converter = None  # lazy singleton — expensive to initialise


def get_converter():
    global _converter
    if _converter is None:
        from docling.document_converter import DocumentConverter
        _converter = DocumentConverter()
    return _converter


def convert_pdf(pdf_path: pathlib.Path) -> str | None:
    """Run docling on pdf_path, return markdown text or None on failure."""
    try:
        result = get_converter().convert(str(pdf_path))
        return result.document.export_to_markdown()
    except Exception as e:
        print(f"  [docling] {pdf_path.name}: {e}")
        return None


# ── per-paper orchestration ───────────────────────────────────────────────────

def process_paper(arxiv_id: str) -> tuple[str, str]:
    """
    Download, convert, save one paper.
    Returns (arxiv_id, status) where status ∈ {'ok', 'skip', 'fail_download',
    'fail_convert', 'fail_empty'}.
    """
    out = md_path(arxiv_id)
    if out.exists():
        return arxiv_id, "skip"

    pdf_dest = _PDFS_DIR / f"{arxiv_stem(arxiv_id)}.pdf"

    # 1. Download
    if not pdf_dest.exists():
        ok = download_pdf(arxiv_id, pdf_dest)
        if not ok:
            return arxiv_id, "fail_download"

    # 2. Convert
    md_text = convert_pdf(pdf_dest)
    if md_text is None:
        pdf_dest.unlink(missing_ok=True)
        return arxiv_id, "fail_convert"
    if len(md_text.strip()) < 200:
        pdf_dest.unlink(missing_ok=True)
        return arxiv_id, "fail_empty"

    # 3. Save
    out.write_text(md_text, encoding="utf-8", errors="replace")

    # 4. Clean up PDF (saves disk space)
    pdf_dest.unlink(missing_ok=True)

    return arxiv_id, "ok"


# ── log failures ─────────────────────────────────────────────────────────────

def append_failure(arxiv_id: str, reason: str):
    write_header = not _FAIL_LOG.exists()
    with open(_FAIL_LOG, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["arxiv_id", "reason", "timestamp"])
        w.writerow([arxiv_id, reason, datetime.utcnow().isoformat()])


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Docling PDF→Markdown pipeline for arXiv papers")
    ap.add_argument("--limit",   type=int, default=0,   help="Max papers to process (0 = all)")
    ap.add_argument("--delay",   type=float, default=3.0, help="Seconds between PDF downloads")
    ap.add_argument("--workers", type=int, default=1,   help="Parallel workers (docling is CPU-heavy; 1-2 recommended)")
    ap.add_argument("--ids",     nargs="*",              help="Process specific arxiv IDs only")
    args = ap.parse_args()

    # ── load gap list
    if args.ids:
        gap = [i.strip() for i in args.ids]
    else:
        print("Computing gap …")
        gap = load_gap_ids()

    if args.limit > 0:
        gap = gap[:args.limit]

    total = len(gap)
    print(f"Papers to process: {total}")
    if total == 0:
        print("Nothing to do.")
        return

    # ── warm up docling
    print("Loading docling converter …")
    get_converter()
    print("Docling ready.\n")

    ok = fail = skip = 0
    t0 = time.time()

    if args.workers > 1:
        # Parallel: download+convert concurrently (rate-limiting downloads is
        # trickier in parallel; trust arXiv CDN won't block at low concurrency)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(process_paper, aid): aid for aid in gap}
            done = 0
            for fut in as_completed(futures):
                aid = futures[fut]
                try:
                    _, status = fut.result()
                except Exception as exc:
                    status = "fail_convert"
                    print(f"  [{aid}] Unexpected error: {exc}")

                done += 1
                if status == "ok":
                    ok += 1
                elif status == "skip":
                    skip += 1
                else:
                    fail += 1
                    append_failure(aid, status)

                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (total - done) / rate if rate > 0 else 0
                print(f"[{done}/{total}] {aid} -> {status}  "
                      f"ok={ok} fail={fail}  "
                      f"ETA {eta/60:.1f} min")
    else:
        # Serial: enforce download delay
        for i, aid in enumerate(gap, 1):
            _, status = process_paper(aid)

            if status == "ok":
                ok += 1
            elif status == "skip":
                skip += 1
            else:
                fail += 1
                append_failure(aid, status)

            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0
            eta = (total - i) / rate if rate > 0 else 0
            print(f"[{i}/{total}] {aid} -> {status}  "
                  f"ok={ok} fail={fail}  "
                  f"ETA {eta/60:.1f} min")

            # Rate-limit downloads (arXiv policy)
            if status in ("ok", "fail_convert", "fail_empty") and i < total:
                time.sleep(args.delay)

    elapsed = time.time() - t0
    print(f"\n{'─'*60}")
    print(f"Done in {elapsed/60:.1f} min")
    print(f"  ok:   {ok}")
    print(f"  skip: {skip}")
    print(f"  fail: {fail}")
    if fail:
        print(f"  Failures logged to {_FAIL_LOG}")


if __name__ == "__main__":
    main()
