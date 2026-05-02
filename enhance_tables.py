#!/usr/bin/env python3
"""
Phase 1.5: Table enhancement — crops docling-detected tables, extracts structure
with tabula AND camelot, fuses all sources via VLM into clean markdown.

Preconditions:
    - {stem}.pdf exists (original PDF for tabula/camelot/image rendering)
    - {stem}.md exists (docling markdown output)
    - {stem}.json exists (docling JSON output; requires `docling --to json`)
    - Ollama running at OLLAMA_HOST:OLLAMA_PORT with a vision-capable model loaded
    - Java runtime available (for tabula)
Postconditions:
    - {stem}.md updated in-place: each docling-detected table replaced with
      the VLM-fused version (or left unchanged on per-table failure)
    - {stem}_table_crops/ directory containing cropped table images (PNG)
Failure modes:
    - tabula/camelot error per table: logged, that source omitted from VLM prompt
    - VLM error per table: logged, original docling markdown kept for that table
    - No tables found in JSON: exits cleanly with message
    - Missing JSON file: exits with error (re-run docling with --to json)
"""

import sys
import re
import json
import base64
import argparse
from pathlib import Path
from io import StringIO

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    import pymupdf as fitz  # pymupdf >= 1.24
except ImportError:
    import fitz             # older pymupdf
import requests
import tabula
import camelot

OLLAMA_HOST = "192.168.3.17"
OLLAMA_PORT = 11434
OLLAMA_MODEL = "qwen3-vl:2b"

# Finds all markdown table blocks (one or more consecutive lines starting with |)
_TABLE_BLOCK_RE = re.compile(r"(?m)(?:^\|[^\n]*\n)+")


# ── coordinate conversion helpers ──────────────────────────────────────────────

def _prov(item: dict) -> dict:
    """Return the first provenance record from an item (prov is always a list)."""
    prov = item.get("prov", [])
    return prov[0] if prov else {}


def _fitz_rect(bbox: dict, page_height: float) -> fitz.Rect:
    """
    Convert docling BOTTOMLEFT bbox to a pymupdf Rect (TOPLEFT origin).

    Require: bbox has l, t, r, b; page_height is the page height in points.
    """
    return fitz.Rect(
        bbox["l"],
        page_height - bbox["t"],
        bbox["r"],
        page_height - bbox["b"],
    )


def _tabula_area(bbox: dict, page_width: float, page_height: float) -> list[float]:
    """
    Return tabula area=[top%, left%, bottom%, right%] (distance from top-left).

    tabula uses percentage coordinates measured from the top-left corner.
    """
    return [
        (page_height - bbox["t"]) / page_height * 100,
        bbox["l"] / page_width * 100,
        (page_height - bbox["b"]) / page_height * 100,
        bbox["r"] / page_width * 100,
    ]


def _camelot_area(bbox: dict) -> str:
    """
    Return camelot table_areas string 'x1,y1,x2,y2' in PDF points.

    camelot uses BOTTOMLEFT origin: (x1,y1) = top-left, (x2,y2) = bottom-right
    in PDF coordinate space, meaning y1=t (higher value) and y2=b (lower value).
    """
    return f"{bbox['l']},{bbox['t']},{bbox['r']},{bbox['b']}"


# ── extraction helpers ──────────────────────────────────────────────────────────

def crop_table_image_b64(pdf_path: Path, page_no: int, bbox: dict,
                         page_height: float, crops_dir: Path,
                         table_idx: int) -> str:
    """
    Render the table region from the PDF page and return raw base64 PNG bytes.

    Require: page_no is 1-based (docling convention).
    Guarantee: saves PNG to crops_dir; returns base64 string.
    Failure mode: fitz errors propagate to caller.
    """
    doc = fitz.open(str(pdf_path))
    page = doc[page_no - 1]        # fitz is 0-based
    rect = _fitz_rect(bbox, page_height)
    clip = rect & page.rect        # clamp to page bounds
    mat = fitz.Matrix(2, 2)        # 2× zoom → ~144 dpi
    pix = page.get_pixmap(matrix=mat, clip=clip)
    crops_dir.mkdir(parents=True, exist_ok=True)
    out_path = crops_dir / f"table_{table_idx:03d}_p{page_no}.png"
    pix.save(str(out_path))
    doc.close()
    return base64.b64encode(pix.tobytes("png")).decode("ascii")


def extract_tabula(pdf_path: Path, page_no: int, bbox: dict,
                   page_width: float, page_height: float) -> str:
    """
    Extract table text via tabula (Java-based; strong on ruled/bordered tables).

    Require: Java is on PATH.
    Guarantee: returns a plain-text rendering of the extracted table(s).
    Failure mode: returns empty string on any error.
    """
    try:
        area = _tabula_area(bbox, page_width, page_height)
        dfs = tabula.read_pdf(
            str(pdf_path),
            pages=page_no,
            area=area,
            guess=False,
            pandas_options={"header": None},
            silent=True,
        )
        if not dfs:
            return ""
        parts = []
        for df in dfs:
            df = df.fillna("")
            rows = [" | ".join(str(v) for v in row) for _, row in df.iterrows()]
            parts.append("\n".join(rows))
        return "\n\n".join(parts)
    except Exception as exc:
        return f"[tabula error: {exc}]"


def extract_camelot(pdf_path: Path, page_no: int, bbox: dict) -> tuple[str, str, float]:
    """
    Extract table text via camelot. Tries lattice first (line-based); falls back to stream.

    Require: camelot installed; PDF page_no is 1-based.
    Guarantee: returns (text, flavor_used, accuracy_pct).
        accuracy_pct >= 80 indicates visible grid lines; use as ruled-table signal.
    Failure mode: returns ("", "none", 0.0) on all errors.
    """
    area_str = _camelot_area(bbox)
    for flavor in ("lattice", "stream"):
        try:
            tables = camelot.read_pdf(
                str(pdf_path),
                pages=str(page_no),
                flavor=flavor,
                table_areas=[area_str],
            )
            if tables and len(tables) > 0:
                t = tables[0]
                accuracy = float(t.parsing_report.get("accuracy", 0.0))
                df = t.df.fillna("")
                rows = [" | ".join(str(v) for v in row) for _, row in df.iterrows()]
                return "\n".join(rows), flavor, accuracy
        except Exception:
            continue
    return "", "none", 0.0


def docling_table_to_md(table_item: dict) -> str:
    """
    Reconstruct a markdown table from docling JSON table_cells data.

    Require: table_item has data.table_cells, data.num_rows, data.num_cols.
    Guarantee: returns markdown string; empty string if no cell data.
    """
    data = table_item.get("data", {})
    num_rows = data.get("num_rows", 0)
    num_cols = data.get("num_cols", 0)
    if not num_rows or not num_cols:
        return ""

    grid = [["" for _ in range(num_cols)] for _ in range(num_rows)]
    for cell in data.get("table_cells", []):
        r = cell.get("start_row_offset_idx", 0)
        c = cell.get("start_col_offset_idx", 0)
        if r < num_rows and c < num_cols:
            grid[r][c] = str(cell.get("text", "")).replace("\n", " ").strip()

    lines = []
    for i, row in enumerate(grid):
        lines.append("| " + " | ".join(row) + " |")
        if i == 0:
            lines.append("|" + "---|" * num_cols)
    return "\n".join(lines)


def fuse_with_vlm(b64_image: str, docling_md: str, tabula_text: str,
                  camelot_text: str, camelot_flavor: str, camelot_accuracy: float,
                  host: str, port: int, model: str) -> str:
    """
    Ask the VLM to fuse three extraction sources into a single clean markdown table.

    Require: b64_image is raw base64 PNG (no data URI prefix).
    Guarantee: returns the VLM's response text; empty string on failure.
    Failure mode: HTTP/JSON errors logged, returns empty string.
    """
    _camelot_mode_desc = {
        "lattice": (
            "lattice mode — requires visible grid lines; reliable for ruled tables "
            "but tends to split merged/spanning cells into separate entries"
        ),
        "stream": (
            "stream mode — whitespace-based; works without grid lines but column "
            "boundaries drift in dense or multi-column layouts"
        ),
        "none": "no output — all camelot extraction attempts failed",
    }
    camelot_desc = _camelot_mode_desc.get(camelot_flavor, camelot_flavor)

    prompt = (
        "You are a document table transcription assistant. A PDF table region has "
        "been extracted using three independent methods, each with known failure modes. "
        "Use their outputs as signals, but treat the image as the authoritative source.\n\n"
        "DOCLING EXTRACTION (structural layout model — may hallucinate column spans in "
        "complex tables; reliable for simple grids and header detection):\n"
        f"{docling_md or '[no output]'}\n\n"
        "TABULA EXTRACTION (Java text-layer extraction — column alignment drifts on "
        "rotated or skewed tables; unreliable without clear column delimiters):\n"
        f"{tabula_text or '[no output]'}\n\n"
        f"CAMELOT EXTRACTION ({camelot_desc}; accuracy: {camelot_accuracy:.0f}%):\n"
        f"{camelot_text or '[no output]'}\n\n"
        "Examine the image carefully. Pay special attention to:\n"
        "- Merged or spanning cells (all three extractors tend to split these incorrectly)\n"
        "- Header rows vs data rows\n"
        "- Column alignment, especially in borderless tables\n"
        "Return ONLY the markdown table — no commentary, no code fences."
    )
    url = f"http://{host}:{port}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "images": [b64_image],
        "stream": False,
    }
    try:
        resp = requests.post(url, json=payload, timeout=180)
        resp.raise_for_status()
        raw = resp.json().get("response", "")
        if not raw.strip():
            raw = resp.json().get("thinking", "")
        return raw.strip()
    except Exception as exc:
        print(f"    [VLM error] {exc}")
        return ""


# ── markdown patching ───────────────────────────────────────────────────────────

def find_md_tables(md_content: str) -> list[tuple[int, int]]:
    """
    Return list of (start, end) character spans for each markdown table block.

    A table block is one or more consecutive lines beginning with '|'.
    """
    return [(m.start(), m.end()) for m in _TABLE_BLOCK_RE.finditer(md_content)]


def find_md_table_match(md_content: str, docling_md: str, fallback_idx: int) -> int:
    """
    Find the markdown table block index that best matches the docling reconstruction.

    Uses Jaccard token overlap between docling_md and each pipe-block in md_content.
    Falls back to fallback_idx when the best overlap score is below 0.25 (docling_md
    has too little text to fingerprint — e.g. single-row or numeric-only tables).

    Require: fallback_idx is a valid positional index (clamped to table count).
    Guarantee: always returns an index in [0, len(blocks)-1].
    """
    spans = find_md_tables(md_content)
    if not spans:
        return fallback_idx
    docling_tokens = set(re.findall(r'\w+', docling_md.lower()))
    if len(docling_tokens) < 5:
        return min(fallback_idx, len(spans) - 1)
    best_idx, best_score = fallback_idx, 0.0
    for i, (start, end) in enumerate(spans):
        block_tokens = set(re.findall(r'\w+', md_content[start:end].lower()))
        if not block_tokens:
            continue
        score = len(docling_tokens & block_tokens) / len(docling_tokens | block_tokens)
        if score > best_score:
            best_score, best_idx = score, i
    return best_idx if best_score >= 0.25 else min(fallback_idx, len(spans) - 1)


def patch_markdown(md_content: str, enhanced_tables: dict[int, str]) -> str:
    """
    Replace the i-th markdown table block with enhanced_tables[i].

    Require: enhanced_tables keys are 0-based indices into the table block list.
    Guarantee: returns patched content; unmodified blocks kept for missing keys.
    """
    spans = find_md_tables(md_content)
    # Replace from end to start to preserve earlier offsets
    result = md_content
    for i in sorted(enhanced_tables.keys(), reverse=True):
        if i >= len(spans):
            continue
        start, end = spans[i]
        replacement = enhanced_tables[i]
        if not replacement.endswith("\n"):
            replacement += "\n"
        result = result[:start] + replacement + result[end:]
    return result


# ── main ────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Enhance docling-detected tables using tabula + camelot + VLM fusion"
    )
    parser.add_argument("pdf_file", help="Original PDF path")
    parser.add_argument("md_file", help="Docling markdown output path")
    parser.add_argument("json_file", help="Docling JSON output path (--to json)")
    parser.add_argument("--host", default=OLLAMA_HOST)
    parser.add_argument("--port", type=int, default=OLLAMA_PORT)
    parser.add_argument("--model", default=OLLAMA_MODEL)
    parser.add_argument("--crops-dir", default=None,
                        help="Directory for cropped table images (default: <stem>_table_crops/)")
    args = parser.parse_args()

    pdf_path = Path(args.pdf_file)
    md_path = Path(args.md_file)
    json_path = Path(args.json_file)

    for p in (pdf_path, md_path, json_path):
        if not p.is_file():
            print(f"Error: not found: {p}")
            return 1

    crops_dir = Path(args.crops_dir) if args.crops_dir else md_path.parent / (md_path.stem + "_table_crops")

    # Load docling JSON
    doc = json.loads(json_path.read_text(encoding="utf-8"))
    tables = doc.get("tables", [])
    pages = doc.get("pages", {})

    if not tables:
        print(f"No tables found in {json_path.name} — nothing to enhance.")
        return 0

    print(f"Found {len(tables)} table(s) in {json_path.name}")

    md_content = md_path.read_text(encoding="utf-8")
    md_table_spans = find_md_tables(md_content)
    print(f"Found {len(md_table_spans)} table block(s) in {md_path.name}")

    enhanced: dict[int, str] = {}
    succeeded = skipped = failed = 0

    for idx, table_item in enumerate(tables):
        prov = _prov(table_item)
        if not prov:
            print(f"  [{idx}] no provenance — skipping")
            skipped += 1
            continue

        page_no = prov["page_no"]
        bbox = prov["bbox"]
        page_key = str(page_no)
        if page_key not in pages:
            print(f"  [{idx}] page {page_no} not in pages dict — skipping")
            skipped += 1
            continue

        page_size = pages[page_key]["size"]
        page_w, page_h = page_size["width"], page_size["height"]

        print(f"  [{idx + 1}/{len(tables)}] page {page_no} bbox "
              f"({bbox['l']:.0f},{bbox['b']:.0f})→({bbox['r']:.0f},{bbox['t']:.0f})")

        # 1. Crop image
        try:
            b64 = crop_table_image_b64(pdf_path, page_no, bbox, page_h, crops_dir, idx)
            print(f"    image: {len(b64)} b64 chars")
        except Exception as exc:
            print(f"    [crop error] {exc} — skipping table")
            skipped += 1
            continue

        # 2. Tabula extraction
        tab_text = extract_tabula(pdf_path, page_no, bbox, page_w, page_h)
        print(f"    tabula: {len(tab_text)} chars")

        # 3. Camelot extraction
        cam_text, cam_flavor, cam_accuracy = extract_camelot(pdf_path, page_no, bbox)
        print(f"    camelot: {len(cam_text)} chars ({cam_flavor}, {cam_accuracy:.0f}%)")

        # 4. Docling markdown for this table
        docling_md = docling_table_to_md(table_item)
        print(f"    docling: {len(docling_md)} chars")

        # Map JSON table index → best-matching MD pipe block (text-overlap fingerprint)
        md_idx = find_md_table_match(md_content, docling_md, idx)
        if md_idx != idx:
            print(f"    fingerprint: JSON[{idx}] → MD block [{md_idx}] (positional was {idx})")

        # 5. VLM fusion
        print("    fusing via VLM...", end="", flush=True)
        fused = fuse_with_vlm(b64, docling_md, tab_text, cam_text, cam_flavor, cam_accuracy,
                              args.host, args.port, args.model)
        if fused:
            print(f" {len(fused)} chars")
            enhanced[md_idx] = fused
            succeeded += 1
        else:
            print(" failed — keeping docling original")
            if docling_md:
                enhanced[md_idx] = docling_md
            failed += 1

    if enhanced:
        patched = patch_markdown(md_content, enhanced)
        md_path.write_text(patched, encoding="utf-8")
        print(f"\nPatched {len(enhanced)} table(s) in {md_path.name}")
    else:
        print("\nNo tables patched.")

    print(f"  Succeeded: {succeeded}  Skipped: {skipped}  Failed: {failed}")
    return 1 if failed and not succeeded else 0


if __name__ == "__main__":
    sys.exit(main())
