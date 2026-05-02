#!/usr/bin/env python3
"""
Phase 3: Generate VLM descriptions for base64 images stored in a CSV file.

Calls qwen3-vl:2b via local Ollama using structured JSON output (Pydantic schema)
and stores both the description and value_added flag back into the CSV.

Preconditions:
    - CSV file exists with columns: image_index, base64_data
    - Ollama running at OLLAMA_HOST:OLLAMA_PORT with OLLAMA_MODEL loaded
Postconditions:
    - CSV updated with 'description' and 'value_added' columns per image
Failure modes:
    - API errors: logged per-image with [ERROR] placeholder, pipeline continues
    - Rows with [ERROR] or [PENDING] prefixes are retried on subsequent runs
    - Rows missing 'value_added' (from a prior schema-less run) are reprocessed
"""

import csv
import json
import sys
import argparse
from pathlib import Path
from pydantic import BaseModel

import requests

# sys.maxsize overflows C long on Windows — binary-search for the max accepted value
_limit = sys.maxsize
while True:
    try:
        csv.field_size_limit(_limit)
        break
    except OverflowError:
        _limit = _limit // 10

OLLAMA_HOST = "192.168.3.17"
OLLAMA_PORT = 11434
OLLAMA_MODEL = "qwen3-vl:2b"
PROMPT = (
    "Describe this image concisely for document transcription. "
    "Focus on any text, data, charts, tables, diagrams, and key visual elements. "
    "Be factual and precise. Use a single paragraph for 'description'.\n\n"
    "Set 'value_added' to false only when the image is purely decorative or "
    "document scaffolding — e.g. page borders, dividing lines, headers or footers "
    "that contain only a logo or page number, watermarks, or blank space. "
    "Set 'value_added' to true when the image contains substantive content such as "
    "a figure, chart, table, diagram, photo, or screenshot."
)
_RETRY_PREFIXES = ("[ERROR]", "[PENDING]")


class ImageDescription(BaseModel):
    """Structured VLM response for a single image."""
    description: str
    value_added: bool


def describe_image(b64: str, host: str, port: int, model: str) -> ImageDescription:
    """
    Call Ollama VLM to generate a structured description of an image.

    Require: b64 is raw base64 (no data URI prefix), Ollama server is reachable.
    Guarantee: returns ImageDescription with description text and value_added flag.
    Failure mode: JSON parse error falls back to raw response with value_added=True.
    """
    url = f"http://{host}:{port}/api/generate"
    payload = {
        "model": model,
        "prompt": PROMPT,
        "images": [b64],
        "stream": False,
        "format": ImageDescription.model_json_schema(),
    }
    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    # qwen3-vl (thinking model) puts structured JSON in 'thinking' when format= is set
    raw = data.get("response", "")
    if not raw.strip():
        raw = data.get("thinking", "")
    try:
        data = json.loads(raw)
        result = ImageDescription.model_validate(data)
        # Normalize description: collapse internal newlines into spaces
        result.description = " ".join(result.description.split())
        return result
    except Exception:
        # Structured output failed — treat raw text as description, assume value-added
        return ImageDescription(
            description=" ".join(raw.split()),
            value_added=True,
        )


def should_skip(row: dict) -> bool:
    """
    Return True if this row already has a complete, successful response.

    Rows missing 'value_added' (produced by an older schema-less run) are
    treated as incomplete and will be reprocessed.
    """
    desc = row.get("description", "")
    if not desc or any(desc.startswith(p) for p in _RETRY_PREFIXES):
        return False
    return bool(row.get("value_added", "").strip())


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate VLM descriptions for images in a CSV file via Ollama"
    )
    parser.add_argument("csv_file", help="Path to the CSV file with base64 image data")
    parser.add_argument("--host", default=OLLAMA_HOST, help="Ollama host")
    parser.add_argument("--port", type=int, default=OLLAMA_PORT, help="Ollama port")
    parser.add_argument("--model", default=OLLAMA_MODEL, help="Ollama model name")
    args = parser.parse_args()

    csv_path = Path(args.csv_file)
    if not csv_path.is_file():
        print(f"Error: CSV file not found: {csv_path}")
        return 1

    rows: list[dict] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            rows.append(dict(row))

    if not rows:
        print("No images found in CSV. Nothing to process.")
        return 0

    pending = [r for r in rows if not should_skip(r)]
    print(f"Found {len(rows)} image(s). {len(pending)} need description.")

    succeeded = skipped = failed = 0

    for i, row in enumerate(rows):
        if should_skip(row):
            skipped += 1
            continue

        img_idx = row.get("image_index", row.get("line_number", "?"))
        print(f"  [{i + 1}/{len(rows)}] image {img_idx}: calling {args.model}...", end="", flush=True)
        try:
            result = describe_image(row["base64_data"], args.host, args.port, args.model)
            row["description"] = result.description
            row["value_added"] = str(result.value_added)
            succeeded += 1
            va_label = "content" if result.value_added else "formatting"
            print(f" done ({len(result.description)} chars, {va_label})")
        except Exception as exc:
            row["description"] = f"[ERROR] {exc}"
            row["value_added"] = ""
            failed += 1
            print(f" FAILED: {exc}")

    # Write back preserving all original columns, appending new ones if needed
    for col in ("description", "value_added"):
        if col not in fieldnames:
            fieldnames.append(col)

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDescriptions saved to: {csv_path}")
    print(f"  Succeeded: {succeeded}  Skipped: {skipped}  Failed: {failed}")

    return 1 if failed and not succeeded else 0


if __name__ == "__main__":
    sys.exit(main())
