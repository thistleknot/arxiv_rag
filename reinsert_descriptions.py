#!/usr/bin/env python3
"""
Phase 4: Reinsert VLM image descriptions into a Markdown document.

Reads <image_NNN> reference tags placed by post_process_md-csv.py and replaces
each with an inline XML span: <image_NNN>\ndescription\n</image_NNN>.

When value_added is False (image is purely decorative / document scaffolding),
the description is replaced with the literal word "formatting" so downstream
LLMs know the image added no content.

Preconditions:
    - Markdown file contains <image_NNN> reference tags (zero-padded sequential index)
    - CSV file has columns: image_index, description, value_added (from vlm_describe.py)
Postconditions:
    - All <image_NNN> tags replaced with '<image_NNN>\ntext\n</image_NNN>'
      where text = description if value_added else "formatting"
    - CSV is not modified
Failure modes:
    - Missing 'description' column: fails fast — run vlm_describe.py first
    - Tag with no matching description: replaced with placeholder text
    - Missing 'value_added' column: assumes True (backward compatible)
"""

import csv
import re
import sys
import argparse
from pathlib import Path

# sys.maxsize overflows C long on Windows — binary-search for the max accepted value
_limit = sys.maxsize
while True:
    try:
        csv.field_size_limit(_limit)
        break
    except OverflowError:
        _limit = _limit // 10

_PLACEHOLDER = "[Image description unavailable]"

# Matches a bare <image_NNN> tag — one NOT immediately followed by \n (i.e., not yet expanded)
_BARE_TAG_RE = re.compile(r"<image_(\d+)>(?!\n)")
# Matches a fully expanded block opened by <image_NNN>\n
_EXPANDED_TAG_RE = re.compile(r"<image_(\d+)>\n")


def count_bare_image_tags(md_content: str) -> int:
    """
    Count <image_NNN> tags that have not been expanded by Phase 4.

    A bare tag looks like '<image_001>' with no newline immediately after.
    An expanded tag looks like '<image_001>\ndescription\n</image_001>'.

    Use this as a pre-condition guard before chunking: if count > 0,
    Phase 4 has not completed and the document should not be chunked.

    Guarantee: returns 0 when all image tokens are populated.
    """
    return len(_BARE_TAG_RE.findall(md_content))


def count_expanded_image_tags(md_content: str) -> int:
    """Return the number of fully expanded <image_NNN>\\ndesc\\n</image_NNN> blocks."""
    return len(_EXPANDED_TAG_RE.findall(md_content))


def load_descriptions(csv_path: Path) -> dict[int, tuple[str, bool]]:
    """
    Load image descriptions from CSV into a {image_index: (description, value_added)} mapping.

    Require: CSV has 'image_index' and 'description' columns.
    Guarantee: returns dict mapping int image_index to (description, value_added) tuples.
    """
    result: dict[int, tuple[str, bool]] = {}
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if "description" not in fieldnames:
            raise ValueError("CSV is missing 'description' column. Run vlm_describe.py first.")
        key_col = "image_index" if "image_index" in fieldnames else "line_number"
        has_va = "value_added" in fieldnames
        for row in reader:
            desc = row.get("description", "").strip()
            va_raw = row.get("value_added", "True") if has_va else "True"
            value_added = va_raw.strip().lower() not in ("false", "0", "")
            result[int(row[key_col])] = (desc, value_added)
    return result


def reinsert(md_path: Path, descriptions: dict[int, tuple[str, bool]]) -> tuple[int, int, int]:
    """
    Replace all <image_NNN> tags in the Markdown file with inline description spans.

    Output format: <image_NNN>\ntext\n</image_NNN>
    where text = description when value_added=True, else "formatting".

    Require: md_path is a readable/writable .md file.
    Guarantee: each tag replaced inline; file written in-place.
    Returns: (replacements_made, missing_count, formatting_count)
    """
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()

    replaced = 0
    missing = 0
    formatting_count = 0

    def replacer(match: re.Match) -> str:
        nonlocal replaced, missing, formatting_count
        img_idx = int(match.group(1))
        entry = descriptions.get(img_idx)
        if entry is None:
            missing += 1
            text = _PLACEHOLDER
        else:
            desc, value_added = entry
            if not value_added:
                formatting_count += 1
                text = "formatting"
            else:
                text = desc or _PLACEHOLDER
        replaced += 1
        return f"<image_{img_idx:03d}>\n{text}\n</image_{img_idx:03d}>"

    modified = re.sub(r"<image_(\d+)>", replacer, content)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(modified)

    return replaced, missing, formatting_count


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reinsert VLM image descriptions into a Markdown document"
    )
    parser.add_argument("md_file", help="Path to the Markdown file with <image N> tags")
    parser.add_argument("csv_file", help="Path to the CSV file with descriptions")
    args = parser.parse_args()

    md_path = Path(args.md_file)
    csv_path = Path(args.csv_file)

    if not md_path.is_file():
        print(f"Error: Markdown file not found: {md_path}")
        return 1
    if not csv_path.is_file():
        print(f"Error: CSV file not found: {csv_path}")
        return 1

    try:
        descriptions = load_descriptions(csv_path)
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    replaced, missing, formatting_count = reinsert(md_path, descriptions)

    content_count = replaced - formatting_count
    print(f"Reinserted {replaced} description(s) into: {md_path}")
    print(f"  Content: {content_count}  Formatting-only: {formatting_count}")
    if missing:
        print(f"  Warning: {missing} tag(s) had no matching description — placeholder inserted.")

    # Post-condition: verify no bare tags remain (guard for future chunking)
    final_content = md_path.read_text(encoding="utf-8")
    bare = count_bare_image_tags(final_content)
    expanded = count_expanded_image_tags(final_content)
    if bare:
        print(f"  ERROR: {bare} bare <image_NNN> tag(s) remain unexpanded — chunking must not proceed.")
        return 1
    print(f"  All {expanded} image token(s) populated — safe to chunk.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
