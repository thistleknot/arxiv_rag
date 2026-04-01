"""
Core Thesis:
    Identify arxiv IDs present in llm.txt but absent from arxiv_data_with_analysis.csv,
    fetch title+abstract via the arxiv HTTP API, extract utility/barriers/thesis
    via copilot-proxy (gpt-4o structured output), then append rows to the CSV.

Workflow:
    1. load_gap_ids()         -- compare llm.txt vs CSV; return list of missing IDs
    2. fetch_arxiv(id)        -- pull title + abstract via arxiv REST API (XML)
    3. extract_fields(paper)  -- call gpt-4o → PaperAnalysis (Pydantic)
    4. append_to_csv(rows)    -- write new rows back to CSV

Dependencies:
    requests, openai, pydantic, pandas, tqdm
"""

from __future__ import annotations

import json
import re
import time
import sys
from pathlib import Path
from typing import List, Optional

import requests
import pandas as pd
from openai import OpenAI
from pydantic import BaseModel, Field
from tqdm import tqdm

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
LLM_TXT    = Path(r"C:\Users\user\Documents\wiki\data science\llm\llm.txt")
CSV_PATH   = Path(r"C:\Users\user\arxiv_id_lists\papers\post_processed\arxiv_data_with_analysis.csv")
STATE_PATH = Path(r"C:\Users\user\arxiv_id_lists\_extractor_state.json")

PROXY_BASE = "http://127.0.0.1:8069/v1"
PROXY_KEY  = "dummy-key"
MODEL      = "gpt-4o"

ARXIV_DELAY      = 5.0   # seconds between arxiv API calls (polite crawling)
LLM_DELAY        = 1.0   # seconds between LLM calls
ARXIV_MAX_RETRY  = 5     # retries on 429
ARXIV_BACKOFF    = 15.0  # extra wait on 429 (seconds)

EXTRACT_PROMPT = """\
You are a research analyst reading an AI/ML arxiv abstract.
Extract exactly the following three fields and return valid JSON only:

{{
  "utility": ["<concrete application or use-case 1>", "..."],
  "barriers": ["<limitation or unsolved problem 1>", "..."],
  "thesis": "<the core contribution/claim in one clear sentence>"
}}

Rules:
- utility: 1-5 items; what this method ENABLES in practice
- barriers: 1-5 items; what it CANNOT do or what it still requires
- thesis: exactly one sentence summarising the central contribution
- Return JSON ONLY — no markdown fences, no commentary

Abstract:
{abstract}
"""

# ──────────────────────────────────────────────
# Pydantic model
# ──────────────────────────────────────────────
class PaperAnalysis(BaseModel):
    arxiv_id:    str
    title:       str
    abstract:    str
    utility:     List[str]  = Field(description="Concrete applications the method enables")
    barriers:    List[str]  = Field(description="Limitations or unsolved problems")
    thesis:      str         = Field(description="Core contribution in one sentence")
    is_complete: bool        = True


# ──────────────────────────────────────────────
# Step 1: Load gap IDs
# ──────────────────────────────────────────────
def load_gap_ids() -> List[str]:
    """Return arxiv IDs in llm.txt that are absent from the CSV."""
    # IDs from llm.txt (via arxiv.org/abs/ URLs)
    text = LLM_TXT.read_text(encoding="utf-8")
    url_ids  = re.findall(r'arxiv\.org/abs/(\d{4}\.\d{4,5})', text)
    bare_ids = re.findall(r'(?<![/\w])(\d{4}\.\d{4,5})(?!\d)', text)
    llm_ids  = set(url_ids + bare_ids)

    # IDs already in CSV
    df = pd.read_csv(CSV_PATH)
    df["arxiv_id_clean"] = (
        df["arxiv_id"].astype(str)
        .str.replace(r'["\s]', "", regex=True)
    )
    csv_ids = set(df["arxiv_id_clean"].tolist())

    gap = sorted(llm_ids - csv_ids)
    print(f"llm.txt: {len(llm_ids)} IDs | CSV: {len(csv_ids)} IDs | Gap: {len(gap)}")
    return gap


# ──────────────────────────────────────────────
# Step 2: Fetch via arxiv package
# ──────────────────────────────────────────────
def fetch_arxiv(arxiv_id: str) -> Optional[dict]:
    """Return {'arxiv_id', 'title', 'abstract'} or None on persistent error.
    Uses Semantic Scholar API (export.arxiv.org is network-blocked in this env).
    Rate limit: ~100 req / 5 min unauthenticated — ARXIV_DELAY=5 keeps us safe.
    """
    url = f"https://api.semanticscholar.org/graph/v1/paper/arXiv:{arxiv_id}"
    params = {"fields": "title,abstract"}

    for attempt in range(ARXIV_MAX_RETRY):
        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 429:
                wait = ARXIV_BACKOFF * (attempt + 1)
                print(f"  [429] Rate limited on {arxiv_id}. Waiting {wait}s (attempt {attempt+1})")
                time.sleep(wait)
                continue
            if resp.status_code == 404:
                print(f"  [404] Not found on SemanticScholar: {arxiv_id}")
                return None
            resp.raise_for_status()
            data = resp.json()
            title    = (data.get("title")    or "").strip().replace("\n", " ")
            abstract = (data.get("abstract") or "").strip().replace("\n", " ")
            if not abstract:
                print(f"  [WARN] Empty abstract for {arxiv_id}")
                return None
            return {"arxiv_id": arxiv_id, "title": title, "abstract": abstract}

        except requests.RequestException as e:
            print(f"  [ERROR] fetch_arxiv({arxiv_id}) attempt {attempt+1}: {e}")
            time.sleep(ARXIV_BACKOFF)

    print(f"  [FAIL] Gave up on {arxiv_id} after {ARXIV_MAX_RETRY} attempts")
    return None


# ──────────────────────────────────────────────
# Step 3: LLM extraction
# ──────────────────────────────────────────────
def extract_fields(paper: dict) -> Optional[PaperAnalysis]:
    """Call gpt-4o via copilot-proxy and parse structured PaperAnalysis."""
    client = OpenAI(api_key=PROXY_KEY, base_url=PROXY_BASE)
    prompt = EXTRACT_PROMPT.format(abstract=paper["abstract"])

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            raw = resp.choices[0].message.content.strip()

            # Strip markdown fences if model added them
            raw = re.sub(r"^```[a-z]*\n?", "", raw)
            raw = re.sub(r"\n?```\s*$", "", raw.rstrip())

            data = json.loads(raw)
            return PaperAnalysis(
                arxiv_id=paper["arxiv_id"],
                title=paper["title"],
                abstract=paper["abstract"],
                utility=data.get("utility", []),
                barriers=data.get("barriers", []),
                thesis=data.get("thesis", ""),
                is_complete=True,
            )
        except json.JSONDecodeError as e:
            print(f"  [WARN] JSON parse failed (attempt {attempt+1}): {e}")
            print(f"  Raw: {raw[:200]}")
            if attempt < 2:
                time.sleep(2)
        except Exception as e:
            print(f"  [ERROR] extract_fields: {e}")
            if attempt < 2:
                time.sleep(2)

    # Return a failed sentinel so we still track the paper
    return PaperAnalysis(
        arxiv_id=paper["arxiv_id"],
        title=paper["title"],
        abstract=paper["abstract"],
        utility=["Validation Failed: Invalid Utility"],
        barriers=["Validation Failed: Invalid Barriers"],
        thesis="Validation Failed: Invalid Thesis",
        is_complete=False,
    )


# ──────────────────────────────────────────────
# Step 4: Append to CSV
# ──────────────────────────────────────────────
def append_to_csv(analyses: List[PaperAnalysis]) -> None:
    """Append new rows to the master CSV."""
    new_rows = []
    for a in analyses:
        new_rows.append({
            "arxiv_id":    a.arxiv_id,
            "title":       a.title,
            "abstract":    a.abstract,
            "utility":     json.dumps(a.utility),
            "barriers":    json.dumps(a.barriers),
            "thesis":      a.thesis,
            "is_complete": a.is_complete,
        })

    df_new = pd.DataFrame(new_rows)
    df_new.to_csv(CSV_PATH, mode="a", header=False, index=False)
    print(f"  Appended {len(new_rows)} rows to CSV.")


# ──────────────────────────────────────────────
# State: track processed IDs across runs
# ──────────────────────────────────────────────
def load_state() -> set:
    """Load set of already-processed IDs from state file."""
    if STATE_PATH.exists():
        try:
            data = json.loads(STATE_PATH.read_text(encoding="utf-8-sig"))
            return set(data.get("processed", []))
        except (json.JSONDecodeError, ValueError):
            print("[WARN] State file unreadable/empty — starting fresh.")
    return set()


def save_state(processed: set) -> None:
    STATE_PATH.write_text(json.dumps({"processed": sorted(processed)}, indent=2))


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main(dry_run: bool = False, limit: Optional[int] = None) -> None:
    gap_ids   = load_gap_ids()
    processed = load_state()

    # Filter already done
    pending = [i for i in gap_ids if i not in processed]
    if limit:
        pending = pending[:limit]

    print(f"Pending: {len(pending)} papers  (processed so far: {len(processed)})")

    if dry_run:
        print("-- DRY RUN -- nothing written")
        for i in pending[:10]:
            print(f"  {i}")
        return

    batch: List[PaperAnalysis] = []

    for arxiv_id in tqdm(pending, desc="Extracting"):
        # Fetch metadata
        paper = fetch_arxiv(arxiv_id)
        time.sleep(ARXIV_DELAY)

        if paper is None:
            # Mark as processed (failed fetch) so we don't retry forever
            processed.add(arxiv_id)
            save_state(processed)
            continue

        # Extract fields
        analysis = extract_fields(paper)
        time.sleep(LLM_DELAY)

        if analysis:
            batch.append(analysis)

        processed.add(arxiv_id)

        # Flush every 10 papers
        if len(batch) >= 10:
            append_to_csv(batch)
            batch.clear()
            save_state(processed)

    # Final flush
    if batch:
        append_to_csv(batch)
        save_state(processed)

    print(f"\nDone. Total processed this run: {len(pending)}")


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    limit   = None

    # Optional: --limit N to process only first N papers
    if "--limit" in sys.argv:
        idx = sys.argv.index("--limit")
        limit = int(sys.argv[idx + 1])

    main(dry_run=dry_run, limit=limit)
