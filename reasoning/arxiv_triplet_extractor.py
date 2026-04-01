"""
arxiv_triplet_extractor.py — Extract semantic SPO triplets from utility text via Ollama.

Core Thesis:
    Given a short utility description for an arXiv paper, ask a local LLM to extract
    Subject-Predicate-Object triplets that capture the key claims.  Results are cached
    per (arxiv_id, utility_hash) in SQLite so repeated queries cost nothing.

Workflow:
    1. Receive (arxiv_id, utility_text) pairs
    2. Cache-lookup: sha256(utility_text)[:16] → return cached if hit
    3. Call Ollama with structured JSON prompt → parse List[Triplet]
    4. Store in SQLite cache; return result

Necessary Conditions:
    - Ollama running locally (default http://localhost:11434)
    - Model capable of JSON output (e.g. llama3.2, qwen3)

Schema:
    triplets(arxiv_id TEXT, utility_hash TEXT, extracted_at TEXT, triplets_json TEXT,
             PRIMARY KEY (arxiv_id, utility_hash))
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import sqlite3
import sys
import pathlib
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import List, Dict, Optional

import ollama

_ROOT = pathlib.Path(__file__).parent.parent
_CACHE_PATH = _ROOT / "reasoning" / "triplet_cache.sqlite3"

DEFAULT_MODEL = "hf.co/unsloth/Qwen3-1.7B-GGUF:Qwen3-1.7B-Q6_K.gguf"

_EXTRACT_PROMPT = """\
Extract semantic subject-predicate-object (SPO) triplets from the research utility description below.

Rules:
- Return ONLY a JSON array of objects, no prose.
- Each object has exactly three keys: "subject", "predicate", "object"
- Use concise noun phrases for subject/object (3-6 words max)
- Use active-voice verb phrases for predicate (e.g. "enables", "improves", "requires", "produces", "demonstrates", "applies to")
- Extract 2-5 triplets; quality over quantity
- Focus on *causal* and *functional* relationships (what enables what, what requires what)

Utility text:
{utility}

JSON array only:"""


# ── Dataclasses ──────────────────────────────────────────────────────────────

@dataclass
class Triplet:
    subject: str
    predicate: str
    object: str
    arxiv_id: str = ""

    def as_premise(self) -> str:
        return f"{self.subject} {self.predicate} {self.object}"

    def as_dict(self) -> dict:
        return asdict(self)


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _init_cache(path: pathlib.Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS triplets (
            arxiv_id     TEXT NOT NULL,
            utility_hash TEXT NOT NULL,
            extracted_at TEXT NOT NULL,
            triplets_json TEXT NOT NULL,
            PRIMARY KEY (arxiv_id, utility_hash)
        )
    """)
    conn.commit()
    return conn


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _cache_get(conn: sqlite3.Connection, arxiv_id: str, utility_hash: str) -> Optional[List[Triplet]]:
    row = conn.execute(
        "SELECT triplets_json FROM triplets WHERE arxiv_id=? AND utility_hash=?",
        (arxiv_id, utility_hash)
    ).fetchone()
    if row is None:
        return None
    raw = json.loads(row[0])
    return [Triplet(arxiv_id=arxiv_id, **r) for r in raw]


def _cache_put(conn: sqlite3.Connection, arxiv_id: str, utility_hash: str,
               triplets: List[Triplet]) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO triplets (arxiv_id, utility_hash, extracted_at, triplets_json) "
        "VALUES (?, ?, ?, ?)",
        (arxiv_id, utility_hash,
         datetime.now(timezone.utc).isoformat(),
         json.dumps([{"subject": t.subject, "predicate": t.predicate,
                      "object": t.object} for t in triplets]))
    )
    conn.commit()


# ── Extraction ────────────────────────────────────────────────────────────────

def _parse_json_response(text: str, arxiv_id: str) -> List[Triplet]:
    """
    Parse LLM JSON output → List[Triplet].
    Handles: raw JSON array, markdown fenced blocks, <think> reasoning blocks.
    """
    # Strip <think>…</think> reasoning blocks (Qwen3 style)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'<think>.*', '', text, flags=re.DOTALL)  # unclosed think block
    text = re.sub(r'</?think>', '', text).strip()

    # Extract first JSON array
    match = re.search(r'\[.*?\]', text, re.DOTALL)
    if not match:
        return []
    try:
        raw = json.loads(match.group())
    except json.JSONDecodeError:
        return []

    triplets = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        s = str(item.get("subject", "")).strip()
        p = str(item.get("predicate", "")).strip()
        o = str(item.get("object", "")).strip()
        if s and p and o:
            triplets.append(Triplet(subject=s, predicate=p, object=o, arxiv_id=arxiv_id))
    return triplets


def extract_triplets(
    arxiv_id: str,
    utility_text: str,
    conn: sqlite3.Connection,
    model: str = DEFAULT_MODEL,
    verbose: bool = False,
) -> List[Triplet]:
    """
    Extract SPO triplets from a single utility text.
    Cache-first: returns cached result if available.

    Args:
        arxiv_id: Paper identifier (used as cache key + triplet label)
        utility_text: Short utility description for the paper
        conn: Open SQLite connection to triplet cache
        model: Ollama model name
        verbose: Print debug info

    Returns:
        List[Triplet] — may be empty if extraction fails
    """
    if not utility_text or not utility_text.strip():
        return []

    util_hash = _hash(utility_text)

    # Cache hit
    cached = _cache_get(conn, arxiv_id, util_hash)
    if cached is not None:
        if verbose:
            print(f"  [cache] {arxiv_id}: {len(cached)} triplets")
        return cached

    # LLM call — think blocks are stripped in _parse_json_response.
    prompt = _EXTRACT_PROMPT.format(utility=utility_text.strip())
    try:
        _client = ollama.Client(host="http://127.0.0.1:11434")
        response = _client.chat(
            model=model,
            messages=[
                {"role": "user", "content": prompt},
            ],
            options={"temperature": 0.0, "num_predict": 512},
        )
        raw_text = response["message"]["content"]
    except Exception as exc:
        if verbose:
            print(f"  [error] {arxiv_id}: Ollama call failed — {exc}", file=sys.stderr)
        return []

    triplets = _parse_json_response(raw_text, arxiv_id)

    if verbose:
        print(f"  [extract] {arxiv_id}: {len(triplets)} triplets")
        for t in triplets:
            print(f"    {t.subject} | {t.predicate} | {t.object}")

    _cache_put(conn, arxiv_id, util_hash, triplets)
    return triplets


def extract_batch(
    papers: List[Dict],  # List of {"arxiv_id": str, "utility": str}
    conn: sqlite3.Connection,
    model: str = DEFAULT_MODEL,
    verbose: bool = False,
) -> Dict[str, List[Triplet]]:
    """
    Extract triplets for a batch of papers.
    Returns dict mapping arxiv_id → List[Triplet].

    Args:
        papers: List of dicts with 'arxiv_id' and 'utility' keys
        conn: Open SQLite connection to triplet cache
        model: Ollama model name
        verbose: Print progress

    Returns:
        Dict[str, List[Triplet]]
    """
    results: Dict[str, List[Triplet]] = {}
    for paper in papers:
        arxiv_id = paper["arxiv_id"]
        utility  = paper.get("utility", "")
        results[arxiv_id] = extract_triplets(arxiv_id, utility, conn, model, verbose)
    return results


# ── Smoke test ────────────────────────────────────────────────────────────────

def _smoke(n: int = 3, model: str = DEFAULT_MODEL) -> None:
    """Load n papers from cleaned CSV and extract triplets, print results."""
    csv_path = _ROOT / "papers" / "post_processed" / "arxiv_data_with_analysis_cleaned.csv"
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    import csv
    papers = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= n:
                break
            if row.get("utility"):
                papers.append({"arxiv_id": row["arxiv_id"], "utility": row["utility"]})

    conn = _init_cache(_CACHE_PATH)
    try:
        print(f"\nSmoke test: {len(papers)} papers, model={model}\n{'─'*60}")
        results = extract_batch(papers, conn, model=model, verbose=True)
        print(f"\n{'─'*60}")
        print(f"Results: {sum(len(v) for v in results.values())} triplets across {len(results)} papers")
    finally:
        conn.close()


# ── Class wrapper (thin facade around module-level functions) ─────────────────

class ArxivTripletExtractor:
    """
    Stateful triplet extractor that holds an open SQLite connection.

    Usage:
        extractor = ArxivTripletExtractor()
        triplets  = extractor.extract("2301.00001", "text about the utility...")
        batch     = extractor.extract_batch([{"arxiv_id": ..., "utility": ...}])
        extractor.close()

    Or as a context manager:
        with ArxivTripletExtractor() as ext:
            ...
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        cache_path: pathlib.Path = _CACHE_PATH,
        verbose: bool = False,
    ):
        self.model   = model
        self.verbose = verbose
        self._conn   = _init_cache(cache_path)

    def extract(self, arxiv_id: str, utility_text: str) -> List[Triplet]:
        return extract_triplets(arxiv_id, utility_text, self._conn, self.model, self.verbose)

    def extract_batch(self, papers: List[Dict]) -> Dict[str, List[Triplet]]:
        return extract_batch(papers, self._conn, self.model, self.verbose)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ArXiv triplet extractor — smoke test")
    parser.add_argument("--smoke", type=int, default=3, metavar="N",
                        help="Test on N papers from cleaned CSV (default 3)")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()
    _smoke(n=args.smoke, model=args.model)
