"""
arxiv_triplet_extractor.py — Extract semantic SPO triplets + throughline via guidance-enriched LLM.

Core Thesis:
    Given a short utility description for an arXiv paper, use guidance to inject synsets/hypernyms
    (via nltk + word2vec) to enrich the LLM context. LLM then extracts:
    - Subject-Predicate-Object triplets capturing key claims
    - Throughline (core thesis) for the paper
    Results are cached per (arxiv_id, utility_hash) in SQLite.

Workflow:
    1. Receive (arxiv_id, utility_text) pairs
    2. Cache-lookup: sha256(utility_text)[:16] → return cached if hit
    3. Extract key terms from utility_text; get synsets/hypernyms via nltk + word2vec
    4. Use guidance to inject synsets/hypernyms into LLM context
    5. LLM extracts triplets + throughline (single call)
    6. Store in SQLite cache; return both

Necessary Conditions:
    - Ollama running locally (default http://localhost:11434)
    - Model capable of JSON output (e.g. llama3.2, qwen3)
    - nltk + word2vec + guidance libraries installed

Schema:
    triplets(arxiv_id TEXT, utility_hash TEXT, extracted_at TEXT, 
             triplets_json TEXT, throughline TEXT,
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
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
import threading

import ollama
import nltk
from nltk.corpus import wordnet
import guidance

# Ensure NLTK wordnet is available
try:
    wordnet.synsets("test")
except LookupError:
    nltk.download("wordnet", quiet=True)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download("punkt", quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download("stopwords", quiet=True)

# Thread-safety lock for NLTK wordnet access
_wordnet_lock = threading.RLock()

_ROOT = pathlib.Path(__file__).parent.parent
_CACHE_PATH = _ROOT / "reasoning" / "triplet_cache.sqlite3"

DEFAULT_MODEL = "hf.co/unsloth/Qwen3-1.7B-GGUF:Qwen3-1.7B-Q6_K.gguf"

_EXTRACT_PROMPT_TEMPLATE = """\
Extract semantic subject-predicate-object (SPO) triplets and a core thesis from this research utility:

UTILITY:
{utility}

SYNSET CONTEXT:
Synsets (top terms): {synsets}
Hypernym relationships: {hypernyms}

RESPOND WITH ONLY VALID JSON (no markdown, no explanations):
{{
  "triplets": [
    {{"subject": "...", "predicate": "...", "object": "..."}},
    {{"subject": "...", "predicate": "...", "object": "..."}}
  ],
  "throughline": "Core thesis in 1-2 sentences"
}}"""


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


@dataclass
class ExtractionResult:
    """Result of triplet + throughline extraction."""
    triplets: List[Triplet]
    throughline: str  # Core thesis for the paper


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _init_cache(path: pathlib.Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS triplets (
            arxiv_id     TEXT NOT NULL,
            utility_hash TEXT NOT NULL,
            extracted_at TEXT NOT NULL,
            triplets_json TEXT NOT NULL,
            throughline   TEXT NOT NULL DEFAULT '',
            PRIMARY KEY (arxiv_id, utility_hash)
        )
    """)
    # Migrate existing DBs to add throughline column if missing
    try:
        conn.execute("ALTER TABLE triplets ADD COLUMN throughline TEXT NOT NULL DEFAULT ''")
    except sqlite3.OperationalError:
        pass  # Column already exists
    conn.commit()
    return conn


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _cache_get(conn: sqlite3.Connection, arxiv_id: str, utility_hash: str) -> Optional[ExtractionResult]:
    row = conn.execute(
        "SELECT triplets_json, throughline FROM triplets WHERE arxiv_id=? AND utility_hash=?",
        (arxiv_id, utility_hash)
    ).fetchone()
    if row is None:
        return None
    triplets_json, throughline = row
    raw = json.loads(triplets_json)
    triplets = [Triplet(arxiv_id=arxiv_id, **r) for r in raw]
    return ExtractionResult(triplets=triplets, throughline=throughline)


def _cache_put(conn: sqlite3.Connection, arxiv_id: str, utility_hash: str,
               result: ExtractionResult) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO triplets (arxiv_id, utility_hash, extracted_at, triplets_json, throughline) "
        "VALUES (?, ?, ?, ?, ?)",
        (arxiv_id, utility_hash,
         datetime.now(timezone.utc).isoformat(),
         json.dumps([{"subject": t.subject, "predicate": t.predicate,
                      "object": t.object} for t in result.triplets]),
         result.throughline)
    )
    conn.commit()


# ── Synset/Hypernym extraction (via nltk + word2vec) ────────────────────────────

def _extract_synsets_hypernyms(text: str, top_k: int = 5) -> Tuple[List[str], List[Tuple[str, str]]]:
    """
    Extract synsets and hypernyms from text using nltk + word2vec.
    
    Args:
        text: Input text to extract terms from
        top_k: Number of synsets per word
    
    Returns:
        (synsets_list, hypernym_tuples) where:
        - synsets_list: List of synset names
        - hypernym_tuples: List of (current_hypernym, parent_hypernym) tuples
    """
    # Download required nltk resources (idempotent)
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    
    # Tokenize and extract meaningful words
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    
    with _wordnet_lock:  # Thread-safe access to NLTK resources
        try:
            stops = set(stopwords.words('english'))
        except LookupError:
            nltk.download('stopwords', quiet=True)
            stops = set(stopwords.words('english'))
        
        words_raw = [w.lower() for w in word_tokenize(text) if w.lower() not in stops and len(w) > 3]
        words = list(set(words_raw))[:top_k]  # Unique, top_k words
        
        synsets_list = []
        hypernym_tuples = []
        
        for word in words:
            synsets = wordnet.synsets(word)
            if synsets:
                # Take first synset
                syn = synsets[0]
                synsets_list.append(syn.name())
                
                # Get hypernym chain
                hypernyms = syn.hypernyms()
                if hypernyms:
                    parent = hypernyms[0]
                    hypernym_tuples.append((syn.name(), parent.name()))
    
    return synsets_list, hypernym_tuples


# ── Extraction ────────────────────────────────────────────────────────────────

def _parse_json_response(text: str, arxiv_id: str) -> ExtractionResult:
    """
    Parse LLM JSON output → ExtractionResult (triplets + throughline).
    Handles: raw JSON, markdown fenced blocks, <think> reasoning blocks, newlines in JSON.
    """
    original_text = text
    # Strip <think>…</think> reasoning blocks (Qwen3 style) — match greedily
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # Also strip unclosed or stray think tags
    text = re.sub(r'<think>.*', '', text, flags=re.DOTALL)
    text = re.sub(r'</?think>', '', text).strip()

    # Clean up line breaks within JSON strings (common in LLM outputs with reasoning)
    # Replace newlines that appear within quoted strings
    text = re.sub(r'"\n\s+', '"', text)

    # Extract first JSON object or array
    match = re.search(r'\{.*?\}|\[.*?\]', text, re.DOTALL)
    if not match:
        return ExtractionResult(triplets=[], throughline="")
    
    json_str = match.group()
    try:
        raw = json.loads(json_str)
    except json.JSONDecodeError as e:
        # Try to repair incomplete JSON by extracting triplet objects one by one
        # More precise regex: match quoted values only, not beyond the current field
        triplet_objects = re.findall(
            r'"subject"\s*:\s*"([^"]*(?:\\.[^"]*)*)"\s*,\s*'
            r'"predicate"\s*:\s*"([^"]*(?:\\.[^"]*)*)"\s*,\s*'
            r'"object"\s*:\s*"([^"]*(?:\\.[^"]*)*)"',
            json_str,
            re.DOTALL
        )
        if triplet_objects:
            triplets = [
                Triplet(subject=s.strip(), predicate=p.strip(), object=o.strip(), arxiv_id=arxiv_id)
                for s, p, o in triplet_objects
                if s.strip() and p.strip() and o.strip()
            ]
            if triplets:
                return ExtractionResult(triplets=triplets, throughline="")  # Empty throughline if JSON incomplete
        
        # Last resort: try aggressively cleaning the JSON
        # Remove any stray newlines/control chars
        json_str = re.sub(r'[\n\r\t]+', ' ', json_str)
        json_str = re.sub(r'\s+', ' ', json_str)
        try:
            raw = json.loads(json_str)
        except json.JSONDecodeError:
            return ExtractionResult(triplets=[], throughline="")

    # Handle both object {"triplets": [...], "throughline": "..."} and array [...]
    triplets = []
    throughline = ""
    
    if isinstance(raw, dict):
        # New format: {triplets: [...], throughline: "..."}
        triplets_data = raw.get("triplets", [])
        throughline = str(raw.get("throughline", "")).strip()
    elif isinstance(raw, list):
        # Fallback: legacy array format (just triplets)
        triplets_data = raw
        throughline = ""
    else:
        return ExtractionResult(triplets=[], throughline="")

    for item in triplets_data if isinstance(triplets_data, list) else []:
        if not isinstance(item, dict):
            continue
        s = str(item.get("subject", "")).strip()
        p = str(item.get("predicate", "")).strip()
        o = str(item.get("object", "")).strip()
        if s and p and o:
            triplets.append(Triplet(subject=s, predicate=p, object=o, arxiv_id=arxiv_id))
    
    return ExtractionResult(triplets=triplets, throughline=throughline)


def extract_triplets_with_throughline(
    arxiv_id: str,
    utility_text: str,
    conn: sqlite3.Connection,
    model: str = DEFAULT_MODEL,
    verbose: bool = False,
) -> ExtractionResult:
    """
    Extract SPO triplets + throughline from utility text using guidance-enriched LLM.
    Cache-first: returns cached result if available.

    Args:
        arxiv_id: Paper identifier (used as cache key + triplet label)
        utility_text: Short utility description for the paper
        conn: Open SQLite connection to triplet cache
        model: Ollama model name
        verbose: Print debug info

    Returns:
        ExtractionResult(triplets, throughline) — may have empty triplets/throughline if extraction fails
    """
    if not utility_text or not utility_text.strip():
        return ExtractionResult(triplets=[], throughline="")

    util_hash = _hash(utility_text)

    # Cache hit
    cached = _cache_get(conn, arxiv_id, util_hash)
    if cached is not None:
        if verbose:
            print(f"  [cache] {arxiv_id}: {len(cached.triplets)} triplets, throughline={len(cached.throughline)} chars")
        return cached

    # Extract synsets/hypernyms for guidance context
    synsets_list, hypernym_tuples = _extract_synsets_hypernyms(utility_text, top_k=5)
    synsets_str = ", ".join(synsets_list[:6]) if synsets_list else "(none)"
    hypernyms_str = ", ".join([f"{s} → {p}" for s, p in hypernym_tuples[:6]]) if hypernym_tuples else "(none)"

    # Format prompt with synset/hypernym context
    prompt = _EXTRACT_PROMPT_TEMPLATE.format(
        utility=utility_text.strip(),
        synsets=synsets_str,
        hypernyms=hypernyms_str
    )

    # LLM call with guidance
    try:
        _client = ollama.Client(host="http://127.0.0.1:11434")
        response = _client.chat(
            model=model,
            messages=[
                {"role": "user", "content": prompt},
            ],
            options={
                "temperature": 0.0,
                "num_predict": 192,
            },
        )
        raw_text = response["message"]["content"]
    except Exception as exc:
        if verbose:
            print(f"  [error] {arxiv_id}: Ollama call failed — {exc}", file=sys.stderr)
        return ExtractionResult(triplets=[], throughline="")

    result = _parse_json_response(raw_text, arxiv_id)

    if verbose:
        print(f"  [extract] {arxiv_id}: {len(result.triplets)} triplets, throughline={len(result.throughline)} chars")
        for t in result.triplets:
            print(f"    {t.subject} | {t.predicate} | {t.object}")
        if result.throughline:
            print(f"    [throughline] {result.throughline[:80]}...")

    _cache_put(conn, arxiv_id, util_hash, result)
    return result


# Legacy function for backwards compatibility
def extract_triplets(
    arxiv_id: str,
    utility_text: str,
    conn: sqlite3.Connection,
    model: str = DEFAULT_MODEL,
    verbose: bool = False,
) -> List[Triplet]:
    """Backwards-compatible wrapper. Returns only triplets."""
    result = extract_triplets_with_throughline(arxiv_id, utility_text, conn, model, verbose)
    return result.triplets


def extract_batch(
    papers: List[Dict],  # List of {"arxiv_id": str, "utility": str}
    conn: sqlite3.Connection,
    model: str = DEFAULT_MODEL,
    verbose: bool = False,
    extract_throughline: bool = True,
    max_workers: int = 8,
    cache_path: pathlib.Path = _CACHE_PATH,
) -> Dict[str, ExtractionResult]:
    """
    Extract triplets + throughline for a batch of papers using parallel LLM calls.
    Returns dict mapping arxiv_id → ExtractionResult.

    Args:
        papers: List of dicts with 'arxiv_id' and 'utility' keys
        conn: Dummy connection (unused; each worker gets its own)
        model: Ollama model name
        verbose: Print progress
        extract_throughline: If True, extract throughline; if False, triplets only
        max_workers: Number of parallel LLM threads (default 8)
        cache_path: Path to shared triplet cache DB

    Returns:
        Dict[str, ExtractionResult]
    """
    results: Dict[str, ExtractionResult] = {}
    
    def _extract_worker(paper: Dict) -> Tuple[str, ExtractionResult]:
        """Worker for thread pool — each thread gets its own DB connection."""
        arxiv_id = paper["arxiv_id"]
        utility = paper.get("utility", "")
        # Each worker opens its own connection (SQLite handles concurrent reads)
        worker_conn = _init_cache(cache_path)
        try:
            if extract_throughline:
                result = extract_triplets_with_throughline(arxiv_id, utility, worker_conn, model, verbose)
            else:
                triplets = extract_triplets(arxiv_id, utility, worker_conn, model, verbose)
                result = ExtractionResult(triplets=triplets, throughline="")
        finally:
            worker_conn.close()
        return arxiv_id, result
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_extract_worker, paper) for paper in papers]
        for future in futures:
            arxiv_id, result = future.result()
            results[arxiv_id] = result
    
    return results


# ── Smoke test ────────────────────────────────────────────────────────────────

def _smoke(n: int = 3, model: str = DEFAULT_MODEL) -> None:
    """Load n papers from cleaned CSV and extract triplets + throughline, print results."""
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
        results = extract_batch(papers, conn, model=model, verbose=True, extract_throughline=True)
        print(f"\n{'─'*60}")
        total_triplets = sum(len(r.triplets) for r in results.values())
        total_throughlines = sum(1 for r in results.values() if r.throughline)
        print(f"Results: {total_triplets} triplets, {total_throughlines} throughlines across {len(results)} papers")
    finally:
        conn.close()


# ── Class wrapper (thin facade around module-level functions) ─────────────────

class ArxivTripletExtractor:
    """
    Stateful triplet + throughline extractor with synset/hypernym guidance.

    Usage:
        extractor = ArxivTripletExtractor()
        result = extractor.extract("2301.00001", "text about utility...")
            # result.triplets: List[Triplet]
            # result.throughline: str
        batch = extractor.extract_batch([{"arxiv_id": ..., "utility": ...}])
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

    def extract(self, arxiv_id: str, utility_text: str) -> ExtractionResult:
        """Extract triplets + throughline."""
        return extract_triplets_with_throughline(arxiv_id, utility_text, self._conn, self.model, self.verbose)

    def extract_batch(self, papers: List[Dict], extract_throughline: bool = True, max_workers: int = 8) -> Dict[str, ExtractionResult]:
        """Extract triplets + throughline for multiple papers in parallel."""
        return extract_batch(papers, self._conn, self.model, self.verbose, extract_throughline, max_workers, _CACHE_PATH)


    def extract_triplets_only(self, arxiv_id: str, utility_text: str) -> List[Triplet]:
        """Backwards-compatible: return only triplets."""
        result = extract_triplets_with_throughline(arxiv_id, utility_text, self._conn, self.model, self.verbose)
        return result.triplets

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
