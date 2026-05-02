#!/usr/bin/env python3
"""
arXiv Bridge — upstream semantic search layer for the arxiv_rag retrieval stack.

Extends local CSV retrieval with upstream paper discovery via:
  1. Semantic Scholar Graph API  (semantic ranking, arXiv IDs)
  2. arXiv Atom API fallback     (BM25 keyword, title/abstract field scoping)

For novel papers (not in local corpus), utility/barriers/thesis are derived
live via LLM (gpt-4o via copilot-proxy) and cached so each paper is scored
at most once. Only top-N uncached papers trigger LLM calls.

Preconditions:
    - requests, openai installed in the Python environment
    - papers/post_processed/arxiv_data_with_analysis_cleaned.csv reachable
    - checkpoints/ directory writable (for utility cache DB)
    - copilot-proxy running at localhost:8069
Postconditions:
    - Returns BridgeResult list sorted by relevance (upstream rank position)
    - local=True papers carry utility/barriers/thesis from local CSV
    - local=False papers: top top_n_derive have LLM-derived fields; rest have []/"" placeholders
Failure modes:
    - S2 API unavailable/rate-limited → falls back to arXiv Atom API
    - Both APIs fail → returns empty list with warning
    - LLM call fails → result carries is_complete=False, utility=["Extraction failed"]
"""

import csv
import json
import re
import sqlite3
import sys
import time
import xml.etree.ElementTree as ET
import os
from dataclasses import dataclass, field as dc_field
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests
from openai import OpenAI

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_ROOT     = Path(__file__).resolve().parent
_CSV      = _ROOT / "papers" / "post_processed" / "arxiv_data_with_analysis_cleaned.csv"
_CACHE_DB = _ROOT / "checkpoints" / "utility_cache.db"

S2_SEARCH  = "https://api.semanticscholar.org/graph/v1/paper/search"
ARXIV_API  = "http://export.arxiv.org/api/query"
S2_FIELDS  = "title,abstract,externalIds,year"
_ATOM_NS   = "http://www.w3.org/2005/Atom"

PROXY_BASE = "http://127.0.0.1:8069/v1"
PROXY_KEY  = "dummy-key"
LLM_MODEL  = "gpt-4o"

_S2_KEY = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")

# Same extraction prompt used in arxiv_pipeline/arxiv_gap_extractor.py
_EXTRACT_PROMPT = """\
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


@dataclass
class BridgeResult:
    arxiv_id: str
    title: str
    abstract: str
    year: Optional[int]
    relevance_score: float      # 0–1, linear decay from upstream rank position
    source: str                 # "local", "s2", "arxiv_api", "cached"
    local: bool                 # True if already in local CSV
    utility: list = dc_field(default_factory=list)    # list[str], from CSV or LLM
    barriers: list = dc_field(default_factory=list)   # list[str]
    thesis: str = ""
    is_complete: bool = False   # True once utility has been LLM-derived or loaded from CSV
    pdf_url: str = ""


def _norm_id(aid: str) -> str:
    """Normalise to dot form without version: '2601_09113v2' → '2601.09113'."""
    aid = aid.strip()
    aid = re.sub(r"v\d+$", "", aid)
    aid = re.sub(r"^(\d{4})_(\d)", r"\1.\2", aid)
    return aid


def _load_local_ids() -> dict[str, dict]:
    """Load local CSV as {norm_arxiv_id: row_dict}."""
    if not _CSV.is_file():
        return {}
    out = {}
    with open(_CSV, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            aid = _norm_id(row.get("arxiv_id", "").strip().strip('"'))
            if aid:
                out[aid] = row
    return out


# ── SQLite utility cache ───────────────────────────────────────────────────────

def _open_cache() -> sqlite3.Connection:
    """Open (and migrate if needed) the utility cache DB."""
    _CACHE_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_CACHE_DB), timeout=10)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS utility_cache (
            arxiv_id     TEXT PRIMARY KEY,
            title        TEXT,
            utility_json TEXT,
            barriers_json TEXT,
            thesis       TEXT,
            is_complete  INTEGER DEFAULT 0,
            cached_at    TEXT
        )
    """)
    # Migrate old schema (utility REAL column) if present
    cols = {r[1] for r in conn.execute("PRAGMA table_info(utility_cache)")}
    if "utility" in cols and "utility_json" not in cols:
        conn.execute("ALTER TABLE utility_cache RENAME TO _utility_cache_old")
        conn.execute("""
            CREATE TABLE utility_cache (
                arxiv_id     TEXT PRIMARY KEY,
                title        TEXT,
                utility_json TEXT,
                barriers_json TEXT,
                thesis       TEXT,
                is_complete  INTEGER DEFAULT 0,
                cached_at    TEXT
            )
        """)
        conn.execute("DROP TABLE _utility_cache_old")
    conn.commit()
    return conn


def _get_cached(arxiv_id: str, conn: sqlite3.Connection) -> Optional[dict]:
    """Return cached fields dict or None."""
    row = conn.execute(
        "SELECT utility_json, barriers_json, thesis, is_complete "
        "FROM utility_cache WHERE arxiv_id = ?",
        (_norm_id(arxiv_id),)
    ).fetchone()
    if not row:
        return None
    try:
        utility  = json.loads(row[0]) if row[0] else []
        barriers = json.loads(row[1]) if row[1] else []
    except json.JSONDecodeError:
        utility, barriers = [], []
    return {
        "utility": utility,
        "barriers": barriers,
        "thesis": row[2] or "",
        "is_complete": bool(row[3]),
    }


def _write_cache(arxiv_id: str, title: str, utility: list, barriers: list,
                 thesis: str, is_complete: bool, conn: sqlite3.Connection) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO utility_cache "
        "(arxiv_id, title, utility_json, barriers_json, thesis, is_complete, cached_at) "
        "VALUES (?,?,?,?,?,?,?)",
        (
            _norm_id(arxiv_id), title,
            json.dumps(utility), json.dumps(barriers),
            thesis, int(is_complete),
            datetime.utcnow().isoformat(),
        )
    )
    conn.commit()


# ── LLM utility derivation ────────────────────────────────────────────────────

def _salvage_json(raw: str) -> Optional[dict]:
    """Try to recover a truncated JSON object from LLM output."""
    for suffix in ("}", '"}', '"]}', '"}]}'):
        try:
            d = json.loads(raw + suffix)
            if isinstance(d, dict):
                return d
        except json.JSONDecodeError:
            pass
    last = raw.rfind("}")
    if last > 0:
        try:
            d = json.loads(raw[:last + 1])
            if isinstance(d, dict):
                return d
        except json.JSONDecodeError:
            pass
    return None


def _derive_utility(arxiv_id: str, title: str, abstract: str,
                    conn: sqlite3.Connection, verbose: bool = True) -> dict:
    """
    Call gpt-4o via copilot-proxy to extract utility/barriers/thesis from abstract.

    Preconditions: abstract is non-empty, proxy running at PROXY_BASE.
    Postconditions: result written to cache; returned dict always has all three keys.
    Failure modes: LLM error → returns sentinel with is_complete=False, cached.
    """
    client = OpenAI(api_key=PROXY_KEY, base_url=PROXY_BASE)
    prompt = _EXTRACT_PROMPT.format(abstract=abstract)

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            raw = resp.choices[0].message.content.strip()
            raw = re.sub(r"^```[a-z]*\n?", "", raw)
            raw = re.sub(r"\n?```\s*$", "", raw.rstrip())

            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                data = _salvage_json(raw)
                if data:
                    if verbose:
                        print(f"    [bridge] Salvaged partial JSON for {arxiv_id}")
                else:
                    raise ValueError("JSON parse failed and salvage returned None")

            utility  = data.get("utility",  [])
            barriers = data.get("barriers", [])
            thesis   = data.get("thesis",   "")
            if not isinstance(utility, list):
                utility = [str(utility)]
            if not isinstance(barriers, list):
                barriers = [str(barriers)]
            result = {"utility": utility, "barriers": barriers,
                      "thesis": thesis, "is_complete": True}
            _write_cache(arxiv_id, title, utility, barriers, thesis, True, conn)
            return result

        except Exception as exc:
            if verbose:
                print(f"    [bridge] LLM derive attempt {attempt+1} failed: {exc}")
            if attempt < 2:
                time.sleep(2)

    # Sentinel on total failure
    sentinel = {"utility": ["Extraction failed"], "barriers": [],
                "thesis": "", "is_complete": False}
    _write_cache(arxiv_id, title, sentinel["utility"], [], "", False, conn)
    return sentinel


# ── Semantic Scholar search ───────────────────────────────────────────────────

def _search_s2(query: str, limit: int = 20) -> list[dict]:
    """
    Query Semantic Scholar paper search.

    Require: query non-empty, limit 1-100.
    Failure modes: HTTP errors / rate-limit → raises requests.HTTPError.
    """
    headers = {"x-api-key": _S2_KEY} if _S2_KEY else {}
    params  = {"query": query, "limit": min(limit, 100), "fields": S2_FIELDS}
    resp = requests.get(S2_SEARCH, params=params, headers=headers, timeout=15)
    resp.raise_for_status()
    return resp.json().get("data", [])


def _s2_to_candidates(papers: list[dict]) -> list[dict]:
    out = []
    for i, p in enumerate(papers):
        ext = p.get("externalIds") or {}
        aid = ext.get("ArXiv")
        if not aid:
            continue
        norm = _norm_id(aid)
        out.append({
            "arxiv_id":      norm,
            "title":         p.get("title")    or "",
            "abstract":      p.get("abstract") or "",
            "year":          p.get("year"),
            "rank_position": i,
            "source":        "s2",
            "pdf_url":       f"https://arxiv.org/pdf/{norm}.pdf",
        })
    return out


# ── arXiv Atom API fallback ───────────────────────────────────────────────────

def _search_arxiv_api(query: str, limit: int = 20) -> list[dict]:
    """
    Query arXiv Atom API with ti+abs field scope.

    Failure modes: HTTP errors → raises requests.HTTPError.
    """
    terms = [t.strip() for t in query.split() if t.strip()]
    if not terms:
        return []
    clauses = [f"(ti:{t}+OR+abs:{t})" for t in terms]
    search_q = "+AND+".join(clauses)
    url = f"{ARXIV_API}?search_query={search_q}&max_results={limit}&sortBy=relevance"
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    return _parse_atom(resp.text)


def _parse_atom(xml_text: str) -> list[dict]:
    root = ET.fromstring(xml_text)
    ns   = {"a": _ATOM_NS}
    out  = []
    for i, entry in enumerate(root.findall("a:entry", ns)):
        raw_id = (entry.findtext("a:id", "", ns) or "").strip()
        m = re.search(r"abs/(\S+)", raw_id)
        if not m:
            continue
        aid      = _norm_id(m.group(1))
        title    = (entry.findtext("a:title",   "", ns) or "").strip().replace("\n", " ")
        abstract = (entry.findtext("a:summary", "", ns) or "").strip().replace("\n", " ")
        year_str = entry.findtext("a:published", "", ns) or ""
        year     = int(year_str[:4]) if year_str else None
        pdf_url  = ""
        for link in entry.findall("a:link", ns):
            if link.get("title") == "pdf":
                pdf_url = link.get("href", "")
        out.append({
            "arxiv_id": aid, "title": title, "abstract": abstract,
            "year": year, "rank_position": i,
            "source": "arxiv_api", "pdf_url": pdf_url,
        })
    return out


# ── Merge + score ─────────────────────────────────────────────────────────────

def _rank_to_relevance(rank: int, total: int) -> float:
    """Linear decay: rank 0 → 1.0, rank total-1 → 0.1."""
    if total <= 1:
        return 1.0
    return max(0.1, 1.0 - (rank / max(total - 1, 1)) * 0.9)


def bridge_search(
    query: str,
    limit: int = 20,
    top_n_derive: int = 10,
    verbose: bool = True,
) -> list[BridgeResult]:
    """
    Search for arXiv papers matching query, merging upstream + local sources.

    Preconditions:
        query is non-empty. limit is max candidates per upstream source.
        top_n_derive: max novel papers to call LLM for (0 = skip LLM entirely).
    Postconditions:
        Returns BridgeResult list sorted descending by relevance_score.
        local=True → utility/barriers/thesis from local CSV.
        local=False, rank < top_n_derive → LLM-derived (or cached).
        local=False, rank >= top_n_derive → utility=[], is_complete=False.
    Failure modes:
        S2 unavailable → falls back to arXiv Atom API.
        Both fail → returns [].
        LLM failure → sentinel result with is_complete=False.
    """
    local_ids = _load_local_ids()
    conn      = _open_cache()

    # ── 1. Upstream search ────────────────────────────────────────────────────
    candidates: list[dict] = []

    try:
        if verbose:
            print("  [bridge] Querying Semantic Scholar...", flush=True)
        s2_papers  = _search_s2(query, limit=limit)
        candidates = _s2_to_candidates(s2_papers)
        if verbose:
            print(f"  [bridge] S2 returned {len(s2_papers)} papers, "
                  f"{len(candidates)} with arXiv IDs")
        time.sleep(1.0)
    except Exception as exc:
        if verbose:
            print(f"  [bridge] S2 failed ({exc}), falling back to arXiv API...")
        try:
            candidates = _search_arxiv_api(query, limit=limit)
            if verbose:
                print(f"  [bridge] arXiv API returned {len(candidates)} results")
            time.sleep(3.0)
        except Exception as exc2:
            print(f"  [bridge] arXiv API also failed: {exc2}")
            conn.close()
            return []

    if not candidates:
        if verbose:
            print("  [bridge] No upstream results.")
        conn.close()
        return []

    # ── 2. Build results + derive utility for novel papers ────────────────────
    seen: set[str] = set()
    results: list[BridgeResult] = []
    novel_derive_count = 0
    total = len(candidates)

    for c in candidates:
        aid = c["arxiv_id"]
        if aid in seen:
            continue
        seen.add(aid)

        is_local    = aid in local_ids
        relevance   = _rank_to_relevance(c["rank_position"], total)
        utility: list[str]   = []
        barriers: list[str]  = []
        thesis      = ""
        is_complete = False
        source_label = c["source"]

        if is_local:
            row = local_ids[aid]
            try:
                utility  = json.loads(row.get("utility",  "[]") or "[]")
            except (json.JSONDecodeError, TypeError):
                utility  = []
            try:
                barriers = json.loads(row.get("barriers", "[]") or "[]")
            except (json.JSONDecodeError, TypeError):
                barriers = []
            thesis      = row.get("thesis", "") or ""
            is_complete = True
            source_label = "local"

        else:
            # Check cache first
            cached = _get_cached(aid, conn)
            if cached:
                utility     = cached["utility"]
                barriers    = cached["barriers"]
                thesis      = cached["thesis"]
                is_complete = cached["is_complete"]
                source_label = "cached"

            elif top_n_derive > 0 and novel_derive_count < top_n_derive:
                # Derive live via LLM
                if verbose:
                    print(f"  [bridge] Deriving utility for {aid} "
                          f"({novel_derive_count+1}/{top_n_derive})...", flush=True)
                derived = _derive_utility(
                    aid, c["title"], c["abstract"], conn, verbose=verbose
                )
                utility     = derived["utility"]
                barriers    = derived["barriers"]
                thesis      = derived["thesis"]
                is_complete = derived["is_complete"]
                novel_derive_count += 1
                time.sleep(0.5)   # polite delay between LLM calls

        results.append(BridgeResult(
            arxiv_id        = aid,
            title           = c["title"],
            abstract        = c["abstract"],
            year            = c.get("year"),
            relevance_score = round(relevance, 3),
            source          = source_label,
            local           = is_local,
            utility         = utility,
            barriers        = barriers,
            thesis          = thesis,
            is_complete     = is_complete,
            pdf_url         = c.get("pdf_url", ""),
        ))

    conn.close()
    results.sort(key=lambda r: r.relevance_score, reverse=True)
    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def _cli() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="arXiv bridge — upstream semantic search + live utility")
    ap.add_argument("query", nargs="+", help="Search query")
    ap.add_argument("--limit",      type=int, default=20,
                    help="Max candidates per upstream source (default 20)")
    ap.add_argument("--top-k",      type=int, default=10,
                    help="Top results to display (default 10)")
    ap.add_argument("--derive",     type=int, default=5,
                    help="Max novel papers to derive utility for via LLM (default 5, 0=skip)")
    args = ap.parse_args()

    query = " ".join(args.query)
    print(f"\narXiv Bridge Search: {query!r}")
    print("=" * 70)

    results = bridge_search(query, limit=args.limit, top_n_derive=args.derive)
    if not results:
        print("  No results.")
        return

    show = results[:args.top_k]
    local_count = sum(1 for r in results if r.local)

    print(f"\nTop {len(show)} / {len(results)} candidates:\n")
    print(f"{'#':<3} {'arxiv_id':<14} {'yr':<5} {'rel':>5}  {'✓':1}  {'src':<8}  title")
    print("-" * 90)

    for i, r in enumerate(show, 1):
        flag  = "✓" if r.local else " "
        title = r.title[:48] if r.title else "(no title)"
        print(f"{i:<3} {r.arxiv_id:<14} {str(r.year or '?'):<5} "
              f"{r.relevance_score:>5.2f}  {flag}  {r.source:<8}  {title}")
        if r.utility:
            for u in r.utility[:2]:
                print(f"         ↳ {u[:80]}")

    print()
    print(f"  Local (in corpus): {local_count}/{len(results)}")
    print(f"  Novel (upstream):  {len(results)-local_count}/{len(results)}")
    novel_derived = [r for r in results if not r.local and r.is_complete]
    if novel_derived:
        print(f"  LLM-derived utility: {len(novel_derived)} papers")


if __name__ == "__main__":
    _cli()
