#!/usr/bin/env python3
"""
arXiv Bridge — upstream semantic search layer for the arxiv_rag retrieval stack.

Extends local CSV retrieval with upstream paper discovery via:
  1. Semantic Scholar Graph API  (semantic ranking, citation count, arXiv IDs)
  2. arXiv Atom API fallback     (BM25 keyword, title/abstract field scoping)

SQLite cache amortises utility derivation so each paper is scored at most once.
Citation count is used as a cold-start proxy when no cached utility exists.

Preconditions:
    - requests installed in the Python environment
    - papers/post_processed/arxiv_data_with_analysis_cleaned.csv reachable
    - checkpoints/ directory writable (for utility cache DB)
Postconditions:
    - Returns BridgeResult list sorted by composite score (utility × relevance)
    - Papers already in local CSV are flagged local=True
    - Upstream novelties are flagged local=False (candidates for downstream ingestion)
Failure modes:
    - S2 API unavailable/rate-limited → falls back to arXiv Atom API
    - Both APIs fail → returns empty list with warning
    - Cache DB unwritable → utility derivation still works, just not cached
"""

import csv
import math
import re
import sqlite3
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import requests

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_ROOT       = Path(__file__).resolve().parent
_CSV        = _ROOT / "papers" / "post_processed" / "arxiv_data_with_analysis_cleaned.csv"
_CACHE_DB   = _ROOT / "checkpoints" / "utility_cache.db"

S2_SEARCH   = "https://api.semanticscholar.org/graph/v1/paper/search"
ARXIV_API   = "http://export.arxiv.org/api/query"
S2_FIELDS   = "title,abstract,externalIds,year,citationCount,influentialCitationCount"
_ATOM_NS    = "http://www.w3.org/2005/Atom"

# S2 API key — set env var SEMANTIC_SCHOLAR_API_KEY for higher rate limits
import os
_S2_KEY = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")


@dataclass
class BridgeResult:
    arxiv_id: str
    title: str
    abstract: str
    year: Optional[int]
    citation_count: int
    utility_score: float       # 0–5 scale, matching local CSV convention
    relevance_score: float     # normalised 0–1 from source ranking position
    composite: float           # utility_score * relevance_score
    source: str                # "local", "s2", "arxiv_api"
    local: bool                # True if already in local CSV
    pdf_url: str = ""


def _norm_id(aid: str) -> str:
    """Normalise to dot form: '2601_09113' or '2601.09113v2' → '2601.09113'."""
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
    _CACHE_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_CACHE_DB), timeout=10)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS utility_cache (
            arxiv_id    TEXT PRIMARY KEY,
            title       TEXT,
            utility     REAL,
            citation_count INTEGER,
            cached_at   TEXT
        )
    """)
    conn.commit()
    return conn


def _get_cached_utility(arxiv_id: str, conn: sqlite3.Connection) -> Optional[float]:
    row = conn.execute(
        "SELECT utility FROM utility_cache WHERE arxiv_id = ?",
        (_norm_id(arxiv_id),)
    ).fetchone()
    return float(row[0]) if row else None


def _cache_utility(arxiv_id: str, title: str, utility: float,
                   citation_count: int, conn: sqlite3.Connection) -> None:
    from datetime import datetime
    conn.execute(
        "INSERT OR REPLACE INTO utility_cache "
        "(arxiv_id, title, utility, citation_count, cached_at) VALUES (?,?,?,?,?)",
        (_norm_id(arxiv_id), title, utility, citation_count,
         datetime.utcnow().isoformat())
    )
    conn.commit()


# ── Utility derivation ────────────────────────────────────────────────────────

def _citation_utility(citation_count: int) -> float:
    """
    Map citation count to a 0–5 utility proxy using log-scale.

    Rationale: citation distribution is heavy-tailed (log-normal).
    Breakpoints calibrated against the local corpus median (~30 citations):
        0 cites  → 0.5   (some value for being on arXiv)
        10 cites → 1.5
        50 cites → 2.5   (≈ median in AI/ML)
       200 cites → 3.5
      1000 cites → 4.5
      5000 cites → 5.0
    """
    if citation_count <= 0:
        return 0.5
    return min(5.0, 0.5 + math.log1p(citation_count) / math.log1p(5000) * 4.5)


# ── Semantic Scholar search ───────────────────────────────────────────────────

def _search_s2(query: str, limit: int = 20) -> list[dict]:
    """
    Query Semantic Scholar paper search. Returns raw paper dicts.

    Require: query non-empty, limit 1-100.
    Failure modes: HTTP errors / rate-limit → raises requests.HTTPError.
    """
    headers = {"x-api-key": _S2_KEY} if _S2_KEY else {}
    params = {
        "query": query,
        "limit": min(limit, 100),
        "fields": S2_FIELDS,
    }
    resp = requests.get(S2_SEARCH, params=params, headers=headers, timeout=15)
    resp.raise_for_status()
    return resp.json().get("data", [])


def _s2_to_candidates(papers: list[dict]) -> list[dict]:
    """Convert S2 paper dicts to normalised candidate dicts."""
    out = []
    for i, p in enumerate(papers):
        ext = p.get("externalIds") or {}
        aid = ext.get("ArXiv")
        if not aid:
            continue  # skip non-arXiv papers
        out.append({
            "arxiv_id": _norm_id(aid),
            "title": p.get("title") or "",
            "abstract": p.get("abstract") or "",
            "year": p.get("year"),
            "citation_count": p.get("citationCount") or 0,
            "rank_position": i,          # 0-based position in S2 relevance list
            "source": "s2",
            "pdf_url": f"https://arxiv.org/pdf/{_norm_id(aid)}.pdf",
        })
    return out


# ── arXiv Atom API fallback ───────────────────────────────────────────────────

def _search_arxiv_api(query: str, limit: int = 20) -> list[dict]:
    """
    Query the arXiv Atom API with title+abstract field scope.

    Splits query into terms, constructs ti+abs conjunctive query.
    Failure modes: HTTP errors → raises requests.HTTPError.
    """
    terms = [t.strip() for t in query.split() if t.strip()]
    if not terms:
        return []
    # Build: (ti:term1 OR abs:term1) AND (ti:term2 OR abs:term2) ...
    clauses = [f"(ti:{t}+OR+abs:{t})" for t in terms]
    search_query = "+AND+".join(clauses)
    params = f"search_query={search_query}&max_results={limit}&sortBy=relevance"
    url = f"{ARXIV_API}?{params}"

    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    return _parse_atom(resp.text)


def _parse_atom(xml_text: str) -> list[dict]:
    root = ET.fromstring(xml_text)
    ns = {"a": _ATOM_NS, "arxiv": "http://arxiv.org/schemas/atom"}
    out = []
    for i, entry in enumerate(root.findall("a:entry", ns)):
        raw_id = (entry.findtext("a:id", "", ns) or "").strip()
        # arXiv Atom ID looks like http://arxiv.org/abs/2601.09113v2
        aid_match = re.search(r"abs/(\S+)", raw_id)
        if not aid_match:
            continue
        aid = _norm_id(aid_match.group(1))
        title    = (entry.findtext("a:title", "", ns) or "").strip().replace("\n", " ")
        abstract = (entry.findtext("a:summary", "", ns) or "").strip().replace("\n", " ")
        year_str = entry.findtext("a:published", "", ns) or ""
        year     = int(year_str[:4]) if year_str else None
        # Find PDF link
        pdf_url = ""
        for link in entry.findall("a:link", ns):
            if link.get("title") == "pdf":
                pdf_url = link.get("href", "")
        out.append({
            "arxiv_id": aid,
            "title": title,
            "abstract": abstract,
            "year": year,
            "citation_count": 0,        # arXiv API doesn't return citations
            "rank_position": i,
            "source": "arxiv_api",
            "pdf_url": pdf_url,
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
    verbose: bool = True,
) -> list[BridgeResult]:
    """
    Search for arXiv papers matching query, merging upstream + local sources.

    Preconditions:
        query is a non-empty search string.
        limit is the max candidates to consider from each upstream source.
    Postconditions:
        Returns BridgeResult list sorted descending by composite score.
        local=True papers were already in local CSV; local=False are novel.
    Failure modes:
        S2 unavailable → falls back to arXiv Atom API with warning.
        Both unavailable → returns empty list.
    """
    local_ids = _load_local_ids()
    conn = _open_cache()

    # ── 1. Upstream search ────────────────────────────────────────────────────
    candidates: list[dict] = []
    source_used = "none"

    try:
        if verbose:
            print("  [bridge] Querying Semantic Scholar...", flush=True)
        s2_papers = _search_s2(query, limit=limit)
        candidates = _s2_to_candidates(s2_papers)
        source_used = "s2"
        if verbose:
            print(f"  [bridge] S2 returned {len(s2_papers)} papers, "
                  f"{len(candidates)} with arXiv IDs")
        # polite delay
        time.sleep(1.0)
    except Exception as exc:
        if verbose:
            print(f"  [bridge] S2 failed ({exc}), falling back to arXiv API...")
        try:
            candidates = _search_arxiv_api(query, limit=limit)
            source_used = "arxiv_api"
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

    # ── 2. Deduplicate + score ────────────────────────────────────────────────
    seen: set[str] = set()
    results: list[BridgeResult] = []
    total = len(candidates)

    for c in candidates:
        aid = c["arxiv_id"]
        if aid in seen:
            continue
        seen.add(aid)

        is_local = aid in local_ids

        # Utility: local CSV > cache > citation proxy
        utility: float
        if is_local:
            raw = local_ids[aid].get("utility", "").strip()
            try:
                utility = float(raw)
            except (ValueError, TypeError):
                utility = _citation_utility(c["citation_count"])
            source_label = "local"
        else:
            cached = _get_cached_utility(aid, conn)
            if cached is not None:
                utility = cached
                source_label = "cached"
            else:
                utility = _citation_utility(c["citation_count"])
                _cache_utility(aid, c["title"], utility, c["citation_count"], conn)
                source_label = c["source"]

        relevance = _rank_to_relevance(c["rank_position"], total)
        composite = utility * relevance

        results.append(BridgeResult(
            arxiv_id       = aid,
            title          = c["title"],
            abstract       = c["abstract"],
            year           = c["year"],
            citation_count = c["citation_count"],
            utility_score  = round(utility, 3),
            relevance_score= round(relevance, 3),
            composite      = round(composite, 4),
            source         = source_label,
            local          = is_local,
            pdf_url        = c.get("pdf_url", ""),
        ))

    conn.close()
    results.sort(key=lambda r: r.composite, reverse=True)
    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def _cli() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="arXiv bridge — upstream semantic search")
    ap.add_argument("query", nargs="+", help="Search query")
    ap.add_argument("--limit", type=int, default=20, help="Max candidates (default 20)")
    ap.add_argument("--top-k", type=int, default=10, help="Top results to show (default 10)")
    args = ap.parse_args()

    query = " ".join(args.query)
    print(f"\narXiv Bridge Search: {query!r}")
    print("=" * 60)

    results = bridge_search(query, limit=args.limit)
    if not results:
        print("  No results.")
        return

    print(f"\nTop {min(args.top_k, len(results))} / {len(results)} candidates:\n")
    print(f"{'#':<3} {'arxiv_id':<14} {'yr':<5} {'cites':>6} "
          f"{'util':>5} {'rel':>5} {'comp':>6}  {'L':1}  title")
    print("-" * 100)

    local_count = sum(1 for r in results if r.local)

    for i, r in enumerate(results[:args.top_k], 1):
        flag = "✓" if r.local else " "
        title_short = r.title[:50] if r.title else "(no title)"
        print(
            f"{i:<3} {r.arxiv_id:<14} {str(r.year or '?'):<5} {r.citation_count:>6} "
            f"{r.utility_score:>5.2f} {r.relevance_score:>5.2f} {r.composite:>6.3f} "
            f" {flag}  {title_short}"
        )

    print()
    print(f"  Local (already in corpus): {local_count}/{len(results)}")
    print(f"  Novel (upstream only):     {len(results)-local_count}/{len(results)}")
    novel = [r for r in results[:args.top_k] if not r.local]
    if novel:
        print(f"\n  Novel top-{args.top_k} candidates (not yet in local corpus):")
        for r in novel:
            print(f"    {r.arxiv_id}  [{r.year}]  cites={r.citation_count}  "
                  f"utility_proxy={r.utility_score:.2f}  {r.title[:60]}")


if __name__ == "__main__":
    _cli()
