#!/usr/bin/env python3
"""
Auto-Ingest MCP server for arxiv_rag.

Exposes queue inspection and methods retrieval tools so agentic sessions
can check enrichment state and read extracted methods without needing
direct filesystem access.

Preconditions:
    - checkpoints/ingest_service.db exists (created by ingest_daemon.py)
    - papers/post_processed/ is accessible at REPO_ROOT
    - fastmcp installed in active Python environment
Postconditions:
    - Tools available via stdio (Copilot CLI) or HTTP :PORT/mcp (Codex)
Failure modes:
    - Missing DB: tools return 'daemon not yet started' message
    - Missing _methods.md: get_methods returns None; check_papers shows state
"""

import argparse
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from fastmcp import FastMCP

REPO_ROOT = Path(r"C:\Users\user\arxiv_id_lists")
POST_PROCESSED = REPO_ROOT / "papers" / "post_processed"
DB_PATH = REPO_ROOT / "checkpoints" / "ingest_service.db"

mcp = FastMCP("auto-ingest")


def _open_db(repo_root: str | None = None) -> sqlite3.Connection | None:
    """Open ingest DB, return None if not yet initialised."""
    db = (
        Path(repo_root) / "checkpoints" / "ingest_service.db"
        if repo_root
        else DB_PATH
    )
    if not db.is_file():
        return None
    conn = sqlite3.connect(str(db), timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def _norm(paper_id: str) -> str:
    """Normalise paper_id: '1806.07366' → '1806_07366'."""
    return paper_id.strip().replace(".", "_")


def _methods_path(paper_id: str, repo_root: str | None = None) -> Path:
    base = (
        Path(repo_root) / "papers" / "post_processed"
        if repo_root
        else POST_PROCESSED
    )
    return base / f"{_norm(paper_id)}_methods.md"


@mcp.tool()
def ingest_status(repo_root: str | None = None) -> str:
    """
    Return a summary of the ingestion queue from the daemon checkpoint DB.

    Args:
        repo_root: Absolute path to the arxiv_rag repo root.
                   Defaults to C:\\Users\\user\\arxiv_id_lists.
    Returns:
        Human-readable counts by state (pending, done, error, running).
    """
    conn = _open_db(repo_root)
    if conn is None:
        return (
            "Daemon checkpoint DB not found. "
            "Run `python ingest_daemon.py --status` once to initialise it."
        )
    try:
        rows = conn.execute(
            "SELECT state, COUNT(*) AS n FROM papers GROUP BY state ORDER BY state"
        ).fetchall()
        total = sum(r["n"] for r in rows)
        lines = [f"Ingest queue ({total} total):"]
        for r in rows:
            lines.append(f"  {r['state']:12s}  {r['n']}")
        return "\n".join(lines)
    finally:
        conn.close()


@mcp.tool()
def check_papers(
    paper_ids: list[str], repo_root: str | None = None
) -> str:
    """
    Check enrichment state for a list of paper IDs.

    Args:
        paper_ids: List of arXiv IDs ('1806.07366' or '1806_07366').
        repo_root: Absolute path to the arxiv_rag repo root.
    Returns:
        Per-paper state: 'done', 'pending', 'running', 'error', or 'unknown'.
        'done' means _methods.md exists on disk.
    """
    conn = _open_db(repo_root)
    results: dict[str, str] = {}

    for pid in paper_ids:
        stem = _norm(pid)
        if _methods_path(pid, repo_root).is_file():
            results[pid] = "done"
            continue
        if conn is not None:
            row = conn.execute(
                "SELECT state FROM papers WHERE paper_id = ?", (stem,)
            ).fetchone()
            results[pid] = row["state"] if row else "unknown"
        else:
            results[pid] = "unknown"

    if conn:
        conn.close()

    lines = [f"Paper enrichment states ({len(paper_ids)} checked):"]
    for pid, state in results.items():
        lines.append(f"  {pid:20s}  {state}")
    return "\n".join(lines)


@mcp.tool()
def get_methods(paper_id: str, repo_root: str | None = None) -> str:
    """
    Read and return the extracted _methods.md content for a paper.

    Args:
        paper_id: arXiv ID ('1806.07366' or '1806_07366').
        repo_root: Absolute path to the arxiv_rag repo root.
    Returns:
        Full _methods.md content, or an error message if not yet extracted.
    """
    mp = _methods_path(paper_id, repo_root)
    if not mp.is_file():
        return (
            f"No _methods.md for {paper_id}. "
            "Paper has not completed the enrichment pipeline. "
            "Run `python ingest_daemon.py` or call queue_papers() to schedule it."
        )
    return mp.read_text(encoding="utf-8")


@mcp.tool()
def queue_papers(
    paper_ids: list[str],
    priority: float = 1.0,
    repo_root: str | None = None,
) -> str:
    """
    Add papers to the ingestion queue for the background daemon to process.

    Papers with _methods.md already present are silently skipped.
    Papers already registered in the DB retain their existing state but get a
    priority bump.

    Args:
        paper_ids: List of arXiv IDs to queue.
        priority:  Priority weight (higher = processed sooner). Default 1.0.
                   Use 10.0 to move papers to the front of the queue.
        repo_root: Absolute path to the arxiv_rag repo root.
    Returns:
        Summary: queued / already_queued / already_done / not_found counts.
    """
    base = (
        Path(repo_root) / "papers" / "post_processed"
        if repo_root
        else POST_PROCESSED
    )
    conn = _open_db(repo_root)
    if conn is None:
        return (
            "Daemon checkpoint DB not found. "
            "Run `python ingest_daemon.py --status` once to initialise it."
        )

    queued = skipped_done = not_found = already_queued = 0

    for pid in paper_ids:
        stem = _norm(pid)
        if _methods_path(pid, repo_root).is_file():
            skipped_done += 1
            continue
        md = base / f"{stem}.md"
        csv_p = base / f"{stem}.csv"
        if not md.is_file() or not csv_p.is_file():
            not_found += 1
            continue
        existing = conn.execute(
            "SELECT state FROM papers WHERE paper_id = ?", (stem,)
        ).fetchone()
        now = datetime.utcnow().isoformat()
        if existing:
            conn.execute(
                "UPDATE papers SET priority = MAX(priority, ?), updated_at = ? "
                "WHERE paper_id = ?",
                (priority, now, stem),
            )
            already_queued += 1
        else:
            conn.execute(
                "INSERT INTO papers "
                "(paper_id, md_path, csv_path, priority, state, updated_at) "
                "VALUES (?,?,?,?,'pending',?)",
                (stem, str(md), str(csv_p), priority, now),
            )
            queued += 1

    conn.commit()
    conn.close()
    return (
        f"Queue update: {queued} newly queued, {already_queued} already queued "
        f"(priority bumped to {priority}), {skipped_done} already done, "
        f"{not_found} not found on disk."
    )


@mcp.tool()
def list_errors(limit: int = 10, repo_root: str | None = None) -> str:
    """
    List papers stuck in the error state with their last error message.

    Args:
        limit:     Max rows to return (default 10).
        repo_root: Absolute path to the arxiv_rag repo root.
    Returns:
        Table of paper_id, updated_at, and truncated error message.
    """
    conn = _open_db(repo_root)
    if conn is None:
        return "Daemon checkpoint DB not found."
    try:
        rows = conn.execute(
            "SELECT paper_id, error_msg, updated_at FROM papers "
            "WHERE state = 'error' ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        if not rows:
            return "No papers in error state."
        lines = [f"{'paper_id':<22} {'updated':<20} error"]
        lines.append("-" * 80)
        for r in rows:
            msg = (r["error_msg"] or "")[:55].replace("\n", " ")
            lines.append(
                f"{r['paper_id']:<22} {(r['updated_at'] or '')[:19]:<20} {msg}"
            )
        return "\n".join(lines)
    finally:
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-Ingest MCP server")
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="HTTP port for streamable-http transport (omit for stdio)",
    )
    args = parser.parse_args()

    if args.port:
        mcp.run(
            transport="streamable-http",
            host="0.0.0.0",
            port=args.port,
            path="/mcp",
        )
    else:
        mcp.run()
