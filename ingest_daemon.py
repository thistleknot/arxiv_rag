#!/usr/bin/env python3
"""
Continuous Phase 3-5 enrichment daemon for papers/post_processed/.

For every paper that has a .md + .csv but no _methods.md, runs:
  0. Tag normalization  (<image N>  →  <image_N>)
  3. vlm_describe.py          — Ollama qwen3-vl:2b image descriptions
  4. reinsert_descriptions.py — embed descriptions into Markdown
  5. extract_methods.py       — gpt-4.1 pseudocode extraction

State is checkpointed in checkpoints/ingest_service.db.
Papers are processed in descending utility order (from main CSV).

Usage:
    python ingest_daemon.py              # continuous loop
    python ingest_daemon.py --once       # one batch then exit
    python ingest_daemon.py --limit 5    # cap papers per batch
    python ingest_daemon.py --poll 120   # poll interval seconds (default 60)
    python ingest_daemon.py --status     # print queue stats and exit

Preconditions:
    - papers/post_processed/ contains .md + .csv files (Phase 1+2 already done)
    - Ollama reachable at 192.168.3.17:11434 (Phase 3)
    - copilot-proxy at localhost:8069 (Phase 5)
Postconditions:
    - Each processed paper has _methods.md in papers/post_processed/
Failure modes:
    - Per-paper errors are recorded in DB with state='error'; daemon continues
    - Interrupted runs resume from last completed phase on restart
"""

import argparse
import csv as csv_module
import re
import sqlite3
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent
POST_PROCESSED = ROOT / "papers" / "post_processed"
DB_PATH = ROOT / "checkpoints" / "ingest_service.db"
MAIN_CSV = POST_PROCESSED / "arxiv_data_with_analysis_cleaned.csv"
PYTHON = sys.executable

# Script paths (same directory as this file)
_VLM = ROOT / "vlm_describe.py"
_REINSERT = ROOT / "reinsert_descriptions.py"
_METHODS = ROOT / "extract_methods.py"

# Regex patterns for old/new image tag formats
_OLD_TAG_RE = re.compile(r"<image (\d+)>")       # old: <image 135>
_NEW_TAG_RE = re.compile(r"<image_(\d+)>")        # new: <image_135>
_CLOSE_TAG_RE = re.compile(r"</image_\d+>")        # Phase 4 done signal

_RETRY_PREFIXES = ("[ERROR]", "[PENDING]")


# ── Helpers ──────────────────────────────────────────────────────────────────

def _ts() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")


def _csv_field_limit():
    limit = sys.maxsize
    while True:
        try:
            csv_module.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


_csv_field_limit()


# ── Database ──────────────────────────────────────────────────────────────────

def open_db() -> sqlite3.Connection:
    """
    Open (or create) the SQLite checkpoint DB.

    Require: DB_PATH parent directory exists or can be created.
    Guarantee: returns connection with row_factory = sqlite3.Row.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS papers (
            paper_id        TEXT PRIMARY KEY,
            md_path         TEXT NOT NULL,
            csv_path        TEXT NOT NULL,
            priority        REAL    DEFAULT 0.0,
            normalize_done  INTEGER DEFAULT 0,
            phase3_done     INTEGER DEFAULT 0,
            phase4_done     INTEGER DEFAULT 0,
            phase5_done     INTEGER DEFAULT 0,
            state           TEXT    DEFAULT 'pending',
            error_msg       TEXT,
            updated_at      TEXT
        )
    """)
    conn.commit()
    return conn


# ── Priority loading ──────────────────────────────────────────────────────────

def load_priorities() -> dict[str, float]:
    """
    Load utility scores from main CSV keyed by normalised paper_id.

    Failure mode: returns empty dict if CSV is missing or unreadable.
    """
    priorities: dict[str, float] = {}
    if not MAIN_CSV.is_file():
        return priorities
    try:
        with open(MAIN_CSV, "r", encoding="utf-8", newline="") as f:
            for row in csv_module.DictReader(f):
                pid = row.get("arxiv_id", "").strip()
                try:
                    u = float(row.get("utility", 0) or 0)
                except (ValueError, TypeError):
                    u = 0.0
                if pid:
                    priorities[pid.replace(".", "_")] = u
    except Exception as exc:
        print(f"[warn] Could not load priorities: {exc}")
    return priorities


# ── Phase completion checks (filesystem-derived) ─────────────────────────────

def _normalize_needed(md_path: Path) -> bool:
    """Return True if MD still contains old-format <image N> tags."""
    try:
        return bool(_OLD_TAG_RE.search(md_path.read_text(encoding="utf-8")))
    except Exception:
        return False


def _phase3_done(csv_path: Path) -> bool:
    """Return True if CSV has at least one successful 'description' entry."""
    try:
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv_module.DictReader(f)
            if "description" not in (reader.fieldnames or []):
                return False
            for row in reader:
                desc = row.get("description", "")
                if desc and not any(desc.startswith(p) for p in _RETRY_PREFIXES):
                    return True
    except Exception:
        pass
    return False


def _phase4_done(md_path: Path) -> bool:
    """Return True if MD contains </image_N> closing tags (Phase 4 wrote them)."""
    try:
        return bool(_CLOSE_TAG_RE.search(md_path.read_text(encoding="utf-8")))
    except Exception:
        return False


# ── Scan for eligible papers ──────────────────────────────────────────────────

def scan_papers(conn: sqlite3.Connection, priorities: dict[str, float]) -> int:
    """
    Scan POST_PROCESSED for papers with .md + .csv but no _methods.md.
    Insert newly discovered papers; update priorities for existing ones.

    Require: conn is open with row_factory = sqlite3.Row.
    Returns: count of newly registered papers.
    """
    new = 0
    for md in sorted(POST_PROCESSED.glob("*.md")):
        if md.stem.endswith("_methods"):
            continue
        csv_path = md.with_suffix(".csv")
        methods_path = md.with_name(md.stem + "_methods.md")

        if not csv_path.is_file():
            continue

        pri = priorities.get(md.stem, 0.0)

        if methods_path.is_file():
            conn.execute(
                "INSERT OR IGNORE INTO papers "
                "(paper_id, md_path, csv_path, priority, normalize_done, "
                " phase3_done, phase4_done, phase5_done, state, updated_at) "
                "VALUES (?,?,?,?,1,1,1,1,'done',?)",
                (md.stem, str(md), str(csv_path), pri, _ts()),
            )
            conn.execute(
                "UPDATE papers SET state='done', phase5_done=1, updated_at=? "
                "WHERE paper_id=? AND state != 'done'",
                (_ts(), md.stem),
            )
            continue

        existing = conn.execute(
            "SELECT paper_id FROM papers WHERE paper_id = ?", (md.stem,)
        ).fetchone()

        if existing:
            conn.execute(
                "UPDATE papers SET priority=? WHERE paper_id=?", (pri, md.stem)
            )
            continue

        # First time seen — derive current phase state from filesystem
        n_done = 0 if _normalize_needed(md) else 1
        p3 = 1 if _phase3_done(csv_path) else 0
        p4 = 1 if _phase4_done(md) else 0

        conn.execute(
            "INSERT INTO papers "
            "(paper_id, md_path, csv_path, priority, normalize_done, "
            " phase3_done, phase4_done, phase5_done, state, updated_at) "
            "VALUES (?,?,?,?,?,?,?,0,'pending',?)",
            (md.stem, str(md), str(csv_path), pri, n_done, p3, p4, _ts()),
        )
        new += 1

    conn.commit()
    return new


# ── Phase runners ─────────────────────────────────────────────────────────────

def _run(cmd: list[str]) -> tuple[bool, str]:
    """
    Run a subprocess.

    Require: cmd is a valid executable + args list.
    Guarantee: returns (success, error_snippet).
    """
    try:
        r = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if r.returncode != 0:
            stderr = (r.stderr or "").strip()
            stdout = (r.stdout or "").strip()
            return False, (stderr or stdout)[-600:]
        # Echo stdout so progress is visible
        if r.stdout.strip():
            print(r.stdout.rstrip())
        return True, ""
    except Exception as exc:
        return False, str(exc)[:600]


def do_normalize(md_path: Path) -> tuple[bool, str]:
    """Convert <image N> (space-separated) tags to <image_N> in-place."""
    try:
        content = md_path.read_text(encoding="utf-8")
        new_content = _OLD_TAG_RE.sub(lambda m: f"<image_{m.group(1)}>", content)
        if new_content != content:
            md_path.write_text(new_content, encoding="utf-8")
            converted = len(_OLD_TAG_RE.findall(content))
            print(f"    Normalized {converted} image tag(s)")
        else:
            print(f"    No old-format tags found (already clean)")
        return True, ""
    except Exception as exc:
        return False, str(exc)


def do_phase3(csv_path: Path) -> tuple[bool, str]:
    return _run([PYTHON, str(_VLM), str(csv_path)])


def do_phase4(md_path: Path, csv_path: Path) -> tuple[bool, str]:
    return _run([PYTHON, str(_REINSERT), str(md_path), str(csv_path)])


def do_phase5(md_path: Path) -> tuple[bool, str]:
    return _run([PYTHON, str(_METHODS), str(md_path)])


# ── Process one paper ─────────────────────────────────────────────────────────

def process_paper(conn: sqlite3.Connection, paper_id: str) -> bool:
    """
    Run all outstanding phases for a single paper.

    Require: paper_id exists in papers table with state='running'.
    Guarantee: on success sets state='done'; on failure sets state='error'.
    Returns: True if paper reached done state.
    """
    row = conn.execute(
        "SELECT * FROM papers WHERE paper_id = ?", (paper_id,)
    ).fetchone()
    if not row:
        return False

    md = Path(row["md_path"])
    csv_p = Path(row["csv_path"])
    methods_path = md.with_name(md.stem + "_methods.md")

    def mark(col: str):
        conn.execute(
            f"UPDATE papers SET {col}=1, updated_at=? WHERE paper_id=?",
            (_ts(), paper_id),
        )
        conn.commit()

    def fail(msg: str):
        conn.execute(
            "UPDATE papers SET state='error', error_msg=?, updated_at=? WHERE paper_id=?",
            (msg[:800], _ts(), paper_id),
        )
        conn.commit()
        print(f"  ✗ ERROR {paper_id}: {msg[:200]}")

    # Phase 0 — normalize old image tags
    if not row["normalize_done"]:
        print(f"  [0] Normalizing image tags...")
        ok, err = do_normalize(md)
        if not ok:
            fail(f"normalize: {err}")
            return False
        mark("normalize_done")

    # Phase 3 — VLM descriptions
    if not row["phase3_done"]:
        print(f"  [3/5] VLM describe ({csv_p.name})")
        ok, err = do_phase3(csv_p)
        if not ok:
            fail(f"phase3: {err}")
            return False
        mark("phase3_done")

    # Phase 4 — reinsert descriptions
    if not row["phase4_done"]:
        print(f"  [4/5] Reinsert descriptions ({md.name})")
        ok, err = do_phase4(md, csv_p)
        if not ok:
            fail(f"phase4: {err}")
            return False
        mark("phase4_done")

    # Phase 5 — extract methods
    if not methods_path.is_file():
        print(f"  [5/5] Extract methods ({md.name})")
        ok, err = do_phase5(md)
        if not ok:
            fail(f"phase5: {err}")
            return False

    conn.execute(
        "UPDATE papers SET phase5_done=1, state='done', updated_at=? WHERE paper_id=?",
        (_ts(), paper_id),
    )
    conn.commit()
    print(f"  ✓ Done: {paper_id}")
    return True


# ── Batch runner ──────────────────────────────────────────────────────────────

def run_batch(conn: sqlite3.Connection, limit: int | None) -> int:
    """
    Claim and process the next batch of pending papers.

    Returns: count of papers attempted.
    """
    q = (
        "SELECT paper_id FROM papers WHERE state='pending' "
        "ORDER BY priority DESC, paper_id ASC"
    )
    if limit:
        q += f" LIMIT {limit}"

    rows = conn.execute(q).fetchall()
    if not rows:
        return 0

    processed = 0
    for row in rows:
        pid = row["paper_id"]
        conn.execute(
            "UPDATE papers SET state='running', updated_at=? WHERE paper_id=? AND state='pending'",
            (_ts(), pid),
        )
        conn.commit()

        print(f"\n[{_ts()}] ── {pid} ──────────────────────────")
        success = process_paper(conn, pid)

        if not success:
            # Reset to pending so it can be retried (unless error was set)
            conn.execute(
                "UPDATE papers SET state='pending' WHERE paper_id=? AND state='running'",
                (pid,),
            )
            conn.commit()

        processed += 1

    return processed


# ── Stats ─────────────────────────────────────────────────────────────────────

def print_stats(conn: sqlite3.Connection):
    states = conn.execute(
        "SELECT state, COUNT(*) AS n FROM papers GROUP BY state ORDER BY state"
    ).fetchall()
    total = sum(r["n"] for r in states)
    print(f"\n  Queue summary ({total} total):")
    for r in states:
        print(f"    {r['state']:12s} {r['n']}")

    errors = conn.execute(
        "SELECT paper_id, error_msg FROM papers WHERE state='error' LIMIT 5"
    ).fetchall()
    if errors:
        print("\n  Recent errors:")
        for e in errors:
            print(f"    {e['paper_id']}: {(e['error_msg'] or '')[:80]}")


# ── Entrypoint ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Continuous Phase 3-5 ingestion daemon for papers/post_processed/"
    )
    parser.add_argument("--once", action="store_true", help="Process one batch then exit")
    parser.add_argument("--limit", type=int, default=None, help="Max papers per batch")
    parser.add_argument("--poll", type=int, default=60, help="Poll interval seconds")
    parser.add_argument("--status", action="store_true", help="Print queue stats and exit")
    args = parser.parse_args()

    conn = open_db()
    priorities = load_priorities()
    print(f"[{_ts()}] Ingest daemon — DB: {DB_PATH}")

    if args.status:
        scan_papers(conn, priorities)
        print_stats(conn)
        conn.close()
        return

    print(f"  Scanning {POST_PROCESSED} ...")
    new = scan_papers(conn, priorities)
    if new:
        print(f"  Registered {new} new paper(s).")
    print_stats(conn)
    print()

    try:
        while True:
            pending = conn.execute(
                "SELECT COUNT(*) FROM papers WHERE state='pending'"
            ).fetchone()[0]

            if pending > 0:
                done = run_batch(conn, args.limit)
                print(f"\n[{_ts()}] Batch complete — {done} paper(s) attempted.")
            else:
                print(f"[{_ts()}] Nothing pending.")

            if args.once:
                break

            # Rescan for newly arrived papers before sleeping
            new = scan_papers(conn, priorities)
            if new:
                print(f"[{_ts()}] Registered {new} new paper(s).")
            time.sleep(args.poll)

    except KeyboardInterrupt:
        print(f"\n[{_ts()}] Interrupted — progress saved to {DB_PATH}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
