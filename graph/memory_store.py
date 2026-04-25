"""memory_store.py — SQLite-backed Pages and Throughlines memory tiers.

Implements tier 2 (Pages) and tier 3 (Throughlines) of the agentic_kg_memory
three-tier memory stack.  Tier 1 (Triplets) is handled by nli_graph_retriever.py.

Skill reference: .copilot/skills/agentic_kg_memory/SKILL.md

Design:
  - MemoryStore is pure persistence.  Decision logic (page promotion, throughline
    selection, evidence gating) lives in the caller (syllogism_retriever.py).
  - Embeddings are stored as raw float32 bytes, L2-normalized by the caller.
    Cosine similarity = dot product on L2-normalized vectors.
  - All history is append-only — page_history and throughline_history rows are
    never modified after insertion.
  - MemRL Q-update for page fit_score and throughline q_score uses PAGE_FIT_ALPHA.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple
import sys

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from constants import (
    MEMORY_STORE_DB,
    PAGE_EMBED_SIM_TAU,
    PAGE_FIT_ALPHA,
    THROUGHLINE_SIM_TAU,
)


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _vec_to_blob(vec: np.ndarray) -> bytes:
    return vec.astype(np.float32).tobytes()


def _blob_to_vec(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


@dataclass
class MemoryPage:
    """In-memory view of a memory_pages row."""
    page_id:              str
    goal:                 str
    domain:               str
    intent:               str       = ""
    constraints:          List[str] = field(default_factory=list)
    entity_bag:           List[str] = field(default_factory=list)
    fit_score:             float        = 0.5
    read_count:            int          = 0
    confirmed_read_count:  int          = 0
    update_count:          int          = 0
    wiki_summary:          str          = ""
    embedding_ref:         str          = ""
    bm25_text:             str          = ""
    triplet_sequence_text: str          = ""
    cluster_id:            Optional[str] = None


@dataclass
class Throughline:
    """In-memory view of a throughlines row."""
    throughline_id:          str
    page_id:                 str
    claim_text:              str
    supporting_arxiv_ids:    List[str] = field(default_factory=list)
    supporting_triplet_keys: List[str] = field(default_factory=list)
    supporting_fks:          List[str] = field(default_factory=list)
    source_fks:              List[str] = field(default_factory=list)
    fact_fks:                List[str] = field(default_factory=list)
    q_score:                 float     = 0.5
    merge_score:             float     = 0.0
    merged_into:             Optional[str] = None
    canonical:               bool      = True
    status:                  str       = "active"  # active | competing | deprecated | merged


_SCHEMA = """
CREATE TABLE IF NOT EXISTS memory_pages (
    page_id              TEXT PRIMARY KEY,
    goal                 TEXT NOT NULL,
    domain               TEXT NOT NULL DEFAULT '',
    intent               TEXT NOT NULL DEFAULT '',
    constraints          TEXT NOT NULL DEFAULT '[]',
    entity_bag           TEXT NOT NULL DEFAULT '[]',
    goal_embedding       BLOB,
    bm25_text            TEXT NOT NULL DEFAULT '',
    triplet_sequence_text TEXT NOT NULL DEFAULT '',
    embedding_ref        TEXT NOT NULL DEFAULT '',
    cluster_id           TEXT,
    fit_score            REAL NOT NULL DEFAULT 0.5,
    read_count           INTEGER NOT NULL DEFAULT 0,
    confirmed_read_count INTEGER NOT NULL DEFAULT 0,
    update_count         INTEGER NOT NULL DEFAULT 0,
    wiki_summary         TEXT NOT NULL DEFAULT '',
    created_at           TEXT NOT NULL,
    updated_at           TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS page_triplets (
    page_id       TEXT    NOT NULL,
    triplet_key   TEXT    NOT NULL,
    max_nli_score REAL    NOT NULL DEFAULT 0.0,
    hit_count     INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (page_id, triplet_key),
    FOREIGN KEY (page_id) REFERENCES memory_pages(page_id)
);

CREATE TABLE IF NOT EXISTS page_history (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    page_id    TEXT NOT NULL,
    action     TEXT NOT NULL,
    details    TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS throughlines (
    throughline_id           TEXT PRIMARY KEY,
    page_id                  TEXT NOT NULL,
    claim_text               TEXT NOT NULL,
    claim_embedding          BLOB,
    supporting_arxiv_ids     TEXT NOT NULL DEFAULT '[]',
    supporting_triplet_keys  TEXT NOT NULL DEFAULT '[]',
    supporting_fks           TEXT NOT NULL DEFAULT '[]',
    source_fks               TEXT NOT NULL DEFAULT '[]',
    fact_fks                 TEXT NOT NULL DEFAULT '[]',
    q_score                  REAL NOT NULL DEFAULT 0.5,
    merge_score              REAL NOT NULL DEFAULT 0.0,
    merged_into              TEXT,
    canonical                INTEGER NOT NULL DEFAULT 1,
    status                   TEXT NOT NULL DEFAULT 'active',
    created_at               TEXT NOT NULL,
    updated_at               TEXT NOT NULL,
    FOREIGN KEY (page_id) REFERENCES memory_pages(page_id)
);

CREATE TABLE IF NOT EXISTS throughline_history (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    throughline_id TEXT NOT NULL,
    action         TEXT NOT NULL,
    q_score_before REAL,
    q_score_after  REAL,
    details        TEXT NOT NULL DEFAULT '',
    created_at     TEXT NOT NULL
);
"""

# Columns added after initial release — applied to existing DBs via _migrate().
_MIGRATIONS = [
    ("memory_pages",  "ALTER TABLE memory_pages ADD COLUMN bm25_text TEXT NOT NULL DEFAULT ''"),
    ("memory_pages",  "ALTER TABLE memory_pages ADD COLUMN triplet_sequence_text TEXT NOT NULL DEFAULT ''"),
    ("memory_pages",  "ALTER TABLE memory_pages ADD COLUMN cluster_id TEXT"),
    ("memory_pages",  "ALTER TABLE memory_pages ADD COLUMN intent TEXT NOT NULL DEFAULT ''"),
    ("memory_pages",  "ALTER TABLE memory_pages ADD COLUMN embedding_ref TEXT NOT NULL DEFAULT ''"),
    ("throughlines",  "ALTER TABLE throughlines ADD COLUMN source_fks TEXT NOT NULL DEFAULT '[]'"),
    ("throughlines",  "ALTER TABLE throughlines ADD COLUMN fact_fks TEXT NOT NULL DEFAULT '[]'"),
    ("throughlines",  "ALTER TABLE throughlines ADD COLUMN merge_score REAL NOT NULL DEFAULT 0.0"),
    ("throughlines",  "ALTER TABLE throughlines ADD COLUMN merged_into TEXT"),
    ("throughlines",  "ALTER TABLE throughlines ADD COLUMN canonical INTEGER NOT NULL DEFAULT 1"),
    ("throughlines",  "ALTER TABLE throughlines ADD COLUMN supporting_fks TEXT NOT NULL DEFAULT '[]'"),
]


class MemoryStore:
    """
    SQLite-backed persistence for Pages (tier 2) and Throughlines (tier 3).

    Require: db_path parent directory exists (created automatically).
    Guarantee: schema is created on first connect; subsequent runs load existing data.
    Maintain: history tables are append-only; main rows are upserted, never deleted.

    Args:
        db_path: SQLite file path.  Defaults to MEMORY_STORE_DB from constants.
    """

    def __init__(self, db_path: Optional[Path] = None):
        self._db_path = db_path or MEMORY_STORE_DB
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self._migrate()

    def close(self) -> None:
        """Close the SQLite connection."""
        self._conn.close()

    def _migrate(self) -> None:
        """Apply additive column migrations to existing DBs (idempotent)."""
        for _table, sql in _MIGRATIONS:
            try:
                self._conn.execute(sql)
                self._conn.commit()
            except sqlite3.OperationalError:
                pass  # column already exists

    # ── Pages ──────────────────────────────────────────────────────────────────

    def match_page(
        self,
        goal_vec: np.ndarray,
        domain:   str = "",
    ) -> Optional[str]:
        """Return page_id of the best cosine-matching page, or None if below threshold.

        Precondition: goal_vec is L2-normalized float32 of consistent dimension.
        Returns the page_id with the highest dot-product similarity >= PAGE_EMBED_SIM_TAU,
        or None when no stored page meets the threshold.
        """
        if domain:
            rows = self._conn.execute(
                "SELECT page_id, goal_embedding FROM memory_pages WHERE domain=?",
                (domain,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT page_id, goal_embedding FROM memory_pages"
            ).fetchall()
        best_id, best_sim = None, PAGE_EMBED_SIM_TAU
        for row in rows:
            emb_blob = row["goal_embedding"]
            if not emb_blob:
                continue
            stored = _blob_to_vec(emb_blob)
            if stored.shape != goal_vec.shape:
                continue
            sim = float(np.dot(goal_vec, stored))
            if sim > best_sim:
                best_sim, best_id = sim, row["page_id"]
        return best_id

    def get_page(self, page_id: str) -> Optional[MemoryPage]:
        """Load a MemoryPage from the DB by its page_id, or None if not found.

        Precondition: page_id is a valid hex UUID string.
        Failure mode: returns None on a missing row; never raises.
        """
        row = self._conn.execute(
            """
            SELECT page_id, goal, domain, intent, constraints, entity_bag,
                   fit_score, read_count, confirmed_read_count, update_count,
                   wiki_summary, embedding_ref,
                   bm25_text, triplet_sequence_text, cluster_id
            FROM memory_pages WHERE page_id=?
            """,
            (page_id,),
        ).fetchone()
        if row is None:
            return None
        return MemoryPage(
            page_id=row["page_id"],
            goal=row["goal"],
            domain=row["domain"],
            intent=row["intent"],
            constraints=json.loads(row["constraints"] or "[]"),
            entity_bag=json.loads(row["entity_bag"] or "[]"),
            fit_score=row["fit_score"],
            read_count=row["read_count"],
            confirmed_read_count=row["confirmed_read_count"],
            update_count=row["update_count"],
            wiki_summary=row["wiki_summary"] or "",
            embedding_ref=row["embedding_ref"] or "",
            bm25_text=row["bm25_text"] or "",
            triplet_sequence_text=row["triplet_sequence_text"] or "",
            cluster_id=row["cluster_id"],
        )

    def create_page(
        self,
        goal:        str,
        domain:      str,
        goal_vec:    Optional[np.ndarray] = None,
        intent:      str = "",
        bm25_text:   str = "",
        entity_bag:  Optional[List[str]] = None,
        constraints: Optional[List[str]] = None,
    ) -> str:
        """Insert a new memory page and return its page_id (UUID hex).

        Args:
            goal:       Normalized objective / goal statement.
            domain:     Routing domain label (e.g. 'artificial intelligence').
            goal_vec:   L2-normalized float32 embedding of goal (for future match).
            intent:     Top-level routing label (e.g. 'compare', 'survey'). Kept
                        mutually exclusive from other intents; acts as a routing key.
            bm25_text:  Sparse lexical surface for BM25 retrieval (defaults to goal).
            entity_bag: Optional entity weight hints.
            constraints: Optional constraint tags.
        """
        page_id = uuid.uuid4().hex
        now = _now_utc()
        effective_bm25 = bm25_text or goal
        # embedding_ref is a stable pointer for when a Chroma integration is added.
        embedding_ref = f"chroma://memory-bank/{page_id}"
        self._conn.execute(
            """
            INSERT INTO memory_pages
                (page_id, goal, domain, intent, constraints, entity_bag,
                 goal_embedding, bm25_text, triplet_sequence_text, embedding_ref,
                 fit_score, read_count,
                 confirmed_read_count, update_count,
                 wiki_summary, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, '', ?, 0.5, 0, 0, 0, '', ?, ?)
            """,
            (
                page_id, goal, domain, intent,
                json.dumps(constraints or []),
                json.dumps(entity_bag or []),
                _vec_to_blob(goal_vec) if goal_vec is not None else None,
                effective_bm25,
                embedding_ref,
                now, now,
            ),
        )
        self._conn.commit()
        self.record_page_history(page_id, "create", f"goal={goal[:80]}")
        return page_id

    def increment_read(self, page_id: str) -> None:
        """Increment read_count and update timestamp for a matched page."""
        self._conn.execute(
            "UPDATE memory_pages SET read_count=read_count+1, updated_at=? WHERE page_id=?",
            (_now_utc(), page_id),
        )
        self._conn.commit()

    def reinforce_page(
        self,
        page_id:      str,
        triplet_keys: List[str],
        nli_scores:   List[float],
    ) -> None:
        """Update page fit_score via MemRL, upsert triplet evidence, rebuild triplet_sequence_text.

        Q_new = Q_old + PAGE_FIT_ALPHA * (avg_nli - Q_old)

        Also rebuilds triplet_sequence_text from the top-scoring accumulated triplet
        keys (max 32, ordered by max_nli_score desc) joined with [SEP].

        Precondition: triplet_keys and nli_scores are the same length or both empty.
        Silently no-ops when the page_id is not found.
        """
        avg_nli = float(np.mean(nli_scores)) if nli_scores else 0.5
        row = self._conn.execute(
            "SELECT fit_score FROM memory_pages WHERE page_id=?", (page_id,)
        ).fetchone()
        if row is None:
            return
        old_fit = row["fit_score"]
        new_fit = max(0.0, min(1.0, old_fit + PAGE_FIT_ALPHA * (avg_nli - old_fit)))

        for key, score in zip(triplet_keys, nli_scores):
            self._conn.execute(
                """
                INSERT INTO page_triplets (page_id, triplet_key, max_nli_score, hit_count)
                VALUES (?, ?, ?, 1)
                ON CONFLICT(page_id, triplet_key) DO UPDATE SET
                    max_nli_score = MAX(excluded.max_nli_score, max_nli_score),
                    hit_count = hit_count + 1
                """,
                (page_id, key, score),
            )

        # Rebuild triplet_sequence_text from top accumulated keys (skill spec: S [SEP] P [SEP] O per component)
        top_keys = [
            r[0] for r in self._conn.execute(
                "SELECT triplet_key FROM page_triplets WHERE page_id=? "
                "ORDER BY max_nli_score DESC, hit_count DESC LIMIT 32",
                (page_id,),
            ).fetchall()
        ]
        # Keys are \x1f-delimited: arxiv_id \x1f subject \x1f predicate \x1f object
        # Build ordered S [SEP] P [SEP] O tokens per skill spec.
        spo_tokens: List[str] = []
        for key in top_keys:
            parts = key.split("\x1f")
            if len(parts) >= 4:
                # parts: [arxiv_id, subject, predicate, object]
                spo_tokens.extend([parts[1], parts[2], parts[3]])
            elif len(parts) == 3:
                spo_tokens.extend(parts)
        triplet_seq = " [SEP] ".join(spo_tokens)

        now = _now_utc()
        self._conn.execute(
            "UPDATE memory_pages SET fit_score=?, update_count=update_count+1, "
            "confirmed_read_count=confirmed_read_count+1, "
            "triplet_sequence_text=?, updated_at=? WHERE page_id=?",
            (new_fit, triplet_seq, now, page_id),
        )
        self._conn.commit()
        self.record_page_history(
            page_id, "reinforce",
            f"fit {old_fit:.3f}→{new_fit:.3f}, avg_nli={avg_nli:.3f}, "
            f"triplets={len(triplet_keys)}",
        )

    def record_page_history(
        self, page_id: str, action: str, details: str = ""
    ) -> None:
        """Append an entry to page_history (append-only)."""
        self._conn.execute(
            "INSERT INTO page_history (page_id, action, details, created_at) "
            "VALUES (?, ?, ?, ?)",
            (page_id, action, details, _now_utc()),
        )
        self._conn.commit()

    # ── Throughlines ───────────────────────────────────────────────────────────

    def get_throughlines(self, page_id: str) -> List[Throughline]:
        """Return all active throughlines for a page, ordered by q_score descending."""
        rows = self._conn.execute(
            "SELECT * FROM throughlines WHERE page_id=? AND status='active' "
            "ORDER BY q_score DESC",
            (page_id,),
        ).fetchall()
        return [
            Throughline(
                throughline_id=row["throughline_id"],
                page_id=row["page_id"],
                claim_text=row["claim_text"],
                supporting_arxiv_ids=json.loads(row["supporting_arxiv_ids"]),
                supporting_triplet_keys=json.loads(row["supporting_triplet_keys"]),
                source_fks=json.loads(row["source_fks"]) if row["source_fks"] else [],
                fact_fks=json.loads(row["fact_fks"]) if row["fact_fks"] else [],
                supporting_fks=json.loads(row["supporting_fks"]) if row["supporting_fks"] else [],
                q_score=row["q_score"],
                merge_score=row["merge_score"] if row["merge_score"] is not None else 0.0,
                merged_into=row["merged_into"],
                canonical=bool(row["canonical"]),
                status=row["status"],
            )
            for row in rows
        ]

    def upsert_throughline(
        self,
        page_id:      str,
        claim_text:   str,
        claim_vec:    Optional[np.ndarray],
        arxiv_ids:    List[str],
        triplet_keys: List[str],
        avg_nli:      float,
    ) -> str:
        """Insert or reinforce a throughline; return its throughline_id.

        Matching logic:
          1. If claim_vec provided: find existing non-deprecated throughline with
             cosine similarity >= THROUGHLINE_SIM_TAU → reinforce via MemRL.
          2. No match: create new throughline.  Status = 'active' if page has
             fewer than 3 active throughlines, else 'competing'.
          3. MemRL Q-update on match: Q_new = Q_old + PAGE_FIT_ALPHA * (avg_nli - Q_old).

        Returns throughline_id of the matched or created record.
        """
        now = _now_utc()
        matched_id: Optional[str] = None

        if claim_vec is not None:
            rows = self._conn.execute(
                "SELECT throughline_id, claim_embedding, q_score FROM throughlines "
                "WHERE page_id=? AND status != 'deprecated'",
                (page_id,),
            ).fetchall()
            best_sim = THROUGHLINE_SIM_TAU
            for row in rows:
                emb_blob = row["claim_embedding"]
                if not emb_blob:
                    continue
                stored = _blob_to_vec(emb_blob)
                if stored.shape != claim_vec.shape:
                    continue
                sim = float(np.dot(claim_vec, stored))
                if sim > best_sim:
                    best_sim, matched_id = sim, row["throughline_id"]

        if matched_id:
            row = self._conn.execute(
                "SELECT q_score, source_fks, fact_fks FROM throughlines WHERE throughline_id=?",
                (matched_id,),
            ).fetchone()
            old_q = row["q_score"]
            new_q = max(0.0, min(1.0, old_q + PAGE_FIT_ALPHA * (avg_nli - old_q)))
            # Merge provenance: union old + new (preserve order, deduplicate)
            old_src  = json.loads(row["source_fks"] or "[]")
            old_fact = json.loads(row["fact_fks"] or "[]")
            merged_src  = list(dict.fromkeys(old_src  + arxiv_ids))
            merged_fact = list(dict.fromkeys(old_fact + triplet_keys))
            merged_all  = list(dict.fromkeys(merged_src + merged_fact))
            self._conn.execute(
                "UPDATE throughlines SET q_score=?, updated_at=?, "
                "supporting_arxiv_ids=?, supporting_triplet_keys=?, "
                "source_fks=?, fact_fks=?, supporting_fks=? "
                "WHERE throughline_id=?",
                (
                    new_q, now,
                    json.dumps(arxiv_ids), json.dumps(triplet_keys),
                    json.dumps(merged_src), json.dumps(merged_fact),
                    json.dumps(merged_all),
                    matched_id,
                ),
            )
            self._conn.commit()
            self.record_throughline_history(
                matched_id, "reinforce", old_q, new_q,
                f"avg_nli={avg_nli:.3f}",
            )
            return matched_id

        # New throughline
        active_count = self._conn.execute(
            "SELECT COUNT(*) FROM throughlines WHERE page_id=? AND status='active'",
            (page_id,),
        ).fetchone()[0]
        status = "active" if active_count < 3 else "competing"
        tid = uuid.uuid4().hex
        all_fks = list(dict.fromkeys(arxiv_ids + triplet_keys))
        self._conn.execute(
            """
            INSERT INTO throughlines
                (throughline_id, page_id, claim_text, claim_embedding,
                 supporting_arxiv_ids, supporting_triplet_keys,
                 supporting_fks, source_fks, fact_fks,
                 q_score, merge_score, canonical, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0.0, 1, ?, ?, ?)
            """,
            (
                tid, page_id, claim_text,
                _vec_to_blob(claim_vec) if claim_vec is not None else None,
                json.dumps(arxiv_ids),
                json.dumps(triplet_keys),
                json.dumps(all_fks),      # supporting_fks = union of source+fact on creation
                json.dumps(arxiv_ids),    # source_fks ← arxiv ids (user-visible provenance)
                json.dumps(triplet_keys), # fact_fks ← triplet ids (normalized evidence)
                avg_nli, status, now, now,
            ),
        )
        self._conn.commit()
        self.record_throughline_history(
            tid, "create", None, avg_nli,
            f"page={page_id[:8]}, status={status}",
        )
        return tid

    def record_throughline_history(
        self,
        throughline_id: str,
        action:         str,
        q_before:       Optional[float],
        q_after:        Optional[float],
        details:        str = "",
    ) -> None:
        """Append an entry to throughline_history (append-only)."""
        self._conn.execute(
            "INSERT INTO throughline_history "
            "(throughline_id, action, q_score_before, q_score_after, details, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (throughline_id, action, q_before, q_after, details, _now_utc()),
        )
        self._conn.commit()

    def merge_throughlines(self, winner_id: str, loser_id: str) -> None:
        """Merge loser into winner: union provenance, deprecate loser.

        - Unions source_fks, fact_fks, supporting_arxiv_ids, supporting_triplet_keys
          from both records; deduplicates while preserving order.
        - Sets loser status='merged', merged_into=winner_id, canonical=0.
        - Applies a final MemRL update to winner using loser's q_score as reward signal.
        - Silently no-ops when either ID is not found.
        """
        now = _now_utc()
        winner = self._conn.execute(
            "SELECT * FROM throughlines WHERE throughline_id=?", (winner_id,)
        ).fetchone()
        loser  = self._conn.execute(
            "SELECT * FROM throughlines WHERE throughline_id=?", (loser_id,)
        ).fetchone()
        if winner is None or loser is None:
            return

        def _union(a: str, b: str) -> str:
            la = json.loads(a or "[]")
            lb = json.loads(b or "[]")
            return json.dumps(list(dict.fromkeys(la + lb)))

        new_src   = _union(winner["source_fks"],             loser["source_fks"])
        new_fact  = _union(winner["fact_fks"],               loser["fact_fks"])
        new_all   = _union(
            _union(winner["supporting_fks"], loser["supporting_fks"]),
            json.dumps(list(dict.fromkeys(
                json.loads(new_src) + json.loads(new_fact)
            ))),
        )
        new_arxiv = _union(winner["supporting_arxiv_ids"],   loser["supporting_arxiv_ids"])
        new_trip  = _union(winner["supporting_triplet_keys"],loser["supporting_triplet_keys"])

        # MemRL update: treat loser's q_score as the reward for this merged evidence
        old_q = winner["q_score"]
        new_q = max(0.0, min(1.0, old_q + PAGE_FIT_ALPHA * (loser["q_score"] - old_q)))

        self._conn.execute(
            "UPDATE throughlines SET q_score=?, source_fks=?, fact_fks=?, "
            "supporting_fks=?, supporting_arxiv_ids=?, supporting_triplet_keys=?, updated_at=? "
            "WHERE throughline_id=?",
            (new_q, new_src, new_fact, new_all, new_arxiv, new_trip, now, winner_id),
        )
        self._conn.execute(
            "UPDATE throughlines SET status='merged', merged_into=?, canonical=0, "
            "updated_at=? WHERE throughline_id=?",
            (winner_id, now, loser_id),
        )
        self._conn.commit()
        self.record_throughline_history(
            winner_id, "merge_winner", old_q, new_q,
            f"absorbed={loser_id[:8]}",
        )
        self.record_throughline_history(
            loser_id, "merge_loser", loser["q_score"], None,
            f"merged_into={winner_id[:8]}",
        )
