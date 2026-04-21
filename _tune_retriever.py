"""_tune_retriever.py — Hyperparameter tuning for SyllogismRetriever.

Protocol (hyper-param tuning skill):
  Tune split  : train (99 QA pairs from eval/data/qa_pairs.json, filtered to CSV)
                subsampled to _TUNE_N_QUERIES=20 for Layer 3 speed (seed=42)
  Holdout     : test  (43 QA pairs, full set)
  Scalar obj  : fit_score = (context_precision + context_recall) / 2

  Precision: rank-sensitive ID-based context precision@K — rewards the correct
             paper being ranked near the top of the retrieved list.
             1/rank if reference found at rank r, else 0.0.
             Higher concentration of correct papers near rank 1 → higher score.
  Recall:    binary per query — 1.0 if the correct paper_id appears anywhere in
             the top_k retrieved set, else 0.0.
  No LLM calls required for evaluation. Each "chunk" is a full paper
  (title + abstract) identified by its arxiv_id.

Tuning modes:

  Standard (default):
    Layer 1 — Semantic blend weights (title / abstract / utility).
               MRR@20 proxy (pure embedding math, no LLM). Optuna TPE, 50 trials.
    Layer 2 — Cache Qwen3-1.7B judge scores for all train queries (~28 min,
               resumable, 3-concurrent workers). Keyed by config_hash.
    Layer 3 — Entailment weight + top_k; structured search (11 evals),
               ID-based fit_score objective (instant, no LLM).

  Judge-free (--disable_judge):
    Single Optuna pass over {title_w, abstract_w, utility_w, top_k}.
    Objective: fit_score directly (no MRR proxy, no LLM).
    entailment_weight fixed to 0.0. Completes in ~2-5 min.

Output: best_retriever_params.json
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
import random
import re
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

# ── Constants ─────────────────────────────────────────────────────────────────

_QA_PATH       = _ROOT / "eval" / "data" / "qa_pairs.json"
_CSV_PATH      = _ROOT / "papers" / "post_processed" / "arxiv_data_with_analysis_cleaned.csv"
_CACHE_DB      = _ROOT / "_tune_cache.db"
_OUTPUT_JSON   = _ROOT / "best_retriever_params.json"

_MRR_K            = 20
_L1_TRIALS        = 50
_DEFAULT_N_PAPERS = 13   # φ-spec: L3 outputs exactly top_k=13 papers to SyllogismRetriever
_TUNE_N_QUERIES   = 20   # subsample for Layer 3 speed; holdout always uses full set

# Layer 3 structured search: 2 params × 2 probes + 1 joint + 5 refinements = 11 evals
_L3_CENTER_EW   = 0.5
_L3_SIGMA_EW    = 0.2
_L3_CENTER_TK   = 5
_L3_SIGMA_TK    = 3
_L3_REFINEMENTS = 5

# ── ID normalisation ──────────────────────────────────────────────────────────

def normalize_paper_id(pid: str) -> str:
    """Normalise QA-pair paper_id (YYMM_NNNNN) to arxiv format (YYMM.NNNNN)."""
    pid = pid.strip().strip('"\'')
    m = re.match(r'^(\d{4})_(\d+)$', pid)
    if m:
        return f"{m.group(1)}.{m.group(2)}"
    return pid


# ── Data loading ──────────────────────────────────────────────────────────────

def load_qa_pairs(csv_ids: Optional[set] = None) -> Tuple[List[dict], List[dict]]:
    """Load train and holdout QA pairs from eval/data/qa_pairs.json.

    Normalises paper_id in every entry.  If csv_ids is provided, only pairs
    whose paper_id is in the CSV are returned (unreachable pairs are dropped).
    """
    with open(_QA_PATH, encoding="utf-8") as fh:
        data = json.load(fh)
    train   = [dict(p, paper_id=normalize_paper_id(p["paper_id"])) for p in data["train"]]
    holdout = [dict(p, paper_id=normalize_paper_id(p["paper_id"])) for p in data["test"]]
    if csv_ids is not None:
        train   = [p for p in train   if p["paper_id"] in csv_ids]
        holdout = [p for p in holdout if p["paper_id"] in csv_ids]
    return train, holdout


# ── CSV / embedding loading ───────────────────────────────────────────────────

def _coerce_utility(v: str) -> str:
    if not v:
        return ""
    v = v.strip()
    if v.startswith("["):
        try:
            items = ast.literal_eval(v)
            if isinstance(items, list):
                return ". ".join(str(x).strip().rstrip(".") for x in items if x) + "."
        except (ValueError, SyntaxError):
            pass
    return v


def _clean(v: str) -> str:
    if not v:
        return ""
    s = str(v).strip()
    if s.lower() in {"nan", "none", "null"}:
        return ""
    return s.replace("\n", " ").strip()


def load_rows() -> List[Dict[str, str]]:
    """Load title/abstract/utility rows from the cleaned CSV."""
    rows: List[Dict[str, str]] = []
    with open(_CSV_PATH, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            arxiv_id = str(row.get("arxiv_id", "")).strip().strip('"')
            if not arxiv_id:
                continue
            rows.append({
                "arxiv_id": arxiv_id,
                "title":    _clean(row.get("title", "")),
                "abstract": _clean(row.get("abstract", "")),
                "utility":  _coerce_utility(_clean(row.get("utility", ""))),
            })
    return rows


def build_field_embeddings(
    rows: List[Dict[str, str]],
    embedder,
) -> Dict[str, np.ndarray]:
    """Encode title/abstract/utility for all rows. Returns dict of (N,D) arrays."""
    embs = {}
    for field in ("title", "abstract", "utility"):
        texts = [r[field] for r in rows]
        print(f"  [embed] Encoding {field} ({len(texts)} docs)...", flush=True)
        embs[field] = embedder.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    return embs


# ── Hashing ───────────────────────────────────────────────────────────────────

def _query_hash(question: str) -> str:
    return hashlib.sha1(question.encode()).hexdigest()[:16]


def _judge_config_hash(blend_weights: Dict[str, float], n_papers: int) -> str:
    """Cache key for judge_cache — identifies the blend config used during Layer 2."""
    key = json.dumps({"blend": blend_weights, "n": n_papers}, sort_keys=True)
    return hashlib.sha1(key.encode()).hexdigest()[:16]



# ── Database ──────────────────────────────────────────────────────────────────

def _migrate_judge_cache(conn: sqlite3.Connection, config_hash: str) -> None:
    """Recreate judge_cache with config_hash in the primary key, migrating existing rows."""
    print("[db] Migrating judge_cache → adding config_hash to PK...", flush=True)
    conn.execute("""
        CREATE TABLE judge_cache_new (
            query_hash  TEXT NOT NULL,
            config_hash TEXT NOT NULL,
            paper_id    TEXT NOT NULL,
            judge_score REAL NOT NULL,
            sem_score   REAL NOT NULL,
            PRIMARY KEY (query_hash, config_hash, paper_id)
        )
    """)
    conn.execute(
        "INSERT INTO judge_cache_new "
        "SELECT query_hash, ?, paper_id, judge_score, sem_score FROM judge_cache",
        (config_hash,),
    )
    n = conn.execute("SELECT COUNT(*) FROM judge_cache_new").fetchone()[0]
    conn.execute("DROP TABLE judge_cache")
    conn.execute("ALTER TABLE judge_cache_new RENAME TO judge_cache")
    conn.commit()
    print(f"  Migrated {n} rows with config_hash={config_hash}", flush=True)


def _migrate_completed_queries(conn: sqlite3.Connection, config_hash: str) -> None:
    """Recreate completed_queries with (query_hash, config_hash) composite PK."""
    conn.execute("""
        CREATE TABLE completed_queries_new (
            query_hash  TEXT NOT NULL,
            config_hash TEXT NOT NULL DEFAULT '',
            PRIMARY KEY (query_hash, config_hash)
        )
    """)
    conn.execute(
        "INSERT INTO completed_queries_new SELECT query_hash, ? FROM completed_queries",
        (config_hash,),
    )
    conn.execute("DROP TABLE completed_queries")
    conn.execute("ALTER TABLE completed_queries_new RENAME TO completed_queries")
    conn.commit()


def _init_cache_db(db_path: Path, l2_config_hash: str = "") -> sqlite3.Connection:
    """Open (or create) the tuning cache, migrating legacy schema if needed.

    Legacy judge_cache used (query_hash, paper_id) as PK with no config_hash.
    New schema adds config_hash to the PK so different Layer 1 configs can
    coexist without overwriting each other.
    """
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")

    jc_cols= [r[1] for r in conn.execute("PRAGMA table_info(judge_cache)").fetchall()]
    if jc_cols and "config_hash" not in jc_cols:
        _migrate_judge_cache(conn, l2_config_hash)

    cq_cols = [r[1] for r in conn.execute("PRAGMA table_info(completed_queries)").fetchall()]
    if cq_cols and "config_hash" not in cq_cols:
        _migrate_completed_queries(conn, l2_config_hash)

    conn.executescript("""
        CREATE TABLE IF NOT EXISTS judge_cache (
            query_hash  TEXT NOT NULL,
            config_hash TEXT NOT NULL,
            paper_id    TEXT NOT NULL,
            judge_score REAL NOT NULL,
            sem_score   REAL NOT NULL,
            PRIMARY KEY (query_hash, config_hash, paper_id)
        );

        CREATE TABLE IF NOT EXISTS completed_queries (
            query_hash  TEXT NOT NULL,
            config_hash TEXT NOT NULL DEFAULT '',
            PRIMARY KEY (query_hash, config_hash)
        );

        CREATE TABLE IF NOT EXISTS ragas_cache (
            query_hash        TEXT NOT NULL,
            config_hash       TEXT NOT NULL,
            sampler_take      TEXT NOT NULL,
            context_precision REAL NOT NULL,
            context_recall    REAL NOT NULL,
            PRIMARY KEY (query_hash, config_hash, sampler_take)
        );

        CREATE TABLE IF NOT EXISTS trials (
            trial_id       INTEGER PRIMARY KEY AUTOINCREMENT,
            layer          INTEGER NOT NULL,
            timestamp      TEXT    NOT NULL,
            config_json    TEXT    NOT NULL,
            tune_precision REAL,
            tune_recall    REAL,
            tune_fit_score REAL
        );
    """)
    conn.commit()
    return conn


# ── MRR evaluation (semantic-only, no LLM) ───────────────────────────────────

def semantic_blend_scores(
    query_emb: np.ndarray,
    field_embeddings: Dict[str, np.ndarray],
    blend_weights: Dict[str, float],
) -> np.ndarray:
    """Compute per-paper blend cosine similarity for one query embedding."""
    scores = np.zeros(len(next(iter(field_embeddings.values()))))
    for field, weight in blend_weights.items():
        if weight > 0:
            scores += weight * (field_embeddings[field] @ query_emb)
    return scores


def rank_paper_ids(
    scores: np.ndarray,
    rows: List[Dict[str, str]],
    n_papers: Optional[int],
) -> List[str]:
    """Return paper IDs sorted by blend score descending (optionally top-n_papers)."""
    order = np.argsort(-scores)
    if n_papers and n_papers > 0:
        order = order[:n_papers]
    return [rows[i]["arxiv_id"] for i in order]


def mrr_at_k(ranked_lists: List[List[str]], gt_ids: List[str], k: int = _MRR_K) -> float:
    """Mean reciprocal rank at k. Returns 0.0 if ground truth never found in top-k."""
    total = 0.0
    for ranked, gt in zip(ranked_lists, gt_ids):
        for rank, pid in enumerate(ranked[:k], start=1):
            if pid == gt:
                total += 1.0 / rank
                break
    return total / len(gt_ids) if gt_ids else 0.0


def eval_semantic_mrr(
    blend_weights: Dict[str, float],
    rows: List[Dict[str, str]],
    field_embeddings: Dict[str, np.ndarray],
    embedder,
    qa_pairs: List[dict],
    n_papers: Optional[int] = None,
) -> float:
    """Compute MRR@k using only semantic blend scores (no LLM)."""
    queries   = [p["question"] for p in qa_pairs]
    gt_ids    = [p["paper_id"] for p in qa_pairs]

    q_embs = embedder.encode(
        queries,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    ranked_lists = []
    for q_emb in q_embs:
        scores = semantic_blend_scores(q_emb, field_embeddings, blend_weights)
        ranked_lists.append(rank_paper_ids(scores, rows, n_papers))

    return mrr_at_k(ranked_lists, gt_ids)


# ── Layer 1: Optuna semantic blend optimisation ───────────────────────────────

def run_layer1(
    rows: List[Dict[str, str]],
    field_embeddings: Dict[str, np.ndarray],
    embedder,
    train_pairs: List[dict],
    n_trials: int = _L1_TRIALS,
) -> Dict[str, float]:
    """Optuna TPE search over (title_w, abstract_frac) for best semantic MRR@20.

    Parameterisation (always non-negative, sums to 1.0):
        title_w     = title_w_param
        abstract_w  = abstract_frac * (1 - title_w)
        utility_w   = (1 - title_w) * (1 - abstract_frac)
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: "optuna.Trial") -> float:
        title_w      = trial.suggest_float("title_w",      0.0, 0.95)
        abstract_frac = trial.suggest_float("abstract_frac", 0.0, 1.0)
        abstract_w   = abstract_frac * (1.0 - title_w)
        utility_w    = (1.0 - title_w) * (1.0 - abstract_frac)

        blend_weights = {
            "title":    title_w,
            "abstract": abstract_w,
            "utility":  utility_w,
        }
        return eval_semantic_mrr(
            blend_weights, rows, field_embeddings, embedder, train_pairs,
        )

    study = optuna.create_study(direction="maximize")
    # Seed with the current default (title=0.4, abstract=0.3, utility=0.3)
    study.enqueue_trial({"title_w": 0.4, "abstract_frac": 0.3 / 0.6})  # abstract_frac = 0.3/(1-0.4)

    print(f"\n[Layer 1] Optuna TPE — {n_trials} trials...")
    t0 = time.time()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    elapsed = time.time() - t0

    best = study.best_trial
    title_w       = best.params["title_w"]
    abstract_frac = best.params["abstract_frac"]
    abstract_w    = abstract_frac * (1.0 - title_w)
    utility_w     = (1.0 - title_w) * (1.0 - abstract_frac)

    result = {
        "title":    round(title_w, 4),
        "abstract": round(abstract_w, 4),
        "utility":  round(utility_w, 4),
    }
    print(f"  Best MRR@{_MRR_K}: {best.value:.4f}  (trial {best.number}, {elapsed:.1f}s)")
    print(f"  Best blend_weights: {result}")
    return result, best.value


# ── Layer 2: Cache LLM judge scores ──────────────────────────────────────────

def run_layer2_cache(
    best_blend_weights: Dict[str, float],
    rows: List[Dict[str, str]],
    field_embeddings: Dict[str, np.ndarray],
    embedder,
    train_pairs: List[dict],
    n_papers: int,
    conn: sqlite3.Connection,
    n_workers: int = 3,
) -> None:
    """Run Qwen3-1.7B judge for all train queries and cache scores to SQLite.

    Resumable: queries already cached for this config_hash are skipped.
    Runs up to n_workers=3 judge calls concurrently (each call = one full query
    with n_papers utility strings).  SQLite writes are serialized on main thread.
    Precondition: conn must be open with the new judge_cache schema.
    """
    import concurrent.futures
    from reasoning.nli_entailment import NLIEntailmentScorer

    config_hash = _judge_config_hash(best_blend_weights, n_papers)

    done_set: set = set(
        row[0] for row in conn.execute(
            "SELECT query_hash FROM completed_queries WHERE config_hash=?",
            (config_hash,),
        )
    )
    pending = [p for p in train_pairs if _query_hash(p["question"]) not in done_set]
    print(f"\n[Layer 2] config_hash={config_hash}  {len(pending)}/{len(train_pairs)} pending")
    if not pending:
        print("  All queries already cached.")
        return

    def _score_one(pair: dict):
        """Score a single query; pure compute, no SQLite. Thread-safe."""
        nli = NLIEntailmentScorer(verbose=False)
        q     = pair["question"]
        qhash = _query_hash(q)

        q_emb = embedder.encode(
            [q], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False,
        )[0]
        scores    = semantic_blend_scores(q_emb, field_embeddings, best_blend_weights)
        order     = np.argsort(-scores)[:n_papers]
        cand_ids  = [rows[idx]["arxiv_id"] for idx in order]
        cand_sems = {rows[idx]["arxiv_id"]: float(scores[idx]) for idx in order}

        sem_vals = np.array([cand_sems[aid] for aid in cand_ids], dtype=float)
        lo, hi   = sem_vals.min(), sem_vals.max()
        span     = (hi - lo) if (hi - lo) > 1e-12 else 1.0
        sem_norm = {aid: (cand_sems[aid] - lo) / span for aid in cand_ids}

        utilities_map: Dict[str, str] = {rows[idx]["arxiv_id"]: rows[idx]["utility"] for idx in order}
        try:
            _entailed, rank_scores = nli.rank_utilities(q, utilities_map)
        except Exception as exc:
            print(f"  [warn] judge failed: {exc}", flush=True)
            rank_scores = {}

        return qhash, cand_ids, rank_scores, sem_norm

    total  = len(train_pairs)
    done_n = len(done_set)
    t_start = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_score_one, pair): pair for pair in pending}
        for fut in concurrent.futures.as_completed(futures):
            t0 = time.time()
            try:
                qhash, cand_ids, rank_scores, sem_norm = fut.result()
            except Exception as exc:
                print(f"  [warn] worker exception: {exc}", flush=True)
                done_n += 1
                continue

            conn.executemany(
                "INSERT OR REPLACE INTO judge_cache VALUES (?,?,?,?,?)",
                [
                    (qhash, config_hash, aid,
                     float(rank_scores.get(aid, 0.0)),
                     float(sem_norm.get(aid, 0.0)))
                    for aid in cand_ids
                ],
            )
            conn.execute(
                "INSERT OR IGNORE INTO completed_queries VALUES (?,?)", (qhash, config_hash)
            )
            conn.commit()

            done_n += 1
            elapsed_q = time.time() - t0
            elapsed_total = time.time() - t_start
            rate = elapsed_total / done_n if done_n else 1
            eta  = rate * (total - done_n)
            print(
                f"  [{done_n}/{total}] {elapsed_q:.1f}s  "
                f"judge selected {len(rank_scores)} papers  ETA {eta/60:.1f}min",
                flush=True,
            )

    print("[Layer 2] Caching complete.")


# ── CachedRetriever ───────────────────────────────────────────────────────────

class _ResultDoc:
    """Minimal document object returned by CachedRetriever.search()."""
    __slots__ = ("content", "arxiv_id")

    def __init__(self, row: Dict[str, str]) -> None:
        self.content   = f"{row['title']}\n\n{row['abstract']}"
        self.arxiv_id  = row["arxiv_id"]


class CachedRetriever:
    """Retriever backed by in-memory blend scores and the SQLite judge cache.

    Used during Layer 3 and holdout to avoid re-running the Qwen3-1.7B judge.
    All candidates are ranked by pure blend score (no hard chain/nonchain partition):
        blend = ew * judge_score + rw * sem_norm
    Papers not in judge_cache get judge_score=0.0 and rank by semantic score alone.

    Precondition: judge_cache populated for judge_config_hash (Layer 2 complete).
    Guarantee: .search(question, top_k) returns at most top_k _ResultDoc objects.
    """

    def __init__(
        self,
        rows: List[Dict[str, str]],
        id_to_idx: Dict[str, int],
        field_embeddings: Dict[str, np.ndarray],
        embedder,
        blend_weights: Dict[str, float],
        n_papers: int,
        entailment_weight: float,
        judge_config_hash: str,
        conn: sqlite3.Connection,
    ) -> None:
        self._rows = rows
        self._id_to_idx = id_to_idx
        self._field_embeddings = field_embeddings
        self._embedder = embedder
        self._blend_weights = blend_weights
        self._n_papers = n_papers
        self._ew = entailment_weight
        self._rw = 1.0 - entailment_weight
        self._judge_hash = judge_config_hash
        self._conn = conn

    def search(self, question: str, top_k: int = 5) -> List[_ResultDoc]:
        q_emb = self._embedder.encode(
            [question], convert_to_numpy=True, normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        scores = semantic_blend_scores(q_emb, self._field_embeddings, self._blend_weights)
        order  = np.argsort(-scores)[:self._n_papers]
        cand_ids = [self._rows[i]["arxiv_id"] for i in order]

        sem_vals = np.array([float(scores[i]) for i in order])
        lo, hi   = sem_vals.min(), sem_vals.max()
        span     = (hi - lo) if (hi - lo) > 1e-12 else 1.0
        sem_norm = {self._rows[i]["arxiv_id"]: float((scores[i] - lo) / span) for i in order}

        qhash = _query_hash(question)
        judge_scores: Dict[str, float] = {
            r[0]: r[1] for r in self._conn.execute(
                "SELECT paper_id, judge_score FROM judge_cache "
                "WHERE query_hash=? AND config_hash=?",
                (qhash, self._judge_hash),
            ).fetchall()
        }

        def _blend(aid: str) -> float:
            return self._ew * judge_scores.get(aid, 0.0) + self._rw * sem_norm.get(aid, 0.0)

        ranked = sorted(cand_ids, key=lambda aid: _blend(aid), reverse=True)[:top_k]
        return [_ResultDoc(self._rows[self._id_to_idx[aid]]) for aid in ranked if aid in self._id_to_idx]


# ── ID-based evaluation (no LLM) ──────────────────────────────────────────────

def _rank_sensitive_precision(retrieved_ids: List[str], reference_id: str) -> float:
    """Rank-sensitive context precision@K for a single reference document.

    Implements the RAGAS formula with exact ID matching:
        precision@K = sum_{k=1}^{K}(precision@k * v_k) / total_relevant_in_top_K
    For a single reference document:
        score = 1/rank  if reference_id found at 1-indexed rank r, else 0.0

    Precondition: retrieved_ids are already normalised (dots, not underscores).
    """
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid == reference_id:
            return 1.0 / rank
    return 0.0


def _persist_trial(
    conn: sqlite3.Connection,
    layer: int,
    config: dict,
    result: dict,
) -> None:
    conn.execute(
        "INSERT INTO trials (layer, timestamp, config_json, tune_precision, tune_recall, tune_fit_score) "
        "VALUES (?,?,?,?,?,?)",
        (
            layer,
            datetime.now(timezone.utc).isoformat(),
            json.dumps(config),
            result.get("precision"),
            result.get("recall"),
            result.get("fit_score"),
        ),
    )
    conn.commit()


def eval_retrieval_config(
    config: dict,
    qa_pairs: List[dict],
    rows: List[Dict[str, str]],
    id_to_idx: Dict[str, int],
    field_embeddings: Dict[str, np.ndarray],
    embedder,
    conn: sqlite3.Connection,
    verbose: bool = True,
) -> Dict[str, object]:
    """Evaluate a retrieval config using ID-based precision and recall.

    No LLM calls required. Each query's ground-truth is the paper_id field
    of the QA pair. Precision is rank-sensitive (1/rank if found); recall is
    binary (1 if paper found anywhere in top_k, else 0).

    fit_score = (mean_precision + mean_recall) / 2

    Precondition: qa_pairs have normalised paper_id fields (dots, not underscores).
    Guarantee: deterministic; calling twice with same config returns same result.
    """
    top_k             = config["top_k"]
    blend_weights     = config["blend_weights"]
    n_papers          = config["n_papers"]
    entailment_weight = config["entailment_weight"]
    judge_hash        = _judge_config_hash(blend_weights, n_papers)

    retriever = CachedRetriever(
        rows=rows,
        id_to_idx=id_to_idx,
        field_embeddings=field_embeddings,
        embedder=embedder,
        blend_weights=blend_weights,
        n_papers=n_papers,
        entailment_weight=entailment_weight,
        judge_config_hash=judge_hash,
        conn=conn,
    )

    precisions: List[float] = []
    recalls: List[float]    = []
    for pair in qa_pairs:
        ref_id = normalize_paper_id(pair["paper_id"])
        docs   = retriever.search(pair["question"], top_k=top_k)
        retrieved_ids = [normalize_paper_id(d.arxiv_id) for d in docs]
        precisions.append(_rank_sensitive_precision(retrieved_ids, ref_id))
        recalls.append(1.0 if ref_id in retrieved_ids else 0.0)

    mean_p    = sum(precisions) / len(precisions) if precisions else 0.0
    mean_r    = sum(recalls)    / len(recalls)    if recalls    else 0.0
    fit_score = (mean_p + mean_r) / 2

    if verbose:
        print(
            f"    prec={mean_p:.4f} rec={mean_r:.4f} fit={fit_score:.4f}  (n={len(precisions)})",
            flush=True,
        )
    return {"precision": mean_p, "recall": mean_r, "fit_score": fit_score}


# ── Layer 3: Structured search over entailment_weight + top_k ────────────────

def run_layer3_structured(
    best_blend: Dict[str, float],
    n_papers: int,
    train_pairs: List[dict],
    conn: sqlite3.Connection,
    rows: List[Dict[str, str]],
    id_to_idx: Dict[str, int],
    field_embeddings: Dict[str, np.ndarray],
    embedder,
    n_refinements: int = _L3_REFINEMENTS,
) -> Tuple[float, int, float]:
    """Structured search (2×2+7 = 11 evals) over (entailment_weight, top_k).

    Objective: ID-based fit_score = (rank_sensitive_precision + binary_recall) / 2.
    Deterministic — no LLM calls, no sampler takes.

    Returns (best_entailment_weight, best_top_k, best_fit_score).
    """
    rng        = np.random.default_rng(42)
    eval_cache: Dict[Tuple[float, int], float] = {}

    def _eval(ew: float, tk: int) -> float:
        ew = round(float(np.clip(ew, 0.0, 1.0)), 4)
        tk = int(np.clip(tk, 3, n_papers))
        if (ew, tk) in eval_cache:
            return eval_cache[(ew, tk)]
        cfg = {
            "blend_weights":     best_blend,
            "n_papers":          n_papers,
            "entailment_weight": ew,
            "top_k":             tk,
        }
        result = eval_retrieval_config(cfg, train_pairs, rows, id_to_idx, field_embeddings, embedder, conn)
        _persist_trial(conn, layer=3, config=cfg, result=result)
        score = result["fit_score"]
        eval_cache[(ew, tk)] = score
        print(f"    → fit={score:.4f} (prec={result['precision']:.3f} rec={result['recall']:.3f})", flush=True)
        return score

    # Baseline
    print(f"\n[Layer 3] Structured search — center: ew={_L3_CENTER_EW} top_k={_L3_CENTER_TK}", flush=True)
    baseline = _eval(_L3_CENTER_EW, _L3_CENTER_TK)

    # One-at-a-time directional probes
    probes = [
        ("ew+", _L3_CENTER_EW + _L3_SIGMA_EW, _L3_CENTER_TK, False),
        ("ew-", _L3_CENTER_EW - _L3_SIGMA_EW, _L3_CENTER_TK, False),
        ("tk+", _L3_CENTER_EW, _L3_CENTER_TK + _L3_SIGMA_TK, True),
        ("tk-", _L3_CENTER_EW, _L3_CENTER_TK - _L3_SIGMA_TK, True),
    ]
    probe_scores: Dict[str, float] = {}
    for name, ew, tk, _ in probes:
        print(f"  [L3] probe {name}: ew={ew:.2f} top_k={tk}", flush=True)
        probe_scores[name] = _eval(ew, tk)

    # Winning direction per factor (beats baseline)
    winner_ew = _L3_CENTER_EW
    if probe_scores["ew+"] >= probe_scores["ew-"] and probe_scores["ew+"] > baseline:
        winner_ew = round(float(np.clip(_L3_CENTER_EW + _L3_SIGMA_EW, 0.0, 1.0)), 4)
    elif probe_scores["ew-"] > baseline:
        winner_ew = round(float(np.clip(_L3_CENTER_EW - _L3_SIGMA_EW, 0.0, 1.0)), 4)

    winner_tk = _L3_CENTER_TK
    if probe_scores["tk+"] >= probe_scores["tk-"] and probe_scores["tk+"] > baseline:
        winner_tk = int(np.clip(_L3_CENTER_TK + _L3_SIGMA_TK, 3, n_papers))
    elif probe_scores["tk-"] > baseline:
        winner_tk = int(np.clip(_L3_CENTER_TK - _L3_SIGMA_TK, 3, n_papers))

    # Joint candidate
    print(f"  [L3] joint: ew={winner_ew:.2f} top_k={winner_tk}", flush=True)
    _eval(winner_ew, winner_tk)

    # Identify current best across all evaluated configs
    best_score = max(eval_cache.values())
    best_ew, best_tk = max(eval_cache, key=lambda k: eval_cache[k])

    # Local refinements around best
    print(f"  [L3] refinements ×{n_refinements} around ew={best_ew:.3f} top_k={best_tk}", flush=True)
    for _ in range(n_refinements):
        ew_r = round(float(np.clip(best_ew + rng.uniform(-_L3_SIGMA_EW / 2, _L3_SIGMA_EW / 2), 0.0, 1.0)), 4)
        tk_r = int(np.clip(best_tk + int(rng.integers(-2, 3)), 3, n_papers))
        s = _eval(ew_r, tk_r)
        if s > best_score:
            best_score = s
            best_ew    = ew_r
            best_tk    = tk_r

    print(f"\n[Layer 3] Best: ew={best_ew:.4f} top_k={best_tk}  fit_score={best_score:.4f}")
    return best_ew, best_tk, best_score


# ── Deterministic tuning (--disable_judge) ───────────────────────────────────

def run_deterministic_tuning(
    rows: List[Dict[str, str]],
    id_to_idx: Dict[str, int],
    field_embeddings: Dict[str, np.ndarray],
    embedder,
    tune_pairs: List[dict],
    n_papers: int,
    conn: sqlite3.Connection,
    n_trials: int = 50,
    warm_start: Optional[Dict[str, float]] = None,
) -> Tuple[Dict[str, float], int, float]:
    """Tune blend weights + top_k without LLM judge (ew=0, pure semantic).

    Pre-encodes all query embeddings once outside the trial loop, then runs
    Optuna TPE over:
        title_w, abstract_w  (utility_w = 1 - title_w - abstract_w, must be ≥0.05)
        top_k                (3 .. n_papers)
    Objective: fit_score = (rank_sensitive_precision + binary_recall) / 2

    Rank-sensitive precision rewards concentration of correct papers near rank 1:
        score = 1/rank  if reference found at rank r, else 0.0

    No SQL queries, no LLM calls — fully deterministic embedding math.
    Completes in ~2-5 min for 20 queries × 50 trials.

    Args:
        warm_start: Optional dict with saved blend_weights to enqueue as first trial.
    Returns:
        (best_blend_weights, best_top_k, best_fit_score)
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Pre-encode query embeddings and normalise reference IDs once
    print(f"[Det. Tuning] Pre-encoding {len(tune_pairs)} query embeddings...", flush=True)
    questions = [p["question"] for p in tune_pairs]
    query_embs = embedder.encode(
        questions, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False,
    )
    ref_ids = [normalize_paper_id(p["paper_id"]) for p in tune_pairs]
    csv_ids_set = {r["arxiv_id"] for r in rows}
    # Filter to pairs whose reference paper is actually in the CSV
    valid = [(q_emb, ref_id) for q_emb, ref_id in zip(query_embs, ref_ids) if ref_id in csv_ids_set]
    if not valid:
        print("[Det. Tuning] No valid pairs — aborting.")
        return {"title": 0.3237, "abstract": 0.5803, "utility": 0.096}, 5, 0.0
    q_embs_arr = np.array([v[0] for v in valid])
    valid_refs  = [v[1] for v in valid]

    def _fast_fit(title_w: float, abstract_w: float, top_k: int) -> float:
        utility_w = 1.0 - title_w - abstract_w
        blend = {"title": title_w, "abstract": abstract_w, "utility": utility_w}
        precisions, recalls = [], []
        for q_emb, ref_id in zip(q_embs_arr, valid_refs):
            scores = semantic_blend_scores(q_emb, field_embeddings, blend)
            order  = np.argsort(-scores)[:n_papers]
            ranked = [rows[i]["arxiv_id"] for i in order[:top_k]]
            precisions.append(_rank_sensitive_precision(ranked, ref_id))
            recalls.append(1.0 if ref_id in ranked else 0.0)
        prec = float(np.mean(precisions))
        rec  = float(np.mean(recalls))
        return (prec + rec) / 2.0

    def objective(trial: "optuna.Trial") -> float:
        title_w    = trial.suggest_float("title",    0.05, 0.70)
        abstract_w = trial.suggest_float("abstract", 0.20, 0.75)
        utility_w  = 1.0 - title_w - abstract_w
        if utility_w < 0.05 or utility_w > 0.50:
            raise optuna.exceptions.TrialPruned()
        top_k = trial.suggest_int("top_k", 3, n_papers)
        score = _fast_fit(title_w, abstract_w, top_k)
        print(
            f"  [trial {trial.number:>3}] title={title_w:.3f} abs={abstract_w:.3f} "
            f"util={utility_w:.3f} tk={top_k}  fit={score:.4f}",
            flush=True,
        )
        return score

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    if warm_start and "blend_weights" in warm_start:
        bw = warm_start["blend_weights"]
        study.enqueue_trial({
            "title":    bw.get("title",    0.3237),
            "abstract": bw.get("abstract", 0.5803),
            "top_k":    warm_start.get("top_k", 5),
        })

    print(f"[Det. Tuning] Optuna TPE — {n_trials} trials × {len(valid)} queries...", flush=True)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_trial
    title_w    = best.params["title"]
    abstract_w = best.params["abstract"]
    utility_w  = 1.0 - title_w - abstract_w
    best_blend = {
        "title":    round(title_w,    4),
        "abstract": round(abstract_w, 4),
        "utility":  round(utility_w,  4),
    }
    best_tk  = best.params["top_k"]
    best_fit = best.value

    cfg = {"blend_weights": best_blend, "n_papers": n_papers, "entailment_weight": 0.0, "top_k": best_tk}
    _persist_trial(conn, layer=1, config=cfg, result={"precision": 0.0, "recall": 0.0, "fit_score": best_fit})

    print(f"\n[Det. Tuning] Best: blend={best_blend}  top_k={best_tk}  fit_score={best_fit:.4f}")
    return best_blend, best_tk, best_fit


# ── Holdout evaluation ────────────────────────────────────────────────────────

def run_holdout_eval(
    best_blend: Dict[str, float],
    best_ew: float,
    best_tk: int,
    n_papers: int,
    holdout_pairs: List[dict],
    conn: sqlite3.Connection,
    rows: List[Dict[str, str]],
    id_to_idx: Dict[str, int],
    field_embeddings: Dict[str, np.ndarray],
    embedder,
) -> Dict[str, object]:
    """Evaluate the final config on holdout using the ID-based objective.

    Holdout is evaluated once, after all tuning layers are frozen.
    """
    print(f"\n[Holdout] Evaluating {len(holdout_pairs)} queries...")
    config = {
        "blend_weights":     best_blend,
        "n_papers":          n_papers,
        "entailment_weight": round(best_ew, 4),
        "top_k":             best_tk,
    }
    result = eval_retrieval_config(config, holdout_pairs, rows, id_to_idx, field_embeddings, embedder, conn)
    print(f"  Holdout fit_score:   {result['fit_score']:.4f}")
    print(f"  Holdout precision:   {result['precision']:.4f}")
    print(f"  Holdout recall:      {result['recall']:.4f}")
    return result


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Tune SyllogismRetriever hyperparameters")
    ap.add_argument("--skip_layer1",     action="store_true")
    ap.add_argument("--skip_layer2",     action="store_true")
    ap.add_argument("--skip_layer3",     action="store_true")
    ap.add_argument("--skip_holdout",    action="store_true")
    ap.add_argument("--disable_judge",   action="store_true",
                    help="Skip LLM judge entirely. Tune blend weights + top_k "
                         "deterministically (ew=0). Completes in ~2-5 min.")
    ap.add_argument("--n_papers",  type=int, default=_DEFAULT_N_PAPERS)
    ap.add_argument("--l1_trials", type=int, default=_L1_TRIALS)
    args = ap.parse_args()

    print("[init] Loading QA pairs and CSV...")
    rows = load_rows()
    csv_ids = {r["arxiv_id"] for r in rows}
    train_pairs, holdout_pairs = load_qa_pairs(csv_ids=csv_ids)
    id_to_idx = {r["arxiv_id"]: i for i, r in enumerate(rows)}
    print(f"  {len(rows)} papers  train={len(train_pairs)}  holdout={len(holdout_pairs)}")

    print("[init] Loading embedder and building field embeddings...")
    from sentence_transformers import SentenceTransformer
    embedder         = SentenceTransformer("all-MiniLM-L6-v2")
    field_embeddings = build_field_embeddings(rows, embedder)

    saved: dict = {}
    if _OUTPUT_JSON.exists():
        with open(_OUTPUT_JSON, encoding="utf-8") as fh:
            saved = json.load(fh)

    # Subsample for tuning (same 20 across all modes, seed=42)
    random.seed(42)
    tune_pairs = random.sample(train_pairs, min(_TUNE_N_QUERIES, len(train_pairs)))

    # Open DB (needed even in disable_judge mode to persist trials)
    l2_config_hash = _judge_config_hash(saved.get("blend_weights", {}), args.n_papers)
    conn = _init_cache_db(_CACHE_DB, l2_config_hash=l2_config_hash)

    if args.disable_judge:
        # ── Judge-free path: single deterministic Optuna pass ──────────────
        print(f"\n[Det. Tuning] Tuning on {len(tune_pairs)}/{len(train_pairs)} queries (seed=42)")
        best_blend, best_tk, l_fit = run_deterministic_tuning(
            rows, id_to_idx, field_embeddings, embedder,
            tune_pairs, args.n_papers, conn,
            n_trials=args.l1_trials,
            warm_start=saved if saved else None,
        )
        best_ew = 0.0
        l1_mrr  = 0.0
        l3_fit  = l_fit

    else:
        # ── Standard layerwise path ─────────────────────────────────────────
        # Layer 1: Semantic blend (MRR@20 proxy)
        if args.skip_layer1 and "blend_weights" in saved:
            best_blend = saved["blend_weights"]
            l1_mrr     = saved.get("tune_sem_mrr20", 0.0)
            print(f"[Layer 1] Using saved: {best_blend}  MRR@{_MRR_K}={l1_mrr:.4f}")
        else:
            best_blend, l1_mrr = run_layer1(rows, field_embeddings, embedder, train_pairs, args.l1_trials)

        # Reopen DB with correct config_hash for the chosen blend
        conn.close()
        l2_config_hash = _judge_config_hash(best_blend, args.n_papers)
        conn = _init_cache_db(_CACHE_DB, l2_config_hash=l2_config_hash)

        # Layer 2: LLM judge caching
        if not args.skip_layer2:
            run_layer2_cache(best_blend, rows, field_embeddings, embedder,
                             train_pairs, args.n_papers, conn)
        else:
            print("[Layer 2] Skipped (--skip_layer2)")

        # Layer 3: Structured search — entailment_weight + top_k
        print(f"[Layer 3] Tuning on {len(tune_pairs)}/{len(train_pairs)} queries (seed=42)")
        if not args.skip_layer3:
            best_ew, best_tk, l3_fit = run_layer3_structured(
                best_blend, args.n_papers, tune_pairs, conn,
                rows, id_to_idx, field_embeddings, embedder,
            )
        else:
            best_ew = saved.get("entailment_weight", _L3_CENTER_EW)
            best_tk = saved.get("top_k",             _L3_CENTER_TK)
            l3_fit  = saved.get("tune_fit_score",    0.0)
            print(f"[Layer 3] Skipped — ew={best_ew} top_k={best_tk}")

    # ── Holdout ──────────────────────────────────────────────────────────────
    if not args.skip_holdout:
        h = run_holdout_eval(best_blend, best_ew, best_tk, args.n_papers,
                             holdout_pairs, conn, rows, id_to_idx, field_embeddings, embedder)
        holdout_fit  = h["fit_score"]
        holdout_prec = h["precision"]
        holdout_rec  = h["recall"]
    else:
        holdout_fit  = saved.get("holdout_fit_score",  0.0)
        holdout_prec = saved.get("holdout_precision",  0.0)
        holdout_rec  = saved.get("holdout_recall",     0.0)
        print("[Holdout] Skipped")

    conn.close()

    output = {
        "blend_weights":     best_blend,
        "n_papers":          args.n_papers,
        "entailment_weight": round(best_ew, 4),
        "retrieval_weight":  round(1.0 - best_ew, 4),
        "top_k":             best_tk,
        "tune_sem_mrr20":    round(l1_mrr, 4),
        "tune_fit_score":    round(l3_fit, 4),
        "holdout_fit_score": round(holdout_fit, 4),
        "holdout_precision": round(holdout_prec, 4),
        "holdout_recall":    round(holdout_rec, 4),
    }

    with open(_OUTPUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2)
    print(f"\n[done] Best params → {_OUTPUT_JSON}")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
