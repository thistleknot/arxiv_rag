"""
nli_graph_retriever.py — BM25 + NLI graph expansion over the KG cache.

Replaces the Ollama-dependent GraphRetriever in Stage 7 of the syllogism
pipeline.  Instead of live LLM extraction, this module:

  1. Loads all pre-built KG cache JSON files (graph/kg_cache/*.json).
  2. Builds a BM25 index over spaCy-lemmatized triplet fields
     (subject lemma + predicate lemma + object lemma).
  3. At query time:
       a. BM25-retrieve top candidates from the index, blended with triplet
          confidence: triplet_score = bm25_score * confidence.
       b. Optionally filter to a specific set of paper IDs.
       c. Verbalize each triplet → natural language premise.
       d. NLI-score premises against the query using DeBERTa cross-encoder.
       e. Return top NLI-ranked premises, grouped by paper.
  4. After retrieval, reinforce_from_last_query() applies MemRL Q-updates:
       confidence_new = confidence_old + α * (r_nli - confidence_old)
     Triplets that consistently support correct entailment gain confidence;
     those that score low decay.  Updates persist to SQLite.

Design follows the agentic_kg_memory skill spec:
  - Index-side: lemma columns (synset/hypernym expansion deferred to v2).
  - BM25 handles IDF weighting; stop words are NOT stripped manually.
  - Epistemic tag: KG-cache triplets default to Observed (confidence 1.0).
  - Verbalization converts raw SPO text to readable premise strings before
    passing to NLI — raw SPO concatenation degrades cross-encoder performance.
  - Triplet confidence is mutable: initialized from epistemic prior, updated
    by downstream NLI signal via MemRL Q-learning rule.

Skill reference: .copilot/skills/agentic_kg_memory/SKILL.md

Usage:
    retriever = NLIGraphRetriever()
    premises_by_paper = retriever.retrieve_context_by_paper(
        query="NLI reasoning agentic memory",
        paper_ids=["2412.17029", "2502.03283"],
        top_k_per_paper=3,
    )
    retriever.reinforce_from_last_query()  # update confidence from this pass
"""

from __future__ import annotations

import json
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from constants import (
    BM25_MIN_SCORE,
    BM25_TOP_K,
    EPISTEMIC_OBSERVED,
    KG_CONFIDENCE_DB,
    MEMRL_ALPHA,
    NLI_ENTAIL_IDX,
    NLI_MODEL,
    TRIPLET_REINFORCE_TAU,
    TRIPLET_WEAKEN_TAU,
)

_KG_CACHE_DIR = Path(__file__).resolve().parent / "kg_cache"

# ── Relation-type → natural language verb phrase ──────────────────────────────
# Handles the most common relation types produced by the KG extractor prompt.
_REL_VERBS: Dict[str, str] = {
    "uses":                "uses",
    "use":                 "uses",
    "applied_to":          "is applied to",
    "based_on":            "is based on",
    "is_based_on":         "is based on",
    "extends":             "extends",
    "outperforms":         "outperforms",
    "achieves":            "achieves",
    "evaluates_on":        "evaluates on",
    "introduces":          "introduces",
    "proposes":            "proposes",
    "leverages":           "leverages",
    "consists_of":         "consists of",
    "enables":             "enables",
    "improves":            "improves",
    "improves_on":         "improves on",
    "reduces":             "reduces",
    "addresses":           "addresses",
    "requires":            "requires",
    "trained_on":          "is trained on",
    "compared_to":         "is compared to",
    "integrates":          "integrates",
    "combines":            "combines",
    "builds_on":           "builds on",
    "applied_in":          "is applied in",
}


def _verbalize(source: str, relation_type: str, target: str) -> str:
    """Convert an SPO triplet into a readable English premise string."""
    verb = _REL_VERBS.get(
        relation_type.lower().strip(),
        relation_type.replace("_", " ").lower().strip(),
    )
    return f"{source} {verb} {target}.".strip()


@dataclass
class TripletRecord:
    """One entry in the BM25 index.

    epistemic  — static extraction-time prior (OBSERVED=1.0, INFERRED=0.5).
    confidence — mutable learned posterior; starts at epistemic, updated by
                 downstream NLI signal via MemRL Q-rule.
    """
    arxiv_id:      str
    source:        str
    relation_type: str
    target:        str
    premise:       str          # verbalized natural-language string
    tokens:        List[str]    # spaCy-lemmatized tokens for BM25
    epistemic:     float = EPISTEMIC_OBSERVED
    confidence:    float = field(default=0.0, init=False)  # set in __post_init__

    def __post_init__(self) -> None:
        self.confidence = self.epistemic


class NLIGraphRetriever:
    """
    BM25 + DeBERTa NLI graph expansion over the pre-built KG triplet cache.

    Retrieval blends BM25 relevance with per-triplet confidence:
        triplet_score = bm25_score * confidence

    Confidence starts at the epistemic prior (1.0 for observed facts) and
    is updated after each query via reinforce_from_last_query().  Confidence
    values persist to SQLite so they accumulate across sessions.

    The index is built lazily on first use and cached in memory for the
    lifetime of the object.  Re-instantiate to pick up new cache files.

    Args:
        cache_dir:        Path to graph/kg_cache directory.
        nli_model:        HuggingFace cross-encoder model name.
        top_k_bm25:       BM25 candidate pool before NLI re-ranking.
        min_bm25_score:   Minimum raw BM25 score to include in NLI pass.
        confidence_db:    SQLite path for triplet confidence persistence.
    """

    def __init__(
        self,
        cache_dir:      Optional[Path] = None,
        nli_model:      str            = NLI_MODEL,
        top_k_bm25:     int            = BM25_TOP_K,
        min_bm25_score: float          = BM25_MIN_SCORE,
        confidence_db:  Optional[Path] = None,
    ):
        self._cache_dir      = cache_dir or _KG_CACHE_DIR
        self._nli_model      = nli_model
        self._top_k_bm25     = top_k_bm25
        self._min_bm25_score = min_bm25_score
        self._confidence_db  = confidence_db or KG_CONFIDENCE_DB

        self._records:      List[TripletRecord]             = []
        self._last_scored:  List[Tuple[TripletRecord, float]] = []
        self._bm25      = None   # rank_bm25.BM25Okapi — lazy
        self._nlp       = None   # spaCy — lazy
        self._encoder   = None   # sentence_transformers CrossEncoder — lazy
        self._index_ok  = False

    # ── Public API ────────────────────────────────────────────────────────────

    def retrieve_context_by_paper(
        self,
        query:           str,
        paper_ids:       Optional[List[str]] = None,
        top_k_per_paper: int                 = 3,
    ) -> Dict[str, List[str]]:
        """
        Return the top NLI-entailed premises per paper.

        Args:
            query:           The retrieval query / hypothesis string.
            paper_ids:       If given, only triplets from these papers are returned.
                             Pass None to search across the full corpus.
            top_k_per_paper: How many premises to return per paper.

        Returns:
            Dict[arxiv_id, List[premise_string]] — ordered by NLI entailment score.
        """
        self._ensure_index()
        if not self._records:
            return {}

        candidates = self._bm25_candidates(query, paper_ids)
        if not candidates:
            return {}

        scored = self._nli_score(query, candidates)
        return self._group_top_k(scored, top_k_per_paper)

    def retrieve_context_str(
        self,
        query:     str,
        paper_ids: Optional[List[str]] = None,
        top_k:     int                 = 10,
    ) -> str:
        """
        Flat string of top NLI-ranked premises (for backward-compat with Stage 7).

        Returns newline-separated premise strings, or "" when no matches.
        """
        by_paper = self.retrieve_context_by_paper(query, paper_ids, top_k_per_paper=top_k)
        lines = []
        for aid, premises in by_paper.items():
            for p in premises:
                lines.append(f"[{aid}] {p}")
        return "\n".join(lines)

    def index_size(self) -> int:
        """Number of triplets in the index."""
        self._ensure_index()
        return len(self._records)

    # ── Index construction ────────────────────────────────────────────────────

    def _ensure_index(self) -> None:
        if self._index_ok:
            return
        self._build_index()
        self._index_ok = True

    def _build_index(self) -> None:
        """Load all KG cache JSONs and build BM25Okapi index."""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            return

        self._nlp = self._load_spacy()

        self._records = []
        for json_path in sorted(self._cache_dir.glob("*.json")):
            arxiv_id = json_path.stem.replace("_", ".")
            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue

            for rel in data.get("relations", []):
                src = str(rel.get("source", "")).strip()
                rel_type = str(rel.get("relation_type", "")).strip()
                tgt = str(rel.get("target", "")).strip()
                if not (src and rel_type and tgt):
                    continue

                premise = _verbalize(src, rel_type, tgt)
                tokens  = self._lemmatize(f"{src} {rel_type} {tgt}")
                self._records.append(TripletRecord(
                    arxiv_id      = arxiv_id,
                    source        = src,
                    relation_type = rel_type,
                    target        = tgt,
                    premise       = premise,
                    tokens        = tokens,
                ))

        if not self._records:
            return

        corpus = [r.tokens for r in self._records]
        self._bm25 = BM25Okapi(corpus)
        self._load_confidence_db()

    def _lemmatize(self, text: str) -> List[str]:
        """Return spaCy lemma tokens (no stop-word stripping — BM25 IDF handles it)."""
        if self._nlp is None:
            return text.lower().split()
        doc = self._nlp(text.lower())
        return [tok.lemma_ for tok in doc if not tok.is_space and not tok.is_punct]

    @staticmethod
    def _load_spacy():
        try:
            import spacy
            return spacy.load("en_core_web_sm", disable=["ner", "parser"])
        except Exception:
            return None

    # ── BM25 retrieval ────────────────────────────────────────────────────────

    def _bm25_candidates(
        self,
        query:     str,
        paper_ids: Optional[List[str]],
    ) -> List[TripletRecord]:
        """BM25 top-k, blended with triplet confidence, then filtered to paper_ids.

        triplet_score = bm25_score * confidence  (agentic_kg_memory: Ranking Surface)
        Triplets below min_bm25_score are excluded before confidence blending.
        """
        if self._bm25 is None:
            return []

        query_tokens = self._lemmatize(query)
        raw_scores   = self._bm25.get_scores(query_tokens)

        # Filter below threshold first, then sort by confidence-blended score.
        pairs = [
            (rec, bm25)
            for rec, bm25 in zip(self._records, raw_scores.tolist())
            if bm25 >= self._min_bm25_score
        ]
        pairs.sort(key=lambda x: x[1] * x[0].confidence, reverse=True)

        id_set = set(paper_ids) if paper_ids else None
        candidates: List[TripletRecord] = []
        for rec, _score in pairs:
            if id_set is not None and rec.arxiv_id not in id_set:
                continue
            candidates.append(rec)
            if len(candidates) >= self._top_k_bm25:
                break

        return candidates

    # ── NLI scoring ──────────────────────────────────────────────────────────

    def _load_encoder(self) -> None:
        if self._encoder is None:
            from sentence_transformers import CrossEncoder
            self._encoder = CrossEncoder(self._nli_model)

    def _nli_score(
        self,
        query:      str,
        candidates: List[TripletRecord],
    ) -> List[Tuple[TripletRecord, float]]:
        """
        Score each candidate premise against the query via DeBERTa NLI.

        Returns list of (TripletRecord, entailment_prob) sorted descending.
        Triplet confidence (mutable learned posterior) weights the raw NLI
        probability: effective_score = entail_prob * confidence.

        Also stores result in self._last_scored for reinforce_from_last_query().
        """
        self._load_encoder()

        pairs = [[query, rec.premise] for rec in candidates]
        raw   = np.array(self._encoder.predict(pairs), dtype=np.float32)
        if raw.ndim == 1:
            raw = raw[np.newaxis, :]

        # Softmax for stable probabilities
        raw   = raw - raw.max(axis=1, keepdims=True)
        exp_r = np.exp(raw)
        probs = exp_r / exp_r.sum(axis=1, keepdims=True)
        entail_probs = probs[:, NLI_ENTAIL_IDX]

        scored = [
            (rec, float(ep) * rec.confidence)
            for rec, ep in zip(candidates, entail_probs)
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        self._last_scored = scored
        return scored

    # ── Grouping ──────────────────────────────────────────────────────────────

    @staticmethod
    def _group_top_k(
        scored:         List[Tuple[TripletRecord, float]],
        top_k_per_paper: int,
    ) -> Dict[str, List[str]]:
        """Group top-k premises per paper, preserving global NLI rank order."""
        result:   Dict[str, List[str]] = {}
        counters: Dict[str, int]       = {}
        for rec, _score in scored:
            aid = rec.arxiv_id
            n   = counters.get(aid, 0)
            if n >= top_k_per_paper:
                continue
            result.setdefault(aid, []).append(rec.premise)
            counters[aid] = n + 1
        return result

    # ── MemRL confidence update ───────────────────────────────────────────────

    def reinforce_from_last_query(self) -> None:
        """Apply MemRL Q-update to triplets scored during the last retrieval pass.

        Precondition: retrieve_context_by_paper() must have been called at least
        once to populate self._last_scored.
        """
        self.reinforce_from_scores(self._last_scored)

    def get_last_triplet_keys(
        self, threshold: float = 0.5
    ) -> Tuple[List[str], List[float]]:
        """Return (triplet_keys, nli_scores) for triplets from the last retrieval pass.

        Only includes triplets with effective_score >= threshold.
        Effective score = entail_prob * confidence (as stored in _last_scored).
        Call immediately after retrieve_context_by_paper() to capture results
        before a second pass overwrites _last_scored.
        """
        keys: List[str] = []
        scores: List[float] = []
        for rec, score in self._last_scored:
            if score >= threshold:
                keys.append(
                    self._triplet_key(
                        rec.arxiv_id, rec.source, rec.relation_type, rec.target
                    )
                )
                scores.append(score)
        return keys, scores

    def reinforce_from_scores(
        self, scored: List[Tuple[TripletRecord, float]]
    ) -> None:
        """Apply MemRL Q-update to an explicit (TripletRecord, nli_score) list.

        Q_new = Q_old + MEMRL_ALPHA * (r_nli - Q_old)

        Reinforces triplets with NLI score >= REINFORCE_TAU.
        Weakens triplets with NLI score <= WEAKEN_TAU.
        Neutral band (WEAKEN_TAU, REINFORCE_TAU): no update.
        Clamps confidence to [0.0, 1.0].  Persists updates to SQLite.
        """
        if not scored:
            return

        updates: List[Tuple[str, str, str, str, float]] = []
        for rec, nli_score in scored:
            if nli_score >= TRIPLET_REINFORCE_TAU or nli_score <= TRIPLET_WEAKEN_TAU:
                new_conf = rec.confidence + MEMRL_ALPHA * (nli_score - rec.confidence)
                rec.confidence = max(0.0, min(1.0, new_conf))
                updates.append((
                    rec.arxiv_id, rec.source, rec.relation_type,
                    rec.target, rec.confidence,
                ))

        if updates:
            self._persist_confidence(updates)

    @staticmethod
    def _triplet_key(
        arxiv_id: str, source: str, relation_type: str, target: str
    ) -> str:
        """Stable lookup key for a triplet across sessions."""
        return f"{arxiv_id}\x1f{source}\x1f{relation_type}\x1f{target}"

    def _load_confidence_db(self) -> None:
        """Hydrate persisted confidence values into in-memory TripletRecord objects.

        Called once after _build_index.  Silently skips if the DB does not exist
        yet (first run) or if the table is missing.
        """
        if not self._confidence_db.exists():
            return
        try:
            conn = sqlite3.connect(self._confidence_db)
            rows = conn.execute(
                "SELECT triplet_key, confidence FROM triplet_confidence"
            ).fetchall()
            conn.close()
        except Exception:
            return
        conf_map = {row[0]: row[1] for row in rows}
        for rec in self._records:
            key = self._triplet_key(
                rec.arxiv_id, rec.source, rec.relation_type, rec.target
            )
            if key in conf_map:
                rec.confidence = conf_map[key]

    def _persist_confidence(
        self, updates: List[Tuple[str, str, str, str, float]]
    ) -> None:
        """Upsert triplet confidence values to SQLite.

        Args:
            updates: List of (arxiv_id, source, relation_type, target, confidence).
        """
        conn = sqlite3.connect(self._confidence_db)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS triplet_confidence (
                triplet_key TEXT PRIMARY KEY,
                confidence  REAL NOT NULL,
                updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        for arxiv_id, source, rel_type, target, confidence in updates:
            key = self._triplet_key(arxiv_id, source, rel_type, target)
            conn.execute(
                "INSERT OR REPLACE INTO triplet_confidence "
                "(triplet_key, confidence, updated_at) VALUES (?, ?, datetime('now'))",
                (key, confidence),
            )
        conn.commit()
        conn.close()
