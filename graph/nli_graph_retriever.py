"""
nli_graph_retriever.py — BM25 + NLI graph expansion over the KG cache.

Replaces the Ollama-dependent GraphRetriever in Stage 7 of the syllogism
pipeline.  Instead of live LLM extraction, this module:

  1. Loads all pre-built KG cache JSON files (graph/kg_cache/*.json).
  2. Builds a BM25 index over spaCy-lemmatized triplet fields
     (subject lemma + predicate lemma + object lemma).
  3. At query time:
       a. BM25-retrieve top candidates from the index.
       b. Optionally filter to a specific set of paper IDs.
       c. Verbalize each triplet → natural language premise.
       d. NLI-score premises against the query using DeBERTa cross-encoder.
       e. Return top NLI-ranked premises, grouped by paper.

Design follows the agentic-nli-memory skill spec:
  - Index-side: lemma columns (synset/hypernym expansion deferred to v2).
  - BM25 handles IDF weighting; stop words are NOT stripped manually.
  - Epistemic tag: all KG-cache triplets default to Observed (weight 1.0).
  - Verbalization converts raw SPO text to readable premise strings before
    passing to NLI — raw SPO concatenation degrades cross-encoder performance.

Usage:
    retriever = NLIGraphRetriever()
    premises_by_paper = retriever.retrieve_context_by_paper(
        query="NLI reasoning agentic memory",
        paper_ids=["2412.17029", "2502.03283"],
        top_k_per_paper=3,
    )
    # returns {"2412.17029": ["GraphAgent uses knowledge graph for reasoning.", ...], ...}
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_KG_CACHE_DIR = Path(__file__).resolve().parent / "kg_cache"
_NLI_MODEL    = "cross-encoder/nli-deberta-v3-small"
_ENTAIL_IDX   = 1   # DeBERTa label order: contradiction=0, entailment=1, neutral=2

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
    """One entry in the BM25 index."""
    arxiv_id:      str
    source:        str
    relation_type: str
    target:        str
    premise:       str          # verbalized natural-language string
    tokens:        List[str]    # spaCy-lemmatized tokens for BM25
    epistemic:     float = 1.0  # 1.0 = Observed (all KG-cache triplets are O)


class NLIGraphRetriever:
    """
    BM25 + DeBERTa NLI graph expansion over the pre-built KG triplet cache.

    The index is built lazily on first use and cached in memory for the
    lifetime of the object.  Re-instantiate to pick up new cache files.

    Args:
        cache_dir:      Path to graph/kg_cache directory.
        nli_model:      HuggingFace cross-encoder model name.
        top_k_bm25:     BM25 candidate pool before NLI re-ranking.
        min_bm25_score: Minimum BM25 score to include in NLI pass (filters noise).
    """

    def __init__(
        self,
        cache_dir:      Optional[Path] = None,
        nli_model:      str            = _NLI_MODEL,
        top_k_bm25:     int            = 50,
        min_bm25_score: float          = 0.0,
    ):
        self._cache_dir      = cache_dir or _KG_CACHE_DIR
        self._nli_model      = nli_model
        self._top_k_bm25     = top_k_bm25
        self._min_bm25_score = min_bm25_score

        self._records:  List[TripletRecord] = []
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
        """BM25 top-k, then filter to paper_ids if given."""
        if self._bm25 is None:
            return []

        query_tokens = self._lemmatize(query)
        scores       = self._bm25.get_scores(query_tokens)

        # Pair (record, score) and sort descending
        ranked = sorted(
            zip(self._records, scores.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )

        # Filter by paper_ids
        id_set = set(paper_ids) if paper_ids else None
        candidates: List[TripletRecord] = []
        for rec, score in ranked:
            if score < self._min_bm25_score:
                break
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
        Epistemic weight (Observed=1.0) is applied to the raw NLI probability.
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
        entail_probs = probs[:, _ENTAIL_IDX]

        scored = [
            (rec, float(ep) * rec.epistemic)
            for rec, ep in zip(candidates, entail_probs)
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
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
