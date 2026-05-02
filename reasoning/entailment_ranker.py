"""
entailment_ranker.py — Re-rank RetrievedDocs using syllogism necessity scores.

Core Thesis:
    Papers whose utility premises are *necessary* to the entailment chain score highly
    and rise to the top.  Non-chain papers retain their original retrieval score but
    are ordered below chain papers.

Blend formula:
    final = 0.5 * entailment_score + 0.5 * original_final_score

Sort order:
    1. Chain papers ranked by entailment_score DESC (most necessary first)
    2. Non-chain papers ranked by original final_score DESC

Necessary Conditions:
    - RetrievedDoc.final_score set by upstream pipeline
    - SyllogismResult.paper_scores maps arxiv_id → necessity float
    - metadata['paper_id'] in RetrievedDoc contains the arxiv_id
"""

from __future__ import annotations

import sys
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

from reasoning.syllogism_former import ChainLink, SyllogismResult

_ROOT_ER = __import__("pathlib").Path(__file__).resolve().parent.parent
if str(_ROOT_ER) not in sys.path:
    sys.path.insert(0, str(_ROOT_ER))
try:
    from constants import (  # type: ignore
        AGENT_ENTAILMENT_WEIGHT,
        AGENT_RETRIEVAL_WEIGHT,
        AGENT_CONFIDENCE_WEIGHT,
    )
except ImportError:
    AGENT_ENTAILMENT_WEIGHT = 0.5
    AGENT_RETRIEVAL_WEIGHT  = 0.3
    AGENT_CONFIDENCE_WEIGHT = 0.2

# Import RetrievedDoc from its canonical location
try:
    from retrieval.gist_retriever import RetrievedDoc
except ImportError:
    # Fallback stub for isolated testing
    from dataclasses import dataclass, field
    from typing import Any

    @dataclass
    class RetrievedDoc:
        doc_id: str
        content: str
        metadata: Dict[str, Any] = field(default_factory=dict)
        bm25_score: Optional[float] = None
        dense_score: Optional[float] = None
        bm25_rank: Optional[int] = None
        dense_rank: Optional[int] = None
        gist_rank: Optional[int] = None
        rrf_score: Optional[float] = None
        colbert_score: Optional[float] = None
        cross_encoder_score: Optional[float] = None
        final_score: Optional[float] = None

# Module-level defaults (imported from constants; kept for backward-compat import)
ENTAILMENT_WEIGHT  = AGENT_ENTAILMENT_WEIGHT
RETRIEVAL_WEIGHT   = AGENT_RETRIEVAL_WEIGHT
CONFIDENCE_WEIGHT  = AGENT_CONFIDENCE_WEIGHT


def _get_paper_id(doc: RetrievedDoc) -> str:
    """Extract arxiv_id from doc metadata or doc_id."""
    # ArxivRetriever stores paper_id in metadata
    if "paper_id" in doc.metadata:
        return str(doc.metadata["paper_id"]).strip('"')
    # Fallback: doc_id may contain the arxiv_id as prefix (e.g. "2301.00001_0_3")
    return doc.doc_id.split("_")[0].strip('"')


def _norm_final_score(docs: List[RetrievedDoc]) -> Dict[str, float]:
    """
    Min-max normalise final_score values to [0,1].
    Returns dict: doc_id → normalised score.
    """
    scores = [d.final_score or 0.0 for d in docs]
    lo, hi = min(scores), max(scores)
    span = hi - lo
    if span < 1e-9:
        return {d.doc_id: 1.0 for d in docs}
    return {d.doc_id: ((d.final_score or 0.0) - lo) / span for d in docs}


class EntailmentRanker:
    """
    Re-ranks RetrievedDocs by blending retrieval score with syllogism necessity.

    3-way blend (tuned weights from best_config_id=8):
        final = entailment_weight * judge_score
              + retrieval_weight  * norm_retrieval_score
              + confidence_weight * nli_cross_encoder_score

    Weights map to composite_weights:
        entailment_weight  = geo_alpha      (LLM judge selection quality)
        retrieval_weight   = selection_prec (initial retrieval rank precision)
        confidence_weight  = confidence_bonus (cross-encoder max entailment prob)

    Usage:
        ranker = EntailmentRanker()
        reranked = ranker.rerank(docs, syllogism_result, nli_scores)
    """

    def __init__(
        self,
        entailment_weight: float = ENTAILMENT_WEIGHT,
        retrieval_weight:  float = RETRIEVAL_WEIGHT,
        confidence_weight: float = CONFIDENCE_WEIGHT,
        verbose: bool = False,
    ):
        self._ew = entailment_weight
        self._rw = retrieval_weight
        self._cw = confidence_weight
        self._verbose = verbose

    def rerank(
        self,
        docs:       List[RetrievedDoc],
        result:     SyllogismResult,
        nli_scores: Optional[Dict[str, float]] = None,
    ) -> List[RetrievedDoc]:
        """
        Re-rank docs using a 3-way blend of judge score, retrieval score, and NLI confidence.

        blend = entailment_weight * judge_score
              + retrieval_weight  * norm_retrieval_score
              + confidence_weight * nli_cross_encoder_score

        Papers absent from paper_scores get judge_score=0.0 (rank below judge selections).
        Papers absent from nli_scores get nli_score=0.0.

        Args:
            docs:       Candidate RetrievedDoc list from Stage 1.
            result:     SyllogismResult with paper_scores (judge position weights).
            nli_scores: Optional dict of arxiv_id → cross-encoder max entailment probability.
                        When provided, adds the confidence_bonus component to the blend.

        Returns:
            New list of RetrievedDoc with updated final_score, sorted descending.
        """
        if not docs:
            return docs

        _nli = nli_scores or {}
        norm = _norm_final_score(docs)
        paper_scores = result.paper_scores

        scored: List[Tuple[float, RetrievedDoc]] = []
        for doc in docs:
            d = deepcopy(doc)
            paper_id = _get_paper_id(d)
            entailment_score = paper_scores.get(paper_id, 0.0)
            retrieval_score  = norm.get(d.doc_id, 0.0)
            nli_conf         = _nli.get(paper_id, 0.0)
            blended = (
                self._ew * entailment_score
                + self._rw * retrieval_score
                + self._cw * nli_conf
            )

            if self._verbose:
                in_chain = paper_id in paper_scores
                marker = "*" if in_chain else " "
                print(
                    f"  {marker} {paper_id}: entail={entailment_score:.3f} "
                    f"retr={retrieval_score:.3f} nli_conf={nli_conf:.3f} -> blend={blended:.3f}"
                )

            d.metadata["original_final_score"] = doc.final_score
            d.metadata["entailment_score"]      = entailment_score
            d.metadata["nli_confidence"]         = nli_conf
            d.metadata["blended_score"]          = blended
            d.final_score = blended
            scored.append((blended, d))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in scored]
