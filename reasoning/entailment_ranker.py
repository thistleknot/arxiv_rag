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

ENTAILMENT_WEIGHT = 0.5
RETRIEVAL_WEIGHT  = 0.5


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

    Usage:
        ranker = EntailmentRanker()
        reranked = ranker.rerank(docs, syllogism_result)
    """

    def __init__(
        self,
        entailment_weight: float = ENTAILMENT_WEIGHT,
        retrieval_weight:  float = RETRIEVAL_WEIGHT,
        verbose: bool = False,
    ):
        self._ew = entailment_weight
        self._rw = retrieval_weight
        self._verbose = verbose

    def rerank(
        self,
        docs:   List[RetrievedDoc],
        result: SyllogismResult,
    ) -> List[RetrievedDoc]:
        """
        Re-rank docs using syllogism necessity scores.

        Chain papers: entailment_score = paper_scores[arxiv_id]
        Non-chain papers: entailment_score = 0.0
        Blend: final = e_weight * entailment + r_weight * norm_retrieval_score

        Args:
            docs: List from ArxivRetriever (already ranked by retrieval pipeline)
            result: SyllogismResult with .paper_scores (arxiv_id → float)

        Returns:
            New list of RetrievedDoc with updated final_score, sorted descending.
        """
        if not docs:
            return docs

        norm = _norm_final_score(docs)
        paper_scores = result.paper_scores

        scored: List[Tuple[float, RetrievedDoc]] = []
        for doc in docs:
            d = deepcopy(doc)
            paper_id = _get_paper_id(d)
            entailment_score = paper_scores.get(paper_id, 0.0)
            retrieval_score  = norm.get(d.doc_id, 0.0)
            blended = self._ew * entailment_score + self._rw * retrieval_score

            if self._verbose:
                in_chain = paper_id in paper_scores
                marker = "*" if in_chain else " "
                print(
                    f"  {marker} {paper_id}: entail={entailment_score:.3f} "
                    f"retr={retrieval_score:.3f} -> blend={blended:.3f}"
                )

            # Overwrite final_score with blended value; preserve original in metadata
            d.metadata["original_final_score"] = doc.final_score
            d.metadata["entailment_score"]      = entailment_score
            d.metadata["blended_score"]          = blended
            d.final_score = blended
            scored.append((blended, d))

        # Sort: chain papers first (entailment_score > 0), then non-chain — both by blended DESC
        chain_ids = set(paper_scores.keys())
        chain_docs    = [(s, d) for s, d in scored if _get_paper_id(d) in chain_ids]
        nonchain_docs = [(s, d) for s, d in scored if _get_paper_id(d) not in chain_ids]

        chain_docs.sort(key=lambda x: x[0], reverse=True)
        nonchain_docs.sort(key=lambda x: x[0], reverse=True)

        return [d for _, d in chain_docs + nonchain_docs]
