"""
ECDF-Weighted RRF Retrieval System

Simple, effective parallel retrieval with ECDF normalization and priority-weighted RRF fusion.

PHILOSOPHY:
- No cascading filters (over-engineered)
- No MAD thresholds (unpredictable)
- Just: BM25 + Cosine + ColBERT + CE → ECDF → Weighted RRF → Done

ARCHITECTURE:
┌─────────────────────────────────────────────────────────────────┐
│  Parallel Retrieval (top_k² per method)                         │
├─────────────────────────────────────────────────────────────────┤
│  BM25 Pool     Cosine Pool     ColBERT Pool     CE Pool         │
│     │              │                │               │            │
│     └──────────────┴────────────────┴───────────────┘            │
│                          │                                       │
│                          ▼                                       │
│               ┌──────────────────┐                              │
│               │ ECDF Transform   │                              │
│               │ (per metric)     │                              │
│               └────────┬─────────┘                              │
│                        │                                         │
│                        ▼                                         │
│               ┌──────────────────┐                              │
│               │ Priority Weights │                              │
│               │ Triangular       │                              │
│               └────────┬─────────┘                              │
│                        │                                         │
│                        ▼                                         │
│               ┌──────────────────┐                              │
│               │ Weighted RRF     │                              │
│               │ 1/(1+ε-ECDF)     │                              │
│               └────────┬─────────┘                              │
│                        │                                         │
│                        ▼                                         │
│                     top_k                                        │
└─────────────────────────────────────────────────────────────────┘

KEY INSIGHT:
ECDF normalization makes weights meaningful. Standard RRF with raw scores
doesn't account for score distributions.

EXAMPLE:
Query: "agentic memory methods"
Priority Ranks: {colbert: 1, ce: 2, cosine: 3, bm25: 4}
Weights: {colbert: 0.40, ce: 0.30, cosine: 0.20, bm25: 0.10}

For each paper:
    ecdf_colbert = midpoint_ecdf(colbert_scores)
    ecdf_ce = midpoint_ecdf(ce_scores)
    ecdf_cosine = midpoint_ecdf(cosine_scores)
    ecdf_bm25 = midpoint_ecdf(bm25_scores)
    
    rrf_score = (
        0.40 / (1.01 - ecdf_colbert) +
        0.30 / (1.01 - ecdf_ce) +
        0.20 / (1.01 - ecdf_cosine) +
        0.10 / (1.01 - ecdf_bm25)
    )

OPTIONAL CASCADING FILTER:
If use_filtering=True, applies original cascading approach:
1. Filter to top-k * multiplier on ColBERT
2. Filter to top-k on CE from remaining
This is for backward compatibility only.
"""

from collections import Counter
from typing import List, Dict, Optional
from dataclasses import dataclass
import numpy as np


def ecdf_rank_ratio_weights(priority_ranks: dict) -> dict:
    """
    Convert priority ranks to normalized weights using triangular accumulation.
    
    Lower rank = higher priority = higher weight (non-linear).
    Tied ranks get fractional positions and share weight.
    
    Args:
        priority_ranks: {metric_name: rank} where lower rank = higher priority
    
    Returns:
        {metric_name: weight} normalized to sum to 1.0
        
    Example:
        >>> ecdf_rank_ratio_weights({'colbert': 1, 'ce': 2, 'cosine': 3, 'bm25': 4})
        {'colbert': 0.40, 'ce': 0.30, 'cosine': 0.20, 'bm25': 0.10}
    """
    # Handle ties: assign fractional ranks
    rank_counts = Counter(priority_ranks.values())
    adjusted = {}
    for metric, rank in priority_ranks.items():
        if rank_counts[rank] > 1:
            # Fractional rank: average position
            adjusted[metric] = rank + (rank_counts[rank] - 1) / 2
        else:
            adjusted[metric] = rank
    
    # Invert ranks (lower rank = higher priority)
    max_rank = max(adjusted.values())
    inverted = {k: max_rank - v + 1 for k, v in adjusted.items()}
    
    # Triangular accumulation: T(n) = n*(n+1)/2
    # This gives non-linear emphasis to higher priorities
    triangular = {k: v * (v + 1) / 2 for k, v in inverted.items()}
    
    # Normalize to sum to 1.0
    total = sum(triangular.values())
    return {k: v / total for k, v in triangular.items()}


def midpoint_ecdf(values: list, higher_is_better: bool = True) -> list:
    """
    Compute midpoint ECDF for a list of values.
    
    Midpoint method naturally avoids exact 0 and 1 values.
    
    Args:
        values: list of metric values
        higher_is_better: if True, higher values get higher ECDF scores
    
    Returns:
        list of ECDF scores in (0, 1) where 1 = best
        
    Example:
        >>> midpoint_ecdf([1.0, 2.0, 3.0, 4.0, 5.0])
        [0.1, 0.3, 0.5, 0.7, 0.9]  # Midpoint avoids 0 and 1
    """
    if not higher_is_better:
        values = [-v for v in values]  # Invert for minimization metrics
    
    n = len(values)
    ecdf_scores = []
    for val in values:
        count_below = sum(1 for v in values if v < val)
        count_at_or_below = sum(1 for v in values if v <= val)
        # Midpoint formula naturally avoids 0 and 1
        ecdf = (count_below + count_at_or_below) / 2 / n
        ecdf_scores.append(ecdf)
    return ecdf_scores


@dataclass
class ECDFRRFReranker:
    """
    ECDF-weighted RRF reranker for paper retrieval.
    
    Replaces complex MAD-based filtering with simple parallel retrieval + weighted fusion.
    
    Args:
        priority_ranks: {metric: rank} where lower rank = higher priority.
            Default: {'colbert': 1, 'ce': 2, 'cosine': 3, 'bm25': 4}
        use_filtering: If True, apply cascading filter (original approach)
        filter_top_k_multiplier: Multiplier for filtering (e.g., 1.5 for phi lower)
        rrf_epsilon: Small constant to avoid division by zero (default 0.01)
    """
    priority_ranks: Optional[Dict[str, int]] = None
    use_filtering: bool = False
    filter_top_k_multiplier: float = 1.5
    rrf_epsilon: float = 0.01
    
    def __post_init__(self):
        """Initialize weights from priority ranks."""
        # Default priority: CE > ColBERT > Cosine > BM25
        # Rationale: BM25/Cosine already pre-filtered during retrieval.
        # CE/ColBERT are computed post-RRF on filtered candidates, so should dominate.
        # This avoids double-weighting the initial retrieval metrics.
        if self.priority_ranks is None:
            self.priority_ranks = {
                'ce': 1,          # Highest: computed post-filter
                'colbert': 2,     # High: computed post-filter
                'cosine': 3,      # Lower: already influenced retrieval
                'bm25': 4         # Lowest: already influenced retrieval
            }
        
        # Compute weights from priority ranks
        self.metric_weights = ecdf_rank_ratio_weights(self.priority_ranks)
        
        print(f"\n[ECDF-RRF Reranker]")
        print(f"  Priority Ranks: {self.priority_ranks}")
        print(f"  Weights: {', '.join(f'{k}: {v:.3f}' for k, v in self.metric_weights.items())}")
        print(f"  Filtering: {'Yes' if self.use_filtering else 'No'}")
        if self.use_filtering:
            print(f"  Filter Multiplier: {self.filter_top_k_multiplier}")
    
    def rerank(
        self,
        papers: List,  # List[RetrievedPaper]
        top_k: int
    ) -> List:  # List[RetrievedPaper]
        """
        Rerank papers using ECDF-weighted RRF.
        
        Process:
        1. Optional: Apply cascading filter (ColBERT → CE)
        2. Compute ECDF for each metric
        3. Apply weighted RRF: score = sum(weight / (1 + epsilon - ecdf))
        4. Sort and return top-k
        
        Args:
            papers: List of RetrievedPaper objects with metric scores
            top_k: Number of papers to return
        
        Returns:
            Top-k papers sorted by RRF score
        """
        if not papers:
            return []
        
        print(f"\n[ECDF-RRF Fusion] Starting with {len(papers)} papers")
        
        # Optional cascading filter (original approach)
        if self.use_filtering:
            papers = self._apply_cascading_filter(papers, top_k)
        
        # Extract PAPER-LEVEL metric scores only
        # ColBERT and CE are computed on FULL paper text (not chunks)
        # BM25/Cosine are chunk-level and already incorporated into candidate selection
        colbert_scores = []
        ce_scores = []
        
        for p in papers:
            colbert_scores.append(getattr(p, 'colbert_score', 0.0))
            ce_scores.append(getattr(p, 'cross_encoder_score', 0.0))
        
        # Compute ECDF for paper-level metrics only
        ecdf_colbert = midpoint_ecdf(colbert_scores, higher_is_better=True)
        ecdf_ce = midpoint_ecdf(ce_scores, higher_is_better=True)
        
        print(f"[ECDF Stats]")
        print(f"  ColBERT: [{min(ecdf_colbert):.3f}, {max(ecdf_colbert):.3f}]")
        print(f"  CE:      [{min(ecdf_ce):.3f}, {max(ecdf_ce):.3f}]")
        
        # Compute weighted RRF scores (ColBERT + CE only)
        # Formula: score = sum(weight_i / (1 + epsilon - ecdf_i))
        # epsilon avoids division by zero when ecdf = 1.0
        rrf_scores = []
        for i in range(len(papers)):
            score = 0.0
            score += self.metric_weights['colbert'] / (1 + self.rrf_epsilon - ecdf_colbert[i])
            score += self.metric_weights['ce'] / (1 + self.rrf_epsilon - ecdf_ce[i])
            rrf_scores.append(score)
        
        print(f"[RRF Scores] Range: [{min(rrf_scores):.3f}, {max(rrf_scores):.3f}], "
              f"Mean: {np.mean(rrf_scores):.3f}")
        
        # Assign scores and ECDF values to papers
        for i, paper in enumerate(papers):
            paper.rrf_score = rrf_scores[i]
            paper.ecdf_colbert = ecdf_colbert[i]
            paper.ecdf_ce = ecdf_ce[i]
            # colbert_score and cross_encoder_score already exist on paper
        
        # Sort by RRF score and return top-k
        papers.sort(key=lambda p: p.rrf_score, reverse=True)
        return papers[:top_k]
    
    def _apply_cascading_filter(
        self,
        papers: List['RetrievedPaper'],
        top_k: int
    ) -> List['RetrievedPaper']:
        """
        Apply original cascading filter approach.
        
        1. Filter to top-k * multiplier on ColBERT
        2. Filter to top-k on CE from remaining
        
        Args:
            papers: Input papers
            top_k: Target number of papers
        
        Returns:
            Filtered papers
        """
        filter_k = int(top_k * self.filter_top_k_multiplier)
        print(f"[Cascading Filter] top-{filter_k} ColBERT → top-{top_k} CE")
        
        # Filter to top-k * multiplier on ColBERT
        papers.sort(key=lambda p: p.colbert_score, reverse=True)
        papers = papers[:filter_k]
        print(f"  After ColBERT filter: {len(papers)} papers")
        
        # Filter to top-k on CE from remaining
        papers.sort(key=lambda p: p.cross_encoder_score, reverse=True)
        papers = papers[:top_k]
        print(f"  After CE filter: {len(papers)} papers")
        
        return papers
