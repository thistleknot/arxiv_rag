"""
GIST Retriever: Greedy Information Selection with Topic Diversity

=============================================================================
ARCHITECTURE OVERVIEW
=============================================================================

This module implements a retrieval pipeline based on the GIST principle:
select documents that maximize query relevance while minimizing redundancy.

The key insight is the FEATURE SELECTION ANALOGY:
  - Documents = features (predictors)
  - Query similarity = response variable y
  - Doc-doc similarity = feature correlation matrix X'X

Goal: Select k documents that:
  1. Correlate with query (UTILITY: query relevance)
  2. Are not collinear with each other (COVERAGE: diversity)

=============================================================================
PIPELINE FLOW
=============================================================================

┌─────────────────────────────────────────────────────────────────────────┐
│  RETRIEVAL (Parallel, Independent Pools)                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  BM25 Pool (top_k²)                    Embedding Pool (top_k²)          │
│       │                                      │                          │
│       ▼                                      ▼                          │
│  ┌─────────────────┐                  ┌─────────────────┐              │
│  │ Coverage Matrix │                  │ Coverage Matrix │              │
│  │ BM25(doc_i, doc_j)                │ cosine(doc_i, doc_j)           │
│  │ for all i,j in pool               │ for all i,j in pool            │
│  │ (excludes query)                  │ (excludes query)               │
│  └────────┬────────┘                  └────────┬────────┘              │
│           │                                    │                        │
│  ┌────────▼────────┐                  ┌────────▼────────┐              │
│  │ Utility Vector  │                  │ Utility Vector  │              │
│  │ BM25(doc, query)│                  │ cosine(doc, query)            │
│  │ = response term │                  │ = response term │              │
│  └────────┬────────┘                  └────────┬────────┘              │
│           │                                    │                        │
│  ┌────────▼────────┐                  ┌────────▼────────┐              │
│  │ GIST Selection  │                  │ GIST Selection  │              │
│  │ Iterative greedy│                  │ Iterative greedy│              │
│  │ max utility     │                  │ max utility     │              │
│  │ min collinearity│                  │ min collinearity│              │
│  └────────┬────────┘                  └────────┬────────┘              │
│           │                                    │                        │
│           └──────────────┬─────────────────────┘                        │
│                          ▼                                              │
│                 ┌────────────────┐                                      │
│                 │  RRF Fusion    │                                      │
│                 │  Merge ranks   │                                      │
│                 │  (≤ φ·top_k²)  │                                      │
│                 └───────┬────────┘                                      │
│                         ▼                                               │
│                 ┌────────────────┐                                      │
│                 │ Late Interact. │ ColBERT MaxSim                       │
│                 │ Token-level    │                                      │
│                 └───────┬────────┘                                      │
│                         ▼                                               │
│                 ┌────────────────┐                                      │
│                 │ Cross-Encoder  │ MS-MARCO                             │
│                 │ Joint encoding │                                      │
│                 └───────┬────────┘                                      │
│                         ▼                                               │
│                      top_k                                              │
└─────────────────────────────────────────────────────────────────────────┘

=============================================================================
GIST SELECTION ALGORITHM (Feature Selection Analogy)
=============================================================================

The GIST selection step is analogous to forward stepwise feature selection
with VIF (Variance Inflation Factor) control:

  Input:
    - Coverage matrix C: n×n doc-doc similarities (X'X analog)
    - Utility vector u: n×1 doc-query similarities (X'y analog)
    - k: number to select

  Algorithm:
    S = {}  # selected set
    for i in 1..k:
        for each candidate doc d not in S:
            utility = u[d]                           # correlation with query
            collinearity = max(C[d, s] for s in S)   # max correlation with selected
            score[d] = utility - collinearity        # partial correlation proxy
        
        best = argmax(score)
        S = S ∪ {best}
    
    return S

This is equivalent to MMR (Maximal Marginal Relevance) but framed through
the lens of multicollinearity control in regression.

=============================================================================
FIBONACCI CASCADE
=============================================================================

The pipeline uses Fibonacci numbers for stage sizing:
  - Retrieval: top_k² (broad recall)
  - GIST selection: φ·top_k² where φ ≈ 0.618 (one Fib lower)
  - ColBERT: top_k
  - Cross-encoder: Fib lower than top_k

Example for top_k=21:
  - Retrieve: 441 (21²)
  - GIST: 377 (Fib below 441)
  - ColBERT: 21
  - Cross-encoder: 13 (Fib below 21)

=============================================================================
USAGE
=============================================================================

Subclass GISTRetriever and implement the abstract methods for your backend:

    class MyRetriever(GISTRetriever):
        def _retrieve_bm25(self, query, limit): ...
        def _retrieve_dense(self, query, limit): ...
        def _get_bm25_scores(self, doc_ids, query): ...
        def _get_dense_embeddings(self, doc_ids): ...
        def _get_query_embedding(self, query): ...

Then use:

    retriever = MyRetriever(config)
    results = retriever.search("my query", top_k=21)

=============================================================================
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set, Any
from collections import defaultdict
import numpy as np


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class GISTConfig:
    """
    Configuration for GIST retrieval pipeline.
    
    Attributes:
        rrf_k: RRF fusion parameter (default 60)
        gist_lambda: Balance between utility and coverage in GIST selection.
                     Higher = more weight on utility (relevance).
                     Lower = more weight on coverage (diversity).
                     Default 0.7 favors relevance slightly.
        use_colbert: Enable ColBERT late interaction reranking
        use_cross_encoder: Enable cross-encoder final reranking
        colbert_model: ColBERT model name
        cross_encoder_model: Cross-encoder model name
        colbert_batch_size: Batch size for ColBERT scoring
        cross_encoder_batch_size: Batch size for cross-encoder scoring
        fibonacci_sequence: Fibonacci numbers for cascade sizing
    """
    rrf_k: int = 60
    gist_lambda: float = 0.7
    use_colbert: bool = True
    use_cross_encoder: bool = True
    colbert_model: str = "bert-base-uncased"
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    colbert_batch_size: int = 8
    cross_encoder_batch_size: int = 8
    fibonacci_sequence: List[int] = field(default_factory=lambda: [
        1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597
    ])


@dataclass
class RetrievedDoc:
    """
    Represents a retrieved document with all scoring metadata.
    
    Attributes:
        doc_id: Unique document identifier
        content: Document text content
        metadata: Additional metadata (paper_id, section_idx, etc.)
        bm25_score: BM25 relevance score (if from BM25 pool)
        dense_score: Dense embedding similarity (if from dense pool)
        bm25_rank: Rank in BM25 pool (1-indexed)
        dense_rank: Rank in dense pool (1-indexed)
        gist_rank: Rank after GIST selection (1-indexed)
        rrf_score: Score after RRF fusion
        colbert_score: ColBERT late interaction score
        cross_encoder_score: Cross-encoder score
        final_score: Final ranking score
    """
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


# =============================================================================
# GIST Selection Algorithm
# =============================================================================

def gist_select(
    coverage_matrix: np.ndarray,
    utility_vector: np.ndarray,
    k: int,
    lambda_param: float = 0.7
) -> List[int]:
    """
    GIST: Greedy selection balancing utility (relevance) and coverage (diversity).
    
    This implements forward stepwise selection with collinearity control,
    analogous to feature selection with VIF constraints.
    
    Formula: score(d) = λ * utility(d) - (1-λ) * max_sim(d, S)
    
    Where:
        - utility(d) = doc-query similarity (normalized to [0,1])
        - max_sim(d, S) = maximum similarity to any already-selected doc
        - λ = tradeoff parameter (higher = favor relevance)
    
    This is equivalent to MMR (Maximal Marginal Relevance).
    
    Args:
        coverage_matrix: n×n pairwise doc-doc similarity matrix.
                         Should be symmetric with diagonal = 1.
                         Does NOT include query - purely doc-to-doc.
        utility_vector: n-dim vector of doc-query similarities.
                        Higher = more relevant to query.
        k: Number of documents to select.
        lambda_param: Tradeoff between utility and diversity.
                      Range [0, 1]. Default 0.7.
                      - 1.0 = pure utility (no diversity)
                      - 0.0 = pure diversity (ignore relevance)
                      - 0.7 = slight preference for relevance
    
    Returns:
        List of selected indices in selection order.
        First element is most relevant, subsequent elements balance
        relevance with diversity from already-selected.
    
    Complexity: O(k * n) where n = number of candidates
    
    Example:
        >>> coverage = cosine_similarity(doc_embeddings)  # n×n
        >>> utility = cosine_similarity(doc_embeddings, query_embedding).flatten()  # n
        >>> selected = gist_select(coverage, utility, k=10, lambda_param=0.7)
        >>> diverse_docs = [docs[i] for i in selected]
    """
    n = len(utility_vector)
    
    if k >= n:
        # Select all, sorted by utility
        return list(np.argsort(utility_vector)[::-1])
    
    if k <= 0:
        return []
    
    # Normalize utility to [0, 1] for consistent weighting
    u_min, u_max = utility_vector.min(), utility_vector.max()
    if u_max > u_min:
        utility_norm = (utility_vector - u_min) / (u_max - u_min)
    else:
        utility_norm = np.ones(n)  # All equal utility
    
    selected_indices = []
    remaining = set(range(n))
    
    # Track max similarity to selected set for each candidate
    # Initially zero (no docs selected yet)
    max_sim_to_selected = np.zeros(n)
    
    for iteration in range(k):
        best_idx = None
        best_score = float('-inf')
        
        for idx in remaining:
            # Utility: query relevance (normalized)
            utility = utility_norm[idx]
            
            # Collinearity penalty: max similarity to already-selected
            collinearity = max_sim_to_selected[idx]
            
            # MMR-style score: weighted combination
            # score = λ * relevance - (1-λ) * redundancy
            score = lambda_param * utility - (1 - lambda_param) * collinearity
            
            if score > best_score:
                best_score = score
                best_idx = idx
        
        if best_idx is None:
            break  # No valid candidates remaining
        
        # Add best to selected set
        selected_indices.append(best_idx)
        remaining.remove(best_idx)
        
        # Update max similarity for remaining candidates
        # Each remaining doc's penalty increases if it's similar to newly selected
        for idx in remaining:
            sim_to_new = coverage_matrix[idx, best_idx]
            if sim_to_new > max_sim_to_selected[idx]:
                max_sim_to_selected[idx] = sim_to_new
    
    return selected_indices


def compute_rrf_score(
    ranks: List[Optional[int]],
    k: int = 60
) -> float:
    """
    Compute Reciprocal Rank Fusion score from multiple rank lists.
    
    RRF(d) = Σ 1 / (k + rank_i(d))
    
    Args:
        ranks: List of ranks from different retrieval methods.
               None if doc not present in that method's results.
        k: RRF parameter (default 60, per original paper)
    
    Returns:
        RRF score (higher = better)
    """
    score = 0.0
    for rank in ranks:
        if rank is not None:
            score += 1.0 / (k + rank)
    return score


def get_fibonacci_lower(n: int, fib_sequence: List[int]) -> int:
    """
    Get the largest Fibonacci number strictly less than n.
    
    Args:
        n: Upper bound
        fib_sequence: List of Fibonacci numbers
    
    Returns:
        Largest Fibonacci number < n, or n if none found
    """
    candidates = [f for f in fib_sequence if f < n]
    return candidates[-1] if candidates else n


def get_fibonacci_upper(n: int, fib_sequence: List[int]) -> int:
    """
    Get the smallest Fibonacci number greater than or equal to n.
    
    Args:
        n: Lower bound
        fib_sequence: List of Fibonacci numbers
    
    Returns:
        Smallest Fibonacci number >= n, or n if none found
    """
    candidates = [f for f in fib_sequence if f >= n]
    return candidates[0] if candidates else n


# =============================================================================
# Reranker Components
# =============================================================================

class ColBERTScorer:
    """
    ColBERT Late Interaction Scorer.
    
    Computes token-level MaxSim: Score(Q,D) = Σᵢ maxⱼ sim(qᵢ, dⱼ)
    
    Each query token finds its best-matching document token,
    then scores are summed. This captures fine-grained semantic matching.
    """
    
    def __init__(self, model_name: str = "bert-base-uncased", batch_size: int = 8):
        self.model_name = model_name
        self.batch_size = batch_size
        self.available = False
        self.model = None
        self.tokenizer = None
        self.linear = None
        self.device = None
        self.dim = 128
        
        self._initialize()
    
    def _initialize(self) -> None:
        """Load BERT model and projection layer."""
        try:
            import torch
            from transformers import AutoTokenizer, AutoModel
            
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            
            print(f"Loading ColBERT ({self.model_name}) on {self.device}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            
            hidden_size = self.model.config.hidden_size
            self.linear = torch.nn.Linear(hidden_size, self.dim, bias=False)
            torch.nn.init.xavier_uniform_(self.linear.weight)
            
            self.model = self.model.to(self.device)
            self.model.eval()
            self.linear = self.linear.to(self.device)
            
            self.available = True
            print(f"ColBERT initialized.")
            
        except ImportError as e:
            print(f"ColBERT unavailable (missing dependencies): {e}")
            self.available = False
        except Exception as e:
            print(f"ColBERT initialization failed: {e}")
            self.available = False
    
    def _encode(self, texts: List[str], max_length: int, is_query: bool = False):
        """Encode texts to token embeddings."""
        import torch
        
        # Add marker tokens
        if is_query:
            texts = [f"[Q] {t}" for t in texts]
        else:
            texts = [f"[D] {t}" for t in texts]
        
        encoded = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = outputs.last_hidden_state
        
        # Project to lower dimension
        embeddings = self.linear(embeddings)
        
        # L2 normalize
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
        
        # Mask padding
        mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size())
        embeddings = embeddings * mask_expanded.float()
        
        return embeddings, attention_mask
    
    def score(self, query: str, documents: List[str]) -> np.ndarray:
        """
        Score documents against query using ColBERT MaxSim.
        
        Args:
            query: Query text
            documents: List of document texts
        
        Returns:
            Array of scores (higher = more relevant)
        """
        if not self.available:
            return np.zeros(len(documents))
        
        if not documents:
            return np.array([])
        
        import torch
        
        try:
            # Encode query (once)
            query_emb, query_mask = self._encode([query], max_length=32, is_query=True)
            
            all_scores = []
            
            # Batch encode documents
            for i in range(0, len(documents), self.batch_size):
                batch = documents[i:i + self.batch_size]
                doc_embs, doc_masks = self._encode(batch, max_length=512, is_query=False)
                
                # MaxSim: for each query token, find max similarity to any doc token
                n_docs = doc_embs.size(0)
                query_expanded = query_emb.expand(n_docs, -1, -1)
                
                # Similarity matrix: (n_docs, query_len, doc_len)
                sim_matrix = torch.bmm(query_expanded, doc_embs.transpose(1, 2))
                
                # Mask padding in documents
                doc_mask_expanded = doc_masks.unsqueeze(1).expand(-1, query_emb.size(1), -1)
                sim_matrix = sim_matrix.masked_fill(~doc_mask_expanded.bool(), float('-inf'))
                
                # Max over document tokens
                max_sim_per_query_token, _ = sim_matrix.max(dim=-1)
                
                # Mask padding in query
                query_mask_expanded = query_mask.expand(n_docs, -1)
                max_sim_per_query_token = max_sim_per_query_token.masked_fill(
                    ~query_mask_expanded.bool(), 0.0
                )
                
                # Sum over query tokens
                batch_scores = max_sim_per_query_token.sum(dim=-1)
                all_scores.append(batch_scores.detach().cpu().numpy())
            
            return np.concatenate(all_scores)
            
        except Exception as e:
            print(f"ColBERT scoring failed: {e}")
            return np.zeros(len(documents))


class CrossEncoderScorer:
    """
    Cross-Encoder Reranker using MS MARCO model.
    
    Encodes query+document pairs jointly for fine-grained relevance scoring.
    More expensive but more accurate than bi-encoders.
    """
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", batch_size: int = 8):
        self.model_name = model_name
        self.batch_size = batch_size
        self.available = False
        self.model = None
        self.device = None
        
        self._initialize()
    
    def _initialize(self) -> None:
        """Load cross-encoder model."""
        try:
            import torch
            from sentence_transformers import CrossEncoder
            
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            
            print(f"Loading Cross-Encoder ({self.model_name}) on {self.device}...")
            self.model = CrossEncoder(self.model_name, max_length=512, device=self.device)
            
            self.available = True
            print(f"Cross-Encoder initialized.")
            
        except ImportError as e:
            print(f"Cross-Encoder unavailable (missing dependencies): {e}")
            self.available = False
        except Exception as e:
            print(f"Cross-Encoder initialization failed: {e}")
            self.available = False
    
    def score(self, query: str, documents: List[str]) -> np.ndarray:
        """
        Score query-document pairs with cross-encoder.
        
        Args:
            query: Query text
            documents: List of document texts
        
        Returns:
            Array of scores (higher = more relevant)
        """
        if not self.available:
            return np.zeros(len(documents))
        
        if not documents:
            return np.array([])
        
        try:
            pairs = [[query, doc] for doc in documents]
            scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
            return scores
            
        except Exception as e:
            print(f"Cross-Encoder scoring failed: {e}")
            return np.zeros(len(documents))


# =============================================================================
# Grouped Result Container
# =============================================================================

@dataclass
class RetrievedGroup:
    """
    Represents a grouped retrieval result (e.g., full section or full quote).
    
    Contains all chunks belonging to this group, reconstructed full text,
    and scoring metadata.
    
    Attributes:
        group_id: Unique group identifier (e.g., "paper_id:section_idx" or "quote_id")
        group_key: Tuple of grouping field values
        full_text: Reconstructed full text from all chunks
        chunks: List of RetrievedDoc chunks belonging to this group
        rrf_score: Aggregated RRF score from chunks
        colbert_score: ColBERT score on full_text
        cross_encoder_score: Cross-encoder score on full_text
        final_score: Final ranking score
        metadata: Additional metadata
    """
    group_id: str
    group_key: Tuple
    full_text: str
    chunks: List[RetrievedDoc] = field(default_factory=list)
    rrf_score: Optional[float] = None
    colbert_score: Optional[float] = None
    cross_encoder_score: Optional[float] = None
    final_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievedPaper:
    """
    Represents a paper with all its sections.
    
    Attributes:
        paper_id: Paper identifier
        sections: List of RetrievedGroup sections belonging to this paper
        full_text: Concatenated text from all sections
        rrf_score: Aggregated RRF score from all sections
        colbert_score: Max ColBERT score across sections
        cross_encoder_score: Cross-encoder score on full paper text
        final_score: Final ranking score
        metadata: Additional metadata
    """
    paper_id: str
    sections: List[RetrievedGroup] = field(default_factory=list)
    full_text: str = ""
    rrf_score: Optional[float] = None
    colbert_score: Optional[float] = None
    cross_encoder_score: Optional[float] = None
    final_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Abstract GIST Retriever
# =============================================================================

class GISTRetriever(ABC):
    """
    Abstract base class for GIST retrieval with grouping support.
    
    Implements the full GIST pipeline:
        1. Parallel retrieval (BM25 + dense) at CHUNK level
        2. GIST selection on each pool (iterative greedy with collinearity control)
        3. RRF fusion of chunks
        4. GROUP chunks by parent (section, quote, etc.)
        5. RECONSTRUCT full text for each group
        6. ColBERT late interaction reranking on FULL TEXT
        7. Cross-encoder final reranking on FULL TEXT
    
    Subclasses must implement:
        - Backend methods: _retrieve_bm25, _retrieve_dense, etc.
        - Grouping methods: _get_group_key, _fetch_all_chunks_for_group
    """
    
    def __init__(self, config: Optional[GISTConfig] = None):
        self.config = config or GISTConfig()
        
        # Lazy-load rerankers
        self._colbert_scorer = None
        self._cross_encoder_scorer = None
    
    @property
    def colbert_scorer(self) -> ColBERTScorer:
        """Lazy-load ColBERT scorer."""
        if self._colbert_scorer is None:
            self._colbert_scorer = ColBERTScorer(
                model_name=self.config.colbert_model,
                batch_size=self.config.colbert_batch_size
            )
        return self._colbert_scorer
    
    @property
    def cross_encoder_scorer(self) -> CrossEncoderScorer:
        """Lazy-load cross-encoder scorer."""
        if self._cross_encoder_scorer is None:
            self._cross_encoder_scorer = CrossEncoderScorer(
                model_name=self.config.cross_encoder_model,
                batch_size=self.config.cross_encoder_batch_size
            )
        return self._cross_encoder_scorer
    
    # =========================================================================
    # Abstract Methods (Backend-Specific)
    # =========================================================================
    
    @abstractmethod
    def _retrieve_bm25(self, query: str, limit: int) -> List[RetrievedDoc]:
        """
        Retrieve top documents from BM25 index.
        
        Args:
            query: Query text
            limit: Maximum number to retrieve
        
        Returns:
            List of RetrievedDoc with bm25_score and bm25_rank populated
        """
        pass
    
    @abstractmethod
    def _retrieve_dense(self, query: str, limit: int) -> List[RetrievedDoc]:
        """
        Retrieve top documents from dense (embedding) index.
        
        Args:
            query: Query text
            limit: Maximum number to retrieve
        
        Returns:
            List of RetrievedDoc with dense_score and dense_rank populated
        """
        pass
    
    @abstractmethod
    def _get_bm25_doc_doc_scores(self, doc_ids: List[str]) -> np.ndarray:
        """
        Compute pairwise BM25 similarity matrix for given documents.
        
        This is the coverage matrix for the BM25 pool (X'X analog).
        Should NOT include query - purely doc-to-doc similarity.
        
        Args:
            doc_ids: List of document IDs
        
        Returns:
            n×n symmetric similarity matrix, diagonal = 1
        """
        pass
    
    @abstractmethod
    def _get_dense_embeddings(self, doc_ids: List[str]) -> np.ndarray:
        """
        Get dense embeddings for given documents.
        
        Args:
            doc_ids: List of document IDs
        
        Returns:
            n×d embedding matrix
        """
        pass
    
    @abstractmethod
    def _get_query_embedding(self, query: str) -> np.ndarray:
        """
        Get dense embedding for query.
        
        Args:
            query: Query text
        
        Returns:
            1×d embedding vector
        """
        pass
    
    @abstractmethod
    def _get_bm25_query_scores(self, doc_ids: List[str], query: str) -> np.ndarray:
        """
        Get BM25 scores for documents against query.
        
        This is the utility vector for the BM25 pool (X'y analog).
        
        Args:
            doc_ids: List of document IDs
            query: Query text
        
        Returns:
            n-dim array of BM25 scores
        """
        pass
    
    @abstractmethod
    def _get_group_key(self, doc: RetrievedDoc) -> Tuple:
        """
        Extract grouping key from a retrieved chunk.
        
        For arxiv: (paper_id, section_idx)
        For quotes: (quote_id,)
        
        Args:
            doc: Retrieved chunk
        
        Returns:
            Tuple of grouping field values
        """
        pass
    
    @abstractmethod
    def _fetch_all_chunks_for_group(self, group_key: Tuple) -> List[RetrievedDoc]:
        """
        Fetch ALL chunks belonging to a group (for reconstruction).
        
        Args:
            group_key: Tuple from _get_group_key
        
        Returns:
            List of all chunks for this group, ordered by chunk_idx
        """
        pass
    
    @abstractmethod
    def _group_key_to_id(self, group_key: Tuple) -> str:
        """
        Convert group key tuple to string identifier.
        
        For arxiv: "paper_id:section_idx"
        For quotes: "quote_id"
        
        Args:
            group_key: Tuple from _get_group_key
        
        Returns:
            String identifier for the group
        """
        pass
    
    # =========================================================================
    # GIST Pipeline
    # =========================================================================
    
    def search(
        self,
        query: str,
        top_k: int = 21,
        return_chunks: bool = False
    ) -> List[RetrievedPaper]:
        """
        Execute full GIST retrieval pipeline with paper-level grouping.
        
        Pipeline:
            1. Retrieve top_k² CHUNKS from BM25 and dense indexes
            2. GIST selection on each pool (chunk-level)
            3. RRF fusion (chunk-level)
            4. GROUP chunks into SECTIONS
            5. GROUP sections into PAPERS
            6. Select sections from top (top_k × φ) papers for ColBERT
            7. ColBERT rerank on section full text
            8. Aggregate sections back to papers
            9. Cross-encoder rerank papers
            10. Return top_k PAPERS
        
        Args:
            query: Query text
            top_k: Number of PAPERS to return
            return_chunks: If True, include chunk details in results
        
        Returns:
            List of RetrievedPaper sorted by final_score (descending)
        """
        fib = self.config.fibonacci_sequence
        
        # Fibonacci cascade sizing
        retrieval_limit = top_k * top_k  # Broad recall (chunks)
        gist_limit = get_fibonacci_lower(retrieval_limit, fib)  # Post-GIST (chunks)
        section_limit = get_fibonacci_lower(gist_limit, fib)  # Sections to create
        
        # For paper-level cascade
        paper_pool_size = top_k  # Papers to consider for ColBERT
        colbert_papers = top_k  # ColBERT operates on sections from top_k papers
        final_papers = get_fibonacci_lower(top_k, fib)  # Final output: fib_lower(top_k) papers
        
        print(f"GIST Pipeline: retrieve {retrieval_limit} chunks → GIST {gist_limit} → sections {section_limit} → papers {paper_pool_size} → ColBERT {colbert_papers} papers → Final {final_papers} papers")
        
        # Step 1: Parallel retrieval (chunks)
        print(f"  [1/8] Retrieving {retrieval_limit} chunks from each pool...")
        bm25_pool = self._retrieve_bm25(query, retrieval_limit)
        dense_pool = self._retrieve_dense(query, retrieval_limit)
        
        print(f"        BM25: {len(bm25_pool)}, Dense: {len(dense_pool)}")
        
        # Step 2: GIST selection on BM25 pool
        print(f"  [2/8] GIST selection on BM25 pool...")
        bm25_selected = self._gist_select_pool(
            pool=bm25_pool,
            query=query,
            k=gist_limit // 2,
            pool_type='bm25'
        )
        print(f"        Selected {len(bm25_selected)} chunks from BM25")
        
        # Step 3: GIST selection on dense pool
        print(f"  [3/8] GIST selection on dense pool...")
        dense_selected = self._gist_select_pool(
            pool=dense_pool,
            query=query,
            k=gist_limit // 2,
            pool_type='dense'
        )
        print(f"        Selected {len(dense_selected)} chunks from dense")
        
        # Step 4: RRF fusion (chunks)
        print(f"  [4/8] RRF fusion...")
        fused_chunks = self._rrf_fusion(bm25_selected, dense_selected)
        print(f"        Fused: {len(fused_chunks)} unique chunks")
        
        # Step 5: Group chunks into sections
        print(f"  [5/8] Grouping chunks into sections...")
        sections = self._group_and_reconstruct(fused_chunks, limit=section_limit)
        print(f"        Created {len(sections)} sections")
        
        # Step 6: Group sections into papers and select top papers
        print(f"  [6/8] Grouping sections into papers...")
        papers = self._group_sections_into_papers(sections, paper_pool_size)
        print(f"        Created {len(papers)} papers from top scoring papers")
        
        # Step 7: ColBERT rerank sections within selected papers
        if self.config.use_colbert and len(papers) > 0:
            print(f"  [7/8] ColBERT reranking sections from {len(papers)} papers...")
            papers = self._colbert_rerank_papers(query, papers)
        else:
            print(f"  [7/8] ColBERT skipped")
        
        # Step 8: Cross-encoder rerank papers
        if self.config.use_cross_encoder and len(papers) > 0:
            print(f"  [8/8] Cross-Encoder reranking papers...")
            papers = self._cross_encoder_rerank_papers(query, papers, final_papers)
        else:
            print(f"  [8/8] Cross-Encoder skipped")
        
        # Ensure we don't exceed requested top_k
        results = papers[:top_k]
        
        # Set final scores
        for paper in results:
            if paper.final_score is None:
                paper.final_score = paper.rrf_score
        
        # Optionally strip chunks to reduce memory
        if not return_chunks:
            for paper in results:
                for section in paper.sections:
                    section.chunks = []
        
        print(f"  Returning {len(results)} papers")
        return results
    
    def _gist_select_pool(
        self,
        pool: List[RetrievedDoc],
        query: str,
        k: int,
        pool_type: str  # 'bm25' or 'dense'
    ) -> List[RetrievedDoc]:
        """
        Apply GIST selection to a single retrieval pool.
        
        Args:
            pool: Retrieved documents from one method
            query: Query text
            k: Number to select
            pool_type: 'bm25' or 'dense' (determines how to compute matrices)
        
        Returns:
            Selected documents with gist_rank populated
        """
        if len(pool) <= k:
            for rank, doc in enumerate(pool, 1):
                doc.gist_rank = rank
            return pool
        
        doc_ids = [doc.doc_id for doc in pool]
        
        # Build coverage matrix (doc-doc similarity)
        if pool_type == 'bm25':
            coverage_matrix = self._get_bm25_doc_doc_scores(doc_ids)
            utility_vector = self._get_bm25_query_scores(doc_ids, query)
        else:  # dense
            embeddings = self._get_dense_embeddings(doc_ids)
            query_emb = self._get_query_embedding(query)
            
            from sklearn.metrics.pairwise import cosine_similarity
            coverage_matrix = cosine_similarity(embeddings)
            utility_vector = cosine_similarity(embeddings, query_emb.reshape(1, -1)).flatten()
        
        # GIST selection
        selected_indices = gist_select(
            coverage_matrix=coverage_matrix,
            utility_vector=utility_vector,
            k=k,
            lambda_param=self.config.gist_lambda
        )
        
        # Build result list with GIST ranks
        selected = []
        for rank, idx in enumerate(selected_indices, 1):
            doc = pool[idx]
            doc.gist_rank = rank
            selected.append(doc)
        
        return selected
    
    def _rrf_fusion(
        self,
        bm25_selected: List[RetrievedDoc],
        dense_selected: List[RetrievedDoc]
    ) -> List[RetrievedDoc]:
        """
        Fuse two GIST-selected pools using Reciprocal Rank Fusion.
        """
        bm25_by_id = {doc.doc_id: doc for doc in bm25_selected}
        dense_by_id = {doc.doc_id: doc for doc in dense_selected}
        
        all_ids = set(bm25_by_id.keys()) | set(dense_by_id.keys())
        
        fused = []
        for doc_id in all_ids:
            bm25_doc = bm25_by_id.get(doc_id)
            dense_doc = dense_by_id.get(doc_id)
            
            doc = bm25_doc or dense_doc
            
            if bm25_doc:
                doc.bm25_score = bm25_doc.bm25_score
                doc.bm25_rank = bm25_doc.bm25_rank
            if dense_doc:
                doc.dense_score = dense_doc.dense_score
                doc.dense_rank = dense_doc.dense_rank
            
            ranks = [
                bm25_doc.gist_rank if bm25_doc else None,
                dense_doc.gist_rank if dense_doc else None
            ]
            doc.rrf_score = compute_rrf_score(ranks, k=self.config.rrf_k)
            
            fused.append(doc)
        
        fused.sort(key=lambda d: d.rrf_score, reverse=True)
        return fused
    
    def _group_and_reconstruct(
        self,
        chunks: List[RetrievedDoc],
        limit: int
    ) -> List[RetrievedGroup]:
        """
        Group chunks by parent and reconstruct full text.
        
        Args:
            chunks: RRF-fused chunks
            limit: Maximum number of groups to process
        
        Returns:
            List of RetrievedGroup with full_text populated
        """
        # Group chunks by key
        groups_dict = defaultdict(lambda: {'chunks': [], 'rrf_sum': 0.0})
        
        for chunk in chunks:
            key = self._get_group_key(chunk)
            groups_dict[key]['chunks'].append(chunk)
            groups_dict[key]['rrf_sum'] += chunk.rrf_score or 0.0
        
        # Sort groups by aggregated RRF score
        sorted_keys = sorted(
            groups_dict.keys(),
            key=lambda k: groups_dict[k]['rrf_sum'],
            reverse=True
        )[:limit]
        
        # Reconstruct full text for top groups
        groups = []
        for key in sorted_keys:
            matched_chunks = groups_dict[key]['chunks']
            
            # Fetch ALL chunks for this group (section or quote)
            all_chunks = self._fetch_all_chunks_for_group(key)
            all_chunks.sort(key=lambda c: c.metadata.get('chunk_idx', 0))
            
            # Reconstruct full text
            full_text = "\n".join([c.content for c in all_chunks])
            
            group = RetrievedGroup(
                group_id=self._group_key_to_id(key),
                group_key=key,
                full_text=full_text,
                chunks=matched_chunks,
                rrf_score=groups_dict[key]['rrf_sum'],
                metadata={
                    'num_chunks': len(all_chunks),
                    'num_matched': len(matched_chunks)
                }
            )
            groups.append(group)
        
        return groups
    
    def _group_sections_into_papers(
        self,
        sections: List[RetrievedGroup],
        max_papers: int
    ) -> List[RetrievedPaper]:
        """
        Group sections by paper and select top papers.
        
        Args:
            sections: List of section groups
            max_papers: Maximum number of papers to include
        
        Returns:
            List of RetrievedPaper with sections grouped by paper_id
        """
        # Group sections by paper_id (first element of group_key)
        papers_dict = defaultdict(lambda: {'sections': [], 'rrf_sum': 0.0})
        
        for section in sections:
            # Extract paper_id from group_key (paper_id, section_idx)
            paper_id = section.group_key[0]
            papers_dict[paper_id]['sections'].append(section)
            papers_dict[paper_id]['rrf_sum'] += section.rrf_score or 0.0
        
        # Sort papers by aggregated RRF score
        sorted_paper_ids = sorted(
            papers_dict.keys(),
            key=lambda p: papers_dict[p]['rrf_sum'],
            reverse=True
        )[:max_papers]
        
        # Build RetrievedPaper objects
        papers = []
        for paper_id in sorted_paper_ids:
            sections_list = papers_dict[paper_id]['sections']
            sections_list.sort(key=lambda s: s.group_key[1])  # Sort by section_idx
            
            # Concatenate all section text
            full_text = "\n\n".join([s.full_text for s in sections_list])
            
            paper = RetrievedPaper(
                paper_id=paper_id,
                sections=sections_list,
                full_text=full_text,
                rrf_score=papers_dict[paper_id]['rrf_sum'],
                metadata={
                    'num_sections': len(sections_list)
                }
            )
            papers.append(paper)
        
        return papers
    
    def _colbert_rerank_groups(
        self,
        query: str,
        groups: List[RetrievedGroup],
        limit: int
    ) -> List[RetrievedGroup]:
        """Rerank groups using ColBERT on full_text."""
        if not self.colbert_scorer.available:
            print("        ColBERT unavailable, skipping...")
            return groups[:limit]
        
        texts = [g.full_text for g in groups]
        scores = self.colbert_scorer.score(query, texts)
        
        for group, score in zip(groups, scores):
            group.colbert_score = float(score)
        
        groups.sort(key=lambda g: g.colbert_score, reverse=True)
        return groups[:limit]
    
    def _cross_encoder_rerank_groups(
        self,
        query: str,
        groups: List[RetrievedGroup],
        limit: int
    ) -> List[RetrievedGroup]:
        """Rerank groups using cross-encoder on full_text."""
        if not self.cross_encoder_scorer.available:
            print("        Cross-Encoder unavailable, skipping...")
            return groups[:limit]
        
        texts = [g.full_text for g in groups]
        scores = self.cross_encoder_scorer.score(query, texts)
        
        for group, score in zip(groups, scores):
            group.cross_encoder_score = float(score)
            group.final_score = float(score)
        
        groups.sort(key=lambda g: g.cross_encoder_score, reverse=True)
        return groups[:limit]
    
    def _colbert_rerank_papers(
        self,
        query: str,
        papers: List[RetrievedPaper]
    ) -> List[RetrievedPaper]:
        """Rerank papers by reranking their sections with ColBERT."""
        if not self.colbert_scorer.available:
            print("        ColBERT unavailable, skipping...")
            return papers
        
        # Rerank all sections across all papers
        all_sections = []
        for paper in papers:
            all_sections.extend(paper.sections)
        
        texts = [s.full_text for s in all_sections]
        scores = self.colbert_scorer.score(query, texts)
        
        for section, score in zip(all_sections, scores):
            section.colbert_score = float(score)
        
        # Update paper scores: max ColBERT score across sections
        for paper in papers:
            if paper.sections:
                paper.colbert_score = max(s.colbert_score for s in paper.sections if s.colbert_score is not None)
        
        # Sort papers by max section score
        papers.sort(key=lambda p: p.colbert_score or 0.0, reverse=True)
        return papers
    
    def _cross_encoder_rerank_papers(
        self,
        query: str,
        papers: List[RetrievedPaper],
        limit: int
    ) -> List[RetrievedPaper]:
        """Rerank papers using cross-encoder on full paper text."""
        if not self.cross_encoder_scorer.available:
            print("        Cross-Encoder unavailable, skipping...")
            return papers[:limit]
        
        # Score full paper text (concatenated sections)
        texts = [p.full_text for p in papers]
        scores = self.cross_encoder_scorer.score(query, texts)
        
        for paper, score in zip(papers, scores):
            paper.cross_encoder_score = float(score)
            paper.final_score = float(score)
        
        papers.sort(key=lambda p: p.cross_encoder_score, reverse=True)
        return papers[:limit]
    
    # =========================================================================
    # Backward Compatibility: Chunk-Level Search
    # =========================================================================
    
    def search_chunks(
        self,
        query: str,
        top_k: int = 21
    ) -> List[RetrievedDoc]:
        """
        Search at chunk level (no grouping/reconstruction).
        
        For cases where you want raw chunk results without grouping.
        ColBERT and cross-encoder operate on individual chunks.
        """
        fib = self.config.fibonacci_sequence
        
        retrieval_limit = top_k * top_k
        gist_limit = get_fibonacci_lower(retrieval_limit, fib)
        colbert_limit = top_k
        final_limit = get_fibonacci_lower(top_k, fib)
        
        # Retrieve
        bm25_pool = self._retrieve_bm25(query, retrieval_limit)
        dense_pool = self._retrieve_dense(query, retrieval_limit)
        
        # GIST select
        bm25_selected = self._gist_select_pool(bm25_pool, query, gist_limit // 2, 'bm25')
        dense_selected = self._gist_select_pool(dense_pool, query, gist_limit // 2, 'dense')
        
        # Fuse
        fused = self._rrf_fusion(bm25_selected, dense_selected)
        
        # ColBERT on chunks
        if self.config.use_colbert and len(fused) > colbert_limit:
            fused = self._colbert_rerank_chunks(query, fused, colbert_limit)
        
        # Cross-encoder on chunks
        if self.config.use_cross_encoder and len(fused) > final_limit:
            fused = self._cross_encoder_rerank_chunks(query, fused, final_limit)
        
        results = fused[:top_k]
        for doc in results:
            if doc.final_score is None:
                doc.final_score = doc.rrf_score
        
        return results
    
    def _colbert_rerank_chunks(
        self,
        query: str,
        docs: List[RetrievedDoc],
        limit: int
    ) -> List[RetrievedDoc]:
        """Rerank chunks using ColBERT."""
        if not self.colbert_scorer.available:
            return docs[:limit]
        
        contents = [doc.content for doc in docs]
        scores = self.colbert_scorer.score(query, contents)
        
        for doc, score in zip(docs, scores):
            doc.colbert_score = float(score)
        
        docs.sort(key=lambda d: d.colbert_score, reverse=True)
        return docs[:limit]
    
    def _cross_encoder_rerank_chunks(
        self,
        query: str,
        docs: List[RetrievedDoc],
        limit: int
    ) -> List[RetrievedDoc]:
        """Rerank chunks using cross-encoder."""
        if not self.cross_encoder_scorer.available:
            return docs[:limit]
        
        contents = [doc.content for doc in docs]
        scores = self.cross_encoder_scorer.score(query, contents)
        
        for doc, score in zip(docs, scores):
            doc.cross_encoder_score = float(score)
            doc.final_score = float(score)
        
        docs.sort(key=lambda d: d.cross_encoder_score, reverse=True)
        return docs[:limit]


# =============================================================================
# Utility Functions
# =============================================================================

def format_results_markdown(results: List[RetrievedDoc]) -> str:
    """
    Format chunk-level retrieval results as markdown.
    
    Args:
        results: List of RetrievedDoc from search_chunks()
    
    Returns:
        Markdown-formatted string
    """
    lines = ["# Retrieval Results\n"]
    
    for i, doc in enumerate(results, 1):
        lines.append(f"## {i}. {doc.doc_id}\n")
        
        # Scores
        scores = []
        if doc.bm25_score is not None:
            scores.append(f"BM25: {doc.bm25_score:.4f}")
        if doc.dense_score is not None:
            scores.append(f"Dense: {doc.dense_score:.4f}")
        if doc.rrf_score is not None:
            scores.append(f"RRF: {doc.rrf_score:.4f}")
        if doc.colbert_score is not None:
            scores.append(f"ColBERT: {doc.colbert_score:.4f}")
        if doc.cross_encoder_score is not None:
            scores.append(f"CrossEnc: {doc.cross_encoder_score:.4f}")
        if doc.final_score is not None:
            scores.append(f"**Final: {doc.final_score:.4f}**")
        
        lines.append(f"**Scores:** {' | '.join(scores)}\n")
        
        # Metadata
        if doc.metadata:
            meta_str = ", ".join([f"{k}={v}" for k, v in doc.metadata.items()])
            lines.append(f"**Metadata:** {meta_str}\n")
        
        # Content preview
        preview = doc.content[:500] + "..." if len(doc.content) > 500 else doc.content
        lines.append(f"**Content:**\n{preview}\n")
        
        lines.append("---\n")
    
    return "\n".join(lines)


def format_groups_markdown(groups: List[RetrievedGroup], include_full_text: bool = False) -> str:
    """
    Format group-level retrieval results as markdown.
    
    Args:
        groups: List of RetrievedGroup from search()
        include_full_text: If True, include full reconstructed text
    
    Returns:
        Markdown-formatted string
    """
    lines = ["# Retrieval Results (Grouped)\n"]
    
    for i, group in enumerate(groups, 1):
        lines.append(f"## {i}. {group.group_id}\n")
        
        # Scores
        scores = []
        if group.rrf_score is not None:
            scores.append(f"RRF: {group.rrf_score:.4f}")
        if group.colbert_score is not None:
            scores.append(f"ColBERT: {group.colbert_score:.4f}")
        if group.cross_encoder_score is not None:
            scores.append(f"CrossEnc: {group.cross_encoder_score:.4f}")
        if group.final_score is not None:
            scores.append(f"**Final: {group.final_score:.4f}**")
        
        lines.append(f"**Scores:** {' | '.join(scores)}\n")
        
        # Metadata
        if group.metadata:
            meta_str = ", ".join([f"{k}={v}" for k, v in group.metadata.items()])
            lines.append(f"**Metadata:** {meta_str}\n")
        
        # Content
        if include_full_text:
            lines.append(f"### Full Text\n\n{group.full_text}\n")
        else:
            preview = group.full_text[:500] + "..." if len(group.full_text) > 500 else group.full_text
            lines.append(f"**Preview:**\n{preview}\n")
        
        lines.append("---\n")
    
    return "\n".join(lines)


def format_papers_markdown(papers: List[RetrievedPaper], include_sections: bool = True) -> str:
    """
    Format paper-level retrieval results as markdown.
    
    Args:
        papers: List of RetrievedPaper from search()
        include_sections: If True, show individual sections
    
    Returns:
        Markdown-formatted string
    """
    lines = ["# Retrieval Results (Papers)\n"]
    
    for i, paper in enumerate(papers, 1):
        lines.append(f"## {i}. {paper.paper_id}\n")
        
        # Scores
        scores = []
        if paper.rrf_score is not None:
            scores.append(f"RRF: {paper.rrf_score:.4f}")
        if paper.colbert_score is not None:
            scores.append(f"ColBERT: {paper.colbert_score:.4f}")
        if paper.cross_encoder_score is not None:
            scores.append(f"CrossEnc: {paper.cross_encoder_score:.4f}")
        if paper.final_score is not None:
            scores.append(f"**Final: {paper.final_score:.4f}**")
        
        lines.append(f"**Scores:** {' | '.join(scores)}\n")
        
        # Metadata
        if paper.metadata:
            meta_str = ", ".join([f"{k}={v}" for k, v in paper.metadata.items()])
            lines.append(f"**Metadata:** {meta_str}\n")
        
        # Sections
        if include_sections and paper.sections:
            lines.append(f"\n**Sections ({len(paper.sections)}):**\n")
            for j, section in enumerate(paper.sections, 1):
                section_scores = []
                if section.rrf_score is not None:
                    section_scores.append(f"RRF: {section.rrf_score:.4f}")
                if section.colbert_score is not None:
                    section_scores.append(f"ColBERT: {section.colbert_score:.4f}")
                
                lines.append(f"  {j}. {section.group_id} | {' | '.join(section_scores)}")
                preview = section.full_text[:200] + "..." if len(section.full_text) > 200 else section.full_text
                lines.append(f"     {preview}\n")
        
        lines.append("---\n")
    
    return "\n".join(lines)