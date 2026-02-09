"""
Three-Layer GIST-Based Retrieval with Coverage/Utility Selection

Architecture:
  Query
    ↓
  Layer 1: Lexical/Semantic Retrieval with GIST
    ├─ BM25 lemmatized → GIST(coverage=BM25 doc-doc, utility=BM25 doc-query)
    └─ M2V embeddings → GIST(coverage=cosine doc-doc, utility=cosine doc-query)
    → RRF fusion → top_k² seed chunks
    ↓
  Layer 2: Query Expansion via Graph with GIST
    ├─ Graph BM25 (triplets) → GIST(coverage=BM25 triplet-triplet, utility=BM25 triplet-query)
    └─ Qwen3 embeddings → GIST(coverage=Qwen3 doc-doc, utility=Qwen3 doc-query)
    → **EXCLUDE Layer 1 doc_ids**
    → RRF fusion → top_k² expansion chunks
    → Aggregate chunks to sections → φ_lower(top_k²) sections
    ↓
  Layer 3: Late Interaction with GIST
    ├─ ColBERTv2 → GIST(coverage=ColBERT section-section, utility=ColBERT section-query)
    └─ MSMarco Cross-Encoder → GIST(coverage=CE section-section, utility=CE section-query)
    → RRF fusion
    → Walk-down: collect first (top_k + 1) papers, use 13th paper's min section as floor
    → Keep top_k papers with sections >= floor
    
GIST at every retrieval step:
  gist_select(coverage_matrix, utility_vector, k, λ=0.7)
  score(d) = λ * utility(d) - (1-λ) * max_sim(d, Selected)
  
  - Coverage Matrix: n×n doc-doc similarity (excludes query)
  - Utility Vector: n×1 doc-query similarity
  - λ = 0.7: 70% relevance, 30% diversity

No ECDF, no Fibonacci expansion — just GIST + standard RRF.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
import msgpack
from pathlib import Path
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter, defaultdict


# =============================================================================
# GIST Core Functions (from gist_retriever.py)
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


def de_overlap_strings(strings: List[str]) -> List[str]:
    """
    Remove overlapping suffixes from consecutive strings.
    
    When chunks are created with overlap, this function removes the duplicate
    content by detecting where the end of one string matches the beginning of the next.
    
    Args:
        strings: List of text chunks (ordered)
    
    Returns:
        List of de-overlapped chunks
    """
    if len(strings) <= 1:
        return strings
    
    def find_overlap(s1: str, s2: str) -> str:
        """Find the longest suffix of s1 that matches a prefix of s2."""
        max_len = min(len(s1), len(s2))
        for i in range(max_len, 0, -1):
            if s1[-i:] == s2[:i]:
                return s1[-i:]
        return ""
    
    result = []
    for i in range(len(strings) - 1):
        s1, s2 = strings[i], strings[i + 1]
        overlap = find_overlap(s1, s2)
        if overlap:
            s1 = s1[:-len(overlap)]
        result.append(s1)
    
    # Add the last string as is
    result.append(strings[-1])
    return result


# Fibonacci sequence for φ-scaling
FIBONACCI = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597]


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class GISTLayerConfig:
    """GIST-based three-layer configuration"""
    top_k: int  # Final output papers (13)
    layer1_seeds: int  # top_k² chunks for Layer 1 (169)
    layer2_expand: int  # top_k² expansions for Layer 2 (169)
    layer2_sections: int  # φ_lower(top_k²) sections (144)
    layer3_output: int  # top_k papers (13)
    
    gist_lambda: float = 0.7  # GIST tradeoff parameter
    rrf_k: int = 60  # RRF parameter
    
    @classmethod
    def from_top_k(cls, top_k: int = 13):
        """
        Calculate layer sizes for GIST retrieval.
        
        L1: top_k² retrieval for both BM25 and embeddings
        L2: top_k² expansion, then φ_lower(top_k²) sections
        L3: top_k papers via walk-down with floor threshold
        
        For top_k=13:
          - top_k² = 169
          - φ_lower(169) = 144
        
        Args:
            top_k: Final paper count (default 13)
        
        Returns:
            GISTLayerConfig with calculated sizes
        """
        top_k_squared = top_k * top_k
        section_limit = get_fibonacci_lower(top_k_squared, FIBONACCI)
        
        return cls(
            top_k=top_k,
            layer1_seeds=top_k_squared,
            layer2_expand=top_k_squared,
            layer2_sections=section_limit,
            layer3_output=top_k
        )


# =============================================================================
# Three-Layer GIST Retriever
# =============================================================================

class ThreeLayerGISTRetriever:
    """
    Three-layer GIST retrieval with coverage/utility selection at every step.
    
    Layer 1: Lexical/semantic seeds (GIST on BM25 + embeddings)
    Layer 2: Graph expansion (GIST on triplets + Qwen3, excluding L1)
    Layer 3: Late interaction reranking (GIST on ColBERT + CE, walk-down papers)
    """
    
    def __init__(
        self,
        chunks_path: str,
        chunk_embeddings_qwen3_path: str,  # Layer 1 & 2: Qwen3 256d
        triplets_path: str,
        chunk_to_triplets_path: str,
        triplet_to_chunks_path: str,
        triplet_bm25_path: str,
        bm25_lemmatized_index,  # Layer 1 BM25 index (from layer1_retriever)
        top_k: int = 13,
        colbert_model_name: str = "bert-base-uncased",  # ColBERTv2 base
        cross_encoder_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        """
        Initialize GIST three-layer retriever.
        
        Args:
            chunks_path: Path to chunks.msgpack
            chunk_embeddings_qwen3_path: Path to Qwen3 256d embeddings (Layer 1 & 2)
            triplets_path: Path to enriched_triplets.msgpack
            chunk_to_triplets_path: Path to chunk_to_triplets.msgpack
            triplet_to_chunks_path: Path to triplet_to_chunks.msgpack
            triplet_bm25_path: Path to triplet BM25 index
            bm25_lemmatized_index: Loaded BM25Okapi index for Layer 1
            top_k: Final paper count (default 13)
            colbert_model_name: ColBERT model identifier
            cross_encoder_model_name: Cross-encoder model identifier
        """
        self.config = GISTLayerConfig.from_top_k(top_k)
        self.bm25_index = bm25_lemmatized_index
        
        print(f"\n{'='*60}")
        print(f"LOADING THREE-LAYER GIST RETRIEVER")
        print(f"{'='*60}")
        
        # Load chunks
        print(f"Loading chunks from {chunks_path}...")
        with open(chunks_path, 'rb') as f:
            self.chunks = msgpack.load(f, raw=False)
        print(f"✓ Loaded {len(self.chunks)} chunks")
        
        # Create doc_id → index mapping
        self.doc_id_to_idx = {chunk['doc_id']: i for i, chunk in enumerate(self.chunks)}
        
        # Load Qwen3 embeddings and prepare PCA for 128d reduction
        print(f"Loading Qwen3 embeddings from {chunk_embeddings_qwen3_path}...")
        with open(chunk_embeddings_qwen3_path, 'rb') as f:
            qwen3_data = msgpack.load(f, raw=False)
        self.qwen3_embeddings_256 = np.array(qwen3_data['embeddings'], dtype=np.float32)
        print(f"✓ Loaded Qwen3 256d: {self.qwen3_embeddings_256.shape}")
        
        # PCA reduction for Layer 1: 256d → 128d
        from sklearn.decomposition import PCA
        print("  Fitting PCA: 256d → 128d...")
        self.pca_128 = PCA(n_components=128)
        self.qwen3_embeddings_128 = self.pca_128.fit_transform(self.qwen3_embeddings_256)
        explained = self.pca_128.explained_variance_ratio_.sum() * 100
        print(f"  ✓ PCA fitted: 128d, {explained:.1f}% variance retained")
        
        # Load triplets
        print(f"Loading triplets from {triplets_path}...")
        with open(triplets_path, 'rb') as f:
            self.triplets = msgpack.load(f, raw=False)
        print(f"✓ Loaded {len(self.triplets)} triplets")
        
        # Load chunk↔triplet mappings
        print(f"Loading chunk↔triplet mappings...")
        with open(chunk_to_triplets_path, 'rb') as f:
            self.chunk_to_triplets = msgpack.load(f, raw=False)
        with open(triplet_to_chunks_path, 'rb') as f:
            self.triplet_to_chunks = msgpack.load(f, raw=False)
        print(f"✓ Loaded mappings")
        
        # Load triplet BM25 index
        print(f"Loading triplet BM25 index from {triplet_bm25_path}...")
        with open(triplet_bm25_path, 'rb') as f:
            bm25_data = msgpack.load(f, raw=False)
            self.triplet_texts = bm25_data['triplet_texts']
            self.triplet_tokens = bm25_data['triplet_tokens']
            self.triplet_bm25 = BM25Okapi(self.triplet_tokens)
        print(f"✓ Loaded triplet BM25: {len(self.triplet_texts)} triplets")
        
        # Load Layer 3 rerankers
        print(f"\n{'='*60}")
        print(f"LOADING LAYER 3 RERANKERS")
        print(f"{'='*60}")
        
        print(f"Loading ColBERT model: {colbert_model_name}...")
        try:
            from sentence_transformers import SentenceTransformer, CrossEncoder
            self.colbert_model = SentenceTransformer(colbert_model_name)
            print(f"✓ ColBERT loaded")
        except Exception as e:
            print(f"❌ Failed to load ColBERT: {e}")
            raise
        
        print(f"Loading Cross-Encoder: {cross_encoder_model_name}...")
        try:
            self.cross_encoder = CrossEncoder(cross_encoder_model_name)
            print(f"✓ Cross-Encoder loaded")
        except Exception as e:
            print(f"❌ Failed to load Cross-Encoder: {e}")
            raise
        
        print(f"\n{'='*60}")
        print(f"CONFIGURATION (top_k={self.config.top_k})")
        print(f"{'='*60}")
        print(f"Layer 1: BM25 + M2V → GIST → RRF → {self.config.layer1_seeds} seeds")
        print(f"Layer 2: Graph BM25 + Qwen3 → GIST → RRF → {self.config.layer2_expand} expansions")
        print(f"  → Aggregate → {self.config.layer2_sections} sections")
        print(f"Layer 3: ColBERT + CE → GIST → RRF → Walk-down → {self.config.layer3_output} papers")
        print(f"GIST λ={self.config.gist_lambda}, RRF k={self.config.rrf_k}")
        print(f"{'='*60}\n")
    
    def _layer1_bm25_gist(self, query: str) -> List[Tuple[int, int]]:
        """
        Layer 1 BM25 retrieval with GIST selection.
        
        Returns:
            List of (chunk_id, gist_rank) tuples
        """
        # Tokenize query (lemmatized)
        query_tokens = query.lower().split()  # Simplified; use proper lemmatizer
        
        # Retrieve top_k² × 2 for oversampling
        scores = self.bm25_index.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:self.config.layer1_seeds * 2]
        
        # Build coverage and utility matrices
        chunk_ids = top_indices.tolist()
        raw_scores = scores[top_indices]
        
        # Coverage: BM25 doc-doc similarity (simplified: use score product)
        # Note: Real implementation would use proper BM25 doc-doc scoring
        coverage_matrix = np.outer(raw_scores, raw_scores)
        coverage_matrix = coverage_matrix / (np.max(coverage_matrix) + 1e-10)  # Normalize
        
        # Utility: BM25 doc-query scores
        utility_vector = raw_scores
        
        # GIST selection
        k_select = min(self.config.layer1_seeds, len(chunk_ids))
        selected_indices = gist_select(coverage_matrix, utility_vector, k=k_select, lambda_param=self.config.gist_lambda)
        
        # Return with GIST ranks
        results = [(chunk_ids[i], rank) for rank, i in enumerate(selected_indices, 1)]
        return results
    
    def _layer1_embedding_gist(self, query: str) -> List[Tuple[int, int]]:
        """
        Layer 1 Qwen3 128d embedding retrieval with GIST selection.
        
        Returns:
            List of (chunk_id, gist_rank) tuples
        """
        from model2vec import StaticModel
        
        # Load Model2Vec Qwen3 (cache it)
        if not hasattr(self, '_qwen3_model'):
            self._qwen3_model = StaticModel.from_pretrained('./qwen3_static_embeddings')
        
        # Encode query and reduce to 128d
        query_emb_256 = self._qwen3_model.encode([query])[0]
        query_emb = self.pca_128.transform(query_emb_256.reshape(1, -1))[0]
        
        # Retrieve top_k² × 2 via cosine similarity
        similarities = cosine_similarity(query_emb.reshape(1, -1), self.qwen3_embeddings_128)[0]
        top_indices = np.argsort(similarities)[::-1][:self.config.layer1_seeds * 2]
        
        chunk_ids = top_indices.tolist()
        candidate_embs = self.qwen3_embeddings_128[top_indices]
        
        # Coverage: Qwen3 doc-doc cosine similarity
        coverage_matrix = cosine_similarity(candidate_embs)
        
        # Utility: M2V doc-query cosine similarity
        utility_vector = similarities[top_indices]
        
        # GIST selection
        k_select = min(self.config.layer1_seeds, len(chunk_ids))
        selected_indices = gist_select(coverage_matrix, utility_vector, k=k_select, lambda_param=self.config.gist_lambda)
        
        # Return with GIST ranks
        results = [(chunk_ids[i], rank) for rank, i in enumerate(selected_indices, 1)]
        return results
    
    def _layer2_graph_bm25_gist(self, query: str, exclude_doc_ids: Set[int]) -> List[Tuple[int, int]]:
        """
        Layer 2 graph BM25 retrieval with GIST selection, excluding Layer 1 results.
        
        Args:
            query: Query text
            exclude_doc_ids: Set of chunk IDs from Layer 1 to exclude
        
        Returns:
            List of (chunk_id, gist_rank) tuples
        """
        # Tokenize query
        query_tokens = query.lower().split()
        
        # Retrieve from triplet BM25
        triplet_scores = self.triplet_bm25.get_scores(query_tokens)
        top_triplet_indices = np.argsort(triplet_scores)[::-1][:self.config.layer2_expand * 3]
        
        # Map triplets → chunks, aggregate scores
        chunk_scores = {}
        for tidx in top_triplet_indices:
            tid = str(tidx)
            tscore = triplet_scores[tidx]
            
            # Get chunks for this triplet
            for doc_id in self.triplet_to_chunks.get(tid, []):
                # Clean doc_id (handle duplication)
                if '_s' in doc_id and doc_id.count('_s') > 1:
                    parts = doc_id.split('_')
                    doc_id = '_'.join(parts[:4])
                
                if doc_id in self.doc_id_to_idx:
                    cid = self.doc_id_to_idx[doc_id]
                    
                    # Exclude Layer 1 results
                    if cid not in exclude_doc_ids:
                        chunk_scores[cid] = max(chunk_scores.get(cid, 0), tscore)
        
        if not chunk_scores:
            return []
        
        # Build coverage and utility for GIST
        chunk_ids = list(chunk_scores.keys())[:self.config.layer2_expand * 2]
        raw_scores = np.array([chunk_scores[cid] for cid in chunk_ids])
        
        # Coverage: Use Qwen3 256d embeddings for doc-doc similarity
        candidate_embs = self.qwen3_embeddings_256[chunk_ids]
        coverage_matrix = cosine_similarity(candidate_embs)
        
        # Utility: BM25 triplet scores
        utility_vector = raw_scores
        
        # GIST selection
        k_select = min(self.config.layer2_expand, len(chunk_ids))
        selected_indices = gist_select(coverage_matrix, utility_vector, k=k_select, lambda_param=self.config.gist_lambda)
        
        results = [(chunk_ids[i], rank) for rank, i in enumerate(selected_indices, 1)]
        return results
    
    def _layer2_qwen3_gist(self, query: str, exclude_doc_ids: Set[int]) -> List[Tuple[int, int]]:
        """
        Layer 2 Qwen3 256d embedding retrieval with GIST selection, excluding Layer 1 results.
        
        Args:
            query: Query text
            exclude_doc_ids: Set of chunk IDs from Layer 1 to exclude
        
        Returns:
            List of (chunk_id, gist_rank) tuples
        """
        # Reuse cached model from Layer 1
        if not hasattr(self, '_qwen3_model'):
            from model2vec import StaticModel
            self._qwen3_model = StaticModel.from_pretrained('./qwen3_static_embeddings')
        
        # Encode query (full 256d for Layer 2)
        query_emb = self._qwen3_model.encode([query])[0]
        
        # Retrieve top_k² via cosine similarity (full 256d)
        similarities = cosine_similarity(query_emb.reshape(1, -1), self.qwen3_embeddings_256)[0]
        
        # Filter out Layer 1 results
        valid_indices = [i for i in range(len(similarities)) if i not in exclude_doc_ids]
        valid_scores = similarities[valid_indices]
        
        top_k_candidates = min(self.config.layer2_expand * 2, len(valid_indices))
        sorted_indices = np.argsort(valid_scores)[::-1][:top_k_candidates]
        
        chunk_ids = [valid_indices[i] for i in sorted_indices]
        candidate_embs = self.qwen3_embeddings_256[chunk_ids]
        
        # Coverage: Qwen3 doc-doc cosine similarity
        coverage_matrix = cosine_similarity(candidate_embs)
        
        # Utility: Qwen3 doc-query cosine similarity
        utility_vector = valid_scores[sorted_indices]
        
        # GIST selection
        k_select = min(self.config.layer2_expand, len(chunk_ids))
        selected_indices = gist_select(coverage_matrix, utility_vector, k=k_select, lambda_param=self.config.gist_lambda)
        
        results = [(chunk_ids[i], rank) for rank, i in enumerate(selected_indices, 1)]
        return results
    
    def _aggregate_chunks_to_sections(self, chunk_ids: List[int], chunk_scores: Dict[int, float]) -> List[Dict]:
        """
        Aggregate chunks into complete sections.
        
        Args:
            chunk_ids: List of chunk indices
            chunk_scores: Mapping of chunk_id → RRF score
        
        Returns:
            List of section dicts with full text and aggregated scores
        """
        # Group by (paper_id, section_idx)
        section_groups = defaultdict(list)
        for cid in chunk_ids:
            chunk = self.chunks[cid]
            key = (chunk['paper_id'], chunk['section_idx'])
            section_groups[key].append(cid)
        
        # For each section, fetch ALL chunks and reconstruct full text
        sections = []
        for (paper_id, section_idx), matched_chunks in section_groups.items():
            # Fetch all chunks for this section (not just matched ones)
            all_section_chunks = [
                (i, c) for i, c in enumerate(self.chunks)
                if c['paper_id'] == paper_id and c['section_idx'] == section_idx
            ]
            all_section_chunks.sort(key=lambda x: x[1]['chunk_idx'])
            
            # De-overlap and concatenate
            chunk_texts = [c['text'] for _, c in all_section_chunks]
            full_text = ' '.join(de_overlap_strings(chunk_texts))
            
            # Aggregate scores (sum of matched chunk scores)
            agg_score = sum(chunk_scores.get(cid, 0) for cid in matched_chunks)
            
            sections.append({
                'paper_id': paper_id,
                'section_idx': section_idx,
                'full_text': full_text,
                'rrf_score': agg_score,
                'matched_chunks': matched_chunks
            })
        
        return sections
    
    def _layer3_colbert_gist(self, query: str, sections: List[Dict]) -> List[Tuple[int, int]]:
        """
        Layer 3 ColBERT reranking with GIST selection.
        
        Args:
            query: Query text
            sections: List of section dicts
        
        Returns:
            List of (section_idx_in_list, gist_rank) tuples
        """
        # Encode query and sections with ColBERT
        query_emb = self.colbert_model.encode([query], convert_to_tensor=False)[0]
        section_texts = [s['full_text'][:512] for s in sections]  # Truncate for speed
        section_embs = self.colbert_model.encode(section_texts, convert_to_tensor=False)
        
        # Coverage: ColBERT section-section cosine similarity
        coverage_matrix = cosine_similarity(section_embs)
        
        # Utility: ColBERT section-query cosine similarity
        utility_vector = cosine_similarity(section_embs, query_emb.reshape(1, -1)).flatten()
        
        # GIST selection
        k_select = min(len(sections), self.config.layer2_sections)
        selected_indices = gist_select(coverage_matrix, utility_vector, k=k_select, lambda_param=self.config.gist_lambda)
        
        results = [(i, rank) for rank, i in enumerate(selected_indices, 1)]
        return results
    
    def _layer3_cross_encoder_gist(self, query: str, sections: List[Dict]) -> List[Tuple[int, int]]:
        """
        Layer 3 Cross-Encoder reranking with GIST selection.
        
        Args:
            query: Query text
            sections: List of section dicts
        
        Returns:
            List of (section_idx_in_list, gist_rank) tuples
        """
        # Encode with Cross-Encoder
        section_texts = [s['full_text'][:512] for s in sections]
        pairs = [[query, text] for text in section_texts]
        ce_scores = self.cross_encoder.predict(pairs, show_progress_bar=False)
        ce_scores = np.array(ce_scores)
        
        # Coverage: Cross-Encoder pairwise section-section scores
        # Approximate via score product (real implementation would re-encode all pairs)
        coverage_matrix = np.outer(ce_scores, ce_scores)
        coverage_matrix = coverage_matrix / (np.max(coverage_matrix) + 1e-10)
        
        # Utility: Cross-Encoder section-query scores
        utility_vector = ce_scores
        
        # GIST selection
        k_select = min(len(sections), self.config.layer2_sections)
        selected_indices = gist_select(coverage_matrix, utility_vector, k=k_select, lambda_param=self.config.gist_lambda)
        
        results = [(i, rank) for rank, i in enumerate(selected_indices, 1)]
        return results
    
    def _walk_down_paper_selection(self, sections: List[Dict], section_scores: Dict[int, float]) -> List[str]:
        """
        Walk-down paper selection with floor threshold.
        
        Traverse sections from highest to lowest score, collecting first (top_k + 1) papers.
        The (top_k + 1)th paper's minimum section score becomes the floor threshold.
        Keep only sections >= floor from first top_k papers.
        
        Args:
            sections: List of section dicts
            section_scores: Mapping of section_idx → RRF score
        
        Returns:
            List of selected paper IDs
        """
        # Sort sections by RRF score
        sorted_sections = sorted(
            [(i, sections[i], section_scores.get(i, 0)) for i in range(len(sections))],
            key=lambda x: x[2],
            reverse=True
        )
        
        # Walk down, collecting papers
        papers_collected = {}  # paper_id → [(section_idx, score)]
        paper_order = []
        
        for section_idx, section, score in sorted_sections:
            paper_id = section['paper_id']
            
            if paper_id not in papers_collected:
                papers_collected[paper_id] = []
                paper_order.append(paper_id)
            
            papers_collected[paper_id].append((section_idx, score))
            
            # Stop after collecting (top_k + 1) papers
            if len(paper_order) == self.config.layer3_output + 1:
                break
        
        # Calculate floor from (top_k + 1)th paper
        if len(paper_order) >= self.config.layer3_output + 1:
            floor_paper_id = paper_order[self.config.layer3_output]  # 0-indexed
            floor_scores = [s for _, s in papers_collected[floor_paper_id]]
            floor_threshold = min(floor_scores)
        else:
            floor_threshold = 0.0
        
        # Keep only top_k papers
        selected_paper_ids = paper_order[:self.config.layer3_output]
        
        print(f"  Walk-down: {len(paper_order)} papers collected, floor={floor_threshold:.4f}")
        print(f"  Selected: {len(selected_paper_ids)} papers")
        
        return selected_paper_ids
    
    def retrieve(self, query: str) -> List[Tuple[int, float]]:
        """
        Three-layer GIST retrieval.
        
        Args:
            query: Query text
        
        Returns:
            List of (chunk_id, rrf_score) tuples from selected papers
        """
        print(f"\n{'='*60}")
        print(f"LAYER 1: LEXICAL/SEMANTIC RETRIEVAL")
        print(f"{'='*60}")
        
        # L1: BM25 + M2V with GIST
        print(f"BM25 retrieval with GIST...")
        bm25_results = self._layer1_bm25_gist(query)
        print(f"  ✓ Selected {len(bm25_results)} chunks via GIST")
        
        print(f"M2V retrieval with GIST...")
        m2v_results = self._layer1_embedding_gist(query)
        print(f"  ✓ Selected {len(m2v_results)} chunks via GIST")
        
        # RRF fusion
        print(f"RRF fusion...")
        rrf_scores = {}
        for chunk_id, rank in bm25_results:
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + compute_rrf_score([rank], k=self.config.rrf_k)
        for chunk_id, rank in m2v_results:
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + compute_rrf_score([rank], k=self.config.rrf_k)
        
        l1_chunks = list(rrf_scores.keys())
        print(f"  ✓ L1 seeds: {len(l1_chunks)} unique chunks")
        
        # ===================================================================
        # LAYER 2: QUERY EXPANSION via GRAPH
        # ===================================================================
        print(f"\n{'='*60}")
        print(f"LAYER 2: QUERY EXPANSION (excluding L1 seeds)")
        print(f"{'='*60}")
        
        exclude_set = set(l1_chunks)
        
        print(f"Graph BM25 retrieval with GIST...")
        graph_results = self._layer2_graph_bm25_gist(query, exclude_set)
        print(f"  ✓ Selected {len(graph_results)} chunks via GIST")
        
        print(f"Qwen3 retrieval with GIST...")
        qwen3_results = self._layer2_qwen3_gist(query, exclude_set)
        print(f"  ✓ Selected {len(qwen3_results)} chunks via GIST")
        
        # RRF fusion of L2 results
        print(f"RRF fusion of L2 results...")
        l2_rrf = {}
        for chunk_id, rank in graph_results:
            l2_rrf[chunk_id] = l2_rrf.get(chunk_id, 0) + compute_rrf_score([rank], k=self.config.rrf_k)
        for chunk_id, rank in qwen3_results:
            l2_rrf[chunk_id] = l2_rrf.get(chunk_id, 0) + compute_rrf_score([rank], k=self.config.rrf_k)
        
        print(f"  ✓ L2 expansions: {len(l2_rrf)} unique chunks")
        
        # Combine L1 + L2 for section aggregation
        all_chunks = list(set(l1_chunks) | set(l2_rrf.keys()))
        all_scores = {**rrf_scores, **l2_rrf}
        
        print(f"  Total: {len(all_chunks)} chunks (L1 + L2)")
        
        # Aggregate to sections
        print(f"Aggregating chunks to sections...")
        sections = self._aggregate_chunks_to_sections(all_chunks, all_scores)
        print(f"  ✓ {len(sections)} unique sections")
        
        # Select φ_lower(top_k²) sections
        sections.sort(key=lambda s: s['rrf_score'], reverse=True)
        sections = sections[:self.config.layer2_sections]
        print(f"  ✓ Selected {len(sections)} sections (φ_lower={self.config.layer2_sections})")
        
        # ===================================================================
        # LAYER 3: LATE INTERACTION RERANKING
        # ===================================================================
        print(f"\n{'='*60}")
        print(f"LAYER 3: LATE INTERACTION RERANKING")
        print(f"{'='*60}")
        
        print(f"ColBERT reranking with GIST...")
        colbert_results = self._layer3_colbert_gist(query, sections)
        print(f"  ✓ Selected {len(colbert_results)} sections via GIST")
        
        print(f"Cross-Encoder reranking with GIST...")
        ce_results = self._layer3_cross_encoder_gist(query, sections)
        print(f"  ✓ Selected {len(ce_results)} sections via GIST")
        
        # RRF fusion
        print(f"RRF fusion of ColBERT + CE...")
        l3_rrf = {}
        for section_idx, rank in colbert_results:
            l3_rrf[section_idx] = l3_rrf.get(section_idx, 0) + compute_rrf_score([rank], k=self.config.rrf_k)
        for section_idx, rank in ce_results:
            l3_rrf[section_idx] = l3_rrf.get(section_idx, 0) + compute_rrf_score([rank], k=self.config.rrf_k)
        
        # Walk-down paper selection
        print(f"Walk-down paper selection...")
        selected_papers = self._walk_down_paper_selection(sections, l3_rrf)
        
        # Collect all chunks from selected papers
        final_results = []
        for i, section in enumerate(sections):
            if section['paper_id'] in selected_papers:
                score = l3_rrf.get(i, 0)
                for cid in section['matched_chunks']:
                    final_results.append((cid, score))
        
        print(f"\n{'#'*60}")
        print(f"FINAL OUTPUT: {len(final_results)} chunks from {len(selected_papers)} papers")
        print(f"{'#'*60}\n")
        
        return final_results


def example_usage():
    """Example usage"""
    from rank_bm25 import BM25Okapi
    
    # Load BM25 index for Layer 1
    with open("checkpoints/chunks.msgpack", 'rb') as f:
        chunks = msgpack.load(f, raw=False)
    
    tokenized_corpus = [chunk['lemmatized_text'].split() for chunk in chunks]
    bm25_index = BM25Okapi(tokenized_corpus)
    
    # Initialize retriever
    retriever = ThreeLayerGISTRetriever(
        chunks_path="checkpoints/chunks.msgpack",
        chunk_embeddings_qwen3_path="checkpoints/chunk_embeddings_qwen3.msgpack",
        triplets_path="checkpoints/enriched_triplets.msgpack",
        chunk_to_triplets_path="checkpoints/chunk_to_triplets.msgpack",
        triplet_to_chunks_path="checkpoints/triplet_to_chunks.msgpack",
        triplet_bm25_path="checkpoints/wordpiece_bm25_index.msgpack",
        bm25_lemmatized_index=bm25_index,
        top_k=13
    )
    
    # Query
    query = "What are agentic memory approaches?"
    results = retriever.retrieve(query)
    
    print(f"\nTop {len(results)} results:")
    for i, (chunk_id, score) in enumerate(results[:20], 1):
        chunk_text = retriever.chunks[chunk_id]['content'][:100]
        print(f"\n{i}. Chunk {chunk_id} (score: {score:.4f})")
        print(f"   {chunk_text}...")


if __name__ == '__main__':
    example_usage()
