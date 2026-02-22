"""
Base GIST Retriever: GraphRAG Pipeline (Dataset-Agnostic)

=============================================================================
ARCHITECTURE
=============================================================================

GraphRAG Pipeline (7 layers):
  [1] Hybrid Retrieval: BM25 + GIST (lexical + semantic) → top_k² each
  [2] RRF Fusion → Hybrid Seeds: prev_fib(top_k²) seeds  (e.g. 169 → 144)
  [3] L2 Expansion: ECDF-weighted BM25 triplets + Dense centroid
        each path queries hybrid_seeds×2 from DB, excludes seeds,
        returns up to hybrid_seeds NEW chunks  (2×144 = 288 expansion target)
  [4] RRF Fusion (Hybrid Seeds + L2 expansion → fused pool: 2×hybrid_seeds)
  [5] Section/Quote Reconstruction (dataset-specific abstract method)
  [6] Late Interaction on reconstructed documents (ColBERT scoring)
  [7] Document Selection (dataset-specific abstract method)

This base class provides layers 1-4 and 6.
Subclasses implement layers 5 and 7 for their specific aggregation hierarchy.

=============================================================================
SUBCLASS RESPONSIBILITIES
=============================================================================

Dataset-specific subclasses must implement:

1. _reconstruct_documents_from_chunks(chunks) -> List[Dict]
   - Arxiv: chunk → section (group by paper_id, section_idx)
   - Quotes: chunk → quote (group by quote_id)

2. _select_final_documents(scored_docs, top_k) -> List[RetrievedDoc]
   - Arxiv: section → paper (iterate until K unique papers)
   - Quotes: scored quotes → top K quotes

=============================================================================
"""

from typing import List, Dict, Any
import numpy as np
from pgvector_retriever import PGVectorRetriever, RetrievedDoc


class BaseGISTRetriever(PGVectorRetriever):
    """
    Base class for GraphRAG pipeline.
    
    Implements dataset-agnostic layers:
      - Hybrid retrieval (BM25 + GIST)
      - L2 expansion via pgvector triplet sparsevecs + dense centroid
      - RRF fusion
      - Late interaction scoring
    
    Subclasses implement:
      - Document reconstruction (chunk → section/quote)
      - Final document selection (aggregation logic)
    """
    
    def search(
        self,
        query: str,
        top_k: int = 13,
        return_chunks: bool = False
    ) -> List[RetrievedDoc]:
        """
        GraphRAG search pipeline.
        
        Args:
            query: Search query
            top_k: Number of final documents to return
            return_chunks: Whether to include chunks in results
        
        Returns:
            List of RetrievedDoc (papers or quotes depending on subclass)
        """
        # Calculate parameters
        retrieval_limit = top_k ** 2  # k² = 169 for k=13
        
        # Fibonacci cascade: use previous Fibonacci number
        def get_previous_fibonacci(n):
            """Get the largest Fibonacci number less than n."""
            fib = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597]
            for f in reversed(fib):
                if f < n:
                    return f
            return 1
        
        hybrid_seeds = get_previous_fibonacci(retrieval_limit)  # 169 -> 144
        
        _D = "\u2500" * 66
        print(f"Pipeline  top_k={top_k}  pool={retrieval_limit}  seeds={hybrid_seeds}")
        print(_D)

        # =================================================================
        # Layer 1: BM25 + GIST Retrieval (parallel arms)
        # =================================================================
        print("[L1 Retrieval]")
        bm25_pool = self._retrieve_bm25(query, retrieval_limit)
        print(f"  \u251c\u2500 BM25   layer1_bm25_sparse       \u2192 {len(bm25_pool):4d} chunks")

        gist_pool = self._retrieve_dense(query, retrieval_limit)
        print(f"  \u2514\u2500 Dense  layer1_embeddings_128d   \u2192 {len(gist_pool):4d} chunks")

        # RRF Fusion → Hybrid Seeds (Fibonacci cascade)
        hybrid_pool = self._rrf_fusion(bm25_pool, gist_pool)
        hybrid_seeds_pool = hybrid_pool[:hybrid_seeds]  # Take top Fibonacci seeds
        print(f"     RRF(L1)                          \u2192 {len(hybrid_seeds_pool):4d} hybrid seeds")
        print(_D)

        # =================================================================
        # Layer 2: ECDF-Weighted Dual Expansion from Hybrid Seeds
        #   Each arm queries hybrid_seeds*2, excludes seeds,
        #   returns up to hybrid_seeds NEW chunks → pool = 2×hybrid_seeds
        # =================================================================
        print(f"[L2 Expansion]  seeds={hybrid_seeds}  target={hybrid_seeds * 2} new  (excludes seeds)")
        seed_scores = [doc.rrf_score for doc in hybrid_seeds_pool]
        l2_bm25  = self._expand_layer2_bm25(hybrid_seeds_pool, seed_scores, hybrid_seeds)
        l2_dense = self._expand_layer2_dense(hybrid_seeds_pool, seed_scores, hybrid_seeds)
        print(f"  \u251c\u2500 BM25   layer2_triplet_bm25      \u2192 {len(l2_bm25):4d} new chunks")
        print(f"  \u2514\u2500 Dense  GIST centroid (128d)     \u2192 {len(l2_dense):4d} new chunks")

        graph_expanded = self._rrf_fusion(l2_bm25, l2_dense)
        print(f"     RRF(L2-BM25, L2-Dense)           \u2192 {len(graph_expanded):4d} merged")

        fused_chunks = self._rrf_fusion(hybrid_seeds_pool, graph_expanded)
        print(f"     RRF(seeds + L2)                  \u2192 {len(fused_chunks):4d} fused pool")
        print(_D)

        # =================================================================
        # Layer 3: Reconstruct → Score (ColBERT + CE) → Select
        # =================================================================
        print("[L3 Scoring]")
        documents = self._reconstruct_documents_from_chunks(fused_chunks)
        print(f"  \u251c\u2500 Reconstruct                     \u2192 {len(documents):4d} documents")

        if self.config.use_colbert and len(documents) > 0:
            scored_documents = self._score_documents(query, documents, len(documents))
        else:
            scored_documents = documents
        print(f"     Scored {len(scored_documents):4d} documents")

        results = self._select_final_documents(scored_documents, top_k)
        print(f"     Select top-{top_k:<3d}                    \u2192 {len(results):4d} final")
        print(_D)
        return results
    
    def _rrf_fusion(
        self,
        pool_a: List[RetrievedDoc],
        pool_b: List[RetrievedDoc],
        k: int = 60
    ) -> List[RetrievedDoc]:
        """
        Fuse two pools using Reciprocal Rank Fusion.
        
        Args:
            pool_a: First pool (e.g., BM25 or hybrid seeds)
            pool_b: Second pool (e.g., GIST or graph expansion)
            k: RRF constant (default 60)
        
        Returns:
            Fused list sorted by RRF score
        """
        pool_a_by_id = {doc.doc_id: doc for doc in pool_a}
        pool_b_by_id = {doc.doc_id: doc for doc in pool_b}
        
        all_ids = set(pool_a_by_id.keys()) | set(pool_b_by_id.keys())
        
        fused = []
        for doc_id in all_ids:
            # Get ranks (1-based)
            rank_a = next((i + 1 for i, d in enumerate(pool_a) if d.doc_id == doc_id), None)
            rank_b = next((i + 1 for i, d in enumerate(pool_b) if d.doc_id == doc_id), None)
            
            # RRF score
            score = 0.0
            if rank_a is not None:
                score += 1.0 / (k + rank_a)
            if rank_b is not None:
                score += 1.0 / (k + rank_b)
            
            # Use document from pool_a if available, else pool_b
            doc = pool_a_by_id.get(doc_id) or pool_b_by_id.get(doc_id)
            doc.rrf_score = score
            doc.final_score = score
            fused.append(doc)
        
        # Sort by RRF score (descending)
        fused.sort(key=lambda d: d.rrf_score, reverse=True)
        return fused
    
    def _score_documents(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Score documents using ColBERT + MS-MARCO cross-encoder with RRF fusion.

        L3 scoring pipeline:
          1. ColBERTv2 token-level matching → full ranking
          2. cross-encoder/ms-marco-MiniLM-L-6-v2 → full ranking
          3. RRF merge (k=60) of both rankings
          4. Sort by RRF score (_select_final_documents handles top_k cutoff)

        Args:
            query: Search query
            documents: List of document dicts with 'text' field
            top_k: Unused here; _select_final_documents handles cutoff

        Returns:
            All documents sorted by RRF score (no top_k cap)
        """
        if not documents:
            return documents

        doc_texts = [d['text'] for d in documents]

        # --- ColBERT scores ---
        if self.config.use_colbert:
            colbert_scores = self.colbert_scorer.score(query, doc_texts)
        else:
            colbert_scores = np.zeros(len(documents))

        # --- Cross-encoder scores (ms-marco) ---
        ce_scorer = self.cross_encoder_scorer if self.config.use_cross_encoder else None
        if ce_scorer is not None and ce_scorer.available:
            ce_scores = ce_scorer.score(query, doc_texts)
        else:
            ce_scores = None

        if ce_scores is not None:
            # RRF merge ColBERT + CE rankings (k=60)
            rrf_k = 60
            colbert_order = np.argsort(colbert_scores)[::-1]  # highest-first
            ce_order      = np.argsort(ce_scores)[::-1]

            colbert_rank = {int(idx): rank + 1 for rank, idx in enumerate(colbert_order)}
            ce_rank      = {int(idx): rank + 1 for rank, idx in enumerate(ce_order)}

            for i, doc in enumerate(documents):
                rrf = (1.0 / (rrf_k + colbert_rank[i])
                       + 1.0 / (rrf_k + ce_rank[i]))
                doc['score']                = rrf
                doc['colbert_score']        = float(colbert_scores[i])
                doc['cross_encoder_score']  = float(ce_scores[i])

            print(f"  \u251c\u2500 ColBERT   late interaction")
            print(f"  \u2514\u2500 MS-MARCO CE + RRF merged")
        else:
            # CE unavailable — ColBERT only
            for doc, score in zip(documents, colbert_scores):
                doc['score'] = float(score)
            print(f"  \u2514\u2500 ColBERT   late interaction        (CE unavailable)")

        documents.sort(key=lambda d: d['score'], reverse=True)
        return documents
    
    # =================================================================
    # ABSTRACT METHODS (Must be implemented by subclasses)
    # =================================================================
    
    def _reconstruct_documents_from_chunks(
        self,
        chunks: List[RetrievedDoc]
    ) -> List[Dict[str, Any]]:
        """
        Reconstruct full documents from retrieved chunks.
        
        Dataset-specific aggregation:
          - Arxiv: Group by (paper_id, section_idx) → fetch all chunks → rebuild section
          - Quotes: Group by quote_id → fetch all chunks → rebuild quote
        
        Args:
            chunks: List of retrieved chunk documents
        
        Returns:
            List of document dicts with reconstructed text
        
        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError("Subclass must implement _reconstruct_documents_from_chunks()")
    
    def _select_final_documents(
        self,
        scored_documents: List[Dict[str, Any]],
        top_k: int
    ) -> List[RetrievedDoc]:
        """
        Select final top-K documents with dataset-specific logic.
        
        Dataset-specific selection:
          - Arxiv: Iterate scored sections until K unique papers
          - Quotes: Take top K scored quotes directly
        
        Args:
            scored_documents: Documents sorted by relevance score
            top_k: Number of documents to select
        
        Returns:
            List of RetrievedDoc in final format
        
        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError("Subclass must implement _select_final_documents()")
