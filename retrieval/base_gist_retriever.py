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
        
        print(f"Pipeline: BM25+GIST {retrieval_limit} -> RRF {hybrid_seeds} hybrid seeds -> L2 triplet-BM25+dense (target {hybrid_seeds} new each = {hybrid_seeds*2} total pool) -> RRF fusion -> Reconstruct documents -> Late Interaction -> Select {top_k} final")
        
        # =================================================================
        # Layer 1: BM25 Retrieval (Lexical)
        # =================================================================
        print(f"  [1/7] BM25 retrieval (lexical matching)...")
        bm25_pool = self._retrieve_bm25(query, retrieval_limit)
        print(f"        Retrieved {len(bm25_pool)} chunks")
        
        # =================================================================
        # Layer 2: GIST Retrieval (Semantic)
        # =================================================================
        print(f"  [2/7] GIST retrieval (semantic similarity)...")
        gist_pool = self._retrieve_dense(query, retrieval_limit)
        print(f"        Retrieved {len(gist_pool)} chunks")
        
        # =================================================================
        # Layer 3: RRF Fusion → Hybrid Seeds (Fibonacci cascade)
        # =================================================================
        print(f"  [3/7] RRF fusion (BM25 + GIST) -> {hybrid_seeds} hybrid seeds...")
        hybrid_pool = self._rrf_fusion(bm25_pool, gist_pool)
        hybrid_seeds_pool = hybrid_pool[:hybrid_seeds]  # Take top Fibonacci seeds
        print(f"        Hybrid seeds: {len(hybrid_seeds_pool)} chunks")
        
        # =================================================================
        # Layer 4: L2 ECDF-Weighted Dual Expansion from Hybrid Seeds
        # Expansion target = hybrid_seeds (not top_k):
        #   hybrid_seeds = prev_fib(top_k²) = 144 for top_k=13
        #   Each path queries hybrid_seeds*2 from DB, excludes seeds,
        #   returns up to hybrid_seeds NEW chunks → total pool: 2×144 = 288
        # =================================================================
        print(f"  [4/7] L2 expansion from hybrid seeds (BM25 triplet + Dense centroid)...")
        seed_scores = [doc.rrf_score for doc in hybrid_seeds_pool]
        l2_bm25  = self._expand_layer2_bm25(hybrid_seeds_pool, seed_scores, hybrid_seeds)
        l2_dense = self._expand_layer2_dense(hybrid_seeds_pool, seed_scores, hybrid_seeds)
        print(f"        L2 BM25: {len(l2_bm25)} | L2 Dense: {len(l2_dense)}")

        # RRF merge the two L2 paths → combined expansion pool
        graph_expanded = self._rrf_fusion(l2_bm25, l2_dense)
        print(f"        L2 RRF merged: {len(graph_expanded)} chunks")

        # =================================================================
        # Layer 5: RRF Fusion (Hybrid Seeds + L2 Expansion)
        # =================================================================
        print(f"  [5/7] RRF fusion (Hybrid seeds + L2 expansion)...")
        fused_chunks = self._rrf_fusion(hybrid_seeds_pool, graph_expanded)
        # Keep more chunks after graph expansion (don't limit to k²)
        print(f"        Fused pool: {len(fused_chunks)} chunks")
        
        # =================================================================
        # Layer 6: Document Reconstruction (Dataset-Specific)
        # =================================================================
        print(f"  [6/7] Reconstructing documents from chunks...")
        documents = self._reconstruct_documents_from_chunks(fused_chunks)
        print(f"        Reconstructed {len(documents)} unique documents")
        
        # =================================================================
        # Layer 7: Late Interaction on Documents (ColBERT)
        # =================================================================
        print(f"  [7/7] Late Interaction on documents (ColBERT scoring)...")
        if self.config.use_colbert and len(documents) > 0:
            # Score ALL documents - let final selection choose top_k
            scored_documents = self._score_documents(query, documents, len(documents))
        else:
            # Fallback: keep documents as-is (already have base scores from chunks)
            scored_documents = documents
        print(f"        Scored {len(scored_documents)} documents")
        
        # =================================================================
        # Layer 8: Final Document Selection (Dataset-Specific)
        # =================================================================
        print(f"  [8/8] Selecting {top_k} final documents...")
        results = self._select_final_documents(scored_documents, top_k)
        print(f"        Selected {len(results)} documents")
        
        print(f"  Returning {len(results)} documents")
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
        Score documents using ColBERT late interaction (token-level matching).
        
        Args:
            query: Search query
            documents: List of document dicts with 'text' field
            top_k: Number of top documents to return
        
        Returns:
            Sorted list of documents with scores
        """
        if not self.config.use_colbert or not documents:
            return documents[:top_k]
        
        # Extract document texts
        doc_texts = [d['text'] for d in documents]
        
        # Score using ColBERT (token-level matching)
        scores = self.colbert_scorer.score(query, doc_texts)
        
        # Assign scores
        for doc, score in zip(documents, scores):
            doc['score'] = float(score)
        
        # Sort by score (descending)
        documents.sort(key=lambda d: d['score'], reverse=True)
        
        return documents[:top_k]
    
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
