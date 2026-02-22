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
from gist_retriever import gist_select


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

        # GIST per arm: each arm selects hybrid_seeds diverse docs independently.
        # After RRF union, len(hybrid_pool) >= hybrid_seeds (arms may partially
        # overlap), so slicing [:hybrid_seeds] always yields exactly hybrid_seeds.
        bm25_selected  = self._gist_select_pool(bm25_pool,  query, hybrid_seeds, 'bm25')
        dense_selected = self._gist_select_pool(gist_pool,  query, hybrid_seeds, 'dense')
        print(f"     GIST(BM25)  \u2192 {len(bm25_selected):4d}  GIST(Dense) \u2192 {len(dense_selected):4d}")

        # RRF Fusion → exactly hybrid_seeds seeds
        hybrid_pool = self._rrf_fusion(bm25_selected, dense_selected)
        hybrid_seeds_pool = hybrid_pool[:hybrid_seeds]
        print(f"     RRF(L1-GIST)                     \u2192 {len(hybrid_seeds_pool):4d} hybrid seeds")
        print(_D)

        # =================================================================
        # Layer 2: ECDF-Weighted Dual Expansion from Hybrid Seeds
        #   Each arm independently GIST-selects 2×hybrid_seeds NEW chunks
        #   (seeds excluded). Two arms × 2×seeds = 4×seeds total before RRF.
        #   RRF(BM25-arm, Dense-arm) deduplicates → top hybrid_seeds unique
        #   section_idx selected by _reconstruct_documents_from_chunks.
        # =================================================================
        l2_arm_size = hybrid_seeds * 2   # 288 for top_k=13
        print(f"[L2 Expansion]  seeds={hybrid_seeds}  arm={l2_arm_size}  target={l2_arm_size * 2} combined  (excludes seeds)")
        seed_scores = [doc.rrf_score for doc in hybrid_seeds_pool]
        l2_bm25  = self._expand_layer2_bm25(hybrid_seeds_pool, seed_scores, l2_arm_size)
        l2_dense = self._expand_layer2_dense(hybrid_seeds_pool, seed_scores, l2_arm_size)
        print(f"  \u251c\u2500 BM25   layer2_triplet_bm25      \u2192 {len(l2_bm25):4d} new chunks")
        print(f"  \u2514\u2500 Dense  GIST centroid (256d)      \u2192 {len(l2_dense):4d} new chunks")

        graph_expanded = self._rrf_fusion(l2_bm25, l2_dense)
        print(f"     RRF(L2-BM25, L2-Dense)           \u2192 {len(graph_expanded):4d} merged")

        fused_chunks = self._rrf_fusion(hybrid_seeds_pool, graph_expanded)
        print(f"     RRF(seeds + L2)                  \u2192 {len(fused_chunks):4d} fused pool")
        print(_D)

        # =================================================================
        # Layer 3: Reconstruct → Score (ColBERT + CE) → Select
        # =================================================================
        print("[L3 Scoring]")
        # Section expansion: rank fused chunks by score, take top hybrid_seeds
        # unique (paper_id, section_idx) keys, reconstruct full section text.
        documents = self._reconstruct_documents_from_chunks(fused_chunks, hybrid_seeds)
        print(f"  \u251c\u2500 Reconstruct                     \u2192 {len(documents):4d} sections")

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

        # Build section-level cosine coverage matrix for GIST diversity
        n_docs          = len(documents)
        doc_texts_trunc = [d['text'][:512] for d in documents]
        sec_embs        = self.embedder.encode(doc_texts_trunc)   # (n, dim), L2-normalised
        coverage_matrix = sec_embs @ sec_embs.T                   # cosine similarity
        gist_lam        = getattr(self.config, 'gist_lambda', 0.7)
        rrf_k           = 60

        if ce_scores is not None:
            # GIST per arm → RRF via GIST rank
            colbert_gist = gist_select(coverage_matrix, colbert_scores, n_docs, gist_lam)
            ce_gist      = gist_select(coverage_matrix, ce_scores,      n_docs, gist_lam)

            colbert_rank = {int(idx): rank + 1 for rank, idx in enumerate(colbert_gist)}
            ce_rank      = {int(idx): rank + 1 for rank, idx in enumerate(ce_gist)}

            for i, doc in enumerate(documents):
                rrf = (1.0 / (rrf_k + colbert_rank[i])
                       + 1.0 / (rrf_k + ce_rank[i]))
                doc['score']                = rrf
                doc['colbert_score']        = float(colbert_scores[i])
                doc['cross_encoder_score']  = float(ce_scores[i])

            print(f"  \u251c\u2500 ColBERT   GIST late interaction")
            print(f"  \u2514\u2500 MS-MARCO  GIST + RRF merged")
        else:
            # CE unavailable — ColBERT GIST only
            colbert_gist = gist_select(coverage_matrix, colbert_scores, n_docs, gist_lam)
            colbert_rank = {int(idx): rank + 1 for rank, idx in enumerate(colbert_gist)}
            for i, doc in enumerate(documents):
                doc['score']         = 1.0 / (rrf_k + colbert_rank[i])
                doc['colbert_score'] = float(colbert_scores[i])
            print(f"  \u2514\u2500 ColBERT   GIST late interaction   (CE unavailable)")

        documents.sort(key=lambda d: d['score'], reverse=True)
        return documents
    
    # =================================================================
    # ABSTRACT METHODS (Must be implemented by subclasses)
    # =================================================================
    
    def _reconstruct_documents_from_chunks(
        self,
        chunks: List[RetrievedDoc],
        target_sections: int,
    ) -> List[Dict[str, Any]]:
        """
        Reconstruct full sections from the top-target_sections unique section keys
        in the fused chunk pool.

        Spec (Feature 17):
          Sort chunks by score desc (tiebreak: chunk_idx, section_idx, doc_id).
          Walk sorted list, collect unique (paper_id, section_idx) until
          target_sections reached.  Then fetch ALL chunks per section to rebuild
          full section text.

        Dataset-specific aggregation:
          - Arxiv: section key = (paper_id, section_idx)
          - Quotes: section key = (quote_id,)

        Args:
            chunks: Fused chunk pool sorted by RRF score
            target_sections: Max unique sections to reconstruct (= hybrid_seeds)

        Returns:
            List of section dicts with reconstructed text

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
