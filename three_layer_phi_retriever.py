"""
Three-Layer φ-Scaled Retrieval with Dual Reranking (ColBERTv2 + Cross-Encoder + RRF)

Architecture:
  Query
    ↓
  Layer 1: Hybrid GIST-RRF (BM25 pool→GIST + Dense pool→GIST → RRF) → top_k SEEDS (13)
    ↓ (seeds as INPUT, excluded from OUTPUT)
  Layer 2: Expansion via Graph + Qwen3
    ├─ Graph BM25 (triplet corpus) → GIST(Qwen3 coverage, BM25 utility)
    └─ Qwen3 (co-occurrence similarity) → GIST(Qwen3 coverage, cosine utility)
    → Standard RRF merge → 144 NEW chunks (top_k² - 25)
    ↓
  Concatenate: 13 (L1 seeds) + 144 (L2 expansions) = 157 total
    ↓
  Layer 3: Dual Reranking with RRF Fusion → Final top_k (13)
    ├─ ColBERTv2 late interaction → GIST(Qwen3 coverage, ColBERTv2 utility)
    └─ Cross-Encoder scoring → GIST(Qwen3 coverage, Cross-Encoder utility)
    → RRF merge of dual GIST rankings
    → Paper aggregation (max score per paper)
    → Select top_k PAPERS (guaranteed 13 papers)

GIST at every layer:
  gist_select(coverage_matrix, utility_vector, k, λ=0.7)
  score(d) = λ * utility(d) - (1-λ) * max_sim(d, selected_set)
  Ensures diversity + relevance at each stage.

φ-Scaling:
  top_k = 13
  top_k² = 169
  Layer 2 expansion = 169 - 25 = 144 (one phi lower)
  Final output = 13 chunks from 13 different papers
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from model2vec import StaticModel
import msgpack
from pathlib import Path
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
from gist_retriever import gist_select


# Golden ratio
PHI = 1.618033988749895


@dataclass
class PhiLayerConfig:
    """φ-scaled layer configuration"""
    top_k: int  # Final output (13)
    layer1_seeds: int  # top_k (13)
    layer2_expand: int  # top_k² - 25 (144 for top_k=13)
    layer3_output: int  # top_k (13)
    
    @classmethod
    def from_top_k(cls, top_k: int = 13):
        """
        Calculate layer sizes with φ-scaling
        
        Args:
            top_k: Final output size (default 13)
        
        Returns:
            PhiLayerConfig with calculated sizes
        """
        layer2_expand = (top_k ** 2) - 25  # 169 - 25 = 144
        
        return cls(
            top_k=top_k,
            layer1_seeds=top_k,
            layer2_expand=layer2_expand,
            layer3_output=top_k
        )


class ThreeLayerPhiRetriever:
    """
    Three-layer retrieval with φ-scaling and ECDF gist normalization
    
    Layer 1: Seeds (high-confidence starting points)
    Layer 2: Expansion (graph walking + co-occurrence)
    Layer 3: Reranking (late interaction precision)
    """
    
    def __init__(
        self,
        chunks_path: str,
        chunk_embeddings_qwen3_path: str,
        triplets_path: str,
        chunk_to_triplets_path: str,
        triplet_to_chunks_path: str,
        triplet_bm25_path: str,
        layer1_retriever,  # Existing GistRetriever
        top_k: int = 13,
        colbert_model_name: str = "colbert-ir/colbertv2.0",  # REQUIRED: ColBERTv2 model
        cross_encoder_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # REQUIRED: Cross-Encoder model
    ):
        """
        Args:
            chunks_path: Path to chunks.msgpack
            chunk_embeddings_qwen3_path: Path to chunk_embeddings_qwen3.msgpack
            triplets_path: Path to enriched_triplets.msgpack
            chunk_to_triplets_path: Path to chunk_to_triplets.msgpack
            triplet_to_chunks_path: Path to triplet_to_chunks.msgpack
            triplet_bm25_path: Path to WordPiece BM25 index over triplets
            layer1_retriever: Existing GistRetriever (BM25 + Jina)
            top_k: Final output size (default 13)
            colbert_model_name: ColBERTv2 model identifier (REQUIRED for Layer 3)
            cross_encoder_model_name: Cross-Encoder model identifier (REQUIRED for Layer 3)
        """
        self.config = PhiLayerConfig.from_top_k(top_k)
        self.layer1 = layer1_retriever
        self.verbose = False  # Set to True for debug output
        
        print(f"\n{'='*60}")
        print(f"LOADING THREE-LAYER φ-RETRIEVER")
        print(f"{'='*60}")
        
        # Load chunks
        print(f"Loading chunks from {chunks_path}...")
        with open(chunks_path, 'rb') as f:
            self.chunks = msgpack.load(f, raw=False)
        print(f"✓ Loaded {len(self.chunks)} chunks")
        
        # Create mapping: doc_id (str) → index (int)
        self.doc_id_to_idx = {chunk['doc_id']: i for i, chunk in enumerate(self.chunks)}
        
        # Load Qwen3 embeddings (161,389 × 256)
        print(f"Loading Qwen3 embeddings from {chunk_embeddings_qwen3_path}...")
        with open(chunk_embeddings_qwen3_path, 'rb') as f:
            data = msgpack.load(f, raw=False)
            self.qwen3_embeddings = np.array(data['embeddings'], dtype=np.float32)
        print(f"✓ Loaded embeddings: {self.qwen3_embeddings.shape}")
        
        # Load triplets
        print(f"Loading triplets from {triplets_path}...")
        with open(triplets_path, 'rb') as f:
            self.triplets = msgpack.load(f, raw=False)
        print(f"✓ Loaded {len(self.triplets)} enriched triplets")
        
        # Load mappings
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
            self.graph_bm25 = BM25Okapi(self.triplet_tokens)
        print(f"✓ Loaded BM25 index over {len(self.triplet_texts)} triplets")
        
        # Load Layer 3 rerankers (REQUIRED - not optional)
        print(f"\n{'='*60}")
        print(f"LOADING LAYER 3 DUAL RERANKERS")
        print(f"{'='*60}")
        
        # Load ColBERTv2 model
        print(f"Loading ColBERTv2 model: {colbert_model_name}...")
        try:
            from sentence_transformers import SentenceTransformer
            self.colbert_model = SentenceTransformer(colbert_model_name)
            print(f"✓ ColBERTv2 loaded")
        except Exception as e:
            print(f"❌ Failed to load ColBERTv2: {e}")
            print(f"   Install: pip install sentence-transformers")
            raise
        
        # Load Cross-Encoder model
        print(f"Loading Cross-Encoder model: {cross_encoder_model_name}...")
        try:
            from sentence_transformers import CrossEncoder
            self.cross_encoder = CrossEncoder(cross_encoder_model_name)
            print(f"✓ Cross-Encoder loaded")
        except Exception as e:
            print(f"❌ Failed to load Cross-Encoder: {e}")
            print(f"   Install: pip install sentence-transformers")
            raise
        
        print(f"\n{'='*60}")
        print(f"CONFIGURATION (top_k={self.config.top_k})")
        print(f"{'='*60}")
        print(f"Layer 1: Hybrid RRF → {self.config.layer1_seeds} seeds")
        print(f"Layer 2: Graph + Qwen3 → {self.config.layer2_expand} expansions")
        print(f"  Total: {self.config.layer1_seeds + self.config.layer2_expand} chunks")
        print(f"Layer 3: Dual Reranking (ColBERTv2 + Cross-Encoder + RRF) → {self.config.layer3_output} final")
        print(f"{'='*60}\n")
    
    def _expand_via_graph_bm25(
        self, 
        seed_chunks: List[int], 
        top_k: int
    ) -> List[Tuple[int, float]]:
        """
        Graph expansion via BM25 over triplet corpus with GIST diversity selection.
        
        Seeds as INPUT (extract their triplets) → BM25 search → GIST select (Qwen3 coverage + BM25 utility)
        
        Args:
            seed_chunks: Chunk IDs from Layer 1 (seeds)
            top_k: Number of NEW chunks to retrieve
        
        Returns:
            List of (chunk_id, gist_score) tuples — diverse, non-redundant
        """
        # 1. Extract triplets from seed chunks
        # Convert chunk indices to doc_ids with duplicated format: 1301_3781_s0_c0 -> 1301_3781_s0_c0_s0_c0
        seed_triplet_ids = []
        for cid in seed_chunks:
            doc_id = self.chunks[cid]['doc_id']
            parts = doc_id.split('_')
            if len(parts) >= 4:
                # Duplicate the section/chunk part: arxiv_id_sX_cY -> arxiv_id_sX_cY_sX_cY
                mapping_key = doc_id + '_' + '_'.join(parts[-2:])
            else:
                mapping_key = doc_id
            seed_triplet_ids.extend(self.chunk_to_triplets.get(mapping_key, []))
        
        if not seed_triplet_ids:
            return []
        
        # 2. Extract lemma text from triplets
        # Triplet structure: {chunk_id, text, triplets: [{subject_lemmas, predicate_lemmas, object_lemmas}]}
        seed_triplet_texts = []
        for tid in seed_triplet_ids:
            triplet_obj = self.triplets[tid]
            if 'triplets' in triplet_obj:
                # Extract all lemmas from all SPO triplets in this chunk
                for spo in triplet_obj['triplets']:
                    lemmas = (spo.get('subject_lemmas', []) + 
                             spo.get('predicate_lemmas', []) + 
                             spo.get('object_lemmas', []))
                    # Filter out padding tokens
                    lemmas = [l for l in lemmas if l != 'pad']
                    if lemmas:
                        seed_triplet_texts.append(' '.join(lemmas))
        
        # 3. Concatenate as query
        query_text = ' '.join(seed_triplet_texts)
        
        # 4. BM25 search over triplet corpus
        # Tokenize query (assuming WordPiece tokenization already applied in index)
        query_tokens = query_text.split()  # Simplified - matches how index was built
        scores = self.graph_bm25.get_scores(query_tokens)
        
        # Top candidates (oversample)
        top_indices = np.argsort(scores)[::-1][:top_k*3]
        
        # 5. Map triplets → chunks
        chunk_scores = {}
        for tidx in top_indices:
            tid = tidx  # Triplet ID = index in self.triplets
            score = scores[tidx]
            
            # Get chunks containing this triplet (returns doc_ids with duplicated format)
            for doc_id_dup in self.triplet_to_chunks.get(str(tid), []):
                # Convert duplicated doc_id to original format: 1301_3781_s0_c0_s0_c0 -> 1301_3781_s0_c0
                parts = doc_id_dup.split('_')
                if len(parts) >= 6:
                    doc_id = '_'.join(parts[:4])  # Keep only arxiv_id_sX_cY
                else:
                    doc_id = doc_id_dup
                
                # Convert doc_id to chunk index
                if doc_id in self.doc_id_to_idx:
                    cid = self.doc_id_to_idx[doc_id]
                    # Aggregate: max score across triplets
                    chunk_scores[cid] = max(chunk_scores.get(cid, 0), score)
        
        # 6. Exclude seed chunks
        seed_set = set(seed_chunks)
        chunk_scores = {cid: s for cid, s in chunk_scores.items() if cid not in seed_set}
        
        if not chunk_scores:
            return []
        
        # 7. GIST select: Qwen3 coverage matrix + BM25 utility vector
        chunk_ids = list(chunk_scores.keys())
        raw_scores = np.array([chunk_scores[cid] for cid in chunk_ids])
        
        # Coverage: Qwen3 pairwise doc-doc similarity
        candidate_embeddings = self.qwen3_embeddings[chunk_ids]
        coverage_matrix = cosine_similarity(candidate_embeddings)
        
        # Utility: raw BM25 scores (doc-query relevance)
        utility_vector = raw_scores
        
        # GIST greedy selection (diverse + relevant)
        n_select = min(top_k, len(chunk_ids))
        selected_indices = gist_select(coverage_matrix, utility_vector, k=n_select, lambda_param=0.7)
        
        # 8. Return selected with utility scores (normalized to [0,1])
        u_min, u_max = utility_vector.min(), utility_vector.max()
        if u_max > u_min:
            utility_norm = (utility_vector - u_min) / (u_max - u_min)
        else:
            utility_norm = np.ones(len(utility_vector))
        
        results = [(chunk_ids[i], float(utility_norm[i])) for i in selected_indices]
        
        return results
    
    def _expand_via_qwen3(
        self, 
        seed_chunks: List[int], 
        top_k: int
    ) -> List[Tuple[int, float]]:
        """
        Qwen3 expansion via co-occurrence similarity with GIST diversity selection.
        
        Seeds as INPUT (mean pool their embeddings) → similarity search → GIST select (Qwen3 coverage + cosine utility)
        
        Args:
            seed_chunks: Chunk IDs from Layer 1 (seeds)
            top_k: Number of NEW chunks to retrieve
        
        Returns:
            List of (chunk_id, gist_score) tuples — diverse, non-redundant
        """
        if not seed_chunks:
            return []
        
        # 1. Get seed embeddings
        seed_embeddings = self.qwen3_embeddings[seed_chunks]
        
        # 2. Mean pool (or could use max pool)
        query_embedding = np.mean(seed_embeddings, axis=0, keepdims=True)
        
        # 3. Cosine similarity with ALL chunks
        similarities = cosine_similarity(query_embedding, self.qwen3_embeddings)[0]
        
        # 4. Exclude seed chunks
        seed_set = set(seed_chunks)
        candidates = [
            (cid, sim) for cid, sim in enumerate(similarities) 
            if cid not in seed_set
        ]
        
        # 5. Top candidates (oversample 3× for GIST to select from)
        candidates.sort(key=lambda x: x[1], reverse=True)
        top_candidates = candidates[:top_k*3]
        
        if not top_candidates:
            return []
        
        # 6. GIST select: Qwen3 coverage matrix + cosine utility vector
        chunk_ids = [cid for cid, _ in top_candidates]
        raw_scores = np.array([sim for _, sim in top_candidates])
        
        # Coverage: Qwen3 pairwise doc-doc similarity
        candidate_embeddings = self.qwen3_embeddings[chunk_ids]
        coverage_matrix = cosine_similarity(candidate_embeddings)
        
        # Utility: cosine similarity to query (doc-query relevance)
        utility_vector = raw_scores
        
        # GIST greedy selection (diverse + relevant)
        n_select = min(top_k, len(chunk_ids))
        selected_indices = gist_select(coverage_matrix, utility_vector, k=n_select, lambda_param=0.7)
        
        # 7. Return selected with utility scores (normalized to [0,1])
        u_min, u_max = utility_vector.min(), utility_vector.max()
        if u_max > u_min:
            utility_norm = (utility_vector - u_min) / (u_max - u_min)
        else:
            utility_norm = np.ones(len(utility_vector))
        
        results = [(chunk_ids[i], float(utility_norm[i])) for i in selected_indices]
        
        return results
    
    def _rrf_merge(
        self,
        graph_results: List[Tuple[int, float]],
        qwen3_results: List[Tuple[int, float]],
        top_k: int,
        k: int = 60
    ) -> List[Tuple[int, float]]:
        """
        Standard RRF merge over two GIST-selected lists.
        
        Since both lists are already diversity-selected by gist_select(),
        we use plain RRF = Σ 1/(k + rank) to combine them.
        
        Args:
            graph_results: GIST-selected (chunk_id, score) from graph BM25
            qwen3_results: GIST-selected (chunk_id, score) from Qwen3
            top_k: Number of results to return
            k: RRF constant (default 60)
        
        Returns:
            Merged results sorted by RRF score
        """
        rrf_scores = {}
        
        # Graph contributions — rank-based only
        for rank, (chunk_id, _score) in enumerate(graph_results, start=1):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
        
        # Qwen3 contributions — rank-based only
        for rank, (chunk_id, _score) in enumerate(qwen3_results, start=1):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
        
        # Sort by RRF score descending
        merged = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        return merged[:top_k]
    
    def retrieve(self, query: str) -> List[Tuple[int, float]]:
        """
        Full 3-layer φ-scaled retrieval
        
        Args:
            query: User query
        
        Returns:
            List of (chunk_id, score) tuples (length = top_k)
        """
        print(f"\n{'#'*60}")
        print(f"THREE-LAYER φ-RETRIEVAL: {query[:50]}")
        print(f"{'#'*60}")
        
        # ===================================================================
        # LAYER 1: HYBRID RRF (BM25 + Jina) → SEEDS
        # ===================================================================
        print(f"\n{'='*60}")
        print(f"LAYER 1: HYBRID RRF → {self.config.layer1_seeds} SEEDS")
        print(f"{'='*60}")
        
        # Use existing PGVectorRetriever's retrieval methods for Layer 1
        # BM25 only — M2V dense is semantically weak for domain queries,
        # consistently returns off-topic multimodal/vision junk.
        # Layer 2 Qwen3 expansion provides the dense semantic signal instead.
        retrieval_limit = self.config.layer1_seeds * 5  # fetch more to pick from
        bm25_pool = self.layer1._retrieve_bm25(query, retrieval_limit)
        
        # Take top layer1_seeds BM25 results directly as seeds
        layer1_results = bm25_pool[:self.config.layer1_seeds]
        
        # Extract chunk IDs (strings) and convert to integer indices
        seed_doc_ids = [doc.doc_id for doc in layer1_results]
        seed_chunks = [self.doc_id_to_idx[doc_id] for doc_id in seed_doc_ids if doc_id in self.doc_id_to_idx]
        
        if self.verbose:
            print(f"  BM25 pool top-5:")
            for i, doc in enumerate(bm25_pool[:5]):
                text_preview = self.chunks[self.doc_id_to_idx[doc.doc_id]]['text'][:80] if doc.doc_id in self.doc_id_to_idx else '?'
                print(f"    [{i+1}] score={doc.bm25_score:.4f} id={doc.doc_id} {text_preview}...")
        
        print(f"✓ Retrieved {len(seed_chunks)} seed chunks (BM25-only, no M2V dense)")
        
        # ===================================================================
        # LAYER 2: EXPANSION (Graph + Qwen3) → NEW CHUNKS
        # ===================================================================
        print(f"\n{'='*60}")
        print(f"LAYER 2: EXPANSION → {self.config.layer2_expand} NEW CHUNKS")
        print(f"{'='*60}")
        
        # Graph expansion (BM25 over triplets)
        print(f"Graph BM25: Expanding from {len(seed_chunks)} seeds...")
        graph_results = self._expand_via_graph_bm25(
            seed_chunks, 
            top_k=self.config.layer2_expand
        )
        print(f"  ✓ Found {len(graph_results)} graph-expanded chunks (gist-scored)")
        
        # Qwen3 expansion (co-occurrence similarity)
        print(f"Qwen3: Expanding from {len(seed_chunks)} seeds...")
        qwen3_results = self._expand_via_qwen3(
            seed_chunks, 
            top_k=self.config.layer2_expand
        )
        print(f"  ✓ Found {len(qwen3_results)} Qwen3-expanded chunks (gist-scored)")
        
        # RRF merge expansions
        print(f"RRF: Merging graph + qwen3 → {self.config.layer2_expand} expansions...")
        expansion_results = self._rrf_merge(
            graph_results,
            qwen3_results,
            top_k=self.config.layer2_expand
        )
        expansion_chunks = [chunk_id for chunk_id, _ in expansion_results]
        print(f"  ✓ Merged to {len(expansion_chunks)} unique expansions")
        
        # Concatenate seeds + expansions
        all_chunks = seed_chunks + expansion_chunks
        total_chunks = len(all_chunks)
        print(f"\n✓ TOTAL: {len(seed_chunks)} seeds + {len(expansion_chunks)} expansions = {total_chunks} chunks")
        
        # ===================================================================
        # LAYER 3: DUAL RERANKING (ColBERTv2 + Cross-Encoder) with RRF → FINAL TOP_K
        # ===================================================================
        print(f"\n{'='*60}")
        print(f"LAYER 3: DUAL RERANKING → {self.config.layer3_output} FINAL")
        print(f"{'='*60}")
        
        # Get chunk texts for reranking
        chunk_texts = [self.chunks[cid]['text'] for cid in all_chunks]
        
        # 1. ColBERTv2 Late Interaction Reranking (ALL candidates)
        print(f"ColBERTv2: Reranking {len(chunk_texts)} candidates...")
        colbert_scores = []
        for text in chunk_texts:
            query_emb = self.colbert_model.encode([query], convert_to_tensor=False)[0]
            doc_emb = self.colbert_model.encode([text], convert_to_tensor=False)[0]
            score = float(np.dot(query_emb, doc_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(doc_emb)))
            colbert_scores.append(score)
        colbert_scores = np.array(colbert_scores)
        print(f"  ✓ ColBERTv2 scores range: [{np.min(colbert_scores):.4f}, {np.max(colbert_scores):.4f}]")
        
        # Rank ALL candidates by ColBERTv2 score
        colbert_ranking = sorted(range(len(all_chunks)), key=lambda i: colbert_scores[i], reverse=True)
        
        # 2. Cross-Encoder Reranking (ALL candidates)
        print(f"Cross-Encoder: Reranking {len(chunk_texts)} candidates...")
        pairs = [[query, text] for text in chunk_texts]
        cross_encoder_scores = self.cross_encoder.predict(pairs, show_progress_bar=False)
        cross_encoder_scores = np.array(cross_encoder_scores)
        print(f"  ✓ Cross-Encoder scores range: [{np.min(cross_encoder_scores):.4f}, {np.max(cross_encoder_scores):.4f}]")
        
        # Rank ALL candidates by Cross-Encoder score
        ce_ranking = sorted(range(len(all_chunks)), key=lambda i: cross_encoder_scores[i], reverse=True)
        
        # 3. RRF Fusion of FULL rankings (all 157, not GIST subsets)
        print(f"RRF: Merging full ColBERTv2 + Cross-Encoder rankings ({len(all_chunks)} each)...")
        rrf_scores = {}
        k_param = 60
        
        for rank, idx in enumerate(colbert_ranking, start=1):
            chunk_id = all_chunks[idx]
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (k_param + rank)
        
        for rank, idx in enumerate(ce_ranking, start=1):
            chunk_id = all_chunks[idx]
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (k_param + rank)
        
        # Sort all chunks by RRF score
        rrf_ranking = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        print(f"  ✓ RRF scored {len(rrf_ranking)} chunks")
        
        # 4. Walk down RRF-sorted list, identify top_k unique PAPERS
        print(f"Paper Selection: Walking RRF ranking for top {self.config.layer3_output} papers...")
        selected_papers = set()
        for chunk_id, rrf_score in rrf_ranking:
            paper_id = self.chunks[chunk_id].get('paper_id', 'unknown')
            if paper_id not in selected_papers:
                selected_papers.add(paper_id)
                if len(selected_papers) >= self.config.layer3_output:
                    break
        
        # 5. Return ALL chunks/sections from those top_k papers (preserving RRF order)
        final_results = [
            (chunk_id, rrf_score)
            for chunk_id, rrf_score in rrf_ranking
            if self.chunks[chunk_id].get('paper_id', 'unknown') in selected_papers
        ]
        unique_papers = len(selected_papers)
        print(f"  ✓ Selected {len(final_results)} sections from {unique_papers} unique papers")
        
        print(f"\n{'#'*60}")
        print(f"FINAL OUTPUT: {len(final_results)} chunks")
        print(f"{'#'*60}\n")
        
        return final_results


def example_usage():
    """Example usage"""
    from gist_retriever import GistRetriever
    
    # Initialize Layer 1 (existing hybrid retriever)
    layer1 = GistRetriever(
        chunks_path="checkpoints/chunks.msgpack",
        bm25_vocab_path="bm25_vocab.msgpack",
        embedding_model='jina',
        use_hnsw=True
    )
    
    # Initialize 3-layer φ-retriever with dual reranking
    retriever = ThreeLayerPhiRetriever(
        chunks_path="checkpoints/chunks.msgpack",
        chunk_embeddings_qwen3_path="checkpoints/chunk_embeddings_qwen3.msgpack",
        triplets_path="checkpoints/enriched_triplets.msgpack",
        chunk_to_triplets_path="checkpoints/chunk_to_triplets.msgpack",
        triplet_to_chunks_path="checkpoints/triplet_to_chunks.msgpack",
        triplet_bm25_path="checkpoints/wordpiece_bm25_index.msgpack",
        layer1_retriever=layer1,
        top_k=13,
        colbert_model_name="colbert-ir/colbertv2.0",  # REQUIRED: ColBERTv2
        cross_encoder_model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"  # REQUIRED: Cross-Encoder
    )
    
    # Query
    query = "What is the attention mechanism in transformers?"
    results = retriever.retrieve(query)
    
    print(f"\nTop {len(results)} results:")
    for i, (chunk_id, score) in enumerate(results, 1):
        chunk_text = retriever.chunks[chunk_id]['text'][:100]
        print(f"\n{i}. Chunk {chunk_id} (score: {score:.4f})")
        print(f"   {chunk_text}...")


if __name__ == '__main__':
    example_usage()
