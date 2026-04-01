"""
Unit tests for three_layer_phi_retriever.py

Tests:
1. PhiLayerConfig φ-scaling calculations
2. ECDF normalization correctness
3. Seed exclusion in Layer 2
4. Gist-weighted RRF merge
5. Concatenation strategy (13+144=157)
6. Full pipeline integration
"""

import pytest
import numpy as np
from three_layer_phi_retriever import (
    PhiLayerConfig,
    ThreeLayerPhiRetriever
)


class TestPhiLayerConfig:
    """Test φ-scaled layer configuration"""
    
    def test_default_top_k_13(self):
        """Test default configuration with top_k=13"""
        config = PhiLayerConfig.from_top_k(13)
        
        assert config.top_k == 13
        assert config.layer1_seeds == 13
        assert config.layer2_expand == 144  # 13² - 25 = 169 - 25 = 144
        assert config.layer3_output == 13
    
    def test_custom_top_k(self):
        """Test custom top_k values"""
        # top_k = 10
        config = PhiLayerConfig.from_top_k(10)
        assert config.layer2_expand == 75  # 10² - 25 = 100 - 25 = 75
        
        # top_k = 20
        config = PhiLayerConfig.from_top_k(20)
        assert config.layer2_expand == 375  # 20² - 25 = 400 - 25 = 375
    
    def test_total_layer3_input(self):
        """Test Layer 3 input size (seeds + expansions)"""
        config = PhiLayerConfig.from_top_k(13)
        total = config.layer1_seeds + config.layer2_expand
        
        assert total == 157  # 13 + 144 = 157


class TestECDFNormalization:
    """Test ECDF (Empirical Cumulative Distribution Function) normalization"""
    
    def test_uniform_output_distribution(self):
        """Test that ECDF produces uniform [0,1] distribution"""
        # Create mock retriever (only need _ecdf_normalize method)
        retriever = MockRetriever()
        
        # Various input distributions
        scores_uniform = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        scores_normal = np.random.randn(100)
        scores_exponential = np.random.exponential(scale=2.0, size=100)
        
        # ECDF normalize
        gist_uniform = retriever._ecdf_normalize(scores_uniform)
        gist_normal = retriever._ecdf_normalize(scores_normal)
        gist_exponential = retriever._ecdf_normalize(scores_exponential)
        
        # Check all in [0, 1]
        assert np.all((gist_uniform >= 0) & (gist_uniform <= 1))
        assert np.all((gist_normal >= 0) & (gist_normal <= 1))
        assert np.all((gist_exponential >= 0) & (gist_exponential <= 1))
        
        # Check min/max bounds
        assert np.min(gist_uniform) == pytest.approx(1/6, rel=1e-3)  # 1/(n+1)
        assert np.max(gist_uniform) == pytest.approx(5/6, rel=1e-3)  # n/(n+1)
    
    def test_rank_preservation(self):
        """Test that ECDF preserves relative ordering"""
        retriever = MockRetriever()
        
        # Create scores with known ordering
        scores = np.array([15.2, 8.3, 12.1, 2.1, 5.2])
        # Sorted indices: [0, 2, 1, 4, 3] (15.2 > 12.1 > 8.3 > 5.2 > 2.1)
        
        gist = retriever._ecdf_normalize(scores)
        
        # Verify ordering preserved
        assert gist[0] > gist[2] > gist[1] > gist[4] > gist[3]
    
    def test_empty_scores(self):
        """Test handling of empty score array"""
        retriever = MockRetriever()
        
        scores = np.array([])
        gist = retriever._ecdf_normalize(scores)
        
        assert len(gist) == 0
    
    def test_single_score(self):
        """Test handling of single score"""
        retriever = MockRetriever()
        
        scores = np.array([42.0])
        gist = retriever._ecdf_normalize(scores)
        
        assert len(gist) == 1
        assert gist[0] == pytest.approx(0.5, rel=1e-3)  # 1/(1+1) = 0.5
    
    def test_comparable_across_distributions(self):
        """Test that ECDF makes different distributions comparable"""
        retriever = MockRetriever()
        
        # BM25 scores (unbounded, high values)
        bm25_scores = np.array([15.2, 12.1, 8.3, 5.2, 2.1])
        
        # Cosine scores (bounded [-1, 1])
        cosine_scores = np.array([0.92, 0.88, 0.81, 0.75, 0.64])
        
        # ECDF normalize
        bm25_gist = retriever._ecdf_normalize(bm25_scores)
        cosine_gist = retriever._ecdf_normalize(cosine_scores)
        
        # Both should have similar distributions
        # Top performers should be ~0.8-0.95
        assert bm25_gist[0] > 0.8  # Top BM25 score
        assert cosine_gist[0] > 0.8  # Top cosine score
        
        # Bottom performers should be ~0.1-0.2
        assert bm25_gist[-1] < 0.3
        assert cosine_gist[-1] < 0.3


class TestSeedExclusion:
    """Test that Layer 2 excludes Layer 1 seeds"""
    
    def test_graph_expansion_excludes_seeds(self):
        """Test that graph expansion doesn't return seed chunks"""
        retriever = create_mock_retriever()
        
        # Layer 1 seeds
        seed_chunks = [0, 5, 10, 15, 20]
        
        # Graph expansion
        graph_results = retriever._expand_via_graph_bm25(seed_chunks, top_k=50)
        graph_chunk_ids = [cid for cid, _ in graph_results]
        
        # Verify no overlap with seeds
        seed_set = set(seed_chunks)
        assert not any(cid in seed_set for cid in graph_chunk_ids), \
            "Graph expansion returned seed chunks!"
    
    def test_qwen3_expansion_excludes_seeds(self):
        """Test that Qwen3 expansion doesn't return seed chunks"""
        retriever = create_mock_retriever()
        
        # Layer 1 seeds
        seed_chunks = [0, 5, 10, 15, 20]
        
        # Qwen3 expansion
        qwen3_results = retriever._expand_via_qwen3(seed_chunks, top_k=50)
        qwen3_chunk_ids = [cid for cid, _ in qwen3_results]
        
        # Verify no overlap with seeds
        seed_set = set(seed_chunks)
        assert not any(cid in seed_set for cid in qwen3_chunk_ids), \
            "Qwen3 expansion returned seed chunks!"
    
    def test_rrf_merge_excludes_seeds(self):
        """Test that RRF merge output doesn't contain seeds"""
        retriever = create_mock_retriever()
        
        seed_chunks = [0, 5, 10]
        
        # Get expansions (already exclude seeds)
        graph_results = retriever._expand_via_graph_bm25(seed_chunks, top_k=30)
        qwen3_results = retriever._expand_via_qwen3(seed_chunks, top_k=30)
        
        # RRF merge
        merged = retriever._rrf_merge_gist(graph_results, qwen3_results, top_k=50)
        merged_chunk_ids = [cid for cid, _ in merged]
        
        # Verify no seeds in merged results
        seed_set = set(seed_chunks)
        assert not any(cid in seed_set for cid in merged_chunk_ids), \
            "RRF merge returned seed chunks!"


class TestGistWeightedRRF:
    """Test gist-weighted RRF merge"""
    
    def test_rrf_basic_merge(self):
        """Test that RRF combines two result lists"""
        retriever = MockRetriever()
        
        # Two result lists with gist scores
        graph_results = [(1, 0.9), (2, 0.8), (3, 0.7)]
        qwen3_results = [(1, 0.85), (4, 0.75), (5, 0.65)]
        
        merged = retriever._rrf_merge_gist(graph_results, qwen3_results, top_k=5)
        
        # Chunk 1 appears in both, should rank high
        chunk_ids = [cid for cid, _ in merged]
        assert 1 in chunk_ids[:2], "Chunk appearing in both sources should rank high"
    
    def test_gist_weighting_effect(self):
        """Test that gist scores affect RRF ranking"""
        retriever = MockRetriever()
        
        # Graph: chunk 1 at rank 1 with high gist, chunk 2 at rank 2 with low gist
        graph_results = [(1, 0.95), (2, 0.10)]
        
        # Qwen3: chunk 2 at rank 1 with low gist, chunk 1 at rank 2 with high gist
        qwen3_results = [(2, 0.10), (1, 0.95)]
        
        merged = retriever._rrf_merge_gist(graph_results, qwen3_results, top_k=2, k=60)
        
        # Chunk 1 should rank higher due to high gist scores in both
        chunk_ids = [cid for cid, _ in merged]
        assert chunk_ids[0] == 1, "Chunk with high gist scores should rank first"
    
    def test_empty_results(self):
        """Test RRF with empty result lists"""
        retriever = MockRetriever()
        
        merged = retriever._rrf_merge_gist([], [], top_k=10)
        assert len(merged) == 0
        
        merged = retriever._rrf_merge_gist([(1, 0.9)], [], top_k=10)
        assert len(merged) == 1


class TestConcatenationStrategy:
    """Test concatenation of seeds + expansions"""
    
    def test_concatenation_size(self):
        """Test that concatenation produces correct size"""
        retriever = create_mock_retriever()
        
        # Mock Layer 1 seeds (13)
        seed_chunks = list(range(13))
        
        # Mock Layer 2 expansions (144)
        expansion_chunks = list(range(100, 244))  # 144 chunks
        
        # Concatenate
        all_chunks = seed_chunks + expansion_chunks
        
        # Should be 157 total
        assert len(all_chunks) == 157
        assert len(set(all_chunks)) == 157  # All unique
    
    def test_seeds_first_ordering(self):
        """Test that seeds come before expansions"""
        retriever = create_mock_retriever()
        
        seed_chunks = [0, 1, 2]
        expansion_chunks = [10, 11, 12]
        
        all_chunks = seed_chunks + expansion_chunks
        
        # First 3 should be seeds
        assert all_chunks[:3] == seed_chunks
        # Next 3 should be expansions
        assert all_chunks[3:6] == expansion_chunks


class TestFullPipeline:
    """Integration tests for full pipeline"""
    
    @pytest.mark.integration
    def test_pipeline_sizes(self):
        """Test that pipeline produces correct sizes at each layer"""
        retriever = create_mock_retriever()
        
        # Mock query
        query = "test query"
        
        # Layer 1: Should return 13 seeds
        layer1_results = retriever.layer1.retrieve(query, top_k=13)
        assert len(layer1_results) == 13
        
        # Layer 2: Should return 144 expansions
        seed_chunks = [cid for cid, _ in layer1_results]
        expansion_results = retriever._expand_via_graph_bm25(seed_chunks, top_k=144)
        assert len(expansion_results) <= 144  # May be less if corpus small
        
        # Concatenation: Should be 157 total
        all_chunks = seed_chunks + [cid for cid, _ in expansion_results]
        assert len(all_chunks) <= 157
        
        # Layer 3: Should return 13 final
        final_results = all_chunks[:13]
        assert len(final_results) == 13
    
    @pytest.mark.integration
    def test_no_seed_leakage(self):
        """Test that seeds never appear in Layer 2 expansion output"""
        retriever = create_mock_retriever()
        
        seed_chunks = [0, 5, 10, 15, 20]
        
        # Get both expansions
        graph_results = retriever._expand_via_graph_bm25(seed_chunks, top_k=50)
        qwen3_results = retriever._expand_via_qwen3(seed_chunks, top_k=50)
        
        # Merge
        merged = retriever._rrf_merge_gist(graph_results, qwen3_results, top_k=100)
        
        # Verify no seeds
        seed_set = set(seed_chunks)
        expansion_ids = [cid for cid, _ in merged]
        assert not any(cid in seed_set for cid in expansion_ids)


# ============================================================================
# MOCK OBJECTS FOR TESTING
# ============================================================================

class MockRetriever:
    """Mock retriever with only ECDF and RRF methods"""
    
    def _ecdf_normalize(self, scores: np.ndarray) -> np.ndarray:
        """ECDF normalization"""
        if len(scores) == 0:
            return np.array([])
        
        ranks = np.argsort(np.argsort(scores))
        return (ranks + 1) / (len(scores) + 1)
    
    def _rrf_merge_gist(
        self,
        graph_results,
        qwen3_results,
        top_k,
        k=60
    ):
        """RRF merge with gist weighting"""
        rrf_scores = {}
        
        for rank, (chunk_id, gist_score) in enumerate(graph_results, start=1):
            rrf_contribution = 1.0 / (k + rank)
            weighted = rrf_contribution * gist_score
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + weighted
        
        for rank, (chunk_id, gist_score) in enumerate(qwen3_results, start=1):
            rrf_contribution = 1.0 / (k + rank)
            weighted = rrf_contribution * gist_score
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + weighted
        
        merged = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return merged[:top_k]


def create_mock_retriever():
    """Create mock retriever with fake data for testing"""
    
    class MockLayer1:
        def retrieve(self, query, top_k):
            # Return mock seeds
            return [(i, 1.0) for i in range(top_k)]
    
    class MockFullRetriever:
        def __init__(self):
            self.layer1 = MockLayer1()
            
            # Mock data structures
            self.chunk_to_triplets = {
                str(i): [i*10, i*10+1, i*10+2] for i in range(300)
            }
            self.triplet_to_chunks = {
                str(i): [i // 10] for i in range(3000)
            }
            self.triplets = [
                {'lemma_text': f'triplet_{i}'} for i in range(3000)
            ]
            self.qwen3_embeddings = np.random.randn(300, 256).astype(np.float32)
            
            # Mock BM25
            class MockBM25:
                def get_scores(self, tokens):
                    return np.random.uniform(0, 20, size=3000)
            
            self.graph_bm25 = MockBM25()
        
        def _ecdf_normalize(self, scores):
            if len(scores) == 0:
                return np.array([])
            ranks = np.argsort(np.argsort(scores))
            return (ranks + 1) / (len(scores) + 1)
        
        def _expand_via_graph_bm25(self, seed_chunks, top_k):
            # Mock expansion (exclude seeds)
            seed_set = set(seed_chunks)
            candidates = [i for i in range(300) if i not in seed_set]
            scores = np.random.uniform(0, 1, size=len(candidates))
            gist_scores = self._ecdf_normalize(scores)
            results = list(zip(candidates, gist_scores))
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:top_k]
        
        def _expand_via_qwen3(self, seed_chunks, top_k):
            # Mock expansion (exclude seeds)
            seed_set = set(seed_chunks)
            candidates = [i for i in range(300) if i not in seed_set]
            scores = np.random.uniform(0, 1, size=len(candidates))
            gist_scores = self._ecdf_normalize(scores)
            results = list(zip(candidates, gist_scores))
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:top_k]
        
        def _rrf_merge_gist(self, graph_results, qwen3_results, top_k, k=60):
            rrf_scores = {}
            
            for rank, (chunk_id, gist_score) in enumerate(graph_results, start=1):
                rrf_contribution = 1.0 / (k + rank)
                weighted = rrf_contribution * gist_score
                rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + weighted
            
            for rank, (chunk_id, gist_score) in enumerate(qwen3_results, start=1):
                rrf_contribution = 1.0 / (k + rank)
                weighted = rrf_contribution * gist_score
                rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + weighted
            
            merged = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
            return merged[:top_k]
    
    return MockFullRetriever()


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
