"""
Graph Expansion via Enriched Triplet BM25

Adds graph expansion stage to hybrid retrieval pipeline:
1. Hybrid GIST RRF → K results
2. Graph expand using enriched triplet similarity → 2K results
3. Rerank back to K using triplet BM25 scores
4. Return for ColBERT/cross-encoder late interaction

The "graph" is implicit: chunks sharing enriched terms = connected nodes.
No explicit graph construction needed.
"""

import pickle
import numpy as np
from typing import List, Dict, Tuple
from rank_bm25 import BM25Okapi
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer


class GraphExpander:
    """
    Expand retrieval results using enriched triplet similarity.
    """
    
    def __init__(self, index_path: str):
        """
        Load pre-built enriched triplet BM25 index.
        
        Args:
            index_path: Path to triplet_bm25_index.msgpack (from build_triplet_bm25.py)
        """
        print(f"Loading enriched triplet BM25 index from {index_path}...")
        with open(index_path, 'rb') as f:
            data = pickle.load(f)
        
        self.bm25 = data['bm25']
        self.corpus = data['corpus']  # List[List[str]] - enriched terms per chunk
        self.chunk_ids = data['chunk_ids']
        
        # Build reverse mapping: chunk_id → corpus index
        self.chunk_id_to_idx = {cid: idx for idx, cid in enumerate(self.chunk_ids)}
        
        print(f"✅ Loaded index with {len(self.chunk_ids)} chunks")
        
        # Initialize enrichment tools (for query processing)
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def enrich_query(self, query_tokens: List[str]) -> List[str]:
        """
        Apply same enrichment as corpus: stopwords, lemmas, synsets, hypernyms.
        
        Args:
            query_tokens: Raw query tokens
            
        Returns:
            Enriched query terms
        """
        enriched = []
        
        for token in query_tokens:
            if token.lower() in self.stop_words:
                continue
            
            term = token.lower()
            enriched.append(term)
            
            lemma = self.lemmatizer.lemmatize(term)
            if lemma != term:
                enriched.append(lemma)
            
            synsets = wordnet.synsets(lemma)
            if synsets:
                synset = synsets[0]
                synset_name = synset.name().split('.')[0]
                if synset_name not in enriched:
                    enriched.append(synset_name)
                
                hypernyms = synset.hypernyms()
                if hypernyms:
                    hypernym_name = hypernyms[0].name().split('.')[0]
                    if hypernym_name not in enriched:
                        enriched.append(hypernym_name)
        
        return enriched
    
    def expand(self, initial_results: List[Tuple[str, float]], query: str, 
               expand_factor: float = 2.0) -> List[Tuple[str, float]]:
        """
        Expand initial results using enriched triplet similarity.
        
        Args:
            initial_results: List of (chunk_id, score) from hybrid RRF
            query: Original query string
            expand_factor: Multiply result count (default 2.0 = double)
            
        Returns:
            Expanded and reranked list of (chunk_id, score)
        """
        k_initial = len(initial_results)
        k_expanded = int(k_initial * expand_factor)
        
        # Enrich query
        query_tokens = query.split()
        enriched_query = self.enrich_query(query_tokens)
        
        # Get BM25 scores for all chunks
        all_scores = self.bm25.get_scores(enriched_query)
        
        # Combine with initial results
        # Strategy: Start with initial results, then add high-scoring neighbors
        initial_chunk_ids = set([cid for cid, _ in initial_results])
        
        # Get top-K_expanded chunks by BM25 score
        top_indices = np.argsort(all_scores)[::-1][:k_expanded * 2]  # Over-sample
        
        # Build candidate set
        candidates = {}
        
        # Add initial results (preserve their RRF scores)
        for chunk_id, rrf_score in initial_results:
            if chunk_id in self.chunk_id_to_idx:
                idx = self.chunk_id_to_idx[chunk_id]
                bm25_score = all_scores[idx]
                # Combine RRF and BM25: weighted average
                combined_score = 0.7 * rrf_score + 0.3 * (bm25_score / (1 + bm25_score))
                candidates[chunk_id] = combined_score
        
        # Add expansion candidates (high BM25 score but not in initial results)
        for idx in top_indices:
            chunk_id = self.chunk_ids[idx]
            if chunk_id not in candidates:
                bm25_score = all_scores[idx]
                # Pure BM25 score for new candidates
                candidates[chunk_id] = bm25_score / (1 + bm25_score)  # Normalize
                
                if len(candidates) >= k_expanded:
                    break
        
        # Sort by combined score
        expanded_results = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        
        # Take top k_expanded
        expanded_results = expanded_results[:k_expanded]
        
        # Rerank back to k_initial using pure BM25 scores
        # (This ensures we keep best semantically-matched chunks)
        reranked = []
        for chunk_id, _ in expanded_results:
            idx = self.chunk_id_to_idx[chunk_id]
            bm25_score = all_scores[idx]
            reranked.append((chunk_id, bm25_score))
        
        # Sort by BM25 and take top k_initial
        reranked = sorted(reranked, key=lambda x: x[1], reverse=True)[:k_initial]
        
        return reranked
    
    def get_neighbors(self, chunk_id: str, query: str, k: int = 10) -> List[Tuple[str, float]]:
        """
        Find k-nearest neighbors of a chunk via enriched triplet similarity.
        
        Args:
            chunk_id: Source chunk
            query: Query context
            k: Number of neighbors
            
        Returns:
            List of (neighbor_chunk_id, similarity_score)
        """
        if chunk_id not in self.chunk_id_to_idx:
            return []
        
        idx = self.chunk_id_to_idx[chunk_id]
        chunk_terms = self.corpus[idx]
        
        # Enrich query for context
        query_tokens = query.split()
        enriched_query = self.enrich_query(query_tokens)
        
        # Combine chunk terms and query terms (gives context-aware neighbors)
        combined_query = list(set(chunk_terms + enriched_query))
        
        # Score all other chunks
        scores = self.bm25.get_scores(combined_query)
        
        # Get top-k (excluding self)
        top_indices = np.argsort(scores)[::-1]
        neighbors = []
        
        for other_idx in top_indices:
            if other_idx == idx:
                continue
            
            other_chunk_id = self.chunk_ids[other_idx]
            score = scores[other_idx]
            neighbors.append((other_chunk_id, score))
            
            if len(neighbors) >= k:
                break
        
        return neighbors


# Integration with existing retriever
def add_graph_expansion_to_retriever(retriever_class):
    """
    Decorator to add graph expansion to any retriever.
    
    Usage:
        @add_graph_expansion_to_retriever
        class MyRetriever:
            def retrieve(self, query, k):
                # Existing hybrid retrieval
                results = self.hybrid_retrieve(query, k)
                
                # Add graph expansion
                if hasattr(self, 'graph_expander'):
                    results = self.graph_expander.expand(results, query)
                
                return results
    """
    original_init = retriever_class.__init__
    
    def new_init(self, *args, graph_index_path=None, **kwargs):
        original_init(self, *args, **kwargs)
        
        if graph_index_path:
            self.graph_expander = GraphExpander(graph_index_path)
        else:
            self.graph_expander = None
    
    retriever_class.__init__ = new_init
    return retriever_class


if __name__ == '__main__':
    # Test graph expander
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python graph_expander.py <index_path>")
        sys.exit(1)
    
    index_path = sys.argv[1]
    
    print("Testing GraphExpander...")
    expander = GraphExpander(index_path)
    
    # Simulate initial results
    initial_results = [
        ("chunk_001", 0.95),
        ("chunk_042", 0.87),
        ("chunk_123", 0.76)
    ]
    
    query = "machine learning neural networks"
    
    print(f"\nQuery: {query}")
    print(f"Initial results: {len(initial_results)}")
    
    expanded = expander.expand(initial_results, query, expand_factor=2.0)
    
    print(f"Expanded results: {len(expanded)}")
    print("\nTop 5 after expansion:")
    for chunk_id, score in expanded[:5]:
        print(f"  {chunk_id}: {score:.4f}")
