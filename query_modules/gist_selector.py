"""
GIST Selection Module

Core algorithm for Greedy Information Selection with Topic diversity.
Balances utility (relevance) and coverage (diversity) via iterative selection.

Reference: gist_retriever.py lines 236-340
"""

import numpy as np
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass


@dataclass
class GISTResult:
    """Result of GIST selection."""
    selected_indices: List[int]  # Indices in selection order
    selection_scores: List[float]  # Score at time of selection
    final_utility_scores: List[float]  # Utility scores for selected items
    final_coverage_penalties: List[float]  # Coverage penalties for selected items


def gist_select(
    coverage_matrix: np.ndarray,
    utility_vector: np.ndarray,
    k: int,
    lambda_param: float = 0.7,
    verbose: bool = False
) -> GISTResult:
    """
    GIST: Greedy Information Selection with Topic diversity.
    
    Iteratively select k items balancing utility (relevance) and coverage (diversity).
    
    Mathematical formulation (MMR-style with VIF control):
        score(d) = λ * utility(d) - (1-λ) * max_sim(d, S)
    
    Where:
        - utility(d) = normalized relevance score (doc-query similarity)
        - max_sim(d, S) = maximum similarity to any already-selected doc
        - λ = tradeoff parameter (0=diversity only, 1=relevance only)
    
    Args:
        coverage_matrix: n×n matrix of doc-doc similarities
        utility_vector: n×1 vector of doc-query similarities
        k: Number of items to select
        lambda_param: Tradeoff parameter (default: 0.7)
        verbose: Print selection details
    
    Returns:
        GISTResult with selected indices and scores
    
    Algorithm:
        1. Normalize utility scores to [0,1]
        2. Initialize collinearity penalties to 0
        3. For each selection iteration:
            a. Score remaining items: λ*utility - (1-λ)*collinearity
            b. Select highest scoring item
            c. Update collinearity penalties for remaining items
        4. Return selected indices in order
    """
    n = len(utility_vector)
    
    if k >= n:
        # Select all items
        return GISTResult(
            selected_indices=list(range(n)),
            selection_scores=[1.0] * n,
            final_utility_scores=utility_vector.tolist(),
            final_coverage_penalties=[0.0] * n
        )
    
    # Normalize utility scores to [0, 1]
    util_min = utility_vector.min()
    util_max = utility_vector.max()
    if util_max - util_min > 1e-9:
        utility_norm = (utility_vector - util_min) / (util_max - util_min)
    else:
        utility_norm = np.ones_like(utility_vector) * 0.5
    
    # Track selection
    selected_indices = []
    selected_mask = np.zeros(n, dtype=bool)
    max_sim_to_selected = np.zeros(n)  # Collinearity penalty tracker
    
    selection_scores = []
    final_utility_scores = []
    final_coverage_penalties = []
    
    for iteration in range(k):
        # Calculate scores for remaining items
        remaining_mask = ~selected_mask
        remaining_indices = np.where(remaining_mask)[0]
        
        if len(remaining_indices) == 0:
            break
        
        # Score = λ * utility - (1-λ) * collinearity
        utility_component = utility_norm[remaining_indices]
        coverage_component = max_sim_to_selected[remaining_indices]
        
        scores = lambda_param * utility_component - (1 - lambda_param) * coverage_component
        
        # Select best
        best_idx_in_remaining = np.argmax(scores)
        best_idx = remaining_indices[best_idx_in_remaining]
        best_score = scores[best_idx_in_remaining]
        
        # Record selection
        selected_indices.append(int(best_idx))
        selected_mask[best_idx] = True
        selection_scores.append(float(best_score))
        final_utility_scores.append(float(utility_norm[best_idx]))
        final_coverage_penalties.append(float(max_sim_to_selected[best_idx]))
        
        if verbose:
            print(f"  [{iteration+1}/{k}] Selected idx={best_idx}, "
                  f"score={best_score:.4f}, "
                  f"utility={utility_norm[best_idx]:.4f}, "
                  f"coverage={max_sim_to_selected[best_idx]:.4f}")
        
        # Update collinearity penalties for remaining items
        new_similarities = coverage_matrix[best_idx, :]
        max_sim_to_selected = np.maximum(max_sim_to_selected, new_similarities)
    
    return GISTResult(
        selected_indices=selected_indices,
        selection_scores=selection_scores,
        final_utility_scores=final_utility_scores,
        final_coverage_penalties=final_coverage_penalties
    )


def build_coverage_matrix_from_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """
    Build coverage matrix from embeddings using cosine similarity.
    
    Args:
        embeddings: n×d matrix of embeddings
    
    Returns:
        n×n cosine similarity matrix
    """
    from sklearn.metrics.pairwise import cosine_similarity
    return cosine_similarity(embeddings)


def build_utility_vector_from_embeddings(
    embeddings: np.ndarray,
    query_embedding: np.ndarray
) -> np.ndarray:
    """
    Build utility vector from embeddings using cosine similarity.
    
    Args:
        embeddings: n×d matrix of document embeddings
        query_embedding: 1×d query embedding
    
    Returns:
        n×1 vector of cosine similarities
    """
    from sklearn.metrics.pairwise import cosine_similarity
    return cosine_similarity(embeddings, query_embedding.reshape(1, -1)).flatten()


def test_gist_selector():
    """Unit test for GIST selector."""
    print("\n" + "="*60)
    print("TESTING GIST SELECTOR")
    print("="*60)
    
    # Create test data: 10 items, 3-dimensional
    np.random.seed(42)
    n_items = 10
    
    # Utility: Items 0-2 highly relevant, 3-6 medium, 7-9 low
    utility = np.array([0.9, 0.85, 0.88, 0.6, 0.65, 0.62, 0.58, 0.3, 0.25, 0.28])
    
    # Coverage: Create similarity matrix with some clusters
    # Items 0-2 similar (high relevance cluster)
    # Items 3-6 similar (medium relevance cluster)
    # Items 7-9 similar (low relevance cluster)
    coverage = np.zeros((n_items, n_items))
    
    # High relevance cluster (0-2): high similarity within cluster
    for i in range(3):
        for j in range(3):
            coverage[i, j] = 0.9 if i != j else 1.0
    
    # Medium relevance cluster (3-6): high similarity within cluster
    for i in range(3, 7):
        for j in range(3, 7):
            coverage[i, j] = 0.85 if i != j else 1.0
    
    # Low relevance cluster (7-9): high similarity within cluster
    for i in range(7, 10):
        for j in range(7, 10):
            coverage[i, j] = 0.8 if i != j else 1.0
    
    # Inter-cluster similarities (moderate)
    for i in range(3):
        for j in range(3, 7):
            coverage[i, j] = coverage[j, i] = 0.4
        for j in range(7, 10):
            coverage[i, j] = coverage[j, i] = 0.2
    
    for i in range(3, 7):
        for j in range(7, 10):
            coverage[i, j] = coverage[j, i] = 0.3
    
    # Test 1: High lambda (favor relevance)
    print("\nTest 1: λ=0.9 (favor relevance)")
    result_high_lambda = gist_select(coverage, utility, k=5, lambda_param=0.9, verbose=True)
    print(f"  Selected: {result_high_lambda.selected_indices}")
    print(f"  Expected: Mostly from high relevance cluster (0-2)")
    
    # Test 2: Low lambda (favor diversity)
    print("\nTest 2: λ=0.3 (favor diversity)")
    result_low_lambda = gist_select(coverage, utility, k=5, lambda_param=0.3, verbose=True)
    print(f"  Selected: {result_low_lambda.selected_indices}")
    print(f"  Expected: One from each cluster for diversity")
    
    # Test 3: Balanced (default)
    print("\nTest 3: λ=0.7 (balanced)")
    result_balanced = gist_select(coverage, utility, k=5, lambda_param=0.7, verbose=True)
    print(f"  Selected: {result_balanced.selected_indices}")
    print(f"  Expected: Mostly high relevance, but diverse within that cluster")
    
    print("\n✓ GIST selector tests complete")


if __name__ == '__main__':
    test_gist_selector()
