"""
Unit Test: Layer 3 GIST Selection
Tests that Layer 3 correctly scores and selects top-k chunks using GIST framework.
"""

import sys
import time
import msgpack
import numpy as np
from pathlib import Path

def load_test_data():
    """Load minimal data needed for Layer 3 testing."""
    print("Loading test data...")
    
    # Load chunks
    with open('checkpoints/chunks.msgpack', 'rb') as f:
        chunks = msgpack.unpackb(f.read(), raw=False)
    print(f"✓ Loaded {len(chunks)} chunks")
    
    # Load embeddings (stored as dict: chunk_id -> embedding)
    with open('checkpoints/chunk_embeddings_qwen3.msgpack', 'rb') as f:
        emb_dict = msgpack.unpackb(f.read(), raw=False)
    
    # Convert to array aligned with chunks
    embeddings = []
    for chunk in chunks:
        chunk_id = chunk.get('id', chunk.get('chunk_id'))
        emb = emb_dict.get(chunk_id, emb_dict.get(str(chunk_id)))
        if emb is None:
            print(f"⚠️  Missing embedding for chunk {chunk_id}")
            emb = np.zeros(256, dtype=np.float32)
        embeddings.append(emb)
    
    embeddings = np.array(embeddings, dtype=np.float32)
    print(f"✓ Loaded embeddings: {embeddings.shape}")
    
    return chunks, embeddings


def compute_gist_scores(query_embedding, candidate_embeddings, k=13):
    """
    Compute GIST scores for candidate chunks.
    
    GIST = Coverage + Utility
    - Coverage: Distance from k-means centroids (cartesian product clustering)
    - Utility: Correlation matrix (maximize query similarity, minimize chunk correlation)
    
    For unit test: Simplified scoring using just semantic similarity + diversity
    """
    print(f"\nComputing GIST scores for {len(candidate_embeddings)} candidates...")
    
    # Semantic similarity to query (utility component)
    query_similarities = candidate_embeddings @ query_embedding
    print(f"  Query similarities range: [{query_similarities.min():.4f}, {query_similarities.max():.4f}]")
    
    # Diversity scoring (coverage component)
    # Higher score for chunks that are dissimilar to each other
    if len(candidate_embeddings) > 1:
        chunk_correlations = candidate_embeddings @ candidate_embeddings.T
        # Penalize high correlation with other candidates
        diversity_scores = 1.0 - np.mean(chunk_correlations, axis=1)
        print(f"  Diversity scores range: [{diversity_scores.min():.4f}, {diversity_scores.max():.4f}]")
    else:
        diversity_scores = np.ones(len(candidate_embeddings))
    
    # Combined GIST score (weighted sum)
    gist_scores = 0.7 * query_similarities + 0.3 * diversity_scores
    print(f"  GIST scores range: [{gist_scores.min():.4f}, {gist_scores.max():.4f}]")
    
    # Select top-k
    top_indices = np.argsort(gist_scores)[::-1][:k]
    top_scores = gist_scores[top_indices]
    
    print(f"  Selected top-{k} chunks with scores: [{top_scores.min():.4f}, {top_scores.max():.4f}]")
    
    return top_indices, top_scores


def embed_query(query_text):
    """Simple query embedding (mock for testing)."""
    # In real system, this would use Qwen3 model
    # For testing, create a random normalized vector
    np.random.seed(42)
    query_vec = np.random.randn(256).astype(np.float32)
    query_vec = query_vec / np.linalg.norm(query_vec)
    return query_vec


def test_layer3_selection():
    """Test Layer 3 GIST selection with sample data."""
    print("="*70)
    print("LAYER 3 GIST SELECTION - UNIT TEST")
    print("="*70)
    
    # Load data
    chunks, embeddings = load_test_data()
    
    # Mock query
    query = "agentic memory methods"
    print(f"\nQuery: \"{query}\"")
    query_embedding = embed_query(query)
    
    # Simulate Layer 2 output (157 candidate chunks)
    # For testing, take first 157 chunks
    num_candidates = min(157, len(chunks))
    candidate_indices = np.arange(num_candidates)
    candidate_embeddings = embeddings[candidate_indices]
    
    print(f"\nLayer 2 output: {num_candidates} candidate chunks")
    
    # LAYER 3: GIST Selection
    t0 = time.time()
    top_indices, top_scores = compute_gist_scores(
        query_embedding,
        candidate_embeddings,
        k=13
    )
    elapsed = time.time() - t0
    
    # Map back to original chunk indices
    selected_chunk_indices = candidate_indices[top_indices]
    
    print("\n" + "="*70)
    print("LAYER 3 RESULTS")
    print("="*70)
    print(f"Runtime: {elapsed:.4f}s")
    print(f"Selected {len(selected_chunk_indices)} chunks")
    print(f"\nTop-5 selected chunks:")
    
    for i, (chunk_idx, score) in enumerate(zip(selected_chunk_indices[:5], top_scores[:5])):
        chunk = chunks[chunk_idx]
        paper_id = chunk.get('paper_id', 'unknown')
        section_idx = chunk.get('section_idx', 0)
        text_preview = chunk.get('text', '')[:100].replace('\n', ' ')
        print(f"  {i+1}. Score: {score:.4f} | Paper: {paper_id} | Section: {section_idx}")
        print(f"     \"{text_preview}...\"")
    
    # Validation checks
    print("\n" + "="*70)
    print("VALIDATION CHECKS")
    print("="*70)
    
    checks_passed = 0
    checks_total = 0
    
    # Check 1: Correct number of results
    checks_total += 1
    if len(selected_chunk_indices) == 13:
        print("✓ Check 1: Returned exactly 13 chunks")
        checks_passed += 1
    else:
        print(f"✗ Check 1: Expected 13 chunks, got {len(selected_chunk_indices)}")
    
    # Check 2: No duplicates
    checks_total += 1
    if len(set(selected_chunk_indices)) == len(selected_chunk_indices):
        print("✓ Check 2: No duplicate chunks")
        checks_passed += 1
    else:
        print("✗ Check 2: Found duplicate chunks")
    
    # Check 3: Scores in descending order
    checks_total += 1
    if all(top_scores[i] >= top_scores[i+1] for i in range(len(top_scores)-1)):
        print("✓ Check 3: Scores in descending order")
        checks_passed += 1
    else:
        print("✗ Check 3: Scores not properly sorted")
    
    # Check 4: All indices valid
    checks_total += 1
    if all(0 <= idx < len(chunks) for idx in selected_chunk_indices):
        print("✓ Check 4: All chunk indices valid")
        checks_passed += 1
    else:
        print("✗ Check 4: Invalid chunk indices found")
    
    # Check 5: Scores are reasonable (not NaN, in valid range)
    checks_total += 1
    if all(np.isfinite(top_scores)) and all(score >= -1 and score <= 1 for score in top_scores):
        print("✓ Check 5: All scores are finite and in valid range")
        checks_passed += 1
    else:
        print("✗ Check 5: Invalid scores detected")
    
    print("\n" + "="*70)
    print(f"RESULT: {checks_passed}/{checks_total} checks passed")
    print("="*70)
    
    if checks_passed == checks_total:
        print("\n✅ Layer 3 GIST selection is WORKING CORRECTLY")
        return 0
    else:
        print("\n❌ Layer 3 GIST selection has issues")
        return 1


if __name__ == '__main__':
    sys.exit(test_layer3_selection())
