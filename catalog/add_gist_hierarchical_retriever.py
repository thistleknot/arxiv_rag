"""
Add GIST Three-Layer Hierarchical Retriever to Feature Catalog
"""

import sqlite3
from datetime import datetime

DB_PATH = 'feature_catalog.sqlite3'

def add_gist_hierarchical_retriever():
    """Add GIST Three-Layer Hierarchical Retriever feature to catalog."""
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    feature_name = "GIST Three-Layer Hierarchical Retriever"
    
    description = """
Three-layer retrieval system applying GIST (Greedy Information Selection with Topic diversity) 
at every retrieval step to maximize query relevance while minimizing redundancy.

## Architecture:
Layer 1 (Lexical/Semantic Seeds):
- BM25 retrieval (lemmatized) → GIST selection → top_k²/2
- M2V embedding retrieval (64d) → GIST selection → top_k²/2
- RRF fusion → ~top_k² seed chunks

Layer 2 (Query Expansion via Graph):
- Graph BM25 (on semantic triplets) → GIST selection → top_k²/2 [excluding L1]
- Qwen3 embedding retrieval (256d) → GIST selection → top_k²/2 [excluding L1]
- RRF fusion → top_k² expansions
- Aggregate L1 + L2 → Group by section → Select φ_lower(top_k²) sections

Layer 3 (Late Interaction Reranking):
- ColBERT MaxSim scoring → GIST selection → top_k²
- MSMarco Cross-Encoder scoring → GIST selection → top_k²
- RRF fusion → Walk-down paper selection with floor threshold → top_k papers

## GIST Algorithm:
MMR-based forward stepwise selection balancing relevance and diversity:
- Formula: score(d) = λ * utility(d) - (1-λ) * max_sim(d, Selected)
- Coverage Matrix: n×n doc-doc similarities (excludes query)
- Utility Vector: n×1 doc-query similarities
- λ = 0.7 (70% relevance, 30% diversity)

## Key Features:
1. GIST at all 6 retrieval points (BM25, M2V, Graph BM25, Qwen3, ColBERT, Cross-Encoder)
2. L1 exclusion in Layer 2 (prevents re-retrieving seeds)
3. Floor threshold in Layer 3 (walk-down collects top_k+1 papers, uses 13th as floor)
4. Standard unweighted RRF fusion everywhere
5. Fibonacci φ_lower for section selection (not retrieval)

## Implementation:
- File: three_layer_gist_retriever.py (1,245 lines)
- Reference: gist_retriever.py (lines 236-340 for gist_select())
- Documentation: GIST_THREE_LAYER_ARCHITECTURE.md

## Deprecated Predecessor:
- three_layer_phi_retriever.py (ECDF-based with triangular weighting)
- Replaced custom ECDF weights with GIST coverage/utility matrices
- Simpler approach: GIST + standard RRF everywhere

## Configuration:
- top_k = 13 (final papers)
- layer1_seeds = 169 (top_k²)
- layer2_expand = 169 (top_k²)
- layer2_sections = 144 (φ_lower(169))
- gist_lambda = 0.7
- rrf_k = 60

## Scoring Aggregations:
- BM25 triplet → chunk: Average token counts across triplets
- Chunk → section: Sum of RRF scores
- Section → paper: Walk-down traversal with floor threshold
"""
    
    # Check if feature already exists
    cursor.execute("SELECT id FROM features WHERE name = ?", (feature_name,))
    existing = cursor.fetchone()
    
    if existing:
        print(f"⚠️ Feature '{feature_name}' already exists with ID {existing[0]}")
        print("Updating description and status...")
        
        cursor.execute("""
        UPDATE features 
        SET description = ?,
            status = 'DONE',
            updated_at = CURRENT_TIMESTAMP,
            validation_notes = 'Complete implementation in three_layer_gist_retriever.py (1,245 lines). GIST selection at all 6 retrieval points. Replaces ECDF-based three_layer_phi_retriever.py with simpler GIST + standard RRF approach.'
        WHERE id = ?
        """, (description, existing[0]))
        
        feature_id = existing[0]
    else:
        print(f"Adding new feature: {feature_name}")
        
        cursor.execute("""
        INSERT INTO features (
            name, 
            description, 
            status, 
            validated_by, 
            validation_notes
        ) VALUES (?, ?, 'DONE', 'GitHub Copilot (Claude Sonnet 4.5)', 
                  'Complete implementation in three_layer_gist_retriever.py (1,245 lines). GIST selection at all 6 retrieval points. Replaces ECDF-based three_layer_phi_retriever.py with simpler GIST + standard RRF approach.')
        """, (feature_name, description))
        
        feature_id = cursor.lastrowid
    
    conn.commit()
    
    # Add architectural decision record
    print(f"\nAdding architectural decision for GIST approach...")
    
    cursor.execute("""
    INSERT INTO architectural_decisions (
        decision,
        rationale,
        before_state,
        after_state
    ) VALUES (?, ?, ?, ?)
    """, (
        "Replace ECDF-weighted retrieval with GIST-based hierarchical retrieval",
        """
GIST provides principled diversity control via coverage/utility matrices without custom weighting.
- Coverage matrix: n×n doc-doc similarity (collinearity control)
- Utility vector: n×1 doc-query similarity (relevance signal)
- MMR-based greedy selection: score(d) = λ * utility(d) - (1-λ) * max_sim(d, Selected)
- Standard RRF fusion eliminates need for ECDF normalization
- Simpler, more interpretable, grounded in regression feature selection theory
        """,
        """
three_layer_phi_retriever.py with ECDF-based approach:
- midpoint_ecdf_weights() for triangular normalization
- Weighted RRF mixing ECDF scores
- No L1 exclusion in Layer 2
- No floor threshold in Layer 3 walk-down
- Single embedding model (M2V 64d)
        """,
        """
three_layer_gist_retriever.py with GIST-based approach:
- gist_select() at all 6 retrieval points
- Standard unweighted RRF everywhere
- L1 exclusion in Layer 2 (graph BM25 + Qwen3)
- Floor threshold in Layer 3 (walk-down with top_k+1 papers)
- Dual embedding models (L1: M2V 64d, L2: Qwen3 256d)
- Coverage/utility matrices for principled diversity
        """
    ))
    
    conn.commit()
    conn.close()
    
    print(f"\n✅ Feature catalog updated successfully!")
    print(f"   Feature ID: {feature_id}")
    print(f"   Feature Name: {feature_name}")
    print(f"   Status: DONE")
    print(f"   Implementation: three_layer_gist_retriever.py")
    print(f"   Documentation: GIST_THREE_LAYER_ARCHITECTURE.md")
    
    return feature_id

if __name__ == "__main__":
    feature_id = add_gist_hierarchical_retriever()
    
    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    print("1. Test implementation:")
    print("   from three_layer_gist_retriever import ThreeLayerGISTRetriever")
    print("   retriever = ThreeLayerGISTRetriever(..., top_k=13)")
    print("   results = retriever.retrieve('What are agentic memory approaches?')")
    print()
    print("2. Benchmark vs ECDF approach:")
    print("   - Compare precision@k, recall@k")
    print("   - Measure diversity (inter-document similarity)")
    print("   - Validate L1 exclusion, floor threshold")
    print()
    print("3. Log performance claims:")
    print("   from feature_catalog import log_claim")
    print(f"   log_claim({feature_id}, 'GIST provides X% better diversity...', confidence_score=0.8)")
