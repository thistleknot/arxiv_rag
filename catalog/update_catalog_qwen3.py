"""
Update Feature Catalog with Model2Vec Qwen3 Embeddings and Three-Layer Architecture

Adds two new features:
1. Model2Vec Qwen3 chunk embeddings (256d)
2. Three-layer golden ratio retrieval architecture
"""

import sqlite3
from datetime import datetime
from pathlib import Path


def add_qwen3_embedding_feature(db_path='feature_catalog.sqlite3'):
    """Add Model2Vec Qwen3 embedding feature"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check if feature already exists
    cursor.execute(
        "SELECT id FROM features WHERE name = ?",
        ("Model2Vec Qwen3 Chunk Embeddings",)
    )
    existing = cursor.fetchone()
    
    if existing:
        print(f"⚠️  Feature already exists: Model2Vec Qwen3 Chunk Embeddings (ID: {existing[0]})")
        conn.close()
        return existing[0]
    
    # Insert feature
    cursor.execute("""
    INSERT INTO features (name, description, status, created_at, updated_at)
    VALUES (?, ?, ?, ?, ?)
    """, (
        "Model2Vec Qwen3 Chunk Embeddings",
        """
        Embed all chunks with Model2Vec distilled Qwen3-Embedding-0.6B (256d).
        
        Captures co-occurrence patterns learned during LLM pre-training,
        enabling associative graph expansion beyond pure semantic similarity.
        
        Files:
        - embed_chunks_qwen3.py: Embedding script
        - checkpoints/chunk_embeddings_qwen3.msgpack: Output embeddings
        - python_compare.py: Comparison with original Qwen3-Embedding-0.6B
        
        Architecture:
        - Model: Model2Vec distilled Qwen3 (768d → 256d)
        - Source: ./qwen3_static_embeddings
        - Output: 256-dimensional static embeddings
        - Speed: ~128 chunks/sec (CPU, no GPU needed)
        - Memory: ~50MB model + ~40MB embeddings (161k chunks × 256d × 4 bytes)
        
        Integration:
        - Used in three_layer_retriever.py Layer 2
        - Retrieves via cosine similarity
        - Merged with Graph BM25 using RRF
        
        Key Advantage:
        - Jina (Layer 1): Semantic similarity
        - Qwen3 (Layer 2): Co-occurrence patterns
        - Combined: Explicit (graph) + implicit (LLM) relationships
        """,
        'TODO',
        datetime.now().isoformat(),
        datetime.now().isoformat()
    ))
    
    conn.commit()
    feature_id = cursor.lastrowid
    print(f"✅ Added feature: Model2Vec Qwen3 Chunk Embeddings (ID: {feature_id})")
    
    conn.close()
    return feature_id


def add_three_layer_architecture_feature(db_path='feature_catalog.sqlite3'):
    """Add three-layer retrieval architecture feature"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check if feature already exists
    cursor.execute(
        "SELECT id FROM features WHERE name = ?",
        ("Three-Layer Golden Ratio Retrieval",)
    )
    existing = cursor.fetchone()
    
    if existing:
        print(f"⚠️  Feature already exists: Three-Layer Golden Ratio Retrieval (ID: {existing[0]})")
        conn.close()
        return existing[0]
    
    # Insert feature
    cursor.execute("""
    INSERT INTO features (name, description, status, created_at, updated_at)
    VALUES (?, ?, ?, ?, ?)
    """, (
        "Three-Layer Golden Ratio Retrieval",
        """
        Progressive refinement architecture with φ⁻¹ scaling at each layer.
        
        Layer 1: Hybrid Gist RRF (BM25 + Jina)
        - Retrieve: top_k²
        - Output: φ⁻¹×top_k² (golden ratio reduction)
        - Purpose: Initial semantic + keyword matching
        
        Layer 2: Graph + Co-occurrence Expansion
        - Retrieve: 2×φ⁻¹×top_k² (Graph BM25 + Model2Vec Qwen3)
        - Output: φ⁻¹×top_k² (same as Layer 1 output)
        - Purpose: Expand to concept neighborhoods
        
        Layer 3: Cross-encoder Reranking
        - Input: φ⁻¹×top_k²
        - Output: top_k
        - Purpose: Fine-grained relevance scoring
        
        Files:
        - three_layer_retriever.py: Complete implementation
        - embed_chunks_qwen3.py: Qwen3 embedding preparation
        
        Example (top_k=10):
        - Layer 1: 100 retrieved → 62 output
        - Layer 2: 124 retrieved → 62 output  
        - Layer 3: 62 input → 10 output
        
        Key Innovation:
        - Layer 2 combines explicit (Graph BM25) and implicit (Qwen3 co-occurrence)
        - Jina (Layer 1) = semantic similarity
        - Qwen3 (Layer 2) = co-occurrence patterns
        - Combined = comprehensive concept expansion
        
        Why Qwen3 Matters:
        - LLM pre-training learns co-occurrence (next-token prediction)
        - Distilled to 256d for fast CPU inference
        - Captures "what appears with" not just "what means similar"
        - Perfect complement to graph-based expansion
        
        Dependencies:
        - gist_retriever.py (Layer 1 - existing)
        - build_triplet_bm25_batched.py (Graph BM25 index)
        - embed_chunks_qwen3.py (Qwen3 embeddings)
        - ColBERTv2/cross-encoder (Layer 3 - optional)
        """,
        'TODO',
        datetime.now().isoformat(),
        datetime.now().isoformat()
    ))
    
    conn.commit()
    feature_id = cursor.lastrowid
    print(f"✅ Added feature: Three-Layer Golden Ratio Retrieval (ID: {feature_id})")
    
    conn.close()
    return feature_id


def add_architectural_decision(db_path='feature_catalog.sqlite3'):
    """Log architectural decision for Qwen3 vs Jina"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
    INSERT INTO architectural_decisions (
        decision, 
        rationale, 
        before_state, 
        after_state,
        decision_timestamp
    ) VALUES (?, ?, ?, ?, ?)
    """, (
        "Use Model2Vec Qwen3 for Layer 2 Graph Expansion",
        """
        Architectural choice to use Model2Vec Qwen3 (256d) for Layer 2 instead of reusing Jina.
        
        Rationale:
        1. LLM-learned co-occurrence patterns
           - Qwen3 pre-trained as LLM (next-token prediction)
           - Captures "what appears with X" not just "what means similar to X"
           - Example: "attention mechanism" → "Transformer", "BERT", "query-key-value"
        
        2. Complementary to Jina (Layer 1)
           - Jina: Pure semantic similarity (trained on sentence pairs)
           - Qwen3: Co-occurrence patterns (learned from LLM objective)
           - Combined: Covers both explicit meaning and implicit associations
        
        3. Perfect for scientific knowledge graphs
           - Need: Find concepts discussed together in papers
           - Not just: Find synonyms of query terms
           - Qwen3: Captures domain-specific co-occurrence from arXiv training
        
        4. Performance characteristics
           - 256d static embeddings (distilled from 768d)
           - CPU inference only (no GPU needed)
           - Fast cosine similarity search
           - Complements graph traversal (explicit vs implicit)
        
        Colleague's insight: "LLMs are already trained on co-occurrence"
        → This is true - and Qwen3 retains that signal while being fast
        
        Alternative considered: Reuse Jina for Layer 2
        → Rejected: Would duplicate Layer 1 signal, missing co-occurrence patterns
        """,
        "Layer 2 uses same embeddings as Layer 1 (Jina only)",
        "Layer 2 uses Model2Vec Qwen3 for co-occurrence expansion",
        datetime.now().isoformat()
    ))
    
    conn.commit()
    decision_id = cursor.lastrowid
    print(f"✅ Logged architectural decision (ID: {decision_id})")
    
    conn.close()
    return decision_id


def main():
    db_path = 'feature_catalog.sqlite3'
    
    # Check if database exists
    if not Path(db_path).exists():
        print(f"⚠️  Database not found: {db_path}")
        print("Creating database...")
        from feature_catalog import init_feature_catalog
        init_feature_catalog(db_path)
    
    print("="*60)
    print("UPDATE FEATURE CATALOG: MODEL2VEC QWEN3 + 3-LAYER RETRIEVAL")
    print("="*60)
    print()
    
    # Add features
    qwen3_id = add_qwen3_embedding_feature(db_path)
    three_layer_id = add_three_layer_architecture_feature(db_path)
    
    print()
    
    # Add architectural decision
    decision_id = add_architectural_decision(db_path)
    
    print()
    print("="*60)
    print("CATALOG UPDATED")
    print("="*60)
    print(f"Features added:")
    print(f"  - Model2Vec Qwen3 Chunk Embeddings (ID: {qwen3_id})")
    print(f"  - Three-Layer Golden Ratio Retrieval (ID: {three_layer_id})")
    print(f"Architectural decision logged (ID: {decision_id})")
    print()
    
    # Show next steps
    print("Next steps:")
    print()
    print("1. Embed chunks with Model2Vec Qwen3:")
    print("   python embed_chunks_qwen3.py \\")
    print("     --chunks checkpoints/chunks.msgpack \\")
    print("     --output checkpoints/chunk_embeddings_qwen3.msgpack")
    print()
    print("2. Update feature status after embedding:")
    print(f"   python -c \"from feature_catalog import update_feature_status; update_feature_status({qwen3_id}, 'DONE')\"")
    print()
    print("3. Implement three_layer_retriever.py Layer 2 methods:")
    print("   - _retrieve_graph_bm25()")
    print("   - _map_triplets_to_chunks()")
    print()
    print("4. Test three-layer retrieval:")
    print("   python three_layer_retriever.py")
    print()
    print("5. Update three-layer feature after testing:")
    print(f"   python -c \"from feature_catalog import update_feature_status; update_feature_status({three_layer_id}, 'DONE')\"")


if __name__ == '__main__':
    main()
