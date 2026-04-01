"""
Simple Test for GIST Three-Layer Hierarchical Retriever

This script demonstrates how to use the ThreeLayerGISTRetriever.

Requirements:
- Data files (chunks, embeddings, triplets, BM25 indexes)
- sentence-transformers library for ColBERT and Cross-Encoder
- rank-bm25 library

If data files are missing, this will show a helpful error message.
"""

import sys
from pathlib import Path

def test_gist_retriever():
    """Test GIST retriever with sample query."""
    
    # Import requirements
    try:
        from three_layer_gist_retriever import ThreeLayerGISTRetriever
        import msgpack
        from rank_bm25 import BM25Okapi
        print("✓ Imports successful")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("\nInstall with:")
        print("  pip install msgpack-python rank-bm25 sentence-transformers")
        return False
    
    # Define data file paths (using available files)
    data_config = {
        'chunks_path': 'checkpoints/chunks.msgpack',
        'chunk_embeddings_m2v_path': 'checkpoints/chunk_embeddings_64d.msgpack',  # M2V 64d embeddings
        'chunk_embeddings_qwen3_path': 'checkpoints/chunk_embeddings_qwen3.msgpack',
        'triplets_path': 'triplet_checkpoints_full/stage4_lemmatized.msgpack',  # Using lemmatized triplets
        'chunk_to_triplets_path': 'checkpoints/chunk_to_triplets.msgpack',
        'triplet_to_chunks_path': 'checkpoints/triplet_to_chunks.msgpack',
        'triplet_bm25_path': 'checkpoints/triplet_bm25_index.msgpack',
        'bm25_lemmatized_path': 'checkpoints/chunk_bm25_sparse.msgpack'  # BM25 index
    }
    
    # Check if data files exist
    print("\nChecking for data files...")
    missing_files = []
    for name, path in data_config.items():
        if not Path(path).exists():
            missing_files.append(f"  - {name}: {path}")
            print(f"❌ Missing: {path}")
        else:
            print(f"✓ Found: {path}")
    
    if missing_files:
        print(f"\n❌ Missing {len(missing_files)} required data file(s):")
        for f in missing_files:
            print(f)
        print("\nTo generate these files, run:")
        print("  python build_arxiv_graph_batched.py")
        print("  python equidistant_chunking.py")
        print("  (and other preprocessing scripts)")
        return False
    
    # Load BM25 index - rebuild from chunks since sparse matrix format exists
    print("\nBuilding BM25 index from chunks...")
    try:
        with open(data_config['chunks_path'], 'rb') as f:
            chunks = msgpack.load(f, raw=False)
        print(f"  Loaded {len(chunks):,} chunks")
        
        # Simple tokenization (lowercase alphanumeric, min 2 chars)
        import re
        def tokenize(text):
            """Simple lowercase alphanumeric tokenization"""
            return re.findall(r'\b[a-z0-9]{2,}\b', text.lower())
        
        print("  Tokenizing corpus...")
        corpus_tokens = [tokenize(chunk['text']) for chunk in chunks]
        
        print("  Building BM25Okapi index...")
        bm25_index = BM25Okapi(corpus_tokens)
        print("✓ BM25 index built")
    except Exception as e:
        print(f"❌ Failed to build BM25: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Initialize retriever
    print("\nInitializing GIST Three-Layer Retriever...")
    try:
        retriever = ThreeLayerGISTRetriever(
            chunks_path=data_config['chunks_path'],
            chunk_embeddings_m2v_path=data_config['chunk_embeddings_m2v_path'],
            chunk_embeddings_qwen3_path=data_config['chunk_embeddings_qwen3_path'],
            triplets_path=data_config['triplets_path'],
            chunk_to_triplets_path=data_config['chunk_to_triplets_path'],
            triplet_to_chunks_path=data_config['triplet_to_chunks_path'],
            triplet_bm25_path=data_config['triplet_bm25_path'],
            bm25_lemmatized_index=bm25_index,
            top_k=13,
            colbert_model_name="bert-base-uncased",
            cross_encoder_model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        print("✓ Retriever initialized successfully!")
    except Exception as e:
        print(f"❌ Failed to initialize retriever: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test query
    print("\n" + "="*60)
    print("RUNNING TEST QUERY")
    print("="*60)
    
    test_query = "What are agentic memory approaches?"
    print(f"\nQuery: '{test_query}'")
    print(f"Expected output: {retriever.config.top_k} papers\n")
    
    try:
        results = retriever.retrieve(test_query)
        
        print(f"\n{'='*60}")
        print(f"RESULTS ({len(results)} papers)")
        print(f"{'='*60}\n")
        
        for i, paper in enumerate(results, 1):
            print(f"{i}. Paper ID: {paper['paper_id']}")
            print(f"   RRF Score: {paper['rrf_score']:.4f}")
            print(f"   Sections: {len(paper['sections'])}")
            
            # Show top section
            if paper['sections']:
                top_section = paper['sections'][0]
                print(f"   Top Section: idx={top_section.get('section_idx', 'N/A')}, "
                      f"score={top_section.get('rrf_score', 0):.4f}")
                # Show snippet
                text = top_section.get('full_text', top_section.get('content', ''))
                snippet = text[:200].replace('\n', ' ') + '...' if len(text) > 200 else text
                print(f"   Snippet: {snippet}")
            print()
        
        print(f"✓ Test completed successfully!")
        print(f"\nRetrieval breakdown:")
        print(f"  - Layer 1 seeds: {retriever.config.layer1_seeds}")
        print(f"  - Layer 2 expansions: {retriever.config.layer2_expand}")
        print(f"  - Layer 2 sections: {retriever.config.layer2_sections}")
        print(f"  - Layer 3 output: {retriever.config.layer3_output} papers")
        
        return True
        
    except Exception as e:
        print(f"❌ Query failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("="*60)
    print("GIST THREE-LAYER HIERARCHICAL RETRIEVER TEST")
    print("="*60)
    print("\nFeature: GIST Three-Layer Hierarchical Retriever")
    print("Feature ID: 13 (in feature_catalog.sqlite3)")
    print("Implementation: three_layer_gist_retriever.py (1,245 lines)")
    print("\nArchitecture:")
    print("  Layer 1: BM25 + M2V embeddings → GIST → RRF → seeds")
    print("  Layer 2: Graph BM25 + Qwen3 → GIST → RRF → sections (exclude L1)")
    print("  Layer 3: ColBERT + Cross-Encoder → GIST → RRF → papers (walk-down)")
    print("\n" + "="*60 + "\n")
    
    success = test_gist_retriever()
    
    if success:
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Tests failed - see errors above")
        sys.exit(1)
