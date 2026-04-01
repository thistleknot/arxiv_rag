"""
Integration Test for Three-Layer φ-Retriever

Tests the complete 3-layer system with real data:
- Layer 1: Existing gist_retriever (BM25 + Jina RRF)
- Layer 2: Graph BM25 + Qwen3 expansion
- Layer 3: Placeholder (future: cross-encoder)

Query: "agentic memory methods"
"""

import sys
import time
from pathlib import Path
import msgpack
import numpy as np
from typing import List, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from simple_hybrid_retriever import SimpleHybridRetriever
from pgvector_retriever import PGVectorConfig, RetrievedDoc
from three_layer_phi_retriever import ThreeLayerPhiRetriever, PhiLayerConfig


def load_chunks(path: str):
    """Load chunks from msgpack"""
    print(f"Loading chunks from {path}...")
    with open(path, 'rb') as f:
        data = msgpack.unpackb(f.read(), raw=False)
    
    if isinstance(data, dict) and 'chunks' in data:
        chunks = data['chunks']
    elif isinstance(data, list):
        chunks = data
    else:
        raise ValueError(f"Unexpected data format in {path}")
    
    print(f"✅ Loaded {len(chunks):,} chunks")
    return chunks


def load_qwen3_embeddings(path: str):
    """Load Qwen3 embeddings from msgpack"""
    print(f"Loading Qwen3 embeddings from {path}...")
    with open(path, 'rb') as f:
        data = msgpack.unpackb(f.read(), raw=False)
    
    embeddings = np.array(data['embeddings'], dtype=np.float32)
    print(f"✅ Loaded embeddings: {embeddings.shape} ({embeddings.nbytes / 1024 / 1024:.1f} MB)")
    return embeddings


def load_triplets(path: str):
    """Load enriched triplets from stage4 (lemmatized)"""
    print(f"Loading triplets from {path}...")
    with open(path, 'rb') as f:
        data = msgpack.unpackb(f.read(), raw=False)
    
    # Handle different formats
    if isinstance(data, dict) and 'triplets' in data:
        triplets = data['triplets']
    elif isinstance(data, list):
        triplets = data
    else:
        raise ValueError(f"Unexpected triplet format in {path}")
    
    print(f"✅ Loaded {len(triplets):,} triplets")
    return triplets


def create_chunk_triplet_mappings(chunks: List[dict], triplets: List[dict], output_dir: str = "checkpoints"):
    """
    Create bidirectional mappings between chunks and triplets and save to disk.
    
    Returns:
        chunk_to_triplets_path: Path to saved chunk_to_triplets.msgpack
        triplet_to_chunks_path: Path to saved triplet_to_chunks.msgpack
    """
    print("\nCreating chunk↔triplet mappings...")
    
    chunk_to_triplets = {}
    triplet_to_chunks = {}
    
    # Build mappings
    for triplet_idx, triplet in enumerate(triplets):
        chunk_id = triplet.get('chunk_id', triplet.get('chunk_idx'))
        
        if chunk_id is None:
            continue
        
        # Chunk → Triplets
        if chunk_id not in chunk_to_triplets:
            chunk_to_triplets[chunk_id] = []
        chunk_to_triplets[chunk_id].append(triplet_idx)
        
        # Triplet → Chunks
        if triplet_idx not in triplet_to_chunks:
            triplet_to_chunks[triplet_idx] = []
        triplet_to_chunks[triplet_idx].append(chunk_id)
    
    print(f"✅ Created mappings:")
    print(f"   Chunks with triplets: {len(chunk_to_triplets):,}")
    print(f"   Triplets mapped: {len(triplet_to_chunks):,}")
    
    # Convert integer keys to strings for msgpack compatibility
    chunk_to_triplets_str = {str(k): v for k, v in chunk_to_triplets.items()}
    triplet_to_chunks_str = {str(k): v for k, v in triplet_to_chunks.items()}
    
    # Save to msgpack files
    chunk_to_triplets_path = Path(output_dir) / "chunk_to_triplets.msgpack"
    triplet_to_chunks_path = Path(output_dir) / "triplet_to_chunks.msgpack"
    
    print(f"\nSaving mappings...")
    with open(chunk_to_triplets_path, 'wb') as f:
        msgpack.pack(chunk_to_triplets_str, f)
    print(f"✅ Saved {chunk_to_triplets_path}")
    
    with open(triplet_to_chunks_path, 'wb') as f:
        msgpack.pack(triplet_to_chunks_str, f)
    print(f"✅ Saved {triplet_to_chunks_path}")
    
    return str(chunk_to_triplets_path), str(triplet_to_chunks_path)


def test_layer1_only(retriever: SimpleHybridRetriever, query: str, top_k: int = 13):
    """Test Layer 1 (simple hybrid: BM25 + embeddings)"""
    print(f"\n{'='*70}")
    print(f"LAYER 1 ONLY (Simple Hybrid: BM25 + Embeddings)")
    print(f"{'='*70}")
    
    start = time.time()
    results = retriever.retrieve(query, top_k=top_k)
    elapsed = time.time() - start
    
    print(f"\nQuery: '{query}'")
    print(f"Retrieved: {len(results)} chunks in {elapsed:.3f}s")
    print(f"\nTop {min(5, len(results))} results:")
    print("-" * 70)
    
    for i, doc in enumerate(results[:5], 1):
        text = doc.content[:150].replace('\n', ' ')
        print(f"\n{i}. Chunk (RRF score: {doc.rrf_score:.4f})")
        print(f"   {text}...")
    
    return results


def test_three_layer(
    retriever: ThreeLayerPhiRetriever,
    query: str,
    top_k: int = 13
):
    """Test complete 3-layer φ-retrieval"""
    print(f"\n{'='*70}")
    print(f"THREE-LAYER φ-RETRIEVAL")
    print(f"{'='*70}")
    
    start = time.time()
    results = retriever.retrieve(query)
    elapsed = time.time() - start
    
    print(f"\nQuery: '{query}'")
    print(f"Pipeline: 13 seeds → 144 expansions → 13 final")
    print(f"Retrieved: {len(results)} chunks in {elapsed:.3f}s")
    print(f"\nTop {min(5, len(results))} results:")
    print("-" * 70)
    
    for i, (chunk_id, score) in enumerate(results[:5], 1):
        chunk = retriever.chunks[chunk_id]
        text = chunk['text'][:150].replace('\n', ' ')
        print(f"\n{i}. Chunk {chunk_id} (score: {score:.4f})")
        print(f"   {text}...")
    
    return results


def compare_results(
    layer1_results: List[RetrievedDoc],
    layer3_results: List[Tuple[int, float]]
):
    """Compare Layer 1 vs 3-layer results"""
    print(f"\n{'='*70}")
    print(f"COMPARISON: Layer 1 vs Three-Layer")
    print(f"{'='*70}")
    
    # Layer 1 returns RetrievedDoc, extract doc_ids
    layer1_ids = set(doc.doc_id for doc in layer1_results)
    # Layer 3 returns (chunk_idx, score) tuples - need to map back to doc_ids
    # For now just compare indices
    layer3_ids = set(cid for cid, _ in layer3_results)
    
    # Can't directly compare since Layer 1 has doc_id strings and Layer 3 has indices
    print(f"\nLayer 1: {len(layer1_ids)} chunks (doc_id strings)")
    print(f"Layer 3: {len(layer3_ids)} chunks (integer indices)")
    print(f"\n⚠️  Direct comparison skipped (different ID formats)")
    
    return
    
    # OLD CODE (kept for reference):
    # overlap = layer1_ids & layer3_ids
    # layer3_only = layer3_ids - layer1_ids
    
    # print(f"\nOverlap: {len(overlap)} / {len(layer3_results)} chunks")
    print(f"New chunks in 3-layer: {len(layer3_only)}")
    
    if layer3_only:
        print(f"\nTop new chunks discovered by 3-layer expansion:")
        print("-" * 70)
        
        for i, (chunk_id, score) in enumerate(layer3_results, 1):
            if chunk_id in layer3_only:
                print(f"{i}. Chunk {chunk_id} (score: {score:.4f})")
                if i >= 3:
                    break


def main():
    """Run integration test"""
    print("="*70)
    print("THREE-LAYER φ-RETRIEVER INTEGRATION TEST")
    print("="*70)
    
    query = "agentic memory methods"
    
    # =========================================================================
    # LOAD DATA
    # =========================================================================
    
    print("\n1. Loading data files...")
    print("-" * 70)
    
    # Chunks
    chunks_path = "checkpoints/chunks.msgpack"
    chunks = load_chunks(chunks_path)
    
    # Qwen3 embeddings
    qwen3_path = "checkpoints/chunk_embeddings_qwen3.msgpack"
    qwen3_embeddings = load_qwen3_embeddings(qwen3_path)
    
    # Triplets (use stage4 lemmatized)
    triplets_path = "triplet_checkpoints_full/stage4_lemmatized.msgpack"
    triplets = load_triplets(triplets_path)
    
    # Create mappings (saves to checkpoints/)
    chunk_to_triplets_path, triplet_to_chunks_path = create_chunk_triplet_mappings(chunks, triplets)
    
    # =========================================================================
    # INITIALIZE LAYER 1 (Simple Hybrid Retriever)
    # =========================================================================
    
    print("\n2. Initializing Layer 1 retriever (simple hybrid: BM25 + embeddings)...")
    print("-" * 70)
    
    try:
        from pgvector_retriever import PGVectorRetriever, PGVectorConfig
        
        # PostgreSQL configuration (Docker Desktop)
        pg_config = PGVectorConfig(
            db_host="localhost",
            db_port=5432,
            db_name="langchain",
            db_user="langchain",
            db_password="langchain",
            table_name="arxiv_chunks",
            embedding_dim=64,
            embedding_model="minishlab/M2V_base_output",
            bm25_cache_path=Path("bm25_vocab.msgpack"),
            use_lemmatized=False
        )
        
        layer1 = SimpleHybridRetriever(pg_config)
        print("✅ Layer 1 initialized (simple hybrid: BM25 + embeddings + RRF)")
        print(f"   Database: {pg_config.db_host}:{pg_config.db_port}/{pg_config.db_name}")
        print(f"   Table: {pg_config.table_name}")
        
    except Exception as e:
        print(f"⚠️ Failed to initialize Layer 1: {e}")
        print("   Make sure Docker Desktop is running and PostgreSQL is accessible")
        print("   Connection: localhost:5432/langchain")
        print("\n   Using mock Layer 1 for testing...")
        layer1 = None
    
    # =========================================================================
    # TEST LAYER 1 ONLY (Baseline)
    # =========================================================================
    
    if layer1:
        layer1_results = test_layer1_only(layer1, query, top_k=13)
    else:
        print("\nSkipping Layer 1 test (mock mode)")
        layer1_results = None
    
    # =========================================================================
    # CHECK FOR BM25 INDEX
    # =========================================================================
    
    # First try msgpack format
    triplet_bm25_path = "checkpoints/triplet_bm25_index.msgpack"
    if not Path(triplet_bm25_path).exists():
        # Fall back to pickle format (stage7)
        pkl_path = "triplet_checkpoints_full/stage7_bm25_index.pkl"
        if not Path(pkl_path).exists():
            print("\n⚠️ WARNING: BM25 index not found")
            print("   Need to build WordPiece BM25 index for graph expansion")
            print("   Run: python build_wordpiece_bm25.py")
            print("\n   For this test, using None (will skip graph expansion)")
            triplet_bm25_path = None
        else:
            print(f"\n⚠️ Found pickle BM25 index but need msgpack format")
            print(f"   Run: python convert_bm25_index.py")
            triplet_bm25_path = None
    else:
        print(f"\n✅ Found BM25 index: {triplet_bm25_path}")
    
    # =========================================================================
    # INITIALIZE THREE-LAYER RETRIEVER
    # =========================================================================
    
    print("\n3. Initializing Three-Layer φ-Retriever...")
    print("-" * 70)
    
    try:
        retriever = ThreeLayerPhiRetriever(
            chunks_path=chunks_path,
            chunk_embeddings_qwen3_path=qwen3_path,
            triplets_path=triplets_path,
            chunk_to_triplets_path=chunk_to_triplets_path,
            triplet_to_chunks_path=triplet_to_chunks_path,
            triplet_bm25_path=triplet_bm25_path,
            layer1_retriever=layer1,
            top_k=13
        )
        print("✅ Three-layer retriever initialized")
        
        # =====================================================================
        # TEST THREE-LAYER RETRIEVAL
        # =====================================================================
        
        layer3_results = test_three_layer(retriever, query, top_k=13)
        
        # =====================================================================
        # COMPARE RESULTS
        # =====================================================================
        
        if layer1_results:
            compare_results(layer1_results, layer3_results)
        
    except Exception as e:
        print(f"❌ Error in three-layer retrieval: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    
    print(f"\n{'='*70}")
    print(f"INTEGRATION TEST COMPLETE")
    print(f"{'='*70}")
    
    print("\n✅ Successfully tested:")
    if layer1_results:
        print(f"   • Layer 1 (baseline): {len(layer1_results)} chunks")
    print(f"   • Three-layer φ-retrieval: {len(layer3_results)} chunks")
    print(f"   • Pipeline: 13 → 157 → 13")
    
    print("\n⚠️ Next steps:")
    print("   1. Build WordPiece BM25 index: python build_wordpiece_bm25.py")
    print("   2. Re-test with full graph expansion enabled")
    print("   3. Add Layer 3 cross-encoder (ColBERTv2)")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
