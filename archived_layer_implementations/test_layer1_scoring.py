"""
Diagnostic Test: Layer 1 Scoring Components

Tests each scoring component individually to find where scores break:
1. BM25 retrieval → should have bm25_score
2. Dense retrieval → should have dense_score  
3. RRF fusion → should have rrf_score
"""

from pathlib import Path
from pgvector_retriever import PGVectorRetriever, PGVectorConfig

def test_bm25_scoring():
    """Test BM25 retrieval scores"""
    print("\n" + "="*70)
    print("TEST 1: BM25 RETRIEVAL SCORES")
    print("="*70)
    
    config = PGVectorConfig(
        db_host="localhost",
        db_port=5432,
        db_name="langchain",
        db_user="langchain",
        db_password="langchain",
        table_name="arxiv_chunks",
        bm25_cache_path=Path("bm25_vocab.msgpack"),
        embedding_model="minishlab/M2V_base_output",
        embedding_dim=64
    )
    
    retriever = PGVectorRetriever(config)
    
    query = "agentic memory methods"
    print(f"\nQuery: '{query}'")
    print(f"Retrieving top 5 BM25 results...")
    
    results = retriever._retrieve_bm25(query, limit=5)
    
    print(f"\nRetrieved {len(results)} results:")
    print("-" * 70)
    
    for i, doc in enumerate(results, 1):
        print(f"\n{i}. Doc ID: {doc.doc_id}")
        print(f"   BM25 score: {doc.bm25_score}")
        print(f"   Dense score: {doc.dense_score}")
        print(f"   RRF score: {doc.rrf_score}")
        print(f"   Content: {doc.content[:80]}...")
    
    # Check if any scores are non-zero
    has_bm25 = any(doc.bm25_score and doc.bm25_score > 0 for doc in results)
    print(f"\n{'✅' if has_bm25 else '❌'} BM25 scores present: {has_bm25}")
    
    return results


def test_dense_scoring():
    """Test dense retrieval scores"""
    print("\n" + "="*70)
    print("TEST 2: DENSE RETRIEVAL SCORES")
    print("="*70)
    
    config = PGVectorConfig(
        db_host="localhost",
        db_port=5432,
        db_name="langchain",
        db_user="langchain",
        db_password="langchain",
        table_name="arxiv_chunks",
        bm25_cache_path=Path("bm25_vocab.msgpack"),
        embedding_model="minishlab/M2V_base_output",
        embedding_dim=64
    )
    
    retriever = PGVectorRetriever(config)
    
    query = "agentic memory methods"
    print(f"\nQuery: '{query}'")
    print(f"Retrieving top 5 dense results...")
    
    results = retriever._retrieve_dense(query, limit=5)
    
    print(f"\nRetrieved {len(results)} results:")
    print("-" * 70)
    
    for i, doc in enumerate(results, 1):
        print(f"\n{i}. Doc ID: {doc.doc_id}")
        print(f"   BM25 score: {doc.bm25_score}")
        print(f"   Dense score: {doc.dense_score}")
        print(f"   RRF score: {doc.rrf_score}")
        print(f"   Content: {doc.content[:80]}...")
    
    # Check if any scores are non-zero
    has_dense = any(doc.dense_score and doc.dense_score > 0 for doc in results)
    print(f"\n{'✅' if has_dense else '❌'} Dense scores present: {has_dense}")
    
    return results


def test_rrf_fusion():
    """Test RRF fusion scores"""
    print("\n" + "="*70)
    print("TEST 3: RRF FUSION SCORES")
    print("="*70)
    
    config = PGVectorConfig(
        db_host="localhost",
        db_port=5432,
        db_name="langchain",
        db_user="langchain",
        db_password="langchain",
        table_name="arxiv_chunks",
        bm25_cache_path=Path("bm25_vocab.msgpack"),
        embedding_model="minishlab/M2V_base_output",
        embedding_dim=64
    )
    
    retriever = PGVectorRetriever(config)
    
    query = "agentic memory methods"
    print(f"\nQuery: '{query}'")
    print(f"Testing RRF fusion...")
    
    # Get both pools
    bm25_results = retriever._retrieve_bm25(query, limit=15)
    dense_results = retriever._retrieve_dense(query, limit=15)
    
    print(f"\nBM25 pool: {len(bm25_results)} results")
    print(f"Dense pool: {len(dense_results)} results")
    
    # Fuse
    fused_results = retriever._rrf_fusion(bm25_results, dense_results)
    
    print(f"\nFused results: {len(fused_results)}")
    print("-" * 70)
    
    for i, doc in enumerate(fused_results[:5], 1):
        print(f"\n{i}. Doc ID: {doc.doc_id}")
        print(f"   BM25 score: {doc.bm25_score}")
        print(f"   BM25 rank: {doc.bm25_rank}")
        print(f"   Dense score: {doc.dense_score}")
        print(f"   Dense rank: {doc.dense_rank}")
        print(f"   RRF score: {doc.rrf_score}")
        print(f"   Content: {doc.content[:80]}...")
    
    # Check if RRF scores are non-zero
    has_rrf = any(doc.rrf_score and doc.rrf_score > 0 for doc in fused_results)
    print(f"\n{'✅' if has_rrf else '❌'} RRF scores present: {has_rrf}")
    
    return fused_results


def main():
    print("\n" + "="*70)
    print("LAYER 1 SCORING DIAGNOSTIC")
    print("="*70)
    
    try:
        # Test 1: BM25
        bm25_results = test_bm25_scoring()
        
        # Test 2: Dense
        dense_results = test_dense_scoring()
        
        # Test 3: RRF
        rrf_results = test_rrf_fusion()
        
        # Summary
        print("\n" + "="*70)
        print("DIAGNOSTIC SUMMARY")
        print("="*70)
        
        bm25_ok = any(doc.bm25_score and doc.bm25_score > 0 for doc in bm25_results)
        dense_ok = any(doc.dense_score and doc.dense_score > 0 for doc in dense_results)
        rrf_ok = any(doc.rrf_score and doc.rrf_score > 0 for doc in rrf_results)
        
        print(f"\nBM25 scoring:  {'✅ PASS' if bm25_ok else '❌ FAIL'}")
        print(f"Dense scoring: {'✅ PASS' if dense_ok else '❌ FAIL'}")
        print(f"RRF scoring:   {'✅ PASS' if rrf_ok else '❌ FAIL'}")
        
        if not bm25_ok:
            print("\n⚠️  BM25 scores are all zero/None - check _retrieve_bm25()")
        if not dense_ok:
            print("\n⚠️  Dense scores are all zero/None - check _retrieve_dense()")
        if not rrf_ok:
            print("\n⚠️  RRF scores are all zero/None - check _rrf_fusion()")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
