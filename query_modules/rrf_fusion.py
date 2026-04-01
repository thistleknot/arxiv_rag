"""
RRF (Reciprocal Rank Fusion) Module

Combines rankings from multiple retrieval methods using reciprocal rank fusion.

Reference: gist_retriever.py lines 1549-1593
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from collections import defaultdict


def compute_rrf_score(ranks: List[int], k: int = 60) -> float:
    """
    Compute RRF score from multiple ranks.
    
    Formula: RRF = Σ 1/(k + rank_i)
    
    Args:
        ranks: List of ranks (0-indexed) from different sources
        k: Constant (default: 60)
    
    Returns:
        Combined RRF score
    """
    return sum(1.0 / (k + rank + 1) for rank in ranks)


def rrf_fusion(
    results_list: List[List[Dict[str, Any]]],
    id_field: str = 'chunk_id',
    k: int = 60,
    preserve_scores: bool = True
) -> List[Dict[str, Any]]:
    """
    Fuse multiple ranked lists using RRF.
    
    Args:
        results_list: List of ranked result lists
        id_field: Field name for document ID
        k: RRF constant (default: 60)
        preserve_scores: Keep original scores from each source
    
    Returns:
        Fused results sorted by RRF score
    
    Example:
        bm25_results = [{'chunk_id': 'a', 'score': 0.9}, ...]
        dense_results = [{'chunk_id': 'b', 'score': 0.8}, ...]
        fused = rrf_fusion([bm25_results, dense_results])
    """
    # Track ranks and scores for each document
    doc_data = defaultdict(lambda: {
        'ranks': [],
        'source_scores': {},
        'doc': None
    })
    
    # Collect ranks from each source
    for source_idx, results in enumerate(results_list):
        for rank, doc in enumerate(results):
            doc_id = doc[id_field]
            doc_data[doc_id]['ranks'].append(rank)
            
            if preserve_scores:
                source_name = f'source_{source_idx}'
                doc_data[doc_id]['source_scores'][source_name] = doc.get('score', 0.0)
            
            if doc_data[doc_id]['doc'] is None:
                doc_data[doc_id]['doc'] = doc
    
    # Compute RRF scores
    fused_results = []
    for doc_id, data in doc_data.items():
        rrf_score = compute_rrf_score(data['ranks'], k=k)
        
        doc = data['doc'].copy()
        doc['rrf_score'] = rrf_score
        doc['rrf_ranks'] = data['ranks']
        
        if preserve_scores:
            for source_name, score in data['source_scores'].items():
                doc[f'{source_name}_score'] = score
        
        fused_results.append(doc)
    
    # Sort by RRF score (descending)
    fused_results.sort(key=lambda x: x['rrf_score'], reverse=True)
    
    return fused_results


def test_rrf_fusion():
    """Unit test for RRF fusion."""
    print("\n" + "="*60)
    print("TESTING RRF FUSION")
    print("="*60)
    
    # Create test results from two sources
    bm25_results = [
        {'chunk_id': 'a', 'score': 0.9, 'text': 'Doc A'},
        {'chunk_id': 'b', 'score': 0.8, 'text': 'Doc B'},
        {'chunk_id': 'c', 'score': 0.7, 'text': 'Doc C'},
        {'chunk_id': 'd', 'score': 0.6, 'text': 'Doc D'},
    ]
    
    dense_results = [
        {'chunk_id': 'b', 'score': 0.95, 'text': 'Doc B'},
        {'chunk_id': 'd', 'score': 0.85, 'text': 'Doc D'},
        {'chunk_id': 'a', 'score': 0.75, 'text': 'Doc A'},
        {'chunk_id': 'e', 'score': 0.65, 'text': 'Doc E'},
    ]
    
    print("\nBM25 ranking: a(0.9) > b(0.8) > c(0.7) > d(0.6)")
    print("Dense ranking: b(0.95) > d(0.85) > a(0.75) > e(0.65)")
    
    # Fuse results
    fused = rrf_fusion([bm25_results, dense_results], k=60)
    
    print("\nRRF Fusion Results:")
    for i, doc in enumerate(fused):
        print(f"  [{i+1}] {doc['chunk_id']}: "
              f"RRF={doc['rrf_score']:.4f}, "
              f"ranks={doc['rrf_ranks']}, "
              f"BM25={doc.get('source_0_score', 'N/A')}, "
              f"Dense={doc.get('source_1_score', 'N/A')}")
    
    # Verify expectations
    print("\nExpectations:")
    print("  - Doc B should rank high (top 2 in both sources)")
    print("  - Doc A should rank high (top 1 in BM25, top 3 in dense)")
    print("  - Doc D should rank high (rank 4 in BM25, rank 2 in dense)")
    print("  - Doc C appears only in BM25 (rank 3)")
    print("  - Doc E appears only in dense (rank 4)")
    
    # Calculate expected RRF scores manually
    print("\nManual RRF calculations:")
    print(f"  Doc A: 1/(60+0) + 1/(60+2) = {1/61 + 1/63:.4f}")
    print(f"  Doc B: 1/(60+1) + 1/(60+0) = {1/62 + 1/61:.4f}")
    print(f"  Doc C: 1/(60+2) = {1/63:.4f}")
    print(f"  Doc D: 1/(60+3) + 1/(60+1) = {1/64 + 1/62:.4f}")
    print(f"  Doc E: 1/(60+3) = {1/64:.4f}")
    
    assert fused[0]['chunk_id'] in ['a', 'b'], "Top result should be A or B"
    assert fused[-1]['chunk_id'] in ['c', 'e'], "Bottom result should be C or E"
    
    print("\n✓ RRF fusion tests complete")


if __name__ == '__main__':
    test_rrf_fusion()
