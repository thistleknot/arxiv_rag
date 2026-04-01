"""
Query arXiv Papers with Full GIST Pipeline

Runs a query through the complete retrieval pipeline:
- BM25 + Dense retrieval
- GIST lemma selection
- RRF fusion
- Grouping by paper
- ColBERT scoring
- Doc-Doc GIST: GIST select using ColBERT doc-doc matrix, GIST select using
                Cross-Encoder doc-doc matrix, then RRF fuse both selections

Usage:
    python query_arxiv.py "agentic memory methods"
    python query_arxiv.py "agentic memory methods" --no-diversity --top-k 21
"""

import sys
import argparse
from pathlib import Path

def query_papers(
    query: str,
    use_diversity: bool = True,
    top_k: int = 13
):
    """Run query and display results.
    
    Args:
        query: Search query
        use_diversity: Enable doc-doc GIST diversity selection
        top_k: Number of papers to retrieve
    """
    try:
        from arxiv_retriever import ArxivRetriever
        from pgvector_retriever import PGVectorConfig
    except ImportError as e:
        print(f"Error importing retriever: {e}")
        print("Make sure arxiv_retriever.py and pgvector_retriever.py are in the current directory.")
        return
    
    print("=" * 80)
    print(f"QUERY: {query}")
    print("=" * 80)
    
    # Configure retriever with unified filtering
    config = PGVectorConfig(
        db_host="localhost",
        db_port=5432,
        db_name="langchain",
        db_user="langchain",
        db_password="langchain",
        table_name="arxiv_chunks",
        use_colbert=True,
        use_cross_encoder=True,
        use_doc_doc_diversity=use_diversity,  # Dual-GIST on doc-doc matrices + RRF
        use_hnsw_diversity=False,  # LEGACY: Disabled
        # Unified filtering (all stages): positive scores only
        bm25_min_score=0.0,
        dense_min_similarity=0.0,
        colbert_min_score=0.0,
        cross_encoder_min_score=0.0,
    )
    
    print(f"\nConfiguration:")
    print(f"  - Database: {config.db_host}:{config.db_port}/{config.db_name}")
    print(f"  - Table: {config.table_name}")
    print(f"  - ColBERT: {'Yes' if config.use_colbert else 'No'}")
    print(f"  - Cross-Encoder: {'Yes' if config.use_cross_encoder else 'No'}")
    print(f"  - Unified Filtering (all stages):")
    print(f"    - BM25 min score: {config.bm25_min_score}")
    print(f"    - Dense min similarity: {config.dense_min_similarity}")
    print(f"    - ColBERT min score: {config.colbert_min_score}")
    print(f"    - Cross-Encoder min score: {config.cross_encoder_min_score}")
    print(f"  - Doc-Doc GIST (CB + CE -> RRF): {'Yes' if config.use_doc_doc_diversity else 'No'}")
    if use_diversity:
        print(f"    - gist_lambda={config.gist_lambda}")
    print(f"  - Top-K: {top_k}")
    
    # Initialize retriever
    print(f"\n{'Initializing retriever...':<40}", end='', flush=True)
    try:
        retriever = ArxivRetriever(config)
        print("OK")
    except Exception as e:
        print(f"FAILED\nError: {e}")
        return
    
    # Run query
    print(f"{'Running query...':<40}", end='', flush=True)
    try:
        results = retriever.search(query, top_k=top_k)
        print("OK")
    except Exception as e:
        print(f"FAILED\nError: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Display results
    print("\n" + "=" * 80)
    print(f"RESULTS: {len(results)} papers")
    print("=" * 80)
    
    for i, paper in enumerate(results, 1):
        # Paper ID is stored in doc_id or metadata
        paper_id = paper.doc_id if hasattr(paper, 'doc_id') else paper.metadata.get('paper_id', 'unknown')
        print(f"\n[{i}] {paper_id}")
        
        # Show scores
        scores = []
        if paper.colbert_score is not None:
            scores.append(f"ColBERT: {paper.colbert_score:.4f}")
        # Check for diversity score in metadata
        combined_div = paper.metadata.get('combined_diversity')
        if combined_div is not None:
            scores.append(f"Diversity: {combined_div:.4f}")
        if paper.cross_encoder_score is not None:
            scores.append(f"Cross-Encoder: {paper.cross_encoder_score:.4f}")
        rrf_final = paper.metadata.get('rrf_final_score')
        if rrf_final is not None:
            scores.append(f"RRF: {rrf_final:.4f}")
        elif paper.final_score is not None:
            scores.append(f"Final: {paper.final_score:.4f}")
        
        if scores:
            print(f"    Scores: {' | '.join(scores)}")
        
        # Show sections
        if paper.sections:
            print(f"    Sections: {len(paper.sections)} matched")
            for j, section in enumerate(paper.sections[:2], 1):  # Show first 2
                # Section index from metadata
                section_idx = section.metadata.get('section_index', j)
                print(f"      [{j}] Section {section_idx}: score {section.final_score:.4f}")
                # Show first 150 chars of section text
                section_text = section.content[:150].replace('\n', ' ') if section.content else '(no text)'
                print(f"          {section_text}...")
        
        print()
    
    # Summary statistics
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    if results:
        avg_sections = sum(len(p.sections) for p in results) / len(results)
        print(f"  • Average sections per paper: {avg_sections:.1f}")
        
        if any(p.colbert_score is not None for p in results):
            avg_colbert = sum(p.colbert_score for p in results if p.colbert_score) / len([p for p in results if p.colbert_score])
            print(f"  • Average ColBERT score: {avg_colbert:.4f}")
        
        if use_diversity and any(hasattr(p, 'diversity_score') and p.diversity_score is not None for p in results):
            diverse_papers = [p for p in results if hasattr(p, 'diversity_score') and p.diversity_score is not None]
            avg_diversity = sum(p.diversity_score for p in diverse_papers) / len(diverse_papers)
            print(f"  • Average diversity score: {avg_diversity:.4f}")
        
        if any(p.cross_encoder_score is not None for p in results):
            avg_ce = sum(p.cross_encoder_score for p in results if p.cross_encoder_score) / len([p for p in results if p.cross_encoder_score])
            print(f"  • Average cross-encoder score: {avg_ce:.4f}")
    
    # Show diversity statistics if available
    div_scores = [p.metadata.get('combined_diversity') for p in results if p.metadata.get('combined_diversity') is not None]
    if div_scores:
        print(f"  • Average diversity score: {sum(div_scores)/len(div_scores):.4f}")
        print(f"  • Diversity range: {min(div_scores):.4f} - {max(div_scores):.4f}")
    
    print(f"\nQuery complete!")


def main():
    parser = argparse.ArgumentParser(
        description='Query arXiv papers with GIST retrieval pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python query_arxiv.py "agentic memory methods"
  python query_arxiv.py "agentic memory methods" --no-diversity --top-k 21
  python query_arxiv.py "transformer attention" --top-k 10
        """
    )
    
    parser.add_argument('query', type=str, help='Search query')
    parser.add_argument('--diversity', dest='use_diversity', action='store_true',
                       help='Enable doc-doc GIST (ColBERT + Cross-Encoder -> RRF) [default]')
    parser.add_argument('--no-diversity', dest='use_diversity', action='store_false',
                       help='Disable doc-doc GIST, use only ColBERT ranking')
    parser.add_argument('--top-k', type=int, default=13,
                       help='Number of papers to retrieve (default: 13)')
    
    parser.set_defaults(use_diversity=True)
    
    args = parser.parse_args()
    
    query_papers(
        args.query,
        args.use_diversity,
        args.top_k
    )


if __name__ == '__main__':
    main()
