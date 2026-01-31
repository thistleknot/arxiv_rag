"""
Query arXiv Papers with ECDF-Weighted RRF

Simplified retrieval pipeline:
- Parallel retrieval across all metrics (BM25, Cosine, ColBERT, CE)
- ECDF normalization for each metric
- Priority-weighted RRF fusion
- Optional cascading filter for backward compatibility

Usage:
    # Default: ECDF-RRF with ColBERT > CE > Cosine > BM25
    python query_arxiv_ecdf.py "agentic memory methods"
    
    # With cascading filter (original approach)
    python query_arxiv_ecdf.py "agentic memory methods" --use-filtering
    
    # Custom priorities (CE-first)
    python query_arxiv_ecdf.py "agentic memory methods" \
        --priority-ranks "ce:1,colbert:2,cosine:3,bm25:4"
    
    # Export full results
    python query_arxiv_ecdf.py "agentic memory methods" --export
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime


def parse_priority_ranks(rank_str: str) -> dict:
    """
    Parse priority ranks from string format.
    
    Args:
        rank_str: "metric1:rank1,metric2:rank2,..."
    
    Returns:
        {metric: rank} dictionary
        
    Example:
        >>> parse_priority_ranks("ce:1,colbert:2,cosine:3,bm25:4")
        {'ce': 1, 'colbert': 2, 'cosine': 3, 'bm25': 4}
    """
    ranks = {}
    for pair in rank_str.split(','):
        metric, rank = pair.split(':')
        ranks[metric.strip()] = int(rank.strip())
    return ranks


def query_papers(
    query: str,
    top_k: int = 13,
    use_filtering: bool = False,
    filter_multiplier: float = 1.5,
    priority_ranks: dict = None,
    export: bool = False
):
    """
    Run query with ECDF-weighted RRF.
    
    Args:
        query: Search query
        top_k: Number of papers to retrieve
        use_filtering: Enable cascading filter (ColBERT → CE)
        filter_multiplier: Multiplier for filtering (e.g., 1.5)
        priority_ranks: {metric: rank} for custom priorities
        export: Export full results to markdown
    """
    try:
        from pgvector_retriever import PGVectorRetriever, PGVectorConfig
        from ecdf_rrf_retriever import ECDFRRFReranker
    except ImportError as e:
        print(f"Error importing: {e}")
        return
    
    print("=" * 80)
    print(f"QUERY: {query}")
    print("=" * 80)
    
    # Configure retriever
    # ECDF-RRF approach: No chunk-level filtering by default
    # IF filtering requested: Use MAD * 2 adaptive thresholds (not hardcoded)
    config = PGVectorConfig(
        db_host="localhost",
        db_port=5432,
        db_name="postgres",
        db_user="postgres",
        db_password="postgres",
        table_name="arxiv_papers_lemma_fullembed",
        use_colbert=True,
        use_cross_encoder=True,
        # ECDF-RRF: No stage-level filtering
        use_doc_doc_diversity=False,
        use_hnsw_diversity=False,
        # Disable joint filtering by setting negative thresholds (skip filtering)
        bm25_min_score=-float('inf') if not use_filtering else None,  # Will compute MAD if None
        dense_min_similarity=-float('inf') if not use_filtering else None,
        colbert_min_score=-float('inf'),  # Never filter ColBERT/CE at chunk level
        cross_encoder_min_score=-float('inf'),
    )
    
    print(f"\nConfiguration:")
    print(f"  - Database: {config.db_host}:{config.db_port}/{config.db_name}")
    print(f"  - Table: {config.table_name}")
    print(f"  - Retrieval Strategy: ECDF-Weighted RRF")
    print(f"  - Top-K: {top_k}")
    
    # Initialize retriever
    print(f"\n{'Initializing retriever...':<40}", end='', flush=True)
    try:
        retriever = PGVectorRetriever(config)
        print("OK")
    except Exception as e:
        print(f"FAILED\nError: {e}")
        return
    
    # Initialize ECDF-RRF reranker
    print(f"{'Initializing ECDF-RRF reranker...':<40}", end='', flush=True)
    try:
        reranker = ECDFRRFReranker(
            priority_ranks=priority_ranks,
            use_filtering=use_filtering,
            filter_top_k_multiplier=filter_multiplier
        )
        print("OK")
    except Exception as e:
        print(f"FAILED\nError: {e}")
        return
    
    # Run standard retrieval pipeline
    # This will get us papers with all metric scores already computed
    print(f"\n{'Running retrieval pipeline...':<40}", end='', flush=True)
    try:
        # Use larger top_k for retrieval to get more candidates
        retrieval_k = top_k * 2  # Get 2x papers for ECDF-RRF to rerank
        
        papers = retriever.search(query, top_k=retrieval_k)
        print(f"OK ({len(papers)} papers)")
    except Exception as e:
        print(f"FAILED\nError: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Apply ECDF-RRF reranking on the retrieved papers
    print(f"\n{'Applying ECDF-RRF reranking...':<40}", end='', flush=True)
    try:
        results = reranker.rerank(papers, top_k=top_k)
        print(f"OK ({len(results)} papers)")
    except Exception as e:
        print(f"FAILED\nError: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Display results
    print("\n" + "=" * 80)
    print(f"TOP {len(results)} RESULTS (ECDF-Weighted RRF)")
    print("=" * 80)
    
    for i, paper in enumerate(results, 1):
        # Get paper title and ID
        title = getattr(paper, 'title', getattr(paper, 'paper_id', 'Unknown'))
        paper_id = getattr(paper, 'paper_id', getattr(paper, 'arxiv_id', 'Unknown'))
        
        print(f"\n{i}. {title}")
        print(f"   Paper ID: {paper_id}")
        
        # Show all metric scores (ColBERT + CE only)
        print(f"   Scores:")
        print(f"     RRF:     {paper.rrf_score:.4f}")
        print(f"     ColBERT: {paper.colbert_score:.4f} (ECDF: {paper.ecdf_colbert:.3f})")
        print(f"     CE:      {paper.cross_encoder_score:.4f} (ECDF: {paper.ecdf_ce:.3f})")
        
        # Show snippet from sections if available
        if hasattr(paper, 'sections') and paper.sections:
            section = paper.sections[0]
            if hasattr(section, 'chunks') and section.chunks:
                snippet = section.chunks[0].content[:200]
            elif hasattr(section, 'text'):
                snippet = section.text[:200]
            else:
                snippet = None
            
            if snippet:
                print(f"\n   Snippet: {snippet}...")
    
    # Export if requested
    if export:
        export_results(query, results, use_filtering, priority_ranks)
    
    print("\n" + "=" * 80)


def export_results(query: str, results: list, use_filtering: bool, priority_ranks: dict):
    """Export full results to markdown file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"ecdf_rrf_results_{timestamp}.md"
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"# ECDF-RRF Search Results\n\n")
        f.write(f"**Query**: {query}\n\n")
        f.write(f"**Strategy**: ECDF-Weighted RRF\n")
        f.write(f"**Filtering**: {'Yes' if use_filtering else 'No'}\n")
        f.write(f"**Priority Ranks**: {priority_ranks}\n\n")
        f.write(f"**Retrieved**: {len(results)} papers\n\n")
        f.write("---\n\n")
        
        for i, paper in enumerate(results, 1):
            # Get paper info
            title = getattr(paper, 'title', getattr(paper, 'paper_id', 'Unknown'))
            paper_id = getattr(paper, 'paper_id', getattr(paper, 'arxiv_id', 'Unknown'))
            
            f.write(f"## {i}. {title}\n\n")
            f.write(f"**Paper ID**: {paper_id}\n\n")
            
            # Scores
            f.write("**Scores**:\n")
            f.write(f"- RRF: {paper.rrf_score:.4f}\n")
            f.write(f"- ColBERT: {paper.colbert_score:.4f} (ECDF: {paper.ecdf_colbert:.3f})\n")
            f.write(f"- CE: {paper.cross_encoder_score:.4f} (ECDF: {paper.ecdf_ce:.3f})\n")
            f.write("\n")
            
            # Sections
            if hasattr(paper, 'sections') and paper.sections:
                f.write("**Retrieved Text**:\n\n")
                for section in paper.sections[:3]:  # Limit to 3 sections
                    # Get section name
                    section_idx = getattr(section, 'group_key', ('', 0))[1] if hasattr(section, 'group_key') else 0
                    section_name = getattr(section, 'section_name', f"Section {section_idx}")
                    f.write(f"### {section_name}\n\n")
                    
                    # Get text - try full_text first (chunks are stripped), then chunks, then text
                    text = None
                    if hasattr(section, 'full_text') and section.full_text:
                        text = section.full_text[:1000] + "..." if len(section.full_text) > 1000 else section.full_text
                    elif hasattr(section, 'chunks') and section.chunks:
                        text = section.chunks[0].content[:1000] + "..."
                    elif hasattr(section, 'text') and section.text:
                        text = section.text[:1000] + "..."
                    
                    if not text:
                        text = "No text available"
                    
                    f.write(f"{text}\n\n")
            
            f.write("---\n\n")
    
    print(f"\nResults exported to: {filename}")


def main():
    parser = argparse.ArgumentParser(
        description="Query arXiv papers with ECDF-weighted RRF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default priorities (ColBERT > CE > Cosine > BM25)
  python query_arxiv_ecdf.py "agentic memory methods"
  
  # With cascading filter (original approach)
  python query_arxiv_ecdf.py "agentic memory methods" --use-filtering
  
  # Custom priorities (CE-first)
  python query_arxiv_ecdf.py "agentic memory methods" \\
    --priority-ranks "ce:1,colbert:2,cosine:3,bm25:4"
  
  # Export full results
  python query_arxiv_ecdf.py "agentic memory methods" --export
        """
    )
    
    parser.add_argument("query", help="Search query")
    parser.add_argument("--top-k", type=int, default=13, 
                       help="Number of papers to return (default: 13)")
    parser.add_argument("--use-filtering", action="store_true",
                       help="Enable cascading filter (ColBERT → CE)")
    parser.add_argument("--filter-multiplier", type=float, default=1.5,
                       help="Multiplier for cascading filter (default: 1.5)")
    parser.add_argument("--priority-ranks", type=str,
                       help="Custom priority ranks (e.g., 'ce:1,colbert:2,cosine:3,bm25:4')")
    parser.add_argument("--export", action="store_true",
                       help="Export full results to markdown file")
    
    args = parser.parse_args()
    
    # Parse priority ranks if provided
    priority_ranks = None
    if args.priority_ranks:
        try:
            priority_ranks = parse_priority_ranks(args.priority_ranks)
        except Exception as e:
            print(f"Error parsing priority ranks: {e}")
            print("Format: 'metric1:rank1,metric2:rank2,...'")
            return
    
    # Run query
    query_papers(
        query=args.query,
        top_k=args.top_k,
        use_filtering=args.use_filtering,
        filter_multiplier=args.filter_multiplier,
        priority_ranks=priority_ranks,
        export=args.export
    )


if __name__ == "__main__":
    main()
