"""
Demonstration of Hierarchical Paper→Section Retrieval

This script shows how the PGVectorRetriever (which implements GIST)
groups chunks into sections, then aggregates sections into papers,
showing the hierarchical structure.

Output: hierarchical_retrieval_demo.md
"""

import msgpack
from pathlib import Path
from datetime import datetime
from pgvector_retriever import PGVectorRetriever, PGVectorConfig
from simple_hybrid_retriever import SimpleHybridRetriever


def format_paper_hierarchy(papers, query):
    """Format papers with their sections as markdown."""
    lines = []
    lines.append("# Hierarchical Retrieval Demo")
    lines.append(f"\n**Query:** `{query}`")
    lines.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"\n**Papers Retrieved:** {len(papers)}")
    
    total_sections = sum(len(paper.sections) for paper in papers)
    lines.append(f"**Total Sections:** {total_sections}")
    lines.append("\n---\n")
    
    for idx, paper in enumerate(papers, 1):
        lines.append(f"## Paper {idx}: {paper.paper_id}")
        lines.append(f"\n**Aggregated RRF Score:** {paper.rrf_score:.4f}")
        lines.append(f"**Number of Sections:** {len(paper.sections)}")
        lines.append(f"\n### Sections\n")
        
        for sec_idx, section in enumerate(paper.sections, 1):
            section_num = section.group_key[1]  # Extract section_idx
            lines.append(f"#### Section {sec_idx} (Index: {section_num})")
            lines.append(f"\n**Section RRF Score:** {section.rrf_score:.4f}")
            lines.append(f"**Matched Chunks:** {len(section.matched_chunks)}")
            lines.append(f"**Total Chunks in Section:** {section.metadata.get('num_chunks', 'N/A')}")
            
            # Show first 500 chars of section text
            text_preview = section.full_text[:500]
            if len(section.full_text) > 500:
                text_preview += "..."
            
            lines.append(f"\n**Text Preview:**")
            lines.append(f"```")
            lines.append(text_preview)
            lines.append(f"```\n")
        
        lines.append("\n---\n")
    
    return "\n".join(lines)


def main():
    """Run hierarchical retrieval demo."""
    print("="*60)
    print("HIERARCHICAL RETRIEVAL DEMONSTRATION")
    print("="*60)
    
    query = "agentic memory methods"
    print(f"\nQuery: {query}")
    
    # Initialize retrievers
    print("\nInitializing PGVector GIST retriever...")
    
    config = PGVectorConfig(
        db_name="langchain",
        db_user="langchain",
        db_password="langchain",
        db_host="localhost",
        db_port=5432,
        table_name="arxiv_chunks",
        graph_path="data/arxiv_graph_sparse.msgpack",
        bm25_vocab_path="data/graph_vocab_adaptive.msgpack"
    )
    
    retriever = PGVectorRetriever(config)
    
    # Run retrieval
    print("\nRunning retrieval with hierarchical grouping...")
    print("(This demonstrates: RRF → sections → papers grouping)")
    
    # For demo purposes, we'll run simple hybrid retrieval (Layer 1)
    # then group results to show hierarchical structure
    
    print("\n" + "="*60)
    print("RETRIEVING...")
    print("="*60)
    
    # Get Layer 1 seeds using retriever's internal methods
    print("Running BM25 + Dense retrieval...")
    retrieval_limit = 13 * 3  # Over-retrieve for RRF
    bm25_results = retriever._retrieve_bm25(query, retrieval_limit)
    dense_results = retriever._retrieve_dense(query, retrieval_limit)
    
    # RRF fusion
    fused_results = retriever._rrf_fusion(bm25_results, dense_results)
    layer1_results = fused_results[:13]  # Top 13 seeds
    print(f"✓ Retrieved {len(layer1_results)} seed chunks via hybrid RRF")
    
    # For this demo, we'll group just these results to show structure
    # (Full GIST does graph expansion first, but this demonstrates the pattern)
    
    print("\nGrouping chunks into sections...")
    sections = retriever._group_and_reconstruct(
        chunks=layer1_results,
        limit=30  # Get more sections for demo
    )
    print(f"✓ Grouped into {len(sections)} sections")
    
    print("\nAggregating sections into papers...")
    papers = retriever._group_sections_into_papers(
        sections=sections,
        max_papers=10  # Top 10 papers for demo
    )
    print(f"✓ Selected top {len(papers)} papers")
    
    # Format as markdown
    print("\nGenerating markdown output...")
    markdown = format_paper_hierarchy(papers, query)
    
    # Write to file
    output_file = Path("hierarchical_retrieval_demo.md")
    output_file.write_text(markdown, encoding='utf-8')
    
    print(f"\n{'='*60}")
    print(f"✓ Output written to: {output_file}")
    print(f"{'='*60}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Total papers: {len(papers)}")
    print(f"Total sections: {sum(len(p.sections) for p in papers)}")
    
    # Show paper distribution
    print("\nPaper Statistics:")
    for idx, paper in enumerate(papers, 1):
        print(f"  Paper {idx} ({paper.paper_id}): {len(paper.sections)} sections, RRF={paper.rrf_score:.4f}")
    
    print("\n" + "="*60)
    print("HIERARCHICAL STRUCTURE DEMONSTRATED")
    print("="*60)
    print("\nThe output shows:")
    print("  1. Papers ranked by aggregated RRF score")
    print("  2. Each paper contains ALL its sections")
    print("  3. Each section shows matched chunks + full reconstructed text")
    print("  4. This is what Layer 3 would process for cross-encoding")
    print("\nReview the markdown file to see the complete structure!")


if __name__ == "__main__":
    main()
