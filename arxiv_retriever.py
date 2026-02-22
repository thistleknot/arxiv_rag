"""
Arxiv GIST Retriever: Paper Retrieval with Section Aggregation

=============================================================================
HIERARCHY
=============================================================================

Chunk → Section → Paper (3-level aggregation)

1. Chunks: Retrieved from database (3-5 paragraphs, overlapping)
2. Sections: Reconstructed from chunks (paper_id, section_idx)
3. Papers: Selected from scored sections (iterate until K unique papers)

=============================================================================
"""

from typing import List, Dict, Any
from collections import defaultdict
import msgpack
import numpy as np
from pathlib import Path
from rank_bm25 import BM25Okapi
from base_gist_retriever import BaseGISTRetriever, RetrievedDoc


class ArxivRetriever(BaseGISTRetriever):
    """
    Arxiv-specific retriever with 3-level aggregation:
      chunk → section → paper
    
    Graph expansion uses semantic triplet extraction (not Node2Vec).
    """
    
    def __init__(self, config):
        """
        Initialize with triplet graph data for Layer 2 expansion.
        
        Args:
            config: PGVectorConfig with database settings
        """
        super().__init__(config)
        
        # Load triplet graph data (for graph expansion)
        try:
            # Load arxiv_graph_sparse.msgpack (triplet-based graph)
            graph_path = Path("arxiv_graph_sparse.msgpack")
            if graph_path.exists():
                with open(graph_path, 'rb') as f:
                    graph_data = msgpack.unpackb(f.read(), raw=False, strict_map_key=False)
                    self.chunk_to_triplets = graph_data.get('forward_index', {})
                    self.triplet_to_chunks = graph_data.get('inverse_index', {})
                    metadata = graph_data.get('metadata', {})
                    
                    # Note: This sparse format doesn't include triplet text,
                    # just the mappings. We'll need to load triplet texts separately
                    # or use a different approach for BM25.
                    print(f"    Loaded graph: {len(self.triplet_to_chunks):,} triplets, {len(self.chunk_to_triplets):,} chunks")
                    self.graph_bm25 = None  # TODO: Load triplet corpus for BM25
            else:
                print(f"    Warning: {graph_path} not found - graph expansion disabled")
                self.chunk_to_triplets = {}
                self.triplet_to_chunks = {}
                self.graph_bm25 = None
        except Exception as e:
            print(f"    Warning: Failed to load graph data: {e}")
            self.chunk_to_triplets = {}
            self.triplet_to_chunks = {}
            self.graph_bm25 = None
    
    def _reconstruct_documents_from_chunks(
        self,
        chunks: List[RetrievedDoc]
    ) -> List[Dict[str, Any]]:
        """
        Reconstruct full sections from retrieved chunks.
        
        Groups chunks by (paper_id, section_idx), then fetches all chunks
        in each section to rebuild complete section text.
        
        Args:
            chunks: List of retrieved chunk documents
        
        Returns:
            List of section dicts with full reconstructed text
        """
        # Extract unique (paper_id, section_idx) tuples
        section_keys = set()
        for chunk in chunks:
            paper_id = chunk.metadata.get('paper_id')
            section_idx = chunk.metadata.get('section_idx')
            if paper_id and section_idx is not None:
                section_keys.add((paper_id, section_idx))
        
        sections = []
        for paper_id, section_idx in section_keys:
            # Query database for ALL chunks in this section
            with self.conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT 
                        chunk_id,
                        content,
                        chunk_idx
                    FROM {self.pg_config.table_name}
                    WHERE paper_id = %s AND section_idx = %s
                    ORDER BY chunk_idx ASC
                    """,
                    (paper_id, section_idx)
                )
                
                result = cur.fetchall()
            
            if result:
                # Reconstruct full section text from all chunks
                section_text = ' '.join([row[1] for row in result])
                
                sections.append({
                    'section_id': f"{paper_id}_s{section_idx}",
                    'paper_id': paper_id,
                    'text': section_text,
                    'heading': '',  # No heading column in database
                    'section_index': section_idx,
                    'score': 0.0  # Will be scored by late interaction
                })
        
        return sections
    
    def _retrieve_graph(
        self,
        query: str,
        limit: int
    ) -> List[RetrievedDoc]:
        """
        Graph expansion via semantic triplet connections.
        
        Uses forward/inverse index to expand from hybrid seed chunks.
        Triplets act as edges connecting related chunks.
        
        Args:
            query: Search query (not used - we expand from hybrid seeds)
            limit: Maximum number of chunks to retrieve
        
        Returns:
            List of expanded chunks from graph traversal
        """
        if not self.chunk_to_triplets or not self.triplet_to_chunks:
            return []  # Graph data not loaded
        
        # Note: We can't access hybrid seeds here without refactoring BaseGISTRetriever
        # For now, return empty list - graph expansion happens via triplet BM25 in ThreeLayerPhiRetriever
        # TODO: Refactor base class to pass hybrid seeds to _retrieve_graph()
        
        return []
    
    def _select_final_documents(
        self,
        scored_sections: List[Dict[str, Any]],
        top_k: int
    ) -> List[RetrievedDoc]:
        """
        Select top K papers by iterating scored sections.
        
        Iterates through sections sorted by score until we have K unique papers.
        Each paper gets all its relevant sections (not all sections from the paper).
        
        Args:
            scored_sections: Sections sorted by relevance score
            top_k: Number of unique papers to select
        
        Returns:
            List of paper RetrievedDoc objects with sections
        """
        seen_papers = set()
        selected_sections = []
        
        # Iterate scored sections until K unique papers
        for section in scored_sections:
            paper_id = section['paper_id']
            
            # Track unique papers
            if paper_id not in seen_papers:
                seen_papers.add(paper_id)
            
            # Stop after K unique papers
            if len(seen_papers) > top_k:
                break
            
            # Keep this section
            selected_sections.append(section)
        
        # Group sections by paper
        papers_dict = defaultdict(list)
        for section in selected_sections:
            papers_dict[section['paper_id']].append(section)
        
        # Convert to RetrievedDoc format
        results = []
        for paper_id, sections in papers_dict.items():
            # Sort sections by index within paper
            sections.sort(key=lambda s: s.get('section_index', 0))
            
            # Calculate paper-level score (average of section scores)
            avg_score = sum(s['score'] for s in sections) / len(sections)
            
            # Create paper RetrievedDoc
            paper_doc = RetrievedDoc(
                doc_id=paper_id,
                content='',  # Content is in sections
                metadata={
                    'paper_id': paper_id,
                    'total_sections': len(sections)
                }
            )
            paper_doc.final_score = avg_score
            paper_doc.rrf_score = avg_score
            
            # Add sections
            paper_doc.sections = []
            for section_dict in sections:
                section_doc = RetrievedDoc(
                    doc_id=section_dict['section_id'],
                    content=section_dict['text'],
                    metadata={
                        'section_id': section_dict['section_id'],
                        'paper_id': section_dict['paper_id'],
                        'heading': section_dict['heading'],
                        'section_index': section_dict['section_index']
                    }
                )
                section_doc.final_score = section_dict['score']
                section_doc.chunks = []  # No chunks in GraphRAG architecture
                paper_doc.sections.append(section_doc)
            
            results.append(paper_doc)
        
        # Sort papers by score (descending)
        results.sort(key=lambda p: p.final_score, reverse=True)
        
        return results[:top_k]


if __name__ == "__main__":
    import argparse
    import datetime
    import time

    parser = argparse.ArgumentParser(
        description="ArxivRetriever — full 3-layer pipeline (L1 BM25+GIST, L2 ECDF, L3 ColBERT)"
    )
    parser.add_argument("--search", type=str, required=True, help="Search query")
    parser.add_argument("--top-k", type=int, default=13, help="Number of papers to return")
    parser.add_argument("--save", type=str, default=None, help="Save markdown results to this file")
    parser.add_argument("--no-colbert", action="store_true", help="Disable ColBERT reranking")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=5432)
    parser.add_argument("--db", type=str, default="langchain")
    parser.add_argument("--user", type=str, default="langchain")
    parser.add_argument("--password", type=str, default="langchain")
    parser.add_argument("--table", type=str, default="arxiv_chunks")
    args = parser.parse_args()

    from pgvector_retriever import PGVectorConfig

    config = PGVectorConfig(
        db_host=args.host,
        db_port=args.port,
        db_name=args.db,
        db_user=args.user,
        db_password=args.password,
        table_name=args.table,
        use_colbert=not args.no_colbert,
        use_cross_encoder=True,
        use_doc_doc_diversity=True,
        use_hnsw_diversity=False,
        bm25_min_score=0.0,
        dense_min_similarity=0.0,
        colbert_min_score=0.0,
        cross_encoder_min_score=0.0,
    )

    retriever = ArxivRetriever(config)

    t0 = time.time()
    results = retriever.search(args.search, top_k=args.top_k)
    elapsed = time.time() - t0

    total_sections = sum(len(p.sections) for p in results)
    score_lo = results[-1].final_score if results else 0.0
    score_hi = results[0].final_score if results else 0.0

    print(f"\nQuery : {args.search}")
    print(f"Papers: {len(results)}  |  Sections: {total_sections}  |  {elapsed:.1f}s")
    print(f"Scores: {score_lo:.4f} – {score_hi:.4f}\n")
    for i, p in enumerate(results, 1):
        secs = p.sections or []
        sec_scores = " ".join(f"{s.final_score:.4f}" for s in secs)
        print(f"  {i:2d}. {p.doc_id:<20s}  paper={p.final_score:.4f}  sections=[{sec_scores}]")

    if args.save:
        lines = [
            "# Retrieval Results",
            "",
            f"**Query:** {args.search}",
            f"**Run:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "**Pipeline:** L1 BM25+GIST→RRF | L2 BM25-triplet+Dense-centroid→RRF "
            "| L3 ColBERT+Cross-Encoder+GIST-diversity",
            f"**Time:** {elapsed:.1f}s",
            f"**Papers:** {len(results)}",
            "",
            "> Scores are ColBERT late-interaction composites computed at **section level**.",
            "> Paper score = avg(section scores).",
            "",
            "## Summary",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Unique papers | {len(results)} |",
            f"| Total sections | {total_sections} |",
            f"| Avg sections / paper | {total_sections / max(len(results), 1):.1f} |",
            f"| Score range | {score_lo:.4f} – {score_hi:.4f} |",
            "",
            "## Rankings",
            "",
        ]
        for rank, p in enumerate(results, 1):
            secs = p.sections or []
            lines += [
                f"### [{rank}] {p.doc_id}",
                "",
                f"**Paper score:** {p.final_score:.4f} &nbsp;|&nbsp; **Sections:** {len(secs)}",
                "",
            ]
            for j, sec in enumerate(secs, 1):
                meta = sec.metadata or {}
                heading = meta.get("heading", "") or ""
                sidx = meta.get("section_index", j)
                content = (sec.content or "").strip()
                preview = (content[:300] + "…") if len(content) > 300 else content
                heading_str = f" — *{heading}*" if heading else ""
                lines += [
                    f"**Section {j}** "
                    f"(section_idx={sidx}, ColBERT score={sec.final_score:.4f}){heading_str}",
                    "",
                    f"> {preview.replace(chr(10), ' ')}",
                    "",
                ]
        with open(args.save, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))
        print(f"\nSaved to {args.save}")
