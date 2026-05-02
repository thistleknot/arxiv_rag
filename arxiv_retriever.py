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

from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Any, Optional
from retrieval.base_gist_retriever import BaseGISTRetriever, RetrievedDoc

# Default location for pre-computed pipeline outputs (Phase 4 enriched MD + Phase 5 methods)
PAPERS_DIR = Path(__file__).parent / "papers" / "post_processed"


def load_methods_content(paper_id: str, papers_dir: Path) -> Optional[str]:
    """
    Load pre-computed Phase 5 methods pseudocode for a paper if available.

    Tries {paper_id}_methods.md with both underscore and dot normalizations so
    that arxiv IDs stored as "1806_07366" or "1806.07366" both resolve.

    Require: papers_dir is a readable directory path (need not exist).
    Guarantee: returns file text if found, None otherwise.
    """
    variants = [paper_id, paper_id.replace(".", "_"), paper_id.replace("_", ".")]
    for vid in dict.fromkeys(variants):  # deduplicated, order-preserving
        p = papers_dir / f"{vid}_methods.md"
        if p.is_file():
            return p.read_text(encoding="utf-8")
    return None


def load_enriched_md(paper_id: str, papers_dir: Path) -> Optional[str]:
    """
    Load the Phase 4 enriched Markdown (with inline image descriptions) if available.

    Require: papers_dir is a readable directory path (need not exist).
    Guarantee: returns file text if found, None otherwise.
    """
    variants = [paper_id, paper_id.replace(".", "_"), paper_id.replace("_", ".")]
    for vid in dict.fromkeys(variants):
        p = papers_dir / f"{vid}.md"
        if p.is_file():
            return p.read_text(encoding="utf-8")
    return None


class ArxivRetriever(BaseGISTRetriever):
    """
    Arxiv-specific retriever with 3-level aggregation:
      chunk → section → paper
    
    L2 expansion queries layer2_triplet_bm25 (pgvector sparsevec, 118k triplets).
    """
    
    def __init__(self, config):
        """
        Args:
            config: PGVectorConfig with database settings
        """
        super().__init__(config)
    
    def _reconstruct_documents_from_chunks(
        self,
        chunks: List[RetrievedDoc],
        target_sections: int,
    ) -> List[Dict[str, Any]]:
        """
        Section expansion: select top-target_sections unique sections from the
        fused chunk pool, then reconstruct each section's full text.

        Spec (Feature 17):
          Sort chunks by score desc (tiebreak: chunk_idx, section_idx, doc_id).
          Walk sorted list, collecting unique (paper_id, section_idx) keys until
          target_sections reached.  For each collected key, fetch ALL chunks in
          that section from the DB to rebuild the full section text.

        This ensures we get exactly target_sections sections (= hybrid_seeds = 144
        for top_k=13), ranked by the best-scoring chunk in each section.

        Args:
            chunks: Fused chunk pool (RRF-sorted)
            target_sections: Max unique sections to reconstruct (= hybrid_seeds)

        Returns:
            List of section dicts (up to target_sections), each with full text
        """
        # Sort by score desc; tiebreak by chunk_idx, section_idx, doc_id
        sorted_chunks = sorted(
            chunks,
            key=lambda c: (
                -c.final_score,
                c.metadata.get('chunk_idx', 0),
                c.metadata.get('section_idx', 0),
                c.doc_id,
            )
        )

        # Walk sorted chunks, collect unique section keys (preserves rank order)
        seen: Dict[tuple, None] = {}  # ordered dict as ordered set
        for chunk in sorted_chunks:
            pid = chunk.metadata.get('paper_id')
            sidx = chunk.metadata.get('section_idx')
            if pid is not None and sidx is not None:
                key = (pid, sidx)
                if key not in seen:
                    seen[key] = None
                    if len(seen) >= target_sections:
                        break

        # Reconstruct each selected section from DB
        sections: List[Dict[str, Any]] = []
        for paper_id, section_idx in seen:
            with self.conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT content, chunk_idx
                    FROM {self.pg_config.table_name}
                    WHERE paper_id = %s AND section_idx = %s
                    ORDER BY chunk_idx ASC
                    """,
                    (paper_id, section_idx)
                )
                rows = cur.fetchall()

            if rows:
                section_text = ' '.join(r[0] for r in rows)
                sections.append({
                    'section_id': f"{paper_id}_s{section_idx}",
                    'paper_id': paper_id,
                    'text': section_text,
                    'heading': '',
                    'section_index': section_idx,
                    'score': 0.0,
                })

        return sections
    
    def _select_final_documents(
        self,
        scored_sections: List[Dict[str, Any]],
        top_k: int
    ) -> List[RetrievedDoc]:
        """
        Select top_k papers' worth of sections from L3-scored sections.

        Pass 1: walk sections in descending L3 RRF score order, collecting
                unique paper_ids until top_k distinct papers are identified.
        Pass 2: collect ALL sections (from `scored_sections`) that belong to
                those top_k papers — including sections that ranked below
                an (top_k+1)th paper's first occurrence.

        Each paper's sections are sorted by section_index for coherent reading.
        Papers are sorted by their best-section L3 score (descending).
        """
        # Pass 1: identify top_k paper_ids by L3 rank order
        top_paper_ids: list = []
        seen: set = set()
        for section in scored_sections:
            pid = section['paper_id']
            if pid not in seen:
                seen.add(pid)
                top_paper_ids.append(pid)
            if len(top_paper_ids) == top_k:
                break

        # Pass 2: collect ALL sections belonging to those top_k papers
        papers_dict = defaultdict(list)
        for section in scored_sections:
            if section['paper_id'] in seen:
                papers_dict[section['paper_id']].append(section)
        
        # Convert to RetrievedDoc format
        results = []
        for paper_id, sections in papers_dict.items():
            # Sort sections by index within paper for coherent reading
            sections.sort(key=lambda s: s.get('section_index', 0))

            # Paper-level score = best section score (paper rank derived from top section)
            best_score = max(s['score'] for s in sections)
            
            # Create paper RetrievedDoc
            paper_doc = RetrievedDoc(
                doc_id=paper_id,
                content='',
                metadata={
                    'paper_id': paper_id,
                    'total_sections': len(sections)
                }
            )
            paper_doc.final_score = best_score
            paper_doc.rrf_score = best_score
            
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
    import sys
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
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
    parser.add_argument(
        "--papers-dir",
        type=str,
        default=str(PAPERS_DIR),
        help="Directory containing pre-processed paper MDs and _methods.md files",
    )
    parser.add_argument(
        "--no-methods",
        action="store_true",
        help="Skip injecting pre-computed methods pseudocode into results",
    )
    args = parser.parse_args()
    papers_dir = Path(args.papers_dir)

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

    # Attach pre-computed enrichments to each paper's metadata
    if not args.no_methods:
        for p in results:
            p.metadata["methods"] = load_methods_content(p.doc_id, papers_dir)
            p.metadata["enriched_md"] = load_enriched_md(p.doc_id, papers_dir)

    total_sections = sum(len(p.sections) for p in results)
    enriched_count = sum(1 for p in results if p.metadata.get("methods")) if not args.no_methods else 0
    score_lo = results[-1].final_score if results else 0.0
    score_hi = results[0].final_score if results else 0.0

    print(f"\nQuery : {args.search}")
    print(f"Papers: {len(results)}  |  Sections: {total_sections}  |  {elapsed:.1f}s")
    print(f"Scores: {score_lo:.4f} – {score_hi:.4f}")
    if not args.no_methods:
        print(f"Enriched: {enriched_count}/{len(results)} papers have pre-computed methods")
    print()
    for i, p in enumerate(results, 1):
        secs = p.sections or []
        sec_scores = " ".join(f"{s.final_score:.4f}" for s in secs)
        has_methods = "✓methods" if p.metadata.get("methods") else ""
        has_img = "✓images" if p.metadata.get("enriched_md") else ""
        enrichment_tag = f"  [{' '.join(t for t in [has_methods, has_img] if t)}]" if (has_methods or has_img) else ""
        print(f"  {i:2d}. {p.doc_id:<20s}  paper={p.final_score:.4f}  sections=[{sec_scores}]{enrichment_tag}")

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
            f"| Papers with extracted methods | {enriched_count} |",
            "",
            "## Rankings",
            "",
        ]
        for rank, p in enumerate(results, 1):
            secs = p.sections or []
            methods_text = p.metadata.get("methods")
            lines += [
                f"### [{rank}] {p.doc_id}",
                "",
                f"**Paper score:** {p.final_score:.4f} &nbsp;|&nbsp; **Sections:** {len(secs)}"
                + (" &nbsp;|&nbsp; **Methods: ✓**" if methods_text else ""),
                "",
            ]
            for j, sec in enumerate(secs, 1):
                meta = sec.metadata or {}
                heading = meta.get("heading", "") or ""
                sidx = meta.get("section_index", j)
                content = (sec.content or "").strip()
                heading_str = f" — *{heading}*" if heading else ""
                # Blockquote: prefix every non-empty line with "> "
                blockquote = "\n".join(
                    f"> {ln}" if ln.strip() else ">"
                    for ln in content.splitlines()
                )
                lines += [
                    f"**Section {j}** "
                    f"(section_idx={sidx}, ColBERT score={sec.final_score:.4f}){heading_str}",
                    "",
                    blockquote,
                    "",
                ]
            if methods_text:
                lines += [
                    "#### Extracted Methods (pseudocode)",
                    "",
                    methods_text.strip(),
                    "",
                ]
        with open(args.save, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))
        print(f"\nSaved to {args.save}")
