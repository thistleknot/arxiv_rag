"""
Generate a retrieval results markdown file.

Usage:
    python gen_results.py "your query here"
    python gen_results.py "attention mechanism" --top-k 13 --out results.md
"""

import argparse
import datetime
import time

from arxiv_retriever import ArxivRetriever
from pgvector_retriever import PGVectorConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="?", default="attention mechanism transformer")
    parser.add_argument("--top-k", type=int, default=13)
    parser.add_argument("--out", default="_retrieval_results.md")
    parser.add_argument(
        "--no-colbert", action="store_true", help="Disable ColBERT reranking"
    )
    args = parser.parse_args()

    config = PGVectorConfig(
        db_host="localhost",
        db_port=5432,
        db_name="langchain",
        db_user="langchain",
        db_password="langchain",
        table_name="arxiv_chunks",
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
    results = retriever.search(args.query, top_k=args.top_k)
    elapsed = time.time() - t0

    lines = build_markdown(args.query, results, elapsed)
    text = "\n".join(lines)

    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(text)

    total_sections = sum(len(p.sections) for p in results)
    print(
        f"Written to {args.out}  "
        f"| papers={len(results)} sections={total_sections} "
        f"| {elapsed:.1f}s"
    )


def build_markdown(query: str, results: list, elapsed: float) -> list[str]:
    lines: list[str] = []

    total_sections = sum(len(p.sections) for p in results)
    score_lo = results[-1].final_score if results else 0.0
    score_hi = results[0].final_score if results else 0.0

    lines += [
        "# Retrieval Results",
        "",
        f"**Query:** {query}",
        f"**Run:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "**Pipeline:** L1 BM25+GIST→RRF | L2 BM25-triplet+Dense-centroid→RRF "
        "| L3 ColBERT+Cross-Encoder+GIST-diversity",
        f"**Time:** {elapsed:.1f}s",
        f"**Papers:** {len(results)}",
        "",
        "## Breadth / Depth Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Unique papers | {len(results)} |",
        f"| Total sections | {total_sections} |",
        f"| Avg sections / paper | {total_sections / max(len(results), 1):.1f} |",
        f"| Score range | {score_lo:.4f} – {score_hi:.4f} |",
        "",
        "> Scores are ColBERT late-interaction composites computed at **section level**.",
        "> Paper score = avg of its retrieved section scores.",
        "",
        "## Paper Rankings",
        "",
    ]

    for rank, paper in enumerate(results, 1):
        pid = getattr(paper, "doc_id", "?")
        sections = getattr(paper, "sections", []) or []
        paper_score = paper.final_score or 0.0
        n_sec = len(sections)

        lines += [
            f"### [{rank}] {pid}",
            "",
            f"**Paper score:** {paper_score:.4f} &nbsp;|&nbsp; "
            f"**Sections retrieved:** {n_sec}",
            "",
        ]

        for j, sec in enumerate(sections, 1):
            sec_score = sec.final_score or 0.0
            sec_meta = getattr(sec, "metadata", {}) or {}
            heading = sec_meta.get("heading", "") or ""
            sidx = sec_meta.get("section_index", j)
            content = (getattr(sec, "content", "") or "").strip()
            preview = (content[:300] + "…") if len(content) > 300 else content
            heading_str = f" — *{heading}*" if heading else ""

            lines += [
                f"**Section {j}** "
                f"(section_idx={sidx}, ColBERT score={sec_score:.4f}){heading_str}",
                "",
                f"> {preview.replace(chr(10), ' ')}",
                "",
            ]

    return lines


if __name__ == "__main__":
    main()
