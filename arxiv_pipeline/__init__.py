"""
arxiv_pipeline — modular arXiv syllogism retrieval pipeline.

Exposes the main entry points for the end-to-end workflow:
  1. arxiv_gap_extractor  — extract missing paper IDs from llm.txt
  2. docling_pipeline     — convert PDFs to markdown via Docling
  3. warm_kg_cache        — pre-extract KNWLER triplets to graph/kg_cache/
  4. syllogism_retriever  — 9-stage query → ranked papers + synthesis report
  5. run                  — orchestrating entry point (warmup + retrieve)

Usage from arxiv_id_lists root:
    python run_arxiv.py "your query here"
    python run_arxiv.py "transformer attention" --top_k 10
    python run_arxiv.py "your query" --warmup --dry_run
"""
from arxiv_pipeline.syllogism_retriever import SyllogismRetriever
from arxiv_pipeline.run import main as run_main

__all__ = ["SyllogismRetriever", "run_main"]
