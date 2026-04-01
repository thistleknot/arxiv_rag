"""
run_arxiv.py — root-level entry point for the arXiv syllogism pipeline.

Delegates entirely to arxiv_pipeline/run.py so the pipeline can be
invoked from the arxiv_id_lists root (or any directory via absolute path)
without knowing the internal subfolder structure.

Usage (from arxiv_id_lists root):
    python run_arxiv.py "methods for long context large language models"
    python run_arxiv.py "transformer attention" --top_k 10 --output results.md
    python run_arxiv.py "your query" --warmup --warmup_limit 20
    python run_arxiv.py --dry_run
"""
import sys
from pathlib import Path

# Ensure arxiv_id_lists/ is on sys.path so arxiv_pipeline.* is importable
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from arxiv_pipeline.run import main

if __name__ == "__main__":
    main()
