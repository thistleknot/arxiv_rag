"""
paper_loader.py — Load full paper markdown for surviving arxiv IDs.

Core Thesis:
    After retrieval and entailment filtering, load the full structured markdown
    for each surviving paper so downstream reasoning (ReAct agent, thesis synthesis)
    can access complete content rather than just the utility snippet.

File naming convention:
    arxiv_id '1504.04788'  →  papers/post_processed/1504_04788.md
    Dots are replaced with underscores to match the on-disk filenames.
"""

from __future__ import annotations

import pathlib
from typing import Dict, List, Optional

_PAPERS_DIR = pathlib.Path(__file__).parent.parent / "papers" / "post_processed"


def _arxiv_to_filename(arxiv_id: str) -> str:
    """'1504.04788' → '1504_04788.md'"""
    return arxiv_id.replace(".", "_") + ".md"


def load_papers(
    arxiv_ids:  List[str],
    papers_dir: Optional[pathlib.Path] = None,
) -> Dict[str, str]:
    """
    Load full markdown content for each arxiv_id.

    Args:
        arxiv_ids:  List of arxiv IDs, e.g. ['1504.04788', '1701.06538'].
        papers_dir: Override the default papers/post_processed directory.

    Returns:
        Dict[arxiv_id, markdown_content] — only IDs with a corresponding .md file.
        Missing files are silently skipped.
    """
    base   = papers_dir or _PAPERS_DIR
    result: Dict[str, str] = {}
    for arxiv_id in arxiv_ids:
        path = base / _arxiv_to_filename(arxiv_id)
        if path.exists():
            content = path.read_text(encoding="utf-8")
            cut = content.find("\n## References")
            if cut == -1:
                cut = content.lower().find("\n## references")
            if cut != -1:
                content = content[:cut]
            result[arxiv_id] = content
    return result
