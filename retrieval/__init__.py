"""retrieval — arxiv retrieval sub-package.

Adds this directory to sys.path so that all intra-package bare imports
(e.g. ``from pgvector_retriever import ...``) continue to resolve correctly
whether the module is imported as ``retrieval.X`` or run as a script directly.
"""
import sys
from pathlib import Path

_here = str(Path(__file__).parent)
if _here not in sys.path:
    sys.path.insert(0, _here)

from .gist_retriever import GISTRetriever, GISTConfig          # noqa: E402,F401
from .pgvector_retriever import PGVectorRetriever, PGVectorConfig  # noqa: E402,F401
from .base_gist_retriever import BaseGISTRetriever              # noqa: E402,F401

__all__ = [
    "GISTRetriever", "GISTConfig",
    "PGVectorRetriever", "PGVectorConfig",
    "BaseGISTRetriever",
]
