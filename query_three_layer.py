"""
DEPRECATED: This query script has been archived.

This script used the standalone msgpack-based three-layer GIST retriever,
which has been superseded by the production PostgreSQL + pgvector backend.

ARCHIVED LOCATION:
    archived_layer_implementations/standalone_gist/query_three_layer.py

PRODUCTION REPLACEMENT:
    query_arxiv.py  (main query interface)
    query_arxiv_ecdf.py  (ECDF-weighted RRF fusion)

USE INSTEAD:
    python query_arxiv.py "your search query here"

MIGRATION NOTES:
    - Production system uses PostgreSQL + pgvector (not msgpack files)
    - ECDF weighting is enabled between Layer 1 → Layer 2
    - HNSW index for dense vectors, GIN index for BM25
    - Scalable and actively maintained

For historical reference, see the archived script in:
    archived_layer_implementations/standalone_gist/query_three_layer.py

Documentation: README.md "Production System Clarification" section
"""

print("=" * 70)
print("DEPRECATED: query_three_layer.py has been archived")
print("=" * 70)
print()
print("This script used the standalone msgpack-based three-layer retriever,")
print("which has been superseded by the PostgreSQL + pgvector backend.")
print()
print("USE INSTEAD:")
print("  python query_arxiv.py \"your search query here\"")
print()
print("ARCHIVED LOCATION:")
print("  archived_layer_implementations/standalone_gist/query_three_layer.py")
print()
print("See README.md 'Production System Clarification' for details.")
print("=" * 70)
