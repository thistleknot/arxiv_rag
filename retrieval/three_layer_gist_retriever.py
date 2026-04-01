"""
DEPRECATED: This module has been archived.

The standalone msgpack-based three-layer GIST retriever has been superseded
by the production PostgreSQL + pgvector backend.

ARCHIVED LOCATION:
    archived_layer_implementations/standalone_gist/three_layer_gist_retriever.py

PRODUCTION REPLACEMENT:
    query_modules/pgvector_retriever.py

KEY DIFFERENCES:
    Standalone (Archived)          | Production (pgvector_retriever.py)
    ------------------------------|-----------------------------------
    Msgpack files                 | PostgreSQL + pgvector
    No ECDF weighting             | ECDF-weighted Layer 2 queries
    In-memory only                | Scalable database backend
    Simple RRF fusion             | ECDF-weighted RRF
    
MIGRATION GUIDE:
    Instead of:
        from three_layer_gist_retriever import ThreeLayerGISTRetriever
        retriever = ThreeLayerGISTRetriever(...)
    
    Use:
        from query_modules.pgvector_retriever import PGVectorRetriever
        from arxiv_retriever import ArxivRetriever
        
        retriever = ArxivRetriever(top_k_final=20)
        results = retriever.retrieve(query)
    
    See query_arxiv.py for complete usage example.

DOCUMENTATION:
    - README.md: "Production System Clarification" section
    - README.md: Feature 19 (GIST Diversity Selection Algorithm)
    - archived_layer_implementations/standalone_gist/README_DEPRECATED.md

For historical reference, see the archived implementation in:
    archived_layer_implementations/standalone_gist/
"""

raise ImportError(
    "three_layer_gist_retriever has been archived. "
    "Use query_modules.pgvector_retriever.PGVectorRetriever instead. "
    "See docstring above for migration guide."
)
