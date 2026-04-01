"""
Graph transformer pipeline for structured knowledge retrieval.

Pipeline steps:
    1. normalize_entities.py  — extract + cluster entity/predicate vocabulary
    2. build_kg_dataset.py    — construct PyG Data object
    3. train_gat.py           — train GATv2Encoder via link prediction
    4. graph_retriever.py     — query-time retrieval and context formatting

Typical usage:
    from graph.graph_retriever import GraphRetriever
    retriever = GraphRetriever()
    context = retriever.retrieve_context("transformer attention", top_k=15)
"""
from graph.graph_retriever import GraphRetriever

__all__ = ["GraphRetriever"]
