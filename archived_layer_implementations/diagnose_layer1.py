"""
Diagnostic: What does Layer 1 return for 'agentic memory methods'?
Also check if the corpus even contains agentic memory papers.
"""
import msgpack
import numpy as np
import sys

def main():
    query = sys.argv[1] if len(sys.argv) > 1 else "agentic memory methods"
    
    # Load chunks
    print("Loading chunks...")
    with open("checkpoints/chunks.msgpack", "rb") as f:
        chunks = msgpack.unpack(f, raw=False)
    print(f"Total chunks: {len(chunks)}")
    
    # ========================================
    # PART 1: Check corpus for agentic memory content
    # ========================================
    print(f"\n{'='*60}")
    print("CORPUS SCAN: Searching for agentic memory papers")
    print(f"{'='*60}")
    
    keywords = [
        "agentic memory", "agent memory", "memory mechanism",
        "reflexion", "memgpt", "memory bank", "mem0",
        "zettelkasten", "memory stream", "generative agents",
        "tool calling", "agent workflow"
    ]
    
    for kw in keywords:
        kw_lower = kw.lower()
        hits = []
        for i, chunk in enumerate(chunks):
            text = chunk.get('text', '').lower()
            if kw_lower in text:
                hits.append(i)
        if hits:
            print(f"  '{kw}': {len(hits)} chunks (first 3: {hits[:3]})")
            # Show a snippet from first hit
            first_text = chunks[hits[0]]['text'][:200]
            print(f"    Sample: {first_text}...")
        else:
            print(f"  '{kw}': 0 chunks")
    
    # ========================================
    # PART 2: Run Layer 1 retrieval and inspect seeds
    # ========================================
    print(f"\n{'='*60}")
    print(f"LAYER 1 RETRIEVAL: '{query}'")
    print(f"{'='*60}")
    
    from pathlib import Path
    from pgvector_retriever import PGVectorRetriever, PGVectorConfig
    
    pg_config = PGVectorConfig(
        db_host="localhost",
        db_port=5432,
        db_name="langchain",
        db_user="langchain",
        db_password="langchain",
        table_name="arxiv_chunks",
        embedding_dim=64,
        embedding_model="minishlab/M2V_base_output",
        bm25_cache_path=Path("bm25_vocab.msgpack"),
        use_lemmatized=False,
    )
    retriever = PGVectorRetriever(pg_config)
    
    # Get raw BM25 results (before GIST)
    print("\n--- RAW BM25 Pool (top 20) ---")
    bm25_raw = retriever._retrieve_bm25(query, limit=20)
    for i, doc in enumerate(bm25_raw[:20]):
        text_preview = doc.content[:120].replace('\n', ' ')
        print(f"  {i+1}. [{doc.doc_id}] bm25={doc.bm25_score:.4f} | {text_preview}")
    
    # Get raw dense results (before GIST)
    print("\n--- RAW DENSE Pool (top 20) ---")
    dense_raw = retriever._retrieve_dense(query, limit=20)
    for i, doc in enumerate(dense_raw[:20]):
        text_preview = doc.content[:120].replace('\n', ' ')
        print(f"  {i+1}. [{doc.doc_id}] dense={doc.dense_score:.4f} | {text_preview}")
    
    # Run full hybrid with GIST (the actual Layer 1)
    print("\n--- GIST HYBRID (Layer 1 output, top 13) ---")
    from simple_hybrid_retriever import SimpleHybridRetriever
    hybrid = SimpleHybridRetriever(pg_config)
    results = hybrid.retrieve(query, top_k=13)
    for i, doc in enumerate(results):
        text_preview = doc.content[:120].replace('\n', ' ')
        print(f"  {i+1}. [{doc.doc_id}] rrf={doc.rrf_score:.4f} bm25={doc.bm25_score or 0:.4f} dense={doc.dense_score or 0:.4f} | {text_preview}")


if __name__ == '__main__':
    main()
