"""
Query Quotes Database with GraphRAG

CLI for searching quotes using GraphRAG retrieval:
- BM25 + GIST hybrid retrieval
- Graph expansion from seeds
- Late interaction on full quotes
"""

import argparse
from pathlib import Path
from quotes_retriever import QuotesRetriever
from pgvector_retriever import PGVectorConfig


def main():
    parser = argparse.ArgumentParser(
        description="Search quotes database with GraphRAG"
    )
    parser.add_argument(
        'query',
        type=str,
        help='Search query'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=10,
        help='Number of quotes to return (default: 10)'
    )
    parser.add_argument(
        '--table',
        type=str,
        default='quotes',
        help='Table name (default: quotes)'
    )
    
    args = parser.parse_args()
    query = args.query
    top_k = args.top_k
    
    # Configuration
    config = PGVectorConfig(
        db_host='localhost',
        db_port=5432,
        db_name='postgres',
        db_user='postgres',
        db_password='postgres',
        table_name=args.table,
        embedding_model='model2vec_jina',
        embedding_dim=64,
        bm25_cache_path=Path('data/quotes_bm25_vocab.msgpack'),
        use_lemmatized=False,  # Quotes use full BERT vocab
        use_full_embed=False,   # Use GIST
        use_colbert=False       # No ColBERT for now (just hybrid+graph)
    )
    
    print(f"\n{'='*80}")
    print(f"QUOTES GRAPHRAG SEARCH")
    print(f"{'='*80}")
    print(f"Query: \"{query}\"")
    print(f"Top-K: {top_k}")
    print(f"Table: {args.table}")
    print(f"{'='*80}\n")
    
    # Initialize retriever
    retriever = QuotesRetriever(config)
    
    # Search
    print("Searching...")
    results = retriever.search(query, top_k=top_k)
    
    # Display results
    print(f"\n{'='*80}")
    print(f"RESULTS: {len(results)} quotes")
    print(f"{'='*80}\n")
    
    for i, result in enumerate(results, 1):
        score = result.final_score if hasattr(result, 'final_score') else result.rrf_score
        
        print(f"[{i}] {result.doc_id} - Score: {score:.4f}")
        print(f"    \"{result.content}\"")
        print()


if __name__ == '__main__':
    main()
