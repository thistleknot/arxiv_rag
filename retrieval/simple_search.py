"""Simple search script for pgvector_retriever"""
import sys
from pgvector_retriever import PGVectorRetriever, PGVectorConfig

# Get query from command line
if len(sys.argv) < 2:
    print("Usage: python simple_search.py \"your query here\" [top_k]")
    sys.exit(1)

query = sys.argv[1]
top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 21

# Configure database connection
config = PGVectorConfig(
    db_host="localhost",
    db_port=5432,
    db_name="postgres",
    db_user="postgres",
    db_password="postgres",
    table_name="arxiv_papers_lemma_fullembed"
)

# Search
print(f"Searching for: '{query}' (top_k={top_k})")
print("="*70)

retriever = PGVectorRetriever(config)
results = retriever.search(query, top_k=top_k)

# Display results
print(f"\nFound {len(results)} papers\n")
for i, doc in enumerate(results[:top_k], 1):
    print(f"[{i}] {doc.metadata.get('paper_id', doc.doc_id)}")
    print(f"    Score: {doc.final_score:.4f}")
    print(f"    {doc.content[:150]}...\n")
