import sys
sys.path.insert(0, r'C:\Users\user\arxiv_id_lists')
from graph.graph_retriever import GraphRetriever

r = GraphRetriever()
results = r.retrieve('anger wisdom pain', top_k=10)
for t in results:
    print(f"  {t[0]} | {t[1]} | {t[2]}")
