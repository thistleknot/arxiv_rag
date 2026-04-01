"""Smoke test: Stage 7 KG extraction with real utility strings from CSV."""
from graph.graph_retriever import GraphRetriever
from syllogism_retriever import _coerce_utility
import csv
import time

csv_path = "papers/post_processed/arxiv_data_with_analysis_cleaned.csv"
samples = {}
with open(csv_path, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        u = row.get("utility", "")
        if u:
            arxiv_id = row["arxiv_id"].strip('"')
            samples[arxiv_id] = _coerce_utility(u)
        if len(samples) >= 5:
            break

print("Sample utility lengths:", {k: len(v) for k, v in samples.items()})
print()

gr = GraphRetriever()
t0 = time.time()
result = gr.retrieve_context(
    query="transformer attention mechanism",
    top_k=10,
    paper_texts=samples,
)
elapsed = time.time() - t0
print(f"Done in {elapsed:.1f}s")
print("Result:")
print(result if result else "(empty)")
