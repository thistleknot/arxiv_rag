"""Warm KG cache for the 33 newly-ingested papers only."""
import json, sys, time
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parent
KG_DIR = ROOT / "graph" / "kg_cache"
KG_DIR.mkdir(parents=True, exist_ok=True)
CSV = ROOT / "papers" / "post_processed" / "arxiv_data_with_analysis_cleaned.csv"

new_ids = {s.strip().strip('"') for s in json.loads(open(ROOT / "_new_ids.json").read())}
df = pd.read_csv(CSV)
df["arxiv_id"] = df["arxiv_id"].astype(str).str.strip().str.strip('"')
sub = df[df["arxiv_id"].isin(new_ids)].to_dict("records")
print(f"Targeted papers: {len(sub)} / {len(new_ids)} requested")

uncached = [p for p in sub if not (KG_DIR / f"{p['arxiv_id'].replace('/', '_')}.json").exists()]
print(f"Already cached: {len(sub) - len(uncached)}, remaining: {len(uncached)}")
if not uncached:
    sys.exit(0)

from graph.graph_retriever import GraphRetriever
gr = GraphRetriever(
    ollama_host="http://127.0.0.1:11434",
    model="qwen2.5-coder:1.5b",
    cache_dir=KG_DIR,
)

t0 = time.time()
ok = err = 0
for i, p in enumerate(uncached, 1):
    aid = p["arxiv_id"]
    try:
        gr._extract_paper_graph(aid, p.get("utility", ""))
        ok += 1
        status = "ok"
    except Exception as e:
        err += 1
        status = f"FAIL: {e}"
    elapsed = time.time() - t0
    avg = elapsed / i
    eta = avg * (len(uncached) - i) / 60
    print(f"[{i}/{len(uncached)}] {aid} -> {status}  ok={ok} err={err}  ETA {eta:.1f} min", flush=True)

print(f"\nDone in {(time.time()-t0)/60:.1f} min  ok={ok} err={err}")
