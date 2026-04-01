"""
Show the retrieved subgraph for a single extracted quote.

Pipeline trace:
  quote text
    → embed → seed entity nodes (cosine sim)
    → 1-hop subgraph extraction (adjacency walk)
    → GATv2 forward  (node embeddings updated by neighborhood)
    → triplet scoring (query · z_subj + query · z_obj)
    → ranked S|P|O output
"""
import sys, msgpack, numpy as np
sys.path.insert(0, r"C:\Users\user\arxiv_id_lists")

from graph.graph_retriever import GraphRetriever

# ─── 1. Pick a quote from the extracted corpus ───────────────────────────────
MSGPACK = r"C:\Users\user\arxiv_id_lists\checkpoints\quotes_triplets.msgpack"
with open(MSGPACK, "rb") as f:
    records = msgpack.unpackb(f.read(), raw=False)

# Find a record with ≥3 triplets for a richer subgraph
record = next(r for r in records if len(r.get("triplets", [])) >= 3)
quote       = record.get("quote", "")
author      = record.get("author", "unknown")
triplets_in = record.get("triplets", [])

print("=" * 70)
print(f"QUOTE  : {quote[:200]}")
print(f"AUTHOR : {author}")
print(f"\nExtracted triplets from this quote ({len(triplets_in)}):")
for t in triplets_in:
    print(f"  S: {t['subject']!r:30s}  P: {t['predicate']!r:25s}  O: {t['object']!r}")

# ─── 2. Load retriever ───────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("Loading GraphRetriever …")
ret = GraphRetriever()

# ─── 3. Embed the quote ──────────────────────────────────────────────────────
query     = quote[:256]
query_emb = ret._embed_query(query)
print(f"\nQuery embedding:  shape={query_emb.shape}  norm={np.linalg.norm(query_emb):.4f}")

# ─── 4. Seed nodes ───────────────────────────────────────────────────────────
N_SEEDS = 8
seed_ids = ret._find_seed_nodes(query_emb, top_k=N_SEEDS)
sims     = ret.entity_embs_n @ query_emb

print(f"\nTop-{N_SEEDS} seed entity nodes (cosine sim to query):")
for rank, eid in enumerate(seed_ids, 1):
    text = ret.entity_id_to_text.get(eid, f"[{eid}]")
    print(f"  {rank:2d}.  sim={sims[eid]:.4f}  [{eid:4d}]  {text!r}")

# ─── 5. 1-hop subgraph ───────────────────────────────────────────────────────
local_triplets, local_nodes, node_remap = ret._extract_subgraph(seed_ids)

print(f"\n1-hop subgraph:")
print(f"  Nodes (entities) : {len(local_nodes):4d}")
print(f"  Edges (triplets) : {len(local_triplets):4d}")

# Sample up to 10 edges
print(f"\n  Sample edges (S --[P]--> O):")
for eid_s, pid, eid_o in local_triplets[:10]:
    s = ret.entity_id_to_text.get(eid_s, f"[{eid_s}]")
    p = ret.pred_id_to_text.get(pid,   f"[{pid}]")
    o = ret.entity_id_to_text.get(eid_o, f"[{eid_o}]")
    print(f"    {s!r:30s}  --[{p}]-->  {o!r}")

# ─── 6. GATv2 forward ─────────────────────────────────────────────────────────
import torch, torch.nn.functional as F

z = ret._run_gat(local_nodes, local_triplets, node_remap)  # (n_nodes, 256)

print(f"\nGATv2 output embeddings:")
print(f"  z shape           : {z.shape}        (nodes × hidden_dim)")
print(f"  z norm range      : [{np.linalg.norm(z, axis=1).min():.4f}, "
                              f"{np.linalg.norm(z, axis=1).max():.4f}]  (should be ~1.0 after L2-norm)")

# Show the seed nodes' GAT embeddings vs raw embeddings (cosine similarity to themselves)
print(f"\n  Seed node embedding shift (raw → GATv2):")
print(f"  {'entity':<30s}  raw·z    (how much GAT changed the representation)")
for eid in seed_ids[:5]:
    if eid not in node_remap:   # isolated node (no neighbors) — skip
        continue
    li  = node_remap[eid]
    raw = ret.entity_embs_n[eid]        # L2-normalised raw, shape (512,)
    gz  = z[li]                          # GATv2 output, shape (256,)
    # project raw into 256-dim via lin_in for a fair comparison
    with torch.no_grad():
        dev = next(ret.model.parameters()).device
        raw_t = torch.from_numpy(ret.entity_embs[eid:eid+1]).to(dev)
        raw_proj = F.normalize(ret.model.lin_in(raw_t), dim=-1).squeeze(0).cpu().numpy()
    cos = float(raw_proj @ gz)
    text = ret.entity_id_to_text.get(eid, f"[{eid}]")
    print(f"  {text!r:<30s}  cos(raw_proj, z)={cos:.4f}")

# ─── 7. Score triplets ───────────────────────────────────────────────────────
with torch.no_grad():
    dev = next(ret.model.parameters()).device
    q_t = torch.from_numpy(query_emb).unsqueeze(0).to(dev)
    query_emb_z = F.relu(ret.model.lin_in(q_t)).squeeze(0).cpu().numpy()
    query_emb_z = query_emb_z / (np.linalg.norm(query_emb_z) + 1e-9)

scored = ret._score_triplets(query_emb_z, z, local_nodes, local_triplets, node_remap)

print(f"\n" + "=" * 70)
print(f"Scored triplets (query·z_subj + query·z_obj), top 15:")
print(f"  {'score':>7s}  {'subject':<28s}  {'predicate':<22s}  object")
print(f"  {'-'*7}  {'-'*28}  {'-'*22}  ------")
seen = set()
shown = 0
for score, eid_s, pid, eid_o in scored:
    key = (eid_s, pid, eid_o)
    if key in seen:
        continue
    seen.add(key)
    s = ret.entity_id_to_text.get(eid_s, f"[{eid_s}]")[:28]
    p = ret.pred_id_to_text.get(pid,   f"[{pid}]")[:22]
    o = ret.entity_id_to_text.get(eid_o, f"[{eid_o}]")[:35]
    print(f"  {score:7.4f}  {s:<28s}  {p:<22s}  {o}")
    shown += 1
    if shown >= 15:
        break
