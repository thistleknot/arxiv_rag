import json, pickle, random
from pathlib import Path

g = Path('graph')
ent_idx  = json.loads((g / 'entity_index.json').read_text())
pred_idx = json.loads((g / 'predicate_index.json').read_text())
tmap = pickle.loads((g / 'triplet_map.pkl').read_bytes())

# entity_index.json schema: {"span_to_id": {span: int}, "n_nodes": int}
span2ent  = ent_idx['span_to_id']
span2pred = pred_idx['span_to_id']

# reverse map: id -> first canonical span
id2ent: dict = {}
for span, nid in span2ent.items():
    if nid not in id2ent:
        id2ent[nid] = span

id2pred: dict = {}
for span, nid in span2pred.items():
    if nid not in id2pred:
        id2pred[nid] = span

all_trips = [(es, p, eo) for trips in tmap.values() for (es, p, eo) in trips]
print(f"Total triplets in map: {len(all_trips):,}")
print(f"Unique entity nodes:   {ent_idx['n_nodes']:,}")
print(f"Unique predicate nodes:{pred_idx['n_nodes']:,}")
print()

print("── 50 random triplets ──────────────────────────────────────────────────")
random.seed(42)
for es, p, eo in random.sample(all_trips, 50):
    s = id2ent.get(es, "?")
    r = id2pred.get(p,  "?")
    o = id2ent.get(eo,  "?")
    print(f"  {s:<40} | {r:<25} | {o}")

print()
print("── Top 20 most-connected entities (hub check) ─────────────────────────")
from collections import Counter
hub = Counter()
for es, _, eo in all_trips:
    hub[id2ent.get(es, "?")] += 1
    hub[id2ent.get(eo, "?")] += 1
for span, cnt in hub.most_common(20):
    print(f"  {cnt:>6}  {span}")
