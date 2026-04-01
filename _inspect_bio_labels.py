import msgpack

with open("checkpoints/quotes_bio_training.msgpack", "rb") as f:
    raw = msgpack.unpack(f, raw=False)

data = raw["training_data"]

# Distinguish true quote wrappers vs. apostrophes in contractions
open_quotes  = {'\u201c', '\u201d', '\u2018', '"'}   # standalone quote wrappers
apostrophes  = {'\u2019', "'"}                         # apostrophe / right-single (contractions)
entity_keys  = ('B-SUBJ', 'I-SUBJ', 'B-OBJ', 'I-OBJ')

quote_hits = []   # opening/closing quote labeled as entity
apos_hits  = 0    # apostrophe inside contraction span (legitimate)

for ex in data:
    toks = ex["tokens"]
    labs = ex["labels"]
    for i, t in enumerate(toks):
        for k in entity_keys:
            vec = labs.get(k, [])
            if not (i < len(vec) and vec[i]):
                continue
            if t in open_quotes:
                ctx = toks[max(0, i-2):i+5]
                quote_hits.append((t, k, ctx))
            elif t in apostrophes:
                apos_hits += 1
            break

print("=== Opening/closing quote chars labeled as entity ===")
for t, k, ctx in quote_hits[:12]:
    print(f"  {repr(t):8s}  {k:8s}  {ctx}")
print(f"\nTotal quote-wrapper entity hits : {len(quote_hits)}")
print(f"Apostrophe-in-contraction hits  : {apos_hits}  (these are fine — interior of span)")
print(f"\nTotal training examples         : {len(data)}")
