import msgpack
with open('checkpoints/quotes_triplets.msgpack', 'rb') as f:
    records = msgpack.unpack(f, raw=False)
HOLLOW = ('life in', 'way in', 'time in', 'place in', 'state in', 'role in', 'part in', 'form in', 'manner in')
life_in = []
for r in records:
    for t in r.get('triplets', []):
        for field in ('subject', 'object'):
            v = (t.get(field) or '').lower()
            if any(v.startswith(h) for h in HOLLOW):
                life_in.append((field, v, t.get('predicate', '')))
seen = sorted(set((f2, v, p) for f2, v, p in life_in))
print(f'Hollow-head spans in msgpack: {len(seen)}')
for f2, v, p in seen:
    print(f'  [{f2}] "{v}"  pred={p}')
