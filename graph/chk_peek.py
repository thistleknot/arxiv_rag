import msgpack
with open(r'C:\Users\user\arxiv_id_lists\checkpoints\bio_triplets_checkpoint.msgpack','rb') as f:
    data = msgpack.unpackb(f.read(), raw=False)
print(f'Records: {len(data)}')
r = data[0]
print(f'Keys: {list(r.keys())}')
trips = r.get("triplets",[])
print(f'Triplets in record[0]: {len(trips)}')
t = trips[0] if trips else {}
print(f'Triplet keys: {list(t.keys())}')
print(f'Sample: {t}')
total_trips = sum(len(r.get("triplets",[])) for r in data)
print(f'Total triplets: {total_trips}')
