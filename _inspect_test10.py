import msgpack

with open('data/bio_training_test10.msgpack', 'rb') as f:
    data = msgpack.unpackb(f.read(), raw=False)

print(f"Total examples: {len(data['training_data'])}")
print(f"Label names: {data['label_names']}")
print(f"\nStats: {data.get('stats', {})}")

print(f"\nFirst 5 triplet samples:")
for i, ex in enumerate(data['training_data'][:5]):
    print(f"{i+1}. {ex['triplets']}")

print(f"\nLast 3 triplet samples:")
for i, ex in enumerate(data['training_data'][-3:]):
    print(f"{len(data['training_data'])-2+i}. {ex['triplets']}")
