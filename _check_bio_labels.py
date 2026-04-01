import msgpack

with open('data/bio_training_test10.msgpack', 'rb') as f:
    data = msgpack.unpackb(f.read(), raw=False)

# Check example 1 in detail
ex = data['training_data'][0]
print(f"Sentence: {ex['sentence']}")
print(f"\nTriplets: {ex['triplets']}")
print(f"\nTokens: {ex['tokens']}")
print(f"\nB-SUBJ labels: {ex['labels']['B-SUBJ']}")
print(f"\nToken-label pairs for subjects:")
for i, tok in enumerate(ex['tokens']):
    if ex['labels']['B-SUBJ'][i] == 1 or ex['labels']['I-SUBJ'][i] == 1:
        tag = 'B-SUBJ' if ex['labels']['B-SUBJ'][i] == 1 else 'I-SUBJ'
        print(f"  {i}: '{tok}' → {tag}")

# Check example 2 with "existing ke methods"
ex2 = data['training_data'][1]
print(f"\n\nExample 2:")
print(f"Sentence: {ex2['sentence']}")
print(f"Triplets: {ex2['triplets']}")
print(f"\nToken-label pairs for subjects:")
for i, tok in enumerate(ex2['tokens']):
    if ex2['labels']['B-SUBJ'][i] == 1 or ex2['labels']['I-SUBJ'][i] == 1:
        tag = 'B-SUBJ' if ex2['labels']['B-SUBJ'][i] == 1 else 'I-SUBJ'
        print(f"  {i}: '{tok}' → {tag}")
