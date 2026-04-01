import msgpack

# Load v3 data
with open('data/bio_training_test10_v3.msgpack', 'rb') as f:
    data = msgpack.unpackb(f.read(), raw=False)

training_data = data['training_data']

# Check example 4 (which shows "may be cumbersome" in CSV)
ex = training_data[4]
print("Example 4:")
print(f"Sentence: {ex['sentence']}")
print(f"\nRaw triplets (cleaned by clean_triplets):")
for t in ex['triplets']:
    print(f"  S: '{t['subject']}' | P: '{t['predicate']}' | O: '{t['object']}'")

print(f"\nLabels dict:")
tokens = ex['tokens']
labels = ex['labels']

# Find labeled tokens
subj_tokens = [tokens[i] for i in range(len(tokens)) if labels['B-SUBJ'][i] == 1 or labels['I-SUBJ'][i] == 1]
pred_tokens = [tokens[i] for i in range(len(tokens)) if labels['B-PRED'][i] == 1 or labels['I-PRED'][i] == 1]
obj_tokens = [tokens[i] for i in range(len(tokens)) if labels['B-OBJ'][i] == 1 or labels['I-OBJ'][i] == 1]

print(f"Subject tokens: {subj_tokens}")
print(f"Predicate tokens: {pred_tokens}")
print(f"Object tokens: {obj_tokens}")

# Show WHY these tokens were labeled
print(f"\nOriginal sentence:")
print(ex['sentence'])

print("\n\nExample 18:")
ex = training_data[18]
print(f"Sentence: {ex['sentence']}")
print(f"\nRaw triplets (cleaned):")
for t in ex['triplets']:
    print(f"  S: '{t['subject']}' | P: '{t['predicate']}' | O: '{t['object']}'")

subj_tokens = [tokens[i] for i in range(len(ex['tokens'])) if ex['labels']['B-SUBJ'][i] == 1 or ex['labels']['I-SUBJ'][i] == 1]
pred_tokens = [tokens[i] for i in range(len(ex['tokens'])) if ex['labels']['B-PRED'][i] == 1 or ex['labels']['I-PRED'][i] == 1]
obj_tokens = [tokens[i] for i in range(len(ex['tokens'])) if ex['labels']['B-OBJ'][i] == 1 or ex['labels']['I-OBJ'][i] == 1]

print(f"Subject tokens: {subj_tokens}")
print(f"Predicate tokens: {pred_tokens}")
print(f"Object tokens: {obj_tokens}")
