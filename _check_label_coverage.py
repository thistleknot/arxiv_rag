import msgpack

with open('data/bio_training_test10.msgpack', 'rb') as f:
    data = msgpack.unpackb(f.read(), raw=False)

total_examples = len(data['training_data'])
total_triplets = sum(len(ex['triplets']) for ex in data['training_data'])

# Count examples with no labels at all (failed matching)
zero_label_examples = 0
partial_label_examples = 0
full_label_examples = 0

for ex in data['training_data']:
    # Check if any label has a 1
    has_any_label = any(
        any(ex['labels'][label_name])
        for label_name in ['B-SUBJ', 'I-SUBJ', 'B-PRED', 'I-PRED', 'B-OBJ', 'I-OBJ']
    )
    
    # Count number of triplets
    n_triplets = len(ex['triplets'])
    
    # Count number of label types present
    label_types_present = sum(
        1 for label_name in ['B-SUBJ', 'I-SUBJ', 'B-PRED', 'I-PRED', 'B-OBJ', 'I-OBJ']
        if any(ex['labels'][label_name])
    )
    
    if not has_any_label:
        zero_label_examples += 1
    elif label_types_present < 6:
        partial_label_examples += 1
    else:
        full_label_examples += 1

print(f"Total examples: {total_examples}")
print(f"Total triplets: {total_triplets}")
print(f"\nLabel matching success:")
print(f"  Zero labels (complete failure): {zero_label_examples} ({zero_label_examples/total_examples*100:.1f}%)")
print(f"  Partial labels (some spans matched): {partial_label_examples} ({partial_label_examples/total_examples*100:.1f}%)")
print(f"  Full labels (all spans matched): {full_label_examples} ({full_label_examples/total_examples*100:.1f}%)")

# Show examples of failed matching
print(f"\nExamples with zero labels:")
for i, ex in enumerate(data['training_data']):
    has_any_label = any(
        any(ex['labels'][label_name])
        for label_name in ['B-SUBJ', 'I-SUBJ', 'B-PRED', 'I-PRED', 'B-OBJ', 'I-OBJ']
    )
    if not has_any_label:
        print(f"\n{i+1}. Triplets: {ex['triplets']}")
        print(f"   Sentence: {ex['sentence'][:100]}...")
