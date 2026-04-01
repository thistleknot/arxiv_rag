import msgpack

# Load training data
with open('data/bio_training_test10.msgpack', 'rb') as f:
    data = msgpack.unpackb(f.read())

training_data = data['training_data']

print("Checking examples with missing subjects/objects:\n")
print("="*80)

for i, example in enumerate(training_data):
    labels = example['labels']
    
    has_subj = any(labels['B-SUBJ'])
    has_pred = any(labels['B-PRED'])
    has_obj = any(labels['B-OBJ'])
    
    # Show examples missing subject or object
    if not has_subj or not has_obj:
        print(f"\nExample {i}:")
        print(f"  Sentence: {example['sentence'][:100]}...")
        print(f"  Raw triplets from Stanza:")
        for triplet in example.get('triplets', []):
            print(f"    S: {triplet.get('subject', '?')}")
            print(f"    P: {triplet.get('predicate', '?')}")
            print(f"    O: {triplet.get('object', '?')}")
        print(f"  Missing: {'Subject' if not has_subj else ''} {'Object' if not has_obj else ''}")
        print(f"  Has labels: S={has_subj}, P={has_pred}, O={has_obj}")

print("\n" + "="*80)
print("\nChecking weird predicates:\n")

for i, example in enumerate(training_data):
    tokens = example['tokens']
    labels = example['labels']
    
    # Extract predicate tokens
    pred_tokens = []
    for j, token in enumerate(tokens):
        if labels['B-PRED'][j] == 1 or labels['I-PRED'][j] == 1:
            pred_tokens.append(token)
    
    pred_text = ' '.join(pred_tokens)
    
    # Flag multi-word predicates or unusual ones
    if len(pred_tokens) > 2 or any(word in pred_text for word in ['through', 'with', 'from', 'conclude']):
        print(f"\nExample {i}: {pred_text}")
        print(f"  Raw triplet: {example.get('triplets', [{}])[0].get('predicate', 'N/A')}")
        print(f"  Sentence: {example['sentence'][:80]}...")
