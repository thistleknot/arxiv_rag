import msgpack

with open('data/bio_training_test10.msgpack', 'rb') as f:
    data = msgpack.unpackb(f.read(), raw=False)

print(f"Total examples: {len(data['training_data'])}\n")
print("=" * 100)

for i, ex in enumerate(data['training_data'][:10]):  # First 10 examples
    tokens = ex['tokens']
    labels = ex['labels']
    
    # Extract tokens for each entity type
    subject_tokens = []
    predicate_tokens = []
    object_tokens = []
    
    for idx, token in enumerate(tokens):
        if labels['B-SUBJ'][idx] == 1 or labels['I-SUBJ'][idx] == 1:
            subject_tokens.append(token)
        if labels['B-PRED'][idx] == 1 or labels['I-PRED'][idx] == 1:
            predicate_tokens.append(token)
        if labels['B-OBJ'][idx] == 1 or labels['I-OBJ'][idx] == 1:
            object_tokens.append(token)
    
    subject = ' '.join(subject_tokens) if subject_tokens else '[none]'
    predicate = ' '.join(predicate_tokens) if predicate_tokens else '[none]'
    obj = ' '.join(object_tokens) if object_tokens else '[none]'
    
    print(f"{i+1}. ORIGINAL SENTENCE:")
    print(f"   {ex['sentence']}")
    print(f"\n   STANZA TRIPLETS (what was extracted):")
    for triplet in ex['triplets']:
        print(f"   - S: '{triplet['subject']}' | P: '{triplet['predicate']}' | O: '{triplet['object']}'")
    print(f"\n   BIO LABELED TOKENS (what matched):")
    print(f"   - SUBJECT: {subject}")
    print(f"   - PREDICATE: {predicate}")
    print(f"   - OBJECT: {obj}")
    print("\n" + "=" * 100)
