"""Debug fuzzy matching - test specific failing cases"""
import msgpack

# Load the data
with open('data/bio_training_test10_fuzzy.msgpack', 'rb') as f:
    data = msgpack.unpackb(f.read(), raw=False)

training = data['training_data']

# Example 6 from diagnostic - should have object but doesn't
example_6 = None
for i, ex in enumerate(training):
    if 'ke provides fine-grained controllability' in ex['sentence'].lower():
        example_6 = ex
        print(f"Found example at index {i}")
        break

if example_6:
    print("\n" + "="*80)
    print("Example 6 Analysis (should have object but doesn't)")
    print("="*80)
    print(f"\nSentence: {example_6['sentence'][:100]}...")
    print(f"\nRaw triplets:")
    for triplet in example_6.get('triplets', []):
        print(f"  S: {triplet['subject']}")
        print(f"  P: {triplet['predicate']}")
        print(f"  O: {triplet['object']}")
    
    print(f"\nTokens: {example_6['tokens'][:20]}...")
    
    # Check labels
    has_subj = any(example_6['labels']['B-SUBJ'])
    has_pred = any(example_6['labels']['B-PRED'])
    has_obj = any(example_6['labels']['B-OBJ'])
    
    print(f"\nLabels:")
    print(f"  Has subject: {has_subj}")
    print(f"  Has predicate: {has_pred}")
    print(f"  Has object: {has_obj}")
    
    # Show where labels are 1
    if has_subj:
        subj_indices = [i for i, v in enumerate(example_6['labels']['B-SUBJ']) if v == 1]
        print(f"  Subject tokens: {[example_6['tokens'][i] for i in subj_indices]}")
    if has_pred:
        pred_indices = [i for i, v in enumerate(example_6['labels']['B-PRED']) if v == 1]
        print(f"  Predicate tokens: {[example_6['tokens'][i] for i in pred_indices]}")
    if has_obj:
        obj_indices = [i for i, v in enumerate(example_6['labels']['B-OBJ']) if v == 1]
        print(f"  Object tokens: {[example_6['tokens'][i] for i in obj_indices]}")
    
    # Check normalized versions
    def normalize_for_matching(text):
        normalized = text.lower()
        normalized = normalized.replace(' - ', '-')
        normalized = normalized.replace(' , ', ',')
        normalized = normalized.replace(' . ', '.')
        normalized = normalized.replace(' ; ', ';')
        normalized = normalized.replace(' : ', ':')
        return normalized
    
    sent_lower = example_6['sentence'].lower()
    sent_norm = normalize_for_matching(sent_lower)
    
    obj_raw = example_6['triplets'][0]['object'] if example_6['triplets'] else None
    if obj_raw and obj_raw != '?':
        obj_norm = normalize_for_matching(obj_raw)
        print(f"\n  Object raw: '{obj_raw}'")
        print(f"  Object normalized: '{obj_norm}'")
        print(f"  Found in sentence (exact): {obj_norm in sent_norm}")
        
        # Try word-by-word
        obj_words = obj_norm.split()
        print(f"  Object words ({len(obj_words)}): {obj_words[:5]}...")
        sent_words = sent_norm.split()
        print(f"  Sentence words ({len(sent_words)}): {sent_words[:10]}...")
        
        # Check if first few words found
        for word in obj_words[:3]:
            if word in sent_words:
                idx = sent_words.index(word)
                print(f"  Found '{word}' at position {idx}")
            else:
                print(f"  NOT FOUND: '{word}'")

# Check a few more failing cases
print("\n" + "="*80)
print("Checking other examples with missing objects:")
print("="*80)

missing_count = 0
for i, ex in enumerate(training):
    has_obj = any(ex['labels']['B-OBJ'])
    triplet_has_obj = any(t['object'] != '?' for t in ex.get('triplets', []))
    
    if not has_obj and triplet_has_obj:
        missing_count += 1
        if missing_count <= 5:
            print(f"\nExample {i}:")
            print(f"  Sentence: {ex['sentence'][:80]}...")
            if ex.get('triplets'):
                print(f"  Raw object: '{ex['triplets'][0]['object'][:80]}...'")

print(f"\nTotal examples with object in raw triplet but not labeled: {missing_count}")
