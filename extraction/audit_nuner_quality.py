"""Quality audit of NuNER training data extractions"""
import json
import sys

fname = sys.argv[1] if len(sys.argv) > 1 else 'nuner_training_data.json'
print(f"Auditing: {fname}")
with open(fname) as f:
    data = json.load(f)

print(f"Total examples: {len(data)}")
print()

# Quality checks
issues = {
    'single_token_subject': 0,
    'single_token_object': 0, 
    'no_subject': 0,
    'no_object': 0,
    'no_predicate': 0,
    'object_too_long': 0,  # > 10 tokens = probably garbage
    'subject_too_long': 0,
    'overlapping_roles': 0,
    'good': 0,
}

for i, ex in enumerate(data):
    tokens = ex['tokenized_text']
    ner = ex['ner']
    
    # Separate by role
    subj_idxs = [s for s, e, l in ner if l == 'subject']
    pred_idxs = [s for s, e, l in ner if l == 'predicate']
    obj_idxs = [s for s, e, l in ner if l == 'object']
    
    has_issue = False
    
    if not subj_idxs:
        issues['no_subject'] += 1
        has_issue = True
    if not obj_idxs:
        issues['no_object'] += 1
        has_issue = True
    if not pred_idxs:
        issues['no_predicate'] += 1
        has_issue = True
    
    if len(subj_idxs) == 1:
        issues['single_token_subject'] += 1
        # Not necessarily bad, but worth noting
    
    if len(obj_idxs) == 1:
        issues['single_token_object'] += 1
    
    if len(obj_idxs) > 10:
        issues['object_too_long'] += 1
        has_issue = True
    
    if len(subj_idxs) > 10:
        issues['subject_too_long'] += 1
        has_issue = True
    
    # Check for overlapping token indices across roles
    all_sets = [set(subj_idxs), set(pred_idxs), set(obj_idxs)]
    if (all_sets[0] & all_sets[1]) or (all_sets[0] & all_sets[2]) or (all_sets[1] & all_sets[2]):
        issues['overlapping_roles'] += 1
        has_issue = True
    
    if not has_issue:
        issues['good'] += 1

print("=== QUALITY AUDIT ===")
for k, v in sorted(issues.items(), key=lambda x: -x[1]):
    pct = v / len(data) * 100
    print(f"  {k}: {v} ({pct:.0f}%)")

print()
print("=== SHOWING SUSPICIOUS EXAMPLES ===")
print()

# Show examples with very long objects (likely garbage spans)
print("--- Object > 10 tokens (likely bad span) ---")
shown = 0
for i, ex in enumerate(data):
    tokens = ex['tokenized_text']
    obj_tokens = [tokens[s] for s, e, l in ex['ner'] if l == 'object' and s < len(tokens)]
    subj_tokens = [tokens[s] for s, e, l in ex['ner'] if l == 'subject' and s < len(tokens)]
    pred_tokens = [tokens[s] for s, e, l in ex['ner'] if l == 'predicate' and s < len(tokens)]
    
    if len(obj_tokens) > 10:
        print(f"  Ex {i}: subj=[{' '.join(subj_tokens)}] pred=[{' '.join(pred_tokens)}] obj=[{' '.join(obj_tokens)}]")
        shown += 1
        if shown >= 5:
            break

print()
print("--- Examples with overlapping roles (token assigned multiple roles) ---")
shown = 0
for i, ex in enumerate(data):
    tokens = ex['tokenized_text']
    subj_set = set(s for s, e, l in ex['ner'] if l == 'subject')
    pred_set = set(s for s, e, l in ex['ner'] if l == 'predicate')
    obj_set = set(s for s, e, l in ex['ner'] if l == 'object')
    
    overlap = (subj_set & pred_set) | (subj_set & obj_set) | (pred_set & obj_set)
    if overlap:
        overlap_tokens = [f"{tokens[idx]}(idx={idx})" for idx in sorted(overlap) if idx < len(tokens)]
        print(f"  Ex {i}: overlapping tokens: {overlap_tokens}")
        shown += 1
        if shown >= 5:
            break

print()
print("--- Random sample of 'good' examples ---")
import random
random.seed(42)
good_idxs = []
for i, ex in enumerate(data):
    tokens = ex['tokenized_text']
    subj = [tokens[s] for s, e, l in ex['ner'] if l == 'subject' and s < len(tokens)]
    pred = [tokens[s] for s, e, l in ex['ner'] if l == 'predicate' and s < len(tokens)]
    obj = [tokens[s] for s, e, l in ex['ner'] if l == 'object' and s < len(tokens)]
    
    if 1 < len(subj) <= 8 and 1 <= len(pred) <= 3 and 1 < len(obj) <= 8:
        good_idxs.append(i)

random.shuffle(good_idxs)
print(f"Examples with reasonable span lengths (2-8 subj, 1-3 pred, 2-8 obj): {len(good_idxs)}/{len(data)}")
for idx in good_idxs[:10]:
    ex = data[idx]
    tokens = ex['tokenized_text']
    subj = ' '.join(tokens[s] for s, e, l in ex['ner'] if l == 'subject' and s < len(tokens))
    pred = ' '.join(tokens[s] for s, e, l in ex['ner'] if l == 'predicate' and s < len(tokens))
    obj = ' '.join(tokens[s] for s, e, l in ex['ner'] if l == 'object' and s < len(tokens))
    print(f"  [{subj}] --({pred})--> [{obj}]")
