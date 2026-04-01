"""Convert GLiNER NER spans to proper BIO labels"""
import json
import numpy as np
from typing import List, Dict

def spans_to_bio_labels(tokens: List[str], ner_spans: List[List]) -> Dict[str, List[int]]:
    """Convert NER span annotations to BIO multi-hot labels.
    
    Args:
        tokens: List of BERT tokens
        ner_spans: List of [start_idx, end_idx, "ROLE"] annotations
    
    Returns:
        {
            'B-SUBJ': [0, 1, 0, ...],  # Binary for each token
            'I-SUBJ': [0, 0, 1, ...],
            'B-PRED': [...],
            'I-PRED': [...],
            'B-OBJ': [...],
            'I-OBJ': [...]
        }
    """
    # Initialize all labels to 0
    labels = {
        'B-SUBJ': [0] * len(tokens),
        'I-SUBJ': [0] * len(tokens),
        'B-PRED': [0] * len(tokens),
        'I-PRED': [0] * len(tokens),
        'B-OBJ': [0] * len(tokens),
        'I-OBJ': [0] * len(tokens),
    }
    
    # Process each NER span
    for start_idx, end_idx, role in ner_spans:
        # Ensure valid range
        if start_idx < 0 or end_idx >= len(tokens):
            continue
        
        # Convert role to label key
        role_map = {
            'SUBJ': ('B-SUBJ', 'I-SUBJ'),
            'PRED': ('B-PRED', 'I-PRED'),
            'OBJ': ('B-OBJ', 'I-OBJ'),
        }
        
        if role not in role_map:
            continue
        
        b_label, i_label = role_map[role]
        
        # First token gets B- label
        labels[b_label][start_idx] = 1
        
        # Continuation tokens get I- labels
        for idx in range(start_idx + 1, end_idx + 1):
            labels[i_label][idx] = 1
    
    return labels

# Test on Example 1
print("="*80)
print("CONVERTING NER SPANS TO BIO LABELS")
print("="*80)

with open('bio_training_250chunks_gliner.json', 'r') as f:
    examples = [json.loads(line) for line in f]

ex = examples[0]
tokens = ex['tokenized_text']
ner_spans = ex['ner']

print(f"\nExample 1:")
print(f"Tokens: {tokens}")
print(f"\nNER Spans:")
for span in ner_spans:
    start, end, role = span
    span_text = ' '.join(tokens[start:end+1]).replace(' ##', '')
    print(f"  {span}: {role:4s} = '{span_text}'")

# Convert to BIO
bio_labels = spans_to_bio_labels(tokens, ner_spans)

print(f"\nBIO Labels:")
label_names = ['B-SUBJ', 'I-SUBJ', 'B-PRED', 'I-PRED', 'B-OBJ', 'I-OBJ']
for i, token in enumerate(tokens):
    active = [name for name in label_names if bio_labels[name][i] == 1]
    status = ', '.join(active) if active else 'O'
    print(f"  {i:2d}  {token:15s}  {status}")

# Verify "deep learning models" gets proper BIO
print("\n" + "="*80)
print("VERIFICATION: 'deep learning models' [19, 21]")
print("="*80)
print(f"Token 19 'deep':      B-SUBJ={bio_labels['B-SUBJ'][19]} (expected 1)")
print(f"Token 20 'learning':  I-SUBJ={bio_labels['I-SUBJ'][20]} (expected 1)")
print(f"Token 21 'models':    I-SUBJ={bio_labels['I-SUBJ'][21]} (expected 1)")

if (bio_labels['B-SUBJ'][19] == 1 and 
    bio_labels['I-SUBJ'][20] == 1 and 
    bio_labels['I-SUBJ'][21] == 1):
    print("\n✓ SUCCESS! Proper BIO tags for contiguous entity")
else:
    print("\n✗ FAILED - Check logic")

# Convert all 416 examples and save
print("\n" + "="*80)
print("CONVERTING ALL 416 EXAMPLES")
print("="*80)

all_data = []

for i, ex in enumerate(examples):
    tokens = ex['tokenized_text']
    ner_spans = ex['ner']
    
    bio_labels = spans_to_bio_labels(tokens, ner_spans)
    
    training_example = {
        'sentence': ' '.join(tokens).replace(' ##', ''),  # Reconstruct sentence
        'tokens': tokens,
        'labels': bio_labels,
        'ner_spans': ner_spans,  # Keep original spans for reference
    }
    
    all_data.append(training_example)

# Save in msgpack format for compatibility
import msgpack

with open('bio_training_416_gliner_fixed.msgpack', 'wb') as f:
    data = {'training_data': all_data}
    f.write(msgpack.packb(data, use_bin_type=True))

print(f"\n✓ Saved 416 examples to bio_training_416_gliner_fixed.msgpack")

# Show statistics
print(f"\nStatistics:")
labeled_examples = sum(1 for ex in all_data if any(sum(bio_labels[lbl]) for lbl in bio_labels for bio_labels in [ex['labels']]))
avg_labels = np.mean([sum(1 for lbl in ex['labels'].values() for val in lbl if val == 1) for ex in all_data])
avg_spans = np.mean([len(ex['ner_spans']) for ex in all_data])

print(f"  Avg labels per example: {avg_labels:.1f}")
print(f"  Avg NER spans per example: {avg_spans:.1f}")
print(f"\nFormat:")
print(f"  Example 1 keys: {list(all_data[0].keys())}")
print(f"  Label format: {all_data[0]['labels']}")
