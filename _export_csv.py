import msgpack
import pandas as pd

# Load training data
with open('data/bio_training_test10_v15.msgpack', 'rb') as f:
    data = msgpack.unpackb(f.read())

training_data = data['training_data']

def merge_wordpieces(tokens):
    """Merge BERT wordpiece tokens back into readable text"""
    if not tokens:
        return ''
    
    merged = []
    for token in tokens:
        if token.startswith('##'):
            # Continuation token - append to previous
            if merged:
                merged[-1] += token[2:]
            else:
                merged.append(token[2:])
        else:
            merged.append(token)
    
    return ' '.join(merged)

# Extract labeled tokens for each SPO component
rows = []
for i, example in enumerate(training_data):
    tokens = example['tokens']
    labels = example['labels']
    
    # Extract subject tokens
    subj_tokens = []
    for j, token in enumerate(tokens):
        if labels['B-SUBJ'][j] == 1 or labels['I-SUBJ'][j] == 1:
            subj_tokens.append(token)
    
    # Extract predicate tokens
    pred_tokens = []
    for j, token in enumerate(tokens):
        if labels['B-PRED'][j] == 1 or labels['I-PRED'][j] == 1:
            pred_tokens.append(token)
    
    # Extract object tokens
    obj_tokens = []
    for j, token in enumerate(tokens):
        if labels['B-OBJ'][j] == 1 or labels['I-OBJ'][j] == 1:
            obj_tokens.append(token)
    
    rows.append({
        'example_id': i,
        'subject': merge_wordpieces(subj_tokens),
        'predicate': merge_wordpieces(pred_tokens),
        'object': merge_wordpieces(obj_tokens),
        'sentence': example['sentence']
    })

# Create DataFrame
df = pd.DataFrame(rows)

# Output to CSV
df.to_csv('output.csv', index=False, encoding='utf-8')

print(f"✓ Exported {len(df)} examples to output.csv")
print(f"\nSummary:")
print(f"  Examples with subject: {sum(1 for _, row in df.iterrows() if row['subject'])}")
print(f"  Examples with predicate: {sum(1 for _, row in df.iterrows() if row['predicate'])}")
print(f"  Examples with object: {sum(1 for _, row in df.iterrows() if row['object'])}")
print(f"  Complete S+P+O: {sum(1 for _, row in df.iterrows() if row['subject'] and row['predicate'] and row['object'])}")
