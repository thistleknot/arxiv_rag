import msgpack
import pandas as pd

with open('data/bio_training_test10.msgpack', 'rb') as f:
    data = msgpack.unpackb(f.read(), raw=False)

rows = []
for i, ex in enumerate(data['training_data']):
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
    
    rows.append({
        'subject': ' '.join(subject_tokens) if subject_tokens else '',
        'predicate': ' '.join(predicate_tokens) if predicate_tokens else '',
        'object': ' '.join(object_tokens) if object_tokens else ''
    })

df = pd.DataFrame(rows)

# Set display options for full output
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 120)
pd.set_option('display.max_colwidth', 50)

print(f"Total examples: {len(df)}\n")
print(df)

print(f"\n\nSummary:")
print(f"Examples with subject: {(df['subject'] != '').sum()}")
print(f"Examples with predicate: {(df['predicate'] != '').sum()}")
print(f"Examples with object: {(df['object'] != '').sum()}")
print(f"Complete S+P+O: {((df['subject'] != '') & (df['predicate'] != '') & (df['object'] != '')).sum()}")
