import msgpack
from collections import Counter

# Load data
with open('bio_training_250chunks_complete_FIXED.msgpack', 'rb') as f:
    data = msgpack.unpackb(f.read(), raw=False)

examples = data['training_data']
print(f'Total examples: {len(examples)}\n')

# Analyze predicate lengths
pred_lengths = []
ipred_total = 0

for ex in examples:
    labels_dict = ex['labels']
    b_pred = labels_dict['B-PRED']
    i_pred = labels_dict['I-PRED']
    
    i = 0
    while i < len(b_pred):
        if b_pred[i] == 1:  # B-PRED starts here
            pred_len = 1
            i += 1
            # Count consecutive I-PRED tokens
            while i < len(i_pred) and i_pred[i] == 1:
                pred_len += 1
                ipred_total += 1
                i += 1
            pred_lengths.append(pred_len)
        else:
            i += 1

print(f'ALL {len(examples)} EXAMPLES:')
print(f'  Total predicates: {len(pred_lengths)}')
print(f'  Total I-PRED tokens: {ipred_total}')
print(f'  Single-token predicates: {sum(1 for l in pred_lengths if l == 1)} ({sum(1 for l in pred_lengths if l == 1)/len(pred_lengths)*100:.1f}%)')
print(f'  Multi-token predicates: {sum(1 for l in pred_lengths if l > 1)} ({sum(1 for l in pred_lengths if l > 1)/len(pred_lengths)*100:.1f}%)')
print(f'  Average tokens per predicate: {sum(pred_lengths)/len(pred_lengths):.2f}\n')

print('Predicate length distribution:')
for length, count in sorted(Counter(pred_lengths).items()):
    pct = count/len(pred_lengths)*100
    print(f'  {length} token(s): {count:4d} predicates ({pct:5.1f}%)')

# Show some examples of multi-token predicates
print('\n\nSample multi-token predicates:')
count = 0
for ex in examples:
    if count >= 10:
        break
    labels_dict = ex['labels']
    b_pred = labels_dict['B-PRED']
    i_pred = labels_dict['I-PRED']
    tokens = ex['tokens']
    
    i = 0
    while i < len(b_pred):
        if b_pred[i] == 1:  # B-PRED
            pred_start = i
            pred_tokens = [tokens[i]]
            i += 1
            while i < len(i_pred) and i_pred[i] == 1:  # I-PRED
                pred_tokens.append(tokens[i])
                i += 1
            if len(pred_tokens) > 1:
                print(f'  {" ".join(pred_tokens)}')
                count += 1
                if count >= 10:
                    break
        else:
            i += 1
