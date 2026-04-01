import msgpack

with open('data/bio_training_test10_v4.msgpack', 'rb') as f:
    data = msgpack.unpackb(f.read(), raw=False)
td = data['training_data']

ex = td[6]
tokens = ex['tokens']
labels = ex['labels']
subj_toks = [tokens[i] for i in range(len(tokens)) if labels['B-SUBJ'][i] or labels['I-SUBJ'][i]]
pred_toks = [tokens[i] for i in range(len(tokens)) if labels['B-PRED'][i] or labels['I-PRED'][i]]
obj_toks  = [tokens[i] for i in range(len(tokens)) if labels['B-OBJ'][i]  or labels['I-OBJ'][i]]
print('Sentence:', ex['sentence'])
print('Raw triplets:', ex['triplets'])
print('Subject tokens:', subj_toks)
print('Predicate tokens:', pred_toks)
print('Object tokens:', obj_toks)
