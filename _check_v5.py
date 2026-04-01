import msgpack

with open('data/bio_training_test10_v5.msgpack', 'rb') as f:
    data = msgpack.unpack(f, raw=False)

td = data['training_data']

for i, ex in enumerate(td):
    sent = ex['sentence']
    if 'controllability' in sent:
        tokens = ex['tokens']
        labels = ex['labels']
        trip   = ex['triplets']
        obj_b  = labels['B-OBJ']
        obj_i  = labels['I-OBJ']
        obj_toks = [tokens[j] for j in range(len(tokens)) if obj_b[j] or obj_i[j]]
        print(f"idx={i}")
        print(f"  sentence: {sent[:90]}")
        print(f"  triplets: {trip}")
        print(f"  Object tokens: {obj_toks}")
        print()

print("Overall stats:")
print(f"  Total: {len(td)}")
complete = sum(1 for ex in td
               if any(ex['labels']['B-SUBJ']) and
                  any(ex['labels']['B-PRED']) and
                  any(ex['labels']['B-OBJ']))
print(f"  Complete S+P+O: {complete} ({100*complete//len(td)}%)")
