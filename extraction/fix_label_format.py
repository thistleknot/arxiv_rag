"""Fix label format in existing msgpack file - convert list to dict format"""
import msgpack

print("Loading bio_training_1500chunks_atomic.msgpack...")
with open('bio_training_1500chunks_atomic.msgpack', 'rb') as f:
    data = msgpack.unpackb(f.read(), raw=False)

label_names = ['B-SUBJ', 'I-SUBJ', 'B-PRED', 'I-PRED', 'B-OBJ', 'I-OBJ']

print(f"Converting {len(data['training_data'])} examples...")

for ex in data['training_data']:
    # Convert from [[0,0,1,0,0,0], [0,0,0,1,0,0], ...] 
    # to {'B-SUBJ': [0,0,...], 'I-SUBJ': [0,0,...], ...}
    labels_list = ex['labels']
    labels_dict = {
        label_names[i]: [token_labels[i] for token_labels in labels_list]
        for i in range(6)
    }
    ex['labels'] = labels_dict

print("Saving fixed file...")
with open('bio_training_1500chunks_atomic.msgpack', 'wb') as f:
    f.write(msgpack.packb(data, use_bin_type=True))

print("✅ Fixed label format!")
