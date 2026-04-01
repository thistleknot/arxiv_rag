"""Fix labels in bio_training_250chunks_clean.msgpack using proper character-span mapping."""

import msgpack
import numpy as np
from transformers import BertTokenizer
from tqdm import tqdm

# Load tokenizer
print("Loading BERT tokenizer...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
print("✓ Loaded")

# Load existing data
print("\nLoading bio_training_250chunks_clean.msgpack...")
with open('bio_training_250chunks_clean.msgpack', 'rb') as f:
    data = msgpack.unpackb(f.read(), raw=False)

training_data = data['training_data']
print(f"Loaded {len(training_data)} examples")

# Fix labels for each example
print("\nRegenerating labels with correct character-span mapping...")
for example in tqdm(training_data):
    sentence = example['sentence']
    triplets = example['triplets']
    
    # Tokenize with BERT
    encoding = tokenizer(sentence, add_special_tokens=False)
    bert_tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'])
    
    # Manually build offset mapping
    sentence_lower = sentence.lower()
    offset_mapping = []
    char_pos = 0
    
    for tok in bert_tokens:
        clean_tok = tok.replace('##', '')
        
        if not tok.startswith('##'):
            # Fresh token - find it in sentence
            next_pos = sentence_lower.find(clean_tok, char_pos)
            if next_pos != -1:
                char_pos = next_pos
        
        # Record this token's character span
        tok_start = char_pos
        tok_end = char_pos + len(clean_tok)
        offset_mapping.append((tok_start, tok_end))
        
        if not tok.startswith('##'):
            char_pos = tok_end
    
    # Initialize labels: [n_bert_tokens, 6]
    labels = np.zeros((len(bert_tokens), 6), dtype=np.int32)
    
    # Collect all entity spans from triplets
    entity_spans = []  # List of (char_start, char_end, entity_type_idx)
    
    for triplet in triplets:
        subj = triplet['subject'].lower()
        pred = triplet['predicate'].lower()
        obj = triplet['object'].lower()
        
        # Find subject spans
        if subj != '?':
            pos = 0
            while True:
                pos = sentence_lower.find(subj, pos)
                if pos == -1:
                    break
                entity_spans.append((pos, pos + len(subj), 0))  # 0 = SUBJ
                pos += 1
        
        # Find predicate spans
        if pred != '?':
            pos = 0
            while True:
                pos = sentence_lower.find(pred, pos)
                if pos == -1:
                    break
                entity_spans.append((pos, pos + len(pred), 1))  # 1 = PRED
                pos += 1
        
        # Find object spans
        if obj != '?':
            pos = 0
            while True:
                pos = sentence_lower.find(obj, pos)
                if pos == -1:
                    break
                entity_spans.append((pos, pos + len(obj), 2))  # 2 = OBJ
                pos += 1
    
    # Map entity spans to token indices
    for entity_start, entity_end, entity_type in entity_spans:
        first_token_idx = None
        
        for tok_idx, (tok_start, tok_end) in enumerate(offset_mapping):
            # Check if token overlaps with entity span
            if tok_start < entity_end and tok_end > entity_start:
                if first_token_idx is None:
                    # First token covering this entity → B-tag
                    first_token_idx = tok_idx
                    b_col = entity_type * 2  # 0, 2, 4 for SUBJ, PRED, OBJ
                    labels[tok_idx, b_col] = 1
                else:
                    # Continuation token → I-tag
                    i_col = entity_type * 2 + 1  # 1, 3, 5 for SUBJ, PRED, OBJ
                    labels[tok_idx, i_col] = 1
    
    # Convert to dict format
    label_names = ['B-SUBJ', 'I-SUBJ', 'B-PRED', 'I-PRED', 'B-OBJ', 'I-OBJ']
    labels_dict = {
        label_names[i]: [int(token_labels[i]) for token_labels in labels]
        for i in range(6)
    }
    
    # Update example with new labels
    example['labels'] = labels_dict

# Save fixed data
output_file = 'bio_training_250chunks_complete.msgpack'
print(f"\nSaving to {output_file}...")
with open(output_file, 'wb') as f:
    f.write(msgpack.packb(data, use_bin_type=True))

print(f"✅ Done! Saved {len(training_data)} examples with complete labels")
print(f"\n📊 Verification - checking label density in first 5 examples:")

for i in range(min(5, len(training_data))):
    example = training_data[i]
    label_sum = sum(sum(example['labels'][k]) for k in example['labels'])
    num_tokens = len(example['labels']['B-SUBJ'])
    coverage = label_sum / num_tokens * 100 if num_tokens > 0 else 0
    print(f"  Example {i+1}: {len(example['triplets'])} triplets, {num_tokens} tokens, {label_sum} labels ({coverage:.1f}% coverage)")
