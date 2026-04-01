"""Convert msgpack triplets to proper GLiNER JSON NER format.

Correct approach:
- Group adjacent atomic tokens of same role (subject/predicate/object)
- Map each group as ONE entity span [start_token_idx, end_token_idx, role]
- Convert to JSON with 'tokenized_text' and 'ner' fields

Example:
  Atomic triplets: (deep, tend, due), (learning, tend, inductive), (models, tend, bias)
  Tokens: ["deep", "learning", "models", "tend", ...]
  
  Correct NER annotations:
    [0, 2, "SUBJ"]    # "deep learning models"
    [3, 3, "PRED"]    # "tend"
    (Objects would be elsewhere in token list)
  
NOT:
    [0, 0, "SUBJ"]    # just "deep"
    [1, 1, "SUBJ"]    # just "learning"
    [2, 2, "SUBJ"]    # just "models"
"""

import json
import msgpack
from transformers import BertTokenizer
from typing import List, Dict, Tuple

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def find_token_indices(tokens: List[str], sentence_lower: str, char_offset: int = 0) -> List[Tuple[int, int]]:
    """Find character positions of tokens in sentence."""
    positions = []
    pos = 0
    
    for token in tokens:
        pos = sentence_lower.find(token, pos)
        if pos == -1:
            continue
        positions.append((pos, pos + len(token)))
        pos += len(token)
    
    return positions

def group_adjacent_tokens(sentence_lower: str, atomic_tokens: List[str], max_gap: int = 1) -> List[Tuple[int, int]]:
    """Group adjacent character positions into spans."""
    positions = find_token_indices(atomic_tokens, sentence_lower)
    
    if not positions:
        return []
    
    # Sort by start position
    positions.sort(key=lambda x: x[0])
    
    # Group adjacent positions
    spans = []
    current_start, current_end = positions[0]
    
    for i in range(1, len(positions)):
        start, end = positions[i]
        gap = start - current_end
        
        if gap <= max_gap:  # Adjacent (allowing whitespace)
            current_end = end  # Extend span
        else:
            # Save current span, start new one
            spans.append((current_start, current_end))
            current_start, current_end = start, end
    
    # Add final span
    spans.append((current_start, current_end))
    return spans

def char_to_token_indices(bert_tokens: List[str], char_spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Convert character-position spans to BERT token indices."""
    # Build character position to token index mapping
    char_to_token = {}
    char_pos = 0
    
    for tok_idx, tok in enumerate(bert_tokens):
        clean_tok = tok.replace('##', '')
        
        if not tok.startswith('##'):
            # Fresh token - estimate start position
            # This is approximate since we're working with lowercased text
            pass
        
        # Map character position to token
        for i in range(len(clean_tok)):
            char_to_token[char_pos + i] = tok_idx
            if not tok.startswith('##') or i == 0:
                char_to_token[char_pos + i] = tok_idx
        
        char_pos += len(clean_tok)
    
    # Convert character spans to token indices
    token_spans = []
    for char_start, char_end in char_spans:
        tok_start = char_to_token.get(char_start, 0)
        tok_end = char_to_token.get(char_end - 1, len(bert_tokens) - 1)
        
        if tok_start <= tok_end:
            token_spans.append((tok_start, tok_end))
    
    return token_spans

def create_gliner_format(sentence: str, triplets: List[Dict]) -> Dict:
    """Convert sentence + triplets to GLiNER JSON format.
    
    Returns:
        {
            "tokenized_text": ["token1", "token2", ...],
            "ner": [[start_idx, end_idx, "ROLE"], ...]
        }
    """
    # Tokenize with BERT
    bert_encoding = tokenizer(sentence, add_special_tokens=False)
    bert_tokens = tokenizer.convert_ids_to_tokens(bert_encoding['input_ids'])
    
    # Manually build offset mapping
    offset_mapping = []
    char_pos = 0
    sentence_lower = sentence.lower()
    
    for tok in bert_tokens:
        clean_tok = tok.replace('##', '')
        
        if not tok.startswith('##'):
            # Find this token in sentence
            next_pos = sentence_lower.find(clean_tok, char_pos)
            if next_pos != -1:
                char_pos = next_pos
        
        tok_start = char_pos
        tok_end = char_pos + len(clean_tok)
        offset_mapping.append((tok_start, tok_end))
        
        if not tok.startswith('##'):
            char_pos = tok_end
    
    sentence_lower = sentence.lower()
    
    # Collect unique atomic tokens by role
    subjects = sorted(set(t['subject'].lower() for t in triplets if t['subject'] != '?'))
    predicates = sorted(set(t['predicate'].lower() for t in triplets if t['predicate'] != '?'))
    objects = sorted(set(t['object'].lower() for t in triplets if t['object'] != '?'))
    
    # Group adjacent tokens
    subject_char_spans = group_adjacent_tokens(sentence_lower, subjects)
    predicate_char_spans = group_adjacent_tokens(sentence_lower, predicates)
    object_char_spans = group_adjacent_tokens(sentence_lower, objects)
    
    # Convert character spans to token indices
    ner_annotations = []
    
    for char_start, char_end in subject_char_spans:
        # Find which tokens overlap with this character span
        tok_start = None
        tok_end = None
        
        for tok_idx, (off_start, off_end) in enumerate(offset_mapping):
            if off_start < char_end and off_end > char_start:  # Overlap
                if tok_start is None:
                    tok_start = tok_idx
                tok_end = tok_idx
        
        if tok_start is not None and tok_end is not None:
            ner_annotations.append([tok_start, tok_end, "SUBJ"])
    
    for char_start, char_end in predicate_char_spans:
        tok_start = None
        tok_end = None
        
        for tok_idx, (off_start, off_end) in enumerate(offset_mapping):
            if off_start < char_end and off_end > char_start:
                if tok_start is None:
                    tok_start = tok_idx
                tok_end = tok_idx
        
        if tok_start is not None and tok_end is not None:
            ner_annotations.append([tok_start, tok_end, "PRED"])
    
    for char_start, char_end in object_char_spans:
        tok_start = None
        tok_end = None
        
        for tok_idx, (off_start, off_end) in enumerate(offset_mapping):
            if off_start < char_end and off_end > char_start:
                if tok_start is None:
                    tok_start = tok_idx
                tok_end = tok_idx
        
        if tok_start is not None and tok_end is not None:
            ner_annotations.append([tok_start, tok_end, "OBJ"])
    
    return {
        "tokenized_text": bert_tokens,
        "ner": ner_annotations
    }

# Test on Example 1
print("="*80)
print("CONVERTING TRAINING DATA TO GLINER JSON FORMAT")
print("="*80)

with open('bio_training_250chunks_complete.msgpack', 'rb') as f:
    data = msgpack.unpackb(f.read(), raw=False)

ex = data['training_data'][0]
sentence = ex['sentence']
triplets = ex['triplets']

print(f"\nEXAMPLE 1:")
print(f"Sentence: {sentence[:100]}...")

gliner_ex = create_gliner_format(sentence, triplets)

print(f"\nTokens ({len(gliner_ex['tokenized_text'])}):")
print(gliner_ex['tokenized_text'])

print(f"\nNER Annotations ({len(gliner_ex['ner'])}):")
for start, end, role in sorted(gliner_ex['ner']):
    tokens_in_span = gliner_ex['tokenized_text'][start:end+1]
    text = ' '.join(tokens_in_span).replace(' ##', '')
    print(f"  [{start:2d}, {end:2d}] = '{text}' → {role}")

# Verify "deep learning models" is properly annotated
print("\n" + "="*80)
print("VERIFICATION:")
print("="*80)

# Find SUBJ annotations
subj_annotations = [ann for ann in gliner_ex['ner'] if ann[2] == 'SUBJ']
print(f"Subject annotations: {len(subj_annotations)}")

for start, end, role in subj_annotations:
    tokens_in_span = gliner_ex['tokenized_text'][start:end+1]
    text = ' '.join(tokens_in_span).replace(' ##', '')
    print(f"  [{start:2d}, {end:2d}] = '{text}'")
    
    if text == "deep learning models":
        print(f"    ✓ SUCCESS! 'deep learning models' correctly annotated as [19, 21]")

# Now convert all 416 training examples to GLiNER JSON
print("\n" + "="*80)
print("CONVERTING ALL 416 TRAINING EXAMPLES...")
print("="*80)

all_examples = []
skipped = 0

for i, ex in enumerate(data['training_data']):
    try:
        gliner_ex = create_gliner_format(ex['sentence'], ex['triplets'])
        all_examples.append(gliner_ex)
    except Exception as e:
        skipped += 1
        if i < 5:  # Show first few errors
            print(f"Example {i}: {str(e)[:100]}")

print(f"\nConverted: {len(all_examples)}/416")
print(f"Skipped: {skipped}")

# Save as JSON
output_file = 'bio_training_250chunks_gliner.json'
with open(output_file, 'w') as f:
    for ex in all_examples:
        f.write(json.dumps(ex) + '\n')

print(f"\n✓ Saved to {output_file}")
print(f"\nFormat verification:")
print(f"  Example 1:")
ex1 = all_examples[0]
print(f"    Tokens: {len(ex1['tokenized_text'])} tokens")
print(f"    NER annotations: {len(ex1['ner'])} spans")
print(f"    Keys: {list(ex1.keys())}")
print(f"\n  Sample annotation: {ex1['ner'][0] if ex1['ner'] else 'None'}")
