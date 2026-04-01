"""
Convert existing GLiNER JSONL (bio_training_250chunks_gliner.json) to NuNER format.

Input:  BERT-subtokened JSONL with span-level NER: [start, end, "SUBJ"]
Output: spaCy-tokenized JSON with token-level NER: [i, i, "subject"]

The GLiNER data already has 416 quality-validated examples.
We just need to re-tokenize with spaCy and remap spans.
"""

import json
import argparse
from typing import List, Dict, Tuple

def load_gliner_jsonl(path: str) -> List[Dict]:
    """Load JSONL format GLiNER data"""
    data = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def bert_tokens_to_text(bert_tokens: List[str]) -> str:
    """Reconstruct raw text from BERT subtokens"""
    text = ""
    for tok in bert_tokens:
        if tok.startswith("##"):
            text += tok[2:]  # strip ## and append without space
        else:
            if text:
                text += " "
            text += tok
    return text

def bert_span_to_text(bert_tokens: List[str], start: int, end: int) -> str:
    """Get the raw text for a BERT subtoken span"""
    span_tokens = bert_tokens[start:end + 1]
    return bert_tokens_to_text(span_tokens)

def map_bert_span_to_spacy_tokens(
    bert_tokens: List[str], 
    bert_start: int, 
    bert_end: int,
    spacy_tokens: List[str],
    spacy_char_spans: List[Tuple[int, int]],
    full_text: str
) -> Tuple[int, int]:
    """
    Map a BERT subtoken span to spaCy token indices.
    
    Strategy: reconstruct the entity text from BERT tokens,
    find it in the full text, then find overlapping spaCy tokens.
    """
    entity_text = bert_span_to_text(bert_tokens, bert_start, bert_end).lower()
    
    # Find entity in full text (case insensitive)
    text_lower = full_text.lower()
    char_start = text_lower.find(entity_text)
    
    if char_start == -1:
        # Try without spaces (handles edge cases)
        entity_nospace = entity_text.replace(" ", "")
        for i in range(len(text_lower)):
            chunk = text_lower[i:i+len(entity_nospace)+10].replace(" ", "")
            if chunk.startswith(entity_nospace):
                char_start = i
                # Find end by counting original chars
                count = 0
                char_end = i
                while count < len(entity_nospace) and char_end < len(text_lower):
                    if text_lower[char_end] != ' ':
                        count += 1
                    char_end += 1
                break
        if char_start == -1:
            return None, None
    else:
        char_end = char_start + len(entity_text)
    
    # Find overlapping spaCy tokens
    spacy_start = None
    spacy_end = None
    for i, (ts, te) in enumerate(spacy_char_spans):
        if ts < char_end and te > char_start:
            if spacy_start is None:
                spacy_start = i
            spacy_end = i
    
    return spacy_start, spacy_end

def convert_to_nuner(gliner_data: List[Dict]) -> List[Dict]:
    """
    Convert GLiNER examples to NuNER format.
    
    GLiNER: BERT subtokens + span-level NER [start, end, LABEL]
    NuNER:  spaCy tokens + token-level NER [i, i, label]
    """
    import spacy
    nlp = spacy.load("en_core_web_sm")
    
    LABEL_MAP = {
        "SUBJ": "subject",
        "PRED": "predicate", 
        "OBJ": "object",
    }
    
    nuner_data = []
    skipped = 0
    
    for ex in gliner_data:
        bert_tokens = ex['tokenized_text']
        ner_spans = ex['ner']
        
        # Reconstruct raw text from BERT subtokens
        raw_text = bert_tokens_to_text(bert_tokens)
        
        # Tokenize with spaCy
        doc = nlp(raw_text)
        spacy_tokens = [t.text for t in doc]
        spacy_char_spans = [(t.idx, t.idx + len(t.text)) for t in doc]
        
        # Convert each NER span
        token_level_ner = []
        valid = True
        
        for span in ner_spans:
            bert_start, bert_end, label = span
            label_lower = LABEL_MAP.get(label, label.lower())
            
            spacy_start, spacy_end = map_bert_span_to_spacy_tokens(
                bert_tokens, bert_start, bert_end,
                spacy_tokens, spacy_char_spans, raw_text
            )
            
            if spacy_start is None:
                # Could not map this span - skip entire example
                valid = False
                break
            
            # NuNER token-level: each token gets its own annotation
            for i in range(spacy_start, spacy_end + 1):
                token_level_ner.append([i, i, label_lower])
        
        if not valid or not token_level_ner:
            skipped += 1
            continue
        
        # Deduplicate and sort
        token_level_ner = sorted(set(tuple(x) for x in token_level_ner))
        token_level_ner = [list(x) for x in token_level_ner]
        
        nuner_data.append({
            "tokenized_text": spacy_tokens,
            "ner": token_level_ner,
        })
    
    print(f"Converted {len(nuner_data)}/{len(gliner_data)} examples ({skipped} skipped)")
    return nuner_data

def main():
    parser = argparse.ArgumentParser(description="Convert GLiNER JSONL to NuNER JSON format")
    parser.add_argument("--input", default="bio_training_250chunks_gliner.json",
                        help="Input GLiNER JSONL file")
    parser.add_argument("--output", default="nuner_training_data.json",
                        help="Output NuNER JSON file")
    args = parser.parse_args()
    
    print(f"Loading {args.input}...")
    gliner_data = load_gliner_jsonl(args.input)
    print(f"Loaded {len(gliner_data)} examples")
    
    print("Converting to NuNER format...")
    nuner_data = convert_to_nuner(gliner_data)
    
    # Save as JSON array (NuNER expects this)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(nuner_data, f, indent=2)
    
    print(f"Saved {len(nuner_data)} examples to {args.output}")
    
    # Show sample
    if nuner_data:
        ex = nuner_data[0]
        tokens = ex['tokenized_text']
        ner = ex['ner']
        roles = {}
        for s, e, l in ner:
            if s < len(tokens):
                roles.setdefault(l, []).append(tokens[s])
        print(f"\nSample: {len(tokens)} tokens, {len(ner)} NER tags")
        for role, words in sorted(roles.items()):
            print(f"  {role}: {' '.join(words)}")

if __name__ == "__main__":
    main()
