"""
Extract NuNER Training Data from ArXiv Chunks

Uses spaCy dependency parsing to extract subject-predicate-object triplets,
then converts to NuNER's token-level format:
{
    "tokenized_text": ["The", "cat", "sat", ...],
    "ner": [(token_idx, token_idx, label), ...]
}

Labels: subject, predicate, object (lowercase as per NuNER convention)

Source data: equidistant_chunking.py output → raw text chunks
"""

import msgpack
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import spacy
from tqdm import tqdm


def load_chunks_from_chunker(max_chunks: int = None) -> List[str]:
    """
    Load raw text chunks by running the equidistant chunking pipeline.
    Falls back to extracting sentences from existing bio_training msgpack.
    
    Returns:
        List of raw text strings
    """
    # Try loading from existing bio training data (has raw sentences)
    candidates = [
        "bio_training_1500chunks_atomic.msgpack",
        "bio_training_250chunks.msgpack", 
        "bio_training_data.msgpack",
    ]
    
    for fname in candidates:
        if Path(fname).exists():
            print(f"Loading sentences from {fname}...")
            with open(fname, 'rb') as f:
                data = msgpack.unpackb(f.read(), raw=False)
            
            if isinstance(data, dict) and 'training_data' in data:
                td = data['training_data']
                # Get unique sentences preserving order
                seen = set()
                sentences = []
                for ex in td:
                    s = ex.get('sentence', '')
                    if s and s not in seen:
                        seen.add(s)
                        sentences.append(s)
                print(f"  Found {len(sentences)} unique sentences from {len(td)} examples")
                if max_chunks:
                    sentences = sentences[:max_chunks]
                return sentences
    
    # Fallback: run equidistant chunker directly
    print("No existing data found. Running equidistant chunker...")
    try:
        from equidistant_chunking import chunk_documents
        chunks = chunk_documents(max_chunks=max_chunks)
        return chunks
    except ImportError:
        raise FileNotFoundError(
            "No chunk source found. Need bio_training_*.msgpack or equidistant_chunking.py"
        )

def tokenize_with_spacy(text: str, nlp) -> Tuple[List[str], List[Tuple[int, int]]]:
    """
    Tokenize text with spaCy and return tokens + character spans
    
    Returns:
        tokens: List of token strings
        char_spans: List of (start, end) character positions for each token
    """
    doc = nlp(text)
    tokens = [token.text for token in doc]
    char_spans = [(token.idx, token.idx + len(token.text)) for token in doc]
    return tokens, char_spans

def extract_triplets_openie(text: str, nlp) -> List[Dict]:
    """
    Extract subject-predicate-object triplets using spaCy dependency parsing.
    Also searches prepositional phrases for additional objects.
    
    Returns:
        List of dicts with keys: subject, predicate, object, 
        subject_span, predicate_span, object_span
        Spans are character positions (start, end)
    """
    doc = nlp(text)
    triplets = []
    
    for sent in doc.sents:
        # Find root verb (predicate)
        roots = [token for token in sent if token.dep_ == "ROOT"]
        if not roots:
            continue
        root = roots[0]
        
        # Find subject (full noun phrase via subtree edges)
        subject = None
        for child in root.children:
            if child.dep_ in ("nsubj", "nsubjpass"):
                subject_span = doc[child.left_edge.i:child.right_edge.i + 1]
                subject = {
                    'text': subject_span.text,
                    'start': subject_span.start_char,
                    'end': subject_span.end_char
                }
                break
        
        if not subject:
            continue
        
        # Find predicate span (root + auxiliaries)
        pred_tokens = [root]
        for child in root.children:
            if child.dep_ in ("aux", "auxpass", "neg"):
                pred_tokens.append(child)
        pred_tokens.sort(key=lambda t: t.i)
        pred_start = pred_tokens[0].idx
        pred_end = pred_tokens[-1].idx + len(pred_tokens[-1].text)
        predicate = {
            'text': ' '.join([t.text for t in pred_tokens]),
            'start': pred_start,
            'end': pred_end
        }
        
        # Find direct objects
        for child in root.children:
            if child.dep_ in ("dobj", "attr"):
                obj_span = doc[child.left_edge.i:child.right_edge.i + 1]
                triplets.append({
                    'subject': subject['text'],
                    'predicate': predicate['text'],
                    'object': obj_span.text,
                    'subject_span': (subject['start'], subject['end']),
                    'predicate_span': (predicate['start'], predicate['end']),
                    'object_span': (obj_span.start_char, obj_span.end_char)
                })
            
            # Also check prepositional phrase objects (e.g. "relies on X")
            if child.dep_ == "prep":
                for grandchild in child.children:
                    if grandchild.dep_ == "pobj":
                        obj_span = doc[grandchild.left_edge.i:grandchild.right_edge.i + 1]
                        # Include preposition in predicate
                        full_pred = predicate['text'] + ' ' + child.text
                        triplets.append({
                            'subject': subject['text'],
                            'predicate': full_pred,
                            'object': obj_span.text,
                            'subject_span': (subject['start'], subject['end']),
                            'predicate_span': (predicate['start'], predicate['end']),
                            'object_span': (obj_span.start_char, obj_span.end_char)
                        })
    
    return triplets

def char_span_to_token_span(char_start: int, char_end: int, 
                            char_spans: List[Tuple[int, int]]) -> Tuple[int, int]:
    """
    Convert character span to token indices
    
    Returns:
        (start_token_idx, end_token_idx) - inclusive range
    """
    start_token = None
    end_token = None
    
    for i, (tok_start, tok_end) in enumerate(char_spans):
        # Token overlaps with character span
        if tok_start < char_end and tok_end > char_start:
            if start_token is None:
                start_token = i
            end_token = i
    
    if start_token is None:
        # No overlap found - return closest token
        for i, (tok_start, tok_end) in enumerate(char_spans):
            if tok_start >= char_start:
                return (i, i)
        return (len(char_spans) - 1, len(char_spans) - 1)
    
    return (start_token, end_token)

def create_nuner_example(text: str, triplets: List[Dict], nlp) -> Dict:
    """
    Create NuNER training example with token-level annotations
    
    NuNER format:
    {
        "tokenized_text": ["The", "cat", "sat", ...],
        "ner": [(token_idx, token_idx, label), ...]
    }
    
    Token-level means each token gets its own annotation (not span-level)
    """
    tokens, char_spans = tokenize_with_spacy(text, nlp)
    
    # Convert triplets to token-level annotations
    ner = []
    for triplet in triplets:
        # Subject
        subj_start, subj_end = char_span_to_token_span(
            triplet['subject_span'][0], 
            triplet['subject_span'][1], 
            char_spans
        )
        for i in range(subj_start, subj_end + 1):
            ner.append((i, i, "subject"))
        
        # Predicate
        pred_start, pred_end = char_span_to_token_span(
            triplet['predicate_span'][0],
            triplet['predicate_span'][1],
            char_spans
        )
        for i in range(pred_start, pred_end + 1):
            ner.append((i, i, "predicate"))
        
        # Object
        obj_start, obj_end = char_span_to_token_span(
            triplet['object_span'][0],
            triplet['object_span'][1],
            char_spans
        )
        for i in range(obj_start, obj_end + 1):
            ner.append((i, i, "object"))
    
    # Remove duplicates (same token annotated multiple times)
    ner = list(set(ner))
    ner.sort(key=lambda x: x[0])
    
    return {
        "tokenized_text": tokens,
        "ner": ner
    }

def extract_nuner_training_data(chunks: List[str], 
                                max_chunks: int = None,
                                min_triplets: int = 1) -> List[Dict]:
    """
    Extract NuNER training examples from chunks
    
    Args:
        chunks: List of text chunks
        max_chunks: Maximum number of chunks to process (None = all)
        min_triplets: Minimum triplets required to include example
    
    Returns:
        List of NuNER training examples
    """
    print("Loading spaCy model...")
    nlp = spacy.load("en_core_web_sm")
    
    if max_chunks:
        chunks = chunks[:max_chunks]
    
    training_data = []
    
    print(f"Processing {len(chunks)} chunks...")
    for chunk in tqdm(chunks, desc="Extracting triplets"):
        # Skip empty or very short chunks
        if not chunk or len(chunk) < 50:
            continue
        
        # Extract triplets
        triplets = extract_triplets_openie(chunk, nlp)
        
        if len(triplets) >= min_triplets:
            example = create_nuner_example(chunk, triplets, nlp)
            training_data.append(example)
    
    print(f"\nExtracted {len(training_data)} training examples")
    return training_data

def main():
    parser = argparse.ArgumentParser(description="Extract NuNER training data from ArXiv chunks")
    parser.add_argument("--output", default="nuner_training_data.json", help="Output JSON file")
    parser.add_argument("--chunks", type=int, default=None, help="Max sentences to process (default: all)")
    parser.add_argument("--min-triplets", type=int, default=1, help="Min triplets per example")
    
    args = parser.parse_args()
    
    # Load raw text sentences
    chunks = load_chunks_from_chunker(max_chunks=args.chunks)
    
    # Extract training data
    training_data = extract_nuner_training_data(
        chunks, 
        max_chunks=None,  # already limited by loader
        min_triplets=args.min_triplets
    )
    
    # Save
    print(f"\nSaving to {args.output}...")
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, indent=2)
    
    print(f"✅ Saved {len(training_data)} examples")
    
    # Print sample
    if training_data:
        print("\n" + "="*70)
        print("SAMPLE EXAMPLE:")
        print("="*70)
        sample = training_data[0]
        print(f"Tokens ({len(sample['tokenized_text'])}): {sample['tokenized_text'][:20]}...")
        print(f"NER annotations ({len(sample['ner'])}): {sample['ner'][:10]}...")

if __name__ == "__main__":
    main()
