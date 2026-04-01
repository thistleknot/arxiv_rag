"""
Generate BIO training data from english_quotes dataset for domain-specific NER.

Strategy: WEAK SUPERVISION via transfer learning
  1. Load 2,508 quotes from HuggingFace Abirate/english_quotes
  2. Split into sentences → ~25k examples
  3. Use existing bio_tagger_atomic.pt (arxiv-trained) to generate noisy labels
  4. Save in BIO format compatible with train_bert_bio_optuna.py
  5. Fine-tune new model: quotes_bio_tagger.pt (domain adaptation)

Output: checkpoints/quotes_bio_training.msgpack
  Format: list of dicts:
    {
      'tokens': ['The', 'only', 'way', 'to', 'do', ...],
      'labels': ['O', 'O', 'O', 'O', 'B-predicate', ...],
      'text': 'The only way to do great work is to love what you do.',
      'source': 'quote',
      'author': 'Steve Jobs',
      'tags': ['work', 'love']
    }

Usage:
    # Generate training data (uses existing arxiv tagger for weak labels)
    python extraction/create_quotes_bio_training.py --n 2508 --threshold 0.4585
    
    # Train quotes-specific tagger with Optuna
    python training/train_bert_bio_optuna.py \\
        --train checkpoints/quotes_bio_training.msgpack \\
        --output quotes_bio_tagger.pt \\
        --trials 30 \\
        --epochs 15
    
    # Then rebuild graph with quotes-tuned model
    python apply_bio_corpus.py --model quotes_bio_tagger.pt --msgpack checkpoints/quotes.msgpack
"""

import sys
import re
import argparse
import msgpack
from pathlib import Path
from tqdm import tqdm
import torch

# ── sys.path ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'training'))

from inference_bio_tagger import BIOTripletExtractor

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--n', type=int, default=2508,
                    help='Number of quotes to load (default: all 2508)')
parser.add_argument('--threshold', type=float, default=0.4585,
                    help='BIO threshold for weak label generation')
parser.add_argument('--model', type=str, default='bio_tagger_atomic.pt',
                    help='Existing model to use for weak label generation')
parser.add_argument('--output', type=str, 
                    default='checkpoints/quotes_bio_training.msgpack',
                    help='Output path for BIO training data')
parser.add_argument('--min-length', type=int, default=10,
                    help='Minimum sentence length in characters')
args = parser.parse_args()

# ── Load quotes from HuggingFace ──────────────────────────────────────────────

def load_quotes(n: int):
    """Load english_quotes dataset from HuggingFace."""
    try:
        from datasets import load_dataset
        print(f"Loading english_quotes from HuggingFace (n={n})...")
        ds = load_dataset("Abirate/english_quotes", split="train")
        
        quotes = []
        for row in ds:
            q = (row.get("quote") or "").strip()
            if q and len(q) >= args.min_length:
                quotes.append({
                    "quote": q,
                    "author": row.get("author", "Unknown"),
                    "tags": row.get("tags", []) or [],
                })
        
        print(f"  Loaded {len(quotes)} valid quotes")
        return quotes[:n]
    
    except Exception as e:
        print(f"  ERROR loading from HuggingFace: {e}")
        sys.exit(1)


def split_into_sentences(text: str):
    """
    Split quote into sentences using regex.
    Same logic as apply_bio_corpus.py simple_sentence_split().
    """
    sents = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in sents if len(s.strip()) >= args.min_length]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Load quotes
    quotes = load_quotes(args.n)
    
    # Split into sentences
    sentences = []
    for q in quotes:
        sents = split_into_sentences(q['quote'])
        if not sents:
            sents = [q['quote']]  # Keep as single unit if no split
        
        for sent in sents:
            sentences.append({
                'text': sent,
                'author': q['author'],
                'tags': q['tags'],
            })
    
    print(f"\nSplit into {len(sentences):,} sentences\n")
    
    # Load weak labeler (existing arxiv-trained model)
    model_path = str(ROOT / args.model)
    print(f"Loading weak labeler from {model_path}...")
    extractor = BIOTripletExtractor(model_path, device='cuda')
    print(f"  Active labels: {extractor.active_labels}\n")
    
    # Generate weak BIO labels
    print("Generating weak BIO labels via existing model...")
    training_data = []
    
    for sent_meta in tqdm(sentences, desc='Weak labeling', unit='sent'):
        text = sent_meta['text']
        
        # Get BIO predictions from existing model
        # extractor.extract_triplets_batch returns triplets, but we need token-level labels
        # We'll use the underlying predict_bio method directly
        
        # Tokenize
        encoding = extractor.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding='max_length'
        )
        
        input_ids = encoding['input_ids'].to(extractor.device)
        attention_mask = encoding['attention_mask'].to(extractor.device)
        
        # Get BIO predictions
        with torch.no_grad():
            probs, logits = extractor.model(input_ids, attention_mask)  # Returns (probs, logits) tuple
            # probs shape: [1, seq_len, n_active_labels] already sigmoidized
        
        # Decode to BIO labels (using threshold)
        tokens = extractor.tokenizer.convert_ids_to_tokens(input_ids[0])
        bio_labels_str = []
        
        # Remove batch dimension for easier indexing
        probs = probs[0]  # Now shape: [seq_len, n_labels]
        
        for i, token in enumerate(tokens):
            if token in ['[PAD]', '[CLS]', '[SEP]']:
                bio_labels_str.append('O')
            else:
                # Check which label has highest prob above threshold
                max_prob = 0.0
                best_label = 'O'
                
                for label_idx, label_name in enumerate(extractor.active_labels):
                    prob = probs[i, label_idx].item()
                    if prob > args.threshold and prob > max_prob:
                        max_prob = prob
                        best_label = label_name
                
                bio_labels_str.append(best_label)
        
        # Filter to actual tokens (remove padding, CLS, SEP)
        actual_length = attention_mask[0].sum().item()
        tokens = tokens[1:actual_length-1]  # Skip [CLS] and [SEP]
        bio_labels_str = bio_labels_str[1:actual_length-1]
        
        # Convert to multi-hot format for train_bert_bio_optuna.py
        labels_multihot = {
            'B-SUBJ': [1 if l == 'B-SUBJ' else 0 for l in bio_labels_str],
            'I-SUBJ': [1 if l == 'I-SUBJ' else 0 for l in bio_labels_str],
            'B-PRED': [1 if l == 'B-PRED' else 0 for l in bio_labels_str],
            'I-PRED': [1 if l == 'I-PRED' else 0 for l in bio_labels_str],
            'B-OBJ':  [1 if l == 'B-OBJ'  else 0 for l in bio_labels_str],
            'I-OBJ':  [1 if l == 'I-OBJ'  else 0 for l in bio_labels_str],
        }
        
        # Store training example
        training_data.append({
            'tokens': tokens,
            'labels': labels_multihot,
            'text': text,
            'source': 'quote',
            'author': sent_meta['author'],
            'tags': sent_meta['tags'],
        })
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving {len(training_data):,} training examples to {output_path}...")
    with open(output_path, 'wb') as f:
        msgpack.pack({'training_data': training_data}, f)
    
    # Stats
    label_counts = {'O': 0}
    for label_name in ['B-SUBJ', 'I-SUBJ', 'B-PRED', 'I-PRED', 'B-OBJ', 'I-OBJ']:
        label_counts[label_name] = 0
    
    for example in training_data:
        labels_dict = example['labels']
        n_tokens = len(labels_dict['B-SUBJ'])
        
        for i in range(n_tokens):
            # Check which label is active
            found = False
            for label_name in ['B-SUBJ', 'I-SUBJ', 'B-PRED', 'I-PRED', 'B-OBJ', 'I-OBJ']:
                if labels_dict[label_name][i] == 1:
                    label_counts[label_name] += 1
                    found = True
                    break
            if not found:
                label_counts['O'] += 1
    
    print("\nLabel distribution:")
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / sum(label_counts.values())
        print(f"  {label:15s} {count:8,} ({pct:5.2f}%)")
    
    print("\n✅ Training data ready!")
    print("\nNext steps:")
    print("  1. Train quotes-specific model:")
    print(f"     python training/train_bert_bio_optuna.py \\")
    print(f"       --train {args.output} \\")
    print(f"       --output quotes_bio_tagger.pt \\")
    print(f"       --trials 30 --epochs 15")
    print("\n  2. Extract triplets with new model:")
    print(f"     python demo_quotes_triplets.py --n 2508 \\")
    print(f"       --model quotes_bio_tagger.pt \\")
    print(f"       --output checkpoints/quotes_triplets_clean.msgpack")


if __name__ == '__main__':
    main()
