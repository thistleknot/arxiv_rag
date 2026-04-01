"""
Extract training data for triplet seq2seq model.

Process:
  1. Sample N chunks from database
  2. Extract triplets with OpenIE
  3. Reduce triplets with synset mapper
  4. Linearize to format: [P] pred [S] subj [O] obj [/T] ...
  5. Save to msgpack
"""

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import msgpack
import psycopg2
import numpy as np
from tqdm import tqdm
import argparse
from collections import defaultdict
from synset_reducer import SynsetReducer
from nltk.corpus import wordnet as wn

DB_CONFIG = {
    'dbname': 'postgres',
    'user': 'postgres',
    'password': 'postgres',
    'host': 'localhost',
    'port': 5432
}
TABLE_NAME = 'arxiv_papers_lemma_fullembed'

parser = argparse.ArgumentParser()
parser.add_argument('--chunks', type=int, default=3000, help='Number of chunks to extract')
parser.add_argument('--batch-size', type=int, default=8, help='Parallel extraction workers')
parser.add_argument('--output', type=str, default='triplet_training_data.msgpack', help='Output file')
args = parser.parse_args()


def extract_triplets_openie(chunk_id, content):
    """Extract raw triplets using OpenIE"""
    from extract_graph_to_msgpack import extract_and_process_triplets, get_thread_extractor
    
    try:
        extractor, nlp = get_thread_extractor()
        triplets = extract_and_process_triplets(content, chunk_id)
        return triplets
    except Exception as e:
        print(f"\n  Error on chunk {chunk_id}: {e}")
        return []


def build_synset_vocabulary(reducer, max_vocab_size=10000):
    """
    Build constrained vocabulary of all possible synset-canonical lemmas.
    
    This exploits the deterministic synset selection:
    - For any word, synsets(word)[0] is ALWAYS the same (most common)
    - Extract canonical lemma, tokenize
    - Result: Fixed vocabulary of ~3-5k concept tokens
    
    Returns:
        vocab_tokens (set): Wordpiece token IDs that can appear in synset-canonical forms
        synset_canonical_lemmas (set): Human-readable canonical lemma strings
    """
    print("\n[VOCAB] Building synset-constrained vocabulary...")
    
    synset_vocab = set()
    
    # Get all lemma names from WordNet (potential input space)
    all_lemmas = set()
    for synset in wn.all_synsets():
        for lemma in synset.lemmas():
            lemma_name = lemma.name().replace('_', ' ')
            all_lemmas.add(lemma_name)
    
    print(f"  [VOCAB] WordNet contains {len(all_lemmas):,} unique lemmas")
    
    # For each lemma, get its deterministic synset-canonical form
    for lemma_name in tqdm(all_lemmas, desc="  [VOCAB] Mapping lemmas to synsets"):
        # This mimics what synset_reducer does (ALWAYS picks synsets[0])
        synsets = wn.synsets(lemma_name.replace(' ', '_'))
        if synsets:
            # ALWAYS pick first (most common) - deterministic
            canonical_lemma = synsets[0].lemmas()[0].name().replace('_', ' ')
            synset_vocab.add(canonical_lemma)
    
    print(f"  [VOCAB] Reduced to {len(synset_vocab):,} synset-canonical lemmas")
    
    # Tokenize all canonical lemmas
    vocab_tokens = set()
    for canonical_lemma in tqdm(synset_vocab, desc="  [VOCAB] Tokenizing canonical lemmas"):
        tokens = reducer.tokenizer.encode(canonical_lemma, add_special_tokens=False)
        vocab_tokens.update(tokens)
    
    print(f"  [VOCAB] Final vocabulary size: {len(vocab_tokens):,} wordpiece tokens")
    print(f"  [VOCAB] Reduction: {30522} BERT tokens → {len(vocab_tokens)} synset tokens ({100*len(vocab_tokens)/30522:.1f}%)")
    
    return vocab_tokens, synset_vocab


def linearize_triplet(subject_toks, predicate_toks, object_toks, special_tokens):
    """
    Convert reduced triplet to linearized token sequence.
    
    Format: [P] pred_toks [S] subj_toks [O] obj_toks [/T]
    (Object optional for partial triplets)
    """
    sequence = []
    
    # Predicate first (most important)
    if predicate_toks:
        sequence.append(special_tokens['[P]'])
        sequence.extend(predicate_toks)
    
    # Subject
    if subject_toks:
        sequence.append(special_tokens['[S]'])
        sequence.extend(subject_toks)
    
    # Object (optional)
    if object_toks:
        sequence.append(special_tokens['[O]'])
        sequence.extend(object_toks)
    
    # Triplet delimiter
    sequence.append(special_tokens['[/T]'])
    
    return sequence


def main():
    print("="*70)
    print("EXTRACT TRAINING DATA FOR TRIPLET SEQ2SEQ")
    print("="*70)
    print(f"Chunks: {args.chunks}")
    print(f"Output: {args.output}")
    
    # Initialize synset reducer
    print("\n[1/6] Initializing synset reducer...")
    reducer = SynsetReducer()
    
    # Build synset-constrained vocabulary FIRST
    synset_vocab_tokens, synset_canonical_lemmas = build_synset_vocabulary(reducer)
    
    # Special tokens
    special_tokens = {
        '[P]': 999990,   # Predicate marker
        '[S]': 999991,   # Subject marker
        '[O]': 999992,   # Object marker
        '[/T]': 999993,  # Triplet delimiter
        '[PAD]': 0,      # Padding
        '[EOS]': 999994  # End of sequence
    }
    
    print("  Special tokens:")
    for name, idx in special_tokens.items():
        print(f"    {name}: {idx}")
    
    # Get random chunk IDs
    print("\n[2/6] Fetching chunk IDs...")
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute(f"""
        SELECT id 
        FROM {TABLE_NAME} 
        ORDER BY RANDOM() 
        LIMIT {args.chunks}
    """)
    chunk_ids = [row[0] for row in cur.fetchall()]
    cur.close()
    conn.close()
    
    print(f"  ✓ Fetched {len(chunk_ids)} chunk IDs")
    
    # Extract triplets with OpenIE + reduce
    print("\n[3/6] Extracting triplets with OpenIE...")
    
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    training_data = []
    vocab_counts = defaultdict(int)  # Track token frequency
    
    for chunk_id in tqdm(chunk_ids, desc="  Processing"):
        try:
            # Get content
            cur.execute(f"SELECT content FROM {TABLE_NAME} WHERE id = %s", (chunk_id,))
            row = cur.fetchone()
            if not row:
                continue
            
            content = row[0]
            
            # Extract raw triplets with OpenIE
            raw_triplets = extract_triplets_openie(chunk_id, content)
            
            if not raw_triplets:
                continue
            
            # Reduce each triplet
            linearized_sequence = []
            
            for triplet in raw_triplets:
                subj = triplet.get('subject', '')
                pred = triplet.get('predicate', '')
                obj = triplet.get('object', '')
                
                # Reduce via synset
                reduced = reducer.reduce_triplet(subj, pred, obj)
                
                # Skip if no predicate (predicate is required)
                if not reduced['predicate']:
                    continue
                
                # Linearize
                triplet_seq = linearize_triplet(
                    reduced['subject'],
                    reduced['predicate'],
                    reduced['object'],
                    special_tokens
                )
                
                linearized_sequence.extend(triplet_seq)
                
                # Track vocab
                for tok in triplet_seq:
                    vocab_counts[tok] += 1
            
            # Add EOS
            linearized_sequence.append(special_tokens['[EOS]'])
            
            if len(linearized_sequence) > 1:  # More than just EOS
                training_data.append({
                    'chunk_id': chunk_id,
                    'text': content,
                    'target_sequence': linearized_sequence
                })
        
        except Exception as e:
            print(f"\n  Error on chunk {chunk_id}: {e}")
            continue
    
    cur.close()
    conn.close()
    
    print(f"\n  ✓ Extracted {len(training_data)} training samples")
    
    # Build vocabulary from extraction (validate against synset vocab)
    print("\n[4/6] Building vocabulary from extracted tokens...")
    
    # Keep tokens appearing >= 2 times AND in synset vocabulary
    valid_tokens = set()
    invalid_tokens = set()
    
    for tok, count in vocab_counts.items():
        if tok in special_tokens.values():
            valid_tokens.add(tok)  # Always keep special tokens
        elif tok in synset_vocab_tokens and count >= 2:
            valid_tokens.add(tok)  # Keep if in synset vocab AND frequent
        else:
            invalid_tokens.add(tok)
    
    vocab = {tok: idx for idx, tok in enumerate(sorted(valid_tokens))}
    
    # Add special tokens if not present
    for name, tok_id in special_tokens.items():
        if tok_id not in vocab:
            vocab[tok_id] = len(vocab)
    
    print(f"  Total unique tokens extracted: {len(vocab_counts):,}")
    print(f"  Valid synset tokens (freq>=2): {len(valid_tokens):,}")
    print(f"  Invalid/rare tokens filtered: {len(invalid_tokens):,}")
    print(f"  Special tokens: {len(special_tokens)}")
    print(f"  Final vocab size: {len(vocab):,}")
    
    # Statistics
    avg_triplets = np.mean([
        sample['target_sequence'].count(special_tokens['[/T]'])
        for sample in training_data
    ])
    avg_seq_len = np.mean([len(s['target_sequence']) for s in training_data])
    
    print(f"\n  Avg triplets/chunk: {avg_triplets:.1f}")
    print(f"  Avg sequence length: {avg_seq_len:.1f}")
    print(f"  Max sequence length: {max(len(s['target_sequence']) for s in training_data)}")
    
    # Save
    print(f"\n[5/6] Saving to {args.output}...")
    
    save_data = {
        'training_data': training_data,
        'vocab': vocab,
        'synset_vocab_tokens': list(synset_vocab_tokens),  # All valid synset tokens
        'synset_canonical_lemmas': list(synset_canonical_lemmas),  # Human-readable
        'special_tokens': special_tokens,
        'stats': {
            'num_samples': len(training_data),
            'vocab_size': len(vocab),
            'synset_vocab_size': len(synset_vocab_tokens),
            'synset_canonical_lemmas_count': len(synset_canonical_lemmas),
            'avg_triplets': float(avg_triplets),
            'avg_seq_len': float(avg_seq_len),
            'reduction_ratio': len(synset_vocab_tokens) / 30522
        }
    }
    
    with open(args.output, 'wb') as f:
        msgpack.pack(save_data, f)
    
    print("  ✓ Saved")
    
    print("\n" + "="*70)
    print("COMPLETE")
    print("="*70)
    print(f"Training samples: {len(training_data)}")
    print(f"Final vocab size: {len(vocab):,}")
    print(f"Synset vocab size: {len(synset_vocab_tokens):,} tokens")
    print(f"Synset canonical lemmas: {len(synset_canonical_lemmas):,}")
    print(f"BERT vocab reduction: {30522} → {len(synset_vocab_tokens)} ({100*len(synset_vocab_tokens)/30522:.1f}%)")
    print("\n✅ Vocabulary is now constrained to synset-canonical tokens only!")
    print("\nNext steps:")
    print("  1. Train model: python train_triplet_model.py")
    print("  2. Run inference: python predict_triplets.py")


if __name__ == "__main__":
    main()
