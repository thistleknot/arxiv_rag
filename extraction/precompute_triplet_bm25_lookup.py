"""
Pre-compute Triplet BM25 Lookup Table (ONE-TIME)

Computes BM25 sparse vectors for ALL triplets once, saves as msgpack lookup.
This avoids O(n²) complexity in the main ingestion pipeline.

Runtime: ~5-10 minutes (one-time cost)
Output: checkpoints/triplet_bm25_lookup.msgpack (~100-200 MB)
"""

import msgpack
import numpy as np
from pathlib import Path
from tqdm import tqdm
from rank_bm25 import BM25Okapi
import json

# Configuration
CHECKPOINT_DIR = Path('checkpoints')
TRIPLET_BM25_INDEX = CHECKPOINT_DIR / 'triplet_bm25_index.msgpack'
OUTPUT_FILE = CHECKPOINT_DIR / 'triplet_bm25_lookup.msgpack'

def load_triplet_index():
    """Load triplet BM25 index."""
    print(f"\nLoading triplet BM25 index from {TRIPLET_BM25_INDEX}...")
    with open(TRIPLET_BM25_INDEX, 'rb') as f:
        data = msgpack.unpackb(f.read(), strict_map_key=False)
    
    triplet_tokens = data['triplet_tokens']
    triplet_texts = data.get('triplet_texts', [])
    
    print(f"  ✓ Loaded {len(triplet_tokens):,} triplets")
    
    return triplet_tokens, triplet_texts


def build_bm25(triplet_tokens):
    """Build BM25 index from triplet tokens."""
    print("\nBuilding BM25 index...")
    bm25 = BM25Okapi(triplet_tokens)
    print(f"  ✓ BM25 index built")
    return bm25


def precompute_all_triplet_scores(bm25, triplet_tokens):
    """
    Pre-compute BM25 sparse vectors for ALL triplets.
    
    Returns: dict mapping triplet_id → sparse_vector_dict
    """
    print(f"\nPre-computing BM25 scores for {len(triplet_tokens):,} triplets...")
    print("  (This is the expensive O(n²) operation - done ONCE)")
    
    triplet_lookup = {}
    
    # Process in batches with progress bar
    batch_size = 5000
    total_batches = (len(triplet_tokens) + batch_size - 1) // batch_size
    
    for i in tqdm(range(0, len(triplet_tokens), batch_size), 
                  desc="  Computing", 
                  total=total_batches):
        batch_tokens = triplet_tokens[i:i+batch_size]
        
        for j, tokens in enumerate(batch_tokens):
            triplet_id = i + j
            
            # Compute BM25 scores for this triplet against corpus
            scores = bm25.get_scores(tokens)
            
            # Convert to sparse format (only non-zero entries)
            non_zero_indices = np.where(scores > 0)[0]
            
            if len(non_zero_indices) > 0:
                # Store as dict: {triplet_idx: score}
                sparse_dict = {
                    int(idx): float(scores[idx])
                    for idx in non_zero_indices
                }
                triplet_lookup[triplet_id] = sparse_dict
            else:
                # Empty sparse vector
                triplet_lookup[triplet_id] = {}
    
    # Statistics
    non_empty = sum(1 for v in triplet_lookup.values() if v)
    total_entries = sum(len(v) for v in triplet_lookup.values())
    avg_sparsity = total_entries / len(triplet_lookup) if triplet_lookup else 0
    
    print(f"\n  ✓ Pre-computed {len(triplet_lookup):,} triplet vectors")
    print(f"  ✓ Non-empty: {non_empty:,} ({100*non_empty/len(triplet_lookup):.1f}%)")
    print(f"  ✓ Average sparsity: {avg_sparsity:.1f} non-zero entries per triplet")
    
    return triplet_lookup


def save_lookup(triplet_lookup, output_file):
    """Save pre-computed lookup to msgpack."""
    print(f"\nSaving lookup to {output_file}...")
    
    with open(output_file, 'wb') as f:
        packed = msgpack.packb(triplet_lookup, use_bin_type=True)
        f.write(packed)
    
    # File size
    size_mb = output_file.stat().st_size / (1024 * 1024)
    print(f"  ✓ Saved: {output_file.name} ({size_mb:.1f} MB)")


def main():
    print("="*60)
    print("PRE-COMPUTE TRIPLET BM25 LOOKUP (ONE-TIME)")
    print("="*60)
    
    # Load
    triplet_tokens, triplet_texts = load_triplet_index()
    
    # Build BM25
    bm25 = build_bm25(triplet_tokens)
    
    # Pre-compute (expensive operation done ONCE)
    triplet_lookup = precompute_all_triplet_scores(bm25, triplet_tokens)
    
    # Save
    save_lookup(triplet_lookup, OUTPUT_FILE)
    
    print("\n" + "="*60)
    print("✓ PRE-COMPUTATION COMPLETE")
    print("="*60)
    print(f"\nNext step: Run ingest_layer2_fast.py to use this lookup")
    print(f"  (Should take ~30 seconds instead of hanging)")


if __name__ == '__main__':
    main()
