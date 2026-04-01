"""
Append New Documents to Existing BM25 Index

Efficiently adds new chunks to an existing triplet BM25 index by:
1. Loading existing Stage 6 checkpoint (enriched terms)
2. Processing only new chunks through stages 1-6
3. Merging old + new enriched terms
4. Rebuilding BM25 from combined corpus

Note: BM25 IDF scores depend on total corpus, so index must be rebuilt.
However, we avoid re-processing old chunks (saves ~4-5 hours).

Usage:
    python append_to_bm25_index.py \
        --existing-checkpoint triplet_checkpoints_full/stage6_with_hypernyms.msgpack \
        --new-chunks new_chunks.msgpack \
        --bio-model bio_tagger_best.pt \
        --output-dir triplet_checkpoints_updated
"""

import argparse
import msgpack
import pickle
from pathlib import Path
from typing import List, Dict
import numpy as np
from tqdm import tqdm

from rank_bm25 import BM25Okapi

# Import processing stages from main pipeline
from build_triplet_bm25_checkpointed import (
    load_chunks,
    create_chunk_id,
    stage1_extract_triplets,
    stage2_lowercase_and_clean,
    stage3_remove_stopwords_tokenize,
    stage4_lemmatize,
    stage5_add_synsets,
    stage6_add_hypernyms,
)


def merge_stage6_data(existing_data: List[Dict], new_data: List[Dict]) -> List[Dict]:
    """
    Merge existing and new Stage 6 data (enriched terms).
    
    Returns combined list, removing duplicates by chunk_id.
    """
    print("\nMerging existing and new data...")
    
    # Create lookup by chunk_id
    existing_ids = {item['chunk_id'] for item in existing_data}
    
    # Keep all existing
    merged = list(existing_data)
    
    # Add only new chunks (skip duplicates)
    new_count = 0
    duplicate_count = 0
    for item in new_data:
        if item['chunk_id'] not in existing_ids:
            merged.append(item)
            new_count += 1
        else:
            duplicate_count += 1
    
    print(f"  Existing chunks: {len(existing_data)}")
    print(f"  New chunks added: {new_count}")
    print(f"  Duplicates skipped: {duplicate_count}")
    print(f"  Total chunks: {len(merged)}")
    
    return merged


def rebuild_bm25_from_stage6(stage6_data: List[Dict], output_path: str):
    """
    Rebuild BM25 index from Stage 6 enriched data.
    
    Same as Stage 7, but as standalone function.
    """
    print("\n" + "="*80)
    print("REBUILDING BM25 INDEX")
    print("="*80)
    
    # Build corpus
    corpus = []
    chunk_ids = []
    
    for item in tqdm(stage6_data, desc="Building corpus"):
        # Flatten all triplets in chunk
        all_terms = []
        for triplet in item['triplets']:
            all_terms.extend(triplet['enriched_terms'])
        
        corpus.append(all_terms)
        chunk_ids.append(item['chunk_id'])
    
    # Build BM25
    print("Building BM25Okapi index...")
    bm25 = BM25Okapi(corpus)
    
    # Save
    output_data = {
        'bm25': bm25,
        'corpus': corpus,
        'chunk_ids': chunk_ids,
        'stats': {
            'num_chunks': len(chunk_ids),
            'avg_terms_per_chunk': np.mean([len(doc) for doc in corpus]),
            'max_terms': max([len(doc) for doc in corpus]),
            'min_terms': min([len(doc) for doc in corpus])
        }
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(output_data, f)
    
    print(f"✅ BM25 index rebuilt")
    print(f"   Chunks indexed: {len(chunk_ids)}")
    print(f"   Avg terms per chunk: {output_data['stats']['avg_terms_per_chunk']:.1f}")
    print(f"   Saved to: {output_path}")
    
    return output_data


def main():
    parser = argparse.ArgumentParser(description="Append new documents to existing BM25 index")
    parser.add_argument('--existing-checkpoint', required=True, 
                       help="Path to existing stage6_with_hypernyms.msgpack")
    parser.add_argument('--new-chunks', required=True, 
                       help="Path to new chunks.msgpack to add")
    parser.add_argument('--bio-model', required=True, 
                       help="Path to trained BIO tagger")
    parser.add_argument('--output-dir', default='triplet_checkpoints_updated',
                       help="Output directory for updated checkpoints")
    parser.add_argument('--max-chunks', type=int, 
                       help="Limit new chunks for testing")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("="*80)
    print("APPEND TO BM25 INDEX")
    print("="*80)
    print(f"Existing checkpoint: {args.existing_checkpoint}")
    print(f"New chunks: {args.new_chunks}")
    print(f"BIO Model: {args.bio_model}")
    print(f"Output Dir: {args.output_dir}")
    if args.max_chunks:
        print(f"Max new chunks: {args.max_chunks}")
    print()
    
    # Load existing Stage 6 data
    print("Loading existing enriched data...")
    with open(args.existing_checkpoint, 'rb') as f:
        existing_stage6 = msgpack.unpack(f, raw=False)
    print(f"Loaded {len(existing_stage6)} existing chunks\n")
    
    # Load new chunks
    new_chunks = load_chunks(args.new_chunks, args.max_chunks)
    print(f"Loaded {len(new_chunks)} new chunks\n")
    
    # Process new chunks through stages 1-6
    temp_dir = output_dir / 'temp_new_chunks'
    temp_dir.mkdir(exist_ok=True)
    
    print("Processing new chunks through pipeline...")
    
    stage1_output = temp_dir / 'new_stage1.msgpack'
    stage1_data = stage1_extract_triplets(new_chunks, args.bio_model, stage1_output)
    
    stage2_output = temp_dir / 'new_stage2.msgpack'
    stage2_data = stage2_lowercase_and_clean(stage1_data, stage2_output)
    
    stage3_output = temp_dir / 'new_stage3.msgpack'
    stage3_data = stage3_remove_stopwords_tokenize(stage2_data, stage3_output)
    
    stage4_output = temp_dir / 'new_stage4.msgpack'
    stage4_data = stage4_lemmatize(stage3_data, stage4_output)
    
    stage5_output = temp_dir / 'new_stage5.msgpack'
    stage5_data = stage5_add_synsets(stage4_data, stage5_output)
    
    stage6_output = temp_dir / 'new_stage6.msgpack'
    new_stage6_data = stage6_add_hypernyms(stage5_data, stage6_output)
    
    # Merge old + new
    merged_stage6 = merge_stage6_data(existing_stage6, new_stage6_data)
    
    # Save merged Stage 6
    merged_stage6_path = output_dir / 'stage6_with_hypernyms.msgpack'
    with open(merged_stage6_path, 'wb') as f:
        msgpack.pack(merged_stage6, f)
    print(f"\n✅ Saved merged Stage 6: {merged_stage6_path}")
    
    # Rebuild BM25 from merged data
    bm25_output = output_dir / 'stage7_bm25_index.pkl'
    rebuild_bm25_from_stage6(merged_stage6, bm25_output)
    
    print("\n" + "="*80)
    print("✅ UPDATE COMPLETE")
    print("="*80)
    print(f"Updated index: {bm25_output}")
    print(f"Total chunks in index: {len(merged_stage6)}")
    print(f"\nNote: Old checkpoint at {args.existing_checkpoint} unchanged.")


if __name__ == '__main__':
    main()
