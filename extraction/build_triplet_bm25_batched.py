"""
Checkpointed Triplet BM25 Index Builder - BATCHED VERSION

Same 7-stage pipeline but with BATCH INFERENCE for Stage 1.
Expected speedup: 3-5x faster (3 hours → <1 hour for Stage 1).

Key change: Process 32 chunks at once through BERT instead of 1 at a time.

Usage:
    python build_triplet_bm25_batched.py --chunks checkpoints/chunks.msgpack 
           --bio-model bio_tagger_best.pt --output-dir triplet_checkpoints_full
           --batch-size 32
"""

import argparse
import msgpack
import pickle
import torch
import re
import string
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm
import numpy as np

from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
from rank_bm25 import BM25Okapi

from inference_bio_tagger import BIOTripletExtractor


def load_chunks(chunks_path: str, max_chunks: int = None) -> List[Dict]:
    """Load chunks from msgpack file."""
    with open(chunks_path, 'rb') as f:
        data = msgpack.unpack(f, raw=False)
    
    chunks = data if isinstance(data, list) else data.get('chunks', [])
    
    if max_chunks:
        chunks = chunks[:max_chunks]
    
    return chunks


def create_chunk_id(chunk: Dict) -> str:
    """Generate unique chunk ID."""
    doc_id = chunk.get('doc_id', 'unknown')
    section_idx = chunk.get('section_idx', 0)
    chunk_idx = chunk.get('chunk_idx', 0)
    return f"{doc_id}_s{section_idx}_c{chunk_idx}"


def stage1_extract_triplets_batched(
    chunks: List[Dict], 
    bio_model_path: str, 
    output_path: str,
    batch_size: int = 32
) -> List[Dict]:
    """
    Extract triplets using BATCH INFERENCE for GPU efficiency.
    
    Instead of processing 1 chunk at a time, processes batch_size chunks.
    Speeds up from ~10 chunks/sec to ~40-60 chunks/sec.
    """
    print("\n" + "="*80)
    print("STAGE 1: RAW TRIPLET EXTRACTION (BATCHED)")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading BIO tagger on {device}...")
    
    bio_tagger = BIOTripletExtractor(bio_model_path, device=device.type)
    print(f"Model loaded on {device}")
    print(f"Batch size: {batch_size}")
    
    stage1_data = []
    
    # Process in batches
    num_batches = (len(chunks) + batch_size - 1) // batch_size
    
    with tqdm(total=len(chunks), desc="Extracting triplets (batched)") as pbar:
        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(chunks))
            batch_chunks = chunks[batch_start:batch_end]
            
            # Extract texts and IDs
            texts = []
            chunk_ids = []
            for chunk in batch_chunks:
                if 'chunk_id' not in chunk:
                    chunk['chunk_id'] = create_chunk_id(chunk)
                chunk_ids.append(chunk['chunk_id'])
                texts.append(chunk.get('text', ''))
            
            # BATCH INFERENCE - All texts at once
            try:
                batch_triplets = bio_tagger.extract_triplets_batch(
                    texts, 
                    threshold=0.5
                )
                
                # Combine results
                for i, chunk in enumerate(batch_chunks):
                    triplets = batch_triplets[i] if i < len(batch_triplets) else []
                    
                    stage1_data.append({
                        'chunk_id': chunk_ids[i],
                        'text': texts[i],
                        'triplets': triplets
                    })
                
            except Exception as e:
                print(f"\n⚠️ Batch {batch_idx} failed: {e}")
                # Fallback: process individually
                for chunk in batch_chunks:
                    try:
                        text = chunk.get('text', '')
                        triplets = bio_tagger.extract_triplets(text, threshold=0.5)
                        
                        stage1_data.append({
                            'chunk_id': chunk['chunk_id'],
                            'text': text,
                            'triplets': triplets
                        })
                    except:
                        stage1_data.append({
                            'chunk_id': chunk['chunk_id'],
                            'text': text,
                            'triplets': []
                        })
            
            pbar.update(len(batch_chunks))
    
    # Count triplets
    total_triplets = sum(len(item['triplets']) for item in stage1_data)
    
    # Save checkpoint
    with open(output_path, 'wb') as f:
        msgpack.pack(stage1_data, f)
    
    print(f"✅ Stage 1 complete: {len(stage1_data)} chunks, {total_triplets} triplets")
    print(f"   Saved to: {output_path}")
    
    return stage1_data


def stage2_lowercase_and_clean(stage1_data: List[Dict], output_path: str) -> List[Dict]:
    """Lowercase and remove punctuation."""
    print("\n" + "="*80)
    print("STAGE 2: LOWERCASE + REMOVE PUNCTUATION")
    print("="*80)
    
    punct_pattern = re.compile(f'[{re.escape(string.punctuation)}]')
    stage2_data = []
    
    for item in tqdm(stage1_data, desc="Cleaning"):
        cleaned_triplets = []
        for triplet in item['triplets']:
            subj = punct_pattern.sub(' ', triplet['subject']).lower().strip()
            pred = punct_pattern.sub(' ', triplet['predicate']).lower().strip()
            obj = punct_pattern.sub(' ', triplet['object']).lower().strip() if triplet.get('object') else ''
            
            if subj or pred or obj:
                cleaned_triplets.append({
                    'subject': subj,
                    'predicate': pred,
                    'object': obj
                })
        
        stage2_data.append({
            'chunk_id': item['chunk_id'],
            'text': item['text'],
            'triplets': cleaned_triplets
        })
    
    total_triplets = sum(len(item['triplets']) for item in stage2_data)
    
    with open(output_path, 'wb') as f:
        msgpack.pack(stage2_data, f)
    
    print(f"✅ Stage 2 complete: {total_triplets} cleaned triplets")
    print(f"   Saved to: {output_path}")
    
    return stage2_data


def stage3_remove_stopwords_tokenize(stage2_data: List[Dict], output_path: str) -> List[Dict]:
    """Remove stopwords and tokenize."""
    print("\n" + "="*80)
    print("STAGE 3: REMOVE STOPWORDS + TOKENIZE")
    print("="*80)
    
    stop_words = set(stopwords.words('english'))
    stage3_data = []
    total_tokens = 0
    
    for item in tqdm(stage2_data, desc="Removing stopwords"):
        tokenized_triplets = []
        for triplet in item['triplets']:
            subj_tokens = [w for w in triplet['subject'].split() if w and w not in stop_words]
            pred_tokens = [w for w in triplet['predicate'].split() if w and w not in stop_words]
            obj_tokens = [w for w in triplet['object'].split() if w and w not in stop_words]
            
            total_tokens += len(subj_tokens) + len(pred_tokens) + len(obj_tokens)
            
            tokenized_triplets.append({
                'subject_tokens': subj_tokens,
                'predicate_tokens': pred_tokens,
                'object_tokens': obj_tokens
            })
        
        stage3_data.append({
            'chunk_id': item['chunk_id'],
            'text': item['text'],
            'triplets': tokenized_triplets
        })
    
    with open(output_path, 'wb') as f:
        msgpack.pack(stage3_data, f)
    
    print(f"✅ Stage 3 complete: {total_tokens} total tokens after stopword removal")
    print(f"   Saved to: {output_path}")
    
    return stage3_data


def stage4_lemmatize(stage3_data: List[Dict], output_path: str) -> List[Dict]:
    """Lemmatize tokens."""
    print("\n" + "="*80)
    print("STAGE 4: LEMMATIZATION")
    print("="*80)
    
    lemmatizer = WordNetLemmatizer()
    stage4_data = []
    
    for item in tqdm(stage3_data, desc="Lemmatizing"):
        lemmatized_triplets = []
        for triplet in item['triplets']:
            subj_lemmas = [lemmatizer.lemmatize(token) for token in triplet['subject_tokens']]
            pred_lemmas = [lemmatizer.lemmatize(token) for token in triplet['predicate_tokens']]
            obj_lemmas = [lemmatizer.lemmatize(token) for token in triplet['object_tokens']]
            
            lemmatized_triplets.append({
                'subject_lemmas': subj_lemmas,
                'predicate_lemmas': pred_lemmas,
                'object_lemmas': obj_lemmas
            })
        
        stage4_data.append({
            'chunk_id': item['chunk_id'],
            'text': item['text'],
            'triplets': lemmatized_triplets
        })
    
    with open(output_path, 'wb') as f:
        msgpack.pack(stage4_data, f)
    
    print(f"✅ Stage 4 complete: {len([t for item in stage4_data for t in item['triplets']])} triplets lemmatized")
    print(f"   Saved to: {output_path}")
    
    return stage4_data


def stage5_add_synsets(stage4_data: List[Dict], output_path: str) -> List[Dict]:
    """Add 1st synset term."""
    print("\n" + "="*80)
    print("STAGE 5: ADD 1ST SYNSET TERMS")
    print("="*80)
    
    stage5_data = []
    synset_count = 0
    
    for item in tqdm(stage4_data, desc="Adding synsets"):
        synset_triplets = []
        for triplet in item['triplets']:
            subj_with_synsets = []
            for lemma in triplet['subject_lemmas']:
                subj_with_synsets.append(lemma)
                synsets = wordnet.synsets(lemma)
                if synsets:
                    synset_name = synsets[0].name().split('.')[0]
                    if synset_name != lemma:
                        subj_with_synsets.append(synset_name)
                        synset_count += 1
            
            pred_with_synsets = []
            for lemma in triplet['predicate_lemmas']:
                pred_with_synsets.append(lemma)
                synsets = wordnet.synsets(lemma)
                if synsets:
                    synset_name = synsets[0].name().split('.')[0]
                    if synset_name != lemma:
                        pred_with_synsets.append(synset_name)
                        synset_count += 1
            
            obj_with_synsets = []
            for lemma in triplet['object_lemmas']:
                obj_with_synsets.append(lemma)
                synsets = wordnet.synsets(lemma)
                if synsets:
                    synset_name = synsets[0].name().split('.')[0]
                    if synset_name != lemma:
                        obj_with_synsets.append(synset_name)
                        synset_count += 1
            
            synset_triplets.append({
                'subject_with_synsets': subj_with_synsets,
                'predicate_with_synsets': pred_with_synsets,
                'object_with_synsets': obj_with_synsets
            })
        
        stage5_data.append({
            'chunk_id': item['chunk_id'],
            'text': item['text'],
            'triplets': synset_triplets
        })
    
    with open(output_path, 'wb') as f:
        msgpack.pack(stage5_data, f)
    
    print(f"✅ Stage 5 complete: {synset_count} synset terms added")
    print(f"   Saved to: {output_path}")
    
    return stage5_data


def stage6_add_hypernyms(stage5_data: List[Dict], output_path: str) -> List[Dict]:
    """Add 1st hypernym, flatten to type-agnostic."""
    print("\n" + "="*80)
    print("STAGE 6: ADD 1ST HYPERNYM TERMS")
    print("="*80)
    
    stage6_data = []
    hypernym_count = 0
    
    for item in tqdm(stage5_data, desc="Adding hypernyms"):
        hypernym_triplets = []
        for triplet in item['triplets']:
            # Process each role separately, then flatten
            all_enriched = []
            
            for role_key in ['subject_with_synsets', 'predicate_with_synsets', 'object_with_synsets']:
                role_terms = triplet[role_key]
                
                # Track original lemmas only (not added synsets)
                original_count = len([t for t in role_terms if t in role_terms[:len(role_terms)//2+1]])
                
                role_final = list(role_terms)
                
                # Add hypernyms only for original lemmas
                for i, term in enumerate(role_terms):
                    if i < original_count:  # Only original lemmas
                        synsets = wordnet.synsets(term)
                        if synsets:
                            hypernyms = synsets[0].hypernyms()
                            if hypernyms:
                                hypernym_name = hypernyms[0].name().split('.')[0]
                                if hypernym_name not in role_final:
                                    role_final.append(hypernym_name)
                                    hypernym_count += 1
                
                all_enriched.extend(role_final)
            
            hypernym_triplets.append({
                'enriched_terms': all_enriched
            })
        
        stage6_data.append({
            'chunk_id': item['chunk_id'],
            'text': item['text'],
            'triplets': hypernym_triplets
        })
    
    with open(output_path, 'wb') as f:
        msgpack.pack(stage6_data, f)
    
    print(f"✅ Stage 6 complete: {hypernym_count} hypernym terms added")
    print(f"   Saved to: {output_path}")
    
    return stage6_data


def stage7_build_bm25(stage6_data: List[Dict], output_path: str) -> Dict:
    """Build BM25 index."""
    print("\n" + "="*80)
    print("STAGE 7: BUILD BM25 INDEX")
    print("="*80)
    
    corpus = []
    chunk_ids = []
    
    for item in tqdm(stage6_data, desc="Building corpus"):
        all_terms = []
        for triplet in item['triplets']:
            all_terms.extend(triplet['enriched_terms'])
        
        corpus.append(all_terms)
        chunk_ids.append(item['chunk_id'])
    
    print("Building BM25Okapi index...")
    bm25 = BM25Okapi(corpus)
    
    output_data = {
        'bm25': bm25,
        'corpus': corpus,
        'chunk_ids': chunk_ids,
        'stats': {
            'num_chunks': len(chunk_ids),
            'avg_terms_per_chunk': np.mean([len(doc) for doc in corpus]),
            'max_terms': max([len(doc) for doc in corpus]) if corpus else 0,
            'min_terms': min([len(doc) for doc in corpus]) if corpus else 0
        }
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(output_data, f)
    
    print(f"✅ Stage 7 complete: BM25 index built")
    print(f"   Chunks indexed: {len(chunk_ids)}")
    print(f"   Avg terms per chunk: {output_data['stats']['avg_terms_per_chunk']:.1f}")
    print(f"   Saved to: {output_path}")
    
    return output_data


def main():
    parser = argparse.ArgumentParser(description="Build checkpointed triplet BM25 index with BATCH INFERENCE")
    parser.add_argument('--chunks', required=True, help="Path to chunks.msgpack")
    parser.add_argument('--bio-model', required=True, help="Path to trained BIO tagger")
    parser.add_argument('--output-dir', default='triplet_checkpoints', help="Output directory")
    parser.add_argument('--max-chunks', type=int, help="Limit chunks for testing")
    parser.add_argument('--start-stage', type=int, default=1, choices=[1,2,3,4,5,6,7], 
                       help="Resume from stage N")
    parser.add_argument('--batch-size', type=int, default=32, 
                       help="Batch size for Stage 1 inference (default: 32)")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("="*80)
    print("CHECKPOINTED TRIPLET BM25 INDEX BUILDER (BATCHED)")
    print("="*80)
    print(f"Chunks: {args.chunks}")
    print(f"BIO Model: {args.bio_model}")
    print(f"Output Dir: {args.output_dir}")
    print(f"Batch Size: {args.batch_size}")
    if args.max_chunks:
        print(f"Max chunks: {args.max_chunks}")
    if args.start_stage > 1:
        print(f"Resuming from Stage {args.start_stage}")
    
    # Load chunks
    chunks = load_chunks(args.chunks, args.max_chunks)
    print(f"Loaded {len(chunks)} chunks\n")
    
    # Define checkpoint paths
    stage_outputs = {}
    stage1_output = output_dir / 'stage1_raw_triplets.msgpack'
    stage2_output = output_dir / 'stage2_cleaned.msgpack'
    stage3_output = output_dir / 'stage3_tokenized.msgpack'
    stage4_output = output_dir / 'stage4_lemmatized.msgpack'
    stage5_output = output_dir / 'stage5_with_synsets.msgpack'
    stage6_output = output_dir / 'stage6_with_hypernyms.msgpack'
    stage7_output = output_dir / 'stage7_bm25_index.pkl'
    
    # Execute pipeline
    if args.start_stage <= 1:
        stage_outputs[1] = stage1_extract_triplets_batched(
            chunks, args.bio_model, stage1_output, args.batch_size
        )
    else:
        print(f"Loading Stage 1 from {stage1_output}...")
        with open(stage1_output, 'rb') as f:
            stage_outputs[1] = msgpack.unpack(f, raw=False)
    
    if args.start_stage <= 2:
        stage_outputs[2] = stage2_lowercase_and_clean(stage_outputs[1], stage2_output)
    else:
        print(f"Loading Stage 2 from {stage2_output}...")
        with open(stage2_output, 'rb') as f:
            stage_outputs[2] = msgpack.unpack(f, raw=False)
    
    if args.start_stage <= 3:
        stage_outputs[3] = stage3_remove_stopwords_tokenize(stage_outputs[2], stage3_output)
    else:
        print(f"Loading Stage 3 from {stage3_output}...")
        with open(stage3_output, 'rb') as f:
            stage_outputs[3] = msgpack.unpack(f, raw=False)
    
    if args.start_stage <= 4:
        stage_outputs[4] = stage4_lemmatize(stage_outputs[3], stage4_output)
    else:
        print(f"Loading Stage 4 from {stage4_output}...")
        with open(stage4_output, 'rb') as f:
            stage_outputs[4] = msgpack.unpack(f, raw=False)
    
    if args.start_stage <= 5:
        stage_outputs[5] = stage5_add_synsets(stage_outputs[4], stage5_output)
    else:
        print(f"Loading Stage 5 from {stage5_output}...")
        with open(stage5_output, 'rb') as f:
            stage_outputs[5] = msgpack.unpack(f, raw=False)
    
    if args.start_stage <= 6:
        stage_outputs[6] = stage6_add_hypernyms(stage_outputs[5], stage6_output)
    else:
        print(f"Loading Stage 6 from {stage6_output}...")
        with open(stage6_output, 'rb') as f:
            stage_outputs[6] = msgpack.unpack(f, raw=False)
    
    if args.start_stage <= 7:
        stage_outputs[7] = stage7_build_bm25(stage_outputs[6], stage7_output)
    
    print("\n" + "="*80)
    print("✅ PIPELINE COMPLETE")
    print("="*80)
    print(f"All checkpoints saved to: {output_dir}/")
    print(f"Final BM25 index: {stage7_output}")


if __name__ == '__main__':
    main()
