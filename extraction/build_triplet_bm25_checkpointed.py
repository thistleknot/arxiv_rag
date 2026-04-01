"""
Checkpointed Triplet BM25 Index Builder

Processes all chunks through staged pipeline with intermediate checkpoints:

Stage 1: Raw triplet extraction (BERT BIO tagger)
Stage 2: Lowercase + remove punctuation
Stage 3: Remove stopwords + tokenize
Stage 4: Lemmatization
Stage 5: Add 1st synset terms
Stage 6: Add 1st hypernym terms
Stage 7: Build BM25 index

Each stage saves to disk for inspection.

Usage:
    python build_triplet_bm25_checkpointed.py --chunks checkpoints/chunks.msgpack --bio-model bio_tagger_best.pt --output-dir triplet_checkpoints
"""

import argparse
import msgpack
import pickle
import json
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from tqdm import tqdm
import re
import string

from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
from rank_bm25 import BM25Okapi

from inference_bio_tagger import BIOTripletExtractor
import torch


def load_chunks(chunks_path: str, max_chunks: int = None) -> List[Dict]:
    """Load chunks from msgpack."""
    with open(chunks_path, 'rb') as f:
        chunks_data = msgpack.unpack(f, raw=False)
    
    # Handle both list and dict formats
    if isinstance(chunks_data, list):
        chunks = chunks_data
    elif isinstance(chunks_data, dict) and 'chunks' in chunks_data:
        chunks = chunks_data['chunks']
    else:
        raise ValueError(f"Unexpected chunks format: {type(chunks_data)}")
    
    if max_chunks:
        chunks = chunks[:max_chunks]
    
    return chunks


def create_chunk_id(chunk: Dict) -> str:
    """Create consistent chunk ID."""
    if 'chunk_id' in chunk:
        return chunk['chunk_id']
    else:
        return f"{chunk['doc_id']}_s{chunk['section_idx']}_c{chunk['chunk_idx']}"


def stage1_extract_triplets(chunks: List[Dict], bio_model_path: str, output_path: str):
    """
    Stage 1: Extract raw triplets using BIO tagger.
    
    Output format:
    [
        {
            'chunk_id': str,
            'text': str,
            'triplets': [
                {'subject': str, 'predicate': str, 'object': str},
                ...
            ]
        },
        ...
    ]
    """
    print("\n" + "="*80)
    print("STAGE 1: RAW TRIPLET EXTRACTION")
    print("="*80)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading BIO tagger on {device}...")
    bio_tagger = BIOTripletExtractor(bio_model_path, device=device.type)
    
    # Extract triplets
    results = []
    for chunk in tqdm(chunks, desc="Extracting triplets"):
        chunk_id = create_chunk_id(chunk)
        text = chunk['text']
        
        # Extract triplets (returns list of dicts)
        triplets = bio_tagger.extract_triplets(text, threshold=0.5, apply_synsets=False)
        
        results.append({
            'chunk_id': chunk_id,
            'text': text,
            'triplets': triplets
        })
    
    # Save checkpoint
    with open(output_path, 'wb') as f:
        msgpack.pack(results, f)
    
    print(f"✅ Stage 1 complete: {len(results)} chunks, {sum(len(r['triplets']) for r in results)} triplets")
    print(f"   Saved to: {output_path}")
    
    return results


def stage2_lowercase_and_clean(stage1_data: List[Dict], output_path: str):
    """
    Stage 2: Lowercase all terms and remove punctuation.
    
    Note: BERT already handles most punctuation, but this ensures consistency.
    """
    print("\n" + "="*80)
    print("STAGE 2: LOWERCASE + REMOVE PUNCTUATION")
    print("="*80)
    
    # Punctuation removal pattern (keep only alphanumeric and spaces)
    punct_pattern = re.compile(f'[{re.escape(string.punctuation)}]')
    
    results = []
    for item in tqdm(stage1_data, desc="Cleaning"):
        cleaned_triplets = []
        
        for triplet in item['triplets']:
            # Lowercase and remove punctuation
            subj = punct_pattern.sub(' ', triplet.get('subject', '')).lower().strip()
            pred = punct_pattern.sub(' ', triplet.get('predicate', '')).lower().strip()
            obj = punct_pattern.sub(' ', triplet.get('object', '')).lower().strip()
            
            # Keep if not empty
            if subj or pred or obj:
                cleaned_triplets.append({
                    'subject': subj,
                    'predicate': pred,
                    'object': obj
                })
        
        results.append({
            'chunk_id': item['chunk_id'],
            'text': item['text'],
            'triplets': cleaned_triplets
        })
    
    # Save checkpoint
    with open(output_path, 'wb') as f:
        msgpack.pack(results, f)
    
    print(f"✅ Stage 2 complete: {sum(len(r['triplets']) for r in results)} cleaned triplets")
    print(f"   Saved to: {output_path}")
    
    return results


def stage3_remove_stopwords_tokenize(stage2_data: List[Dict], output_path: str):
    """
    Stage 3: Remove stopwords and tokenize into word lists.
    
    Output: triplets with word lists instead of strings.
    """
    print("\n" + "="*80)
    print("STAGE 3: REMOVE STOPWORDS + TOKENIZE")
    print("="*80)
    
    stop_words = set(stopwords.words('english'))
    
    results = []
    for item in tqdm(stage2_data, desc="Removing stopwords"):
        tokenized_triplets = []
        
        for triplet in item['triplets']:
            # Tokenize and filter stopwords
            subj_tokens = [w for w in triplet['subject'].split() if w and w not in stop_words]
            pred_tokens = [w for w in triplet['predicate'].split() if w and w not in stop_words]
            obj_tokens = [w for w in triplet['object'].split() if w and w not in stop_words]
            
            # Keep if has meaningful content
            if subj_tokens or pred_tokens or obj_tokens:
                tokenized_triplets.append({
                    'subject_tokens': subj_tokens,
                    'predicate_tokens': pred_tokens,
                    'object_tokens': obj_tokens
                })
        
        results.append({
            'chunk_id': item['chunk_id'],
            'text': item['text'],
            'triplets': tokenized_triplets
        })
    
    # Save checkpoint
    with open(output_path, 'wb') as f:
        msgpack.pack(results, f)
    
    total_tokens = sum(
        len(t['subject_tokens']) + len(t['predicate_tokens']) + len(t['object_tokens'])
        for r in results for t in r['triplets']
    )
    
    print(f"✅ Stage 3 complete: {total_tokens} total tokens after stopword removal")
    print(f"   Saved to: {output_path}")
    
    return results


def stage4_lemmatize(stage3_data: List[Dict], output_path: str):
    """
    Stage 4: Lemmatize all tokens.
    """
    print("\n" + "="*80)
    print("STAGE 4: LEMMATIZATION")
    print("="*80)
    
    lemmatizer = WordNetLemmatizer()
    
    results = []
    for item in tqdm(stage3_data, desc="Lemmatizing"):
        lemmatized_triplets = []
        
        for triplet in item['triplets']:
            # Lemmatize each token
            subj_lemmas = [lemmatizer.lemmatize(token) for token in triplet['subject_tokens']]
            pred_lemmas = [lemmatizer.lemmatize(token) for token in triplet['predicate_tokens']]
            obj_lemmas = [lemmatizer.lemmatize(token) for token in triplet['object_tokens']]
            
            lemmatized_triplets.append({
                'subject_lemmas': subj_lemmas,
                'predicate_lemmas': pred_lemmas,
                'object_lemmas': obj_lemmas
            })
        
        results.append({
            'chunk_id': item['chunk_id'],
            'text': item['text'],
            'triplets': lemmatized_triplets
        })
    
    # Save checkpoint
    with open(output_path, 'wb') as f:
        msgpack.pack(results, f)
    
    print(f"✅ Stage 4 complete: {sum(len(r['triplets']) for r in results)} triplets lemmatized")
    print(f"   Saved to: {output_path}")
    
    return results


def stage5_add_synsets(stage4_data: List[Dict], output_path: str):
    """
    Stage 5: Add 1st synset term for each lemma.
    """
    print("\n" + "="*80)
    print("STAGE 5: ADD 1ST SYNSET TERMS")
    print("="*80)
    
    results = []
    synset_count = 0
    
    for item in tqdm(stage4_data, desc="Adding synsets"):
        enriched_triplets = []
        
        for triplet in item['triplets']:
            # For each lemma, add first synset
            subj_enriched = []
            for lemma in triplet['subject_lemmas']:
                subj_enriched.append(lemma)
                synsets = wordnet.synsets(lemma)
                if synsets:
                    synset_name = synsets[0].name().split('.')[0]
                    if synset_name != lemma:
                        subj_enriched.append(synset_name)
                        synset_count += 1
            
            pred_enriched = []
            for lemma in triplet['predicate_lemmas']:
                pred_enriched.append(lemma)
                synsets = wordnet.synsets(lemma)
                if synsets:
                    synset_name = synsets[0].name().split('.')[0]
                    if synset_name != lemma:
                        pred_enriched.append(synset_name)
                        synset_count += 1
            
            obj_enriched = []
            for lemma in triplet['object_lemmas']:
                obj_enriched.append(lemma)
                synsets = wordnet.synsets(lemma)
                if synsets:
                    synset_name = synsets[0].name().split('.')[0]
                    if synset_name != lemma:
                        obj_enriched.append(synset_name)
                        synset_count += 1
            
            enriched_triplets.append({
                'subject_with_synsets': subj_enriched,
                'predicate_with_synsets': pred_enriched,
                'object_with_synsets': obj_enriched
            })
        
        results.append({
            'chunk_id': item['chunk_id'],
            'text': item['text'],
            'triplets': enriched_triplets
        })
    
    # Save checkpoint
    with open(output_path, 'wb') as f:
        msgpack.pack(results, f)
    
    print(f"✅ Stage 5 complete: {synset_count} synset terms added")
    print(f"   Saved to: {output_path}")
    
    return results


def stage6_add_hypernyms(stage5_data: List[Dict], output_path: str):
    """
    Stage 6: Add 1st hypernym term for each lemma's 1st synset.
    """
    print("\n" + "="*80)
    print("STAGE 6: ADD 1ST HYPERNYM TERMS")
    print("="*80)
    
    results = []
    hypernym_count = 0
    
    for item in tqdm(stage5_data, desc="Adding hypernyms"):
        fully_enriched_triplets = []
        
        for triplet in item['triplets']:
            # For each lemma (original terms only), add hypernym
            subj_final = list(triplet['subject_with_synsets'])  # Copy existing
            for lemma in triplet['subject_with_synsets'][:len(triplet['subject_with_synsets'])//2 or 1]:  # Only original lemmas
                synsets = wordnet.synsets(lemma)
                if synsets:
                    hypernyms = synsets[0].hypernyms()
                    if hypernyms:
                        hypernym_name = hypernyms[0].name().split('.')[0]
                        if hypernym_name not in subj_final:
                            subj_final.append(hypernym_name)
                            hypernym_count += 1
            
            pred_final = list(triplet['predicate_with_synsets'])
            for lemma in triplet['predicate_with_synsets'][:len(triplet['predicate_with_synsets'])//2 or 1]:
                synsets = wordnet.synsets(lemma)
                if synsets:
                    hypernyms = synsets[0].hypernyms()
                    if hypernyms:
                        hypernym_name = hypernyms[0].name().split('.')[0]
                        if hypernym_name not in pred_final:
                            pred_final.append(hypernym_name)
                            hypernym_count += 1
            
            obj_final = list(triplet['object_with_synsets'])
            for lemma in triplet['object_with_synsets'][:len(triplet['object_with_synsets'])//2 or 1]:
                synsets = wordnet.synsets(lemma)
                if synsets:
                    hypernyms = synsets[0].hypernyms()
                    if hypernyms:
                        hypernym_name = hypernyms[0].name().split('.')[0]
                        if hypernym_name not in obj_final:
                            obj_final.append(hypernym_name)
                            hypernym_count += 1
            
            # Flatten to single list (type-agnostic)
            flat_enriched = subj_final + pred_final + obj_final
            
            fully_enriched_triplets.append({
                'enriched_terms': flat_enriched
            })
        
        results.append({
            'chunk_id': item['chunk_id'],
            'text': item['text'],
            'triplets': fully_enriched_triplets
        })
    
    # Save checkpoint
    with open(output_path, 'wb') as f:
        msgpack.pack(results, f)
    
    print(f"✅ Stage 6 complete: {hypernym_count} hypernym terms added")
    print(f"   Saved to: {output_path}")
    
    return results


def stage7_build_bm25(stage6_data: List[Dict], output_path: str):
    """
    Stage 7: Build BM25 index from flattened enriched terms.
    """
    print("\n" + "="*80)
    print("STAGE 7: BUILD BM25 INDEX")
    print("="*80)
    
    # Build corpus (one document per chunk)
    corpus = []
    chunk_ids = []
    
    for item in tqdm(stage6_data, desc="Building corpus"):
        # Flatten all triplets in chunk to single term list
        all_terms = []
        for triplet in item['triplets']:
            all_terms.extend(triplet['enriched_terms'])
        
        corpus.append(all_terms)
        chunk_ids.append(item['chunk_id'])
    
    # Build BM25 index
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
    
    print(f"✅ Stage 7 complete: BM25 index built")
    print(f"   Chunks indexed: {len(chunk_ids)}")
    print(f"   Avg terms per chunk: {output_data['stats']['avg_terms_per_chunk']:.1f}")
    print(f"   Saved to: {output_path}")
    
    return output_data


def main():
    parser = argparse.ArgumentParser(description="Build checkpointed triplet BM25 index")
    parser.add_argument('--chunks', required=True, help="Path to chunks.msgpack")
    parser.add_argument('--bio-model', required=True, help="Path to trained BIO tagger")
    parser.add_argument('--output-dir', default='triplet_checkpoints', help="Output directory for checkpoints")
    parser.add_argument('--max-chunks', type=int, help="Limit chunks for testing")
    parser.add_argument('--start-stage', type=int, default=1, help="Start from stage N (1-7)")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("="*80)
    print("CHECKPOINTED TRIPLET BM25 INDEX BUILDER")
    print("="*80)
    print(f"Chunks: {args.chunks}")
    print(f"BIO Model: {args.bio_model}")
    print(f"Output Dir: {args.output_dir}")
    if args.max_chunks:
        print(f"Max chunks: {args.max_chunks}")
    print()
    
    # Load chunks
    chunks = load_chunks(args.chunks, args.max_chunks)
    print(f"Loaded {len(chunks)} chunks\n")
    
    # Run pipeline
    stage_outputs = {}
    
    if args.start_stage <= 1:
        stage1_output = output_dir / 'stage1_raw_triplets.msgpack'
        stage_outputs[1] = stage1_extract_triplets(chunks, args.bio_model, stage1_output)
    else:
        stage1_output = output_dir / 'stage1_raw_triplets.msgpack'
        with open(stage1_output, 'rb') as f:
            stage_outputs[1] = msgpack.unpack(f, raw=False)
        print(f"Loaded stage 1 from {stage1_output}")
    
    if args.start_stage <= 2:
        stage2_output = output_dir / 'stage2_cleaned.msgpack'
        stage_outputs[2] = stage2_lowercase_and_clean(stage_outputs[1], stage2_output)
    else:
        stage2_output = output_dir / 'stage2_cleaned.msgpack'
        with open(stage2_output, 'rb') as f:
            stage_outputs[2] = msgpack.unpack(f, raw=False)
        print(f"Loaded stage 2 from {stage2_output}")
    
    if args.start_stage <= 3:
        stage3_output = output_dir / 'stage3_tokenized.msgpack'
        stage_outputs[3] = stage3_remove_stopwords_tokenize(stage_outputs[2], stage3_output)
    else:
        stage3_output = output_dir / 'stage3_tokenized.msgpack'
        with open(stage3_output, 'rb') as f:
            stage_outputs[3] = msgpack.unpack(f, raw=False)
        print(f"Loaded stage 3 from {stage3_output}")
    
    if args.start_stage <= 4:
        stage4_output = output_dir / 'stage4_lemmatized.msgpack'
        stage_outputs[4] = stage4_lemmatize(stage_outputs[3], stage4_output)
    else:
        stage4_output = output_dir / 'stage4_lemmatized.msgpack'
        with open(stage4_output, 'rb') as f:
            stage_outputs[4] = msgpack.unpack(f, raw=False)
        print(f"Loaded stage 4 from {stage4_output}")
    
    if args.start_stage <= 5:
        stage5_output = output_dir / 'stage5_with_synsets.msgpack'
        stage_outputs[5] = stage5_add_synsets(stage_outputs[4], stage5_output)
    else:
        stage5_output = output_dir / 'stage5_with_synsets.msgpack'
        with open(stage5_output, 'rb') as f:
            stage_outputs[5] = msgpack.unpack(f, raw=False)
        print(f"Loaded stage 5 from {stage5_output}")
    
    if args.start_stage <= 6:
        stage6_output = output_dir / 'stage6_with_hypernyms.msgpack'
        stage_outputs[6] = stage6_add_hypernyms(stage_outputs[5], stage6_output)
    else:
        stage6_output = output_dir / 'stage6_with_hypernyms.msgpack'
        with open(stage6_output, 'rb') as f:
            stage_outputs[6] = msgpack.unpack(f, raw=False)
        print(f"Loaded stage 6 from {stage6_output}")
    
    if args.start_stage <= 7:
        stage7_output = output_dir / 'stage7_bm25_index.pkl'
        stage_outputs[7] = stage7_build_bm25(stage_outputs[6], stage7_output)
    
    print("\n" + "="*80)
    print("✅ PIPELINE COMPLETE")
    print("="*80)
    print(f"All checkpoints saved to: {args.output_dir}/")
    print(f"Final BM25 index: {args.output_dir}/stage7_bm25_index.pkl")


if __name__ == '__main__':
    main()
