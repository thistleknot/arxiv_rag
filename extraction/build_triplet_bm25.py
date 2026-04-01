"""
Build Enriched Triplet BM25 Index

Processes all chunks to create a flat BM25 index where each document is the 
concatenated enriched terms from all SPO triplets in that chunk.

Enrichment pipeline per token:
1. Remove stopwords
2. Lemmatize
3. Add first WordNet synset
4. Add first hypernym of that synset

Graph emerges implicitly: chunks sharing enriched terms = connected nodes.
No explicit graph construction, no Node2Vec, no neural training.

Usage:
    python build_triplet_bm25.py --chunks checkpoints/chunks.msgpack --bio-model bio_tagger_best.pt --output triplet_bm25_index.msgpack
"""

import argparse
import msgpack
import pickle
from pathlib import Path
from typing import List, Tuple
import numpy as np
from tqdm import tqdm
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
from rank_bm25 import BM25Okapi

# Import BIO tagger
from inference_bio_tagger import BIOTripletExtractor
import torch


def enrich_tokens(tokens: List[str], lemmatizer: WordNetLemmatizer, stop_words: set) -> List[str]:
    """
    Enrich tokens with lemmas, synsets, and hypernyms.
    
    Args:
        tokens: Raw tokens from triplet
        lemmatizer: WordNetLemmatizer instance
        stop_words: Set of stopwords to filter
        
    Returns:
        List of enriched terms (original + lemma + synset + hypernym)
    """
    enriched = []
    
    for token in tokens:
        # Skip stopwords
        if token.lower() in stop_words:
            continue
            
        # Keep original (lowercased)
        term = token.lower()
        enriched.append(term)
        
        # Lemmatize
        lemma = lemmatizer.lemmatize(term)
        if lemma != term:
            enriched.append(lemma)
        
        # Add first synset
        synsets = wordnet.synsets(lemma)
        if synsets:
            synset = synsets[0]
            synset_name = synset.name().split('.')[0]  # e.g., "love.n.01" → "love"
            if synset_name not in enriched:
                enriched.append(synset_name)
            
            # Add first hypernym
            hypernyms = synset.hypernyms()
            if hypernyms:
                hypernym = hypernyms[0]
                hypernym_name = hypernym.name().split('.')[0]
                if hypernym_name not in enriched:
                    enriched.append(hypernym_name)
    
    return enriched


def process_chunk(chunk_text: str, bio_tagger: BIOTripletExtractor, lemmatizer: WordNetLemmatizer, 
                  stop_words: set) -> List[str]:
    """
    Extract triplets and enrich to flat term list.
    
    Args:
        chunk_text: Raw text
        bio_tagger: Trained BIO tagger model
        lemmatizer: WordNetLemmatizer instance
        stop_words: Set of stopwords
        
    Returns:
        List of enriched terms from all triplets
    """
    # Extract triplets (returns list of dicts with 'subject', 'predicate', 'object')
    triplets = bio_tagger.extract_triplets(chunk_text, threshold=0.5, apply_synsets=False)
    
    # Flatten and enrich
    all_enriched = []
    for triplet in triplets:
        # Get tokens from each field (split strings into tokens)
        subj_tokens = triplet.get('subject', '').split()
        pred_tokens = triplet.get('predicate', '').split()
        obj_tokens = triplet.get('object', '').split()
        
        # Concatenate S + P + O (type-agnostic)
        all_tokens = subj_tokens + pred_tokens + obj_tokens
        
        # Enrich
        enriched = enrich_tokens(all_tokens, lemmatizer, stop_words)
        all_enriched.extend(enriched)
    
    return all_enriched


def build_index(chunks_path: str, bio_model_path: str, output_path: str, max_chunks: int = None):
    """
    Build BM25 index from enriched triplets.
    
    Args:
        chunks_path: Path to chunks.msgpack
        bio_model_path: Path to trained BIO tagger (bio_tagger_best.pt)
        output_path: Where to save index
        max_chunks: Optional limit for testing
    """
    print("=" * 80)
    print("BUILDING ENRICHED TRIPLET BM25 INDEX")
    print("=" * 80)
    print(f"Chunks: {chunks_path}")
    print(f"BIO Model: {bio_model_path}")
    print(f"Output: {output_path}")
    if max_chunks:
        print(f"Max chunks: {max_chunks}")
    print()
    
    # Load chunks
    print("Loading chunks...")
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
    
    print(f"Loaded {len(chunks)} chunks")
    print()
    
    # Load BIO tagger
    print("Loading BIO tagger...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bio_tagger = BIOTripletExtractor(bio_model_path, device=device.type)
    print(f"Model loaded on {device}")
    print()
    
    # Initialize enrichment tools
    print("Initializing NLTK resources...")
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    print(f"Loaded {len(stop_words)} stopwords")
    print()
    
    # Process all chunks
    print("Processing chunks and enriching triplets...")
    corpus = []
    chunk_ids = []
    
    for chunk in tqdm(chunks, desc="Enriching"):
        # Create chunk ID from available fields
        if 'chunk_id' in chunk:
            chunk_id = chunk['chunk_id']
        else:
            # Build ID from doc_id, section_idx, chunk_idx
            chunk_id = f"{chunk['doc_id']}_s{chunk['section_idx']}_c{chunk['chunk_idx']}"
        
        text = chunk['text']
        
        enriched_terms = process_chunk(text, bio_tagger, lemmatizer, stop_words)
        
        # Store as space-separated string
        corpus.append(enriched_terms)
        chunk_ids.append(chunk_id)
    
    print(f"\nProcessed {len(corpus)} chunks")
    print(f"Example enriched terms (first chunk): {corpus[0][:20]}...")
    print()
    
    # Build BM25 index
    print("Building BM25 index...")
    bm25 = BM25Okapi(corpus)
    print("BM25 index built")
    print()
    
    # Save
    print(f"Saving to {output_path}...")
    output_data = {
        'bm25': bm25,
        'corpus': corpus,
        'chunk_ids': chunk_ids,
        'stats': {
            'num_chunks': len(chunks),
            'bio_model': bio_model_path,
            'chunks_source': chunks_path
        }
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(output_data, f)
    
    print("✅ Index saved successfully")
    print()
    print("Index stats:")
    print(f"  Chunks indexed: {len(chunk_ids)}")
    print(f"  Avg enriched terms per chunk: {np.mean([len(doc) for doc in corpus]):.1f}")
    print(f"  Max enriched terms: {max([len(doc) for doc in corpus])}")
    print(f"  Min enriched terms: {min([len(doc) for doc in corpus])}")


def main():
    parser = argparse.ArgumentParser(description="Build enriched triplet BM25 index")
    parser.add_argument('--chunks', required=True, help="Path to chunks.msgpack")
    parser.add_argument('--bio-model', required=True, help="Path to trained BIO tagger")
    parser.add_argument('--output', default='triplet_bm25_index.msgpack', help="Output path")
    parser.add_argument('--max-chunks', type=int, help="Limit chunks for testing")
    
    args = parser.parse_args()
    
    build_index(args.chunks, args.bio_model, args.output, args.max_chunks)


if __name__ == '__main__':
    main()
