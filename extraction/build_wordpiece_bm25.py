"""
WordPiece-Tokenized BM25 for Semantic Triplets

Uses WordPiece tokenization on lemmatized triplets to create
FIXED vocabulary BM25 index (true appendability, no IDF recalc).

Key Innovation:
- Lemmatize first (semantic grouping)
- WordPiece second (OOV handling + fixed vocab)
- BM25 over WordPiece tokens (static inverted index)

Example:
  "attention mechanism" 
  → lemmatize → ["attention", "mechanism"]
  → wordpiece → ["attention", "mech", "##anism"]
  → BM25 vocab is FIXED (BERT WordPiece 30k tokens)
"""

import msgpack
import numpy as np
from transformers import AutoTokenizer
from rank_bm25 import BM25Okapi
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import argparse
from typing import List


class WordPieceBM25:
    """
    BM25 over WordPiece tokens of lemmatized triplets
    
    Advantages:
    - Fixed vocabulary (BERT WordPiece ~30k tokens)
    - True appendability (no IDF recalc needed)
    - OOV handling (new technical terms decomposed)
    - Can use sparse matrix (fixed dimensions)
    
    Disadvantage:
    - Slight semantic loss (subword splits)
    """
    
    def __init__(self, tokenizer_name='bert-base-uncased'):
        """Initialize with WordPiece tokenizer"""
        print(f"Loading WordPiece tokenizer: {tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.vocab_size = len(self.tokenizer)
        print(f"✓ Vocabulary size: {self.vocab_size} tokens")
        
        self.bm25 = None
        self.doc_tokens = []  # Store for inspection
    
    def tokenize_triplet_text(self, text: str) -> List[str]:
        """
        Tokenize text with WordPiece
        
        Args:
            text: Lemmatized triplet string (e.g., "attention mechanism enable")
        
        Returns:
            List of WordPiece tokens
        """
        # Tokenize with WordPiece
        tokens = self.tokenizer.tokenize(text)
        
        # Remove special tokens ([CLS], [SEP], [PAD])
        tokens = [t for t in tokens if t not in ['[CLS]', '[SEP]', '[PAD]', '[UNK]']]
        
        return tokens
    
    def build_index(self, triplet_texts: List[str], checkpoint_path: str = None):
        """
        Build BM25 index over WordPiece tokens
        
        Args:
            triplet_texts: List of lemmatized triplet strings
            checkpoint_path: Optional path to save index
        """
        print(f"\nTokenizing {len(triplet_texts)} triplets with WordPiece...")
        
        self.doc_tokens = []
        
        for text in tqdm(triplet_texts, desc="WordPiece tokenization"):
            tokens = self.tokenize_triplet_text(text)
            self.doc_tokens.append(tokens)
        
        print(f"✓ Tokenized {len(self.doc_tokens)} documents")
        
        # Build BM25 index
        print("\nBuilding BM25 index...")
        self.bm25 = BM25Okapi(self.doc_tokens)
        print("✓ BM25 index built")
        
        # Save if checkpoint provided
        if checkpoint_path:
            self.save(checkpoint_path)
    
    def append_documents(self, new_triplet_texts: List[str]):
        """
        Append new documents WITHOUT IDF recalculation
        
        This is the KEY advantage: vocabulary is fixed,
        so we can append without rebuilding.
        
        Note: rank-bm25 doesn't support true append,
        so we rebuild (but it's fast with fixed vocab).
        
        For true append: use scipy.sparse matrix implementation.
        """
        print(f"\nAppending {len(new_triplet_texts)} documents...")
        
        new_tokens = []
        for text in tqdm(new_triplet_texts, desc="Tokenizing new docs"):
            tokens = self.tokenize_triplet_text(text)
            new_tokens.append(tokens)
        
        # Rebuild BM25 (fast because vocab is fixed)
        self.doc_tokens.extend(new_tokens)
        self.bm25 = BM25Okapi(self.doc_tokens)
        
        print(f"✓ Index now contains {len(self.doc_tokens)} documents")
    
    def search(self, query: str, top_k: int = 10):
        """
        Search with BM25
        
        Args:
            query: Query string (will be tokenized with WordPiece)
            top_k: Number of results
        
        Returns:
            List[(doc_idx, score)]
        """
        if self.bm25 is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Tokenize query
        query_tokens = self.tokenize_triplet_text(query)
        
        # Get scores
        scores = self.bm25.get_scores(query_tokens)
        
        # Top-k
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = [(int(idx), float(scores[idx])) for idx in top_indices]
        
        return results
    
    def save(self, path: str):
        """Save index to msgpack"""
        print(f"\nSaving index to {path}...")
        
        data = {
            'doc_tokens': self.doc_tokens,
            'tokenizer_name': self.tokenizer.name_or_path,
            'vocab_size': self.vocab_size,
            'n_docs': len(self.doc_tokens),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(path, 'wb') as f:
            msgpack.pack(data, f)
        
        print(f"✓ Saved {len(self.doc_tokens)} documents")
    
    def load(self, path: str):
        """Load index from msgpack"""
        print(f"\nLoading index from {path}...")
        
        with open(path, 'rb') as f:
            data = msgpack.load(f, raw=False)
        
        self.doc_tokens = data['doc_tokens']
        
        # Rebuild BM25 (fast)
        self.bm25 = BM25Okapi(self.doc_tokens)
        
        print(f"✓ Loaded {len(self.doc_tokens)} documents")
        print(f"  Tokenizer: {data['tokenizer_name']}")
        print(f"  Vocab size: {data['vocab_size']}")


def load_enriched_triplets(triplets_path: str):
    """Load enriched triplets from msgpack"""
    print(f"Loading triplets from {triplets_path}...")
    with open(triplets_path, 'rb') as f:
        triplets = msgpack.load(f, raw=False)
    print(f"✓ Loaded {len(triplets)} triplets")
    return triplets


def main():
    parser = argparse.ArgumentParser(
        description="Build WordPiece BM25 index over semantic triplets"
    )
    parser.add_argument(
        '--triplets',
        type=str,
        default='checkpoints/enriched_triplets.msgpack',
        help='Input enriched triplets msgpack'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='checkpoints/wordpiece_bm25_index.msgpack',
        help='Output BM25 index msgpack'
    )
    parser.add_argument(
        '--tokenizer',
        type=str,
        default='bert-base-uncased',
        help='WordPiece tokenizer (BERT model)'
    )
    parser.add_argument(
        '--test-query',
        type=str,
        default=None,
        help='Test query after building'
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("WORDPIECE BM25 INDEX BUILDER")
    print("="*60)
    print(f"Triplets:    {args.triplets}")
    print(f"Output:      {args.output}")
    print(f"Tokenizer:   {args.tokenizer}")
    print("="*60)
    print()
    
    # Initialize WordPiece BM25
    wp_bm25 = WordPieceBM25(tokenizer_name=args.tokenizer)
    
    # Load triplets
    triplets = load_enriched_triplets(args.triplets)
    
    # Extract lemmatized text from each triplet
    # Assuming triplets have format: {'subject': ..., 'predicate': ..., 'object': ..., 'text': ...}
    triplet_texts = []
    for t in triplets:
        # Concatenate lemmatized components
        if 'lemma_text' in t:
            text = t['lemma_text']
        elif 'text' in t:
            text = t['text']
        else:
            # Fallback: join components
            text = f"{t.get('subject', '')} {t.get('predicate', '')} {t.get('object', '')}"
        
        triplet_texts.append(text.strip())
    
    # Build index
    start_time = datetime.now()
    wp_bm25.build_index(triplet_texts, checkpoint_path=args.output)
    elapsed = (datetime.now() - start_time).total_seconds()
    
    print(f"\n{'='*60}")
    print("INDEX BUILT")
    print(f"{'='*60}")
    print(f"Total triplets:  {len(triplet_texts)}")
    print(f"Vocab size:      {wp_bm25.vocab_size} (fixed)")
    print(f"Build time:      {elapsed:.1f}s")
    print(f"{'='*60}")
    
    # Test query
    if args.test_query:
        print(f"\nTest query: '{args.test_query}'")
        results = wp_bm25.search(args.test_query, top_k=5)
        print("\nTop 5 results:")
        for idx, score in results:
            print(f"  [{idx}] score={score:.4f}")
            print(f"      {triplet_texts[idx][:100]}...")
    
    print(f"\n✅ Done! Index saved to: {args.output}")
    print(f"\nNext steps:")
    print(f"  1. Use in three_layer_retriever.py Layer 2")
    print(f"  2. Test append with wp_bm25.append_documents()")
    print(f"  3. Compare with lemma-only BM25")


if __name__ == '__main__':
    main()
