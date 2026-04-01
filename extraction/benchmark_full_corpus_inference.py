"""
Benchmark BIO Tagger Inference on Full Chunk Corpus

Tests how long it takes to:
1. Load all chunks from arxiv corpus
2. Split chunks into sentences
3. Run BIO inference on each sentence
4. Extract triplets
5. Build inverted index: chunk_id -> [(sentence_idx, triplet), ...]

Goal: Validate inference speed before building full LCA graph pipeline.
"""

import time
import torch
import msgpack
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import re

# Import BIO tagger components
from inference_bio_tagger import BIOTripletExtractor, Span

def simple_sentence_split(text: str) -> List[str]:
    """
    Simple sentence splitter using regex.
    Splits on . ! ? followed by space and capital letter.
    Good enough for benchmarking.
    """
    # Split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    # Filter empty and very short sentences
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    return sentences

def load_chunks_from_msgpack(chunks_path: str = 'checkpoints/chunks.msgpack', 
                             max_chunks: Optional[int] = None) -> List[Dict]:
    """
    Load chunks from msgpack file.
    
    Returns:
        List of dicts with 'text' key (raw chunk text)
    """
    with open(chunks_path, 'rb') as f:
        data = msgpack.unpackb(f.read(), raw=False)
    
    if not isinstance(data, list):
        raise ValueError(f"Expected list of chunks, got {type(data)}")
    
    chunks = data[:max_chunks] if max_chunks else data
    print(f"Loaded {len(chunks)} chunks from {chunks_path}")
    return chunks

def extract_triplet_from_dict(triplet_dict: Dict) -> Optional[Tuple[str, str, str]]:
    """
    Extract (subj, pred, obj) triplet from BIOTripletExtractor output dict.
    Returns None if incomplete triplet.
    """
    subj = triplet_dict.get('subject')
    pred = triplet_dict.get('predicate')
    obj = triplet_dict.get('object')
    
    # Require subject and predicate, object is optional
    if subj and pred:
        return (subj, pred, obj or '')
    return None

def benchmark_inference(
    model_path: str = 'bio_tagger_best.pt',
    chunks_path: str = 'checkpoints/chunks.msgpack',
    max_chunks: Optional[int] = None,
    batch_size: int = 32
):
    """
    Benchmark BIO inference on full chunk corpus from msgpack.
    
    Args:
        model_path: Path to trained model
        chunks_path: Path to chunks msgpack file
        max_chunks: Optional limit for testing (None = all chunks)
        batch_size: Batch size for inference
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load model
    print(f"\nLoading model from {model_path}...")
    extractor = BIOTripletExtractor(model_path, device=str(device))
    
    # Load chunks from msgpack
    chunks = load_chunks_from_msgpack(chunks_path=chunks_path, max_chunks=max_chunks)
    
    # Build inverted index: chunk_idx -> [(sentence_idx, triplet), ...]
    inverted_index = defaultdict(list)
    
    # Statistics
    total_chunks = len(chunks)
    total_sentences = 0
    total_triplets = 0
    incomplete_triplets = 0
    
    print(f"\n{'='*60}")
    print("STARTING BENCHMARK")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    # Process chunks
    for chunk_idx, chunk in enumerate(chunks):
        # Extract text from chunk (msgpack dict with 'text' key)
        text = chunk.get('text', '') or chunk.get('chunk_text', '')
        
        if not text or not text.strip():
            continue
        
        # Split into sentences
        sentences = simple_sentence_split(text)
        total_sentences += len(sentences)
        
        # Process each sentence
        for sent_idx, sentence in enumerate(sentences):
            # Run BIO inference
            triplet_dicts = extractor.extract_triplets(sentence, threshold=0.5)
            
            # Extract first valid triplet (sentences typically have 1 main triplet)
            for triplet_dict in triplet_dicts:
                triplet = extract_triplet_from_dict(triplet_dict)
                
                if triplet:
                    total_triplets += 1
                    inverted_index[chunk_idx].append((sent_idx, triplet))
                    break  # One triplet per sentence for now
            else:
                incomplete_triplets += 1
        
        # Progress report every 100 chunks
        if (chunk_idx + 1) % 100 == 0:
            elapsed = time.time() - start_time
            rate = (chunk_idx + 1) / elapsed
            sent_rate = total_sentences / elapsed
            print(f"Processed {chunk_idx + 1}/{total_chunks} chunks | "
                  f"{total_sentences} sentences | "
                  f"{total_triplets} triplets | "
                  f"{rate:.1f} chunks/sec | "
                  f"{sent_rate:.1f} sent/sec")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Final statistics
    print(f"\n{'='*60}")
    print("BENCHMARK RESULTS")
    print(f"{'='*60}")
    print(f"Total chunks processed:     {total_chunks:,}")
    print(f"Total sentences:            {total_sentences:,}")
    print(f"Total triplets extracted:   {total_triplets:,}")
    print(f"Incomplete triplets:        {incomplete_triplets:,}")
    print(f"Triplet extraction rate:    {total_triplets / total_sentences * 100:.1f}%")
    print(f"\nTotal time:                 {total_time:.2f} seconds")
    print(f"Chunks per second:          {total_chunks / total_time:.2f}")
    print(f"Sentences per second:       {total_sentences / total_time:.2f}")
    print(f"Triplets per second:        {total_triplets / total_time:.2f}")
    print(f"\nAvg sentences per chunk:    {total_sentences / total_chunks:.1f}")
    print(f"Avg triplets per chunk:     {total_triplets / total_chunks:.1f}")
    print(f"{'='*60}\n")
    
    # Sample inverted index entries
    print("Sample inverted index entries:")
    for chunk_idx in list(inverted_index.keys())[:3]:
        print(f"\nChunk {chunk_idx}: {len(inverted_index[chunk_idx])} triplets")
        for sent_idx, triplet in inverted_index[chunk_idx][:3]:
            s, p, o = triplet
            print(f"  Sent {sent_idx}: ({s}, {p}, {o})")
    
    return inverted_index

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Benchmark BIO inference on full corpus from msgpack')
    parser.add_argument('--model', default='bio_tagger_best.pt', 
                        help='Path to trained model')
    parser.add_argument('--chunks', default='checkpoints/chunks.msgpack',
                        help='Path to chunks msgpack file')
    parser.add_argument('--max-chunks', type=int, default=None,
                        help='Limit number of chunks for testing (None = all)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for inference')
    
    args = parser.parse_args()
    
    benchmark_inference(
        model_path=args.model,
        chunks_path=args.chunks,
        max_chunks=args.max_chunks,
        batch_size=args.batch_size
    )
    
    benchmark_inference(
        model_path=args.model,
        chunks_path=args.chunks,
        max_chunks=args.max_chunks,
        batch_size=args.batch_size
    )
