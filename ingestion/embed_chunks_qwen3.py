"""
Embed Chunks with Model2Vec Qwen3 (256d)

Applies Model2Vec distilled Qwen3-Embedding-0.6B to all chunks,
creating fast co-occurrence-aware embeddings for Layer 2 graph expansion.

Output: checkpoints/chunk_embeddings_qwen3.msgpack
"""

import msgpack
import numpy as np
from model2vec import StaticModel
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import argparse


def load_chunks(chunks_path: str):
    """Load chunks from msgpack file"""
    print(f"Loading chunks from {chunks_path}...")
    with open(chunks_path, 'rb') as f:
        chunks = msgpack.load(f, raw=False)
    print(f"✓ Loaded {len(chunks)} chunks")
    return chunks


def embed_chunks_batch(
    chunks: list, 
    model: StaticModel,
    batch_size: int = 128,
    max_chunks: int = None
) -> np.ndarray:
    """
    Embed chunks in batches with Model2Vec Qwen3
    
    Args:
        chunks: List of text chunks
        model: Model2Vec StaticModel
        batch_size: Batch size for encoding
        max_chunks: Optional limit for testing
    
    Returns:
        np.ndarray of shape [n_chunks, 256]
    """
    if max_chunks:
        chunks = chunks[:max_chunks]
    
    print(f"\nEmbedding {len(chunks)} chunks (batch_size={batch_size})...")
    
    embeddings = []
    
    for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding batches"):
        batch = chunks[i:i+batch_size]
        batch_texts = [chunk['text'] for chunk in batch]
        
        # Encode batch
        batch_embs = model.encode(batch_texts)
        embeddings.append(batch_embs)
    
    # Concatenate all batches
    embeddings = np.vstack(embeddings)
    
    print(f"✓ Embeddings shape: {embeddings.shape}")
    return embeddings


def save_embeddings(embeddings: np.ndarray, output_path: str, metadata: dict = None):
    """
    Save embeddings to msgpack
    
    Args:
        embeddings: np.ndarray of shape [n_chunks, 256]
        output_path: Output file path
        metadata: Optional metadata dict
    """
    print(f"\nSaving embeddings to {output_path}...")
    
    # Convert to list for msgpack serialization
    embeddings_list = embeddings.tolist()
    
    data = {
        'embeddings': embeddings_list,
        'shape': embeddings.shape,
        'dtype': str(embeddings.dtype),
        'metadata': metadata or {}
    }
    
    with open(output_path, 'wb') as f:
        msgpack.pack(data, f)
    
    print(f"✓ Saved {embeddings.shape[0]} embeddings")
    
    # Calculate file size
    file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"✓ File size: {file_size_mb:.1f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Embed chunks with Model2Vec Qwen3 (256d)"
    )
    parser.add_argument(
        '--chunks',
        type=str,
        default='checkpoints/chunks.msgpack',
        help='Input chunks msgpack file'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='./qwen3_static_embeddings',
        help='Path to Model2Vec Qwen3 model'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='checkpoints/chunk_embeddings_qwen3.msgpack',
        help='Output embeddings msgpack file'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=128,
        help='Batch size for encoding'
    )
    parser.add_argument(
        '--max-chunks',
        type=int,
        default=None,
        help='Maximum chunks to process (for testing)'
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("EMBED CHUNKS WITH MODEL2VEC QWEN3")
    print("="*60)
    print(f"Chunks:      {args.chunks}")
    print(f"Model:       {args.model}")
    print(f"Output:      {args.output}")
    print(f"Batch size:  {args.batch_size}")
    if args.max_chunks:
        print(f"Max chunks:  {args.max_chunks}")
    print("="*60)
    print()
    
    # Load Model2Vec Qwen3
    print(f"Loading Model2Vec Qwen3 from {args.model}...")
    model = StaticModel.from_pretrained(args.model)
    print(f"✓ Model loaded (256 dimensions)")
    
    # Load chunks
    chunks = load_chunks(args.chunks)
    
    # Embed chunks
    start_time = datetime.now()
    embeddings = embed_chunks_batch(
        chunks, 
        model, 
        batch_size=args.batch_size,
        max_chunks=args.max_chunks
    )
    end_time = datetime.now()
    
    elapsed = (end_time - start_time).total_seconds()
    chunks_per_sec = len(embeddings) / elapsed
    
    print(f"\n{'='*60}")
    print(f"EMBEDDING COMPLETE")
    print(f"{'='*60}")
    print(f"Total chunks:     {len(embeddings)}")
    print(f"Embedding dim:    {embeddings.shape[1]}")
    print(f"Total time:       {elapsed:.1f}s")
    print(f"Throughput:       {chunks_per_sec:.1f} chunks/sec")
    print(f"{'='*60}")
    
    # Save embeddings with metadata
    metadata = {
        'model': 'Model2Vec Qwen3 (256d)',
        'model_path': args.model,
        'source_chunks': args.chunks,
        'n_chunks': len(embeddings),
        'embedding_dim': embeddings.shape[1],
        'batch_size': args.batch_size,
        'timestamp': datetime.now().isoformat(),
        'elapsed_seconds': elapsed,
        'chunks_per_sec': chunks_per_sec
    }
    
    save_embeddings(embeddings, args.output, metadata)
    
    print(f"\n✅ Done! Embeddings saved to: {args.output}")
    print(f"\nNext steps:")
    print(f"  1. Use embeddings in three_layer_retriever.py Layer 2")
    print(f"  2. Test co-occurrence expansion vs semantic similarity")
    print(f"  3. Compare with Jina embeddings (Layer 1)")


if __name__ == '__main__':
    main()
