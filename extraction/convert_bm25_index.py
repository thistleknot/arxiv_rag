"""
Convert BM25 index from pickle to msgpack format for three-layer retriever.
"""

import pickle
import msgpack
from pathlib import Path

def convert_bm25_index():
    """Convert stage7_bm25_index.pkl to msgpack format."""
    
    pkl_path = Path("triplet_checkpoints_full/stage7_bm25_index.pkl")
    msgpack_path = Path("checkpoints/triplet_bm25_index.msgpack")
    
    print(f"Loading BM25 index from {pkl_path}...")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Data type: {type(data)}")
    if isinstance(data, dict):
        print(f"Keys: {list(data.keys())}")
        for k, v in data.items():
            print(f"  {k}: {type(v)}, len={len(v) if hasattr(v, '__len__') else 'N/A'}")
    
    # Extract only the data needed for BM25 reconstruction
    print(f"\nExtracting BM25 data...")
    
    # Get the corpus (tokenized texts)
    corpus = data['corpus']  # This should be the tokenized triplet texts
    
    # Create simplified format for msgpack
    bm25_data = {
        'triplet_tokens': corpus,  # Tokenized texts
        'triplet_texts': [' '.join(tokens) for tokens in corpus]  # Reconstruct text
    }
    
    print(f"Extracted:")
    print(f"  {len(bm25_data['triplet_tokens'])} tokenized triplets")
    print(f"  {len(bm25_data['triplet_texts'])} text representations")
    
    # Save to msgpack
    print(f"\nSaving to {msgpack_path}...")
    with open(msgpack_path, 'wb') as f:
        msgpack.pack(bm25_data, f)
    
    print(f"✅ Converted BM25 index to msgpack format")
    print(f"   Size: {msgpack_path.stat().st_size / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    convert_bm25_index()
