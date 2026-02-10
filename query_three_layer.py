"""
Query the Three-Layer GIST Retriever Pipeline.

Pipeline:
  Layer 1: BM25 + Qwen3 128d embeddings (PCA) → GIST selection → RRF fusion → top_k² seeds
  Layer 2: Graph BM25 + Qwen3 256d embeddings (full) → GIST selection → RRF fusion → expand + rerank
  Layer 3: ColBERT + Cross-Encoder → GIST selection → RRF fusion → walk-down papers

Prerequisites:
  - Checkpoints in checkpoints/ folder:
    * chunks.msgpack
    * chunk_embeddings_qwen3.msgpack (Qwen3 256d, used for both Layer 1 128d and Layer 2 256d)
    * triplet_bm25_index.msgpack
    * chunk_to_triplets.msgpack (auto-generated if missing)
    * triplet_to_chunks.msgpack (auto-generated if missing)
  - triplet_checkpoints_full/stage4_lemmatized.msgpack

Usage:
  python query_three_layer.py "agentic memory methods"
  python query_three_layer.py "transformer attention mechanisms" --top-k 20
  python query_three_layer.py "retrieval augmented generation" --top-k 10 --verbose
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import msgpack
import numpy as np
from rank_bm25 import BM25Okapi

sys.path.insert(0, str(Path(__file__).parent))

from three_layer_gist_retriever import ThreeLayerGISTRetriever, GISTLayerConfig


# =============================================================================
# DATA LOADING
# =============================================================================

def load_msgpack(path: str, key: str = None):
    """Load data from msgpack file, optionally extracting a key."""
    with open(path, 'rb') as f:
        data = msgpack.unpackb(f.read(), raw=False)
    if key and isinstance(data, dict) and key in data:
        return data[key]
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and 'chunks' in data:
        return data['chunks']
    return data


def load_qwen3_embeddings(path: str) -> np.ndarray:
    """Load Qwen3 embeddings as float32 numpy array."""
    with open(path, 'rb') as f:
        data = msgpack.unpackb(f.read(), raw=False)
    return np.array(data['embeddings'], dtype=np.float32)


# =============================================================================
# MAPPING BUILDER
# =============================================================================

def ensure_mappings(chunks, triplets, output_dir: str = "checkpoints"):
    """Build chunk<->triplet mappings if they don't exist on disk."""
    c2t_path = Path(output_dir) / "chunk_to_triplets.msgpack"
    t2c_path = Path(output_dir) / "triplet_to_chunks.msgpack"

    if c2t_path.exists() and t2c_path.exists():
        return str(c2t_path), str(t2c_path)

    print("Building chunk<->triplet mappings...")
    chunk_to_triplets = {}
    triplet_to_chunks = {}

    for i, chunk in enumerate(chunks):
        chunk_id = chunk.get('id', str(i))
        triplet_ids = []
        if i < len(triplets):
            t = triplets[i]
            if isinstance(t, list):
                for j, _ in enumerate(t):
                    tid = f"{chunk_id}_t{j}"
                    triplet_ids.append(tid)
                    triplet_to_chunks[tid] = [chunk_id]
            elif isinstance(t, dict):
                tid = f"{chunk_id}_t0"
                triplet_ids.append(tid)
                triplet_to_chunks[tid] = [chunk_id]
        chunk_to_triplets[chunk_id] = triplet_ids

    Path(output_dir).mkdir(exist_ok=True)
    for p, d in [(c2t_path, chunk_to_triplets), (t2c_path, triplet_to_chunks)]:
        with open(p, 'wb') as f:
            f.write(msgpack.packb(d, use_bin_type=True))

    print(f"  Saved {len(chunk_to_triplets):,} chunk mappings, {len(triplet_to_chunks):,} triplet mappings")
    return str(c2t_path), str(t2c_path)


# =============================================================================
# DISPLAY
# =============================================================================

def display_results(results, chunks, query: str, verbose: bool = False):
    """Pretty-print retrieval results grouped by paper and section."""
    print(f"\n{'='*70}")
    print(f"RESULTS FOR: \"{query}\"")
    print(f"{'='*70}")
    print(f"Returned {len(results)} chunks\n")

    # Build chunk lookup
    chunk_lookup = {}
    for i, c in enumerate(chunks):
        cid = c.get('id', str(i))
        chunk_lookup[cid] = c

    # Group by paper_id and section_idx
    from collections import defaultdict
    papers = defaultdict(lambda: defaultdict(list))
    
    for chunk_id, score in results:
        chunk = chunk_lookup.get(chunk_id, chunk_lookup.get(str(chunk_id), {}))
        paper_id = chunk.get('paper_id', 'unknown')
        section_idx = chunk.get('section_idx', 0)
        
        papers[paper_id][section_idx].append({
            'chunk_id': chunk_id,
            'score': score,
            'text': chunk.get('text', chunk.get('content', ''))
        })
    
    # Display grouped by paper and section
    for paper_id in sorted(papers.keys()):
        print(f"\n## Paper: {paper_id}\n")
        
        sections = sorted(papers[paper_id].keys())
        prev_section = None
        
        for section_idx in sections:
            # Insert ellipses if section jumps by more than 1
            if prev_section is not None and section_idx > prev_section + 1:
                print("...\n")
            
            print(f"### Section {section_idx}\n")
            
            # Print all chunks in this section (full text, no truncation)
            for chunk_data in papers[paper_id][section_idx]:
                print(chunk_data['text'])
                print()  # Blank line between chunks
            
            prev_section = section_idx


def save_results_to_markdown(results, chunks, query: str, output_path: str, runtime: float):
    """Save retrieval results to markdown file grouped by paper and section."""
    from collections import defaultdict
    
    # Build chunk lookup
    chunk_lookup = {}
    for i, c in enumerate(chunks):
        cid = c.get('id', str(i))
        chunk_lookup[cid] = c
    
    # Group by paper_id and section_idx
    papers = defaultdict(lambda: defaultdict(list))
    
    for chunk_id, score in results:
        chunk = chunk_lookup.get(chunk_id, chunk_lookup.get(str(chunk_id), {}))
        paper_id = chunk.get('paper_id', 'unknown')
        section_idx = chunk.get('section_idx', 0)
        
        papers[paper_id][section_idx].append({
            'chunk_id': chunk_id,
            'score': score,
            'text': chunk.get('text', chunk.get('content', ''))
        })
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"# Three-Layer GIST Retrieval Results\n\n")
        f.write(f"**Query:** \"{query}\"\\\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d')}\\\n")
        f.write(f"**Runtime:** {runtime:.2f}s\\\n")
        f.write(f"**Results:** {len(results)} chunks from {len(papers)} papers\n\n")
        f.write("---\n\n")
        
        # Write grouped results
        for paper_id in sorted(papers.keys()):
            f.write(f"## Paper: {paper_id}\n\n")
            
            sections = sorted(papers[paper_id].keys())
            prev_section = None
            
            for section_idx in sections:
                # Insert ellipses if section jumps by more than 1
                if prev_section is not None and section_idx > prev_section + 1:
                    f.write("...\n\n")
                
                f.write(f"### Section {section_idx}\n\n")
                
                # Write all chunks in this section (full text)
                for chunk_data in papers[paper_id][section_idx]:
                    f.write(chunk_data['text'])
                    f.write("\n\n")
                
                prev_section = section_idx
    
    print(f"\n✅ Results saved to: {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def run_query(query: str, top_k: int = 13, verbose: bool = False, output: str = None):
    """Execute a three-layer retrieval query."""
    t0 = time.time()

    # --- Paths ---
    chunks_path = "checkpoints/chunks.msgpack"
    qwen3_path = "checkpoints/chunk_embeddings_qwen3.msgpack"
    triplets_path = "triplet_checkpoints_full/stage4_lemmatized.msgpack"
    triplet_bm25_path = "checkpoints/triplet_bm25_index.msgpack"

    for p in [chunks_path, qwen3_path, triplets_path]:
        if not Path(p).exists():
            print(f"❌ Missing checkpoint: {p}")
            print("   Run build_arxiv_graph_batched.py first.")
            return 1

    # --- Load data ---
    if verbose:
        print("Loading data...")
    chunks = load_msgpack(chunks_path)
    triplets = load_msgpack(triplets_path)

    if verbose:
        print(f"  {len(chunks):,} chunks, {len(triplets):,} triplets")

    # --- Build BM25 index from lemmatized chunks ---
    if verbose:
        print("Building BM25 index...")
    
    # Simple tokenization (lowercase alphanumeric, min 2 chars)
    import re
    def tokenize(text):
        """Simple lowercase alphanumeric tokenization"""
        return re.findall(r'\b[a-z0-9]{2,}\b', text.lower())
    
    tokenized_corpus = [tokenize(chunk['text']) for chunk in chunks]
    bm25_index = BM25Okapi(tokenized_corpus)
    if verbose:
        print(f"  BM25 index built with {len(tokenized_corpus):,} documents")

    # --- Mappings ---
    c2t_path, t2c_path = ensure_mappings(chunks, triplets)

    # --- Three-Layer GIST Retriever ---
    retriever = ThreeLayerGISTRetriever(
        chunks_path=chunks_path,
        chunk_embeddings_qwen3_path=qwen3_path,
        triplets_path=triplets_path,
        chunk_to_triplets_path=c2t_path,
        triplet_to_chunks_path=t2c_path,
        triplet_bm25_path=triplet_bm25_path,
        bm25_lemmatized_index=bm25_index,
        top_k=top_k,
        colbert_model_name="colbert-ir/colbertv2.0",
        cross_encoder_model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    retriever.verbose = verbose  # Enable verbose output if -v flag set

    # --- Execute ---
    if verbose:
        print(f"\nQuerying: \"{query}\" (top_k={top_k})")

    results = retriever.retrieve(query)
    elapsed = time.time() - t0

    # --- Display ---
    display_results(results, chunks, query, verbose=verbose)
    print(f"⏱  {elapsed:.2f}s total")
    
    # --- Save to markdown if requested ---
    if output:
        save_results_to_markdown(results, chunks, query, output, elapsed)

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Query the Three-Layer GIST Retriever pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python query_three_layer.py "agentic memory methods"
  python query_three_layer.py "transformer attention" --top-k 20
  python query_three_layer.py "retrieval augmented generation" -v
        """
    )
    parser.add_argument("query", help="Search query string")
    parser.add_argument("--top-k", type=int, default=13, help="Number of results (default: 13)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--output", type=str, help="Save results to markdown file (e.g., results.md)")

    args = parser.parse_args()
    sys.exit(run_query(args.query, top_k=args.top_k, verbose=args.verbose, output=args.output))


if __name__ == "__main__":
    main()
