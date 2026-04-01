"""Apply trained BIO tagger (bio_tagger_atomic.pt) to all 161K corpus chunks.

Pipeline per chunk:
  1. Split chunk text into sentences (regex)
  2. Batch-infer BIO probs through BERT (batch_size sentences at a time)
  3. Threshold at 0.4585 (tuned via HPO)
  4. Extract spans → reconstruct triplets (inference_bio_tagger.py pipeline)
     includes: copula recovery, possession rewrite, deduplication
  5. Join BERT wordpiece tokens into clean text

Input:  checkpoints/chunks.msgpack (161,389 chunks)
Model:  bio_tagger_atomic.pt  (Macro F1=0.7887, threshold=0.4585)
Output: checkpoints/bio_triplets_full_corpus.msgpack

Checkpoint: checkpoints/bio_triplets_checkpoint.msgpack  (every 5000 chunks)
  → delete this file after successful run to free space
"""

import sys
import os
import re
import json
import time
import msgpack
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Optional

# ──────────────────────────────────────────────────────────────────────────────
# sys.path: point at training/ so inference_bio_tagger.py can import BIOTagger
# ──────────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / 'training'))

from inference_bio_tagger import BIOTripletExtractor

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
MODEL_PATH       = str(ROOT / 'bio_tagger_atomic.pt')
CHUNKS_PATH      = str(ROOT / 'checkpoints' / 'chunks.msgpack')
OUTPUT_PATH      = str(ROOT / 'checkpoints' / 'bio_triplets_full_corpus.msgpack')
CHECKPOINT_PATH  = str(ROOT / 'checkpoints' / 'bio_triplets_checkpoint.msgpack')
HP_PATH          = str(ROOT / 'best_hyperparams.json')
CHECKPOINT_EVERY = 5_000      # save progress every N chunks
SENTENCE_BATCH   = 128        # sentences per BERT forward pass (GPU-optimal)
DEFAULT_THRESHOLD = 0.4585


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_threshold() -> float:
    """Read tuned threshold from best_hyperparams.json if available."""
    hp = Path(HP_PATH)
    if hp.exists():
        with open(hp) as f:
            data = json.load(f)
        # Support both top-level and nested 'params' key
        t = data.get('threshold') or data.get('params', {}).get('threshold')
        if t is not None:
            return float(t)
    return DEFAULT_THRESHOLD


def simple_sentence_split(text: str) -> List[str]:
    """
    Fast regex sentence splitter.
    Splits on . ! ? followed by whitespace + capital letter.
    Long sentences are left intact; BERT handles truncation at 512 tokens.
    """
    sents = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in sents if len(s.strip()) > 10]


def join_wordpieces(text: str) -> str:
    """
    Join BERT wordpiece tokens back into clean text.
    'transform ##er model' → 'transformer model'
    """
    return re.sub(r'\s*##', '', text)


def clean_triplet(t: Dict) -> Dict:
    """
    Post-process a single triplet dict:
    - join wordpiece artifacts in all text fields
    - remove empty-string fields
    """
    out = {}
    for field in ('subject', 'predicate', 'object'):
        val = t.get(field)
        if val:
            val = join_wordpieces(val).strip()
        out[field] = val if val else None

    # Carry through metadata fields
    for field in ('arity', 'relation_type'):
        if field in t:
            out[field] = t[field]

    return out


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    threshold = load_threshold()
    print(f"Threshold: {threshold:.4f}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device:    {device}")

    # Load model
    print(f"\nLoading model from {MODEL_PATH} ...")
    extractor = BIOTripletExtractor(MODEL_PATH, device=device)
    print(f"Active labels: {extractor.active_labels}\n")

    # Load chunks
    print(f"Loading chunks from {CHUNKS_PATH} ...")
    with open(CHUNKS_PATH, 'rb') as f:
        chunks = msgpack.unpackb(f.read(), raw=False)
    total_chunks = len(chunks)
    print(f"Loaded {total_chunks:,} chunks\n")

    # Resume from checkpoint if present
    results: List[Dict] = []
    start_idx = 0
    if Path(CHECKPOINT_PATH).exists():
        print(f"Found checkpoint — resuming from {CHECKPOINT_PATH} ...")
        with open(CHECKPOINT_PATH, 'rb') as f:
            results = msgpack.unpackb(f.read(), raw=False)
        start_idx = len(results)
        print(f"Resuming from chunk {start_idx:,} / {total_chunks:,}\n")

    # ── Processing loop ───────────────────────────────────────────────────────
    start_time = time.time()
    total_triplets = sum(len(r['triplets']) for r in results)

    for chunk_idx in tqdm(range(start_idx, total_chunks),
                          desc='BIO inference', unit='chunk',
                          initial=start_idx, total=total_chunks):

        chunk = chunks[chunk_idx]
        text  = chunk.get('text', '') or chunk.get('chunk_text', '') or ''

        meta = {
            'doc_id':      chunk.get('doc_id', ''),
            'paper_id':    chunk.get('paper_id', ''),
            'section_idx': chunk.get('section_idx', 0),
            'chunk_idx':   chunk.get('chunk_idx', chunk_idx),
        }

        if not text.strip():
            results.append({**meta, 'triplets': []})
            continue

        # Split chunk into sentences
        sentences = simple_sentence_split(text)
        if not sentences:
            sentences = [text[:512]]

        # Batch inference
        all_triplets: List[Dict] = []
        for i in range(0, len(sentences), SENTENCE_BATCH):
            batch = sentences[i:i + SENTENCE_BATCH]
            batch_results = extractor.extract_triplets_batch(batch, threshold=threshold)
            for sent_triplets in batch_results:
                all_triplets.extend(sent_triplets)

        # Clean wordpieces + deduplicate across sentences
        seen: set = set()
        deduped: List[Dict] = []
        for t in all_triplets:
            ct = clean_triplet(t)
            key = (ct['subject'], ct['predicate'], ct.get('object'))
            if key not in seen and ct['subject'] and ct['predicate']:
                seen.add(key)
                deduped.append(ct)

        total_triplets += len(deduped)
        results.append({**meta, 'triplets': deduped})

        # Checkpoint every N chunks
        if (chunk_idx + 1) % CHECKPOINT_EVERY == 0:
            with open(CHECKPOINT_PATH, 'wb') as f:
                msgpack.pack(results, f)
            elapsed = time.time() - start_time
            processed = chunk_idx + 1 - start_idx
            rate = processed / elapsed
            eta_s  = (total_chunks - chunk_idx - 1) / rate if rate > 0 else 0
            print(
                f"\n  [{chunk_idx+1:,}/{total_chunks:,}] checkpoint saved"
                f" | {total_triplets:,} triplets"
                f" | {rate:.1f} chunks/sec"
                f" | ETA {eta_s/60:.0f} min"
            )

    # ── Save final output ─────────────────────────────────────────────────────
    print(f"\nSaving {len(results):,} records to {OUTPUT_PATH} ...")
    with open(OUTPUT_PATH, 'wb') as f:
        msgpack.pack(results, f)

    elapsed = time.time() - start_time
    non_empty = sum(1 for r in results if r['triplets'])
    avg_triplets = total_triplets / max(1, non_empty)

    print(f"\n{'='*60}")
    print(f"DONE")
    print(f"{'='*60}")
    print(f"  Chunks processed:      {len(results):,}")
    print(f"  Chunks with triplets:  {non_empty:,}  ({non_empty/len(results)*100:.1f}%)")
    print(f"  Total triplets:        {total_triplets:,}")
    print(f"  Avg triplets/chunk:    {avg_triplets:.1f}")
    print(f"  Elapsed:               {elapsed:.0f}s  ({elapsed/60:.1f} min)")
    print(f"  Output:                {OUTPUT_PATH}")
    print(f"  Checkpoint (delete):   {CHECKPOINT_PATH}")


if __name__ == '__main__':
    main()
