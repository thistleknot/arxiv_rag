"""
Investigate why adaptive vocabulary has lower similarity than expected
"""
import msgpack
import numpy as np

# Load baseline (100 chunks)
print("Loading baseline extraction (100 chunks)...")
with open('arxiv_graph_sparse.msgpack', 'rb') as f:
    baseline_data = msgpack.unpack(f, raw=False, strict_map_key=False)
    baseline = baseline_data['forward_index']

print(f"  Chunks: {len(baseline)}")
baseline_tokens_per_chunk = [len(c.get('sparse_vector', {})) for c in baseline.values()]
print(f"  Avg tokens/chunk: {np.mean(baseline_tokens_per_chunk):.1f}")
print(f"  Token range: {np.min(baseline_tokens_per_chunk)}-{np.max(baseline_tokens_per_chunk)}")

# Load adaptive vocabulary
print("\nLoading adaptive vocabulary...")
with open('graph_vocab_adaptive.msgpack', 'rb') as f:
    adaptive = msgpack.unpack(f, raw=False, strict_map_key=False)

print(f"  Train size: {adaptive['train_size']}")
print(f"  Vocab size: {len(adaptive['important_tokens'])}")
print(f"  Test similarity: {adaptive['test_similarity']:.4f}")

print("\nTraining history:")
for h in adaptive['history']:
    print(f"  Iter {h['iteration']}: {h['train_size']:3} chunks → {h['vocab_size']:4} tokens → {h['similarity']:.4f} sim (Δ{h['improvement']:+.4f})")

# Compare vocabularies
baseline_vocab = set()
for chunk_data in baseline.values():
    for token_id in chunk_data.get('sparse_vector', {}).keys():
        tid = int(token_id) if isinstance(token_id, str) else token_id
        baseline_vocab.add(tid)

adaptive_vocab = set(adaptive['important_tokens'])

print(f"\nVocabulary comparison:")
print(f"  Baseline (100 chunks): {len(baseline_vocab)} unique tokens")
print(f"  Adaptive (300 chunks): {len(adaptive_vocab)} unique tokens")
print(f"  Overlap: {len(baseline_vocab & adaptive_vocab)} tokens")
print(f"  Baseline coverage: {len(baseline_vocab & adaptive_vocab)/len(baseline_vocab)*100:.1f}%")

# Check token frequency distribution
token_freq = adaptive['token_freq']
freqs = sorted(token_freq.values(), reverse=True)
print(f"\nToken frequency distribution (adaptive):")
print(f"  Total tokens: {len(token_freq)}")
print(f"  Tokens with freq≥2: {len(adaptive_vocab)} ({len(adaptive_vocab)/len(token_freq)*100:.1f}%)")
print(f"  Top 10 frequencies: {freqs[:10]}")
print(f"  Median frequency: {np.median(freqs)}")

# Hypothesis: Test set might be too different from training
print("\n" + "="*70)
print("HYPOTHESIS: Test set distribution mismatch?")
print("="*70)
print("\nPossible issues:")
print("  1. Test set chunks may be fundamentally different from training")
print("  2. 50 test chunks may be too small for stable evaluation")
print("  3. Random sampling might have captured unrepresentative chunks")
print("  4. Vocabulary filtering (freq≥2) might be too aggressive")

print("\nNext steps:")
print("  A. Increase test set to 100-200 chunks for more stable evaluation")
print("  B. Lower frequency threshold (freq≥1) to keep more vocabulary")
print("  C. Use stratified sampling instead of random (ensure diversity)")
print("  D. Compare actual extracted vectors side-by-side for specific chunks")
