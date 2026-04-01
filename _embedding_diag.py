"""
Embedding model diagnostic — is all-MiniLM-L6-v2 the bottleneck?

Tests cosine similarity between the query and utility strings for:
  - Top-ranked papers (should be high)
  - Gap papers, Mamba/SSM utilities (should reveal the gap)
  - Thesis/title strings for gap papers (see if other fields bridge it)
  - Semantic neighbour probes ("efficient sequence modeling" ↔ "long context")
"""
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL = "all-MiniLM-L6-v2"
QUERY = "methods for long context large language models"

# ── probes ──────────────────────────────────────────────────────────────────
PROBES = {
    # Current top-rankers (utility strings from the chain)
    "TOP#1 LongBench utility":
        "Facilitates comprehensive evaluation of large language models' long context "
        "understanding across various tasks and languages.",
    "TOP#2 CoA utility":
        "Enabling effective processing of long contexts for Large Language Models (LLMs), "
        "addressing current limitations in input length reduction and context window extension strategies.",
    "TOP#5 HSA utility":
        "Enables handling ultra-long contexts in text processing. "
        "Achieves over 90% accuracy on in-context retrieval tasks with contexts up to 16M.",

    # Gap paper utilities
    "GAP Mamba utility":
        "general sequence model backbone for various modalities, million-length sequences",
    "GAP Mamba-2 utility":
        "SSM, Mamba-2, 2-8X faster",
    "GAP Stuffed Mamba utility":
        "RNN-based architecture for long context modeling without forgetting",
    "GAP Samba utility":
        "1M without explicit training on such long contexts",
    "GAP RoPE/PI utility":
        "extending context window sizes, position interpolation, positional encoding",

    # Alternative fields for gap papers
    "Samba THESIS":
        "modeling sequences with infinite context length",
    "Stuffed Mamba THESIS":
        "cause of inability to process long context for RNNs",
    "Stuffed Mamba TITLE":
        "RNN-Based Long-Context Modeling",
    "Mamba TITLE":
        "Mamba: Linear-Time Sequence Modeling with Selective State Spaces",
    "Mamba abstract snippet":
        "Foundation models, now powering most of the exciting applications in deep learning, "
        "are almost universally based on the Transformer architecture and its core attention module. "
        "Many subquadratic-time architectures such as linear attention, gated convolution and "
        "recurrent models, and structured state space models (SSMs) have been developed to address "
        "Transformers' computational inefficiency on long sequences.",

    # Semantic neighbour probes — the user's hypothesis
    "PROBE efficient sequence modeling":
        "efficient sequence modeling",
    "PROBE linear recurrence long context":
        "linear recurrence for long context sequences",
    "PROBE state space models":
        "state space models for sequence modeling",
    "PROBE subquadratic attention":
        "subquadratic time attention alternative",
    "PROBE long range dependencies":
        "modeling long range dependencies efficiently",
    "PROBE sliding window attention":
        "sliding window attention for long sequences",
}

def main():
    print(f"Model: {MODEL}")
    print(f"Query: '{QUERY}'\n")
    model = SentenceTransformer(MODEL)

    q_emb = model.encode(QUERY, normalize_embeddings=True)
    labels = list(PROBES.keys())
    texts  = list(PROBES.values())
    embs   = model.encode(texts, normalize_embeddings=True, batch_size=64)

    sims = embs @ q_emb

    # Sort descending
    order = np.argsort(-sims)
    print(f"{'Label':<42}  {'Sim':>6}  {'Context'}")
    print("-" * 90)
    for i in order:
        label = labels[i]
        sim   = sims[i]
        ctx   = "TOP" if label.startswith("TOP") else ("GAP" if label.startswith("GAP") else "probe")
        print(f"{label:<42}  {sim:>6.3f}  [{ctx}]")

    # ── direct pairwise: "efficient sequence modeling" ↔ "long context" ──
    print("\n── Pairwise probes (no query) ──────────────────────────────────────")
    pairs = [
        ("efficient sequence modeling",
         "long context large language models"),
        ("linear recurrence sequence model",
         "long context large language models"),
        ("state space model selective ssm",
         "long context large language models"),
        ("context window extension position interpolation",
         "long context large language models"),
        ("efficient sequence modeling",
         "efficient sequence modeling"),   # self-similarity sanity check
    ]
    pair_texts = list({t for p in pairs for t in p})
    pair_embs  = {t: model.encode(t, normalize_embeddings=True) for t in pair_texts}
    for a, b in pairs:
        sim = float(pair_embs[a] @ pair_embs[b])
        print(f"  {sim:>5.3f}  '{a}'  ←→  '{b}'")

    # ── gap score summary ──
    print("\n── Key gap (what model 'sees' vs what query needs) ──────────────────")
    top_sim  = np.mean([sims[labels.index(l)] for l in labels if l.startswith("TOP")])
    gap_util = np.mean([sims[labels.index(l)] for l in labels if l.startswith("GAP") and "utility" in l.lower()])
    gap_other = np.mean([sims[labels.index(l)] for l in labels
                         if l.startswith(("Samba", "Stuffed", "Mamba")) and "utility" not in l.lower()])
    print(f"  Mean sim — top-ranked utilities:     {top_sim:.3f}")
    print(f"  Mean sim — gap paper utilities:      {gap_util:.3f}  ← this is the bottleneck")
    print(f"  Mean sim — gap paper title/thesis:   {gap_other:.3f}  ← does alternative field help?")
    print(f"\n  Gap magnitude: {top_sim - gap_util:.3f}  ('bad embedding' threshold ≈ 0.15)")

if __name__ == "__main__":
    main()
