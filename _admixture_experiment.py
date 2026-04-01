"""
Admixture Experiment Harness — Index Source Comparison

Core Thesis:
    The retrieval vocabulary mismatch (Mamba/SSM/RoPE papers under-ranked) is caused
    by embedding only the LLM-generated `utility` field. Utility strips domain vocabulary
    in favour of functional benefit language. Other CSV fields (title, thesis, abstract,
    barriers) retain domain vocabulary and may bridge the gap.

Target papers (known to be under-retrieved):
    - 2307.03172  Lost in the Middle      (baseline rank #6  — present but under-weighted)
    - 2306.15595  RoPE/PI                 (baseline rank #18 — barely present)
    - 2312.00752  Mamba                   (NOT in top-22)
    - 2406.07522  Samba unlimited context (NOT in top-22)
    - 2410.07145  Stuffed Mamba LC        (NOT in top-22)
    - 2405.21060  Transformers are SSMs   (NOT in top-22)

Experiments:
    concat_* — concatenate fields → one embedding per paper
    blend_*  — separate embeddings → weighted cosine average

Metrics per configuration:
    - Rank of each target paper (lower = better)
    - Recall@10, Recall@22, Recall@50
    - MRR (mean reciprocal rank) across target set

Usage:
    python _admixture_experiment.py
"""

import ast
import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# ───────────── config ─────────────
CSV = "papers/post_processed/arxiv_data_with_analysis_cleaned.csv"
QUERY = "methods for long context large language models"
MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 256  # fast on CPU

TARGET_IDS = [
    "2307.03172",  # Lost in the Middle
    "2306.15595",  # RoPE / Positional Interpolation
    "2312.00752",  # Mamba
    "2406.07522",  # Samba unlimited context
    "2410.07145",  # Stuffed Mamba (long-context)
    "2405.21060",  # Transformers are SSMs
]


# ───────────── helpers ─────────────
def _coerce(val, field):
    """Parse JSON list → single string; return plain string as-is."""
    if pd.isna(val) or val == "":
        return ""
    s = str(val).strip()
    if s.startswith("["):
        try:
            items = ast.literal_eval(s)
            return " ".join(str(i) for i in items)
        except Exception:
            pass
    return s


def build_text(df, fields):
    """Concatenate multiple CSV columns into one text string per row."""
    parts = []
    for _, row in df.iterrows():
        tokens = []
        for f in fields:
            val = _coerce(row.get(f, ""), f)
            if val:
                tokens.append(val)
        parts.append(" ".join(tokens))
    return parts


def embed(model, texts):
    """Encode, return L2-normalised float32 matrix [N, D]."""
    embs = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embs.astype(np.float32)


def cosine_rank(query_emb, doc_embs):
    """Returns 0-based ranks (0 = best)."""
    scores = doc_embs @ query_emb  # already L2-normalised
    order = np.argsort(-scores)
    return order, scores


def recall_at_k(ranks_of_targets, k, total):
    """Fraction of target papers in top-k."""
    found = sum(1 for r in ranks_of_targets if r < k)
    return found / len(ranks_of_targets)


def mrr(ranks_of_targets):
    """Mean reciprocal rank (1-based rank)."""
    rr = [1.0 / (r + 1) for r in ranks_of_targets]
    return np.mean(rr)


# ───────────── main ─────────────
def run():
    print(f"Loading CSV …")
    df = pd.read_csv(CSV)
    df["arxiv_id"] = df["arxiv_id"].astype(str).str.strip('"')
    df = df.reset_index(drop=True)
    N = len(df)
    print(f"  {N} papers loaded")

    # Map arxiv_id → row index
    id_to_idx = {aid: i for i, aid in enumerate(df["arxiv_id"])}

    # Which targets are in the corpus?
    targets_present = [t for t in TARGET_IDS if t in id_to_idx]
    targets_missing = [t for t in TARGET_IDS if t not in id_to_idx]
    if targets_missing:
        print(f"  [WARN] Not in CSV: {targets_missing}")
    print(f"  Target papers present: {targets_present}\n")

    print(f"Loading sentence-transformer: {MODEL_NAME} …")
    model = SentenceTransformer(MODEL_NAME)

    # Embed the query once
    q_emb = embed(model, [QUERY])[0]  # shape [D]

    # ── experiment definitions ──────────────────────────────────────────────
    # Each entry: (name, mode, spec)
    #   mode="concat"  → spec = list of field names to concatenate
    #   mode="blend"   → spec = list of (field, weight) tuples
    EXPERIMENTS = [
        # ── concat ──
        ("baseline_utility",            "concat", ["utility"]),
        ("title_only",                  "concat", ["title"]),
        ("abstract_only",               "concat", ["abstract"]),
        ("thesis_only",                 "concat", ["thesis"]),
        ("barriers_only",               "concat", ["barriers"]),
        ("title_utility",               "concat", ["title", "utility"]),
        ("utility_thesis",              "concat", ["utility", "thesis"]),
        ("utility_thesis_barriers",     "concat", ["utility", "thesis", "barriers"]),
        ("title_utility_thesis",        "concat", ["title", "utility", "thesis"]),
        ("all_fields",                  "concat", ["title", "abstract", "utility", "thesis", "barriers"]),
        # ── blend ──
        ("blend_title60_util40",        "blend",  [("title", 0.6), ("utility", 0.4)]),
        ("blend_util60_thesis40",       "blend",  [("utility", 0.6), ("thesis", 0.4)]),
        ("blend_util40_abst40_tit20",   "blend",  [("utility", 0.4), ("abstract", 0.4), ("title", 0.2)]),
        ("blend_equal_util_thesis_tit", "blend",  [("utility", 0.34), ("thesis", 0.33), ("title", 0.33)]),
        # ── query expansion × utility ──
        ("qex_utility",                 "qex",    ["utility"]),
    ]

    QUERY_EXPANDED = (
        "methods for long context large language models "
        "context window extension state space model linear recurrence "
        "sparse attention position interpolation RoPE SSM "
        "ultra-long sequence long-range dependency"
    )
    q_emb_expanded = embed(model, [QUERY_EXPANDED])[0]

    # ── pre-embed each field once ────────────────────────────────────────────
    all_fields_needed = {"title", "abstract", "utility", "thesis", "barriers"}
    print("Pre-embedding all fields …")
    field_embs = {}
    for f in sorted(all_fields_needed):
        texts = [_coerce(row.get(f, ""), f) for _, row in df.iterrows()]
        field_embs[f] = embed(model, texts)
        print(f"  {f}: done  shape={field_embs[f].shape}")
    print()

    # ── run experiments ─────────────────────────────────────────────────────
    results = []

    for name, mode, spec in EXPERIMENTS:
        if mode == "concat":
            # Build one combined embedding per paper
            texts = build_text(df, spec)
            doc_embs = embed(model, texts)
            q = q_emb
        elif mode == "qex":
            texts = build_text(df, spec)
            doc_embs = embed(model, texts)
            q = q_emb_expanded
        elif mode == "blend":
            # Weighted sum of pre-computed field embeddings
            doc_embs = np.zeros((N, field_embs["utility"].shape[1]), dtype=np.float32)
            for f, w in spec:
                doc_embs += w * field_embs[f]
            # Re-normalise after blend
            norms = np.linalg.norm(doc_embs, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            doc_embs = doc_embs / norms
            q = q_emb

        order, scores = cosine_rank(q, doc_embs)
        # 0-based rank of each target
        rank_map = {aid: int(np.where(order == id_to_idx[aid])[0][0])
                    for aid in targets_present}

        rec10  = recall_at_k(list(rank_map.values()), k=10,  total=N)
        rec22  = recall_at_k(list(rank_map.values()), k=22,  total=N)
        rec50  = recall_at_k(list(rank_map.values()), k=50,  total=N)
        mrr_v  = mrr(list(rank_map.values()))

        results.append({
            "name": name,
            "mode": mode,
            "spec": str(spec),
            **{f"rank_{t.replace('.','_')}": rank_map.get(t, -1) for t in targets_present},
            "recall@10":  round(rec10,  3),
            "recall@22":  round(rec22,  3),
            "recall@50":  round(rec50,  3),
            "mrr":        round(mrr_v,  4),
        })
        print(f"[{name:35s}]  R@10={rec10:.2f} R@22={rec22:.2f} R@50={rec50:.2f} MRR={mrr_v:.3f}  "
              + "  ".join(f"{t.replace('.','_')[-7:]}=#{rank_map.get(t,-1)+1}" for t in targets_present))

    # ── summary table ───────────────────────────────────────────────────────
    print("\n" + "=" * 110)
    print("SUMMARY (sorted by MRR desc)")
    print("=" * 110)
    res_df = pd.DataFrame(results).sort_values("mrr", ascending=False)
    pd.set_option("display.max_colwidth", 40)
    pd.set_option("display.width", 200)
    print(res_df[["name", "recall@10", "recall@22", "recall@50", "mrr"]].to_string(index=False))

    out = "_admixture_results.json"
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nFull results saved → {out}")


if __name__ == "__main__":
    run()
