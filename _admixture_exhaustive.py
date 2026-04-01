import itertools
import json
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


CSV_PATH = "papers/post_processed/arxiv_data_with_analysis_cleaned.csv"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
QUERY = "methods for long context large language models"
FIELDS = ["title", "abstract", "utility"]
WEIGHT_STEP = 0.1

TARGET_IDS = [
    "2307.03172",  # Lost in the Middle
    "2306.15595",  # PI/YaRN
    "2312.00752",  # Mamba
    "2406.07522",  # Samba
    "2410.07145",  # Stuffed Mamba
    "2405.21060",  # Mamba-2
]


def norm_id(v: str) -> str:
    s = str(v).strip()
    return s.replace("_", "").replace(".", "")


def normalize_text(v: object) -> str:
    if pd.isna(v):
        return ""
    s = str(v)
    if s.strip().lower() in {"nan", "none", "null"}:
        return ""
    return s.replace("\n", " ").strip()


def cosine_scores(query_emb: np.ndarray, doc_embs: np.ndarray) -> np.ndarray:
    qn = np.linalg.norm(query_emb)
    dn = np.linalg.norm(doc_embs, axis=1)
    den = (dn * qn) + 1e-12
    return (doc_embs @ query_emb) / den


def rank_targets(sorted_ids: List[str], targets: List[str]) -> Dict[str, int]:
    pos = {norm_id(aid): i + 1 for i, aid in enumerate(sorted_ids)}
    return {t: pos.get(norm_id(t), 10**9) for t in targets}


def metrics_from_ranks(ranks: Dict[str, int]) -> Dict[str, float]:
    vals = list(ranks.values())
    n = len(vals)
    r10 = sum(r <= 10 for r in vals) / n
    r22 = sum(r <= 22 for r in vals) / n
    r50 = sum(r <= 50 for r in vals) / n
    mrr = sum((1.0 / r) if r < 10**9 else 0.0 for r in vals) / n
    return {"recall@10": r10, "recall@22": r22, "recall@50": r50, "mrr": mrr}


def subset_configs(fields: List[str]) -> List[Tuple[str, List[str]]]:
    out = []
    for r in range(1, len(fields) + 1):
        for combo in itertools.combinations(fields, r):
            name = "concat_" + "_".join(combo)
            out.append((name, list(combo)))
    return out


def weight_grid(step: float) -> List[Tuple[float, float, float]]:
    points = int(round(1.0 / step))
    out = []
    for i in range(points + 1):
        for j in range(points + 1 - i):
            k = points - i - j
            w1, w2, w3 = i * step, j * step, k * step
            if w1 == 0 and w2 == 0 and w3 == 0:
                continue
            out.append((w1, w2, w3))
    return out


def main() -> None:
    t0 = time.time()
    print("Loading CSV ...")
    df = pd.read_csv(CSV_PATH)
    for f in FIELDS + ["arxiv_id"]:
        if f not in df.columns:
            raise ValueError(f"Missing required field: {f}")

    df = df[df["arxiv_id"].notna()].copy()
    df["arxiv_id"] = df["arxiv_id"].astype(str).str.strip().str.strip('"')

    for f in FIELDS:
        df[f] = df[f].map(normalize_text)

    present_norm = set(df["arxiv_id"].map(norm_id).tolist())
    print(f"Rows: {len(df)}")
    print("Targets present:", [t for t in TARGET_IDS if norm_id(t) in present_norm])

    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print("Pre-embedding base fields (in-memory)...")
    field_embs = {}
    for f in FIELDS:
        embs = model.encode(df[f].tolist(), convert_to_numpy=True, show_progress_bar=False)
        field_embs[f] = embs
        print(f"  {f}: {embs.shape}")

    q_emb = model.encode([QUERY], convert_to_numpy=True, show_progress_bar=False)[0]

    results = []

    # 1) Exhaustive concat subsets (2^3 - 1 = 7)
    print("Running exhaustive concat subsets...")
    for name, subset in subset_configs(FIELDS):
        concat_texts = [" ".join([row[f] for f in subset if row[f]]) for _, row in df.iterrows()]
        doc_embs = model.encode(concat_texts, convert_to_numpy=True, show_progress_bar=False)
        scores = cosine_scores(q_emb, doc_embs)
        order = np.argsort(-scores)
        sorted_ids = df.iloc[order]["arxiv_id"].tolist()
        ranks = rank_targets(sorted_ids, TARGET_IDS)
        m = metrics_from_ranks(ranks)
        results.append({"name": name, "mode": "concat", "subset": subset, **ranks, **m})

    # 2) Exhaustive weight blends on title/abstract/utility simplex
    print("Running exhaustive weight-grid blends...")
    w_grid = weight_grid(WEIGHT_STEP)
    for wt, wa, wu in w_grid:
        # Skip pure single-field points already covered by concat singletons only if desired.
        # Keep them here to compare blend machinery apples-to-apples.
        combined = wt * field_embs["title"] + wa * field_embs["abstract"] + wu * field_embs["utility"]
        scores = cosine_scores(q_emb, combined)
        order = np.argsort(-scores)
        sorted_ids = df.iloc[order]["arxiv_id"].tolist()
        ranks = rank_targets(sorted_ids, TARGET_IDS)
        m = metrics_from_ranks(ranks)
        name = f"blend_t{wt:.1f}_a{wa:.1f}_u{wu:.1f}"
        results.append({
            "name": name,
            "mode": "blend",
            "weights": {"title": wt, "abstract": wa, "utility": wu},
            **ranks,
            **m,
        })

    out_df = pd.DataFrame(results)

    # Sort primary by recall@22 (your priority), then MRR, then recall@50
    sorted_primary = out_df.sort_values(["recall@22", "mrr", "recall@50"], ascending=[False, False, False])
    print("\nTop 15 by Recall@22, MRR, Recall@50")
    print(sorted_primary[["name", "mode", "recall@10", "recall@22", "recall@50", "mrr"]].head(15).to_string(index=False))

    best_r22 = sorted_primary.iloc[0]
    best_mrr = out_df.sort_values("mrr", ascending=False).iloc[0]

    print("\nBest by Recall@22:")
    print(best_r22[["name", "mode", "recall@10", "recall@22", "recall@50", "mrr"]].to_dict())

    print("\nBest by MRR:")
    print(best_mrr[["name", "mode", "recall@10", "recall@22", "recall@50", "mrr"]].to_dict())

    # Dump full results
    out_json = {
        "query": QUERY,
        "fields": FIELDS,
        "weight_step": WEIGHT_STEP,
        "num_configs": int(len(results)),
        "results": results,
    }
    with open("_admixture_exhaustive_results.json", "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2)

    out_df.to_csv("_admixture_exhaustive_results.csv", index=False)

    elapsed = time.time() - t0
    print(f"\nSaved: _admixture_exhaustive_results.json, _admixture_exhaustive_results.csv")
    print(f"Configs evaluated: {len(results)}")
    print(f"Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
