"""
Utility Retriever — on-the-fly TF-IDF search over the `utility` column.

Core Thesis:
    Index the flattened utility bullet-points for every paper; rank by cosine
    similarity to a free-text query.  No DB or embeddings required — small
    enough (~2152 rows) to build in memory in under a second.

Usage:
    # interactive loop
    python utility_retriever.py

    # single query
    python utility_retriever.py --query "efficient attention in transformers" --top_k 10

    # show full abstract
    python utility_retriever.py --query "hallucination reduction" --top_k 5 --full
"""

import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CSV = Path(r"C:\Users\user\arxiv_id_lists\papers\post_processed\arxiv_data_with_analysis_cleaned.csv")
DEFAULT_TOP_K = 10
TFIDF_MAX_FEATURES = 8000


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _parse_json_list(raw: str) -> list[str]:
    """Parse a JSON array string into a list of strings."""
    if not isinstance(raw, str):
        return []
    raw = raw.strip()
    # Try JSON first, then ast.literal_eval as fallback
    for parser in (json.loads, ast.literal_eval):
        try:
            result = parser(raw)
            if isinstance(result, list):
                return [str(x) for x in result]
        except Exception:
            pass
    # Last resort: treat as plain text
    return [raw]


def _flatten(points: list[str]) -> str:
    return " ".join(points)


def _strip_quotes(s: str) -> str:
    """Strip surrounding quotes often present in this CSV."""
    if isinstance(s, str):
        return s.strip('"').strip("'")
    return s


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8", encoding_errors="replace")
    df = df.dropna(subset=["utility"])
    df = df[df["utility"].str.strip().str.len() > 4]
    df["arxiv_id"] = df["arxiv_id"].apply(_strip_quotes)
    df["utility_points"] = df["utility"].apply(_parse_json_list)
    df["utility_text"] = df["utility_points"].apply(_flatten)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Index builder
# ---------------------------------------------------------------------------

class UtilityRetriever:
    """TF-IDF retriever indexed on the utility column."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self._vectorizer = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self._matrix = self._vectorizer.fit_transform(df["utility_text"])

    # ------------------------------------------------------------------
    def retrieve(self, query: str, top_k: int = DEFAULT_TOP_K) -> pd.DataFrame:
        qv = self._vectorizer.transform([query])
        scores = cosine_similarity(qv, self._matrix).flatten()
        idx = np.argsort(scores)[::-1][:top_k]
        results = self.df.iloc[idx][
            ["arxiv_id", "title", "abstract", "utility_text", "utility_points"]
        ].copy()
        results["score"] = scores[idx]
        return results

    # ------------------------------------------------------------------
    @classmethod
    def from_csv(cls, path: Path = CSV) -> "UtilityRetriever":
        df = load_csv(path)
        print(f"[UtilityRetriever] Indexed {len(df)} papers on utility column.")
        return cls(df)


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def _truncate(s: str, n: int = 300) -> str:
    return s if len(s) <= n else s[:n] + "…"


def display_results(
    results: pd.DataFrame,
    show_full_abstract: bool = False,
    abstract_len: int = 400,
) -> None:
    if results.empty:
        print("  No results.")
        return
    for rank, (_, row) in enumerate(results.iterrows(), start=1):
        print(f"\n{'─'*72}")
        print(f"  #{rank:2d}  [{row['score']:.4f}]  {row['arxiv_id']}")
        print(f"       {row['title']}")
        abstr = row["abstract"] or ""
        if not show_full_abstract:
            abstr = _truncate(str(abstr), abstract_len)
        print(f"\n  Abstract:\n  {abstr}")
        print(f"\n  Utility:")
        points = row["utility_points"] if isinstance(row["utility_points"], list) else []
        if points:
            for pt in points:
                print(f"    • {pt}")
        else:
            print(f"    {row['utility_text']}")
    print(f"\n{'─'*72}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Utility-column paper retriever")
    p.add_argument("--query", "-q", type=str, default=None, help="Query string")
    p.add_argument("--top_k", "-k", type=int, default=DEFAULT_TOP_K)
    p.add_argument("--full", action="store_true", help="Show full abstract")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    retriever = UtilityRetriever.from_csv()

    if args.query:
        print(f"\nQuery: {args.query!r}  (top {args.top_k})\n")
        results = retriever.retrieve(args.query, top_k=args.top_k)
        display_results(results, show_full_abstract=args.full)
        return

    # Interactive loop
    print("Utility retriever ready. Type a query or 'q' to quit.")
    while True:
        try:
            query = input("\nQuery> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not query or query.lower() in ("q", "quit", "exit"):
            break
        results = retriever.retrieve(query, top_k=args.top_k)
        display_results(results, show_full_abstract=args.full)


if __name__ == "__main__":
    main()
