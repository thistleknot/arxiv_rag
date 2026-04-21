"""generate_qa_pairs_v2.py — QA pair generation from abstract + utility fields.

Motivation:
    The original generator (v1) sampled random paper sections, including tables
    and narrow numerical results.  A semantic retriever indexes title/abstract/
    utility; questions about table values cannot be matched semantically.

    V2 sources every question from abstract + utility — the same text the
    retriever indexes.  This aligns the eval signal with what the retriever can
    actually do.

Design:
    - Reads from papers/post_processed/arxiv_data_with_analysis_cleaned.csv
      (no pgvector required)
    - Source text = abstract  +  utility statements (if present)
    - Prompt steers toward contribution / capability questions (semantic, not
      factual tables)
    - Ollama /api/chat, 3-concurrent workers via asyncio + httpx
    - Output schema matches v1 for drop-in compatibility with _tune_retriever.py
    - Saves to eval/data/qa_pairs_v2.json; swap manually when ready

Usage:
    python eval/generate_qa_pairs_v2.py
    python eval/generate_qa_pairs_v2.py --n_target 150 --model qwen3.5:2b
    python eval/generate_qa_pairs_v2.py --out eval/data/qa_pairs.json  # overwrite
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Optional

import httpx

# ── Config ───────────────────────────────────────────────────────────────────

_ROOT        = Path(__file__).resolve().parent.parent
_CSV_PATH    = _ROOT / "papers" / "post_processed" / "arxiv_data_with_analysis_cleaned.csv"
_OUT_DEFAULT = Path(__file__).parent / "data" / "qa_pairs_v2.json"

OLLAMA_URL   = "http://127.0.0.1:11434/api/chat"
GEN_MODEL    = "qwen3.5:2b"
N_SAMPLE     = 200   # candidate papers to attempt
N_TARGET     = 150   # QA pairs to actually collect (after LLM filter)
TRAIN_FRAC   = 0.70
SEED         = 42
CONCURRENCY  = 3     # parallel Ollama calls
TIMEOUT      = 60.0  # seconds per LLM call

QA_PROMPT = """\
Below is the abstract and key contributions of an academic paper.

ABSTRACT:
{abstract}

KEY CONTRIBUTIONS:
{utility}

Your task:
1. Write ONE question about WHAT THIS PAPER ENABLES, CONTRIBUTES, OR PROPOSES.
   The question must be answerable using only the text above.
2. Write the answer to that question (at least 2 sentences, drawn from the text).

Rules:
- Do NOT ask about specific numbers, tables, or experimental results.
- Do NOT ask a yes/no question.
- Do NOT ask for the paper title or authors.
- The question should test understanding of the paper's main contribution or capability.

Respond in EXACTLY this format (nothing else, no extra text):
QUESTION: <your question>
ANSWER: <your answer>
"""


# ── Helpers ──────────────────────────────────────────────────────────────────

def _coerce_utility(v: str) -> str:
    if not v:
        return ""
    v = v.strip()
    if v.startswith("["):
        try:
            items = json.loads(v)
            if isinstance(items, list):
                return ". ".join(str(x).strip().rstrip(".") for x in items if x) + "."
        except (ValueError, json.JSONDecodeError):
            pass
    return v


def _clean(v: str) -> str:
    if not v:
        return ""
    s = str(v).strip()
    if s.lower() in {"nan", "none", "null"}:
        return ""
    return s.replace("\n", " ").strip()


def load_papers() -> list[dict]:
    """Load all papers from CSV; returns list of {arxiv_id, title, abstract, utility}."""
    rows = []
    with open(_CSV_PATH, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            arxiv_id = str(row.get("arxiv_id", "")).strip().strip('"')
            if not arxiv_id:
                continue
            abstract = _clean(row.get("abstract", ""))
            utility  = _coerce_utility(_clean(row.get("utility", "")))
            if len(abstract.split()) < 50:
                continue  # skip papers without a usable abstract
            rows.append({
                "arxiv_id": arxiv_id,
                "title":    _clean(row.get("title", "")),
                "abstract": abstract,
                "utility":  utility,
            })
    return rows


def _parse_qa(text: str) -> Optional[tuple[str, str]]:
    """Parse QUESTION: / ANSWER: from LLM response."""
    q_match = re.search(r"QUESTION:\s*(.+?)(?=\nANSWER:|$)", text, re.DOTALL)
    a_match = re.search(r"ANSWER:\s*(.+)", text, re.DOTALL)
    if not q_match or not a_match:
        return None
    question = q_match.group(1).strip()
    answer   = a_match.group(1).strip()
    # Remove <think>...</think> blocks that Qwen3 sometimes emits
    answer   = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()
    question = re.sub(r"<think>.*?</think>", "", question, flags=re.DOTALL).strip()
    # Quality filters
    if len(answer.split()) < 15:
        return None
    if question.lower().startswith(("is ", "are ", "was ", "were ", "do ", "does ", "did ", "has ", "have ")):
        return None
    if len(question) < 20:
        return None
    return question, answer


async def _generate_one(
    client: httpx.AsyncClient,
    paper: dict,
    semaphore: asyncio.Semaphore,
    model: str,
) -> Optional[dict]:
    """Call Ollama to generate a (question, answer) pair for one paper."""
    prompt = QA_PROMPT.format(
        abstract=paper["abstract"][:2000],
        utility=paper["utility"][:1000] if paper["utility"] else "(not available)",
    )
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": False,   # disable CoT chain for Qwen3 models
        "options": {"temperature": 0.3, "num_predict": 500},
    }
    async with semaphore:
        try:
            resp = await client.post(OLLAMA_URL, json=payload, timeout=TIMEOUT)
            resp.raise_for_status()
            text = resp.json()["message"]["content"].strip()
        except Exception as exc:
            print(f"  [LLM error] {paper['arxiv_id']}: {exc}", flush=True)
            return None

    parsed = _parse_qa(text)
    if parsed is None:
        return None
    question, answer = parsed
    section_text = (
        f"ABSTRACT: {paper['abstract']}\n\nKEY CONTRIBUTIONS: {paper['utility']}"
        if paper["utility"]
        else f"ABSTRACT: {paper['abstract']}"
    )
    return {
        "question":     question,
        "answer":       answer,
        "section_text": section_text,
        "paper_id":     paper["arxiv_id"].replace(".", "_"),  # normalise to YYMM_NNNNN
        "section_idx":  -1,
        "chunk_ids":    [],
        "source_field": "abstract_utility",
    }


async def generate_all(papers: list[dict], model: str) -> list[dict]:
    """Generate QA pairs for all sampled papers using CONCURRENCY workers."""
    semaphore = asyncio.Semaphore(CONCURRENCY)
    pairs: list[dict] = []

    async with httpx.AsyncClient() as client:
        tasks = [_generate_one(client, p, semaphore, model) for p in papers]
        for i, coro in enumerate(asyncio.as_completed(tasks)):
            result = await coro
            status = "✓" if result else "✗"
            if result:
                pairs.append(result)
            print(
                f"  [{i+1:3d}/{len(tasks)}] {status}  "
                f"({len(pairs)} collected)",
                flush=True,
            )
    return pairs


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Generate QA pairs from abstract+utility")
    ap.add_argument("--n_target",  type=int, default=N_TARGET)
    ap.add_argument("--n_sample",  type=int, default=N_SAMPLE)
    ap.add_argument("--model",     default=GEN_MODEL)
    ap.add_argument("--seed",      type=int, default=SEED)
    ap.add_argument("--out",       default=str(_OUT_DEFAULT))
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    random.seed(args.seed)

    print("=" * 60)
    print("QA Pair Generator v2  (abstract + utility source)")
    print("=" * 60)
    print(f"  Model   : {args.model}")
    print(f"  Target  : {args.n_target} pairs from {args.n_sample} candidates")
    print(f"  Output  : {args.out}")
    print()

    print("[1/3] Loading papers from CSV...")
    all_papers = load_papers()
    print(f"  {len(all_papers)} papers with usable abstracts")

    candidates = random.sample(all_papers, min(args.n_sample, len(all_papers)))
    # Overshoot to account for LLM failures
    candidates = candidates[: args.n_target + 40]
    print(f"  Sampling {len(candidates)} candidates")

    print(f"\n[2/3] Generating QA pairs ({CONCURRENCY} concurrent workers)...")
    pairs = asyncio.run(generate_all(candidates, args.model))
    pairs = pairs[: args.n_target]
    print(f"\n  Generated {len(pairs)} valid QA pairs")

    if not pairs:
        print("  ERROR: too few pairs — aborting. Check Ollama is running.", file=sys.stderr)
        sys.exit(1)

    print("\n[3/3] Splitting and saving...")
    random.shuffle(pairs)
    n_train = int(len(pairs) * TRAIN_FRAC)
    train   = pairs[:n_train]
    test    = pairs[n_train:]

    output = {
        "meta": {
            "n_total":   len(pairs),
            "n_train":   len(train),
            "n_test":    len(test),
            "gen_model": args.model,
            "seed":      args.seed,
            "source":    "abstract_utility",
        },
        "train": train,
        "test":  test,
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)

    print(f"  Saved → {args.out}")
    print(f"  train={len(train)}  test={len(test)}")
    print("\nDone.")


if __name__ == "__main__":
    main()
