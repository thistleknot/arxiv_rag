"""
RAGAS Evaluation Runner

Core Thesis:
    Given a set of QA pairs and an ArxivRetriever instance, run the 4
    standard RAGAS metrics and return aggregate scores.

Metrics (RAGAS 0.2.x):
    - context_precision   : are retrieved contexts relevant to the question?
    - context_recall      : do retrieved contexts cover the reference answer?
    - faithfulness        : is the LLM answer grounded in retrieved contexts?
    (AnswerRelevancy excluded — Copilot proxy has no /v1/embeddings endpoint)

Workflow:
    1. For each QA pair: retriever.search(question) → retrieved contexts
    2. Generate answer from contexts using the same LLM
    3. Build SingleTurnSample(user_input, reference, response, retrieved_contexts)
    4. evaluate() → per-sample scores → aggregate mean

Public API:
    run_eval(retriever, qa_pairs, top_k=5, llm_model=..., n_pairs=None) -> dict
"""

import os
import sys
import json
from typing import Optional

import openai

# ── Imports ───────────────────────────────────────────────────────────────────
from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
)
# Note: AnswerRelevancy is excluded — it requires /v1/embeddings which the
# GitHub Copilot proxy does not expose.  Three LLM-only metrics are used.
from langchain_openai import ChatOpenAI

# ── Config ────────────────────────────────────────────────────────────────────
COPILOT_PROXY = "http://127.0.0.1:8069/v1"
DEFAULT_MODEL = "gpt-4.1"   # via GitHub Copilot proxy

# Set env vars so RAGAS internal OpenAIEmbeddings() fallback also hits the proxy
os.environ.setdefault("OPENAI_API_KEY",  "dummy-key")
os.environ.setdefault("OPENAI_BASE_URL", COPILOT_PROXY)

from ragas import SingleTurnSample


def _extract_contexts(docs) -> list[str]:
    """
    Extract text strings from RetrievedDoc objects.

    Top-level docs aggregated at paper level have content=''; the actual text
    lives in doc.sections[*].content.  Fall back to sections when content is empty.
    """
    contexts: list[str] = []
    for d in docs:
        if d.content:
            contexts.append(d.content)
        elif hasattr(d, "sections") and d.sections:
            for sec in d.sections:
                if sec.content:
                    contexts.append(sec.content)
    return contexts


def _generate_answer(question: str, contexts: list[str],
                     model: str = DEFAULT_MODEL, max_tokens: int = 300) -> str:
    """
    Generate a grounded answer from retrieved contexts via Copilot proxy.
    Used to populate SingleTurnSample.response.
    """
    client = openai.OpenAI(api_key="dummy-key", base_url=COPILOT_PROXY)
    joined = "\n\n---\n\n".join(contexts[:5])
    prompt = (
        "Using only the passages below, answer the question concisely (2-4 sentences).\n\n"
        f"Passages:\n{joined}\n\n"
        f"Question: {question}\n\nAnswer:"
    )
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[generation error: {e}]"


def run_eval(
    retriever,
    qa_pairs: list[dict],
    top_k: int = 5,
    llm_model: str = DEFAULT_MODEL,
    n_pairs: Optional[int] = None,
    verbose: bool = True,
) -> dict:
    """
    Run 4-metric RAGAS evaluation.

    Args:
        retriever    : ArxivRetriever instance (has .search(query, top_k) → List[RetrievedDoc])
        qa_pairs     : list of dicts with keys  question / answer / section_text / ...
        top_k        : chunks to retrieve per question
        llm_model    : Ollama model name for RAGAS LLM backbone
        n_pairs      : subsample to this many pairs (None = all)
        verbose      : print per-pair progress

    Returns:
        dict {
            context_precision: float,
            context_recall:    float,
            faithfulness:      float,
            n_evaluated:       int,
        }
    """
    if n_pairs:
        qa_pairs = qa_pairs[:n_pairs]

    # Build RAGAS LLM + embeddings wrappers pointing at the Copilot proxy
    langchain_llm = ChatOpenAI(
        model=llm_model,
        openai_api_key="dummy-key",
        openai_api_base=COPILOT_PROXY,
        temperature=0.1,
    )
    ragas_llm = LangchainLLMWrapper(langchain_llm)

    # The Copilot proxy does not expose /v1/embeddings, so AnswerRelevancy is
    # excluded.  ContextPrecision, ContextRecall, Faithfulness are LLM-only.
    metrics = [
        LLMContextPrecisionWithReference(),
        LLMContextRecall(),
        Faithfulness(),
    ]

    samples = []
    n_skip = 0
    for i, pair in enumerate(qa_pairs):
        question  = pair["question"]
        reference = pair["answer"]

        # Retrieve
        try:
            docs = retriever.search(question, top_k=top_k)
            contexts = _extract_contexts(docs)
        except Exception as e:
            if verbose:
                print(f"  [{i+1:3d}] SKIP retrieval error: {e}")
            n_skip += 1
            continue

        if not contexts:
            n_skip += 1
            continue

        # Generate grounded answer
        response = _generate_answer(question, contexts, model=llm_model)

        sample = SingleTurnSample(
            user_input=question,
            reference=reference,
            response=response,
            retrieved_contexts=contexts,
        )
        samples.append(sample)

        if verbose:
            print(f"  [{i+1:3d}] ✓  {len(contexts)} ctx  Q: {question[:60]}...", flush=True)

    if not samples:
        raise RuntimeError("No valid samples for evaluation after retrieval.")

    dataset = EvaluationDataset(samples=samples)

    if verbose:
        print(f"\n  Evaluating {len(samples)} samples with RAGAS (skipped {n_skip})...")

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=ragas_llm,
    )

    scores = result.to_pandas()
    # RAGAS metric .name values: llm_context_precision_with_reference, context_recall, faithfulness
    agg = {
        "context_precision": float(scores["llm_context_precision_with_reference"].mean()),
        "context_recall":    float(scores["context_recall"].mean()),
        "faithfulness":      float(scores["faithfulness"].mean()),
        "n_evaluated":       len(samples),
    }
    return agg


# ── CLI entry-point ────────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run RAGAS eval on QA pairs")
    parser.add_argument("--qa-file",  default="eval/data/qa_pairs.json")
    parser.add_argument("--split",    default="test", choices=["train", "test"])
    parser.add_argument("--n-pairs",  type=int, default=None)
    parser.add_argument("--top-k",    type=int, default=5)
    parser.add_argument("--model",    default=DEFAULT_MODEL)
    args = parser.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from arxiv_retriever import ArxivRetriever
    from retrieval.pgvector_retriever import PGVectorConfig

    with open(args.qa_file, encoding="utf-8") as f:
        qa_data = json.load(f)
    pairs = qa_data[args.split]

    print(f"RAGAS Eval — {args.split} split ({len(pairs)} pairs), top_k={args.top_k}, model={args.model}")
    retriever = ArxivRetriever(PGVectorConfig())
    scores = run_eval(retriever, pairs, top_k=args.top_k,
                      llm_model=args.model, n_pairs=args.n_pairs)

    print("\n── Results ──")
    for k, v in scores.items():
        print(f"  {k:<22}: {v:.4f}" if isinstance(v, float) else f"  {k:<22}: {v}")


if __name__ == "__main__":
    main()
