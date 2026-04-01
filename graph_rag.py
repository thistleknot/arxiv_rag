"""
graph_rag.py — Graph-augmented generation using local Ollama.

Workflow:
  1. User query → GraphRetriever builds 1-2 hop subgraph (GAT-scored triplets)
  2. Triplets injected into LLM system prompt as structured relational context
  3. Ollama streams the answer — informed by graph neighbourhood the LLM
     couldn't see from flat similarity alone

Usage:
  python graph_rag.py "what makes a true friend?"
  python graph_rag.py "what makes a true friend?" --model qwen3:1.7b --top_k 20
"""

import argparse
import sys
import pathlib
import textwrap

import ollama

_ROOT = pathlib.Path(__file__).parent

# ── defaults ─────────────────────────────────────────────────────────────────
DEFAULT_MODEL  = "hf.co/unsloth/Qwen3-1.7B-GGUF:Qwen3-1.7B-Q6_K.gguf"
DEFAULT_TOP_K  = 15
DEFAULT_SEEDS  = 10

SYSTEM_PROMPT = textwrap.dedent("""\
    You are a thoughtful assistant with access to a knowledge graph.
    Below is a set of relational facts extracted from the knowledge graph that
    are relevant to the user's question. Each fact is formatted as:
      subject | predicate | object

    Use these facts to reason about the question. The facts give you
    relational context that may go 1-2 hops beyond the surface query —
    treat them as structured background knowledge, not direct answers.
    Synthesise them into a coherent, natural response.

    Knowledge graph context:
    {context}
""")


def load_retriever():
    """Lazy-import so the script fails fast if graph/ isn't built yet."""
    sys.path.insert(0, str(_ROOT))
    from graph.graph_retriever import GraphRetriever
    return GraphRetriever()


def build_prompt(context: str, query: str) -> list[dict]:
    system = SYSTEM_PROMPT.format(context=context)
    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": query},
    ]


def run(query: str, model: str, top_k: int, n_seeds: int, think: bool):
    print(f"\n{'─'*60}")
    print(f"Query : {query}")
    print(f"Model : {model}")
    print(f"{'─'*60}\n")

    # ── 1. Retrieve subgraph ─────────────────────────────────────────────────
    print("Building subgraph from KG …")
    retriever = load_retriever()
    context   = retriever.retrieve_context(query, top_k=top_k, n_seeds=n_seeds)

    if not context:
        print("⚠  No triplets retrieved — answering without graph context.\n")
    else:
        print(f"Subgraph ({context.count(chr(10))+1} triplets):")
        for line in context.splitlines():
            print(f"  {line}")
        print()

    # ── 2. Build prompt ──────────────────────────────────────────────────────
    messages = build_prompt(context or "(no graph context available)", query)

    # ── 3. Stream generation ─────────────────────────────────────────────────
    print(f"{'─'*60}")
    print("Response:")
    print(f"{'─'*60}")

    options = {}
    if not think:
        # suppress <think> blocks on reasoning models (Qwen3 etc.)
        options["think"] = False

    client = ollama.Client(host="http://localhost:11434")
    stream = client.chat(
        model=model,
        messages=messages,
        stream=True,
        options=options if options else None,
    )

    in_think = False
    for chunk in stream:
        token = chunk["message"]["content"]
        # Strip <think>…</think> blocks from output if present
        if "<think>" in token:
            in_think = True
        if not in_think:
            print(token, end="", flush=True)
        if "</think>" in token:
            in_think = False

    print(f"\n{'─'*60}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Graph-augmented generation")
    ap.add_argument("query", nargs="+", help="Query string (words joined)")
    ap.add_argument("--model",   default=DEFAULT_MODEL,
                    help=f"Ollama model name (default: {DEFAULT_MODEL})")
    ap.add_argument("--top_k",  type=int, default=DEFAULT_TOP_K,
                    help=f"Max triplets to inject (default: {DEFAULT_TOP_K})")
    ap.add_argument("--seeds",  type=int, default=DEFAULT_SEEDS,
                    help=f"Seed entity count (default: {DEFAULT_SEEDS})")
    ap.add_argument("--think",  action="store_true",
                    help="Show <think> tokens from reasoning models")
    args = ap.parse_args()

    run(
        query   = " ".join(args.query),
        model   = args.model,
        top_k   = args.top_k,
        n_seeds = args.seeds,
        think   = args.think,
    )
