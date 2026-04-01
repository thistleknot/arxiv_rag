"""
TextGrad Optimization Harness for PGVector 3-Layer GIST Retrieval Pipeline

=============================================================================
WHAT THIS FILE DOES
=============================================================================

Uses TextGrad to optimize the configuration of the PGVectorRetriever pipeline.
The TextGrad "variable" is a structured JSON config embedded in a prompt that
describes how the 3 layers should be configured (thresholds, flags, weights).

TextGrad sends the config through the GIST/Fact → Premise → Syllogism layers,
evaluates with RAGAS (faithfulness, context_precision, context_recall,
answer_relevancy), and uses the gradient signal to mutate the config text.

Loop:
  1. Parse JSON config from TextGrad variable text.
  2. Build PGVectorConfig from parsed params.
  3. Retrieve contexts for each eval question.
  4. Generate RAG answers with gpt-4.1 @ localhost:3000.
  5. Score with RAGAS (4 metrics).
  6. Compute loss = 1.0 - mean(metrics).
  7. Build textual feedback variable.
  8. Inject as gradient → optimizer.step() rewrites the config variable.
  9. Log everything to MLflow.

=============================================================================
HOW TO RUN
=============================================================================

  python textgrad_pgvector_eval.py --help
  python textgrad_pgvector_eval.py --iterations 10
  python textgrad_pgvector_eval.py --iterations 5 --eval-data my_qa.json
  python textgrad_pgvector_eval.py --iterations 10 --output results.json

=============================================================================
EVAL DATA FORMAT (eval_dataset.json)
=============================================================================

  [
    {
      "question": "What is contrastive learning?",
      "ground_truth": "Contrastive learning is a self-supervised technique...",
      "relevant_doc_ids": [],
      "question_type": "factoid"
    },
    ...
  ]

=============================================================================
"""

import argparse
import json
import os
import re
import sys
import time
import warnings

# Must be set before any pydantic-based package imports to suppress
# textgrad's PromptModelConfig "model_name" protected-namespace warning
warnings.filterwarnings("ignore", message=r"Field .* has conflict with protected namespace", category=UserWarning)

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlflow
import numpy as np

import textgrad as tg
from textgrad import Variable
from textgrad.optimizer import TextualGradientDescent

from openai import OpenAI

# PGVectorRetriever is now in retrieval/ sub-package
from retrieval.pgvector_retriever import PGVectorRetriever, PGVectorConfig  # noqa: E402

# =============================================================================
# Constants
# =============================================================================

COPILOT_PROXY_URL = "http://192.168.3.122:8069/v1"
COPILOT_MODEL = "gpt-4.1"
COPILOT_HEADERS = {
    "Editor-Version": "vscode/1.95.0",
    "Editor-Plugin-Version": "copilot-chat/0.22.0",
    "Copilot-Integration-Id": "vscode-chat",
}
MLFLOW_EXPERIMENT = "textgrad-pgvector-retrieval"
MLFLOW_TRACKING_URI = f"file:///{_HERE / 'mlruns'}".replace("\\", "/")

DEFAULT_EVAL_DATA_PATH = _HERE / "eval_dataset.json"
DEFAULT_OUTPUT_PATH = _HERE / "textgrad_pgvector_results.json"

# TextGrad optimization target metrics (RAGAS)
TARGET_METRICS = ["faithfulness", "context_precision", "context_recall"]

# =============================================================================
# System Prompt — tells TextGrad what this pipeline is and what it can change
# =============================================================================

SYSTEM_PROMPT = """
You are optimizing a 3-layer hybrid retrieval pipeline built on PostgreSQL/pgvector.
Your goal is to find the best configuration to maximize RAGAS evaluation metrics.

=== PIPELINE ARCHITECTURE ===

LAYER 1 — GIST/Fact Hybrid Retrieval
  Two parallel candidate pools are built:
    • BM25 Pool: lexical retrieval from a sparsevec(16000) signed hash index, pool size = top_k²
    • Dense Pool: cosine retrieval from a vector(256) IVFFlat index (Qwen3 model2vec), pool size = top_k²

  Each pool independently runs GIST (Greedy Incremental Selection with Threshold) selection:
    score(d) = gist_lambda × utility(d) − (1 − gist_lambda) × max_sim(d, S)
    where utility = doc-query cosine similarity, S = already-selected set

  The two GIST-selected pools are fused with Reciprocal Rank Fusion (RRF):
    RRF(doc) = Σ 1 / (rrf_k + rank_i)  for each ranking in which the doc appears

  Output: ≤ φ × top_k² candidates (φ ≈ 0.618, the golden ratio / Fibonacci contraction)

LAYER 2 — Premise Expansion (optional)
  A semantic triplet index is built from BIO-tagger extracted SPO triplets.
  Query → BM25 matching triplets → map triplets back to source chunks (expansion)
  Expansion chunks are then merged into Layer 1 output via another RRF pass.
  Controlled by: use_layer2_expansion (bool), layer2_expansion_count (int)

LAYER 3 — Syllogism Reranking (optional)
  Stage A: ColBERT MaxSim — late interaction token-level reranking
    Score = Σᵢ maxⱼ cosine(q_token_i, d_token_j)
    Output: top_k best candidates
    Controlled by: use_colbert (bool)

  Stage B: Cross-Encoder MS-MARCO MiniLM-L-6-v2 — joint query+doc encoding
    Output: final top_k
    Controlled by: use_cross_encoder (bool)

=== TUNABLE PARAMETERS ===

  rrf_k (int, 10–120):
    RRF denominator. Lower = more aggressive rank weighting (top ranks dominate).
    Higher = smoother fusion (all ranks contribute more equally). Default: 60.

  gist_lambda (float, 0.0–1.0):
    GIST diversity–relevance tradeoff. 1.0 = pure relevance (no diversity penalty).
    0.0 = pure diversity (ignore relevance). 0.7 = slight relevance preference.

  use_layer2_expansion (bool):
    Enable triplet-based query expansion. Improves recall for multi-hop queries.
    Adds latency. Disable for precision-focused tasks.

  layer2_expansion_count (int, 0–200):
    Number of expansion chunks to add from the triplet index. Only used when
    use_layer2_expansion=true. Null means auto (= top_k²).

  use_colbert (bool):
    Enable ColBERT token-level late interaction reranking (Layer 3A).
    Improves precision, adds ~200ms latency.

  use_cross_encoder (bool):
    Enable Cross-Encoder joint encoding reranking (Layer 3B).
    Best precision boost, adds ~300ms latency. Runs after ColBERT if both enabled.

  top_k (int, 5–34):
    Final number of results. Controls Fibonacci cascade sizing:
    BM25/dense pools = top_k², GIST output ≤ φ×top_k², final = top_k.

  bm25_min_score (float, 0.0–2.0):
    Minimum BM25 score threshold. Positive scores mean lexical match exists.
    0.0 = no filtering. Raise to improve precision at cost of recall.

  dense_min_similarity (float, 0.0–1.0):
    Minimum cosine similarity threshold for dense results. 0.0 = no filtering.

=== EVALUATION METRICS (4 RAGAS metrics, higher is better) ===

  faithfulness:       Does the answer stick to what the retrieved chunks actually say?
  context_precision:  Are the retrieved chunks actually relevant to the question?
  context_recall:     Do the retrieved chunks cover all information needed for the answer?
  answer_relevancy:   Does the generated answer actually address the question asked?

  Composite score = mean(faithfulness, context_precision, context_recall, answer_relevancy)
  Loss = 1.0 − Composite score

=== YOUR TASK ===

Given the current RAGAS scores and the current configuration, update the JSON
configuration block to improve the composite score. Think carefully about which
parameters to change and why, based on which metrics are underperforming.

RULES:
  1. Always return a valid JSON block between ```json and ``` markers.
  2. Include a "rationale" key explaining your changes (string, ≤200 tokens).
  3. Keep rrf_k between 10 and 120 (integer).
  4. Keep gist_lambda between 0.0 and 1.0 (float, 2 decimal places).
  5. Keep top_k between 5 and 34 (integer).
  6. Keep layer2_expansion_count between 0 and 200 (integer or null).
  7. Booleans must be JSON true/false (lowercase), not Python True/False.
"""

# =============================================================================
# Baseline Configuration Prompt (the TextGrad starting variable)
# =============================================================================

BASELINE_CONFIG_PROMPT = """\
You are configuring the 3-layer GIST retrieval pipeline to maximize RAGAS metrics.

Current Configuration:
```json
{
  "rrf_k": 60,
  "gist_lambda": 0.70,
  "use_layer2_expansion": false,
  "layer2_expansion_count": null,
  "use_colbert": true,
  "use_cross_encoder": true,
  "top_k": 13,
  "bm25_min_score": 0.0,
  "dense_min_similarity": 0.0,
  "rationale": "Default balanced configuration. Both reranking stages enabled for best precision. Layer 2 expansion disabled until triplet index is built. top_k=13 gives 169-document candidate pools (Fibonacci cascade: 13²=169, φ×169≈104 after GIST)."
}
```
"""

# =============================================================================
# Config Parsing
# =============================================================================

_CONFIG_KEYS_DEFAULTS = {
    "rrf_k": 60,
    "gist_lambda": 0.70,
    "use_layer2_expansion": False,
    "layer2_expansion_count": None,
    "use_colbert": True,
    "use_cross_encoder": True,
    "top_k": 13,
    "bm25_min_score": 0.0,
    "dense_min_similarity": 0.0,
}


def parse_pipeline_config(prompt_text: str) -> Dict[str, Any]:
    """
    Extract JSON configuration from a TextGrad variable string.

    Strategy (in priority order):
      1. ```json ... ``` code block
      2. First { ... } block that round-trips as valid JSON
      3. Safe defaults — never fails

    Returns a fully-populated dict with all pipeline params.
    """
    # Strategy 1: fenced code block
    fenced = re.search(r"```json\s*(\{.*?\})\s*```", prompt_text, re.DOTALL)
    if fenced:
        try:
            parsed = json.loads(fenced.group(1))
        except json.JSONDecodeError:
            parsed = {}
    else:
        # Strategy 2: bare JSON object
        bare = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)?\}", prompt_text, re.DOTALL)
        if bare:
            try:
                parsed = json.loads(bare.group(0))
            except json.JSONDecodeError:
                parsed = {}
        else:
            parsed = {}

    # Merge with defaults, clamp numeric ranges
    config = dict(_CONFIG_KEYS_DEFAULTS)
    for key, default in _CONFIG_KEYS_DEFAULTS.items():
        if key in parsed:
            config[key] = parsed[key]

    # Clamp and type-coerce
    config["rrf_k"] = int(max(10, min(120, config["rrf_k"] or 60)))
    config["gist_lambda"] = float(max(0.0, min(1.0, config["gist_lambda"] or 0.7)))
    config["top_k"] = int(max(5, min(34, config["top_k"] or 13)))
    config["use_layer2_expansion"] = bool(config["use_layer2_expansion"])
    config["use_colbert"] = bool(config["use_colbert"])
    config["use_cross_encoder"] = bool(config["use_cross_encoder"])
    config["bm25_min_score"] = float(max(0.0, config["bm25_min_score"] or 0.0))
    config["dense_min_similarity"] = float(max(0.0, min(1.0, config["dense_min_similarity"] or 0.0)))
    if config["layer2_expansion_count"] is not None:
        config["layer2_expansion_count"] = int(
            max(0, min(200, config["layer2_expansion_count"]))
        )

    return config


def build_retriever_config(params: Dict[str, Any]) -> PGVectorConfig:
    """Map parsed params dict to PGVectorConfig dataclass."""
    kwargs = dict(
        # DB — hardcoded, not tuned
        db_host="localhost",
        db_port=5432,
        db_name="langchain",
        db_user="langchain",
        db_password="langchain",
        table_name="arxiv_chunks",
        # Pipeline params — tuned by TextGrad
        rrf_k=params["rrf_k"],
        gist_lambda=params["gist_lambda"],
        use_layer2_expansion=params["use_layer2_expansion"],
        use_colbert=params["use_colbert"],
        use_cross_encoder=params["use_cross_encoder"],
        bm25_min_score=params["bm25_min_score"],
        dense_min_similarity=params["dense_min_similarity"],
    )
    if params["layer2_expansion_count"] is not None:
        kwargs["layer2_expansion_count"] = params["layer2_expansion_count"]
    return PGVectorConfig(**kwargs)


# =============================================================================
# LLM Client
# =============================================================================

def build_openai_client() -> OpenAI:
    """Return an OpenAI client pointed at the copilot proxy."""
    return OpenAI(
        base_url=COPILOT_PROXY_URL,
        api_key="copilot",
        default_headers=COPILOT_HEADERS,
    )


# =============================================================================
# Answer Generation
# =============================================================================

def generate_answer(
    question: str,
    contexts: List[str],
    client: OpenAI,
    max_tokens: int = 512,
) -> str:
    """
    Generate a RAG answer using the retrieved contexts.

    Args:
        question:  The user question.
        contexts:  List of retrieved chunk strings.
        client:    OpenAI client (copilot proxy).
        max_tokens: Maximum tokens for generated answer.

    Returns:
        Generated answer string.
    """
    context_block = "\n\n---\n\n".join(
        f"[Chunk {i+1}]:\n{c}" for i, c in enumerate(contexts[:10])  # cap at 10
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You are a precise scientific assistant. "
                "Answer the question using ONLY the provided context chunks. "
                "If the context does not contain enough information, say so honestly. "
                "Be concise and accurate."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Context:\n{context_block}\n\n"
                f"Question: {question}\n\n"
                "Answer:"
            ),
        },
    ]
    try:
        resp = client.chat.completions.create(
            model=COPILOT_MODEL,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.0,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"  [warn] answer generation failed: {e}")
        return f"[generation error: {e}]"


# =============================================================================
# RAGAS Evaluation
# =============================================================================

def run_ragas_eval(
    samples: List[Dict[str, Any]],
    llm_client: OpenAI,
) -> Dict[str, float]:
    """
    Evaluate a list of RAG samples using RAGAS metrics.

    Args:
        samples: List of dicts with keys question, answer, contexts, ground_truth.
        llm_client: OpenAI client (used for RAGAS LLM calls).

    Returns:
        Dict with keys: faithfulness, context_precision, context_recall,
        answer_relevancy, composite.
    """
    try:
        from ragas import evaluate
        from ragas.metrics import (
            Faithfulness,
            LLMContextPrecisionWithReference,
            LLMContextRecall,
        )
        from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
        from ragas.llms import LangchainLLMWrapper
        from langchain_openai import ChatOpenAI as LCChatOpenAI
    except ImportError as e:
        print(f"  [error] RAGAS/langchain import failed: {e}")
        print("  Install: pip install ragas langchain-openai")
        return {m: 0.0 for m in TARGET_METRICS + ["composite"]}

    # Wrap copilot proxy as a LangChain LLM for RAGAS
    lc_llm = LCChatOpenAI(
        model=COPILOT_MODEL,
        openai_api_base=COPILOT_PROXY_URL,
        openai_api_key="copilot",
        temperature=0.0,
        default_headers=COPILOT_HEADERS,
    )
    ragas_llm = LangchainLLMWrapper(lc_llm)

    # Build RAGAS metrics using the proxy LLM
    metrics = [
        Faithfulness(llm=ragas_llm),
        LLMContextPrecisionWithReference(llm=ragas_llm),
        LLMContextRecall(llm=ragas_llm),
    ]
    # metric name mapping: ragas uses its own attribute names
    metric_name_map = {
        "faithfulness": "faithfulness",
        "llm_context_precision_with_reference": "context_precision",
        "context_recall": "context_recall",
    }

    # Build RAGAS dataset
    ragas_samples = []
    for s in samples:
        ragas_samples.append(
            SingleTurnSample(
                user_input=s["question"],
                response=s["answer"],
                retrieved_contexts=s["contexts"],
                reference=s["ground_truth"],
            )
        )
    dataset = EvaluationDataset(samples=ragas_samples)

    try:
        result = evaluate(dataset=dataset, metrics=metrics)
        # result is a dict-like EvaluationResult; convert to plain Python dict
        scores_raw = dict(result)
    except Exception as e:
        print(f"  [error] RAGAS evaluate() failed: {e}")
        return {m: 0.0 for m in TARGET_METRICS + ["composite"]}

    # Normalise key names to TARGET_METRICS
    scores: Dict[str, float] = {}
    for raw_key, canonical in metric_name_map.items():
        if raw_key in scores_raw:
            val = scores_raw[raw_key]
            scores[canonical] = float(val) if val is not None else 0.0
        elif canonical in scores_raw:
            val = scores_raw[canonical]
            scores[canonical] = float(val) if val is not None else 0.0
        else:
            scores[canonical] = 0.0

    scores["composite"] = float(
        np.mean([scores.get(m, 0.0) for m in TARGET_METRICS])
    )
    return scores


# =============================================================================
# Full Retrieval + Eval Round
# =============================================================================

def evaluate_retrieval_config(
    config_prompt: str,
    eval_dataset: List[Dict[str, Any]],
    llm_client: OpenAI,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    End-to-end evaluation of one pipeline configuration.

    1. Parse JSON config from the TextGrad variable text.
    2. Build PGVectorConfig and open a retriever.
    3. For each question: retrieve → generate answer.
    4. Score all samples with RAGAS.
    5. Return scores dict.
    """
    params = parse_pipeline_config(config_prompt)
    if verbose:
        print(f"  [eval] config: rrf_k={params['rrf_k']}, gist_lambda={params['gist_lambda']}, "
              f"layer2={params['use_layer2_expansion']}, colbert={params['use_colbert']}, "
              f"ce={params['use_cross_encoder']}, top_k={params['top_k']}")

    pg_config = build_retriever_config(params)
    top_k = params["top_k"]

    samples: List[Dict[str, Any]] = []
    retriever = PGVectorRetriever(pg_config)

    try:
        for item in eval_dataset:
            question = item["question"]
            ground_truth = item.get("ground_truth", "")
            try:
                results = retriever.search_chunks(question, top_k=top_k)
                contexts = [r.content for r in results]
            except Exception as e:
                print(f"  [warn] retrieval failed for '{question[:60]}': {e}")
                contexts = []

            answer = generate_answer(question, contexts, llm_client)
            samples.append({
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": ground_truth,
            })
            if verbose:
                print(f"    >> '{question[:60]}' -> {len(contexts)} chunks retrieved")
    finally:
        try:
            retriever.close()
        except Exception:
            pass

    if not samples:
        print("  [error] no samples produced, returning zero scores")
        return {m: 0.0 for m in TARGET_METRICS + ["composite"]}

    if verbose:
        print(f"  [eval] running RAGAS on {len(samples)} samples …")
    scores = run_ragas_eval(samples, llm_client)
    if verbose:
        print(f"  [eval] scores: {scores}")
    return scores


# =============================================================================
# TextGrad Engine — copilot proxy patched
# =============================================================================

def build_textgrad_engine() -> Any:
    """
    Build a TextGrad engine backed by the copilot proxy (localhost:3000).

    Patches out the 'stop' parameter which the proxy rejects.
    Follows the same pattern used in textgrad_L1.py.
    """
    import platformdirs
    from textgrad.engine.openai import ChatOpenAI as TGChatOpenAI
    from textgrad.engine.base import CachedEngine

    os.environ.setdefault("OPENAI_API_KEY", "copilot")

    engine = TGChatOpenAI.__new__(TGChatOpenAI)

    # Minimal CachedEngine init
    cache_dir = platformdirs.user_cache_dir("textgrad")
    _safe_model = COPILOT_MODEL.replace(":", "_").replace("/", "_")
    cache_path = os.path.join(cache_dir, f"cache_pgvector_{_safe_model}.db")
    CachedEngine.__init__(engine, cache_path=cache_path)

    engine.system_prompt = TGChatOpenAI.DEFAULT_SYSTEM_PROMPT
    engine.base_url = COPILOT_PROXY_URL
    engine.model_string = COPILOT_MODEL
    engine.is_multimodal = False

    from openai import OpenAI as _OpenAI
    engine.client = _OpenAI(base_url=COPILOT_PROXY_URL, api_key="copilot")

    # Strip 'stop' param — copilot proxy rejects it
    _orig_create = engine.client.chat.completions.create

    def _patched_create(**kwargs):
        kwargs.pop("stop", None)
        return _orig_create(**kwargs)

    engine.client.chat.completions.create = _patched_create
    tg.set_backward_engine(engine)

    return engine


# =============================================================================
# Feedback Builder — turns RAGAS scores into a TextGrad gradient signal
# =============================================================================

def build_feedback_text(scores: Dict[str, float], config_prompt: str) -> str:
    """
    Construct human-readable feedback for the TextGrad gradient variable.

    The feedback describes which metrics are underperforming and which
    parameters are worth adjusting, referencing the current config.
    """
    composite = scores.get("composite", 0.0)
    loss = 1.0 - composite

    metric_lines = "\n".join(
        f"  - {m}: {scores.get(m, 0.0):.4f} (gap to 1.0: {1.0 - scores.get(m, 0.0):.4f})"
        for m in TARGET_METRICS
    )

    # Identify worst metrics for focused feedback
    worst = sorted(TARGET_METRICS, key=lambda m: scores.get(m, 0.0))[:2]
    worst_str = " and ".join(worst)

    # Parse current params for context-aware feedback
    params = parse_pipeline_config(config_prompt)

    param_summary = (
        f"rrf_k={params['rrf_k']}, gist_lambda={params['gist_lambda']:.2f}, "
        f"layer2={'ON' if params['use_layer2_expansion'] else 'OFF'}, "
        f"colbert={'ON' if params['use_colbert'] else 'OFF'}, "
        f"cross_encoder={'ON' if params['use_cross_encoder'] else 'OFF'}, "
        f"top_k={params['top_k']}"
    )

    return (
        f"Evaluation result:\n"
        f"  Composite score: {composite:.4f}/1.0 (loss={loss:.4f})\n"
        f"\nMetric breakdown:\n{metric_lines}\n"
        f"\nCurrent pipeline settings: {param_summary}\n"
        f"\nThe weakest metrics are: {worst_str}.\n"
        f"To close the gap, consider adjusting these parameters:\n"
        f"  - If faithfulness is low: try increasing top_k or lowering bm25_min_score\n"
        f"    so more relevant chunks are included.\n"
        f"  - If context_precision is low: try raising bm25_min_score or lowering rrf_k\n"
        f"    (≤40) to make top-rank results dominate the fusion.\n"
        f"  - If context_recall is low: enable use_layer2_expansion=true to bring in\n"
        f"    triplet-expanded chunks, or increase top_k.\n"
        f"  - If answer_relevancy is low: try use_colbert=true and/or\n"
        f"    use_cross_encoder=true for tighter reranking.\n"
        f"  - gist_lambda: increase toward 1.0 for more relevance, decrease for more diversity.\n"
        f"\nUpdate the JSON configuration block to improve, and provide an updated rationale."
    )


# =============================================================================
# Default Eval Dataset — scaffold if no file provided
# =============================================================================

DEFAULT_EVAL_QUESTIONS = [
    {
        "question": "What is contrastive learning and how is it applied in self-supervised representation learning?",
        "ground_truth": (
            "Contrastive learning is a self-supervised technique that learns representations "
            "by contrasting similar (positive) pairs against dissimilar (negative) pairs. "
            "It applies augmentation-based views of the same sample as positives and "
            "maximizes agreement between them while pushing apart negatives, enabling "
            "representation learning without labeled data."
        ),
        "question_type": "factoid",
    },
    {
        "question": "How does the transformer attention mechanism work and what are its computational complexity properties?",
        "ground_truth": (
            "The transformer attention mechanism computes scaled dot-product attention: "
            "Attention(Q,K,V) = softmax(QKᵀ/√d)V. Self-attention has O(n²d) time and "
            "space complexity in the sequence length n, which becomes a bottleneck for "
            "long sequences. Multi-head attention runs h parallel attention heads and "
            "concatenates their outputs."
        ),
        "question_type": "factoid",
    },
    {
        "question": "What is the difference between BERT pretraining and GPT pretraining objectives?",
        "ground_truth": (
            "BERT uses masked language modeling (MLM) — randomly masking 15% of tokens and "
            "predicting them — plus next sentence prediction (NSP). This enables bidirectional "
            "context. GPT uses causal (autoregressive) language modeling, predicting each "
            "token from all preceding tokens, which is unidirectional but naturally suited "
            "for generation tasks."
        ),
        "question_type": "comparative",
    },
    {
        "question": "How do graph neural networks aggregate node features across neighborhoods?",
        "ground_truth": (
            "Graph neural networks (GNNs) aggregate node features using message passing: "
            "each node collects features from its neighbors (message), combines them with "
            "its own features (aggregation), and applies a neural update. Common aggregation "
            "functions include sum, mean, or max pooling. GCN uses normalized mean aggregation; "
            "GraphSAGE allows different sample sizes; GAT uses attention weights."
        ),
        "question_type": "mechanism",
    },
    {
        "question": "What is zero-shot chain-of-thought prompting and how does it improve reasoning in LLMs?",
        "ground_truth": (
            "Zero-shot chain-of-thought prompting adds 'Let's think step by step' to a prompt, "
            "eliciting intermediate reasoning steps from LLMs without any examples. This "
            "improves accuracy on arithmetic, commonsense, and symbolic reasoning tasks by "
            "allowing the model to decompose the problem before giving a final answer."
        ),
        "question_type": "factoid",
    },
]


def load_eval_dataset(path: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Load QA eval dataset from JSON, or return scaffold defaults.

    Expected format:
      [{"question": ..., "ground_truth": ..., "question_type": ...}, ...]
    """
    if path and Path(path).exists():
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        print(f"[eval] Loaded {len(data)} questions from {path}")
        return data

    default_path = DEFAULT_EVAL_DATA_PATH
    if default_path.exists():
        with open(default_path, encoding="utf-8") as f:
            data = json.load(f)
        print(f"[eval] Loaded {len(data)} questions from {default_path}")
        return data

    print(
        "[eval] No eval dataset found — using 5 scaffold questions.\n"
        f"       Create {default_path} with your QA pairs for real evaluation."
    )
    return DEFAULT_EVAL_QUESTIONS


# =============================================================================
# Main Optimization Loop
# =============================================================================

def optimize_retrieval_prompt(
    baseline_prompt: str,
    eval_data: List[Dict[str, Any]],
    n_iter: int = 10,
    mlflow_run_name: Optional[str] = None,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    TextGrad optimization loop for the GIST pipeline configuration.

    Returns a results dict with baseline/optimized scores, full history, and
    the best-scoring prompt text.

    Args:
        baseline_prompt:  Starting TextGrad variable text (JSON config + rationale).
        eval_data:        List of {question, ground_truth} dicts.
        n_iter:           Maximum optimization iterations.
        mlflow_run_name:  Optional MLflow run name override.
        output_path:      Path to save JSON results.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = mlflow_run_name or f"textgrad-pgvector-{timestamp}"
    output_path = output_path or str(DEFAULT_OUTPUT_PATH)

    llm_client = build_openai_client()
    engine = build_textgrad_engine()  # noqa: F841 — sets backward engine globally

    # TextGrad variable: the thing being optimized
    prompt_variable = Variable(
        baseline_prompt,
        requires_grad=True,
        role_description=(
            "Retrieval pipeline configuration for the 3-layer GIST/Fact/Premise/Syllogism "
            "hybrid search system. Contains a JSON config block and rationale."
        ),
    )

    optimizer = TextualGradientDescent(parameters=[prompt_variable])

    # ── MLflow setup ────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    score_history: List[float] = []
    scores_history: List[Dict[str, float]] = []
    prompt_history: List[str] = []
    config_history: List[Dict[str, Any]] = []

    with mlflow.start_run(run_name=run_name) as mlflow_run:
        mlflow.log_param("model", COPILOT_MODEL)
        mlflow.log_param("proxy_url", COPILOT_PROXY_URL)
        mlflow.log_param("n_iter", n_iter)
        mlflow.log_param("n_eval_questions", len(eval_data))
        mlflow.log_param("target_metrics", ",".join(TARGET_METRICS))
        mlflow.log_text(SYSTEM_PROMPT, "system_prompt.txt")

        print(f"\n{'='*70}")
        print(f"TextGrad GIST Pipeline Optimization — Run: {run_name}")
        print(f"Eval questions: {len(eval_data)} | Max iterations: {n_iter}")
        print(f"MLflow run: {mlflow_run.info.run_id}")
        print(f"{'='*70}\n")

        # ── Baseline evaluation ──────────────────────────────────────────────
        print("=== BASELINE EVALUATION ===")
        baseline_scores = evaluate_retrieval_config(
            prompt_variable.value, eval_data, llm_client
        )
        baseline_composite = baseline_scores.get("composite", 0.0)
        print(f"Baseline composite: {baseline_composite:.4f}")
        mlflow.log_metrics(
            {f"baseline_{k}": v for k, v in baseline_scores.items()},
            step=0,
        )
        mlflow.log_text(prompt_variable.value, "prompt_baseline.txt")

        score_history.append(baseline_composite)
        scores_history.append(baseline_scores)
        prompt_history.append(prompt_variable.value)
        config_history.append(parse_pipeline_config(prompt_variable.value))

        # ── Optimization iterations ──────────────────────────────────────────
        for iteration in range(n_iter):
            print(f"\n--- Iteration {iteration + 1}/{n_iter} ---")

            # Build loss as a textgrad Variable (feedback text)
            feedback_text = build_feedback_text(scores_history[-1], prompt_variable.value)
            loss = Variable(
                feedback_text,
                role_description="RAGAS evaluation feedback and parameter improvement guidance",
            )

            # Gradient injection: feedback Variable becomes the gradient of the prompt
            prompt_variable.reset_gradients()
            prompt_variable.gradients.add(loss)
            prompt_variable.gradients_context[loss] = None

            # Optimizer rewrites the prompt variable based on the gradient text
            optimizer.step()

            print(f"Updated config prompt (first 400 chars):\n  {prompt_variable.value[:400]}…")

            # Evaluate the updated config
            current_scores = evaluate_retrieval_config(
                prompt_variable.value, eval_data, llm_client
            )
            current_composite = current_scores.get("composite", 0.0)
            loss_val = 1.0 - current_composite

            print(
                f"Iteration {iteration + 1}: composite={current_composite:.4f}  "
                f"loss={loss_val:.4f}  delta={current_composite - score_history[-1]:+.4f}"
            )
            for m in TARGET_METRICS:
                print(f"  {m}: {current_scores.get(m, 0.0):.4f}")

            # Log to MLflow
            step = iteration + 1
            mlflow.log_metrics(
                {**current_scores, "loss": loss_val, "composite": current_composite},
                step=step,
            )
            mlflow.log_text(
                prompt_variable.value, f"prompt_iter_{iteration + 1:03d}.txt"
            )
            current_params = parse_pipeline_config(prompt_variable.value)
            mlflow.log_params(
                {f"iter{step}_rrf_k": current_params["rrf_k"],
                 f"iter{step}_gist_lambda": current_params["gist_lambda"],
                 f"iter{step}_layer2": current_params["use_layer2_expansion"],
                 f"iter{step}_colbert": current_params["use_colbert"],
                 f"iter{step}_cross_encoder": current_params["use_cross_encoder"],
                 f"iter{step}_top_k": current_params["top_k"]}
            )

            score_history.append(current_composite)
            scores_history.append(current_scores)
            prompt_history.append(prompt_variable.value)
            config_history.append(current_params)

            # ── Early stopping ───────────────────────────────────────────────
            if len(score_history) >= 4:
                recent_deltas = [
                    score_history[i] - score_history[i - 1]
                    for i in range(-3, 0)
                ]
                all_regressing = all(d < 0 for d in recent_deltas)
                all_flat = all(abs(d) < 0.001 for d in recent_deltas)
                if all_regressing or all_flat:
                    print(
                        f"[EARLY STOP] Last 3 deltas: "
                        f"{[f'{d:+.4f}' for d in recent_deltas]}. Stopping."
                    )
                    break

        # ── Find best result ─────────────────────────────────────────────────
        best_idx = int(np.argmax(score_history))
        best_prompt = prompt_history[best_idx]
        best_scores = scores_history[best_idx]
        best_composite = score_history[best_idx]

        print(f"\n{'='*70}")
        print(f"Optimization complete.")
        print(f"Best composite: {best_composite:.4f} (iteration {best_idx})")
        print(f"Baseline:       {baseline_composite:.4f}")
        print(f"Improvement:    {best_composite - baseline_composite:+.4f}")
        print(f"{'='*70}")

        mlflow.log_metrics({
            "best_composite": best_composite,
            "improvement": best_composite - baseline_composite,
            **{f"best_{k}": v for k, v in best_scores.items()},
        })
        mlflow.log_text(best_prompt, "prompt_best.txt")

    # ── Save results ─────────────────────────────────────────────────────────
    results = {
        "run_name": run_name,
        "timestamp": timestamp,
        "model": COPILOT_MODEL,
        "baseline_composite": baseline_composite,
        "baseline_scores": baseline_scores,
        "best_composite": best_composite,
        "best_scores": best_scores,
        "improvement": best_composite - baseline_composite,
        "best_iteration": best_idx,
        "score_history": score_history,
        "scores_history": scores_history,
        "prompt_history": prompt_history,
        "config_history": config_history,
        "best_prompt": best_prompt,
        "baseline_prompt": baseline_prompt,
        "n_iterations_run": len(score_history) - 1,
        "target_metrics": TARGET_METRICS,
        "n_eval_questions": len(eval_data),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[done] Results saved to {output_path}")
    print(f"[done] MLflow experiment: {MLFLOW_EXPERIMENT}")
    print(f"[done] Tracking URI: {MLFLOW_TRACKING_URI}")
    print(f"       → Run: mlflow ui --backend-store-uri {MLFLOW_TRACKING_URI}")

    return results


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="TextGrad optimization for PGVector 3-layer GIST retrieval pipeline"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Maximum number of TextGrad optimization iterations (default: 10)",
    )
    parser.add_argument(
        "--eval-data",
        type=str,
        default=None,
        help=f"Path to QA eval JSON (default: {DEFAULT_EVAL_DATA_PATH})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_PATH),
        help=f"Path to save JSON results (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="MLflow run name override",
    )
    parser.add_argument(
        "--baseline-prompt",
        type=str,
        default=None,
        help="Path to a .txt file containing the baseline prompt (default: built-in)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and print the baseline config, then exit (no retrieval or LLM calls)",
    )
    args = parser.parse_args()

    # Load baseline prompt
    if args.baseline_prompt and Path(args.baseline_prompt).exists():
        with open(args.baseline_prompt, encoding="utf-8") as f:
            baseline_prompt = f.read()
        print(f"[init] Loaded baseline prompt from {args.baseline_prompt}")
    else:
        baseline_prompt = BASELINE_CONFIG_PROMPT

    # Parse and display baseline config
    params = parse_pipeline_config(baseline_prompt)
    print("\n[init] Baseline configuration:")
    for k, v in params.items():
        print(f"  {k}: {v}")

    if args.dry_run:
        print("\n[dry-run] Exiting. No retrieval or LLM calls made.")
        return

    # Load eval data
    eval_data = load_eval_dataset(args.eval_data)
    if not eval_data:
        print("[error] Eval dataset is empty. Provide questions via --eval-data.")
        sys.exit(1)

    # Run optimization
    optimize_retrieval_prompt(
        baseline_prompt=baseline_prompt,
        eval_data=eval_data,
        n_iter=args.iterations,
        mlflow_run_name=args.run_name,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
