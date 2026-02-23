"""
TextGrad + MLflow Optimization of 3-Layer RAGAS Retrieval Pipeline (v2)

Core Thesis:
    Supply TextGrad with the complete 3-layer retrieval pipeline walkthrough
    so it starts with full toolkit knowledge.  Generate fresh QA pairs by
    running topic queries through the live retriever and calling gpt-4.1 to
    derive question/answer pairs from the retrieved sections.  Evaluate
    4-metric RAGAS (context_precision, context_recall, faithfulness,
    answer_relevancy) and use the mean as the scalar loss for TextGrad\'s
    TextualGradientDescent to iteratively update the == CURRENT CONFIG ==.

TextGrad wiring (fixed vs v1):
    loss_var created with predecessors=[system_prompt] so the backward pass
    correctly flows gradient text from the evaluation feedback back to
    system_prompt.  optimizer.step() then rewrites system_prompt.value with
    the LLM-suggested config block improvements.

MLflow logging per iteration:
    - Metrics : context_precision, context_recall, faithfulness,
                answer_relevancy, mean_ragas  (step=iteration)
    - Params  : logged once at iteration 0 only (no conflict on later iters)
    - Artifacts: system_prompt_v{iter}.txt, retriever code snapshots
"""

import json
import math
import os
import re
import sys
import copy
import shutil
import socket
import tempfile
import time
import random
from typing import Optional

# Force UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import mlflow
import textgrad as tg
from textgrad.engine import LiteLLMEngine
import openai

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Point LiteLLM + RAGAS at the copilot proxy
os.environ["OPENAI_API_BASE"] = os.environ.get("LLM_PROXY_URL", "http://127.0.0.1:8069/v1")
os.environ["OPENAI_API_KEY"]  = os.environ.get("OPENAI_API_KEY", "copilot")

from arxiv_retriever import ArxivRetriever
from retrieval.pgvector_retriever import PGVectorConfig
from eval.ragas_eval import run_eval  # 3-metric baseline

# ── Constants ─────────────────────────────────────────────────────────────────
COPILOT_PROXY = os.environ.get("LLM_PROXY_URL", "http://127.0.0.1:8069/v1")
_BASE_MODEL   = os.environ.get("LLM_MODEL",     "gpt-4.1")
# LiteLLMEngine needs "openai/" prefix to route via OpenAI-compat endpoint;
# gpt-* and already-prefixed names are passed through unchanged.
LITELLM_MODEL = _BASE_MODEL if _BASE_MODEL.startswith(("gpt-", "openai/")) else f"openai/{_BASE_MODEL}"
OPENAI_MODEL  = _BASE_MODEL   # bare name for direct openai.OpenAI() client calls
RASGAS_MODEL  = os.environ.get("RAGAS_MODEL", "gpt-4.1")  # RAGAS eval model
MLFLOW_EXP = "arxiv-3layer-retrieval-optimization"
RETRIEVER_FILES = [
    os.path.join(ROOT, "retrieval", "pgvector_retriever.py"),
    os.path.join(ROOT, "retrieval", "base_gist_retriever.py"),
    os.path.join(ROOT, "retrieval", "gist_retriever.py"),
]

DEFAULT_CONFIG = {
    "top_k": 5,
    "rrf_k": 60,
    "gist_lambda": 0.7,
    "bm25_min_score": 0.0,
    "dense_min_similarity": 0.0,
    "colbert_min_score": 0.0,
    "cross_encoder_min_score": 0.0,
}

# Diverse academic topic queries
TOPIC_QUERIES = [
    "attention mechanism transformer self-attention scaled dot product",
    "diffusion model denoising score matching image generation",
    "graph neural network message passing node classification",
    "reinforcement learning from human feedback reward model RLHF",
    "retrieval augmented generation dense passage retrieval RAG",
    "mixture of experts sparse gating conditional computation scaling",
    "contrastive learning self-supervised SimCLR representation learning",
    "causal language model autoregressive pretraining next-token prediction",
    "vision transformer patch embedding image classification ViT",
    "chain of thought reasoning few-shot prompting emergent abilities",
]

# ── Helpers ──────────────────────────────────────────────────────────────────

def _wait_for_postgres(
    host: str = "localhost",
    port: int = 5432,
    timeout: int = 30,
) -> None:
    """
    Block until Postgres accepts TCP connections on host:port.
    Retries every 1 s up to `timeout` seconds, then raises RuntimeError.
    Called once at startup so the script never fails due to slow container init.
    """
    deadline = time.monotonic() + timeout
    attempt = 0
    while True:
        try:
            with socket.create_connection((host, port), timeout=2):
                if attempt > 0:
                    print(f"  [pg-wait] Postgres ready after {attempt}s.", flush=True)
                return
        except (ConnectionRefusedError, OSError):
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"Postgres at {host}:{port} not ready after {timeout}s — "
                    "is the docker container running?"
                )
            attempt += 1
            if attempt == 1:
                print(f"  [pg-wait] Waiting for Postgres on {host}:{port} ...", flush=True)
            time.sleep(1)


# ── Pipeline code walkthrough ─────────────────────────────────────────────────
PIPELINE_CODE_WALKTHROUGH = """\
== PIPELINE ARCHITECTURE AND CODE WALKTHROUGH ==

This is the full 3-layer academic paper retrieval pipeline (ArxivRetriever).
Each layer is described with its code logic, data flow, and key parameters
so you can reason about which config knobs affect which retrieval step.

------------------------------------------------------------------------------
ENTRY POINT: BaseGISTRetriever.search(query, top_k)
------------------------------------------------------------------------------

    def search(self, query, top_k=13):
        retrieval_limit = top_k ** 2          # k^2 candidates fetched per arm
        hybrid_seeds = prev_fibonacci(retrieval_limit)  # e.g. top_k=13 -> 169 -> 144 seeds

        # Layer 1 ------------------------------------------------------------------
        bm25_pool  = self._retrieve_bm25(query, retrieval_limit)   # L1-BM25 arm
        dense_pool = self._retrieve_dense(query, retrieval_limit)  # L1-Dense arm

        bm25_selected  = self._gist_select_pool(bm25_pool,  query, hybrid_seeds, \'bm25\')
        dense_selected = self._gist_select_pool(dense_pool, query, hybrid_seeds, \'dense\')

        hybrid_pool = self._rrf_fusion(bm25_selected, dense_selected)
        hybrid_seeds_pool = hybrid_pool[:hybrid_seeds]  # exactly hybrid_seeds chunks

        # Layer 2 ------------------------------------------------------------------
        seed_scores = [doc.rrf_score for doc in hybrid_seeds_pool]
        l2_bm25  = self._expand_layer2_bm25 (hybrid_seeds_pool, seed_scores, hybrid_seeds)
        l2_dense = self._expand_layer2_dense(hybrid_seeds_pool, seed_scores, hybrid_seeds)
        graph_expanded = self._rrf_fusion(l2_bm25, l2_dense)
        documents = self._reconstruct_documents_from_chunks(graph_expanded, hybrid_seeds)

        # Layer 3 ------------------------------------------------------------------
        scored_documents = self._score_documents(query, documents, len(documents))
        results = self._select_final_documents(scored_documents, top_k)
        return results

------------------------------------------------------------------------------
LAYER 1 -- Sparse + Dense Seeding (layer1_bm25_sparse, layer1_embeddings_128d)
------------------------------------------------------------------------------

BM25 arm: PGVectorRetriever._retrieve_bm25(query, limit)
  - Tokenise: query.lower().split() -> map to L1 vocab column indices
  - SQL:  SELECT l.chunk_id, SUM((l.sparse_vector->>k)::float) AS score
           FROM layer1_bm25_sparse l, unnest(term_ids) AS k
           WHERE l.sparse_vector ? k
           GROUP BY l.chunk_id
           ORDER BY score DESC LIMIT limit
  - Returns list[RetrievedDoc] with .bm25_score and .bm25_rank
  - bm25_min_score (config): chunks with bm25_score < threshold filtered in L3.

Dense arm: PGVectorRetriever._retrieve_dense(query, limit)
  - Encode: Qwen3-256d model2vec -> PCA 256d->128d -> L2-normalise
  - SQL:  SELECT l.chunk_id,
                 1.0 - (l.embedding <=> query::vector) AS score
           FROM layer1_embeddings_128d l
           ORDER BY l.embedding <=> query::vector   -- HNSW ANN
           LIMIT limit
  - Returns list[RetrievedDoc] with .dense_score and .dense_rank
  - dense_min_similarity (config): chunks with dense_score < threshold filtered in L3.

GIST selection per arm (gist_select formula):
  score(doc_i) = lambda * normalize(corr(doc_i, query))
               - (1 - lambda) * mean_corr(doc_i, selected_set_S)

  Greedy forward selection:
    First pick = highest utility (most relevant to query).
    Each next pick trades off relevance vs mean collinearity to selected docs.
    mean_corr(d, S) = sum_sim_to_selected[d] / |S|   -- updated incrementally.

  gist_lambda (config key):
    1.0 = pure relevance (identical to top-k)
    0.0 = pure diversity (maximum marginal relevance)
    0.7 = default (slight relevance preference)

RRF fusion: _rrf_fusion(pool_a, pool_b, k=rrf_k)
  score(d) = sum_over_arms( 1 / (rrf_k + rank(d, arm)) )
  rrf_k (config key): higher = flatter distribution; lower = top-rank advantage.
  Results deduplicated, sorted by combined RRF score descending.

------------------------------------------------------------------------------
LAYER 2 -- ECDF-weighted Dual Arm Expansion
           (layer2_triplet_bm25, layer2_embeddings_256d)
------------------------------------------------------------------------------

BM25 expansion arm: _expand_layer2_bm25(seed_docs, seed_scores, top_k)
  1. ECDF weights: midpoint_ecdf(seed_rrf_scores) -> weight[i] per seed
  2. Weighted TF profile: weighted average of seed TF vectors using ecdf_w
  3. Build IDF-weighted sparsevec from bm25_l2_stats / bm25_l2_term_df tables
  4. SQL:  -(layer2_triplet_bm25 <#> q_sv) ORDER BY neg-inner-product;
           exclude seeds; LIMIT top_k new candidates
  5. Combined pool = seed_docs + new_candidates
  6. TF coverage matrix: term-frequency L2-norms over query vocab
     coverage[i,j] = cos(tf_vec_i, tf_vec_j)
  7. Utility y_scores = tf_norm @ q_norm  (TF alignment with query terms)
  8. GIST-rank(coverage, y_scores) -> diverse & relevant selection

Dense expansion arm: _expand_layer2_dense(seed_docs, seed_scores, top_k)
  1. ECDF weights -> weighted centroid of seed Qwen3-256d embeddings (L2-norm)
     centroid = sum(ecdf_w[i] * seed_emb[i]) / ||sum||
  2. ANN on layer2_embeddings_256d with centroid; exclude seeds; LIMIT new
  3. Combined pool = seed_docs + new_candidates
  4. Cosine coverage matrix: emb_norm @ emb_norm.T
  5. Utility y_scores = emb_norm @ centroid_norm
  6. GIST-rank(coverage, y_scores) -> diverse & relevant

Layer 2 RRF + section reconstruction:
  graph_expanded = _rrf_fusion(l2_bm25_result, l2_dense_result)
  Walk RRF-sorted chunks; collect chunks per unique section_idx; stop when
  hybrid_seeds unique sections accumulated -> documents list for L3.

------------------------------------------------------------------------------
LAYER 3 -- ColBERT + Cross-Encoder Re-ranking -> Final top_k
------------------------------------------------------------------------------

ColBERT scoring (late interaction, bert-base-uncased -> 128d):
  Score(Q,D) = sum_i  max_j  sim(q_token_i, d_token_j)
  Each query token finds its best-matching doc token; sum of maxima.
  colbert_min_score (config): below threshold ColBERT score excluded.

Cross-encoder scoring (MS-MARCO-MiniLM-L-6-v2):
  Relevance logit from concatenated [query, doc_text].
  cross_encoder_min_score (config): below threshold CE score excluded.

Final selection: _select_final_documents(scored, top_k)
  Combine L1 rrf_score + L2 gist_score + L3 colbert_score + L3 ce_score
  Apply all min-score filters; sort by final_score DESC; return top_k.

------------------------------------------------------------------------------
PGVectorConfig -- TUNABLE PARAMETERS (what you are optimizing)
------------------------------------------------------------------------------

  top_k                   [3-20   int ] Final results count.  Higher = more recall.
  rrf_k                   [10-120 int ] RRF smoothing.  Higher = flatter rank blending.
  gist_lambda             [0.0-1.0    ] Relevance/diversity tradeoff in GIST selection.
  bm25_min_score          [0.0-2.0    ] L1 BM25 score filter.
  dense_min_similarity    [0.0-2.0    ] L1 dense cosine score filter.
  colbert_min_score       [0.0-2.0    ] L3 ColBERT MaxSim score filter.
  cross_encoder_min_score [0.0-2.0    ] L3 cross-encoder relevance filter.
"""


# ── System prompt builder & parser ───────────────────────────────────────────

def build_system_prompt(config: dict) -> str:
    lines = [PIPELINE_CODE_WALKTHROUGH.rstrip(), "", "== CURRENT CONFIG =="]
    for k, v in config.items():
        if isinstance(v, float):
            lines.append(f"{k}: {v:.4f}")
        else:
            lines.append(f"{k}: {v}")
    lines += [
        "",
        "OPTIMIZATION TASK:",
        "Adjust ONLY the numeric values in == CURRENT CONFIG == to improve the mean",
        "of 4 RAGAS metrics: context_precision, context_recall, faithfulness, answer_relevancy.",
        "Do not rename or add keys.  Respect range constraints:",
        "  top_k: 3-20 (integer)  |  rrf_k: 10-120 (integer)  |  gist_lambda: 0.0-1.0",
        "  all min_score / min_similarity: 0.0-2.0",
    ]
    return "\n".join(lines)


def parse_config_from_prompt(text: str) -> dict:
    """
    Extract == CURRENT CONFIG == block and parse key: value pairs.
    Clamps to safe ranges.  Falls back to DEFAULT_CONFIG for missing keys.
    """
    config = copy.copy(DEFAULT_CONFIG)
    block_match = re.search(
        r"== CURRENT CONFIG ==\s*(.*?)(?:==|OPTIMIZATION|$)", text, re.DOTALL
    )
    if not block_match:
        print("  [parse] WARNING: config block not found, using defaults")
        return config

    block = block_match.group(1)
    for match in re.finditer(r"([\w]+):\s*([\d.]+)", block):
        key, raw = match.group(1), match.group(2)
        if key not in DEFAULT_CONFIG:
            continue
        try:
            val = float(raw)
            if key in ("top_k", "rrf_k"):
                val = int(round(val))
            config[key] = val
        except ValueError:
            pass

    # Safety clamping
    config["top_k"] = max(3, min(20, int(config["top_k"])))
    config["rrf_k"] = max(10, min(120, int(config["rrf_k"])))
    config["gist_lambda"] = max(0.0, min(1.0, config["gist_lambda"]))
    for k in ("bm25_min_score", "dense_min_similarity",
              "colbert_min_score", "cross_encoder_min_score"):
        config[k] = max(0.0, min(2.0, config[k]))

    return config


def make_retriever(config: dict) -> ArxivRetriever:
    cfg = PGVectorConfig(
        rrf_k=config["rrf_k"],
        gist_lambda=config["gist_lambda"],
        bm25_min_score=config["bm25_min_score"],
        dense_min_similarity=config["dense_min_similarity"],
        colbert_min_score=config["colbert_min_score"],
        cross_encoder_min_score=config["cross_encoder_min_score"],
    )
    return ArxivRetriever(cfg)


# ── QA generation from live retrieval ────────────────────────────────────────

_QA_GEN_PROMPT = (
    "Below are passages retrieved from academic papers on the topic: \"{topic}\"\n\n"
    "PASSAGES:\n{passages}\n\n"
    "Your task: Generate {n} question-answer pairs each fully answerable from "
    "the passages above.\n\n"
    "Rules:\n"
    "- Questions must be specific (not yes/no, not about authors/title).\n"
    "- Answers must be at least 2 sentences drawn only from the passages.\n"
    "- Number each pair clearly.\n\n"
    "Format (repeat for each pair):\n"
    "Q1: <question>\nA1: <answer>\n"
    "Q2: <question>\nA2: <answer>\n"
    "..."
)


def _parse_qa_pairs(text: str) -> list:
    # Strip <think>...</think> reasoning blocks (qwen3 / o-series thinking models)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    pairs = []
    questions = re.findall(r"Q\d+:\s*(.+?)(?=A\d+:)", text, re.DOTALL)
    answers = re.findall(r"A\d+:\s*(.+?)(?=Q\d+:|$)", text, re.DOTALL)
    for q, a in zip(questions, answers):
        q, a = q.strip(), a.strip()
        if q and len(a.split()) >= 10:
            pairs.append({"question": q, "answer": a})
    return pairs


def generate_qa_from_retrieval(
    retriever,
    n_qa_per_query: int = 3,
    top_k: int = 5,
    verbose: bool = True,
) -> list:
    """
    Run TOPIC_QUERIES through the retriever, collect section text, call
    gpt-4.1 to derive QA pairs from the retrieved passages.
    Returns list of dicts: question, answer, topic, section_texts.
    """
    client = openai.OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", "copilot"),
        base_url=COPILOT_PROXY,
    )
    all_pairs: list = []

    for topic in TOPIC_QUERIES:
        if verbose:
            print(f"  [QA-gen] query: {topic[:60]}", flush=True)
        try:
            docs = retriever.search(topic, top_k=top_k)
        except Exception as e:
            print(f"  [QA-gen] retrieval error: {e}")
            continue

        section_texts: list = []
        for d in docs:
            if d.content:
                section_texts.append(d.content)
            elif hasattr(d, "sections") and d.sections:
                for sec in d.sections:
                    if sec.content:
                        section_texts.append(sec.content)

        if len(section_texts) < 2:
            if verbose:
                print(f"             -> skipped (< 2 sections)")
            continue

        passages = "\n\n---\n\n".join(section_texts[:5])
        prompt = _QA_GEN_PROMPT.format(topic=topic, passages=passages, n=n_qa_per_query)

        try:
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1500,
            )
            raw = response.choices[0].message.content.strip()
        except Exception as e:
            print(f"  [QA-gen] LLM error: {e}")
            continue

        pairs = _parse_qa_pairs(raw)
        for p in pairs:
            p["topic"] = topic
            p["section_texts"] = section_texts[:5]
        all_pairs.extend(pairs)

        if verbose:
            print(f"             -> {len(pairs)} pairs  (total {len(all_pairs)})")

    return all_pairs


# ── 4-metric RAGAS evaluation ─────────────────────────────────────────────────

def run_4metric_eval(
    retriever,
    qa_pairs: list,
    top_k: int = 5,
    n_pairs: Optional[int] = None,
    verbose: bool = True,
    llm_model: str = RASGAS_MODEL,
) -> dict:
    """
    Thin wrapper around run_eval that returns all 4 RAGAS metrics:
      context_precision, context_recall, faithfulness, answer_relevancy.
    answer_relevancy is scored via LLM rubric inside run_eval (no embeddings needed).
    """
    return run_eval(
        retriever,
        qa_pairs,
        top_k=top_k,
        n_pairs=n_pairs,
        verbose=verbose,
        llm_model=llm_model,
    )


# ── TextGrad feedback text ────────────────────────────────────────────────────

def score_to_loss_text(scores: dict, config: dict, iteration: int) -> str:
    cp = scores.get("context_precision", 0.0)
    cr = scores.get("context_recall", 0.0)
    fa = scores.get("faithfulness", 0.0)
    ar = scores.get("answer_relevancy", 0.0)
    mean_score = (cp + cr + fa + ar) / 4.0

    analysis_lines = []
    if cr < 0.60:
        analysis_lines.append(
            "context_recall LOW: retrieved contexts miss ground-truth coverage; "
            "try increasing top_k or reducing dense_min_similarity / bm25_min_score."
        )
    if cp < 0.60:
        analysis_lines.append(
            "context_precision LOW: too many irrelevant chunks retrieved; "
            "try increasing cross_encoder_min_score or bm25_min_score."
        )
    if fa < 0.60:
        analysis_lines.append(
            "faithfulness LOW: answers not grounded in retrieved contexts; "
            "try increasing cross_encoder_min_score to surface only topically relevant chunks."
        )
    if ar < 0.60:
        analysis_lines.append(
            "answer_relevancy LOW: contexts lead to off-topic answers; "
            "try adjusting gist_lambda or top_k for better relevance/diversity balance."
        )
    if not analysis_lines:
        analysis_lines.append(
            "All metrics >= 0.60.  Fine-tune tradeoffs to push mean RAGAS score higher."
        )

    analysis = "\n".join(f"  - {l}" for l in analysis_lines)

    return (
        f"Iteration {iteration} RAGAS evaluation (all metrics: higher = better, target >= 0.70):\n\n"
        f"  context_precision  : {cp:.4f}   (goal >= 0.70)\n"
        f"  context_recall     : {cr:.4f}   (goal >= 0.70)\n"
        f"  faithfulness       : {fa:.4f}   (goal >= 0.75)\n"
        f"  answer_relevancy   : {ar:.4f}   (goal >= 0.70)\n"
        f"  --------------------------------------------------\n"
        f"  MEAN RAGAS (4-metric): {mean_score:.4f}   (target >= 0.70)\n\n"
        f"Current config:\n{json.dumps(config, indent=2)}\n\n"
        f"Analysis:\n{analysis}\n\n"
        "Update == CURRENT CONFIG == numeric values to improve mean RAGAS score.\n"
        "Respect parameter ranges listed in the PIPELINE CODE WALKTHROUGH."
    )


# ── MLflow logging ────────────────────────────────────────────────────────────

def log_iteration(
    iteration: int,
    scores: dict,
    config: dict,
    prompt_text: str,
    tmpdir: str,
) -> None:
    metric_keys = ("context_precision", "context_recall", "faithfulness", "answer_relevancy")
    mlflow.log_metrics(
        {k: v for k, v in scores.items() if k in metric_keys},
        step=iteration,
    )
    mean_score = sum(scores.get(k, 0.0) for k in metric_keys) / 4.0
    mlflow.log_metric("mean_ragas", mean_score, step=iteration)

    # Params only at iteration 0 (avoids MLflow duplicate-key error)
    if iteration == 0:
        mlflow.log_params({f"init_{k}": v for k, v in config.items()})
    else:
        mlflow.log_dict(config, f"configs/iter{iteration:03d}_config.json")

    prompt_path = os.path.join(tmpdir, f"system_prompt_v{iteration:03d}.txt")
    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write(prompt_text)
    mlflow.log_artifact(prompt_path, artifact_path="prompts")

    for fpath in RETRIEVER_FILES:
        if os.path.exists(fpath):
            dest = os.path.join(tmpdir, f"iter{iteration:03d}_{os.path.basename(fpath)}")
            shutil.copy2(fpath, dest)
            mlflow.log_artifact(dest, artifact_path=f"code/iter{iteration:03d}")


# ── Main optimization loop ────────────────────────────────────────────────────

def main(
    n_iterations: int = 10,
    train_batch_size: int = 10,
    eval_top_k: int = 5,
    seed: int = 42,
    n_qa_per_query: int = 3,
) -> None:
    random.seed(seed)

    print("=" * 70)
    print("TextGrad + MLflow  3-Layer Retrieval Optimization  (v2 -- 4-metric RAGAS)")
    print("=" * 70)

    print(f"\n  Initialising TextGrad engine: {LITELLM_MODEL} -> {COPILOT_PROXY}")
    engine = LiteLLMEngine(LITELLM_MODEL)
    tg.set_backward_engine(engine, override=True)

    current_config = copy.copy(DEFAULT_CONFIG)
    current_config["top_k"] = eval_top_k

    _wait_for_postgres()  # guard against slow docker container init

    print(f"\n  Building initial retriever (top_k={eval_top_k})...")
    init_retriever = make_retriever(current_config)

    print(
        f"\n  Generating live QA pairs "
        f"({len(TOPIC_QUERIES)} queries x {n_qa_per_query} QA each)..."
    )
    all_qa = generate_qa_from_retrieval(
        init_retriever,
        n_qa_per_query=n_qa_per_query,
        top_k=eval_top_k,
        verbose=True,
    )
    if not all_qa:
        raise RuntimeError("No QA pairs generated -- check retriever and LLM proxy.")

    random.shuffle(all_qa)
    split = int(0.7 * len(all_qa))
    train_pairs = all_qa[:split]
    test_pairs = all_qa[split:]
    print(
        f"  QA pairs: {len(all_qa)} total  "
        f"| train={len(train_pairs)}  test={len(test_pairs)}"
    )

    prompt_text = build_system_prompt(current_config)
    system_prompt = tg.Variable(
        prompt_text,
        requires_grad=True,
        role_description=(
            "3-layer retrieval pipeline configuration. "
            "Modify == CURRENT CONFIG == numeric values to maximise mean 4-metric RAGAS score."
        ),
    )

    optimizer = tg.optimizer.TextualGradientDescent(
        parameters=[system_prompt],
        engine=engine,
        constraints=[
            "Only change numeric values in == CURRENT CONFIG ==.",
            "Do not rename or add config keys.",
            "top_k must be an integer between 3 and 20.",
            "rrf_k must be an integer between 10 and 120.",
            "gist_lambda must be a float between 0.0 and 1.0.",
            "All min_score / min_similarity values must be floats between 0.0 and 2.0.",
        ],
    )

    best_mean = -1.0
    best_config = copy.copy(current_config)
    tmpdir = tempfile.mkdtemp(prefix="textgrad_mlflow_")

    mlflow.set_experiment(MLFLOW_EXP)
    with mlflow.start_run(run_name=f"textgrad-v2-{n_iterations}iter") as parent_run:
        run_id = parent_run.info.run_id
        mlflow.log_params({
            "n_iterations": n_iterations,
            "train_batch_size": train_batch_size,
            "eval_top_k": eval_top_k,
            "n_qa_per_query": n_qa_per_query,
            "n_train_pairs": len(train_pairs),
            "n_test_pairs": len(test_pairs),
            "textgrad_engine": LITELLM_MODEL,
            "proxy": COPILOT_PROXY,
        })
        mlflow.log_dict(DEFAULT_CONFIG, "default_config.json")

        for iteration in range(n_iterations):
            print(
                f"\n-- Iteration {iteration + 1}/{n_iterations} "
                f"{'--' * 20}"
            )

            current_config = parse_config_from_prompt(system_prompt.value)
            print(f"  Config: {json.dumps(current_config)}")

            retriever = make_retriever(current_config)
            batch = random.sample(train_pairs, min(train_batch_size, len(train_pairs)))

            print(f"  Running 4-metric RAGAS on {len(batch)} pairs...")
            try:
                scores = run_4metric_eval(
                    retriever,
                    batch,
                    top_k=current_config["top_k"],
                    n_pairs=len(batch),
                    verbose=False,
                )
            except Exception as e:
                print(f"  [RAGAS ERROR] {e} -- skipping iteration")
                continue

            cp = scores.get("context_precision", 0.0)
            cr = scores.get("context_recall", 0.0)
            fa = scores.get("faithfulness", 0.0)
            ar = scores.get("answer_relevancy", 0.0)
            mean_score = (cp + cr + fa + ar) / 4.0
            print(
                f"  CP={cp:.3f}  CR={cr:.3f}  F={fa:.3f}  AR={ar:.3f} "
                f"| mean={mean_score:.3f}"
            )

            if mean_score > best_mean:
                best_mean = mean_score
                best_config = copy.copy(current_config)
                print(f"  * New best: {mean_score:.4f}")

            log_iteration(iteration, scores, current_config, system_prompt.value, tmpdir)

            # CRITICAL FIX: predecessors=[system_prompt] wires the computation graph.
            # Without this, loss_var.backward() has no path to system_prompt and
            # optimizer.step() never changes system_prompt.value (no-op optimizer).
            if math.isnan(mean_score):
                print("  [SKIP] All RAGAS scores are nan -- skipping TextGrad step")
                continue

            loss_text = score_to_loss_text(scores, current_config, iteration)
            # requires_grad=True required: TextGrad rejects requires_grad=False
            # when any predecessor has requires_grad=True (system_prompt does).
            # The optimizer is bound to parameters=[system_prompt] only, so
            # loss_var itself is never updated -- it is purely the loss root.
            loss_var = tg.Variable(
                loss_text,
                requires_grad=True,
                predecessors=[system_prompt],
                role_description="RAGAS 4-metric evaluation feedback",
            )
            optimizer.zero_grad()
            loss_var.backward()
            optimizer.step()
            print(f"  TextGrad step complete.")

        # Final held-out test evaluation
        print("\n-- Final held-out test evaluation " + "--" * 18)
        final_retriever = make_retriever(best_config)
        try:
            final_scores = run_4metric_eval(
                final_retriever,
                test_pairs,
                top_k=best_config["top_k"],
                verbose=True,
            )
        except Exception as e:
            print(f"  [Final eval error] {e}")
            final_scores = {}

        if final_scores:
            metric_keys = ("context_precision", "context_recall",
                           "faithfulness", "answer_relevancy")
            final_mean = sum(final_scores.get(k, 0.0) for k in metric_keys) / 4.0
            print(f"\n  Final 4-metric mean RAGAS: {final_mean:.4f}")
            mlflow.log_metrics(
                {f"final_{k}": v for k, v in final_scores.items() if k in metric_keys},
            )
            mlflow.log_metric("final_mean_ragas", final_mean)

        mlflow.log_dict(best_config, "best_config.json")
        best_prompt_path = os.path.join(tmpdir, "best_system_prompt.txt")
        with open(best_prompt_path, "w", encoding="utf-8") as fh:
            fh.write(build_system_prompt(best_config))
        mlflow.log_artifact(best_prompt_path)

        print(f"\n  Best config during training (mean_ragas={best_mean:.4f}):")
        print(json.dumps(best_config, indent=4))
        print(f"  MLflow run_id: {run_id}")

    shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="TextGrad v2: 3-layer retrieval config optimization with 4-metric RAGAS"
    )
    parser.add_argument("--n-iterations",     type=int, default=10)
    parser.add_argument("--train-batch-size", type=int, default=10)
    parser.add_argument("--top-k",            type=int, default=5)
    parser.add_argument("--seed",             type=int, default=42)
    parser.add_argument("--n-qa-per-query",   type=int, default=3,
                        help="QA pairs to generate per topic query at startup")
    args = parser.parse_args()

    main(
        n_iterations=args.n_iterations,
        train_batch_size=args.train_batch_size,
        eval_top_k=args.top_k,
        seed=args.seed,
        n_qa_per_query=args.n_qa_per_query,
    )
