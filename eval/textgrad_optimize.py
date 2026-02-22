"""
TextGrad + MLflow Optimization of 3-Layer RAGAS Retrieval Pipeline

Core Thesis:
    Encode the 3-layer retrieval pipeline architecture and its hyperparameters
    as a TextGrad Variable (system prompt). Use TextualGradientDescent driven
    by RAGAS scores-as-loss feedback to iteratively improve the configuration.
    Log every iteration — scores, hyperparams, system prompt, and full code
    snapshots — to MLflow so all changes are auditable.

== SYSTEM PROMPT TEMPLATE ==
The TextGrad variable text contains a human-readable description of the pipeline
followed by a parseable config block:

    == PIPELINE ARCHITECTURE ==
    ...natural language description...

    == CURRENT CONFIG ==
    top_k: 5
    rrf_k: 60
    gist_lambda: 0.7
    bm25_min_score: 0.0
    dense_min_similarity: 0.0
    colbert_min_score: 0.0
    cross_encoder_min_score: 0.0

TextGrad is instructed to only modify the == CURRENT CONFIG == block.
Values are parsed with regex and mapped to PGVectorConfig kwargs.

MLflow logging per iteration:
    - Metrics : context_precision, context_recall, faithfulness, answer_relevancy
    - Params  : all 7 hyperparameters
    - Artifacts: system_prompt_v{iter}.txt, pgvector_retriever.py snapshot
"""

import json
import os
import re
import sys
import copy
import shutil
import tempfile
import random
from typing import Optional

import mlflow
import textgrad as tg
from textgrad.engine import LiteLLMEngine

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from arxiv_retriever import ArxivRetriever
from retrieval.pgvector_retriever import PGVectorConfig
from eval.ragas_eval import run_eval

# ── Constants ─────────────────────────────────────────────────────────────────
OLLAMA_MODEL   = "granite3.3:latest"
LITELLM_MODEL  = "ollama/" + OLLAMA_MODEL
QA_FILE        = os.path.join(ROOT, "eval", "data", "qa_pairs.json")
MLFLOW_EXP     = "arxiv-3layer-retrieval-optimization"
RETRIEVER_FILES = [
    os.path.join(ROOT, "retrieval", "pgvector_retriever.py"),
    os.path.join(ROOT, "retrieval", "base_gist_retriever.py"),
    os.path.join(ROOT, "retrieval", "gist_retriever.py"),
]

# Default hyperparameters (matched to GISTConfig / PGVectorConfig)
DEFAULT_CONFIG = {
    "top_k":               5,
    "rrf_k":               60,
    "gist_lambda":         0.7,
    "bm25_min_score":      0.0,
    "dense_min_similarity": 0.0,
    "colbert_min_score":   0.0,
    "cross_encoder_min_score": 0.0,
}

# ── System prompt ──────────────────────────────────────────────────────────────
PIPELINE_DESCRIPTION = """\
== PIPELINE ARCHITECTURE ==

This is a 3-layer academic paper retrieval pipeline (ArxivRetriever).

Layer 1 — Sparse + Dense seeding
  - BM25 (PostgreSQL tsvector + GIN index, table: layer1_bm25_sparse)
  - Dense ANN (128-dim model2vec HNSW, table: layer1_embeddings_128d)
  - Results merged with Reciprocal Rank Fusion (RRF, constant: rrf_k)
  - Pool = top_k², seeds = prev_fibonacci(pool)

Layer 2 — GIST branch expansion
  - GIST BM25 path: seed embeddings (256d) → cosine coverage → expand via BM25
  - GIST Dense path: seed centroid (256d) ANN on layer2_embeddings_256d
  - Merged again with RRF; weighted by gist_lambda (relevance) vs 1-gist_lambda (diversity)

Layer 3 — Cross-encoder re-ranking
  - MS-MARCO MiniLM-L-6-v2 cross-encoder scores all candidates
  - Final ranking: RRF of (L1 score, L2 GIST score, L3 CE score)
  - Score thresholds applied: bm25_min_score, dense_min_similarity,
    colbert_min_score, cross_encoder_min_score

Evaluation uses RAGAS 4 metrics on academic paper QA pairs:
  - context_precision, context_recall, faithfulness, answer_relevancy

OPTIMIZATION TASK:
Adjust the == CURRENT CONFIG == values below to improve the mean of all 4 RAGAS metrics.
- top_k: number of final results (integer, 3–20)
- rrf_k: RRF constant (integer, 30–120); higher smooths rank differences
- gist_lambda: float 0–1; 1.0 = pure relevance, 0.0 = pure diversity
- bm25_min_score / dense_min_similarity / colbert_min_score / cross_encoder_min_score:
  score cutoffs (float ≥ 0); higher = fewer but more precise results

IMPORTANT: Only modify the numeric values in == CURRENT CONFIG ==.
Do not change key names. Do not add new keys.
"""

def build_system_prompt(config: dict) -> str:
    lines = [PIPELINE_DESCRIPTION.rstrip(), "", "== CURRENT CONFIG =="]
    for k, v in config.items():
        if isinstance(v, float):
            lines.append(f"{k}: {v:.4f}")
        else:
            lines.append(f"{k}: {v}")
    return "\n".join(lines)


def parse_config_from_prompt(text: str) -> dict:
    """
    Extract == CURRENT CONFIG == block and parse key: value pairs.
    Clamps values to safe ranges.
    Falls back to DEFAULT_CONFIG for any missing/invalid keys.
    """
    config = copy.copy(DEFAULT_CONFIG)
    block_match = re.search(r"== CURRENT CONFIG ==\s*(.*?)(?:==|$)", text, re.DOTALL)
    if not block_match:
        print("  [parse] WARNING: config block not found, using defaults")
        return config

    block = block_match.group(1)
    for match in re.finditer(r"(\w+):\s*([\d.]+)", block):
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

    # Safety clamp
    config["top_k"]   = max(3,   min(20,  config["top_k"]))
    config["rrf_k"]   = max(10,  min(120, config["rrf_k"]))
    config["gist_lambda"] = max(0.0, min(1.0, config["gist_lambda"]))
    for k in ("bm25_min_score", "dense_min_similarity",
              "colbert_min_score", "cross_encoder_min_score"):
        config[k] = max(0.0, min(2.0, config[k]))

    return config


def make_retriever(config: dict) -> ArxivRetriever:
    """Build ArxivRetriever from parsed config dict."""
    cfg = PGVectorConfig(
        rrf_k=config["rrf_k"],
        gist_lambda=config["gist_lambda"],
        bm25_min_score=config["bm25_min_score"],
        dense_min_similarity=config["dense_min_similarity"],
        colbert_min_score=config["colbert_min_score"],
        cross_encoder_min_score=config["cross_encoder_min_score"],
    )
    return ArxivRetriever(cfg)


def score_to_loss_text(scores: dict, config: dict, iteration: int) -> str:
    """
    Format RAGAS scores as natural-language feedback for TextGrad backward pass.
    """
    mean_score = sum(
        scores[k] for k in ("context_precision", "context_recall",
                             "faithfulness", "answer_relevancy")
    ) / 4.0

    feedback = f"""
Iteration {iteration} evaluation results (higher = better for all metrics):

  context_precision   : {scores['context_precision']:.4f}   (goal >= 0.70)
  context_recall      : {scores['context_recall']:.4f}   (goal >= 0.70)
  faithfulness        : {scores['faithfulness']:.4f}   (goal >= 0.75)
  answer_relevancy    : {scores['answer_relevancy']:.4f}   (goal >= 0.70)
  ------------------------------------------
  MEAN RAGAS SCORE    : {mean_score:.4f}   (target >= 0.70)

Current config used:
{json.dumps(config, indent=2)}

Analysis:
{"context_recall is LOW — try increasing top_k or reducing dense_min_similarity to retrieve more context." if scores['context_recall'] < 0.5 else ""}
{"context_precision is LOW — try increasing cross_encoder_min_score or bm25_min_score to filter irrelevant chunks." if scores['context_precision'] < 0.5 else ""}
{"faithfulness is LOW — consider increasing cross_encoder_min_score to ensure more topically relevant chunks." if scores['faithfulness'] < 0.5 else ""}
{"answer_relevancy is LOW — try increasing gist_lambda toward 1.0 to prioritize relevance over diversity." if scores['answer_relevancy'] < 0.5 else ""}

Update == CURRENT CONFIG == to improve the mean RAGAS score.
"""
    return feedback.strip()


def log_iteration(run_id: str, iteration: int, scores: dict,
                  config: dict, prompt_text: str, tmpdir: str) -> None:
    """Log all artifacts and metrics for one TextGrad iteration to MLflow."""
    with mlflow.start_run(run_id=run_id):
        # Metrics
        mlflow.log_metrics(
            {k: v for k, v in scores.items() if isinstance(v, float)},
            step=iteration,
        )
        mean_score = sum(
            scores[k] for k in ("context_precision", "context_recall",
                                 "faithfulness", "answer_relevancy")
        ) / 4.0
        mlflow.log_metric("mean_ragas", mean_score, step=iteration)

        # Params (only on first iteration to avoid overwrite conflicts)
        if iteration == 0:
            mlflow.log_params({
                f"init_{k}": v for k, v in config.items()
            })
        else:
            mlflow.log_params({f"iter{iteration}_{k}": v for k, v in config.items()})

        # System prompt artifact
        prompt_path = os.path.join(tmpdir, f"system_prompt_v{iteration}.txt")
        with open(prompt_path, "w", encoding="utf-8") as f:
            f.write(prompt_text)
        mlflow.log_artifact(prompt_path, artifact_path="prompts")

        # Full code snapshots
        for fpath in RETRIEVER_FILES:
            if os.path.exists(fpath):
                dest = os.path.join(tmpdir, f"iter{iteration}_{os.path.basename(fpath)}")
                shutil.copy2(fpath, dest)
                mlflow.log_artifact(dest, artifact_path=f"code/iter{iteration}")


# ── Main optimization loop ────────────────────────────────────────────────────
def main(
    n_iterations: int = 10,
    train_batch_size: int = 15,
    eval_top_k: int = 5,
    seed: int = 42,
    qa_file: str = QA_FILE,
):
    random.seed(seed)

    print("=" * 60)
    print("TextGrad + MLflow 3-Layer Retrieval Optimization")
    print("=" * 60)

    # Load QA pairs
    with open(qa_file, encoding="utf-8") as f:
        qa_data = json.load(f)
    train_pairs = qa_data["train"]
    test_pairs  = qa_data["test"]
    print(f"  train={len(train_pairs)}  test={len(test_pairs)}")

    # MLflow experiment
    mlflow.set_experiment(MLFLOW_EXP)

    # TextGrad engine
    print(f"\n  Initialising TextGrad engine: {LITELLM_MODEL}")
    engine = LiteLLMEngine(LITELLM_MODEL)
    tg.set_backward_engine(engine, override=True)

    # System prompt Variable (the thing TextGrad will optimise)
    current_config  = copy.copy(DEFAULT_CONFIG)
    current_config["top_k"] = eval_top_k
    prompt_text     = build_system_prompt(current_config)

    system_prompt = tg.Variable(
        prompt_text,
        requires_grad=True,
        role_description=(
            "Retrieval pipeline configuration. "
            "Modify == CURRENT CONFIG == values to maximise RAGAS scores."
        ),
    )

    optimizer = tg.optimizer.TextualGradientDescent(
        parameters=[system_prompt],
        engine=engine,
        constraints=[
            "Only change numeric values in == CURRENT CONFIG ==.",
            "Do not rename or add keys.",
            "Keep top_k between 3 and 20.",
            "Keep rrf_k between 10 and 120.",
            "Keep gist_lambda between 0.0 and 1.0.",
            "Keep all min_score / min_similarity values between 0.0 and 2.0.",
        ],
    )

    best_mean  = -1.0
    best_config = copy.copy(current_config)
    tmpdir = tempfile.mkdtemp(prefix="textgrad_mlflow_")

    # Start a parent MLflow run for the whole optimization
    with mlflow.start_run(run_name=f"textgrad-opt-{n_iterations}iter") as parent_run:
        run_id = parent_run.info.run_id
        mlflow.log_param("n_iterations",      n_iterations)
        mlflow.log_param("train_batch_size",   train_batch_size)
        mlflow.log_param("eval_top_k",         eval_top_k)
        mlflow.log_param("textgrad_engine",    LITELLM_MODEL)
        mlflow.log_dict(DEFAULT_CONFIG,        "default_config.json")

        for iteration in range(n_iterations):
            print(f"\n── Iteration {iteration+1}/{n_iterations} ────────────────────────────")

            # 1. Parse current config from system prompt
            current_config = parse_config_from_prompt(system_prompt.value)
            print(f"  Config: {json.dumps(current_config)}")

            # 2. Build retriever
            retriever = make_retriever(current_config)

            # 3. Sample mini-batch
            batch = random.sample(train_pairs, min(train_batch_size, len(train_pairs)))

            # 4. RAGAS evaluation
            print(f"  Running RAGAS on {len(batch)} pairs...")
            try:
                scores = run_eval(
                    retriever, batch,
                    top_k=current_config["top_k"],
                    n_pairs=len(batch),
                    verbose=False,
                )
            except Exception as e:
                print(f"  [RAGAS ERROR] {e} — skipping iteration")
                continue

            mean_score = sum(
                scores[k] for k in ("context_precision", "context_recall",
                                     "faithfulness", "answer_relevancy")
            ) / 4.0
            print(f"  Scores: CP={scores['context_precision']:.3f}  "
                  f"CR={scores['context_recall']:.3f}  "
                  f"F={scores['faithfulness']:.3f}  "
                  f"AR={scores['answer_relevancy']:.3f}  "
                  f"| mean={mean_score:.3f}")

            # Track best
            if mean_score > best_mean:
                best_mean   = mean_score
                best_config = copy.copy(current_config)
                print(f"  ★ New best: {mean_score:.4f}")

            # 5. MLflow log
            log_iteration(run_id, iteration, scores, current_config,
                          system_prompt.value, tmpdir)

            # 6. TextGrad backward + step
            loss_text = score_to_loss_text(scores, current_config, iteration)
            loss_var  = tg.Variable(
                loss_text,
                requires_grad=False,
                role_description="RAGAS evaluation feedback for retrieval pipeline",
            )
            # TextGrad: attach loss to system_prompt gradient graph
            response_var = tg.Variable(
                system_prompt.value,
                requires_grad=True,
                role_description="retrieval system prompt to optimise",
            )
            # Compute gradient: what should change to reduce the loss?
            optimizer.zero_grad()
            loss_var.backward()
            optimizer.step()

            print(f"  TextGrad step done.")

        # ── Final evaluation on test set ────────────────────────────────────
        print("\n── Final evaluation on test set ──────────────────────────────")
        retriever = make_retriever(best_config)
        try:
            final_scores = run_eval(
                retriever, test_pairs,
                top_k=best_config["top_k"],
                verbose=True,
            )
        except Exception as e:
            print(f"  [Final eval error] {e}")
            final_scores = {}

        if final_scores:
            final_mean = sum(
                final_scores[k] for k in ("context_precision", "context_recall",
                                           "faithfulness", "answer_relevancy")
            ) / 4.0
            print(f"\n  Final mean RAGAS: {final_mean:.4f}")
            mlflow.log_metrics({f"final_{k}": v for k, v in final_scores.items()
                                if isinstance(v, float)})
            mlflow.log_metric("final_mean_ragas", final_mean)

        mlflow.log_dict(best_config, "best_config.json")
        best_prompt_path = os.path.join(tmpdir, "best_system_prompt.txt")
        with open(best_prompt_path, "w", encoding="utf-8") as f:
            f.write(build_system_prompt(best_config))
        mlflow.log_artifact(best_prompt_path)

        print(f"\n  Best config during training: {json.dumps(best_config, indent=2)}")
        print(f"  MLflow run_id: {run_id}")

    shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="TextGrad optimize 3-layer retrieval")
    parser.add_argument("--n-iterations",     type=int, default=10)
    parser.add_argument("--train-batch-size", type=int, default=15)
    parser.add_argument("--top-k",            type=int, default=5)
    parser.add_argument("--qa-file",          default=QA_FILE)
    parser.add_argument("--seed",             type=int, default=42)
    args = parser.parse_args()

    main(
        n_iterations=args.n_iterations,
        train_batch_size=args.train_batch_size,
        eval_top_k=args.top_k,
        seed=args.seed,
        qa_file=args.qa_file,
    )
