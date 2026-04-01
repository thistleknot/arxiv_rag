"""
NuNER Finetuning Script

Core Thesis:
    Finetune NuNerZero on 415 quality-validated S-P-O training examples
    using gliner 0.2.24's model.train_model() API.

Data Format:
    {"tokenized_text": ["word1", ...], "ner": [[start, end, "label"], ...]}
    Labels: "subject", "predicate", "object" (lowercase, token-level)

Verified against:
    gliner==0.2.24, torch==2.8.0+cu128, transformers==4.57.6
    API: model.train_model() -> Trainer (confirmed via inspect)
    evaluate() returns (out, f1) — 2 values
    create_training_args() kwargs confirmed from source

Usage:
    python train_nuner.py
"""

import json
import os
import random

import torch
from gliner import GLiNER

# ── Config ──────────────────────────────────────────────────────────────
DATA_PATH = "nuner_training_data_gliner.json"
MODEL_NAME = "numind/NuNerZero"
OUTPUT_DIR = "nuner_finetuned"

TRAIN_SPLIT = 0.85
SEED = 42

NUM_STEPS = 3000
BATCH_SIZE = 2
LR_ENCODER = 1e-5
LR_OTHERS = 5e-5
WARMUP_RATIO = 0.1
EVAL_EVERY = 500
MAX_GRAD_NORM = 1.0
WEIGHT_DECAY = 0.01


def load_and_split(path, train_ratio, seed):
    """Load training data and split into train/eval sets.

    Expected keys: tokenized_text, ner
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} examples from {path}")

    # Validate format
    for i, ex in enumerate(data):
        assert "tokenized_text" in ex, f"Example {i} missing 'tokenized_text' (keys: {list(ex.keys())})"
        assert "ner" in ex, f"Example {i} missing 'ner'"

    random.seed(seed)
    indices = list(range(len(data)))
    random.shuffle(indices)

    split_idx = int(len(data) * train_ratio)
    train_data = [data[indices[i]] for i in range(split_idx)]
    eval_data = [data[indices[i]] for i in range(split_idx, len(data))]

    print(f"Train: {len(train_data)} | Eval: {len(eval_data)}")
    return train_data, eval_data


def main():
    # ── Load data ───────────────────────────────────────────────────────
    train_data, eval_data = load_and_split(DATA_PATH, TRAIN_SPLIT, SEED)

    # ── Load model ──────────────────────────────────────────────────────
    print(f"\nLoading {MODEL_NAME}...")
    model = GLiNER.from_pretrained(MODEL_NAME, local_files_only=True)
    print(f"Model loaded: {type(model).__name__}")
    print(f"  train_model available: {hasattr(model, 'train_model')}")

    # ── Output dir ──────────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Train using official API ────────────────────────────────────────
    print(f"\nStarting finetuning: {NUM_STEPS} steps")
    print(f"  batch_size={BATCH_SIZE}")
    print(f"  lr_encoder={LR_ENCODER}, lr_others={LR_OTHERS}")
    print(f"  warmup_ratio={WARMUP_RATIO}")
    print(f"  eval/save every {EVAL_EVERY} steps")
    print()

    trainer = model.train_model(
        train_dataset=train_data,
        eval_dataset=eval_data,
        output_dir=OUTPUT_DIR,
        # Schedule
        max_steps=NUM_STEPS,
        lr_scheduler_type="cosine",
        warmup_ratio=WARMUP_RATIO,
        # Batch & optimization
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=8,
        learning_rate=LR_ENCODER,
        others_lr=LR_OTHERS,
        weight_decay=WEIGHT_DECAY,
        max_grad_norm=MAX_GRAD_NORM,
        # Logging & saving
        save_steps=EVAL_EVERY,
        logging_steps=50,
        save_total_limit=3,
        # Precision (bf16 confirmed supported on Quadro RTX 5000)
        bf16=True,
        # Reporting
        report_to="none",
    )

    # ── Final evaluation ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)

    model.eval()
    # evaluate() returns (out, f1) in gliner 0.2.24
    results, f1 = model.evaluate(
        eval_data,
        flat_ner=True,
        threshold=0.5,
        batch_size=12,
    )

    print(f"\nResults:\n{results}")
    print(f"\nOverall F1: {f1:.4f}")

    # ── Save final ──────────────────────────────────────────────────────
    final_path = os.path.join(OUTPUT_DIR, "final")
    os.makedirs(final_path, exist_ok=True)
    model.save_pretrained(final_path)
    print(f"\nFinal model saved to {final_path}")

    return f1


if __name__ == "__main__":
    f1 = main()
    print(f"\n>>> Final F1: {f1:.4f}")
