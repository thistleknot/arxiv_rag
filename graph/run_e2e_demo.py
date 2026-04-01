"""
End-to-end graph transformer demo using the 45K-record checkpoint.

Pipeline:
  1. normalize_entities    → entity/predicate vocab + embeddings
  2. build_kg_dataset      → PyG Data object
  3. train_gat --smoke     → 5-epoch GATv2 sanity check  (10K edges)
  4. train_gat (full)      → 20-epoch GATv2 on full graph
  5. graph_retriever       → live query demo

Usage:
    python graph/run_e2e_demo.py
    python graph/run_e2e_demo.py --skip-full-train   (smoke only)
    python graph/run_e2e_demo.py --query "self-attention mechanism"
"""

import subprocess
import sys
import time
import argparse
from pathlib import Path

PYTHON  = sys.executable
ROOT    = Path(__file__).parent.parent
CHKPT   = ROOT / "checkpoints" / "bio_triplets_checkpoint.msgpack"
FULL    = ROOT / "checkpoints" / "bio_triplets_full_corpus.msgpack"

def run(label: str, cmd: list, cwd=ROOT):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(cwd), capture_output=False)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n[FAILED] exit {result.returncode} after {elapsed:.1f}s")
        sys.exit(result.returncode)
    print(f"\n[OK] {elapsed:.1f}s")
    return elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-full-train", action="store_true",
                        help="Only run smoke test (5 epochs), skip full 20-epoch training")
    parser.add_argument("--query", type=str,
                        default="attention mechanism in transformer models",
                        help="Demo query for the retriever")
    args = parser.parse_args()

    # Decide which msgpack to use
    if FULL.exists():
        src = FULL
        print(f"Using full corpus: {FULL}")
    else:
        src = CHKPT
        print(f"Full corpus not ready — using checkpoint: {CHKPT} (45K records, 378K triplets)")
    print()

    # ── Step 1: Normalize entities ────────────────────────────────────────────
    run("Step 1/5: Normalize entities (embed + cluster spans)",
        [PYTHON, "graph/normalize_entities.py", "--msgpack", str(src)])

    # ── Step 2: Build PyG dataset ─────────────────────────────────────────────
    run("Step 2/5: Build PyTorch Geometric dataset",
        [PYTHON, "graph/build_kg_dataset.py"])

    # ── Step 3: Smoke-test training ───────────────────────────────────────────
    run("Step 3/5: GATv2 smoke test (5 epochs, 10K edges — sanity check)",
        [PYTHON, "graph/train_gat.py", "--smoke-test"])

    # ── Step 4: Full training ─────────────────────────────────────────────────
    if not args.skip_full_train:
        run("Step 4/5: GATv2 full training (20 epochs on full graph)",
            [PYTHON, "graph/train_gat.py"])
    else:
        print("\n[SKIPPED] Step 4: full training (--skip-full-train set)")

    # ── Step 5: Retrieval demo ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Step 5/5: Graph retrieval demo")
    print(f"  Query: '{args.query}'")
    print(f"{'='*60}\n")
    subprocess.run(
        [PYTHON, "graph/graph_retriever.py", args.query, "--top-k", "20", "--n-seeds", "12"],
        cwd=str(ROOT)
    )


if __name__ == "__main__":
    main()
