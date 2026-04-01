"""
Evaluate bio_tagger_atomic.pt on the exact eval split used during training.
Reproduces the stratified split (np.random.seed(42), 20% per label) then
runs full per-class diagnostics.
"""
import sys
import json
import numpy as np
import torch
import msgpack
from collections import defaultdict
from torch.utils.data import DataLoader

sys.path.insert(0, r'c:\Users\user\arxiv_id_lists')

from tune_bio_tagger import (
    BIODataset, BIOTaggerWithDropout, collate_fn, evaluate
)

DATA_PATH   = r'c:\Users\user\arxiv_id_lists\data\bio_training_250_v15.msgpack'
MODEL_PATH  = r'c:\Users\user\arxiv_id_lists\bio_tagger_atomic.pt'
PARAMS_PATH = r'c:\Users\user\arxiv_id_lists\best_hyperparams.json'

# ── load data ──────────────────────────────────────────────────────────────
with open(DATA_PATH, 'rb') as f:
    raw = msgpack.unpackb(f.read(), raw=False)

all_examples = raw['training_data']
full_label_names = raw['label_names']
label_name_to_examples = defaultdict(list)
for idx, ex in enumerate(all_examples):
    for label_idx, (label_name, token_labels) in enumerate(ex['labels'].items()):
        if sum(token_labels) > 0:
            label_name_to_examples[label_idx].append(idx)

eval_indices = set()
np.random.seed(42)

print("\n[90/10 split]")
shuffled = np.random.permutation(len(all_examples)).tolist()
eval_n = max(10, int(0.10 * len(all_examples)))
eval_indices = set(shuffled[:eval_n])

eval_examples = [all_examples[i] for i in sorted(eval_indices)]
train_examples = [all_examples[i] for i in sorted(set(range(len(all_examples))) - eval_indices)]
print(f"  Train: {len(train_examples)} | Eval: {len(eval_examples)}")

# ── load model ─────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")

with open(PARAMS_PATH) as f:
    hp = json.load(f)
threshold = hp['params']['threshold']
print(f"Threshold: {threshold:.4f}")

ckpt = torch.load(MODEL_PATH, map_location=device)
active_labels = ckpt.get('active_labels', full_label_names)
active_label_indices = [full_label_names.index(l) for l in active_labels]

from transformers import BertTokenizerFast
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

model = BIOTaggerWithDropout(
    num_labels=len(active_labels),
    dropout_rate=hp['params']['dropout'],
    unfreeze_layers=hp.get('unfreeze_layers', 12)
)
model.load_state_dict(ckpt['model_state_dict'])
model.to(device)
model.eval()
print(f"Model loaded. Active labels: {active_labels}")

# ── load threshold ─────────────────────────────────────────────────────────

# ── build eval loader ──────────────────────────────────────────────────────
eval_dataset = BIODataset(eval_examples)
eval_loader = DataLoader(
    eval_dataset,
    batch_size=16,
    shuffle=False,
    collate_fn=lambda b: collate_fn(b, tokenizer, None, active_label_indices=active_label_indices)
)

# ── evaluate ───────────────────────────────────────────────────────────────
macro_f1, per_class = evaluate(
    model, eval_loader, device,
    threshold=threshold,
    return_detailed=True,
    active_label_indices=active_label_indices,
    full_label_names=full_label_names
)

# ── print results ──────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("FINAL MODEL — EVAL SET DIAGNOSTICS")
print("=" * 65)
print(f"{'Label':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
print("-" * 55)
for m in per_class:
    print(f"{m['label']:<12} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f} {m['support']:>10}")
print("-" * 55)
print(f"{'Macro F1':<12} {'':>10} {'':>10} {macro_f1:>10.4f}")
print("=" * 65)
