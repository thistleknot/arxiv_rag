#!/usr/bin/env python3
"""
Train an EfficientNet-B0 layout classifier on docling-detected element crops.

Given one or more docling JSON files (with matching PDFs), crops every element
region, labels it by docling element type, applies Box-Cox log-ratio class
weighting, then trains EfficientNet-B0 via the standard protocol:
  1. Optuna sweep on a subset (patience=5, ≤20 epochs) → best hyperparams
  2. Full training with best params (patience=20, ≤100 epochs)
  3. Holdout evaluation

Output: layout_classifier.pt + layout_classifier_classes.json

Usage:
    python train_layout_classifier.py --json-dir <dir_with_docling_json>
    python train_layout_classifier.py --json-files a.json b.json ...
    python train_layout_classifier.py --pdf-dir <dir_with_pdfs>  # runs docling first

Preconditions:
    - JSON files were produced by `docling --to json`; matching PDF lives beside each JSON
    - CUDA preferred; CPU fallback
    - optuna, torch, torchvision, fitz (pymupdf) installed
Postconditions:
    - layout_classifier.pt (state dict with class map)
    - layout_classifier_classes.json (idx→label mapping)
    - crops/ directory of training images (useful for inspection)
Failure modes:
    - Too few classes: warns, proceeds with what is available
    - boxcox requires strictly positive input; handled by adding a small epsilon
    - PDF parse failure per page: logged, page skipped
"""

import sys
import json
import argparse
import subprocess
import sqlite3
from pathlib import Path
from collections import Counter

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    import pymupdf as fitz  # pymupdf >= 1.24
except ImportError:
    import fitz             # older pymupdf
import numpy as np
from scipy.stats import boxcox

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torchvision import models, transforms
from PIL import Image
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)


CHECKPOINT_DB = "checkpoints/layout_classifier.db"

# ── element type label map ──────────────────────────────────────────────────────

# docling label → canonical class name (lowercase for portability)
_LABEL_MAP = {
    "table": "table",
    "picture": "picture",
    "figure": "picture",
    "section_header": "section_header",
    "text": "text",
    "paragraph": "text",
    "list_item": "list_item",
    "caption": "caption",
    "formula": "formula",
    "footnote": "footnote",
    "page_header": "page_header",
    "page_footer": "page_footer",
}


# ── Box-Cox class weighting ─────────────────────────────────────────────────────

def boxcox_class_weights(labels: list[str]) -> dict[str, float]:
    """
    Compute per-class weights via Box-Cox transform of log-inverse-frequency.

    Protocol:
        1. log_inv[c] = log(N / count[c])
        2. Box-Cox transform the log_inv vector (all positive by construction)
        3. Clip to ≥0 and normalize to sum=1

    Require: at least 2 labels; all class counts ≥ 1.
    Guarantee: returns {class_name: weight}; weights sum to ~1.
    Failure mode: single-class → uniform weight {class: 1.0}.
    """
    counts = Counter(labels)
    if len(counts) < 2:
        return {c: 1.0 for c in counts}

    N = len(labels)
    classes = sorted(counts)
    log_inv = np.array([np.log(N / counts[c]) for c in classes], dtype=float)

    # Minority classes have log_inv>0; majority class may be 0 (exactly) → add eps
    log_inv = np.clip(log_inv, 1e-6, None)

    try:
        bc, _ = boxcox(log_inv)
    except Exception:
        bc = log_inv  # fallback: use raw log-inv

    bc_clipped = np.clip(bc, 0.0, None)
    total = bc_clipped.sum()
    if total < 1e-9:
        bc_clipped = np.ones_like(bc_clipped)
        total = bc_clipped.sum()

    return {c: float(bc_clipped[i] / total) for i, c in enumerate(classes)}


# ── image crop generation ───────────────────────────────────────────────────────

def _prov(item: dict) -> dict:
    prov = item.get("prov", [])
    return prov[0] if prov else {}


def _fitz_rect(bbox: dict, page_height: float) -> fitz.Rect:
    """Convert docling BOTTOMLEFT bbox to pymupdf Rect (TOPLEFT origin)."""
    return fitz.Rect(
        bbox["l"],
        page_height - bbox["t"],
        bbox["r"],
        page_height - bbox["b"],
    )


def extract_crops_from_json(json_path: Path, crops_root: Path,
                             all_crops: list[tuple[Path, str]]) -> None:
    """
    Parse one docling JSON file, crop all element regions, append to all_crops.

    Require: matching PDF lives at json_path.with_suffix('.pdf') or .PDF.
    Guarantee: each crop saved to crops_root/<label>/; (path, label) appended.
    Failure mode: missing PDF → skip file; page parse error → skip page.
    """
    try:
        doc_data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"  [skip] cannot read {json_path.name}: {exc}")
        return

    pdf_path = json_path.with_suffix(".pdf")
    if not pdf_path.is_file():
        pdf_path = json_path.with_suffix(".PDF")
    if not pdf_path.is_file():
        print(f"  [skip] no PDF beside {json_path.name}")
        return

    pages = doc_data.get("pages", {})

    try:
        fitz_doc = fitz.open(str(pdf_path))
    except Exception as exc:
        print(f"  [skip] cannot open PDF {pdf_path.name}: {exc}")
        return

    def _crop_items(items: list[dict], default_label: str) -> None:
        for idx_item, item in enumerate(items):
            raw_label = item.get("label", default_label).lower()
            canonical = _LABEL_MAP.get(raw_label, raw_label)

            prov = _prov(item)
            if not prov:
                continue

            page_no = prov.get("page_no")
            bbox = prov.get("bbox")
            if not page_no or not bbox:
                continue

            page_key = str(page_no)
            if page_key not in pages:
                continue

            page_size = pages[page_key]["size"]
            page_h = page_size["height"]

            try:
                page = fitz_doc[page_no - 1]
                rect = _fitz_rect(bbox, page_h) & page.rect
                if rect.is_empty:
                    continue
                mat = fitz.Matrix(2, 2)
                pix = page.get_pixmap(matrix=mat, clip=rect)

                label_dir = crops_root / canonical
                label_dir.mkdir(parents=True, exist_ok=True)
                stem = f"{json_path.stem}_p{page_no}_{idx_item:04d}"
                out_path = label_dir / f"{stem}.png"
                if not out_path.exists():
                    pix.save(str(out_path))
                all_crops.append((out_path, canonical))
            except Exception:
                continue

    # Elements are in: texts, pictures, tables
    _crop_items(doc_data.get("texts", []), "text")
    _crop_items(doc_data.get("pictures", []), "picture")
    _crop_items(doc_data.get("tables", []), "table")

    fitz_doc.close()


# ── dataset ─────────────────────────────────────────────────────────────────────

class CropDataset(Dataset):
    """
    Image dataset built from pre-cropped PNG files.

    Require: all paths in samples exist and are valid images.
    """

    _TRANSFORM = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    def __init__(self, samples: list[tuple[Path, int]]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label_idx = self.samples[idx]
        img = Image.open(str(path)).convert("RGB")
        return self._TRANSFORM(img), label_idx


# ── model ───────────────────────────────────────────────────────────────────────

def build_model(num_classes: int) -> nn.Module:
    """EfficientNet-B0 with replaced classifier head."""
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


# ── training helpers ─────────────────────────────────────────────────────────────

def train_epoch(model: nn.Module, loader: DataLoader,
                optimizer: torch.optim.Optimizer,
                criterion: nn.Module, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
    return total_loss / max(len(loader.dataset), 1)


def eval_accuracy(model: nn.Module, loader: DataLoader,
                  device: torch.device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += len(y)
    return correct / max(total, 1)


def run_training(model: nn.Module, train_loader: DataLoader,
                 val_loader: DataLoader, criterion: nn.Module,
                 lr: float, weight_decay: float, max_epochs: int,
                 patience: int, device: torch.device) -> float:
    """Train with early stopping; return best validation accuracy."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    best_acc = 0.0
    no_improve = 0
    state_dict = None

    for epoch in range(1, max_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_acc = eval_accuracy(model, val_loader, device)
        scheduler.step()

        if val_acc > best_acc:
            best_acc = val_acc
            state_dict = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            break

    if state_dict is not None:
        model.load_state_dict(state_dict)
    return best_acc


# ── optuna objective ─────────────────────────────────────────────────────────────

def make_objective(train_sub: Subset, val_sub: Subset,
                   class_weights_tensor: torch.Tensor,
                   num_classes: int, device: torch.device):
    def objective(trial: optuna.Trial) -> float:
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

        t_loader = DataLoader(train_sub, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
        v_loader = DataLoader(val_sub, batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=True)

        model = build_model(num_classes).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        return run_training(model, t_loader, v_loader, criterion,
                            lr, weight_decay,
                            max_epochs=20, patience=5, device=device)

    return objective


# ── checkpoint helpers ───────────────────────────────────────────────────────────

def _ensure_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE IF NOT EXISTS training_runs "
        "(id INTEGER PRIMARY KEY, phase TEXT, best_acc REAL, "
        " best_params TEXT, finished INTEGER DEFAULT 0)"
    )
    conn.commit()
    return conn


# ── main ─────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Train layout classifier from docling JSON crops")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--json-dir", help="Directory containing docling JSON files")
    src.add_argument("--json-files", nargs="+", help="Explicit list of JSON files")
    src.add_argument("--pdf-dir", help="Run docling --to json on all PDFs in directory first")
    parser.add_argument("--crops-dir", default="crops_layout", help="Output directory for crops")
    parser.add_argument("--output", default="layout_classifier.pt")
    parser.add_argument("--n-trials", type=int, default=15, help="Optuna trials")
    parser.add_argument("--subset-frac", type=float, default=0.3,
                        help="Fraction of data for Optuna sweep (0<f<1)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Step 0: Collect JSON files ──────────────────────────────────────────────
    if args.json_dir:
        json_files = sorted(Path(args.json_dir).glob("*.json"))
    elif args.json_files:
        json_files = [Path(p) for p in args.json_files]
    else:  # --pdf-dir: run docling first
        pdf_dir = Path(args.pdf_dir)
        json_files = []
        for pdf in sorted(pdf_dir.glob("*.pdf")) + sorted(pdf_dir.glob("*.PDF")):
            out_json = pdf.with_suffix(".json")
            if not out_json.exists():
                print(f"  Running docling on {pdf.name}...")
                subprocess.run(
                    ["docling", str(pdf), "--to", "json", "--output", str(pdf.parent)],
                    check=True, capture_output=True
                )
            if out_json.exists():
                json_files.append(out_json)

    if not json_files:
        print("No JSON files found.")
        return 1
    print(f"Processing {len(json_files)} JSON file(s)...")

    # ── Step 1: Generate crops ──────────────────────────────────────────────────
    crops_dir = Path(args.crops_dir)
    all_crops: list[tuple[Path, str]] = []
    for jf in json_files:
        print(f"  {jf.name}")
        extract_crops_from_json(jf, crops_dir, all_crops)

    if not all_crops:
        print("No crops extracted — ensure PDFs are beside JSON files.")
        return 1

    labels = [lbl for _, lbl in all_crops]
    class_counts = Counter(labels)
    classes = sorted(class_counts)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    print(f"\nClass distribution: {dict(class_counts)}")

    # ── Step 2: Box-Cox class weights ──────────────────────────────────────────
    weights = boxcox_class_weights(labels)
    print(f"Box-Cox weights: {weights}")
    w_tensor = torch.tensor(
        [weights.get(c, 1.0) for c in classes], dtype=torch.float32
    ).to(device)

    # ── Step 3: Dataset ─────────────────────────────────────────────────────────
    samples = [(p, class_to_idx[lbl]) for p, lbl in all_crops]
    dataset = CropDataset(samples)

    n = len(dataset)
    n_val = max(1, int(n * 0.15))
    n_test = max(1, int(n * 0.15))
    n_train = n - n_val - n_test
    assert n_train > 0, f"Not enough samples: {n}"

    generator = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test],
                                              generator=generator)
    print(f"Split: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    # ── Step 4: Optuna sweep on subset ─────────────────────────────────────────
    n_sub = max(8, int(len(train_ds) * args.subset_frac))
    sub_indices = list(range(len(train_ds)))[:n_sub]
    train_sub = Subset(train_ds, sub_indices[:int(n_sub * 0.8)])
    val_sub = Subset(train_ds, sub_indices[int(n_sub * 0.8):])

    db_path = Path(CHECKPOINT_DB)
    conn = _ensure_db(db_path)
    optuna_storage = f"sqlite:///{db_path}"
    study_name = "layout_classifier_optuna"

    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=optuna_storage,
        load_if_exists=True,
    )
    existing = len(study.trials)
    remaining = max(0, args.n_trials - existing)
    if remaining > 0:
        print(f"\nOptuna: running {remaining} trial(s) (already have {existing})...")
        study.optimize(
            make_objective(train_sub, val_sub, w_tensor, len(classes), device),
            n_trials=remaining,
            show_progress_bar=False,
        )

    best_params = study.best_params
    print(f"Best params: {best_params}  val_acc={study.best_value:.4f}")
    conn.execute(
        "INSERT OR REPLACE INTO training_runs(phase, best_acc, best_params) VALUES (?,?,?)",
        ("optuna", study.best_value, json.dumps(best_params)),
    )
    conn.commit()

    # ── Step 5: Full training ───────────────────────────────────────────────────
    bs = best_params.get("batch_size", 32)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False,
                            num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False,
                             num_workers=0, pin_memory=True)

    model = build_model(len(classes)).to(device)
    criterion = nn.CrossEntropyLoss(weight=w_tensor)
    print("\nFull training...")
    best_val_acc = run_training(
        model, train_loader, val_loader, criterion,
        lr=best_params["lr"],
        weight_decay=best_params["weight_decay"],
        max_epochs=100, patience=20, device=device,
    )
    print(f"  Best val acc: {best_val_acc:.4f}")

    # ── Step 6: Holdout evaluation ──────────────────────────────────────────────
    test_acc = eval_accuracy(model, test_loader, device)
    print(f"  Holdout test acc: {test_acc:.4f}")

    conn.execute(
        "INSERT OR REPLACE INTO training_runs(phase, best_acc, best_params, finished) VALUES (?,?,?,1)",
        ("full", test_acc, json.dumps(best_params)),
    )
    conn.commit()
    conn.close()

    # ── Step 7: Save ───────────────────────────────────────────────────────────
    out_path = Path(args.output)
    torch.save({"state_dict": model.state_dict(), "classes": classes}, str(out_path))
    classes_path = out_path.with_suffix("").parent / "layout_classifier_classes.json"
    classes_path.write_text(json.dumps({"idx_to_class": {str(i): c for i, c in enumerate(classes)},
                                         "class_to_idx": class_to_idx}, indent=2))
    print(f"\nSaved: {out_path}  ({len(classes)} classes)")
    print(f"Classes: {classes}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
