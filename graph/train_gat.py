"""
Train a GATv2 graph transformer on the KG triplet dataset.

Core Thesis:
    A 2-layer GATv2Conv with edge_dim (edge-aware attention) trained via
    link prediction learns to encode the hard ontology into its weights.
    At inference the model IS the graph: query → seed nodes → GATv2 →
    scored neighborhood → serialized context for LLM injection.

Architecture (GATv2, Brody et al. 2022):
    Unlike GAT (Velickovic 2018) which computes attention as:
        e_ij = a^T · LeakyReLU(W[h_i || h_j])          (static — keys fixed)
    GATv2 computes:
        e_ij = a^T · LeakyReLU(W_1·h_i + W_2·h_j)      (dynamic — joint transform)
    With edge_dim:
        e_ij = a^T · LeakyReLU(W_1·h_i + W_2·h_j + W_e·e_ij)
    The predicate embedding e_ij directly conditions the attention weight,
    which is exactly the semantic bridging described in the conversation:
    "the edges have to be semantically committed before synthesis."

Link Prediction Training:
    Given edge (u, v) with edge_attr e_uv:
        Positive: z_u ⊙ z_v dot-product → sigmoid → 1
        Negative: z_u ⊙ z_rand dot-product → sigmoid → 0
    Binary cross-entropy loss. Self-supervised — uses only the graph structure,
    no external labels needed.

Workflow:
    Load(kg_dataset.pt)
        → RandomLinkSplit (train/val/test)
        → Train(GATv2 model, link prediction loss, N epochs)
        → Validate(val_edge_index, AUC)
        → Save(gat_model.pt, model_config.json)

Dependencies:
    pip install torch torch-geometric scikit-learn tqdm

Usage:
    python graph/train_gat.py
    python graph/train_gat.py --epochs 30 --hidden 256 --heads 4
    python graph/train_gat.py --smoke-test   (5 epochs, tiny subset)
"""

import argparse
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

# ── Paths (absolute so the script is CWD-independent) ─────────────────────────
OUT_DIR      = Path(__file__).parent            # .../graph/
DATASET_PATH = OUT_DIR / "kg_dataset.pt"
MODEL_PATH   = OUT_DIR / "gat_model.pt"
CONFIG_PATH  = OUT_DIR / "model_config.json"

# ── Hyperparameters ───────────────────────────────────────────────────────────
DEFAULT_HIDDEN  = 256
DEFAULT_HEADS   = 4
DEFAULT_LAYERS  = 2
DEFAULT_DROPOUT = 0.1
DEFAULT_LR      = 1e-3
DEFAULT_EPOCHS  = 20
NEG_SAMPLE_RATIO = 1   # negatives per positive edge
EVAL_EVERY       = 5   # validate every N epochs

# ── Optuna tuning ─────────────────────────────────────────────────────────────
OPTUNA_EPOCHS       = 20    # max epochs per trial
OPTUNA_PATIENCE     = 5     # early-stop (eval-check units) during tuning
OPTUNA_SUBSET_RATIO = 0.30  # fraction of non-test edges used for tuning
DEFAULT_N_TRIALS    = 20    # number of Optuna trials

# ── Full training ─────────────────────────────────────────────────────────────
FULL_EPOCHS      = 150
FULL_PATIENCE    = 20   # eval-check units (×5 epochs each) = 100 epoch tolerance
DEFAULT_PATIENCE = FULL_PATIENCE   # backward compat alias


# ── Model ─────────────────────────────────────────────────────────────────────

class GATv2Encoder(torch.nn.Module):
    """
    2-layer GATv2 encoder with edge-feature-aware attention.

    P(S,[O]) expressed as GATv2:
        h_v^{(l+1)} = AGG_{u ∈ N(v)} α_{vu}^{(l)} · W_2^{(l)} · h_u^{(l)}
    where α is computed jointly over (h_v, h_u, e_vu) via dynamic attention.

    Input:
        x           (N, in_channels)      node features
        edge_index  (2, E)                directed edges
        edge_attr   (E, edge_dim)         predicate embeddings
    Output:
        z           (N, out_channels)     node representations
    """

    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int, heads: int = 4,
                 dropout: float = DEFAULT_DROPOUT,
                 edge_dim: int | None = None):
        super().__init__()
        from torch_geometric.nn import GATv2Conv, Linear

        self.dropout = dropout

        # Projection: in_channels → hidden
        self.lin_in = Linear(in_channels, hidden_channels)

        # Edge projection: edge embedding → same hidden dim as node
        # (declared before conv1 so conv1.edge_dim matches projected dim)
        self.edge_proj = torch.nn.Linear(edge_dim or hidden_channels, hidden_channels)

        # GATv2 layer 1: hidden → hidden (multi-head, concat)
        # edge_dim = hidden_channels because ea is projected BEFORE conv1
        self.conv1 = GATv2Conv(
            in_channels  = hidden_channels,
            out_channels = hidden_channels // heads,
            heads        = heads,
            concat       = True,
            dropout      = dropout,
            edge_dim     = hidden_channels,
            add_self_loops = True,
        )

        # GATv2 layer 2: hidden → out (single head, no concat)
        self.conv2 = GATv2Conv(
            in_channels  = hidden_channels,
            out_channels = out_channels,
            heads        = 1,
            concat       = False,
            dropout      = dropout,
            edge_dim     = hidden_channels,
            add_self_loops = True,
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        # Project inputs
        x    = F.elu(self.lin_in(x))
        ea   = F.elu(self.edge_proj(edge_attr))  # project edge embeddings

        # Layer 1
        x    = F.elu(self.conv1(x, edge_index, edge_attr=ea))
        x    = F.dropout(x, p=self.dropout, training=self.training)

        # Layer 2 (ea already projected; edge_index/attr same for both layers)
        x    = self.conv2(x, edge_index, edge_attr=ea)

        return x  # (N, out_channels) — node representations


class LinkPredictor(torch.nn.Module):
    """
    Dot-product link predictor with optional MLP.
    Score(u,v) = sigmoid(z_u · z_v) — classic bilinear link pred.
    EdgeScore(u,v,e) = sigmoid(z_u · e_proj(e_uv) · z_v) includes predicate.
    """

    def forward(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z           (N, D)   node embeddings from GATv2
            edge_index  (2, E)   [src; dst] pairs to score
        Returns:
            scores (E,)  raw logits (apply sigmoid for prob)
        """
        src = z[edge_index[0]]   # (E, D)
        dst = z[edge_index[1]]   # (E, D)
        return (src * dst).sum(dim=-1)   # (E,)


# ── Negative sampling ─────────────────────────────────────────────────────────

def sample_negatives(edge_index: torch.Tensor, num_nodes: int,
                     ratio: int = 1,
                     nodes: torch.Tensor | None = None) -> torch.Tensor:
    """
    Randomly sample negative edges (non-existing pairs).
    If `nodes` is provided, sample only from that subset (connected nodes).
    Best practice: exclude isolated nodes — they have no edges to predict.
    """
    n_pos  = edge_index.shape[1]
    n_neg  = n_pos * ratio
    if nodes is not None:
        pool = nodes.to(edge_index.device)
        idx  = torch.randint(0, len(pool), (n_neg,), device=edge_index.device)
        neg_src = pool[idx]
        idx  = torch.randint(0, len(pool), (n_neg,), device=edge_index.device)
        neg_dst = pool[idx]
    else:
        neg_src = torch.randint(0, num_nodes, (n_neg,), device=edge_index.device)
        neg_dst = torch.randint(0, num_nodes, (n_neg,), device=edge_index.device)
    return torch.stack([neg_src, neg_dst], dim=0)


# ── Train / eval ──────────────────────────────────────────────────────────────

def train_epoch(model: GATv2Encoder, predictor: LinkPredictor,
                data, optimizer: torch.optim.Optimizer,
                neg_ratio: int = NEG_SAMPLE_RATIO) -> float:
    model.train()
    predictor.train()

    # Full-graph forward pass
    z = model(data.x, data.train_pos_edge_index, data.train_pos_edge_attr)

    # Positive scores
    pos_score = predictor(z, data.train_pos_edge_index)

    # Negative sampling — restrict to connected nodes if stored on data object
    _nodes   = getattr(data, 'connected_nodes', None)
    neg_edge  = sample_negatives(data.train_pos_edge_index, data.num_nodes, neg_ratio, nodes=_nodes)
    neg_score = predictor(z, neg_edge)

    # BCE loss
    pos_label = torch.ones_like(pos_score)
    neg_label = torch.zeros_like(neg_score)
    scores    = torch.cat([pos_score, neg_score])
    labels    = torch.cat([pos_label, neg_label])

    loss = F.binary_cross_entropy_with_logits(scores, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def evaluate(model: GATv2Encoder, predictor: LinkPredictor,
             data, pos_edge: torch.Tensor, neg_edge: torch.Tensor) -> float:
    """Compute AUC over given positive/negative edges."""
    from sklearn.metrics import roc_auc_score

    model.eval()
    predictor.eval()

    z = model(data.x, data.train_pos_edge_index, data.train_pos_edge_attr)

    pos_score = torch.sigmoid(predictor(z, pos_edge)).cpu().numpy()
    neg_score = torch.sigmoid(predictor(z, neg_edge)).cpu().numpy()

    scores = np.concatenate([pos_score, neg_score])
    labels = np.concatenate([np.ones(len(pos_score)), np.zeros(len(neg_score))])
    return roc_auc_score(labels, scores)


# ── Data split ────────────────────────────────────────────────────────────────

def split_edges(data, val_ratio: float = 0.05, test_ratio: float = 0.1):
    """
    Manual train/val/test split of edges.
    RandomLinkSplit from PyG also works but produces a more complex object;
    manual split is more transparent.
    """
    E    = data.edge_index.shape[1]
    perm = torch.randperm(E)

    n_test = int(E * test_ratio)
    n_val  = int(E * val_ratio)
    n_train= E - n_val - n_test

    train_idx = perm[:n_train]
    val_idx   = perm[n_train:n_train + n_val]
    test_idx  = perm[n_train + n_val:]

    data.train_pos_edge_index = data.edge_index[:, train_idx]
    data.train_pos_edge_attr  = data.edge_attr[train_idx]
    data.val_pos_edge_index   = data.edge_index[:, val_idx]
    data.test_pos_edge_index  = data.edge_index[:, test_idx]

    # Pre-sample val/test negatives (fixed for consistent reporting)
    data.val_neg_edge_index  = sample_negatives(data.val_pos_edge_index,  data.num_nodes)
    data.test_neg_edge_index = sample_negatives(data.test_pos_edge_index, data.num_nodes)

    print(f"  Train edges: {n_train:,} | Val: {n_val:,} | Test: {n_test:,}")
    return data


# ── Smoke test subset ─────────────────────────────────────────────────────────

def get_smoke_subset(data, max_edges: int = 10_000):
    """Take a small connected subgraph for smoke testing."""
    E     = data.edge_index.shape[1]
    idx   = torch.randperm(E)[:max_edges]
    sub   = data.clone()
    sub.edge_index = data.edge_index[:, idx]
    sub.edge_attr  = data.edge_attr[idx]
    sub.edge_meta  = data.edge_meta[idx]
    # Remap nodes to only those present
    nodes  = sub.edge_index.unique()
    n_new  = len(nodes)
    mapping = torch.full((data.num_nodes,), -1, dtype=torch.long)
    mapping[nodes] = torch.arange(n_new)
    sub.edge_index = mapping[sub.edge_index]
    sub.x          = data.x[nodes]
    sub.num_nodes  = n_new
    print(f"  Smoke subset: {n_new:,} nodes, {max_edges:,} edges")
    return sub


# ── Training helpers ──────────────────────────────────────────────────────────

def build_model(in_dim: int, edge_dim: int, hidden: int, heads: int,
                dropout: float, device: str):
    """Instantiate GATv2Encoder + LinkPredictor and move to device."""
    model = GATv2Encoder(
        in_channels     = in_dim,
        hidden_channels = hidden,
        out_channels    = hidden,
        heads           = heads,
        dropout         = dropout,
        edge_dim        = edge_dim,
    ).to(device)
    predictor = LinkPredictor().to(device)
    return model, predictor


def run_training_loop(model, predictor, data, optimizer, scheduler,
                      epochs: int, patience: int,
                      verbose: bool = False) -> tuple:
    """
    Train with early stopping; return (best_val_auc, best_state).
    verbose=True prints per-epoch progress (used for the full training run).
    """
    best_val_auc = 0.0
    best_state   = None
    no_improve   = 0

    for epoch in range(1, epochs + 1):
        t0   = time.time()
        loss = train_epoch(model, predictor, data, optimizer)
        scheduler.step()

        if epoch % EVAL_EVERY == 0 or epoch == epochs:
            val_auc = evaluate(model, predictor, data,
                               data.val_pos_edge_index,
                               data.val_neg_edge_index)
            elapsed = time.time() - t0

            if verbose:
                print(f"  Epoch {epoch:3d}/{epochs}  loss={loss:.4f}  val_auc={val_auc:.4f}  {elapsed:.1f}s")

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                no_improve   = 0
                best_state   = {
                    "model":     {k: v.cpu().clone() for k, v in model.state_dict().items()},
                    "predictor": {k: v.cpu().clone() for k, v in predictor.state_dict().items()},
                }
            else:
                no_improve += 1
                if patience > 0 and no_improve >= patience:
                    if verbose:
                        print(f"  Early stop (no improvement for {patience} eval checks)")
                    break
        else:
            if verbose:
                elapsed = time.time() - t0
                print(f"  Epoch {epoch:3d}/{epochs}  loss={loss:.4f}  {elapsed:.1f}s")

    return best_val_auc, best_state


def run_optuna(data_x: torch.Tensor, edge_index: torch.Tensor,
               edge_attr: torch.Tensor, in_dim: int, edge_dim: int,
               num_nodes: int, device: str,
               n_trials: int, subset_ratio: float,
               connected_nodes: torch.Tensor | None = None) -> dict:
    """
    Tune hyperparameters with Optuna on a small edge subset.

    Workflow:
        Sample(subset_ratio * non_test_edges) as tuning pool
        For each trial:
            split(tuning_pool, train=0.85, val=0.15)
            Train(max_epochs=OPTUNA_EPOCHS, patience=OPTUNA_PATIENCE)
            Maximize(val_auc)
        Return best_params dict.
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("WARNING: optuna not installed -- skipping tuning, using defaults.")
        print("  pip install optuna")
        return {}

    from torch_geometric.data import Data as _Data

    E      = edge_index.shape[1]
    n_sub  = max(20, int(E * subset_ratio))
    perm   = torch.randperm(E, generator=torch.Generator().manual_seed(99))[:n_sub]
    sub_ei = edge_index[:, perm]
    sub_ea = edge_attr[perm]

    def objective(trial):
        hidden  = trial.suggest_categorical("hidden",  [64, 128, 256])
        heads   = trial.suggest_categorical("heads",   [2, 4])
        dropout = trial.suggest_float("dropout", 0.05, 0.40, step=0.05)
        lr      = trial.suggest_float("lr",      1e-4, 1e-2,  log=True)

        n     = sub_ei.shape[1]
        # Seed per-trial so val split and model init are reproducible across runs.
        # Different trials get different splits (trial.number varies), ensuring
        # Optuna still explores diverse configurations rather than seeing same data.
        _trial_seed = trial.number * 137 + 7
        p     = torch.randperm(n, generator=torch.Generator().manual_seed(_trial_seed))
        n_val = max(10, int(n * 0.35))  # 35% val → ~45 edges (vs 15%/~19): more stable signal
        tr, v = p[:n - n_val], p[n - n_val:]

        torch.manual_seed(trial.number * 37 + 1)  # seed neg sampling + weight init
        td = _Data()
        td.num_nodes            = num_nodes
        td.x                    = data_x.to(device)
        td.train_pos_edge_index = sub_ei[:, tr].to(device)
        td.train_pos_edge_attr  = sub_ea[tr].to(device)
        td.val_pos_edge_index   = sub_ei[:, v].to(device)
        td.val_neg_edge_index   = sample_negatives(sub_ei[:, v], num_nodes, nodes=connected_nodes).to(device)
        if connected_nodes is not None:
            td.connected_nodes  = connected_nodes.to(device)

        m, pred = build_model(in_dim, edge_dim, hidden, heads, dropout, device)
        opt     = torch.optim.Adam(
            list(m.parameters()) + list(pred.parameters()), lr=lr
        )
        sched   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=OPTUNA_EPOCHS)

        val_auc, _ = run_training_loop(m, pred, td, opt, sched,
                                       OPTUNA_EPOCHS, OPTUNA_PATIENCE,
                                       verbose=False)
        return val_auc

    print(f"\nOptuna: {n_trials} trials on {n_sub:,}/{E:,} edges  "
          f"(patience={OPTUNA_PATIENCE}, max_epochs={OPTUNA_EPOCHS})...")
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    bp = study.best_params
    print(f"  Best trial val AUC : {study.best_value:.4f}")
    print(f"  Best params        : {bp}")
    return bp


# ── Main ──────────────────────────────────────────────────────────────────────

def main(n_trials: int = DEFAULT_N_TRIALS,
         holdout_ratio: float = 0.15,
         optuna_subset: float = OPTUNA_SUBSET_RATIO,
         full_epochs: int = FULL_EPOCHS,
         full_patience: int = FULL_PATIENCE,
         smoke_test: bool = False):
    """
    Workflow:
        1. Load KG dataset
        2. Carve permanent holdout test set (seeded, never touched during tuning)
        3. Optuna on small subset of remaining edges  ->  best hyperparams
        4. Full retrain with best params on all non-test edges
        5. Evaluate on holdout  ->  final test AUC
        6. Save model + config
    """
    try:
        from torch_geometric.data import Data
    except ImportError:
        print("ERROR: torch_geometric not installed. pip install torch-geometric")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    assert DATASET_PATH.exists(), f"Run build_kg_dataset.py first (missing {DATASET_PATH})"

    print(f"Loading dataset from {DATASET_PATH}...")
    data = torch.load(DATASET_PATH, weights_only=False)
    print(f"  Nodes: {data.num_nodes:,}  Edges: {data.edge_index.shape[1]:,}")
    print(f"  x: {data.x.shape}  edge_attr: {data.edge_attr.shape}")

    in_dim   = data.x.shape[1]
    edge_dim = data.edge_attr.shape[1]
    N        = data.num_nodes

    if smoke_test:
        print("Smoke test: subsetting to 200 edges...")
        data = get_smoke_subset(data, max_edges=200)
        N    = data.num_nodes

    # Restrict negative sampling to nodes that actually participate in edges.
    # Isolated nodes (71% here) have zero training signal — sampling them as
    # negatives floods the loss with random noise and deflates AUC artificially.
    connected_nodes = torch.unique(data.edge_index).cpu()
    print(f"  Connected nodes: {len(connected_nodes):,} / {N:,}  "
          f"(isolated: {N - len(connected_nodes):,} excluded from neg sampling)")

    # ── 1. Permanent holdout split ─────────────────────────────────────────────
    # Seeded for reproducibility; these edges are NEVER seen during tuning.
    gen          = torch.Generator().manual_seed(42)
    E            = data.edge_index.shape[1]
    perm         = torch.randperm(E, generator=gen)
    n_test       = max(1, int(E * holdout_ratio))
    test_idx     = perm[:n_test]
    non_test_idx = perm[n_test:]

    test_ei  = data.edge_index[:, test_idx]
    torch.manual_seed(42)  # seed holdout negatives — same negatives every run
    test_neg = sample_negatives(test_ei, N, nodes=connected_nodes)

    nt_ei = data.edge_index[:, non_test_idx]
    nt_ea = data.edge_attr[non_test_idx]

    print(f"\nEdge split -- non-test (tuning+train): {len(non_test_idx):,}  "
          f"holdout: {n_test:,}")

    # ── 2. Optuna tuning ───────────────────────────────────────────────────────
    if n_trials > 0:
        best_params = run_optuna(
            data_x          = data.x,
            edge_index      = nt_ei,
            edge_attr       = nt_ea,
            in_dim          = in_dim,
            edge_dim        = edge_dim,
            num_nodes       = N,
            device          = device,
            n_trials        = n_trials,
            subset_ratio    = optuna_subset,
            connected_nodes = connected_nodes,
        )
    else:
        print("\nOptuna skipped (--trials 0).")
        best_params = {}

    hidden  = best_params.get("hidden",  DEFAULT_HIDDEN)
    heads   = best_params.get("heads",   DEFAULT_HEADS)
    dropout = best_params.get("dropout", DEFAULT_DROPOUT)
    lr      = best_params.get("lr",      DEFAULT_LR)

    # ── 3. Full training on all non-test edges ─────────────────────────────────
    n_val   = max(1, int(len(non_test_idx) * 0.15))
    n_tr    = len(non_test_idx) - n_val
    ft_perm = torch.randperm(len(non_test_idx),
                             generator=torch.Generator().manual_seed(0))
    tr_idx  = ft_perm[:n_tr]
    val_idx = ft_perm[n_tr:]

    torch.manual_seed(0)  # seed val neg sampling + model weight init
    full_d = Data()
    full_d.num_nodes            = N
    full_d.x                    = data.x.to(device)
    full_d.train_pos_edge_index = nt_ei[:, tr_idx].to(device)
    full_d.train_pos_edge_attr  = nt_ea[tr_idx].to(device)
    full_d.val_pos_edge_index   = nt_ei[:, val_idx].to(device)
    full_d.val_neg_edge_index   = sample_negatives(
        nt_ei[:, val_idx], N, nodes=connected_nodes).to(device)
    full_d.connected_nodes      = connected_nodes.to(device)

    print(f"\nFull training -- hidden={hidden} heads={heads} "
          f"dropout={dropout:.3f} lr={lr:.2e}")
    print(f"  Train edges: {n_tr:,}  Val edges: {n_val:,}")

    model, predictor = build_model(in_dim, edge_dim, hidden, heads, dropout, device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model params: {n_params:,}")

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(predictor.parameters()), lr=lr
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=full_epochs
    )

    print(f"  Training {full_epochs} epochs "
          f"(patience={full_patience} checks = {full_patience * EVAL_EVERY} epochs)...\n")

    best_val_auc, best_state = run_training_loop(
        model, predictor, full_d, optimizer, scheduler,
        full_epochs, full_patience, verbose=True,
    )

    # ── 4. Holdout evaluation ──────────────────────────────────────────────────
    if best_state:
        model.load_state_dict(best_state["model"])
        predictor.load_state_dict(best_state["predictor"])
    model = model.to(device)

    test_auc = evaluate(model, predictor, full_d,
                        test_ei.to(device), test_neg.to(device))

    print(f"\nBest val AUC : {best_val_auc:.4f}")
    print(f"Holdout AUC  : {test_auc:.4f}")

    # ── 5. Save ────────────────────────────────────────────────────────────────
    print("\nSaving model...")
    config = {
        "in_channels":        in_dim,
        "hidden_channels":    hidden,
        "out_channels":       hidden,
        "heads":              heads,
        "dropout":            dropout,
        "edge_dim":           edge_dim,
        "val_auc":            best_val_auc,
        "test_auc":           test_auc,
        "full_epochs":        full_epochs,
        "optuna_trials":      n_trials,
        "best_optuna_params": best_params,
    }
    torch.save(
        {
            "model":     model.state_dict(),
            "predictor": predictor.state_dict(),
            "config":    config,
        },
        MODEL_PATH,
    )
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)

    print(f"  Model  : {MODEL_PATH}")
    print(f"  Config : {CONFIG_PATH}")
    print(f"  Holdout AUC: {test_auc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train GATv2 with Optuna tuning + full retrain + holdout eval"
    )
    parser.add_argument("--trials",        type=int,   default=DEFAULT_N_TRIALS,
                        help="Optuna trials (0=skip tuning, use default hparams)")
    parser.add_argument("--optuna-subset", type=float, default=OPTUNA_SUBSET_RATIO,
                        help="Fraction of non-test edges for Optuna (default 0.30)")
    parser.add_argument("--full-epochs",   type=int,   default=FULL_EPOCHS,
                        help="Max epochs for full training run")
    parser.add_argument("--full-patience", type=int,   default=FULL_PATIENCE,
                        help="Early-stop patience (eval-check units) for full run")
    parser.add_argument("--holdout-ratio", type=float, default=0.15,
                        help="Fraction of edges held out as final test set")
    parser.add_argument("--smoke-test",    action="store_true",
                        help="Quick sanity check on 200-edge subset")
    args = parser.parse_args()
    main(
        n_trials      = args.trials,
        holdout_ratio = args.holdout_ratio,
        optuna_subset = args.optuna_subset,
        full_epochs   = args.full_epochs,
        full_patience = args.full_patience,
        smoke_test    = args.smoke_test,
    )
