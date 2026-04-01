"""
Build PyTorch Geometric dataset from normalized KG triplets.

Core Thesis:
    Converts normalized (entity_id_subj, predicate_id, entity_id_obj) triplets
    into a PyG homogeneous graph where:
        - Nodes  = entities,    node features = entity embeddings (N_e, D)
        - Edges  = triplets,    edge features = predicate embeddings (N_t, D)
    This is the native substrate for GATv2Conv with edge_dim — attention over
    (source_node, edge_attr, target_node) triples rather than source/target only.

    The predicate embedding as edge_attr IS the hard ontology: typed relational
    structure derived from source context, encoded into the attention computation.

Graph format:
    PyG Data (homogeneous)
        data.x           = entity_embs          float32 (N_e, D)
        data.edge_index  = [src_ids; dst_ids]   long    (2, N_t)
        data.edge_attr   = predicate_embs[pids] float32 (N_t, D)
        data.edge_meta   = [eid_s, pid, eid_o]  long    (N_t, 3)  ← for retrieval

    data.edge_meta stores the (entity_subj, predicate, entity_obj) triple ids
    so the retriever can reconstruct the original text from ids.

Dependencies:
    pip install torch torch-geometric msgpack numpy tqdm

Usage:
    python graph/build_kg_dataset.py
    python graph/build_kg_dataset.py --max-triplets 500000
"""

import argparse
import json
import pickle
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

# ── Paths (absolute so the script is CWD-independent) ─────────────────────────
OUT_DIR       = Path(__file__).parent           # .../graph/
ENTITY_IDX    = OUT_DIR / "entity_index.json"
ENTITY_EMB    = OUT_DIR / "entity_embeddings.npy"
PREDICATE_IDX = OUT_DIR / "predicate_index.json"
PREDICATE_EMB = OUT_DIR / "predicate_embeddings.npy"
TRIPLET_MAP   = OUT_DIR / "triplet_map.pkl"
DATASET_PATH  = OUT_DIR / "kg_dataset.pt"
VOCAB_PATH          = OUT_DIR / "vocab.json"   # id → surface form (for retrieval text reconstruction)
ENTITY_SYNSET_FEATS = OUT_DIR / "entity_synset_feats.npy"   # (N_entities, 6) auxiliary WN features

MSGPACK_PATH  = Path(__file__).parent.parent / "checkpoints/bio_triplets_full_corpus.msgpack"


def load_artifacts():
    """Load all normalized artifacts from normalize_entities.py."""
    assert ENTITY_IDX.exists(),    f"Run normalize_entities.py first (missing {ENTITY_IDX})"
    assert PREDICATE_IDX.exists(), f"Run normalize_entities.py first (missing {PREDICATE_IDX})"
    assert TRIPLET_MAP.exists(),   f"Run normalize_entities.py first (missing {TRIPLET_MAP})"

    print("Loading entity index...")
    with open(ENTITY_IDX) as f:
        entity_data = json.load(f)
    entity_embs = np.load(ENTITY_EMB)

    print("Loading predicate index...")
    with open(PREDICATE_IDX) as f:
        pred_data = json.load(f)
    pred_embs = np.load(PREDICATE_EMB)

    print("Loading triplet map...")
    with open(TRIPLET_MAP, "rb") as f:
        triplet_map = pickle.load(f)

    return entity_data, entity_embs, pred_data, pred_embs, triplet_map


def build_reverse_vocab(span_to_id: dict) -> dict:
    """Invert span→id to id→canonical_span. For equal-id spans, prefer shorter canonical."""
    id_to_span = {}
    for span, sid in span_to_id.items():
        if sid not in id_to_span:
            id_to_span[sid] = span
        else:
            # Keep the entry that came first (highest frequency, since Counter inserts in freq order)
            pass
    return id_to_span


def collect_all_triplets(triplet_map: dict,
                         max_triplets: int | None = None) -> list[tuple]:
    """
    Flatten {chunk_id: [(eid_s, pid, eid_o), ...]} → [(eid_s, pid, eid_o), ...]
    Deduplicates edges (same (eid_s, eid_o) pair, keep all predicates).
    """
    seen   = set()
    result = []
    for chunk_id, triples in tqdm(triplet_map.items(), desc="  Collecting", leave=False):
        for t in triples:
            key = t  # (eid_s, pid, eid_o) — all three form the unique edge
            if key not in seen:
                seen.add(key)
                result.append(t)
            if max_triplets and len(result) >= max_triplets:
                print(f"  [truncated at {max_triplets:,}]")
                return result
    return result


def build_pyg_data(entity_embs: np.ndarray,
                   pred_embs:   np.ndarray,
                   all_triplets: list[tuple]) -> "torch_geometric.data.Data":
    """
    Construct PyG Data object.

    Node features  : entity embeddings (already normalized from clustering step)
    Edge index     : directed src→dst from triplets
    Edge attributes: predicate embeddings looked up by predicate_id
    Edge metadata  : (eid_s, pid, eid_o) for text reconstruction at retrieval time
    """
    from torch_geometric.data import Data

    n_triplets = len(all_triplets)
    print(f"  Building PyG graph: {len(entity_embs):,} nodes, {n_triplets:,} edges...")

    eid_s_arr = np.array([t[0] for t in all_triplets], dtype=np.int64)
    pid_arr   = np.array([t[1] for t in all_triplets], dtype=np.int64)
    eid_o_arr = np.array([t[2] for t in all_triplets], dtype=np.int64)

    x          = torch.from_numpy(entity_embs)              # (N_e, D)
    edge_index = torch.from_numpy(                           # (2, N_t)
        np.stack([eid_s_arr, eid_o_arr], axis=0)
    )
    edge_attr  = torch.from_numpy(pred_embs[pid_arr])       # (N_t, D)
    edge_meta  = torch.from_numpy(                          # (N_t, 3)
        np.stack([eid_s_arr, pid_arr, eid_o_arr], axis=1)
    )

    data = Data(
        x          = x,
        edge_index = edge_index,
        edge_attr  = edge_attr,
        edge_meta  = edge_meta,
        num_nodes  = len(entity_embs),
    )

    print(f"  x:          {data.x.shape}  dtype={data.x.dtype}")
    print(f"  edge_index: {data.edge_index.shape}")
    print(f"  edge_attr:  {data.edge_attr.shape}  dtype={data.edge_attr.dtype}")
    print(f"  edge_meta:  {data.edge_meta.shape}")

    return data


def build_vocab(entity_span_to_id: dict, pred_span_to_id: dict) -> dict:
    """Build id→text vocab for both entities and predicates (for retrieval serialization)."""
    entity_id_to_text = build_reverse_vocab(entity_span_to_id)
    pred_id_to_text   = build_reverse_vocab(pred_span_to_id)
    return {
        "entity":    {str(k): v for k, v in entity_id_to_text.items()},
        "predicate": {str(k): v for k, v in pred_id_to_text.items()},
    }


def print_sample_triplets(all_triplets: list, entity_id_to_text: dict,
                          pred_id_to_text: dict, n: int = 10):
    print(f"\n  Sample triplets:")
    for eid_s, pid, eid_o in all_triplets[:n]:
        s = entity_id_to_text.get(eid_s, f"[{eid_s}]")
        p = pred_id_to_text.get(pid,   f"[{pid}]")
        o = entity_id_to_text.get(eid_o, f"[{eid_o}]")
        print(f"    ({s}) --[{p}]--> ({o})")


def main(max_triplets: int | None = None):
    try:
        import torch_geometric
    except ImportError:
        print("ERROR: torch_geometric not installed.")
        print("  pip install torch-geometric")
        print("  or: pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.x.x+cu12x.html")
        return

    (entity_data, entity_embs, pred_data, pred_embs, triplet_map) = load_artifacts()

    entity_span_to_id = entity_data["span_to_id"]
    pred_span_to_id   = pred_data["span_to_id"]
    print(f"  Entities: {len(entity_embs):,} nodes, emb_dim={entity_embs.shape[1]}")
    print(f"  Predicates: {len(pred_embs):,} types, emb_dim={pred_embs.shape[1]}")
    print(f"  Chunks with triplets: {len(triplet_map):,}")

    # Optional: augment entity node features with WordNet auxiliary features
    if ENTITY_SYNSET_FEATS.exists():
        syn_feats = np.load(ENTITY_SYNSET_FEATS)
        if syn_feats.shape[0] == entity_embs.shape[0]:
            entity_embs = np.concatenate([entity_embs, syn_feats], axis=1)
            print(f"  Augmented entity embs with WN features: {entity_embs.shape} "
                  f"(+{syn_feats.shape[1]} dims: lexname_norm + synset_norm + hypernym_norm)")
        else:
            print(f"  WARNING: WN feats node count mismatch "
                  f"({syn_feats.shape[0]} vs {entity_embs.shape[0]}), skipping augmentation")
    else:
        print("  No WN feature file found — using raw embeddings only")

    print("Collecting all triplets (deduplicating)...")
    all_triplets = collect_all_triplets(triplet_map, max_triplets)
    print(f"  Unique (eid_s, pid, eid_o) edges: {len(all_triplets):,}")

    # Print samples using reverse vocab
    entity_id_to_text = build_reverse_vocab(entity_span_to_id)
    pred_id_to_text   = build_reverse_vocab(pred_span_to_id)
    print_sample_triplets(all_triplets, entity_id_to_text, pred_id_to_text)

    print("\nBuilding PyG dataset...")
    data = build_pyg_data(entity_embs, pred_embs, all_triplets)

    print("Saving vocab...")
    vocab = build_vocab(entity_span_to_id, pred_span_to_id)
    with open(VOCAB_PATH, "w") as f:
        json.dump(vocab, f)

    print("Saving dataset...")
    torch.save(data, DATASET_PATH)

    print(f"\n✓ Done")
    print(f"  Dataset:  {DATASET_PATH}  ({DATASET_PATH.stat().st_size // 1024 // 1024} MB)")
    print(f"  Vocab:    {VOCAB_PATH}")
    print(f"  Nodes:    {data.num_nodes:,}")
    print(f"  Edges:    {data.edge_index.shape[1]:,}")
    print(f"  Feat dim: {data.x.shape[1]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-triplets", type=int, default=None,
                        help="Cap total edges (useful for quick testing)")
    args = parser.parse_args()
    main(max_triplets=args.max_triplets)
