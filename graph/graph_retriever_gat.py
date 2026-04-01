"""
Graph Transformer retriever: query → GATv2 neighborhood scoring → prompt context.

Core Thesis:
    The trained GATv2 model encodes the hard ontology into its weights.
    At query time:
        1. Embed query text with the same static model used during construction
        2. Find seed entity nodes via cosine similarity (semantic entry point)
        3. Extract their 1-hop neighborhood subgraph (ontological expansion)
        4. Run GATv2 on the subgraph → updated node representations
        5. Score each triplet (edge) by: sim(query, subject) + sim(query, object)
           using the GATv2-updated representations (not raw embeddings)
        6. Return top-k triplets serialized as: "subject | predicate | object"

    The LLM then receives ontologically-structured, semantically-ranked context
    rather than cosine-similar text chunks. This is the "1-degree co-occurrence
    that happens as a result from the nodes and edges" with graph structure
    conditioning the relevance scoring.

Usage:
    # In pipeline / prompt construction:
    retriever = GraphRetriever()
    context   = retriever.retrieve_context("attention mechanism in transformers", top_k=15)
    prompt    = f"Context:\\n{context}\\n\\nQuestion: ..."

    # Or get raw triplets:
    triplets = retriever.retrieve("attention mechanism in transformers", top_k=15)
    for subj, pred, obj in triplets:
        print(f"  {subj} --[{pred}]--> {obj}")

    # Integration with existing pgvector_retriever:
    # After L1 retrieval, use L2 graph expansion to enrich context:
    graph_context = retriever.retrieve_for_seeds(seed_texts, top_k=20)
"""

import sys
import json
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Optional

# ── Stopword guard (mirrors normalize_entities.ENTITY_STOPWORDS) ─────────────
_STOPWORDS: frozenset = frozenset({
    "a", "an", "the", "this", "that", "these", "those",
    "i", "me", "my", "we", "our", "you", "your",
    "he", "him", "his", "she", "her", "it", "its", "they", "them", "their",
    "is", "are", "was", "were", "be", "been", "being",
    "and", "or", "but", "so", "yet", "nor", "for",
    "in", "on", "at", "to", "of", "by", "as", "up", "out", "if", "into",
    "more", "less", "most", "least", "very", "much",
    "many", "some", "any", "all", "both", "each", "few",
    "another", "other", "such",
    "just", "also", "even", "still", "back", "now", "then",
    "here", "there", "not", "no", "only", "well", "too",
    "normal", "good", "great", "new", "old", "long", "right", "big", "same",
    "one", "someone", "anyone", "everyone", "no one", "nobody", "somebody",
    "anybody", "nothing", "something", "anything", "everything",
    "what", "which", "who", "whom", "whose", "where", "when", "why", "how",
})

def _is_content(text: str) -> bool:
    """Return True iff at least one token is not a stopword."""
    return any(tok not in _STOPWORDS for tok in text.lower().split())

# ── sys.path: ensure workspace root is importable whether running as script or module ──
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Import model definition (same package)
from graph.train_gat import GATv2Encoder  # noqa: E402

# ── Paths (absolute, derived from __file__) ───────────────────────────────────
GRAPH_DIR      = Path(__file__).parent          # .../graph/
ENTITY_IDX     = GRAPH_DIR / "entity_index.json"
ENTITY_EMB     = GRAPH_DIR / "entity_embeddings.npy"
PREDICATE_IDX  = GRAPH_DIR / "predicate_index.json"
PREDICATE_EMB  = GRAPH_DIR / "predicate_embeddings.npy"
TRIPLET_MAP    = GRAPH_DIR / "triplet_map.pkl"
MODEL_PATH     = GRAPH_DIR / "gat_model.pt"
CONFIG_PATH    = GRAPH_DIR / "model_config.json"
VOCAB_PATH     = GRAPH_DIR / "vocab.json"

EMBEDDER_PATH  = str(_ROOT / "model2vec_jina")  # absolute path


class GraphRetriever:
    """
    Graph Transformer retriever.

    Separates concerns per FOL design:
        Entity embeddings  (static)   = semantic proximity
        GATv2 model        (trained)  = ontological relevance scoring
        Triplet map        (lookup)   = text reconstruction for prompt injection

    Attributes:
        embedder:        static text encoder (matches construction-time encoder)
        entity_embs:     float32 (N_e, D)  — L2-normalized entity node embeddings
        predicate_embs:  float32 (N_p, D)  — L2-normalized predicate embeddings
        entity_to_id:    {span: node_id}
        predicate_to_id: {span: node_id}
        vocab:           {entity: {id: text}, predicate: {id: text}}
        adj:             {entity_id: [(eid_s, pid, eid_o), ...]}  — adjacency list
        model:           GATv2Encoder (trained, eval mode)
        device:          torch device
    """

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._load_artifacts()
        self._build_adjacency()
        self._load_model()
        print(f"GraphRetriever ready on {self.device}")

    # ── Loading ───────────────────────────────────────────────────────────────

    def _load_artifacts(self):
        assert MODEL_PATH.exists(), f"Train model first: python graph/train_gat.py  (missing {MODEL_PATH})"

        print("Loading entity index & embeddings...")
        with open(ENTITY_IDX) as f:
            entity_data = json.load(f)
        self.entity_to_id  = entity_data["span_to_id"]
        self.entity_embs   = np.load(ENTITY_EMB).astype(np.float32)
        norms              = np.linalg.norm(self.entity_embs, axis=1, keepdims=True)
        self.entity_embs_n = self.entity_embs / (norms + 1e-9)   # L2 normalized (512-dim, for query cosine)
        # Augment with WN synset features for GAT input (entity_embs_n stays 512-dim for query cosine)
        _syn_path = GRAPH_DIR / "entity_synset_feats.npy"
        if _syn_path.exists():
            _syn = np.load(_syn_path).astype(np.float32)
            if _syn.shape[0] == self.entity_embs.shape[0]:
                self.entity_embs_gat = np.concatenate([self.entity_embs, _syn], axis=1)
                print(f"  Augmented entity embs for GAT: {self.entity_embs_gat.shape} (+{_syn.shape[1]} WN dims)")
            else:
                self.entity_embs_gat = self.entity_embs
        else:
            self.entity_embs_gat = self.entity_embs

        print("Loading predicate index & embeddings...")
        with open(PREDICATE_IDX) as f:
            pred_data = json.load(f)
        self.predicate_to_id = pred_data["span_to_id"]
        self.pred_embs       = np.load(PREDICATE_EMB).astype(np.float32)

        print("Loading vocab...")
        with open(VOCAB_PATH) as f:
            self.vocab = json.load(f)
        self.entity_id_to_text   = {int(k): v for k, v in self.vocab["entity"].items()}
        self.pred_id_to_text     = {int(k): v for k, v in self.vocab["predicate"].items()}

        print("Loading triplet map...")
        with open(TRIPLET_MAP, "rb") as f:
            self.triplet_map = pickle.load(f)

    def _build_adjacency(self):
        """
        Build {entity_id: [(eid_s, pid, eid_o), ...]} from triplet_map.
        Both subject and object nodes index into the same adjacency list so
        1-hop retrieval works from either end.
        """
        print("Building adjacency index...")
        adj: dict[int, list] = {}
        for chunk_triples in self.triplet_map.values():
            for eid_s, pid, eid_o in chunk_triples:
                adj.setdefault(eid_s, []).append((eid_s, pid, eid_o))
                adj.setdefault(eid_o, []).append((eid_s, pid, eid_o))
        self.adj = adj
        print(f"  Adjacency entries: {len(adj):,}")

    def _load_model(self):
        print(f"Loading GATv2 model from {MODEL_PATH}...")
        checkpoint = torch.load(MODEL_PATH, map_location=self.device, weights_only=False)
        config     = checkpoint["config"]

        self.model = GATv2Encoder(
            in_channels     = config["in_channels"],
            hidden_channels = config["hidden_channels"],
            out_channels    = config["out_channels"],
            heads           = config["heads"],
            dropout         = 0.0,   # eval mode: no dropout
            edge_dim        = config["edge_dim"],
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()
        print(f"  val_auc={config.get('val_auc', 'n/a'):.4f}  "
              f"test_auc={config.get('test_auc', 'n/a'):.4f}")

    # ── Embedder ──────────────────────────────────────────────────────────────

    def _load_embedder_if_needed(self):
        if not hasattr(self, "_embedder"):
            try:
                from model2vec import StaticModel
                self._embedder = StaticModel.from_pretrained(EMBEDDER_PATH)
            except (ImportError, Exception):
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer(EMBEDDER_PATH)
        return self._embedder

    def _embed_query(self, text: str) -> np.ndarray:
        """Embed query → L2-normalized (D,) float32."""
        model = self._load_embedder_if_needed()
        emb   = np.array(model.encode([text.lower()], show_progress_bar=False),
                         dtype=np.float32)[0]
        norm  = np.linalg.norm(emb) + 1e-9
        return emb / norm

    # ── Seed node retrieval ───────────────────────────────────────────────────

    def _keyword_seeds(self, query: str) -> list[int]:
        """
        Substring match: content words from query → entity node IDs.
        Reuses module-level _STOPWORDS + len>=3 filter to extract keywords.
        E.g. "what does anger do" → keyword "anger" ⊆ "holding anger" → matched.
        """
        keywords = [w for w in query.lower().split()
                    if w not in _STOPWORDS and len(w) >= 3]
        if not keywords:
            return []
        matched = []
        for text, eid in self.entity_to_id.items():
            text_words = set(text.lower().split())
            if any(kw in text_words for kw in keywords):
                matched.append(eid)
        return matched

    def _find_seed_nodes(self, query_emb: np.ndarray, top_k: int = 5,
                         query: str = "") -> list[int]:
        """
        Hybrid: dense cosine top-k UNION keyword substring matches.
        Dense retrieval ranks nodes by embedding similarity; keyword retrieval
        catches exact/substring matches diluted in the sentence embedding.
        """
        sims     = self.entity_embs_n @ query_emb           # (N_e,)
        top_idxs = np.argpartition(sims, -top_k)[-top_k:]
        top_idxs = top_idxs[np.argsort(-sims[top_idxs])].tolist()
        kw_ids   = self._keyword_seeds(query) if query else []
        seen     = set(top_idxs)
        return top_idxs + [i for i in kw_ids if i not in seen]

    # ── Subgraph extraction ───────────────────────────────────────────────────

    def _extract_subgraph(self, seed_ids: list[int]) -> tuple:
        """
        Extract 1-hop neighborhood subgraph around seed nodes.

        Returns:
            local_triplets  list of (eid_s, pid, eid_o) — original ids
            local_nodes     sorted list of unique entity ids in subgraph
            node_remap      {global_eid: local_idx} for PyG tensors
        """
        # Collect all neighboring triplets
        seen_triplets = set()
        for eid in seed_ids:
            for t in self.adj.get(eid, []):
                seen_triplets.add(t)

        local_triplets = list(seen_triplets)
        if not local_triplets:
            return [], [], {}

        # Unique nodes in the subgraph
        local_nodes = sorted(set(
            eid for t in local_triplets for eid in (t[0], t[2])
        ))
        node_remap  = {eid: i for i, eid in enumerate(local_nodes)}
        return local_triplets, local_nodes, node_remap

    # ── GATv2 forward on subgraph ─────────────────────────────────────────────

    @torch.no_grad()
    def _run_gat(self, local_nodes: list[int],
                 local_triplets: list[tuple],
                 node_remap: dict) -> np.ndarray:
        """
        Run GATv2 on the extracted subgraph.

        Args:
            local_nodes     global entity ids present in subgraph
            local_triplets  (eid_s, pid, eid_o) using global ids
            node_remap      {global_eid: local_idx}

        Returns:
            z  float32 (n_local_nodes, out_channels)  GATv2 output embeddings
        """
        x         = torch.from_numpy(self.entity_embs_gat[local_nodes]).to(self.device)
        src       = torch.tensor([node_remap[t[0]] for t in local_triplets], dtype=torch.long)
        dst       = torch.tensor([node_remap[t[2]] for t in local_triplets], dtype=torch.long)
        edge_index= torch.stack([src, dst], dim=0).to(self.device)
        edge_attr = torch.from_numpy(
            self.pred_embs[[t[1] for t in local_triplets]]
        ).to(self.device)

        z = self.model(x, edge_index, edge_attr)    # (n_local, out_channels)
        z = F.normalize(z, p=2, dim=-1)             # L2 normalize for dot-product sim
        return z.cpu().numpy()

    # ── Triplet scoring ───────────────────────────────────────────────────────

    def _score_triplets(self, query_emb: np.ndarray, z: np.ndarray,
                        local_nodes: list[int], local_triplets: list[tuple],
                        node_remap: dict) -> list[tuple]:
        """
        Score each triplet by: sim(query, z[subj]) + sim(query, z[obj]).
        Using GATv2-updated embeddings z, which incorporate neighborhood context.

        Returns:
            list of (score, eid_s, pid, eid_o) sorted by score descending
        """
        scored = []
        for eid_s, pid, eid_o in local_triplets:
            li_s = node_remap[eid_s]
            li_o = node_remap[eid_o]
            score = float(query_emb @ z[li_s]) + float(query_emb @ z[li_o])
            scored.append((score, eid_s, pid, eid_o))
        scored.sort(key=lambda x: -x[0])
        return scored

    # ── Public API ────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 15,
                 n_seeds: int = 10) -> list[tuple[str, str, str]]:
        """
        Retrieve top-k relevant triplets for a query.

        Pipeline:
            query → embed → seed nodes → 1-hop subgraph
            → GATv2 forward → score triplets → top-k → text triples

        Args:
            query    natural language query
            top_k    number of triplets to return
            n_seeds  number of seed entity nodes to start from

        Returns:
            list of (subject_text, predicate_text, object_text)
        """
        if not self.adj:
            return []

        query_emb = self._embed_query(query)

        # 1. Seed nodes: dense cosine top-k + keyword substring matches
        seed_ids = self._find_seed_nodes(query_emb, top_k=n_seeds, query=query)

        # 2. 1-hop neighborhood
        local_triplets, local_nodes, node_remap = self._extract_subgraph(seed_ids)
        if not local_triplets:
            return []

        # 3. GATv2 forward
        z = self._run_gat(local_nodes, local_triplets, node_remap)

        # Project query into GAT's latent space for scoring
        with torch.no_grad():
            dev = next(self.model.parameters()).device
            q_t = torch.from_numpy(query_emb).unsqueeze(0).to(dev)
            if self.entity_embs_gat.shape[1] > q_t.shape[-1]:
                pad = torch.zeros(1, self.entity_embs_gat.shape[1] - q_t.shape[-1],
                                  device=dev, dtype=q_t.dtype)
                q_t = torch.cat([q_t, pad], dim=-1)
            query_emb_z = torch.nn.functional.relu(
                self.model.lin_in(q_t)).squeeze(0).cpu().numpy()

        # 4. Score and rank
        scored = self._score_triplets(query_emb_z, z, local_nodes,
                                      local_triplets, node_remap)

        # 5. Convert ids → text
        results = []
        seen    = set()
        for _, eid_s, pid, eid_o in scored:
            key  = (eid_s, pid, eid_o)
            if key in seen:
                continue
            seen.add(key)
            subj = self.entity_id_to_text.get(eid_s, f"[{eid_s}]")
            pred = self.pred_id_to_text.get(pid,   f"[{pid}]")
            obj  = self.entity_id_to_text.get(eid_o, f"[{eid_o}]")
            if not _is_content(subj) or not _is_content(obj):
                continue
            results.append((subj, pred, obj))
            if len(results) >= top_k:
                break

        return results

    def retrieve_context(self, query: str, top_k: int = 15,
                         n_seeds: int = 10) -> str:
        """
        Retrieve and format triplets as a prompt-injection context string.

        Returns:
            Multi-line string like:
                attention mechanism | is used in | transformer model
                self-attention | computes | weighted sum
                ...
        """
        triplets = self.retrieve(query, top_k=top_k, n_seeds=n_seeds)
        if not triplets:
            return ""
        return "\n".join(f"{s} | {p} | {o}" for s, p, o in triplets)

    def retrieve_for_seeds(self, seed_texts: list[str],
                           top_k: int = 20) -> str:
        """
        Retrieve context using multiple seed texts (e.g. from L1 retrieval).
        Merges neighborhoods from all seeds, scores globally.

        Args:
            seed_texts  list of text strings (e.g. from L1 retrieved chunks)
            top_k       total triplets to return across all seeds

        Returns:
            context string for prompt injection
        """
        all_seed_ids = []
        for text in seed_texts:
            q_emb    = self._embed_query(text[:256])   # cap length
            seed_ids = self._find_seed_nodes(q_emb, top_k=3)
            all_seed_ids.extend(seed_ids)
        all_seed_ids = list(set(all_seed_ids))

        # Build merged neighborhood
        seen_triplets = set()
        for eid in all_seed_ids:
            for t in self.adj.get(eid, []):
                seen_triplets.add(t)

        if not seen_triplets:
            return ""

        local_triplets = list(seen_triplets)
        local_nodes    = sorted(set(eid for t in local_triplets for eid in (t[0], t[2])))
        node_remap     = {eid: i for i, eid in enumerate(local_nodes)}

        # Use first seed text as query for scoring
        query_emb = self._embed_query(seed_texts[0][:256])
        z         = self._run_gat(local_nodes, local_triplets, node_remap)

        # Project query into GAT's latent space for scoring
        with torch.no_grad():
            dev = next(self.model.parameters()).device
            q_t = torch.from_numpy(query_emb).unsqueeze(0).to(dev)
            if self.entity_embs_gat.shape[1] > q_t.shape[-1]:
                pad = torch.zeros(1, self.entity_embs_gat.shape[1] - q_t.shape[-1],
                                  device=dev, dtype=q_t.dtype)
                q_t = torch.cat([q_t, pad], dim=-1)
            query_emb_z = torch.nn.functional.relu(
                self.model.lin_in(q_t)).squeeze(0).cpu().numpy()

        scored    = self._score_triplets(query_emb_z, z, local_nodes,
                                         local_triplets, node_remap)

        results = []
        seen    = set()
        for _, eid_s, pid, eid_o in scored:
            key = (eid_s, pid, eid_o)
            if key in seen: continue
            seen.add(key)
            subj = self.entity_id_to_text.get(eid_s, f"[{eid_s}]")
            pred = self.pred_id_to_text.get(pid,   f"[{pid}]")
            obj  = self.entity_id_to_text.get(eid_o, f"[{eid_o}]")
            results.append(f"{subj} | {pred} | {obj}")
            if len(results) >= top_k:
                break

        return "\n".join(results)

    def stats(self) -> dict:
        """Return retriever statistics."""
        return {
            "n_entities":   len(self.entity_embs),
            "n_predicates": len(self.pred_embs),
            "n_chunks":     len(self.triplet_map),
            "n_adj_nodes":  len(self.adj),
            "device":       self.device,
        }


# ── CLI demo ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GraphRetriever demo")
    parser.add_argument("query", nargs="*",
                        help="Query text (multiple words OK)")
    parser.add_argument("--top-k",  type=int, default=15)
    parser.add_argument("--n-seeds",type=int, default=10)
    args = parser.parse_args()

    retriever = GraphRetriever()
    query = " ".join(args.query) if args.query else "attention mechanism in transformer models"

    print(f"\nRetriever stats: {retriever.stats()}")
    print(f"\nQuery: '{query}'")
    print(f"Top {args.top_k} triplets:\n")

    triplets = retriever.retrieve(query, top_k=args.top_k, n_seeds=args.n_seeds)
    for i, (s, p, o) in enumerate(triplets, 1):
        print(f"  {i:2d}. {s} | {p} | {o}")

    print(f"\nFormatted context:\n")
    print(retriever.retrieve_context(query, top_k=args.top_k))
