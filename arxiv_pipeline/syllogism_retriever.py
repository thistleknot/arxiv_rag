"""
syllogism_retriever.py — 9-stage syllogistic reasoning retriever over arXiv paper utilities.

Core Thesis:
    Reasoning pipeline over 1,885+ arXiv papers whose utility phrases are stored in a
    pre-computed semantic vector index (utility_embeddings.npy + utility_catalog.json).
    No vector database required. A single LLM-as-judge call (Qwen via Ollama) replaces
    the cross-encoder NLI path — numbered utilities are presented to the judge; it
    deliberates and emits ranked paper numbers after a '---' separator. Late-stage graph
    extraction (GATv2) provides relational context for the through-line synthesis.

Pipeline (9 stages):
    Stage 0: IntentExtractor.extract(query)
                 → ObjectiveFunction  (goal, domain, requirements)
    Stage 1: Semantic search on pre-computed utility vector index
                 → List[dict] candidates sorted by cosine similarity to intent
    Stage 2: Build flat utility map  {arxiv_id: utility_str}  (rank order preserved)
    Stage 3: NLIEntailmentScorer.rank_utilities(intent, utilities)
                 → LLM judge emits ranked paper numbers (no cross-encoder)
                 → entailed: Dict[arxiv_id, List[str]]
                 → nli_scores: Dict[arxiv_id, float]  (normalised position weights 1.0→1/N)
    Stage 4: SyllogismFormer.form(…, nli_scores)
                 → SyllogismResult (thesis, entailment chain, chain links)
    Stage 5: EntailmentRanker.rerank(docs, result)
                 → blend = 0.5 * nli_score + 0.5 * normalised_retrieval_rank
                 → sorted List[RetrievedDoc] across full [0, 1] range
    Stage 6: load_papers(surviving_ids)
                 → Dict[arxiv_id, markdown]  (docling-converted full text)
    Stage 7: GraphRetriever.retrieve_context(thesis)
                 → KG triplets scored by GATv2  (cached; skipped if unavailable)
    Stage 8: LLM through-line synthesis
                 → natural language philosophy + application paragraph

Usage:
    python syllogism_retriever.py "graph neural networks" --top_k 5 --n_papers 50
    python syllogism_retriever.py --n_papers 0    # use all papers in index
    python syllogism_retriever.py "query" --output report.md  # write Markdown report
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import pathlib
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# Ensure UTF-8 output on Windows (prevents UnicodeEncodeError in console)
_stdout_reconf = getattr(sys.stdout, "reconfigure", None)
if callable(_stdout_reconf):
    _stdout_reconf(encoding="utf-8", errors="replace")
_stderr_reconf = getattr(sys.stderr, "reconfigure", None)
if callable(_stderr_reconf):
    _stderr_reconf(encoding="utf-8", errors="replace")

import numpy as np
import pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

# ── Local imports ─────────────────────────────────────────────────────────────
from reasoning.syllogism_former import ChainLink, SyllogismFormer, SyllogismResult
from reasoning.entailment_ranker import EntailmentRanker
from reasoning.intent_extractor import IntentExtractor, ObjectiveFunction
from reasoning.nli_entailment import NLIEntailmentScorer
from reasoning.paper_loader import load_papers

import httpx

GraphRetriever = None
try:
    from graph.graph_retriever import GraphRetriever
    _GRAPH_AVAILABLE = True
except ImportError:
    _GRAPH_AVAILABLE = False

try:
    from retrieval.gist_retriever import RetrievedDoc as RetrievedDoc  # type: ignore
except ImportError:
    from dataclasses import dataclass as _dc, field as _field
    from typing import Any as _Any
    @_dc
    class RetrievedDoc:  # type: ignore
        doc_id: str = ""
        content: str = ""
        metadata: dict = _field(default_factory=dict)
        final_score: Optional[float] = None

_CSV_PATH = _ROOT / "papers" / "post_processed" / "arxiv_data_with_analysis_cleaned.csv"

# Utility vector index — built once from full CSV, cached to disk
# for sub-second semantic retrieval against encoded query intent.
_UTIL_EMB_PATH     = _ROOT / "papers" / "utility_embeddings.npy"
_UTIL_CATALOG_PATH = _ROOT / "papers" / "utility_catalog.json"

_BLEND_WEIGHTS = {
    "title": 0.3237,
    "abstract": 0.5803,
    "utility": 0.096,
}

OLLAMA_BASE            = "http://127.0.0.1:11434"
_THROUGHLINE_MODEL     = "hf.co/unsloth/Qwen3-4B-128K-GGUF:Qwen3-4B-128K-Q6_K.gguf"

_PAPER_ANGLE_SYSTEM = """\
You are extracting implementation intelligence from a research paper for a software engineering agent.

Produce a structured extraction using these exact headings:

MECHANISM: Name the core algorithm or method precisely (e.g. "sparse top-k attention routing", \
"DPO reward-free preference optimization", "multi-agent hierarchical task decomposition"). \
Explain it in 2-3 sentences — what it does, why it works.

INNOVATION: What this paper contributes that prior work does not. Be specific about the technical delta \
— name the prior technique and state exactly what is improved or replaced.

KEY COMPONENTS: List 2-4 concrete modules, data structures, or computational steps an implementer would \
build. For each, name it and state its function (e.g. "rollout buffer: stores (state, action, reward) \
tuples for off-policy updates; enables decoupling collection from training").

INTEGRATION INTERFACE: Describe the primary input/output contract — what data goes in, what comes out, \
what assumptions are made about the surrounding system or environment.

FAILURE MODES: 2-3 known or implied limitations with specific conditions \
(e.g. "requires >10k labeled preference pairs — breaks on low-resource domains", \
"compute scales quadratically with context length — impractical beyond 8k tokens").

REUSE: One specific module or algorithm from this paper that could be directly lifted into a new system \
with minimal modification — name it and describe its interface.

Write in dense, specific prose. Minimum 250 words total. No hedging. No meta-commentary."""

_THROUGHLINE_SYSTEM = """\
You are synthesizing multiple research paper extractions into an implementation brief for a software \
engineering agent. Produce a comprehensive brief using these exact headings:

THROUGH-LINE: State the shared mathematical or algorithmic principle across all papers. \
What unified problem do they address and why does this class of approach work? \
Name the shared mechanism type. Minimum 3 sentences.

CONVERGENCE POINTS: Where do the approaches agree on data structures, interfaces, or design patterns? \
These are stable foundations to build on first. Be specific — name the structures.

DIVERGENCE POINTS: Where do the approaches disagree or offer complementary alternatives? \
Frame each as a concrete design decision the implementer must make \
(e.g. "synchronous vs. asynchronous inter-agent messaging").

IMPLEMENTATION ROADMAP: A prioritized sequence of 5-7 steps for building a system that captures \
the best ideas from these papers. Each step: name a specific component, explain why it comes before \
the next, and cite which paper(s) it draws from.

RISK REGISTER: 4-5 specific failure modes across the combined approaches. \
For each: (1) the risk, (2) the condition that triggers it, (3) a mitigation strategy.

Write minimum 500 words. Dense, specific, implementable. Every sentence must contain a concrete claim \
or actionable guidance. No hedging. No meta-commentary."""

_DECISION_TREE_SYSTEM = """\
You are recommending which approach to use for a given task based on research papers.
Based on the synthesis provided, write a decision tree with 4-6 yes/no questions that guide selection.
Each leaf node must name a specific method and cite the arXiv ID it comes from.
Each branch must state what condition triggers it (resource constraint, data availability, latency budget, etc.).
Write in plain text. No pseudocode. No equations. Minimum 150 words."""

_THINK_RE = re.compile(r'<think>.*?</think>', re.DOTALL | re.IGNORECASE)
_THINK_OPEN_RE = re.compile(r'<think>', re.IGNORECASE)


def _strip_think(text: str) -> str:
    """Strip Qwen3 chain-of-thought <think>…</think> blocks.

    Handles both complete blocks (open + close tag) and incomplete blocks where
    the model hit its token limit mid-think and never emitted the closing tag.
    """
    # Strip complete blocks first
    text = _THINK_RE.sub('', text)
    # Strip any remaining incomplete block (no closing tag)
    m = _THINK_OPEN_RE.search(text)
    if m:
        text = text[:m.start()]
    return text.strip()


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class SyllogismRetrievalResult:
    """Full output of SyllogismRetriever.retrieve()."""
    query:          str
    papers:         List[Any]           = field(default_factory=list)
    thesis:         str                 = ""
    chain:          List[ChainLink]     = field(default_factory=list)
    objective:      Optional[str]       = None               # ObjectiveFunction.as_text()
    papers_content: Dict[str, str]      = field(default_factory=dict)   # arxiv_id → markdown
    nli_scores:     Dict[str, float]    = field(default_factory=dict)   # arxiv_id → NLI score
    graph_context:  str                 = ""                            # KG triplets from GraphRetriever
    through_line:   str                 = ""                            # LLM-synthesised through-line
    paper_angles:   Dict[str, str]      = field(default_factory=dict)   # arxiv_id → "Philosophy\nApplication"
    decision_tree:  str                 = ""                            # LLM-generated decision tree

    def top_k_summary(self, k: int = 5) -> str:
        lines = [f"Query  : {self.query}"]
        if self.objective:
            lines.append(f"Intent : {self.objective}")
        lines.append(f"Thesis : {self.thesis}")
        lines.append("")
        if self.chain:
            lines.append("Entailment chain:")
            for link in self.chain:
                lines.append(f"  [{link.position}] ({link.necessity_score:.2f}) "
                              f"{link.arxiv_id}: {link.premise_text}")
            lines.append("")
        loaded = len(self.papers_content)
        if loaded:
            lines.append(f"Papers loaded: {loaded} markdown files")
            lines.append("")
        if self.graph_context:
            n_triples = self.graph_context.count("\n") + 1
            lines.append(f"Graph context: {n_triples} triplets")
            lines.append("")
        if self.through_line:
            lines.append(f"Through-line: {self.through_line}")
            lines.append("")
        lines.append(f"Top {k} papers:")
        chain_ids = {l.arxiv_id for l in self.chain}
        for i, doc in enumerate(self.papers[:k], 1):
            pid   = doc.metadata.get("paper_id", doc.doc_id)
            title = doc.metadata.get("title", "")
            es    = doc.metadata.get("entailment_score", 0.0)
            ns    = self.nli_scores.get(pid, 0.0)
            fs    = doc.final_score or 0.0
            mk    = "★" if pid in chain_ids else " "
            lines.append(f"  {mk} {i}. [{pid}] {title[:60]}  "
                         f"(blend={fs:.3f}, entail={es:.3f}, nli={ns:.3f})")
        return "\n".join(lines)

    def to_markdown(self, k: int = 5) -> str:
        """Full retrieval synthesis as a markdown document."""
        md = []
        md.append(f"# Syllogism Retrieval Report")
        md.append("")
        md.append(f"> **Query**: {self.query}")
        md.append("")
        md.append("---")
        md.append("")

        # ── Intent ──
        if self.objective:
            md.append("## Intent")
            md.append("")
            md.append(self.objective)
            md.append("")

        # ── Thesis ──
        md.append("## Thesis")
        md.append("")
        md.append(self.thesis or "*(no thesis formed)*")
        md.append("")

        # ── Entailment chain ──
        if self.chain:
            md.append("## Ranked Evidence Chain")
            md.append("")
            md.append("> **Scoring note:** NLI scores are cross-encoder entailment probabilities "
                      "(cross-encoder/nli-deberta-v3-small). Retrieval score is normalised position "
                      "rank from the initial semantic search. Blend = 0.5 × NLI + 0.5 × retrieval norm.")
            md.append("")
            md.append("| # | arXiv ID | NLI | Premise |")
            md.append("|---|----------|-----|---------|")
            for link in self.chain:
                md.append(f"| {link.position} | `{link.arxiv_id}` "
                          f"| {link.necessity_score:.3f} "
                          f"| {link.premise_text} |")
            md.append("")

        # ── Graph context ──
        if self.graph_context:
            triples = [t.strip() for t in self.graph_context.split("\n") if t.strip()]
            md.append(f"## Knowledge Graph Context ({len(triples)} triplets)")
            md.append("")
            md.append("| Subject | Predicate | Object |")
            md.append("|---------|-----------|--------|")
            for t in triples:
                parts = [p.strip() for p in t.split("|")]
                if len(parts) >= 3:
                    md.append(f"| {parts[0]} | {parts[1]} | {parts[2]} |")
                else:
                    md.append(f"| {t} | | |")
            md.append("")

        # ── Top-k papers ──
        chain_ids = {l.arxiv_id for l in self.chain}
        md.append(f"## Top {k} Papers")
        md.append("")
        for i, doc in enumerate(self.papers[:k], 1):
            pid   = doc.metadata.get("paper_id", doc.doc_id)
            title = doc.metadata.get("title", "(untitled)")
            rs    = doc.metadata.get("original_final_score", 0.0)
            ns    = self.nli_scores.get(pid, 0.0)
            fs    = doc.final_score or 0.0
            in_chain = " ★" if pid in chain_ids else ""
            md.append(f"### {i}. [{pid}] {title}{in_chain}")
            md.append("")
            md.append(f"| Metric | Value |")
            md.append(f"|--------|-------|")
            md.append(f"| Blend score      | {fs:.4f} |")
            md.append(f"| Retrieval (norm) | {rs:.4f} |")
            md.append(f"| NLI score        | {ns:.4f} |")
            md.append("")
            abstract = doc.metadata.get("abstract", "")
            if abstract:
                md.append("**Abstract:**")
                md.append("")
                md.append(abstract)
                md.append("")

        # ── Per-paper Synthesis ──
        if self.paper_angles:
            md.append("## Synthesis")
            md.append("")
            for pid, angle in self.paper_angles.items():
                title = next(
                    (d.metadata.get("title", "") for d in self.papers
                     if d.metadata.get("paper_id", d.doc_id) == pid),
                    ""
                )
                md.append(f"**[{pid}] {title}**")
                md.append("")
                md.append(angle)
                md.append("")

        # ── Through-line ──
        if self.through_line:
            md.append("## Informed Response")
            md.append("")
            md.append(self.through_line)
            md.append("")

        # ── Decision tree ──
        if self.decision_tree:
            md.append("### Practical Recommendation")
            md.append("")
            md.append(self.decision_tree)
            md.append("")

        # ── References ──
        md.append("## References")
        md.append("")
        for i, doc in enumerate(self.papers[:k], 1):
            pid   = doc.metadata.get("paper_id", doc.doc_id)
            title = doc.metadata.get("title", "(untitled)")
            md.append(f"{i}. [{pid}] {title}  ")
            md.append(f"   https://arxiv.org/abs/{pid}")
            md.append("")

        return "\n".join(md)


def _coerce_utility(v: str) -> str:
    """Convert CSV utility value to a plain text string.

    The CSV stores utilities as stringified JSON lists, e.g.:
        "[\"Point one.\", \"Point two.\"]"

    This function parses the list and joins items with ". ".
    Falls back to the raw string if parsing fails.
    """
    if not v:
        return ""
    v = v.strip()
    if v.startswith("["):
        try:
            items = ast.literal_eval(v)
            if isinstance(items, list):
                return ". ".join(str(x).strip().rstrip(".") for x in items if x) + "."
        except (ValueError, SyntaxError):
            pass
    return v


def _clean_text(v: str) -> str:
    if not v:
        return ""
    s = str(v).strip()
    if s.lower() in {"nan", "none", "null"}:
        return ""
    return s.replace("\n", " ").strip()


def _norm_arxiv_id(v: str) -> str:
    return str(v).strip().strip('"')


# ── Orchestrator ──────────────────────────────────────────────────────────────

class SyllogismRetriever:
    """
    End-to-end syllogism-augmented retrieval over arXiv papers.

    Semantic retrieval over arXiv paper utilities (loaded from CSV).

    Stage 1 embeds the query intent against a cached utility index
    (all-MiniLM-L6-v2) to retrieve the top-N most relevant candidates
    before any NLI or LLM work begins.

    Args:
        csv_path: Path to the cleaned CSV with arxiv_id, utility, title columns.
        verbose: Print progress at each stage.
    """

    def __init__(
        self,
        csv_path: pathlib.Path = _CSV_PATH,
        verbose: bool = False,
        blend_weights: Optional[Dict[str, float]] = None,
    ):
        self._csv_path     = csv_path
        self._blend_weights = blend_weights or dict(_BLEND_WEIGHTS)
        self._former    = SyllogismFormer(verbose=verbose)
        self._ranker    = EntailmentRanker(verbose=verbose)
        self._intent    = IntentExtractor(verbose=verbose)
        self._nli       = NLIEntailmentScorer(verbose=verbose)
        self._verbose   = verbose

        # Graph retriever (lazy — heavy init, only if artifacts exist)
        self._graph: Optional[Any] = None
        if _GRAPH_AVAILABLE and callable(GraphRetriever):
            try:
                self._graph = GraphRetriever()
                if self._verbose:
                    print("[init] GraphRetriever loaded")
            except Exception as e:
                if self._verbose:
                    print(f"[init] GraphRetriever unavailable: {e}")
                self._graph = None

        # httpx client for through-line LLM call
        self._llm_client = httpx.Client(base_url=OLLAMA_BASE, timeout=600.0)

        # Sentence-transformer for semantic utility index search (lazy)
        self._embedder = None
        self._rows: List[Dict[str, str]] = []
        self._field_embeddings: Dict[str, np.ndarray] = {}
        self._field_texts: Dict[str, List[str]] = {}
        try:
            from sentence_transformers import SentenceTransformer as _ST
            self._embedder = _ST("all-MiniLM-L6-v2")
        except Exception:
            pass

        self._load_csv_rows()

    def _load_csv_rows(self) -> None:
        """Load minimal retrieval fields from CSV into memory."""
        rows: List[Dict[str, str]] = []
        with open(self._csv_path, encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                arxiv_id = _norm_arxiv_id(row.get("arxiv_id", ""))
                if not arxiv_id:
                    continue
                utility = _coerce_utility(_clean_text(row.get("utility", "")))
                title = _clean_text(row.get("title", ""))
                abstract = _clean_text(row.get("abstract", ""))
                rows.append({
                    "arxiv_id": arxiv_id,
                    "title": title,
                    "abstract": abstract,
                    "utility": utility,
                })
        self._rows = rows

    def _ensure_field_embeddings(self) -> None:
        """Build one-time in-memory embeddings for title/abstract/utility."""
        if self._field_embeddings:
            return
        if self._embedder is None:
            raise RuntimeError("SentenceTransformer embedder is unavailable")

        for field_name in ("title", "abstract", "utility"):
            texts = [r[field_name] for r in self._rows]
            self._field_texts[field_name] = texts
            self._field_embeddings[field_name] = self._embedder.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

    def _semantic_search_blend(
        self,
        intent_text: str,
        n_papers: int,
    ) -> Tuple[List[Any], Dict[str, str]]:
        """
        Stage 1 retrieval with weighted embedding blend:
            score = 0.4 * cos(q, title) + 0.3 * cos(q, abstract) + 0.3 * cos(q, utility)
        Returns ordered candidates and an ordered utility map for Stage 3 judge.
        """
        self._ensure_field_embeddings()

        if self._embedder is None:
            raise RuntimeError("Embedder is unavailable")

        q_emb = self._embedder.encode(
            [intent_text],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]

        title_scores = self._field_embeddings["title"] @ q_emb
        abstract_scores = self._field_embeddings["abstract"] @ q_emb
        utility_scores = self._field_embeddings["utility"] @ q_emb

        blended = (
            self._blend_weights["title"] * title_scores
            + self._blend_weights["abstract"] * abstract_scores
            + self._blend_weights["utility"] * utility_scores
        )

        order = np.argsort(-blended)
        if n_papers and n_papers > 0:
            order = order[:n_papers]

        chosen_scores = blended[order]
        lo = float(chosen_scores.min()) if len(chosen_scores) else 0.0
        hi = float(chosen_scores.max()) if len(chosen_scores) else 1.0
        span = (hi - lo) if (hi - lo) > 1e-12 else 1.0

        docs: List[Any] = []
        utilities_map: Dict[str, str] = {}

        for idx in order.tolist():
            row = self._rows[idx]
            aid = row["arxiv_id"]
            raw = float(blended[idx])
            norm = (raw - lo) / span

            doc = RetrievedDoc(
                doc_id=aid,
                content=row["utility"] or row["abstract"] or row["title"],
                metadata={
                    "paper_id": aid,
                    "title": row["title"],
                    "abstract": row["abstract"],
                    "utility": row["utility"],
                    "retrieval_blend_raw": raw,
                    "retrieval_blend_norm": norm,
                },
                final_score=norm,
            )
            docs.append(doc)
            utilities_map[aid] = row["utility"]

        return docs, utilities_map

    def _build_candidates_from_ids(
        self,
        pre_selected: List[Tuple[str, float]],
    ) -> Tuple[List[Any], Dict[str, str]]:
        """Build candidates and utility map from pre-selected (arxiv_id, score) pairs.

        Used when the 3-layer φ-scaled retriever has already produced a ranked
        candidate set (L3 output → top_k=13 papers).  Skips the cosine search and
        preserves the upstream scorer's score as the retrieval_blend_norm signal.

        Precondition: self._rows is loaded.
        Guarantee: only IDs present in the CSV are returned; unknown IDs are skipped.
        """
        id_to_row = {r["arxiv_id"]: r for r in self._rows}
        docs: List[Any] = []
        utilities_map: Dict[str, str] = {}
        seen: set = set()
        for arxiv_id, score in pre_selected:
            arxiv_id = arxiv_id.strip()
            if not arxiv_id or arxiv_id in seen:
                continue
            seen.add(arxiv_id)
            row = id_to_row.get(arxiv_id)
            if row is None:
                continue
            norm = float(score)
            doc = RetrievedDoc(
                doc_id=arxiv_id,
                content=row["utility"] or row["abstract"] or row["title"],
                metadata={
                    "paper_id": arxiv_id,
                    "title": row["title"],
                    "abstract": row["abstract"],
                    "utility": row["utility"],
                    "retrieval_blend_raw": norm,
                    "retrieval_blend_norm": norm,
                },
                final_score=norm,
            )
            docs.append(doc)
            utilities_map[arxiv_id] = row["utility"]
        return docs, utilities_map

    def retrieve(
        self,
        query: str,
        top_k: int = 8,
        n_papers: int = 13,
        pre_selected: Optional[List[Tuple[str, float]]] = None,
    ) -> SyllogismRetrievalResult:
        """Execute full 9-stage syllogism retrieval pipeline.

        Stage 0: Extract intent from query (goal, domain, requirements).
        Stage 1: Candidate selection — either from pre-selected upstream results
                 (3-layer φ retriever output) or via cosine search over utility
                 embeddings (standalone fallback with n_papers candidates).
        Stage 2: Build utility ranking map (maintain order).
        Stage 3: NLI scoring — LLM judge ranks papers by utility relevance.
        Stage 4: Form syllogism — reason over entailed utilities.
        Stage 5: Rerank by blend (NLI + retrieval scores).
        Stage 6: Load paper markdowns from disk.
        Stage 7: Graph retriever context (optional, if available).
        Stage 8: LLM through-line synthesis over top papers.

        Args:
            query: Natural language retrieval query.
            top_k: Final number of papers to return after reranking.
            n_papers: Candidate pool size for standalone cosine search.
                      Ignored when pre_selected is provided.
                      Default 13 matches the φ-scaled L3 output (top_k=13).
            pre_selected: Optional list of (arxiv_id, score) pairs from an
                          upstream 3-layer retriever (L3 output).  When provided,
                          Stage 1 cosine search is skipped entirely.
        """
        result = SyllogismRetrievalResult(query=query)
        
        if self._verbose:
            print(f"[retrieve] Query: {query}")
        
        # ── Stage 0: Extract intent ──
        if self._verbose:
            print(f"[stage 0] Extracting intent from query...")
        try:
            objective = self._intent.extract(query)
            result.objective = objective.as_text() if hasattr(objective, 'as_text') else str(objective)
        except Exception as e:
            if self._verbose:
                print(f"[stage 0] Intent extraction failed: {e}")
            result.objective = query

        # ── Stage 1: Candidate selection ──
        candidates: List[Any] = []
        utilities_map: Dict[str, str] = {}
        try:
            if pre_selected is not None:
                if self._verbose:
                    print(f"[stage 1] Using {len(pre_selected)} pre-selected candidates "
                          f"from upstream 3-layer retriever")
                candidates, utilities_map = self._build_candidates_from_ids(pre_selected)
            else:
                if self._verbose:
                    print(f"[stage 1] Cosine search over utility index (n_papers={n_papers})...")
                stage1_intent = result.objective or query
                candidates, utilities_map = self._semantic_search_blend(stage1_intent, n_papers)
            if self._verbose:
                print(f"[stage 1] {len(candidates)} candidates ready")
        except Exception as e:
            if self._verbose:
                print(f"[stage 1] Candidate selection failed: {e}")
            candidates = []
            utilities_map = {}

        # ── Stage 2: Build utility map ──
        if self._verbose:
            print(f"[stage 2] Building utility ranking map...")
        if self._verbose:
            print(f"[stage 2] Utility map size: {len(utilities_map)}")

        # ── Stage 3: NLI scoring ──
        if self._verbose:
            print(f"[stage 3] NLI entailment scoring (LLM judge)...")
        try:
            entailed, nli_scores = self._nli.rank_utilities(result.objective or query, utilities_map)
            result.nli_scores = nli_scores
        except Exception as e:
            if self._verbose:
                print(f"[stage 3] NLI scoring failed: {e}")
            entailed = {}
            result.nli_scores = {}

        # ── Stage 4: Form syllogism ──
        if self._verbose:
            print(f"[stage 4] Forming syllogism from entailed utilities...")
        try:
            syllogism = self._former.form(
                query=query,
                premises_by_paper=entailed,
                nli_scores=result.nli_scores,
                objective_text=result.objective or "",
                premise_scores=self._nli.premise_scores,
            )
            result.thesis = syllogism.thesis if hasattr(syllogism, 'thesis') else ""
            result.chain = syllogism.chain if hasattr(syllogism, 'chain') else []
        except Exception as e:
            if self._verbose:
                print(f"[stage 4] Syllogism forming failed: {e}")
            syllogism = SyllogismResult(thesis="", chain=[], paper_scores={})
        try:
            result.papers = self._ranker.rerank(candidates, syllogism)  # type: ignore[arg-type]
        except Exception as e:
            if self._verbose:
                print(f"[stage 5] Rerank failed: {e}")
            result.papers = candidates
        try:
            surviving_ids = [link.arxiv_id for link in result.chain]
            if not surviving_ids:
                surviving_ids = [d.metadata.get("paper_id", d.doc_id) for d in result.papers[:top_k]]
            result.papers_content = load_papers(surviving_ids)
            if self._verbose:
                print(f"[stage 6] Loaded markdown for {len(result.papers_content)} papers")
        except Exception as e:
            if self._verbose:
                print(f"[stage 6] Paper loading failed: {e}")

        # ── Stage 5: Rerank by blend ──
        if self._verbose:
            print(f"[stage 5] Reranking by NLI + retrieval blend...")

        # ── Stage 6: Load papers ──
        if self._verbose:
            print(f"[stage 6] Loading paper markdowns...")

        # ── Stage 7: Graph context ──
        if self._graph:
            if self._verbose:
                print(f"[stage 7] Retrieving graph context...")
            try:
                result.graph_context = self._graph.retrieve_context(result.thesis)
            except Exception as e:
                if self._verbose:
                    print(f"[stage 7] Graph retrieval failed: {e}")

        # ── Stage 8: Through-line synthesis ──
        if self._verbose:
            print(f"[stage 8] LLM through-line synthesis ({len(result.papers)} papers)...")
        try:
            # Per-paper angle extraction (MAP phase: 1024 tokens each)
            for doc in result.papers[:top_k]:
                pid      = doc.metadata.get("paper_id", doc.doc_id)
                title    = doc.metadata.get("title", "")
                abstract = doc.metadata.get("abstract", "")
                utility  = doc.metadata.get("utility", "")
                user_msg = f"/no_think\n\nTitle: {title}\n\nAbstract: {abstract}\n\nUtility: {utility}"
                payload  = {
                    "model":   _THROUGHLINE_MODEL,
                    "system":  _PAPER_ANGLE_SYSTEM,
                    "prompt":  user_msg,
                    "stream":  False,
                    "think":   False,
                    "options": {"num_predict": 1024, "temperature": 0.0},
                }
                resp = self._llm_client.post("/api/generate", json=payload)
                resp.raise_for_status()
                angle = _strip_think(resp.json().get("response", ""))
                result.paper_angles[pid] = angle

            # Capstone through-line synthesis (REDUCE phase: concatenated extractions → 2048 tokens)
            if result.paper_angles:
                angles_text = "\n\n---\n\n".join(
                    f"[{pid}]\n{angle}" for pid, angle in result.paper_angles.items()
                )
                payload = {
                    "model":   _THROUGHLINE_MODEL,
                    "system":  _THROUGHLINE_SYSTEM,
                    "prompt":  f"/no_think\n\nQuery: {result.query}\n\nPer-paper extractions:\n\n{angles_text}",
                    "stream":  False,
                    "think":   False,
                    "options": {"num_predict": 2048, "temperature": 0.2},
                }
                resp = self._llm_client.post("/api/generate", json=payload)
                resp.raise_for_status()
                result.through_line = _strip_think(resp.json().get("response", ""))

            # Decision tree from capstone (512 tokens)
            if result.through_line:
                payload = {
                    "model":   _THROUGHLINE_MODEL,
                    "system":  _DECISION_TREE_SYSTEM,
                    "prompt":  f"/no_think\n\nQuery: {result.query}\n\nSynthesis:\n{result.through_line}",
                    "stream":  False,
                    "think":   False,
                    "options": {"num_predict": 512, "temperature": 0.2},
                }
                resp = self._llm_client.post("/api/generate", json=payload)
                resp.raise_for_status()
                result.decision_tree = _strip_think(resp.json().get("response", ""))
        except Exception as exc:
            if self._verbose:
                print(f"[stage 8] warning: {exc}")

        if self._verbose:
            print(f"[retrieve] Complete. Thesis: {result.thesis[:80]}...")

        return result
