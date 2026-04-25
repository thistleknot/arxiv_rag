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
import collections
import csv
import json
import math
import pathlib
import re
import sys
import time
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
from constants import BLEND_WEIGHTS, OLLAMA_BASE, THROUGHLINE_MODEL, TRIPLET_REINFORCE_TAU
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

NLIGraphRetriever = None
try:
    from graph.nli_graph_retriever import NLIGraphRetriever
    _NLI_GRAPH_AVAILABLE = True
except ImportError:
    _NLI_GRAPH_AVAILABLE = False

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

_PAPER_ANGLE_SYSTEM = """\
You are extracting pseudocode and concrete algorithmic methods from a research paper for a software \
engineering agent that needs to implement this work.

Reproduce the core algorithm(s) from this paper in pseudocode. Use indented pseudocode notation — \
not Python syntax, not LaTeX, not natural language prose. Preserve variable names and notation \
exactly as used in the paper wherever possible.

Use this structure for each algorithm:

ALGORITHM: <exact name from the paper>
PURPOSE: <one sentence — what computational problem does this solve>

PSEUDOCODE:
<indented pseudocode, 10-40 lines, reproducing the paper's core method step by step>

VARIABLES:
<each variable from the pseudocode: name — type and role>

PRECONDITIONS: <what must hold before calling this algorithm>
POSTCONDITIONS: <what is guaranteed after it completes>
COMPLEXITY: <time and space complexity, if stated or derivable from the paper>

If the paper presents multiple distinct algorithms (e.g. training loop + inference loop, or \
two complementary methods), extract each one using the same structure, separated by a blank line.

Be literal. Do not invent steps. Do not paraphrase into prose. If the paper shows an equation \
that is a computational step, render it as a pseudocode assignment. Minimum 200 words of content."""

_DECISION_TREE_SYSTEM = """\
You are recommending which approach to use for a given task based on research papers.
Based on the pseudocode extractions provided, write a decision tree with 4-6 yes/no questions that guide \
selection of which paper's algorithm to implement.
Each leaf node must name a specific algorithm and cite the arXiv ID it comes from.
Each branch must state what condition triggers it (resource constraint, data availability, latency budget, \
scale of input, etc.).
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
    graph_premises: Dict[str, List[str]] = field(default_factory=dict) # arxiv_id → NLI-ranked premise strings
    through_line:   str                 = ""                            # unused — capstone = Synthesis section concatenation
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
            if self.graph_premises:
                # Per-paper NLI-ranked premises (selected papers hit KG cache)
                total = sum(len(v) for v in self.graph_premises.values())
                md.append(f"## Knowledge Graph Premises ({total} NLI-ranked)")
                md.append("")
                md.append("> Premises retrieved via BM25 over KG triplet lemmas, re-ranked by DeBERTa NLI entailment.")
                md.append("")
                for aid, premises in self.graph_premises.items():
                    md.append(f"**[{aid}]**")
                    for p in premises:
                        md.append(f"- {p}")
                    md.append("")
            else:
                # Corpus-wide NLI premises (selected papers not in KG cache)
                lines = [t.strip() for t in self.graph_context.split("\n") if t.strip()]
                # Detect new "[arxiv_id] premise" format vs legacy "|"-delimited triplets
                is_new_fmt = lines and lines[0].startswith("[")
                if is_new_fmt:
                    md.append(f"## Knowledge Graph Context ({len(lines)} NLI-ranked premises)")
                    md.append("")
                    md.append("> Corpus-wide BM25+NLI search (selected papers not yet KG-cached).")
                    md.append("")
                    for line in lines:
                        md.append(f"- {line}")
                    md.append("")
                else:
                    # Legacy: raw triplet table
                    md.append(f"## Knowledge Graph Context ({len(lines)} triplets)")
                    md.append("")
                    md.append("| Subject | Predicate | Object |")
                    md.append("|---------|-----------|--------|")
                    for t in lines:
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


# ── Latency tracking ─────────────────────────────────────────────────────────

class _LatencyTracker:
    """Per-model log-space latency outlier detector.

    Records response times for each model name and flags unusually slow calls
    using log-space median + MAD.  Logs a warning only; never blocks execution.

    Precondition: record() called only on successful completions.
    Failure mode: silently skipped when fewer than 5 samples or MAD == 0.
    """

    def __init__(self, window: int = 50) -> None:
        self._windows: Dict[str, collections.deque] = {}
        self._window_size = window

    def record(self, model: str, secs: float) -> None:
        """Append a successful call duration (seconds) for model."""
        if model not in self._windows:
            self._windows[model] = collections.deque(maxlen=self._window_size)
        self._windows[model].append(secs)

    def check_outlier(self, model: str, secs: float) -> Tuple[bool, float]:
        """Return (is_outlier, threshold) via log-space median+MAD.

        Returns (False, 0.0) when fewer than 5 samples or MAD is zero.
        """
        buf = self._windows.get(model)
        if not buf or len(buf) < 5:
            return False, 0.0
        log_vals = [math.log(max(v, 1e-9)) for v in buf]
        med = float(np.median(log_vals))
        mad = float(np.median([abs(v - med) for v in log_vals]))
        if mad == 0.0:
            return False, 0.0
        threshold = math.exp(med + 3.0 * mad)
        return secs > threshold, threshold


_lat_tracker = _LatencyTracker()


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
        self._blend_weights = blend_weights or dict(BLEND_WEIGHTS)
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

        # NLI graph retriever — BM25+DeBERTa over pre-built KG cache
        self._nli_graph: Optional[Any] = None
        if _NLI_GRAPH_AVAILABLE and callable(NLIGraphRetriever):
            try:
                self._nli_graph = NLIGraphRetriever()
                if self._verbose:
                    print("[init] NLIGraphRetriever loaded")
            except Exception as e:
                if self._verbose:
                    print(f"[init] NLIGraphRetriever unavailable: {e}")
                self._nli_graph = None

        # httpx client for through-line LLM call
        self._llm_client = httpx.Client(base_url=OLLAMA_BASE, timeout=300.0)

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

        # Memory store (Pages + Throughlines tiers) — lazy, non-critical
        self._memory = None
        try:
            from graph.memory_store import MemoryStore
            self._memory = MemoryStore()
        except Exception:
            pass

        # Runtime state for current retrieval (set in Stage 0)
        self._objective_struct = None
        self._goal_vec: Optional[np.ndarray] = None
        self._page_id: Optional[int] = None

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
            self._objective_struct = objective
            result.objective = objective.as_text() if hasattr(objective, 'as_text') else str(objective)
        except Exception as e:
            if self._verbose:
                print(f"[stage 0] Intent extraction failed: {e}")
            self._objective_struct = None
            result.objective = query

        # ── Stage 0b: Memory page matching ──
        self._goal_vec = None
        self._page_id = None
        if self._memory is not None and self._embedder is not None and self._objective_struct is not None:
            try:
                goal_text = getattr(self._objective_struct, 'goal', result.objective)
                domain = getattr(self._objective_struct, 'domain', '')
                self._goal_vec = self._embedder.encode(
                    goal_text, convert_to_numpy=True, normalize_embeddings=True
                ).astype("float32")
                self._page_id = self._memory.match_page(self._goal_vec, domain)
                if self._page_id is not None:
                    self._memory.increment_read(self._page_id)
                    if self._verbose:
                        print(f"[stage 0] memory page matched: page_id={self._page_id}")
            except Exception:
                pass

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
        if self._verbose:
            print(f"[stage 7] Retrieving NLI graph context...")
        try:
            selected_ids = [
                doc.metadata.get("paper_id", doc.doc_id)
                for doc in result.papers[:top_k]
            ]
            if self._nli_graph is not None:
                nli_query = result.objective or query

                # Pass 1: premises filtered to selected papers (for Stage 8 per-paper injection)
                if selected_ids:
                    result.graph_premises = self._nli_graph.retrieve_context_by_paper(
                        query=nli_query,
                        paper_ids=selected_ids,
                        top_k_per_paper=3,
                    )
                # Capture Pass 1 triplet evidence for memory reinforcement before Pass 2 overwrites
                pass1_triplet_keys, pass1_nli_scores = self._nli_graph.get_last_triplet_keys(
                    threshold=TRIPLET_REINFORCE_TAU
                )

                # Pass 2: unfiltered corpus-wide search always populates graph_context.
                # This covers the common case where selected papers are not yet KG-cached.
                corpus_premises = self._nli_graph.retrieve_context_by_paper(
                    query=nli_query,
                    paper_ids=None,   # full corpus
                    top_k_per_paper=2,
                )
                result.graph_context = "\n".join(
                    f"[{aid}] {p}"
                    for aid, premises in corpus_premises.items()
                    for p in premises
                )

                n_per_paper = sum(len(v) for v in result.graph_premises.values())
                if self._verbose:
                    print(
                        f"[stage 7] per-paper premises: {n_per_paper} "
                        f"({len(result.graph_premises)} papers hit KG cache); "
                        f"corpus context: {len(corpus_premises)} papers"
                    )
                # MemRL Q-update: reinforce triplet confidence from corpus-wide NLI scores
                self._nli_graph.reinforce_from_last_query()

                # Update memory store (non-critical — wrapped in _update_memory)
                self._update_memory(result, pass1_triplet_keys, pass1_nli_scores)

            elif self._graph is not None:
                # Legacy fallback: cosine-filtered triplets (requires Ollama)
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
                # Include NLI-ranked KG premises if available for this paper
                premises = result.graph_premises.get(pid, [])
                premises_block = ""
                if premises:
                    premises_block = "\n\nKnowledge Graph Premises (NLI-ranked):\n" + "\n".join(
                        f"- {p}" for p in premises
                    )
                user_msg = (
                    f"/no_think\n\nTitle: {title}\n\nAbstract: {abstract}"
                    f"\n\nUtility: {utility}{premises_block}"
                )
                payload  = {
                    "model":   THROUGHLINE_MODEL,
                    "system":  _PAPER_ANGLE_SYSTEM,
                    "prompt":  user_msg,
                    "stream":  False,
                    "think":   False,
                    "options": {"num_predict": 1024, "temperature": 0.0},
                }
                _t0 = time.perf_counter()
                resp = self._llm_client.post("/api/generate", json=payload)
                resp.raise_for_status()
                _elapsed = time.perf_counter() - _t0
                _lat_tracker.record(THROUGHLINE_MODEL, _elapsed)
                _slow, _thr = _lat_tracker.check_outlier(THROUGHLINE_MODEL, _elapsed)
                if _slow:
                    print(f"[stage 8] latency outlier: {_elapsed:.1f}s > {_thr:.1f}s (model={THROUGHLINE_MODEL})")
                angle = _strip_think(resp.json().get("response", ""))
                result.paper_angles[pid] = angle

            # Capstone: literal concatenation of per-paper pseudocode extractions (no LLM call)
            if result.paper_angles:
                angles_text = "\n\n---\n\n".join(
                    f"[{pid}]\n{angle}" for pid, angle in result.paper_angles.items()
                )

            # Decision tree from concatenated pseudocode (512 tokens)
            if result.paper_angles:
                payload = {
                    "model":   THROUGHLINE_MODEL,
                    "system":  _DECISION_TREE_SYSTEM,
                    "prompt":  f"/no_think\n\nQuery: {result.query}\n\nPseudocode extractions:\n{angles_text}",
                    "stream":  False,
                    "think":   False,
                    "options": {"num_predict": 512, "temperature": 0.2},
                }
                _t0 = time.perf_counter()
                resp = self._llm_client.post("/api/generate", json=payload)
                resp.raise_for_status()
                _elapsed = time.perf_counter() - _t0
                _lat_tracker.record(THROUGHLINE_MODEL, _elapsed)
                _slow, _thr = _lat_tracker.check_outlier(THROUGHLINE_MODEL, _elapsed)
                if _slow:
                    print(f"[stage 8] latency outlier: {_elapsed:.1f}s > {_thr:.1f}s (model={THROUGHLINE_MODEL})")
                result.decision_tree = _strip_think(resp.json().get("response", ""))
        except Exception as exc:
            if self._verbose:
                print(f"[stage 8] warning: {exc}")

        if self._verbose:
            print(f"[retrieve] Complete. Thesis: {result.thesis[:80]}...")

        return result

    # ── Memory helpers ──────────────────────────────────────────────────────────

    def _update_memory(
        self,
        result: "RetrievalResult",
        pass1_triplet_keys: List[str],
        pass1_nli_scores: List[float],
    ) -> None:
        """Upsert memory page and throughline after a completed retrieval.

        Fully wrapped in try/except — memory is non-critical and must never
        break the main retrieval flow.
        """
        if self._memory is None or self._goal_vec is None:
            return
        try:
            from constants import PAGE_EMBED_SIM_TAU
            objective = self._objective_struct
            goal_text = getattr(objective, "goal", result.objective or "")
            domain = getattr(objective, "domain", "")
            intent = getattr(objective, "intent", "")
            chain_ids = [
                doc.metadata.get("paper_id", doc.doc_id)
                for doc in (result.papers or [])
            ]

            # Create page if no match found at Stage 0b
            if self._page_id is None and goal_text:
                bm25_text = f"{goal_text} {domain}".strip()
                self._page_id = self._memory.create_page(
                    goal=goal_text,
                    domain=domain,
                    intent=intent,
                    goal_vec=self._goal_vec,
                    bm25_text=bm25_text,
                )

            if self._page_id is None:
                return

            # Reinforce triplet evidence for this page
            triplet_keys = pass1_triplet_keys or []
            nli_scores = pass1_nli_scores or []
            if triplet_keys:
                self._memory.reinforce_page(
                    page_id=self._page_id,
                    triplet_keys=triplet_keys,
                    nli_scores=nli_scores,
                )

            # Upsert throughline if thesis qualifies
            if _thesis_qualifies(result.thesis, chain_ids) and self._embedder is not None:
                claim_vec = self._embedder.encode(
                    result.thesis, convert_to_numpy=True, normalize_embeddings=True
                ).astype("float32")
                avg_nli = float(sum(nli_scores) / len(nli_scores)) if nli_scores else 0.0
                self._memory.upsert_throughline(
                    page_id=self._page_id,
                    claim_text=result.thesis,
                    claim_vec=claim_vec,
                    arxiv_ids=chain_ids,
                    triplet_keys=triplet_keys,
                    avg_nli=avg_nli,
                )
        except Exception:
            pass  # memory is always non-critical


def _thesis_qualifies(thesis: str, chain: List[str]) -> bool:
    """Return True if the thesis is non-trivial and has supporting evidence."""
    if not thesis or len(chain) < 2:
        return False
    t = thesis.lower().strip()
    junk_patterns = (
        "no thesis", "not enough", "insufficient", "n/a", "none",
        "unable to", "cannot", "no clear", "no single",
    )
    return not any(p in t for p in junk_patterns)
