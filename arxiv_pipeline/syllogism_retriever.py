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
    "title": 0.4,
    "abstract": 0.3,
    "utility": 0.3,
}

OLLAMA_BASE            = "http://127.0.0.1:11434"
_THROUGHLINE_MODEL     = "hf.co/unsloth/Qwen3-4B-128K-GGUF:Qwen3-4B-128K-Q6_K.gguf"

_THROUGHLINE_SYSTEM = """\
You are a systems engineer extracting implementable knowledge from research papers.

Your output has two parts:

PART 1 — Core Idea (2-3 sentences max):
State the shared mathematical or algorithmic principle across the papers in plain language. \
What problem does it solve and why does the approach work? Be specific, not generic.

PART 2 — Shared Mechanism (2-3 sentences):
State the common algorithmic approach across papers in plain English. \
Name the specific mechanism type (e.g., sparse attention, parameter distillation, agent decomposition). \
Identify one failure mode shared or implied across methods. \
No pseudocode. No equations. No variable names. Only output PART 1 and PART 2."""

_PAPER_ANGLE_SYSTEM = """\
You are extracting the core implementable idea from a single research paper.
Answer in exactly 2 lines. No preamble, no markdown, no explanation.

Line 1: Philosophy: <one sentence — what mathematical or algorithmic principle \
does this paper contribute? Name the mechanism, not the application domain.>
Line 2: Application: <one sentence — the primary mechanism or algorithm, ≤30 tokens; \
reproduce notation exactly as it appears in the paper; do not invent notation>"""

_DECISION_TREE_SYSTEM = """\
You are recommending which approach to use for a given task based on research papers.
Based on the synthesis provided, write a decision tree with 3-4 yes/no questions that guide selection.
Each leaf node must name a specific method or paper (use the arXiv IDs provided).
Write in plain text. No pseudocode. No equations. Max 200 words."""


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
    ):
        self._csv_path  = csv_path
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
        self._llm_client = httpx.Client(base_url=OLLAMA_BASE, timeout=240.0)

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
            _BLEND_WEIGHTS["title"] * title_scores
            + _BLEND_WEIGHTS["abstract"] * abstract_scores
            + _BLEND_WEIGHTS["utility"] * utility_scores
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

    def retrieve(self, query: str, top_k: int = 5, n_papers: int = 50) -> SyllogismRetrievalResult:
        """Execute full 9-stage syllogism retrieval pipeline.
        
        Stage 0: Extract intent from query (goal, domain, requirements).
        Stage 1: Semantic search intent against cached utility embeddings.
        Stage 2: Build utility ranking map (maintain order).
        Stage 3: NLI scoring — LLM judge ranks papers by utility relevance.
        Stage 4: Form syllogism — reason over entailed utilities.
        Stage 5: Rerank by blend (NLI + retrieval scores).
        Stage 6: Load paper markdowns from disk.
        Stage 7: Graph retriever context (optional, if available).
        Stage 8: LLM through-line synthesis over top papers.
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

        # ── Stage 1: Semantic search (empty stub for now) ──
        if self._verbose:
            print(f"[stage 1] Semantic search over utility index...")
        candidates: List[Any] = []
        utilities_map: Dict[str, str] = {}
        try:
            stage1_intent = result.objective or query
            candidates, utilities_map = self._semantic_search_blend(stage1_intent, n_papers)
            if self._verbose:
                print(f"[stage 1] Retrieved {len(candidates)} candidates")
        except Exception as e:
            if self._verbose:
                print(f"[stage 1] Semantic search failed: {e}")
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
            # Per-paper angle extraction
            for doc in result.papers[:top_k]:
                pid      = doc.metadata.get("paper_id", doc.doc_id)
                title    = doc.metadata.get("title", "")
                abstract = doc.metadata.get("abstract", "")[:800]
                user_msg = f"Title: {title}\n\nAbstract: {abstract}"
                payload  = {
                    "model":   _THROUGHLINE_MODEL,
                    "system":  _PAPER_ANGLE_SYSTEM,
                    "prompt":  user_msg,
                    "stream":  False,
                    "options": {"num_predict": 120, "temperature": 0.0},
                }
                resp = self._llm_client.post("/api/generate", json=payload)
                resp.raise_for_status()
                angle = resp.json().get("response", "").strip()
                result.paper_angles[pid] = angle

            # Through-line synthesis from all per-paper angles
            if result.paper_angles:
                angles_text = "\n\n".join(
                    f"[{pid}]\n{angle}" for pid, angle in result.paper_angles.items()
                )
                payload = {
                    "model":   _THROUGHLINE_MODEL,
                    "system":  _THROUGHLINE_SYSTEM,
                    "prompt":  f"Query: {result.query}\n\nPaper summaries:\n{angles_text}",
                    "stream":  False,
                    "options": {"num_predict": 400, "temperature": 0.2},
                }
                resp = self._llm_client.post("/api/generate", json=payload)
                resp.raise_for_status()
                result.through_line = resp.json().get("response", "").strip()

            # Decision tree from through-line
            if result.through_line:
                payload = {
                    "model":   _THROUGHLINE_MODEL,
                    "system":  _DECISION_TREE_SYSTEM,
                    "prompt":  f"Query: {result.query}\n\nSynthesis:\n{result.through_line}",
                    "stream":  False,
                    "options": {"num_predict": 400, "temperature": 0.2},
                }
                resp = self._llm_client.post("/api/generate", json=payload)
                resp.raise_for_status()
                result.decision_tree = resp.json().get("response", "").strip()
        except Exception as exc:
            if self._verbose:
                print(f"[stage 8] warning: {exc}")

        if self._verbose:
            print(f"[retrieve] Complete. Thesis: {result.thesis[:80]}...")

        return result
