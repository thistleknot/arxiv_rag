"""
syllogism_former.py — Form a thesis (throughline) from entailed premises via local LLM.

Core Thesis:
    Given pre-filtered entailed premises (selected by NLI cross-encoder) and the user's
    query + structured objective, form a concise thesis sentence that synthesises the
    through-line answer.  Premise selection is now owned by NLIEntailmentScorer;
    this module only forms the thesis and constructs the chain for display.

Architecture:
    - LLM: Qwen3-1.7B via Ollama (local, no API key required)
    - Inputs:  query, objective_text, entailed premises_by_paper, nli_scores
    - Outputs: SyllogismResult(thesis, chain, paper_scores)

Workflow:
    1. Build premise context from premises_by_paper (already NLI-filtered)
    2. Call Qwen3-1.7B: query + objective + premises → single thesis sentence
    3. Construct chain from entailed premises, keyed by NLI scores
    4. Return SyllogismResult with thesis, chain, paper_scores=nli_scores

Necessary Conditions:
    - Ollama running at http://127.0.0.1:11434
    - hf.co/unsloth/Qwen3-1.7B-GGUF:Qwen3-1.7B-Q6_K.gguf available via Ollama
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import httpx

OLLAMA_BASE          = "http://127.0.0.1:11434"
DEFAULT_THESIS_MODEL = "hf.co/unsloth/Qwen3-1.7B-GGUF:Qwen3-1.7B-Q6_K.gguf"

_THESIS_SYSTEM = """\
You synthesize research findings into a concise thesis using structured reasoning.

Given a query, objective, and premises (each with NLI scores E=entailment, C=contradiction):
1. DEDUCTIVE: Identify load-bearing premises (high E, low C). Discard contradictory or neutral ones.
2. INDUCTIVE: Find the common pattern or generalization across validated premises.
3. ABDUCTIVE: Form the most plausible single explanation that accounts for all validated findings.

Name the specific mechanism type (e.g., sparse attention, parameter distillation, agent decomposition). \
Include one key failure mode. Output ONLY one thesis sentence, max 40 words. No preamble. No explanation."""

_FALLBACK_NECESSITY = 0.5   # used when nli_scores are not provided


# ── Pydantic-free dataclasses ─────────────────────────────────────────────────

@dataclass
class ChainLink:
    """A single premise in the entailment chain."""
    premise_text:    str
    arxiv_id:        str
    position:        int           # 0 = highest NLI score (most entailed)
    necessity_score: float = 0.5   # NLI entailment probability [0, 1]


@dataclass
class SyllogismResult:
    """Full output of SyllogismFormer.form()."""
    thesis:       str
    chain:        List[ChainLink]  = field(default_factory=list)
    paper_scores: Dict[str, float] = field(default_factory=dict)

    @property
    def chain_arxiv_ids(self) -> List[str]:
        return [link.arxiv_id for link in self.chain]


# ── Prompt helpers ────────────────────────────────────────────────────────────

def _make_thesis_prompt(
    query:             str,
    premises_by_paper: Dict[str, List[str]],
    objective_text:    str = "",
    premise_scores:    Optional[dict] = None,
) -> str:
    lines = [f"Query: {query}"]
    if objective_text:
        lines.append(f"Objective: {objective_text}")
    lines.append("\nEntailed premises:")
    for arxiv_id, premises in premises_by_paper.items():
        ps_map = {}
        if premise_scores and arxiv_id in premise_scores:
            ps_map = {ps.premise: ps for ps in premise_scores[arxiv_id]}
        for premise in premises:
            ps = ps_map.get(premise)
            if ps:
                lines.append(f"  [{arxiv_id}] (E={ps.entailment:.2f} "
                             f"C={ps.contradiction:.2f}) {premise}")
            else:
                lines.append(f"  [{arxiv_id}] {premise}")
    lines.append("\nThesis:")
    return "\n".join(lines)


def _build_chain(
    premises_by_paper: Dict[str, List[str]],
    nli_scores:        Dict[str, float],
    premise_scores:    Optional[dict] = None,
) -> List[ChainLink]:
    """
    Construct chain from entailed premises, ordered by NLI score descending.
    Each paper contributes its highest-entailment premise (by NLI score).
    Falls back to longest premise when per-premise scores are unavailable.
    """
    best_by_paper: Dict[str, str] = {}
    for arxiv_id, premises in premises_by_paper.items():
        if not premises:
            continue
        if premise_scores and arxiv_id in premise_scores:
            scored = {ps.premise: ps.entailment for ps in premise_scores[arxiv_id]}
            best_by_paper[arxiv_id] = max(
                premises, key=lambda p: scored.get(p, 0.0),
            )
        else:
            best_by_paper[arxiv_id] = max(premises, key=lambda p: len(p))

    sorted_papers = sorted(
        best_by_paper.items(),
        key=lambda kv: nli_scores.get(kv[0], 0.0),
        reverse=True,
    )

    return [
        ChainLink(
            premise_text=premise,
            arxiv_id=arxiv_id,
            position=pos,
            necessity_score=nli_scores.get(arxiv_id, _FALLBACK_NECESSITY),
        )
        for pos, (arxiv_id, premise) in enumerate(sorted_papers)
    ]


# ── Main class ────────────────────────────────────────────────────────────────

class SyllogismFormer:
    """
    Forms a thesis from entailed premises using Qwen3-1.7B (local Ollama).

    In the full pipeline, premises_by_paper should already be filtered to only
    NLI-entailed papers.  When nli_scores is provided, they are used as
    paper_scores for downstream re-ranking.

    Usage:
        sf = SyllogismFormer(verbose=True)
        result = sf.form(query, entailed_premises, nli_scores=nli_dict,
                         objective_text=objective.as_text())
    """

    def __init__(
        self,
        model:   str  = DEFAULT_THESIS_MODEL,
        verbose: bool = False,
        # legacy kwargs accepted but ignored
        base_url: str = "",
        api_key:  str = "",
    ):
        self._model   = model
        self._verbose = verbose
        self._client  = httpx.Client(base_url=OLLAMA_BASE, timeout=120.0)

    def form(
        self,
        query:              str,
        premises_by_paper:  Dict[str, List[str]],
        utilities_by_paper: Optional[Dict[str, str]] = None,  # kept for API compat, unused
        nli_scores:         Optional[Dict[str, float]] = None,
        objective_text:     str = "",
        premise_scores:     Optional[dict] = None,
    ) -> SyllogismResult:
        """
        Form a thesis from (pre-filtered) premises.

        Args:
            query:              The user's raw search query.
            premises_by_paper:  arxiv_id → list of premise strings (NLI-filtered).
            utilities_by_paper: Unused; kept for backward compatibility.
            nli_scores:         arxiv_id → NLI entailment score.  Used as paper_scores
                                and to order the chain.  Defaults to 0.5 for all papers.
            objective_text:     Extracted ObjectiveFunction.as_text() — adds context.

        Returns:
            SyllogismResult with thesis, chain (ordered by NLI score), paper_scores.
        """
        populated = {k: v for k, v in premises_by_paper.items() if v}
        if not populated:
            return SyllogismResult(thesis="No entailed premises available.",
                                   chain=[], paper_scores={})

        effective_nli = nli_scores or {arxiv_id: _FALLBACK_NECESSITY
                                       for arxiv_id in populated}

        user_msg = _make_thesis_prompt(query, populated, objective_text,
                                       premise_scores=premise_scores)

        if self._verbose:
            n_premises = sum(len(v) for v in populated.values())
            print(f"[syllogism] Forming thesis from {n_premises} premises "
                  f"in {len(populated)} papers via {self._model}")

        thesis = self._call_llm(user_msg)
        chain  = _build_chain(populated, effective_nli,
                              premise_scores=premise_scores)

        if self._verbose:
            print(f"[syllogism] Thesis: {thesis}")
            print(f"[syllogism] Chain length: {len(chain)}")
            for link in chain:
                print(f"  [{link.position}] ({link.necessity_score:.2f}) "
                      f"{link.arxiv_id}: {link.premise_text[:80]}")

        return SyllogismResult(
            thesis=thesis,
            chain=chain,
            paper_scores=effective_nli,
        )

    def _call_llm(self, user_msg: str) -> str:
        """Call Qwen3-1.7B via Ollama; return cleaned thesis text."""
        payload = {
            "model":   self._model,
            "messages": [
                {"role": "system", "content": _THESIS_SYSTEM},
                {"role": "user",   "content": user_msg},
            ],
            "stream":  False,
            "options": {"num_predict": 1500, "temperature": 0.2},
        }
        try:
            resp = self._client.post("/api/chat", json=payload)
            resp.raise_for_status()
            raw = resp.json()["message"]["content"] or ""
        except Exception as exc:
            print(f"[syllogism] LLM call failed: {exc}", file=sys.stderr)
            return "Thesis formation failed."

        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
        raw = re.sub(r"<think>.*", "", raw, flags=re.DOTALL)  # unclosed think block
        raw = re.sub(r"</?think>", "", raw).strip()
        raw = re.sub(r"^(thesis\s*:?\s*)", "", raw, flags=re.IGNORECASE).strip()
        return raw or "No thesis generated."

    def close(self) -> None:
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
