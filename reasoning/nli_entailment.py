

"""
nli_entailment.py — Score premise-to-objective entailment via NLI cross-encoder.

Core Thesis:
    For each (objective, premise) pair, an NLI cross-encoder returns a 3-class
    probability vector.  We extract the entailment probability as the relevance
    signal.  Per-paper score = max entailment probability across all that paper's
    premises.  Papers above a threshold are forwarded to thesis formation.

Model:
    cross-encoder/nli-deberta-v3-small — ~85 MB, 3-class NLI trained on MNLI+SNLI.
    Label order: [contradiction=0, entailment=1, neutral=2]

Workflow:
    1. Flatten all (arxiv_id, premise_text) pairs from triplets_by_paper
    2. Run CrossEncoder.predict([(objective, premise), ...])
    3. Softmax → probabilities; extract entailment column
    4. Aggregate: max entailment probability per paper
    5. judge_entailed() presents ranked scores to LLM judge for entailment decision
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import httpx
import numpy as np


@dataclass
class PremiseScore:
    """Per-premise 3-class NLI breakdown (contradiction / entailment / neutral)."""
    arxiv_id: str
    premise: str
    entailment: float
    contradiction: float
    neutral: float

    @property
    def classification(self) -> str:
        """Argmax classification: entailment, contradiction, or neutral."""
        scores = {'entailment': self.entailment, 'contradiction': self.contradiction,
                  'neutral': self.neutral}
        return max(scores, key=scores.get)


NLI_MODEL         = "cross-encoder/nli-deberta-v3-small"
# Label ordering for nli-deberta-v3-small: [contradiction=0, entailment=1, neutral=2]
_ENTAIL_IDX       = 1
DEFAULT_THRESHOLD = 0.4   # retained for filter_entailed(); not used by judge_entailed()

OLLAMA_BASE         = "http://127.0.0.1:11434"
DEFAULT_JUDGE_MODEL = "hf.co/unsloth/Qwen3-1.7B-GGUF:Qwen3-1.7B-Q6_K.gguf"

_JUDGE_SYSTEM = """\
You are an entailment judge performing deductive premise evaluation.

Each premise has 3-class NLI scores:
- E (Entailment): premise supports the objective — load-bearing (TRUE)
- C (Contradiction): premise conflicts with the objective — invalidates (FALSE)
- N (Neutral): premise is unrelated — inconsequential (skip)

Rules:
1. Exclude papers where the strongest premise is contradictory (C > E).
2. Include papers with at least one load-bearing premise (E > C and E > N).

Return ONLY a JSON array of arxiv_id strings for papers with load-bearing premises.
Example: ["1905.07854", "1701.06538"]
If no premises are load-bearing, return: []"""


class NLIEntailmentScorer:
    """
    Score each paper by the maximum NLI entailment probability of any of its
    premises against a given objective statement, then let an LLM judge decide
    which premises are genuinely entailed.

    Usage:
        scorer = NLIEntailmentScorer(verbose=True)
        nli_scores = scorer.score(objective.as_text(), triplets_by_paper)
        entailed   = scorer.judge_entailed(objective.as_text(), triplets_by_paper, nli_scores)
    """

    def __init__(
        self,
        model:       str   = NLI_MODEL,
        threshold:   float = DEFAULT_THRESHOLD,
        judge_model: str   = DEFAULT_JUDGE_MODEL,
        verbose:     bool  = False,
    ):
        self._model_name   = model
        self._threshold    = threshold
        self._judge_model  = judge_model
        self._verbose      = verbose
        self._encoder      = None   # lazy-load; sentence-transformers import is slow
        self._judge_client = None   # lazy-load for Ollama httpx client

    def _load(self) -> None:
        if self._encoder is None:
            from sentence_transformers import CrossEncoder
            if self._verbose:
                print(f"[nli] Loading cross-encoder: {self._model_name}", flush=True)
            self._encoder = CrossEncoder(self._model_name)
            if self._verbose:
                print("[nli] Cross-encoder ready.", flush=True)

    def score(
        self,
        objective_text:    str,
        premises_by_paper: Dict[str, List[str]],
    ) -> Dict[str, float]:
        """
        Return per-paper max entailment probability against objective_text.

        Args:
            objective_text:    The structured objective as a plain-text statement.
            premises_by_paper: arxiv_id → list of premise strings.

        Returns:
            Dict[arxiv_id, float] — entailment probability in [0, 1].
        """
        self._load()

        # Flatten to (paper_id, premise_text) pairs
        pairs: List[Tuple[str, str]] = []
        for arxiv_id, premises in premises_by_paper.items():
            for premise in premises:
                pairs.append((arxiv_id, premise))

        if not pairs:
            return {}

        sentence_pairs = [[objective_text, premise] for _, premise in pairs]

        # predict() returns raw logits (apply_softmax depends on model config)
        raw   = np.array(self._encoder.predict(sentence_pairs), dtype=np.float32)
        if raw.ndim == 1:
            # Single pair edge case
            raw = raw[np.newaxis, :]

        # Manual softmax for stability
        raw   = raw - raw.max(axis=1, keepdims=True)
        exp_r = np.exp(raw)
        probs = exp_r / exp_r.sum(axis=1, keepdims=True)
        entail_probs = probs[:, _ENTAIL_IDX]

        # Store per-premise 3-class breakdown for downstream use
        self._premise_scores: Dict[str, List[PremiseScore]] = {}
        for (arxiv_id, premise), prob_vec in zip(pairs, probs):
            ps = PremiseScore(
                arxiv_id=arxiv_id,
                premise=premise,
                contradiction=float(prob_vec[0]),
                entailment=float(prob_vec[1]),
                neutral=float(prob_vec[2]),
            )
            self._premise_scores.setdefault(arxiv_id, []).append(ps)

        # Aggregate: max entailment score per paper
        paper_scores: Dict[str, float] = {}
        for (arxiv_id, _), prob in zip(pairs, entail_probs):
            prob_f = float(prob)
            if arxiv_id not in paper_scores or paper_scores[arxiv_id] < prob_f:
                paper_scores[arxiv_id] = prob_f

        if self._verbose:
            surviving = sum(1 for v in paper_scores.values() if v >= self._threshold)
            print(
                f"[nli] {len(paper_scores)} papers scored; "
                f"{surviving} survive threshold={self._threshold:.2f}",
                flush=True,
            )

        return paper_scores

    def filter_entailed(
        self,
        premises_by_paper: Dict[str, List[str]],
        paper_scores:      Dict[str, float],
    ) -> Dict[str, List[str]]:
        """Return only papers whose entailment score is >= threshold (legacy)."""
        return {
            arxiv_id: premises
            for arxiv_id, premises in premises_by_paper.items()
            if paper_scores.get(arxiv_id, 0.0) >= self._threshold
        }

    @property
    def premise_scores(self) -> Dict[str, list]:
        """Per-premise 3-class NLI scores from the last score() call."""
        return getattr(self, '_premise_scores', {})

    # ── LLM-as-judge entailment ──────────────────────────────────────────────

    def _load_judge(self) -> None:
        if self._judge_client is None:
            self._judge_client = httpx.Client(base_url=OLLAMA_BASE, timeout=240.0)

    def judge_entailed(
        self,
        objective_text:    str,
        premises_by_paper: Dict[str, List[str]],
        paper_scores:      Dict[str, float],
    ) -> Dict[str, List[str]]:
        """
        LLM-as-judge entailment: present NLI-ranked premises to Qwen3-1.7B,
        let the LLM decide which papers' premises entail the objective.

        NLI scores serve as ranked evidence — no hard threshold gate.
        Falls back to top-1 by NLI score if LLM returns empty/unparseable.
        """
        self._load_judge()

        if not paper_scores:
            return premises_by_paper

        # Rank papers by NLI score descending
        ranked = sorted(paper_scores.items(), key=lambda kv: kv[1], reverse=True)

        # Build prompt with per-premise 3-class NLI evidence
        lines = [f"Objective: {objective_text}", ""]
        ps_map = getattr(self, '_premise_scores', {})
        for arxiv_id, score in ranked:
            premises = premises_by_paper.get(arxiv_id, [])
            detail = {ps.premise: ps for ps in ps_map.get(arxiv_id, [])}
            lines.append(f"Paper [{arxiv_id}]:")
            for premise in premises:
                ps = detail.get(premise)
                if ps:
                    lines.append(f"  - [{ps.classification.upper()}] "
                                 f"(E={ps.entailment:.2f} C={ps.contradiction:.2f} "
                                 f"N={ps.neutral:.2f}) {premise}")
                else:
                    lines.append(f"  - (NLI: {score:.3f}) {premise}")
            lines.append("")
        lines.append("Which papers have load-bearing premises that entail the objective?")

        payload = {
            "model":    self._judge_model,
            "messages": [
                {"role": "system", "content": _JUDGE_SYSTEM},
                {"role": "user",   "content": "\n".join(lines)},
            ],
            "stream":  False,
            "options": {"num_predict": 512, "temperature": 0.1},
        }

        try:
            resp = self._judge_client.post("/api/chat", json=payload)
            resp.raise_for_status()
            raw = resp.json()["message"]["content"] or ""
        except Exception as exc:
            if self._verbose:
                print(f"[nli-judge] LLM call failed: {exc}", flush=True)
            top_id = ranked[0][0]
            return {top_id: premises_by_paper[top_id]}

        # Strip think blocks, parse JSON array
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
        raw = re.sub(r"<think>.*", "", raw, flags=re.DOTALL)  # unclosed think block
        raw = re.sub(r"</?think>", "", raw).strip()

        selected_ids: list = []
        try:
            match = re.search(r'\[.*?\]', raw, re.DOTALL)
            if match:
                selected_ids = json.loads(match.group())
        except (json.JSONDecodeError, TypeError):
            pass

        # Filter to valid IDs present in premises_by_paper
        valid_ids = [
            aid for aid in selected_ids
            if isinstance(aid, str) and aid in premises_by_paper
        ]

        if self._verbose:
            print(f"[nli-judge] LLM selected {len(valid_ids)}/{len(ranked)} papers: "
                  f"{valid_ids}", flush=True)
            if not valid_ids:
                print(f"[nli-judge] Raw LLM response: {raw[:200]}", flush=True)

        if not valid_ids:
            top_id = ranked[0][0]
            if self._verbose:
                print(f"[nli-judge] Fallback to top-1 by NLI: {top_id}", flush=True)
            return {top_id: premises_by_paper[top_id]}

        return {aid: premises_by_paper[aid] for aid in valid_ids}

    def rank_utilities(
        self,
        intent_text: str,
        utilities_by_paper: Dict[str, str],
    ) -> Tuple[Dict[str, List[str]], Dict[str, float]]:
        """
        LLM-as-judge utility ranking: present numbered utility strings and let
        the judge emit ranked paper numbers, dropping non-entailed papers.

        The judge may deliberate freely before a '---' separator line.
        After '---', ranked integers (1-based) are expected one per line.
        Falls back to full-response integer scan if separator is absent,
        and to rank-1 if no valid numbers are found.

        Args:
            intent_text:        User intent / objective as plain text.
            utilities_by_paper: Dict[arxiv_id, utility_string] in cosine-rank order.

        Returns:
            entailed:    Dict[arxiv_id, List[str]] — selected papers mapped to
                         their utility string (wrapped in a list for API compat).
            rank_scores: Dict[arxiv_id, float]     — normalised rank score;
                         1.0 for the top-ranked paper, decaying to 1/N.
        """
        self._load_judge()

        if not utilities_by_paper:
            return {}, {}

        items = list(utilities_by_paper.items())   # preserves cosine-rank order
        N = len(items)
        numbered_lines = "\n".join(
            f"{i + 1}. {utility}" for i, (_, utility) in enumerate(items)
        )

        prompt = (
            "You are a relevance judge for research papers.\n\n"
            f"Intent: {intent_text}\n\n"
            f"Papers (numbered 1-{N}):\n{numbered_lines}\n\n"
            "Which papers directly satisfy or entail the intent?\n"
            "Briefly deliberate, then output the relevant paper numbers ranked\n"
            "from most to least relevant, one number per line, after a line\n"
            "containing only '---'.\n"
            "If none are relevant, output:\n---\nNONE\n"
        )

        raw = ""
        try:
            resp = self._judge_client.post(
                "/api/chat",
                json={
                    "model":    self._judge_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream":   False,
                    "options":  {"num_predict": 512, "temperature": 0.1},
                },
            )
            resp.raise_for_status()
            raw = resp.json()["message"]["content"] or ""
        except Exception as exc:
            if self._verbose:
                print(f"[rank-judge] LLM call failed: {exc}", flush=True)

        if self._verbose:
            print(f"[rank-judge] raw={raw[:300]!r}", flush=True)

        # Parse integers after '---'; fall back to full-response scan
        search_text = raw.split("---", 1)[1] if "---" in raw else raw
        found: List[int] = []
        seen: set = set()
        for m in re.finditer(r'\b(\d+)\b', search_text):
            n = int(m.group(1))
            if 1 <= n <= N and n not in seen:
                found.append(n)
                seen.add(n)

        if not found and "---" in raw:
            # separator present but no valid numbers after it — scan full response
            for m in re.finditer(r'\b(\d+)\b', raw):
                n = int(m.group(1))
                if 1 <= n <= N and n not in seen:
                    found.append(n)
                    seen.add(n)

        if not found:
            found = [1]
            if self._verbose:
                print("[rank-judge] No valid numbers found; falling back to rank-1",
                      flush=True)

        if self._verbose:
            print(f"[rank-judge] Selected {len(found)}/{N}: {found}", flush=True)

        entailed: Dict[str, List[str]] = {}
        rank_scores: Dict[str, float] = {}
        for rank_pos, num in enumerate(found):
            arxiv_id, utility = items[num - 1]
            entailed[arxiv_id] = [utility]
            rank_scores[arxiv_id] = float(N - rank_pos) / N   # 1.0 at rank-1, decays

        return entailed, rank_scores
