"""
intent_extractor.py — Extract a structured ObjectiveFunction from a raw user query.

Core Thesis:
    A raw user query is underspecified for semantic retrieval.  This module uses
    Qwen3-1.7B (local Ollama) to extract a structured ObjectiveFunction
    (goal, domain, constraints) that is more precise for NLI entailment checking
    and utility-field search.

Workflow:
    1. Send raw query to Qwen3-1.7B via Ollama /api/chat
    2. Parse JSON response → ObjectiveFunction dataclass
    3. Fallback: use raw query as goal if LLM call or parse fails

Necessary Conditions:
    - Ollama running at http://127.0.0.1:11434
    - Model specified by DEFAULT_MODEL available in Ollama
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import List, Optional

import httpx

OLLAMA_BASE   = "http://127.0.0.1:11434"
DEFAULT_MODEL = "hf.co/unsloth/Qwen3-1.7B-GGUF:Qwen3-1.7B-Q6_K.gguf"

_SYSTEM = """\
You are an intent parser for research queries.
Extract a structured objective from the user's query.

Respond with ONLY a JSON object — no markdown fences, no prose:
{
  "goal": "<1-2 sentence statement of what the user wants to understand or find>",
  "domain": "<primary research domain, e.g. 'natural language processing'>",
  "constraints": ["<specific requirement or constraint if any>"]
}"""


@dataclass
class ObjectiveFunction:
    """Structured representation of the user's research intent."""

    goal:        str
    domain:      str        = ""
    constraints: List[str]  = field(default_factory=list)

    def as_text(self) -> str:
        """Flat text representation suitable for NLI premise comparison."""
        parts = [self.goal]
        if self.domain:
            parts.append(f"Domain: {self.domain}.")
        if self.constraints:
            parts.append("Requires: " + "; ".join(self.constraints) + ".")
        return " ".join(parts)


class IntentExtractor:
    """Extract ObjectiveFunction from a raw user query via local LLM."""

    def __init__(self, model: str = DEFAULT_MODEL, verbose: bool = False):
        self._model   = model
        self._verbose = verbose
        self._client  = httpx.Client(base_url=OLLAMA_BASE, timeout=60.0)

    def extract(self, query: str) -> ObjectiveFunction:
        """
        Extract structured objective from a query string.

        Args:
            query: Raw user search / research query.

        Returns:
            ObjectiveFunction; falls back to query-as-goal on parse failure.
        """
        payload = {
            "model":   self._model,
            "messages": [
                {"role": "system", "content": _SYSTEM},
                {"role": "user",   "content": f"Query: {query}"},
            ],
            "stream":  False,
            "options": {"num_predict": 1000, "temperature": 0.1},
        }
        try:
            resp = self._client.post("/api/chat", json=payload)
            resp.raise_for_status()
            raw = resp.json()["message"]["content"] or ""
        except Exception as exc:
            if self._verbose:
                print(f"[intent] LLM call failed: {exc}; using raw query.")
            return ObjectiveFunction(goal=query)

        # Strip <think> blocks
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
        raw = re.sub(r"<think>.*", "", raw, flags=re.DOTALL)  # unclosed think block
        raw = re.sub(r"</?think>", "", raw).strip()

        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            if self._verbose:
                print("[intent] No JSON found; using raw query as goal.")
            return ObjectiveFunction(goal=query)

        try:
            data = json.loads(match.group(0))
            obj  = ObjectiveFunction(
                goal        = str(data.get("goal", query) or query),
                domain      = str(data.get("domain", "") or ""),
                constraints = list(data.get("constraints", []) or []),
            )
            if self._verbose:
                print(f"[intent] goal={obj.goal[:80]}")
            return obj
        except (json.JSONDecodeError, ValueError):
            return ObjectiveFunction(goal=query)

    def close(self) -> None:
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
