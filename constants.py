"""constants.py — single source of truth for arxiv RAG pipeline constants.

Skill specifications these constants implement:
  agentic_kg_memory : .copilot/skills/agentic_kg_memory/SKILL.md
  memory-bank       : .copilot/skills/memory-bank/SKILL.md

When the skill specs change, update this file. All pipeline modules reference
these names directly rather than re-defining them.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ── NLI model ─────────────────────────────────────────────────────────────────
# (agentic_kg_memory: Triplet Layer — cross-encoder for 3-class entailment)
NLI_MODEL      = "cross-encoder/nli-deberta-v3-small"
NLI_ENTAIL_IDX = 1  # DeBERTa label order: contradiction=0, entailment=1, neutral=2

# ── Epistemic weights ─────────────────────────────────────────────────────────
# (agentic_kg_memory: Triplet Confidence — extraction-time priors)
EPISTEMIC_OBSERVED = 1.0  # directly observed fact
EPISTEMIC_INFERRED = 0.5  # derived or inferred fact

# ── MemRL confidence update ───────────────────────────────────────────────────
# (agentic_kg_memory: MemRL section)
# Q_new = Q_old + MEMRL_ALPHA * (r_nli - Q_old)
MEMRL_ALPHA          = 0.1
TRIPLET_REINFORCE_TAU = 0.6  # NLI score at or above which confidence is reinforced
TRIPLET_WEAKEN_TAU    = 0.3  # NLI score at or below which confidence is weakened
# Neutral band (WEAKEN_TAU, REINFORCE_TAU): no update

# ── BM25 retrieval ────────────────────────────────────────────────────────────
# (agentic_kg_memory: Ranking Surface — triplet_score = bm25_score * confidence)
BM25_TOP_K    = 50
BM25_MIN_SCORE = 0.0

# ── Confidence persistence ────────────────────────────────────────────────────
KG_CONFIDENCE_DB = ROOT / "graph" / "triplet_confidence.sqlite3"

# ── Ollama endpoint ───────────────────────────────────────────────────────────
# Shared by all Ollama-backed components (intent extractor, syllogism former,
# NLI judge, through-line synthesis).
OLLAMA_BASE = "http://127.0.0.1:11434"

# ── Syllogism pipeline ────────────────────────────────────────────────────────
THROUGHLINE_MODEL = "hf.co/unsloth/Qwen3-4B-128K-GGUF:Qwen3-4B-128K-Q6_K.gguf"

# Blend weights for title/abstract/utility cosine retrieval (Layer 1 tuned).
BLEND_WEIGHTS = {
    "title":    0.3237,
    "abstract": 0.5803,
    "utility":  0.096,
}
