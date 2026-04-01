"""
On-the-fly knowledge graph extraction via Ollama structured output.

Core Thesis:
    Instead of a pre-trained GATv2 model over static artifacts, extract
    entities and relations directly from paper text at query time using
    an LLM with constrained JSON output.  This approach (inspired by
    KNWLER — github.com/Orbifold/knwler) produces a fresh, paper-specific
    knowledge graph every time, cached per arxiv_id for reuse.

Pipeline (per paper):
    1. Chunk markdown text into ~1500-char windows
    2. For each chunk, ask qwen3.5:2b (Ollama structured output)
       to extract entities + relations
    3. Consolidate: deduplicate entities, merge relations
    4. Filter triplets by cosine similarity to the query
    5. Return top-k as "subject | predicate | object" lines

Usage:
    retriever = GraphRetriever()
    context = retriever.retrieve_context(
        "attention mechanism in transformers",
        paper_texts={"2401.12345": "# Transformer paper\n..."},
        top_k=15,
    )

    # Backward-compatible (no paper_texts → returns ""):
    context = retriever.retrieve_context("query", top_k=15, n_seeds=10)
"""

import json
import hashlib
import re
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

try:
    import ollama
    _OLLAMA_AVAILABLE = True
except ImportError:
    _OLLAMA_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False


# ── Cache directory ──────────────────────────────────────────────────────────
_CACHE_DIR = Path(__file__).resolve().parent / "kg_cache"

# ── Pydantic schemas for structured output ───────────────────────────────────
# Defined as plain dicts (JSON Schema) to avoid hard pydantic dependency in the
# extraction loop — the ollama client accepts raw JSON schema.

_EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string"},
                },
                "required": ["name", "type"],
            },
        },
        "relations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "target": {"type": "string"},
                    "relation_type": {"type": "string"},
                },
                "required": ["source", "target", "relation_type"],
            },
        },
    },
    "required": ["entities", "relations"],
}

_EXTRACTION_PROMPT = """\
Extract all important entities and their relations from the following academic text.
Focus on: methods, models, algorithms, datasets, metrics, concepts, and their relationships.

Rules:
- Each entity needs a name and a type (e.g. Method, Model, Dataset, Metric, Concept, Task, Architecture)
- Each relation needs source entity, target entity, and relation_type
- relation_type should be a short verb phrase (e.g. "uses", "outperforms", "is based on", "evaluates on")
- Be thorough — capture all meaningful relationships
- Keep entity names concise but specific

Text:
{chunk}"""


@dataclass
class PaperGraph:
    """Extracted knowledge graph for a single paper."""
    arxiv_id: str
    entities: List[dict] = field(default_factory=list)
    relations: List[dict] = field(default_factory=list)

    @property
    def triplets(self) -> List[Tuple[str, str, str]]:
        """Return relations as (source, relation_type, target) tuples."""
        return [
            (r["source"], r["relation_type"], r["target"])
            for r in self.relations
        ]


class GraphRetriever:
    """
    On-the-fly knowledge graph extractor and retriever.

    Uses Ollama structured output to extract entities and relations
    from paper text, then filters by query relevance.
    """

    def __init__(
        self,
        model: str = "qwen3.5:2b",
        ollama_host: str = "http://127.0.0.1:11434",
        chunk_size: int = 1500,
        chunk_overlap: int = 200,
        max_chunks_per_paper: int = 20,
        embed_model: str = "all-MiniLM-L6-v2",
        cache_dir: Optional[Path] = None,
        device: Optional[str] = None,
    ):
        self._model = model
        self._host = ollama_host
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._max_chunks = max_chunks_per_paper
        self._cache_dir = cache_dir or _CACHE_DIR
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        # In-memory cache: arxiv_id → PaperGraph
        self._mem_cache: Dict[str, PaperGraph] = {}

        # Ollama client
        if _OLLAMA_AVAILABLE:
            self._client = ollama.Client(host=self._host)
        else:
            self._client = None

        # Sentence transformer for query filtering
        self._embedder = None
        if _ST_AVAILABLE:
            try:
                self._embedder = SentenceTransformer(embed_model, device=device)
            except Exception:
                pass

    # ── Public interface (backward-compatible) ────────────────────────────────

    def retrieve_context(
        self,
        query: str,
        top_k: int = 15,
        n_seeds: int = 10,
        paper_texts: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Extract KG from paper_texts and return query-relevant triplets.

        Args:
            query:       Thesis or search query.
            top_k:       Max triplets to return.
            n_seeds:     Ignored (kept for backward compat).
            paper_texts: Dict[arxiv_id, markdown_content].
                         If None, returns empty string (graceful degrade).

        Returns:
            Newline-separated "subject | relation | object" string.
        """
        if not paper_texts:
            return ""
        triplets = self.retrieve(query, top_k=top_k, paper_texts=paper_texts)
        if not triplets:
            return ""
        return "\n".join(f"{s} | {p} | {o}" for s, p, o in triplets)

    def retrieve(
        self,
        query: str,
        top_k: int = 15,
        paper_texts: Optional[Dict[str, str]] = None,
    ) -> List[Tuple[str, str, str]]:
        """
        Extract and retrieve query-relevant triplets from paper texts.

        Returns:
            List of (subject, relation_type, target) tuples, ranked by
            relevance to query.
        """
        if not paper_texts or self._client is None:
            return []

        # 1. Extract graphs from each paper (cached)
        all_triplets: List[Tuple[str, str, str]] = []
        for arxiv_id, text in paper_texts.items():
            pg = self._extract_paper_graph(arxiv_id, text)
            all_triplets.extend(pg.triplets)

        if not all_triplets:
            return []

        # 2. Deduplicate
        seen = set()
        unique: List[Tuple[str, str, str]] = []
        for t in all_triplets:
            key = (t[0].lower().strip(), t[1].lower().strip(), t[2].lower().strip())
            if key not in seen:
                seen.add(key)
                unique.append(t)

        # 3. Rank by query relevance
        ranked = self._rank_triplets(query, unique, top_k)
        return ranked

    def retrieve_for_seeds(
        self,
        seed_texts: List[str],
        top_k: int = 20,
        paper_texts: Optional[Dict[str, str]] = None,
    ) -> str:
        """Multi-seed variant: merge query from all seeds."""
        combined_query = " ".join(seed_texts)
        return self.retrieve_context(
            combined_query, top_k=top_k, paper_texts=paper_texts,
        )

    def load_cached_triplets(self, arxiv_id: str) -> List[Tuple[str, str, str]]:
        """
        Return cached SPO triplets for a paper without live extraction.

        Checks memory cache first, then disk cache.  Returns an empty list if
        this paper has not been KG-extracted yet (run warm_kg_cache.py to
        pre-populate the cache offline).
        """
        if arxiv_id in self._mem_cache:
            return self._mem_cache[arxiv_id].triplets

        cache_path = self._cache_dir / f"{arxiv_id.replace('/', '_')}.json"
        if not cache_path.exists():
            return []

        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            pg = PaperGraph(
                arxiv_id=arxiv_id,
                entities=data.get("entities", []),
                relations=data.get("relations", []),
            )
            self._mem_cache[arxiv_id] = pg
            return pg.triplets
        except (json.JSONDecodeError, KeyError):
            return []

    def build_context_from_cache(
        self,
        arxiv_ids: List[str],
        query: str,
        top_k: int = 15,
    ) -> str:
        """
        Build a graph-context string from cached triplets only.

        No live Ollama extraction is performed.  Papers without a cache entry
        are silently skipped.  Returns a newline-separated "s | p | o" string
        ranked by query relevance (or "" when no cached data exists).
        """
        all_triplets: List[Tuple[str, str, str]] = []
        for aid in arxiv_ids:
            all_triplets.extend(self.load_cached_triplets(aid))

        if not all_triplets:
            return ""

        # Deduplicate
        seen: set = set()
        unique: List[Tuple[str, str, str]] = []
        for t in all_triplets:
            key = (t[0].lower().strip(), t[1].lower().strip(), t[2].lower().strip())
            if key not in seen:
                seen.add(key)
                unique.append(t)

        ranked = self._rank_triplets(query, unique, top_k)
        return "\n".join(f"{s} | {p} | {o}" for s, p, o in ranked)

    # ── Paper graph extraction ────────────────────────────────────────────────

    def _extract_paper_graph(self, arxiv_id: str, text: str) -> PaperGraph:
        """Extract entities and relations from a paper, with caching."""
        # Check memory cache
        if arxiv_id in self._mem_cache:
            return self._mem_cache[arxiv_id]

        # Check disk cache
        cache_path = self._cache_dir / f"{arxiv_id.replace('/', '_')}.json"
        if cache_path.exists():
            try:
                data = json.loads(cache_path.read_text(encoding="utf-8"))
                pg = PaperGraph(
                    arxiv_id=arxiv_id,
                    entities=data.get("entities", []),
                    relations=data.get("relations", []),
                )
                self._mem_cache[arxiv_id] = pg
                return pg
            except (json.JSONDecodeError, KeyError):
                pass

        # Extract fresh
        chunks = self._chunk_text(text)
        all_entities = []
        all_relations = []

        for chunk in chunks[:self._max_chunks]:
            extraction = self._extract_chunk(chunk)
            if extraction:
                all_entities.extend(extraction.get("entities", []))
                all_relations.extend(extraction.get("relations", []))

        # Consolidate (deduplicate entities, normalize names)
        entities, relations = self._consolidate(all_entities, all_relations)

        pg = PaperGraph(arxiv_id=arxiv_id,
                        entities=entities, relations=relations)

        # Cache to disk
        try:
            cache_path.write_text(
                json.dumps({"entities": entities, "relations": relations},
                           ensure_ascii=False, indent=1),
                encoding="utf-8",
            )
        except OSError:
            pass

        self._mem_cache[arxiv_id] = pg
        return pg

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        # Strip markdown artifacts that add noise
        text = re.sub(r"!\[.*?\]\(.*?\)", "", text)       # images
        text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)  # links → text
        text = re.sub(r"#{1,6}\s*", "", text)              # headings
        text = re.sub(r"\n{3,}", "\n\n", text)             # excessive newlines

        chunks = []
        start = 0
        while start < len(text):
            end = start + self._chunk_size
            # Try to break at sentence boundary
            if end < len(text):
                # Look for ". " within last 200 chars of chunk
                break_zone = text[max(start, end - 200):end]
                last_period = break_zone.rfind(". ")
                if last_period != -1:
                    end = max(start, end - 200) + last_period + 2
            chunk = text[start:end].strip()
            if len(chunk) > 50:  # skip tiny fragments
                chunks.append(chunk)
            start = end - self._chunk_overlap
            if start >= len(text):
                break
        return chunks

    def _extract_chunk(self, chunk: str) -> Optional[dict]:
        """Call Ollama to extract entities + relations from a single chunk."""
        if self._client is None:
            return None
        try:
            resp = self._client.chat(
                model=self._model,
                messages=[{
                    "role": "user",
                    "content": _EXTRACTION_PROMPT.format(chunk=chunk),
                }],
                format=_EXTRACTION_SCHEMA,
                options={"num_predict": 8192, "temperature": 0.1},
            )
            raw = resp.message.content
            # Strip think blocks before JSON parse (Qwen3 chain-of-thought)
            raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
            raw = re.sub(r"<think>.*", "", raw, flags=re.DOTALL)  # unclosed think block
            raw = re.sub(r"</?think>", "", raw).strip()
            return json.loads(raw)
        except Exception:
            return None

    # ── Consolidation ─────────────────────────────────────────────────────────

    def _consolidate(
        self,
        entities: List[dict],
        relations: List[dict],
    ) -> Tuple[List[dict], List[dict]]:
        """Deduplicate entities and normalize relation endpoints."""
        # Build canonical name mapping (lowercase → first-seen casing)
        canonical: Dict[str, str] = {}
        unique_entities: Dict[str, dict] = {}

        for e in entities:
            name = e.get("name", "").strip()
            if not name:
                continue
            key = name.lower()
            if key not in canonical:
                canonical[key] = name
                unique_entities[key] = e

        # Normalize relation endpoints to canonical names
        unique_relations = []
        seen_rels = set()
        for r in relations:
            src = r.get("source", "").strip()
            tgt = r.get("target", "").strip()
            rtype = r.get("relation_type", "").strip()
            if not src or not tgt or not rtype:
                continue
            src_canon = canonical.get(src.lower(), src)
            tgt_canon = canonical.get(tgt.lower(), tgt)
            rel_key = (src_canon.lower(), rtype.lower(), tgt_canon.lower())
            if rel_key not in seen_rels:
                seen_rels.add(rel_key)
                unique_relations.append({
                    "source": src_canon,
                    "target": tgt_canon,
                    "relation_type": rtype,
                })

        return list(unique_entities.values()), unique_relations

    # ── Query-relevance ranking ───────────────────────────────────────────────

    def _rank_triplets(
        self,
        query: str,
        triplets: List[Tuple[str, str, str]],
        top_k: int,
    ) -> List[Tuple[str, str, str]]:
        """Rank triplets by semantic similarity to query."""
        if not triplets:
            return []

        if self._embedder is not None:
            return self._rank_by_embedding(query, triplets, top_k)
        return self._rank_by_keyword(query, triplets, top_k)

    def _rank_by_embedding(
        self,
        query: str,
        triplets: List[Tuple[str, str, str]],
        top_k: int,
    ) -> List[Tuple[str, str, str]]:
        """Rank using sentence transformer cosine similarity."""
        triplet_texts = [f"{s} {p} {o}" for s, p, o in triplets]
        query_emb = self._embedder.encode([query], normalize_embeddings=True)
        trip_embs = self._embedder.encode(triplet_texts, normalize_embeddings=True)
        scores = (query_emb @ trip_embs.T).flatten()
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [triplets[i] for i in top_idx]

    def _rank_by_keyword(
        self,
        query: str,
        triplets: List[Tuple[str, str, str]],
        top_k: int,
    ) -> List[Tuple[str, str, str]]:
        """Fallback: rank by keyword overlap with query."""
        query_words = set(query.lower().split())

        def score(t: Tuple[str, str, str]) -> int:
            words = set(f"{t[0]} {t[1]} {t[2]}".lower().split())
            return len(query_words & words)

        ranked = sorted(triplets, key=score, reverse=True)
        return ranked[:top_k]
