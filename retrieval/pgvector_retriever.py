"""
PGVector GIST Retriever: PostgreSQL/pgvector Backend

=============================================================================
OVERVIEW
=============================================================================

This module implements the GISTRetriever interface for PostgreSQL with pgvector.

Storage:
  - Dense vectors: vector(dim) with IVFFlat + cosine
  - Sparse vectors: sparsevec for BM25

Dependencies:
  - psycopg2: PostgreSQL driver
  - pgvector extension in PostgreSQL
  - model2vec: Dense embeddings (or compatible encoder)
  - sklearn: Similarity computations

=============================================================================
SCHEMA
=============================================================================

CREATE TABLE {table_name} (
    id SERIAL PRIMARY KEY,
    chunk_id TEXT UNIQUE NOT NULL,
    paper_id TEXT NOT NULL,
    section_idx INTEGER NOT NULL,
    chunk_idx INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding vector({dim}) NOT NULL,
    bm25_sparse sparsevec({vocab_size}) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX ON {table_name} USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX ON {table_name} (paper_id);

=============================================================================
USAGE
=============================================================================

from pgvector_retriever import PGVectorRetriever, PGVectorConfig

config = PGVectorConfig(
    db_host="localhost",
    db_port=5432,
    db_name="mydb",
    table_name="chunks"
)

retriever = PGVectorRetriever(config)
results = retriever.search("my query", top_k=21)

=============================================================================
"""

import math
import msgpack
import re
import threading
import time
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set, Any
from collections import Counter, defaultdict

import mmh3
import psycopg2
import psycopg2.extras
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector, SparseVector

from gist_retriever import (
    GISTRetriever, GISTConfig, RetrievedDoc, RetrievedGroup, RetrievedPaper,
    gist_select, compute_rrf_score, format_results_markdown, format_groups_markdown,
    format_papers_markdown
)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class PGVectorConfig(GISTConfig):
    """
    Configuration for PGVector GIST retriever.
    
    Inherits all GIST pipeline settings and adds PostgreSQL-specific config.
    """
    # Database connection
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "langchain"
    db_user: str = "langchain"
    db_password: str = "langchain"
    
    # Table
    table_name: str = "arxiv_chunks"
    
    # Embeddings
    embedding_dim: int = 256
    embedding_model: str = "./qwen3_static_embeddings"  # Model2Vec Qwen3 256d
    use_full_embed: bool = False  # If True, disable model2vec and use full sentence embeddings
    
    # BM25 (signed hashing — no vocab bootstrap)
    n_buckets: int = 16_000          # must match sparsevec(N) in schema
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    # Commit scheduler
    commit_size_threshold: int = 5_000   # flush staging when this many docs pending
    commit_max_age_seconds: int = 300    # flush at least every N seconds


# =============================================================================
# BM25 Schema SQL
# =============================================================================

BM25_SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

-- Committed corpus truth
CREATE TABLE IF NOT EXISTS bm25_global_stats (
    id           INT PRIMARY KEY DEFAULT 1,
    n_docs       INT     NOT NULL DEFAULT 0,
    total_tokens BIGINT  NOT NULL DEFAULT 0,
    updated_at   TIMESTAMPTZ     DEFAULT now()
);
INSERT INTO bm25_global_stats (id) VALUES (1) ON CONFLICT DO NOTHING;

-- Per-bucket IDF source (bucket = hash(term) % N_BUCKETS)
CREATE TABLE IF NOT EXISTS bm25_term_global (
    bucket    INT PRIMARY KEY,
    doc_freq  INT NOT NULL DEFAULT 0
);

-- Staging table: UNLOGGED = no WAL (safe if source is replayable)
CREATE UNLOGGED TABLE IF NOT EXISTS bm25_incoming (
    doc_id      TEXT    PRIMARY KEY,
    content     TEXT    NOT NULL,
    token_count INT     NOT NULL,
    bm25_sparse sparsevec(16000)
);
"""


# =============================================================================
# Hashing + Tokenization
# =============================================================================

def hash_token(term: str) -> Tuple[int, float]:
    """
    Returns (bucket, sign).
    Seed 0 → bucket assignment, Seed 1 → sign ∈ {+1, -1}.
    Colliding terms with opposite signs cancel rather than compound
    (unbiased estimator of inner product, Weinberger et al. 2009).
    """
    bucket = mmh3.hash(term, seed=0, signed=False) % 16_000
    sign   = 1.0 if mmh3.hash(term, seed=1, signed=True) >= 0 else -1.0
    return bucket, sign


def tokenize_bm25(text: str) -> List[str]:
    """
    BM25 tokenizer: lowercase + split on whitespace.
    MUST be identical at index time and query time.
    """
    return text.lower().split()


# =============================================================================
# BM25 Hash Manager (ANOVA partition model)
# =============================================================================

class BM25HashManager:
    """
    Owns BM25 corpus statistics via ANOVA-style partition:

      stats_effective = global_table + _pending_deltas

    Entities:
      bm25_global_stats — committed corpus truth (N, total_tokens)
      bm25_term_global  — committed bucket doc_freq (IDF source)
      bm25_incoming     — staged docs (sparsevec computed, not yet global)

    Predicates:
      stage_insert(docs)  → compute sparsevecs, stage to bm25_incoming
      stage_delete(ids)   → accumulate -deltas in Python
      commit_batch()      → collapse all pending → global atomically
      search(query)       → effective stats → query sparsevec → <#> retrieval

    Signed hashing: hash(term) % N_BUCKETS IS the vocabulary — no vocab table,
    no corpus bootstrap, no OOV.
    """

    def __init__(self, conn, k1: float = 1.5, b: float = 0.75,
                 n_buckets: int = 16_000, table_name: str = 'documents'):
        self.conn       = conn
        self.k1         = k1
        self.b          = b
        self.n_buckets  = n_buckets
        self.table_name = table_name
        self._reset_pending()

    # -----------------------------------------------------------------------
    # Effective stats (global + pending)
    # -----------------------------------------------------------------------

    def effective_stats(self) -> Dict:
        """Live corpus stats folding in uncommitted pending deltas."""
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT n_docs, total_tokens FROM bm25_global_stats WHERE id = 1"
            )
            n, tok = cur.fetchone()
        n_eff   = max(n   + self._pending_n_delta,    1)
        tok_eff = max(tok + self._pending_token_delta, 1)
        return {'n': n_eff, 'avgdl': tok_eff / n_eff}

    def get_batch_effective_df(self, buckets: List[int]) -> Dict[int, int]:
        """One DB round-trip for all buckets; folds pending deltas in Python."""
        if not buckets:
            return {}
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT bucket, doc_freq FROM bm25_term_global WHERE bucket = ANY(%s)",
                (buckets,)
            )
            df_map = {row[0]: row[1] for row in cur.fetchall()}
        for b in buckets:
            effective = max(df_map.get(b, 0) + self._pending_bucket_delta.get(b, 0), 0)
            if effective > 0:
                df_map[b] = effective
            elif b in df_map:
                del df_map[b]
        return df_map

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _build_bucket_tf(self, tokens: List[str]) -> Dict[int, float]:
        """Accumulate signed TF per bucket."""
        bucket_tf: Dict[int, float] = defaultdict(float)
        for term in tokens:
            bucket, sign = hash_token(term)
            bucket_tf[bucket] += sign
        return dict(bucket_tf)

    # -----------------------------------------------------------------------
    # Vectorize
    # -----------------------------------------------------------------------

    def vectorize_doc(self, tokens: List[str], stats: Dict) -> SparseVector:
        """
        Full BM25: TF saturation + IDF + document length normalization.
        Uses effective stats for IDF accuracy within staging window.
        """
        n, avgdl  = stats['n'], stats['avgdl']
        doc_len   = len(tokens)
        bucket_tf = self._build_bucket_tf(tokens)
        df_map    = self.get_batch_effective_df(list(bucket_tf.keys()))

        indices, values = [], []
        for bucket, signed_tf in bucket_tf.items():
            tf     = abs(signed_tf)
            df     = df_map.get(bucket, 0)
            idf    = math.log((n - df + 0.5) / (df + 0.5) + 1)
            tf_sat = (tf * (self.k1 + 1)) / (
                tf + self.k1 * (1 - self.b + self.b * doc_len / avgdl)
            )
            score = idf * tf_sat
            if score > 0:
                indices.append(bucket)
                values.append(math.copysign(score, signed_tf))

        return SparseVector(dict(zip(indices, values)), self.n_buckets)

    def vectorize_query(self, tokens: List[str], stats: Dict) -> SparseVector:
        """
        Query vectorization: IDF only.
        No TF saturation or length norm — queries are short.
        """
        n         = stats['n']
        bucket_tf = self._build_bucket_tf(tokens)
        df_map    = self.get_batch_effective_df(list(bucket_tf.keys()))

        indices, values = [], []
        for bucket, signed_tf in bucket_tf.items():
            df  = df_map.get(bucket, 0)
            idf = math.log((n - df + 0.5) / (df + 0.5) + 1)
            if idf > 0:
                indices.append(bucket)
                values.append(math.copysign(idf, signed_tf))

        return SparseVector(dict(zip(indices, values)), self.n_buckets)

    # -----------------------------------------------------------------------
    # Stage insert
    # -----------------------------------------------------------------------

    def stage_insert(self, docs: List[Dict]):
        """
        docs: [{'id': str, 'content': str}, ...]

        Computes sparsevecs using effective stats, stages to bm25_incoming,
        accumulates Python-side deltas. Does NOT touch global tables.
        """
        stats    = self.effective_stats()
        doc_rows = []

        for doc in docs:
            tokens = tokenize_bm25(doc['content'])
            sv     = self.vectorize_doc(tokens, stats)
            doc_rows.append((doc['id'], doc['content'], len(tokens), sv))

            self._pending_n_delta     += 1
            self._pending_token_delta += len(tokens)
            for bucket in self._build_bucket_tf(tokens):
                self._pending_bucket_delta[bucket] += 1

        with self.conn.cursor() as cur:
            psycopg2.extras.execute_values(cur, """
                INSERT INTO bm25_incoming (doc_id, content, token_count, bm25_sparse)
                VALUES %s
                ON CONFLICT (doc_id) DO UPDATE SET
                    content     = EXCLUDED.content,
                    token_count = EXCLUDED.token_count,
                    bm25_sparse = EXCLUDED.bm25_sparse
            """, doc_rows)
        self.conn.commit()

    # -----------------------------------------------------------------------
    # Stage delete
    # -----------------------------------------------------------------------

    def stage_delete(self, doc_ids: List[str]):
        """Marks docs for removal at next commit; derives bucket df decrements."""
        if not doc_ids:
            return
        with self.conn.cursor() as cur:
            cur.execute(
                f"SELECT id, token_count, bm25_sparse FROM {self.table_name} WHERE id = ANY(%s)",
                (doc_ids,)
            )
            rows = cur.fetchall()
        for _, token_count, sv in rows:
            self._pending_n_delta     -= 1
            self._pending_token_delta -= token_count
            for bucket in sv.indices():
                self._pending_bucket_delta[bucket] -= 1
        self._pending_delete_ids.extend(doc_ids)

    # -----------------------------------------------------------------------
    # Commit
    # -----------------------------------------------------------------------

    def commit_batch(self):
        """
        Atomic collapse of all pending state into global tables.

        Order:
          1. Promote bm25_incoming → documents
          2. Apply n / token deltas → bm25_global_stats
          3. Apply bucket df deltas → bm25_term_global (upsert)
          4. Execute pending deletes
          5. Truncate bm25_incoming
          6. Reset Python pending state
        """
        with self.conn.cursor() as cur:
            cur.execute(f"""
                INSERT INTO {self.table_name} (id, content, bm25_sparse)
                SELECT doc_id, content, bm25_sparse FROM bm25_incoming
                ON CONFLICT (id) DO UPDATE SET
                    content     = EXCLUDED.content,
                    bm25_sparse = EXCLUDED.bm25_sparse
            """)
            cur.execute("""
                UPDATE bm25_global_stats SET
                    n_docs       = n_docs       + %s,
                    total_tokens = total_tokens + %s,
                    updated_at   = now()
                WHERE id = 1
            """, (self._pending_n_delta, self._pending_token_delta))
            if self._pending_bucket_delta:
                psycopg2.extras.execute_values(cur, """
                    INSERT INTO bm25_term_global (bucket, doc_freq)
                    VALUES %s
                    ON CONFLICT (bucket) DO UPDATE SET
                        doc_freq = GREATEST(bm25_term_global.doc_freq + EXCLUDED.doc_freq, 0)
                """, list(self._pending_bucket_delta.items()))
            if self._pending_delete_ids:
                cur.execute(
                    f"DELETE FROM {self.table_name} WHERE id = ANY(%s)",
                    (self._pending_delete_ids,)
                )
            cur.execute("TRUNCATE bm25_incoming")
        self.conn.commit()
        self._reset_pending()

    # -----------------------------------------------------------------------
    # Internal
    # -----------------------------------------------------------------------

    def _reset_pending(self):
        self._pending_bucket_delta: Counter = Counter()
        self._pending_n_delta:      int     = 0
        self._pending_token_delta:  int     = 0
        self._pending_delete_ids:   list    = []


# =============================================================================
# Commit Scheduler
# =============================================================================

class CommitScheduler:
    """
    Flushes staging → global when EITHER threshold is crossed.

    size_threshold:  flush when bm25_incoming reaches N docs
    max_age_seconds: flush at least every N seconds regardless of size

    Recommended for large corpora: size_threshold=5000, max_age_seconds=300
    """

    def __init__(self, manager: BM25HashManager,
                 size_threshold: int = 5_000, max_age_seconds: int = 300):
        self.manager        = manager
        self.size_threshold = size_threshold
        self.max_age        = max_age_seconds
        self._last_commit   = time.monotonic()
        self._lock          = threading.Lock()

    def maybe_commit(self):
        with self._lock:
            age    = time.monotonic() - self._last_commit
            staged = self._staged_count()
            if staged >= self.size_threshold or age >= self.max_age:
                self.manager.commit_batch()
                self._last_commit = time.monotonic()

    def force_commit(self):
        """Call at pipeline shutdown to flush remaining staged docs."""
        with self._lock:
            self.manager.commit_batch()
            self._last_commit = time.monotonic()

    def _staged_count(self) -> int:
        with self.manager.conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM bm25_incoming")
            return cur.fetchone()[0]








# =============================================================================
# Dense Embedder
# =============================================================================

class Model2VecEmbedder:
    """
    Dense embeddings using model2vec with optional PCA reduction,
    or full sentence embeddings via sentence-transformers.
    """
    
    def __init__(self, model_name: str = "minishlab/M2V_base_output", target_dim: int = 64, use_full_embed: bool = False):
        self.target_dim = target_dim
        self.use_full_embed = use_full_embed
        self.pca = None
        self._pca_fitted = False
        self.model = self._load_model(model_name)
        
        # Check native dimension
        test = self.encode(["test"])
        self.native_dim = test.shape[1]
    
    def _load_model(self, model_name: str):
        """Load model based on embedding mode."""
        if self.use_full_embed:
            from sentence_transformers import SentenceTransformer
            print(f"Loading full sentence embeddings: {model_name}")
            return SentenceTransformer(model_name)
        else:
            from model2vec import StaticModel
            print(f"Loading model2vec: {model_name}")
            return StaticModel.from_pretrained(model_name)
    
    def fit_pca(self, texts: List[str], sample_size: int = 10000):
        """Fit PCA for dimension reduction (call once during index building)."""
        if self.native_dim == self.target_dim:
            return
        
        from sklearn.decomposition import PCA
        
        n = min(len(texts), sample_size)
        if n < self.target_dim:
            self.pca = None
            return
        
        print(f"  Fitting PCA: {self.native_dim} → {self.target_dim}...")
        embeddings = self.encode(texts[:n])
        
        self.pca = PCA(n_components=self.target_dim)
        self.pca.fit(embeddings)
        self._pca_fitted = True
        
        explained = sum(self.pca.explained_variance_ratio_) * 100
        print(f"  Explained variance: {explained:.1f}%")
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to normalized embeddings."""
        if self.use_full_embed:
            # sentence-transformers returns numpy arrays
            embeddings = self.model.encode(texts, convert_to_numpy=True)
        else:
            # model2vec returns numpy arrays
            embeddings = self.model.encode(texts)
        
        # Apply PCA or truncation if native_dim is set
        if hasattr(self, 'native_dim'):
            if self.pca is not None:
                embeddings = self.pca.transform(embeddings)
            elif self.native_dim != self.target_dim:
                embeddings = embeddings[:, :self.target_dim]
        
        # L2 normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.clip(norms, 1e-10, None)
        
        return embeddings.astype(np.float32)


# =============================================================================
# PGVector GIST Retriever
# =============================================================================

class PGVectorRetriever(GISTRetriever):
    """
    GIST Retriever backed by PostgreSQL/pgvector.
    
    Implements all abstract methods from GISTRetriever using:
      - pgvector for dense similarity search
      - sparsevec for BM25 similarity search
      - model2vec for query embedding
      - Regex tokenizer + BM25 IDF for query sparse vectors
    """
    
    def __init__(self, config: Optional[PGVectorConfig] = None):
        self.pg_config = config or PGVectorConfig()
        super().__init__(self.pg_config)
        
        self.conn = None
        self._embedder = None
        self._bm25_manager = None
        self._scheduler    = None
        self._l1_vocab_cache = None
        self._l1_pca_cache   = None
    
    # =========================================================================
    # Connection Management
    # =========================================================================
    
    def connect(self):
        """Open database connection."""
        if self.conn is None or self.conn.closed:
            self.conn = psycopg2.connect(
                host=self.pg_config.db_host,
                port=self.pg_config.db_port,
                dbname=self.pg_config.db_name,
                user=self.pg_config.db_user,
                password=self.pg_config.db_password
            )
            self.conn.autocommit = True
            register_vector(self.conn)
    
    def close(self):
        """Close database connection."""
        if self.conn and not self.conn.closed:
            self.conn.close()
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, *args):
        self.close()
    
    # =========================================================================
    # Lazy-Loaded Components
    # =========================================================================
    
    @property
    def embedder(self) -> Model2VecEmbedder:
        """Lazy-load embedder, resolving relative model paths against project root."""
        if self._embedder is None:
            model_name = self.pg_config.embedding_model
            # Resolve relative paths against project root (parent of retrieval/)
            model_path = Path(model_name)
            if not model_path.is_absolute() and not model_path.exists():
                resolved = (Path(__file__).parent.parent / model_name).resolve()
                if resolved.exists():
                    model_name = str(resolved)
            self._embedder = Model2VecEmbedder(
                model_name=model_name,
                target_dim=self.pg_config.embedding_dim,
                use_full_embed=self.pg_config.use_full_embed
            )
        return self._embedder
    
    @property
    def bm25_manager(self) -> BM25HashManager:
        """Lazy-load BM25 hash manager."""
        if self._bm25_manager is None:
            self.connect()
            self._bm25_manager = BM25HashManager(
                conn=self.conn,
                k1=self.pg_config.bm25_k1,
                b=self.pg_config.bm25_b,
                n_buckets=self.pg_config.n_buckets,
                table_name=self.pg_config.table_name
            )
        return self._bm25_manager
    
    @property
    def scheduler(self) -> CommitScheduler:
        """Lazy-load commit scheduler."""
        if self._scheduler is None:
            self._scheduler = CommitScheduler(
                self.bm25_manager,
                size_threshold=self.pg_config.commit_size_threshold,
                max_age_seconds=self.pg_config.commit_max_age_seconds
            )
        return self._scheduler
    
    # =========================================================================
    # Abstract Method Implementations
    # =========================================================================
    
    # -------------------------------------------------------------------------
    # BM25 query helpers (sparsevec / signed-hash path)
    # -------------------------------------------------------------------------

    def _q_bm25_stats(self,
                      stat_table: str = 'bm25_global_stats') -> Dict:
        """Read corpus stats from *stat_table*."""
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    f"SELECT n_docs, total_tokens FROM {stat_table} WHERE id = 1"
                )
                row = cur.fetchone()
            if row:
                n, tok = row
                return {'n': max(n, 1), 'avgdl': max(tok, 1) / max(n, 1)}
        except Exception:
            self.conn.rollback()
        return {'n': 1, 'avgdl': 256}

    def _q_bm25_df(self, buckets: List[int],
                   term_table: str = 'bm25_term_global') -> Dict[int, int]:
        """Fetch doc_freq for *buckets* in one round-trip."""
        if not buckets:
            return {}
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    f"SELECT bucket, doc_freq FROM {term_table}"
                    f" WHERE bucket = ANY(%s)",
                    (buckets,)
                )
                return {row[0]: row[1] for row in cur.fetchall()}
        except Exception:
            self.conn.rollback()
            return {}

    def _q_bm25_vectorize(self, text: str,
                          stat_table: str = 'bm25_global_stats',
                          term_table:  str = 'bm25_term_global') -> SparseVector:
        """IDF-only query vector using the signed-hash BM25 scheme."""
        n_buckets = self.pg_config.n_buckets
        tokens    = tokenize_bm25(text)        # module-level helper
        stats     = self._q_bm25_stats(stat_table)
        n         = stats['n']

        bucket_tf: Dict[int, float] = defaultdict(float)
        for term in tokens:
            bucket, sign = hash_token(term)    # module-level helper
            bucket_tf[bucket] += sign

        df_map          = self._q_bm25_df(list(bucket_tf.keys()), term_table)
        indices, values = [], []
        for bucket, signed_tf in bucket_tf.items():
            df  = df_map.get(bucket, 0)
            idf = math.log((n - df + 0.5) / (df + 0.5) + 1)
            if idf > 0:
                indices.append(bucket)
                values.append(math.copysign(idf, signed_tf))

        return SparseVector(dict(zip(indices, values)), n_buckets)

    # -------------------------------------------------------------------------
    # L1 vocab helpers (lemmatized BERT BM25 — layer1_bm25_sparse JSONB)
    # -------------------------------------------------------------------------

    @property
    def _l1_vocab(self) -> Dict[str, int]:
        """Lazy-load L1 BM25 vocabulary (token_str → CSR column index)."""
        if self._l1_vocab_cache is None:
            vocab_path = Path(__file__).parent.parent / 'checkpoints' / 'chunk_bm25_sparse.msgpack'
            with open(vocab_path, 'rb') as f:
                data = msgpack.unpackb(f.read(), strict_map_key=False)
            self._l1_vocab_cache = data['vocab']   # {str: int}
        return self._l1_vocab_cache

    def _l1_bm25_term_ids(self, query: str) -> List[str]:
        """
        Tokenize *query* (lowercase split) and return unique JSONB key strings
        for terms present in the L1 BM25 vocab.
        """
        vocab  = self._l1_vocab
        seen:  Set[int] = set()
        ids:   List[str] = []
        for token in query.lower().split():
            tid = vocab.get(token)
            if tid is not None and tid not in seen:
                ids.append(str(tid))
                seen.add(tid)
        return ids

    # -------------------------------------------------------------------------
    # L1 PCA helper (Qwen3 256d → 128d)
    # -------------------------------------------------------------------------

    @property
    def _l1_pca(self):
        """Lazy-load PCA transform (256d → 128d) from numpy npz checkpoint.

        Returns a lightweight object with a .transform(X) method so call sites
        are unchanged.  Loading from .npz avoids numpy-version pickle issues.
        """
        if self._l1_pca_cache is None:
            npz_path = Path(__file__).parent.parent / 'checkpoints' / 'pca_256to128.npz'
            npz = np.load(str(npz_path))
            components = npz['components'].astype(np.float32)  # (128, 256)
            mean      = npz['mean'].astype(np.float32)         # (256,)

            class _NpzPCA:
                """Minimal PCA wrapper: transform(X) → (X - mean) @ components.T"""
                def __init__(self, C, mu):
                    self.C  = C   # (128, 256)
                    self.mu = mu  # (256,)
                def transform(self, X: np.ndarray) -> np.ndarray:
                    # X: (n_samples, 256) or (1, 256)
                    return (X.astype(np.float32) - self.mu) @ self.C.T  # → (n, 128)

            self._l1_pca_cache = _NpzPCA(components, mean)
        return self._l1_pca_cache

    # -------------------------------------------------------------------------
    # L1 retrieval methods
    # -------------------------------------------------------------------------

    def _retrieve_bm25(self, query: str, limit: int) -> List[RetrievedDoc]:
        """
        Layer 1 BM25: lemmatized JSONB dot product over layer1_bm25_sparse.

        Tokenizes *query* with the same lowercase-split tokenizer used at
        ingest time, maps tokens to CSR column indices via the saved vocab,
        then sums the stored BM25 weights for matching JSONB keys using the
        GIN index on layer1_bm25_sparse.sparse_vector.
        """
        self.connect()

        term_ids = self._l1_bm25_term_ids(query)
        if not term_ids:
            return []

        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT scored.chunk_id,
                       c.content, c.paper_id, c.section_idx, c.chunk_idx,
                       scored.score
                FROM (
                    SELECT l.chunk_id,
                           SUM((l.sparse_vector->>k)::float) AS score
                    FROM layer1_bm25_sparse l,
                         unnest(%s::text[]) AS k
                    WHERE l.sparse_vector ? k
                    GROUP BY l.chunk_id
                ) scored
                JOIN arxiv_chunks c ON c.chunk_id = scored.chunk_id
                ORDER BY scored.score DESC
                LIMIT %s
            """, (term_ids, limit))

            results = []
            for rank, row in enumerate(cur.fetchall(), 1):
                doc = RetrievedDoc(
                    doc_id=row[0],
                    content=row[1],
                    metadata={
                        'paper_id': row[2],
                        'section_idx': row[3],
                        'chunk_idx': row[4],
                    },
                    bm25_score=float(row[5]) if row[5] is not None else 0.0,
                    bm25_rank=rank,
                )
                results.append(doc)

        return results

    def _retrieve_dense(self, query: str, limit: int) -> List[RetrievedDoc]:
        """
        Layer 1 Dense: PCA-reduced Qwen3 128d HNSW search (layer1_embeddings_128d).

        Encodes *query* with the Qwen3 256d embedder, applies the saved PCA
        (256d → 128d), L2-normalises, then queries the HNSW index on
        layer1_embeddings_128d using cosine distance.
        """
        self.connect()

        # Qwen3 256d → PCA 128d → L2-normalised
        raw_emb   = self.embedder.encode([query])[0]                          # (256,)
        query_emb = self._l1_pca.transform(raw_emb.reshape(1, -1))[0]        # (128,)
        norm      = np.linalg.norm(query_emb)
        if norm > 1e-10:
            query_emb = query_emb / norm

        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT l.chunk_id,
                       c.content, c.paper_id, c.section_idx, c.chunk_idx,
                       1.0 - (l.embedding <=> %s::vector) AS score
                FROM layer1_embeddings_128d l
                JOIN arxiv_chunks c ON c.chunk_id = l.chunk_id
                ORDER BY l.embedding <=> %s::vector
                LIMIT %s
            """, (query_emb.tolist(), query_emb.tolist(), limit))

            results = []
            for rank, row in enumerate(cur.fetchall(), 1):
                doc = RetrievedDoc(
                    doc_id=row[0],
                    content=row[1],
                    metadata={
                        'paper_id': row[2],
                        'section_idx': row[3],
                        'chunk_idx': row[4],
                    },
                    dense_score=float(row[5]) if row[5] is not None else 0.0,
                    dense_rank=rank,
                )
                results.append(doc)

        return results
    
    # -------------------------------------------------------------------------
    # Layer 2 Expansion (ECDF-weighted BM25 + Dense centroid)
    # -------------------------------------------------------------------------

    def _q_bm25_vectorize_weighted(
        self,
        term_weights: Dict[str, float],
        stat_table: str = 'bm25_global_stats',
        term_table: str  = 'bm25_term_global',
    ) -> 'SparseVector':
        """Weighted IDF vectorization for L2 expansion (ECDF-weighted TF profile)."""
        n_buckets = self.pg_config.n_buckets
        stats = self._q_bm25_stats(stat_table)
        n = stats['n']
        bucket_acc: Dict[int, float] = defaultdict(float)
        for term, weight in term_weights.items():
            bucket, sign = hash_token(term)
            bucket_acc[bucket] += sign * abs(weight)
        df_map = self._q_bm25_df(list(bucket_acc.keys()), term_table)
        indices, values = [], []
        for bucket, signed_w in bucket_acc.items():
            df    = df_map.get(bucket, 0)
            idf   = math.log((n - df + 0.5) / (df + 0.5) + 1)
            score = idf * abs(signed_w)
            if score > 0:
                indices.append(bucket)
                values.append(math.copysign(score, signed_w))
        return SparseVector(dict(zip(indices, values)), n_buckets)

    def get_chunk_texts(self, chunk_ids: List[str]) -> List[str]:
        """Fetch content for chunk_ids from arxiv_chunks (order-preserving)."""
        if not chunk_ids:
            return []
        self.connect()
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT chunk_id, content FROM arxiv_chunks WHERE chunk_id = ANY(%s)",
                (chunk_ids,)
            )
            text_map = {row[0]: row[1] for row in cur.fetchall()}
        return [text_map.get(cid, '') for cid in chunk_ids]

    def _expand_layer2_bm25(
        self,
        seed_docs: List[RetrievedDoc],
        seed_scores: List[float],
        top_k: int,
    ) -> List[RetrievedDoc]:
        """
        Layer 2 Path A: ECDF-weighted BM25 expansion over layer2_triplet_bm25.

        Builds a weighted TF profile from L1 seed texts, vectorizes it with IDF,
        and queries the triplet BM25 sparse-vector table.  Seeds are excluded.
        """
        from collections import Counter
        import numpy as _np

        if not seed_docs or not seed_scores:
            return []

        self.connect()
        n_buckets = self.pg_config.n_buckets

        # 1. ECDF weights from L1 RRF scores
        ecdf_w = self._midpoint_ecdf_weights(
            _np.array(seed_scores, dtype=_np.float64)
        )

        # 2. Fetch seed texts
        seed_ids   = [doc.doc_id for doc in seed_docs]
        seed_texts = self.get_chunk_texts(seed_ids)

        # 3. Build weighted TF profile
        seed_tfs: List[Counter] = []
        valid_w:  List[float]   = []
        for idx, text in enumerate(seed_texts):
            if text.strip():
                tf = Counter(text.lower().split())
                if tf:
                    seed_tfs.append(tf)
                    valid_w.append(float(ecdf_w[idx]))

        if not seed_tfs:
            return []

        vocab     = sorted(set().union(*seed_tfs))
        vocab_idx = {t: i for i, t in enumerate(vocab)}
        tf_matrix = _np.zeros((len(seed_tfs), len(vocab)), dtype=_np.float64)
        for i, tf in enumerate(seed_tfs):
            for term, cnt in tf.items():
                tf_matrix[i, vocab_idx[term]] = cnt
        weighted_tf  = _np.average(tf_matrix, axis=0, weights=_np.array(valid_w))
        term_weights = {
            vocab[j]: float(weighted_tf[j])
            for j in range(len(vocab)) if weighted_tf[j] > 0
        }
        if not term_weights:
            return []

        # 4. Vectorize and query layer2_triplet_bm25
        q_sv = self._q_bm25_vectorize_weighted(
            term_weights,
            stat_table='bm25_l2_stats',
            term_table='bm25_l2_term_df',
        )
        if not q_sv.indices():
            return []

        seed_set = set(seed_ids)
        with self.conn.cursor() as cur:
            cur.execute(f"""
                SELECT chunk_id,
                       -(triplet_bm25_vector <#> %s::sparsevec({n_buckets})) AS score
                FROM layer2_triplet_bm25
                ORDER BY triplet_bm25_vector <#> %s::sparsevec({n_buckets}) ASC
                LIMIT %s
            """, (q_sv, q_sv, top_k * 2 + len(seed_set)))
            rows = cur.fetchall()

        # 5. Exclude seeds, collect all candidates
        candidates = [
            RetrievedDoc(
                doc_id=row[0], content='',
                metadata={'source': 'layer2_triplet_bm25'},
                bm25_score=float(row[1]),
            )
            for row in rows if row[0] not in seed_set
        ]

        if not candidates:
            return []

        # 6. GIST selection: fetch embeddings via L2→L1 chunk_id mapping
        # L2 triplet ids have doubled suffix: 'paper_sX_cY_sX_cY' → strip last '_sN_cM'
        import re as _re
        cand_ids = [doc.doc_id for doc in candidates]
        l2_to_l1 = {cid: _re.sub(r'_s\d+_c\d+$', '', cid) for cid in cand_ids}
        l1_ids = list(set(l2_to_l1.values()))
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT chunk_id, embedding FROM layer2_embeddings_256d WHERE chunk_id = ANY(%s)",
                (l1_ids,)
            )
            l1_emb = {row[0]: _np.array(row[1], dtype=_np.float64) for row in cur.fetchall()}
        # Map L2 candidate ids → embeddings via L1 key
        emb_map = {
            cid: l1_emb[l2_to_l1[cid]]
            for cid in cand_ids
            if l2_to_l1[cid] in l1_emb
        }

        valid_idx = [i for i, doc in enumerate(candidates) if doc.doc_id in emb_map]
        if len(valid_idx) < 2:
            # Not enough embeddings — fall back to score-sorted
            results = sorted(candidates, key=lambda d: d.bm25_score, reverse=True)[:top_k]
            for rank, doc in enumerate(results, 1):
                doc.bm25_rank = rank
            return results

        emb_matrix = _np.array([emb_map[candidates[i].doc_id] for i in valid_idx])
        scores_arr = _np.array([candidates[i].bm25_score for i in valid_idx])

        # Pairwise cosine similarity as coverage matrix
        norms = _np.linalg.norm(emb_matrix, axis=1, keepdims=True)
        norms = _np.where(norms == 0, 1.0, norms)
        emb_norm = emb_matrix / norms
        coverage_matrix = emb_norm @ emb_norm.T

        # MMR-style selection: relevance + diversity
        selected = gist_select(coverage_matrix, scores_arr, top_k)
        results = [candidates[valid_idx[i]] for i in selected]
        for rank, doc in enumerate(results, 1):
            doc.bm25_rank = rank
        return results

    def _expand_layer2_dense(
        self,
        seed_docs: List[RetrievedDoc],
        seed_scores: List[float],
        top_k: int,
    ) -> List[RetrievedDoc]:
        """
        Layer 2 Path B: ECDF-weighted GIST centroid expansion over layer2_embeddings_256d.

        Takes L1 hybrid seeds, computes a weighted centroid in the 256d Qwen3 model2vec
        space, ANNs against layer2_embeddings_256d (HNSW cosine), then applies GIST
        diversity selection over the retrieved candidates.  Seeds are excluded.

        GIST clustering is done only over the retrieved results — never the full table.
        """
        import numpy as _np

        if not seed_docs or not seed_scores:
            return []

        self.connect()

        # 1. ECDF weights from L1 RRF scores
        ecdf_w   = self._midpoint_ecdf_weights(
            _np.array(seed_scores, dtype=_np.float64)
        )
        seed_ids = [doc.doc_id for doc in seed_docs]

        # 2. Fetch seed embeddings from layer2_embeddings_256d (256d Qwen3 model2vec)
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT chunk_id, embedding FROM layer2_embeddings_256d WHERE chunk_id = ANY(%s)",
                (seed_ids,)
            )
            emb_map = {row[0]: _np.array(row[1], dtype=_np.float64) for row in cur.fetchall()}

        present = [cid for cid in seed_ids if cid in emb_map]
        if not present:
            return []
        vecs    = _np.array([emb_map[cid] for cid in present])
        valid_w = _np.array([ecdf_w[seed_ids.index(cid)] for cid in present])

        # 3. Weighted centroid → cosine ANN in layer2_embeddings_256d
        centroid = _np.average(vecs, axis=0, weights=valid_w)
        emb_str  = '[' + ','.join(map(str, centroid.tolist())) + ']'

        seed_set = set(seed_ids)
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT chunk_id, 1 - (embedding <=> %s::vector(256)) AS score, embedding
                FROM layer2_embeddings_256d
                ORDER BY embedding <=> %s::vector(256)
                LIMIT %s
            """, (emb_str, emb_str, top_k * 2 + len(seed_set)))
            rows = cur.fetchall()

        # 4. Exclude seeds, collect candidates with embeddings for GIST
        candidates = [
            (row[0], float(row[1]), _np.array(row[2], dtype=_np.float64))
            for row in rows if row[0] not in seed_set
        ]

        if not candidates:
            return []

        cand_docs = [
            RetrievedDoc(doc_id=c[0], content='', metadata={'source': 'layer2_dense'}, dense_score=c[1])
            for c in candidates
        ]

        # 5. GIST selection: cluster only the retrieved results, not the full table
        emb_matrix = _np.array([c[2] for c in candidates])
        scores_arr = _np.array([c[1] for c in candidates])

        norms = _np.linalg.norm(emb_matrix, axis=1, keepdims=True)
        norms = _np.where(norms == 0, 1.0, norms)
        emb_norm = emb_matrix / norms
        coverage_matrix = emb_norm @ emb_norm.T

        if len(cand_docs) < 2:
            results = cand_docs[:top_k]
        else:
            selected = gist_select(coverage_matrix, scores_arr, top_k)
            results = [cand_docs[i] for i in selected]

        for rank, doc in enumerate(results, 1):
            doc.dense_rank = rank
        return results

    def _rrf_fusion(
        self,
        bm25_results: List[RetrievedDoc],
        dense_results: List[RetrievedDoc]
    ) -> List[RetrievedDoc]:
        """
        Fuse BM25 and dense results using Reciprocal Rank Fusion.
        
        Args:
            bm25_results: BM25 retrieval results (with bm25_score and bm25_rank)
            dense_results: Dense retrieval results (with dense_score and dense_rank)
        
        Returns:
            Fused results sorted by RRF score
        """
        # Build lookups
        bm25_by_id = {doc.doc_id: doc for doc in bm25_results}
        dense_by_id = {doc.doc_id: doc for doc in dense_results}
        
        # Merge both pools
        all_ids = set(bm25_by_id.keys()) | set(dense_by_id.keys())
        
        fused = []
        for doc_id in all_ids:
            bm25_doc = bm25_by_id.get(doc_id)
            dense_doc = dense_by_id.get(doc_id)
            
            # Use whichever doc we have (prefer BM25 if both)
            doc = bm25_doc or dense_doc
            
            # Preserve component scores and ranks
            if bm25_doc:
                doc.bm25_score = bm25_doc.bm25_score
                doc.bm25_rank = bm25_doc.bm25_rank
            if dense_doc:
                doc.dense_score = dense_doc.dense_score
                doc.dense_rank = dense_doc.dense_rank
            
            # Compute RRF score from ranks (NOT gist_rank!)
            ranks = [
                bm25_doc.bm25_rank if bm25_doc else None,
                dense_doc.dense_rank if dense_doc else None
            ]
            doc.rrf_score = compute_rrf_score(ranks, k=self.config.rrf_k)
            
            fused.append(doc)
        
        # Sort by RRF score (descending)
        fused.sort(key=lambda d: d.rrf_score, reverse=True)
        return fused
    
    def _get_bm25_doc_doc_scores(self, doc_ids: List[str]) -> np.ndarray:
        """Compute pairwise BM25 similarity via sparsevec <#> self-join."""
        self.connect()
        n         = len(doc_ids)
        id_to_idx = {did: i for i, did in enumerate(doc_ids)}
        sim_matrix = np.zeros((n, n))

        with self.conn.cursor() as cur:
            cur.execute(f"""
                SELECT a.chunk_id, b.chunk_id,
                       -(a.bm25_sparse <#> b.bm25_sparse) AS score
                FROM {self.pg_config.table_name} a,
                     {self.pg_config.table_name} b
                WHERE a.chunk_id = ANY(%s)
                  AND b.chunk_id = ANY(%s)
            """, (doc_ids, doc_ids))

            for row in cur.fetchall():
                i = id_to_idx.get(row[0])
                j = id_to_idx.get(row[1])
                if i is not None and j is not None:
                    sim_matrix[i, j] = float(row[2]) if row[2] is not None else 0.0

        max_val = sim_matrix.max()
        if max_val > 0:
            sim_matrix = sim_matrix / max_val
        np.fill_diagonal(sim_matrix, 1.0)
        return sim_matrix

    def _get_dense_embeddings(self, doc_ids: List[str]) -> np.ndarray:
        """Fetch dense embeddings for given doc IDs."""
        self.connect()
        
        with self.conn.cursor() as cur:
            cur.execute(f"""
                SELECT chunk_id, embedding::text
                FROM {self.pg_config.table_name}
                WHERE chunk_id = ANY(%s)
            """, (doc_ids,))
            
            emb_dict = {}
            for row in cur.fetchall():
                # Parse vector string: "[0.1,0.2,0.3]" -> np.array
                vec_str = row[1].strip('[]')
                emb_dict[row[0]] = np.array([float(x) for x in vec_str.split(',')])
        
        # Preserve order
        embeddings = np.array([emb_dict[doc_id] for doc_id in doc_ids])
        return embeddings
    
    def _get_query_embedding(self, query: str) -> np.ndarray:
        """Encode query to dense embedding."""
        return self.embedder.encode([query])[0]
    
    def _get_bm25_query_scores(self, doc_ids: List[str], query: str) -> np.ndarray:
        """Get BM25 scores via sparsevec <#> inner product."""
        self.connect()
        mgr    = self.bm25_manager
        stats  = mgr.effective_stats()
        tokens = tokenize_bm25(query)
        q_sv   = mgr.vectorize_query(tokens, stats)

        with self.conn.cursor() as cur:
            cur.execute(f"""
                SELECT chunk_id,
                       -(bm25_sparse <#> %s) AS score
                FROM {self.pg_config.table_name}
                WHERE chunk_id = ANY(%s)
            """, (q_sv, doc_ids))

            score_dict = {row[0]: float(row[1]) for row in cur.fetchall()}

        return np.array([score_dict.get(did, 0.0) for did in doc_ids])

    # =========================================================================
    # BM25 schema & incremental-update delegation
    # =========================================================================

    def init_bm25_schema(self):
        """Create BM25 rolling-stats tables (idempotent)."""
        self.connect()
        with self.conn.cursor() as cur:
            cur.execute(BM25_SCHEMA_SQL)

    def stage_insert(self, docs: List[Dict]):
        """
        Stage documents for incremental BM25 update.

        Each doc must have keys: doc_id (str), content (str).
        Triggers scheduler; commit happens automatically when threshold is met.
        """
        self.bm25_manager.stage_insert(docs)
        self.scheduler.maybe_commit()

    def stage_delete(self, doc_ids: List[str]):
        """Stage document deletions; triggers auto-commit if threshold met."""
        self.bm25_manager.stage_delete(doc_ids)
        self.scheduler.maybe_commit()

    def commit_batch(self):
        """Force-flush all staged BM25 inserts/deletes."""
        self.scheduler.force_commit()

    def _get_group_key(self, doc: RetrievedDoc) -> Tuple:
        """
        Extract grouping key: (paper_id, section_idx) for arxiv.
        
        Groups chunks by section to maintain section-level granularity.
        Each section becomes a separate retrievable unit.
        """
        return (
            doc.metadata.get('paper_id'),
            doc.metadata.get('section_idx')
        )
    
    def _fetch_all_chunks_for_group(self, group_key: Tuple) -> List[RetrievedDoc]:
        """
        Fetch ALL chunks from a specific section.
        
        Groups by (paper_id, section_idx) to reconstruct complete sections.
        Each section is treated as an independent retrievable unit.
        """
        self.connect()
        paper_id, section_idx = group_key
        
        with self.conn.cursor() as cur:
            cur.execute(f"""
                SELECT chunk_id, content, paper_id, section_idx, chunk_idx
                FROM {self.pg_config.table_name}
                WHERE paper_id = %s AND section_idx = %s
                ORDER BY chunk_idx
            """, (paper_id, section_idx))
            
            chunks = []
            for row in cur.fetchall():
                doc = RetrievedDoc(
                    doc_id=row[0],
                    content=row[1],
                    metadata={
                        'paper_id': row[2],
                        'section_idx': row[3],
                        'chunk_idx': row[4]
                    }
                )
                chunks.append(doc)
        
        return chunks
    
    def _group_key_to_id(self, group_key: Tuple) -> str:
        """
        Convert (paper_id, section_idx) to string identifier.
        
        Format: "paper_id:s{section_idx}" (e.g., "1301_3781:s0")
        """
        paper_id, section_idx = group_key
        return f"{paper_id}:s{section_idx}"
    
    # =========================================================================
    # Index Building
    # =========================================================================

    def build_index(
        self,
        chunks: List[Dict],
        reset: bool = False
    ):
        """
        Build hybrid index (dense + BM25 sparsevec) from chunks.

        Args:
            chunks: List of dicts with keys:
                    - doc_id:       Unique chunk ID
                    - paper_id:     Parent document ID
                    - section_idx:  Section index
                    - chunk_idx:    Chunk index within section
                    - text:         Chunk content
            reset: If True, drop existing table before build.
        """
        from tqdm import tqdm
        self.connect()

        print("=" * 70)
        print("Building PGVector Index")
        print("=" * 70)

        chunk_ids    = [c['doc_id']      for c in chunks]
        paper_ids    = [c['paper_id']    for c in chunks]
        section_idxs = [c['section_idx'] for c in chunks]
        chunk_idxs   = [c['chunk_idx']   for c in chunks]
        contents     = [c['text']        for c in chunks]
        N            = len(contents)

        # ── Step 1: BM25 sparsevec (signed hashing) ──────────────────────────
        print("\n[1/4] Building BM25 sparse vectors (hash trick)...")
        k1, b_param = self.pg_config.bm25_k1, self.pg_config.bm25_b
        n_buckets   = self.pg_config.n_buckets

        all_tokens   = [tokenize_bm25(t) for t in tqdm(contents, desc="  Tokenizing", leave=False)]
        total_tokens = sum(len(t) for t in all_tokens)
        avgdl        = total_tokens / N if N > 0 else 1.0

        bucket_df: Dict[int, int] = defaultdict(int)
        for tokens in all_tokens:
            seen: Set[int] = set()
            for tok in tokens:
                bkt, _ = hash_token(tok)
                if bkt not in seen:
                    bucket_df[bkt] += 1
                    seen.add(bkt)

        bm25_vecs: List[SparseVector] = []
        for tokens in tqdm(all_tokens, desc="  Vectorizing BM25", leave=False):
            dl = len(tokens)
            if dl == 0:
                bm25_vecs.append(SparseVector({}, n_buckets))
                continue
            signed_tf: Dict[int, float] = defaultdict(float)
            for tok in tokens:
                bkt, sign = hash_token(tok)
                signed_tf[bkt] += sign
            idxs, vals = [], []
            for bkt, stf in signed_tf.items():
                df   = bucket_df.get(bkt, 1)
                idf  = math.log(1 + (N - df + 0.5) / (df + 0.5))
                tf_a = abs(stf)
                tf_n = tf_a * (k1 + 1) / (tf_a + k1 * (1 - b_param + b_param * dl / avgdl))
                w    = (1.0 if stf >= 0 else -1.0) * idf * tf_n
                if w != 0.0:
                    idxs.append(bkt)
                    vals.append(w)
            bm25_vecs.append(SparseVector(dict(zip(idxs, vals)), n_buckets))

        # ── Step 2: Dense embeddings ─────────────────────────────────────────
        print("\n[2/4] Creating dense embeddings...")
        embedder = Model2VecEmbedder(
            model_name=self.pg_config.embedding_model,
            target_dim=self.pg_config.embedding_dim
        )
        embedder.fit_pca(contents)
        embeddings = []
        for i in tqdm(range(0, N, 256), desc="  Embedding"):
            embeddings.append(embedder.encode(contents[i:i + 256]))
        embeddings = np.vstack(embeddings)

        # ── Step 3: Create tables ─────────────────────────────────────────────
        print("\n[3/4] Creating PostgreSQL tables...")
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            if reset:
                cur.execute(f"DROP TABLE IF EXISTS {self.pg_config.table_name} CASCADE")
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.pg_config.table_name} (
                    id          SERIAL PRIMARY KEY,
                    chunk_id    TEXT UNIQUE NOT NULL,
                    paper_id    TEXT NOT NULL,
                    section_idx INTEGER NOT NULL,
                    chunk_idx   INTEGER NOT NULL,
                    content     TEXT NOT NULL,
                    embedding   vector({self.pg_config.embedding_dim}) NOT NULL,
                    bm25_sparse sparsevec({n_buckets}) NOT NULL,
                    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
        self.init_bm25_schema()

        # ── Step 4: Insert data ───────────────────────────────────────────────
        print("\n[4/4] Inserting data...")
        data = list(zip(
            chunk_ids, paper_ids, section_idxs, chunk_idxs, contents,
            [emb.tolist() for emb in embeddings],
            bm25_vecs
        ))
        with self.conn.cursor() as cur:
            for i in tqdm(range(0, len(data), 500), desc="  Inserting"):
                execute_values(
                    cur,
                    f"""
                    INSERT INTO {self.pg_config.table_name}
                    (chunk_id, paper_id, section_idx, chunk_idx, content, embedding, bm25_sparse)
                    VALUES %s
                    ON CONFLICT (chunk_id) DO UPDATE SET
                        content     = EXCLUDED.content,
                        embedding   = EXCLUDED.embedding,
                        bm25_sparse = EXCLUDED.bm25_sparse
                    """,
                    data[i:i + 500],
                    template="(%s, %s, %s, %s, %s, %s::vector, %s::sparsevec)"
                )

        # ── Update rolling BM25 stats ─────────────────────────────────────────
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO bm25_global_stats (n_docs, total_tokens)
                VALUES (%s, %s)
                ON CONFLICT (id) DO UPDATE
                    SET n_docs       = EXCLUDED.n_docs,
                        total_tokens = EXCLUDED.total_tokens,
                        updated_at   = now()
            """, (N, total_tokens))
            execute_values(cur, """
                INSERT INTO bm25_term_global (bucket, doc_freq)
                VALUES %s
                ON CONFLICT (bucket) DO UPDATE SET doc_freq = EXCLUDED.doc_freq
            """, list(bucket_df.items()))

        # ── Create indexes ────────────────────────────────────────────────────
        print("  Creating indexes...")
        with self.conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {self.pg_config.table_name}")
            lists = max(1, int(cur.fetchone()[0] ** 0.5))
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS {self.pg_config.table_name}_emb_idx
                ON {self.pg_config.table_name}
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = {lists})
            """)
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS {self.pg_config.table_name}_bm25_idx
                ON {self.pg_config.table_name}
                USING ivfflat (bm25_sparse sparsevec_ip_ops)
                WITH (lists = {lists})
            """)
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS {self.pg_config.table_name}_paper_idx
                ON {self.pg_config.table_name} (paper_id)
            """)

        print("\n" + "=" * 70)
        print(f"Index built: {N:,} chunks")
        print(f"  Dense:  vector({self.pg_config.embedding_dim}) with IVFFlat (cosine)")
        print(f"  Sparse: sparsevec({n_buckets}) BM25 hash with IVFFlat (ip_ops)")
        print("=" * 70)

    def clear_all(self):
        """Drop all tables."""
        self.connect()
        with self.conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {self.pg_config.table_name} CASCADE")
            print(f"Dropped table: {self.pg_config.table_name}")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="PGVector GIST Retriever")
    
    # Actions
    parser.add_argument("--search", type=str, help="Search query")
    parser.add_argument("--build", action="store_true", help="Build/rebuild index from chunks")
    parser.add_argument("--clear", action="store_true", help="Clear all tables")
    
    # Search options
    parser.add_argument("--top_k", type=int, default=21, help="Number of results")
    parser.add_argument("--save", type=str, help="Save results summary to markdown file")
    parser.add_argument("--export", type=str, help="Export full reconstructed text to file")
    parser.add_argument("--chunks", action="store_true", help="Return chunks instead of grouped sections")
    
    # Build options
    parser.add_argument("--reset", action="store_true", help="Drop existing table before build")
    parser.add_argument("--papers_dir", type=str, default=r"C:\Users\user\arxiv_id_lists\papers\post_processed", 
                        help="Directory with papers for --build")
    parser.add_argument("--fullembed", action="store_true", help="Use full sentence embeddings instead of model2vec")
    parser.add_argument("--init-schema", action="store_true", help="Initialize BM25 rolling-stats schema")
    parser.add_argument("--commit", action="store_true", help="Force-flush staged BM25 updates")
    
    # Database connection args
    parser.add_argument("--host", type=str, default="localhost", help="Database host")
    parser.add_argument("--port", type=int, default=5432, help="Database port")
    parser.add_argument("--db", type=str, default="langchain", help="Database name")
    parser.add_argument("--user", type=str, default="langchain", help="Database user")
    parser.add_argument("--password", type=str, default="langchain", help="Database password")
    parser.add_argument("--table", type=str, default="arxiv_chunks", help="Table name")
    
    args = parser.parse_args()
    
    config = PGVectorConfig(
        db_host=args.host,
        db_port=args.port,
        db_name=args.db,
        db_user=args.user,
        db_password=args.password,
        table_name=args.table,
        use_full_embed=args.fullembed
    )

    if args.clear:
        confirm = input("Delete all tables? Type 'yes': ")
        if confirm.lower() == "yes":
            with PGVectorRetriever(config) as retriever:
                retriever.clear_all()

    elif args.init_schema:
        with PGVectorRetriever(config) as retriever:
            retriever.init_bm25_schema()
            print("BM25 rolling-stats schema initialized.")

    elif args.commit:
        with PGVectorRetriever(config) as retriever:
            retriever.commit_batch()
            print("BM25 staged updates committed.")

    elif args.build:
        # Import chunking pipeline
        try:
            from arxiv_chunking_pipeline import chunk_arxiv_papers
        except ImportError:
            print("ERROR: arxiv_chunking_pipeline.py not found in current directory")
            print("Place it alongside pgvector_retriever.py or adjust import path")
            exit(1)
        
        print(f"Chunking papers from: {args.papers_dir}")
        chunks = chunk_arxiv_papers(args.papers_dir)
        
        # Convert to dict format expected by build_index
        chunk_dicts = [
            {
                'doc_id': f"{c.doc_id}_s{c.section_idx}_c{c.chunk_idx}",
                'paper_id': c.doc_id,
                'section_idx': c.section_idx,
                'chunk_idx': c.chunk_idx,
                'text': c.text
            }
            for c in chunks
        ]
        
        # Save chunks to msgpack for diagnostics
        import msgpack
        import os
        os.makedirs("checkpoints", exist_ok=True)
        with open("checkpoints/chunks.msgpack", "wb") as f:
            msgpack.pack(chunk_dicts, f)
        print(f"Saved {len(chunk_dicts)} chunks to checkpoints/chunks.msgpack")
        
        with PGVectorRetriever(config) as retriever:
            retriever.build_index(chunk_dicts, reset=args.reset)
    
    elif args.search:
        with PGVectorRetriever(config) as retriever:
            if args.chunks:
                # Chunk-level search (no grouping)
                results = retriever.search_chunks(args.search, top_k=args.top_k)
                
                for i, doc in enumerate(results, 1):
                    print(f"\n{i}. {doc.doc_id}")
                    print(f"   Score: {doc.final_score:.4f}")
                    print(f"   Preview: {doc.content[:150]}...")
                
                if args.save:
                    md = format_results_markdown(results)
                    with open(args.save, 'w', encoding='utf-8') as f:
                        f.write(md)
                    print(f"\nSummary saved to {args.save}")
                
                if args.export:
                    with open(args.export, 'w', encoding='utf-8') as f:
                        f.write(f"# Search Results: {args.search}\n\n")
                        for i, doc in enumerate(results, 1):
                            f.write(f"## {i}. {doc.doc_id}\n\n")
                            f.write(f"**Score:** {doc.final_score:.4f}\n\n")
                            f.write(doc.content)
                            f.write("\n\n---\n\n")
                    print(f"Full content exported to {args.export}")
            
            else:
                # Paper-level search (returns papers with sections)
                results = retriever.search(args.search, top_k=args.top_k)
                
                for i, paper in enumerate(results, 1):
                    print(f"\n{i}. {paper.paper_id}")
                    print(f"   Score: {paper.final_score:.4f}")
                    print(f"   Sections: {paper.metadata.get('num_sections', '?')}")
                    preview = paper.full_text[:150] + "..." if len(paper.full_text) > 150 else paper.full_text
                    print(f"   Preview: {preview}")
                
                if args.save:
                    md = format_papers_markdown(results, include_sections=True)
                    with open(args.save, 'w', encoding='utf-8') as f:
                        f.write(md)
                    print(f"\nSummary saved to {args.save}")
                
                if args.export:
                    with open(args.export, 'w', encoding='utf-8') as f:
                        f.write(f"# Search Results: {args.search}\n\n")
                        f.write(f"**Query:** {args.search}\n")
                        f.write(f"**Results:** {len(results)} papers\n\n")
                        f.write("---\n\n")
                        
                        for i, paper in enumerate(results, 1):
                            f.write(f"## {i}. {paper.paper_id}\n\n")
                            f.write(f"**Final Score:** {paper.final_score:.4f}\n")
                            if paper.rrf_score is not None:
                                f.write(f"**RRF Score:** {paper.rrf_score:.4f}\n")
                            if paper.colbert_score is not None:
                                f.write(f"**ColBERT Score:** {paper.colbert_score:.4f}\n")
                            if paper.cross_encoder_score is not None:
                                f.write(f"**Cross-Encoder Score:** {paper.cross_encoder_score:.4f}\n")
                            f.write(f"**Sections:** {paper.metadata.get('num_sections', '?')}\n\n")
                            
                            # Write all sections
                            for j, section in enumerate(paper.sections, 1):
                                f.write(f"### Section {j}: {section.group_id}\n\n")
                                if section.rrf_score is not None:
                                    f.write(f"**Section RRF:** {section.rrf_score:.4f}\n")
                                if section.colbert_score is not None:
                                    f.write(f"**Section ColBERT:** {section.colbert_score:.4f}\n")
                                f.write("\n")
                                f.write(section.full_text)
                                f.write("\n\n")
                            
                            f.write("---\n\n")
                    
                    print(f"\nFull papers with sections exported to {args.export}")
    
    else:
        parser.print_help()