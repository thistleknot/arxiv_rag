"""
Quotes GIST Retriever: PostgreSQL/pgvector Backend for Quotes Dataset

=============================================================================
OVERVIEW
=============================================================================

This module implements the GISTRetriever interface for a quotes dataset.

Key differences from arxiv retriever:
  - Groups by: quote_id (single field)
  - Reconstructs: Full quote from chunks
  - Schema: quote_id, chunk_idx (no section_idx)

=============================================================================
SCHEMA
=============================================================================

CREATE TABLE {table_name} (
    id SERIAL PRIMARY KEY,
    chunk_id TEXT UNIQUE NOT NULL,
    quote_id TEXT NOT NULL,
    chunk_idx INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding vector({dim}) NOT NULL,
    bm25_sparse sparsevec({vocab_size}) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX ON {table_name} USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX ON {table_name} (quote_id);

=============================================================================
USAGE
=============================================================================

from quotes_retriever import QuotesRetriever, QuotesConfig

config = QuotesConfig(
    db_host="192.168.3.18",
    db_port=6024,
    table_name="quotes_chunks"
)

retriever = QuotesRetriever(config)
results = retriever.search("wisdom about life", top_k=21)

# Results are grouped by quote_id with full quote text reconstructed

=============================================================================
"""

import re
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set, Any
from collections import Counter, defaultdict
from math import exp

import psycopg2
from psycopg2.extras import execute_values

from gist_retriever import (
    GISTRetriever, GISTConfig, RetrievedDoc, RetrievedGroup,
    gist_select, compute_rrf_score, format_results_markdown, format_groups_markdown
)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class QuotesConfig(GISTConfig):
    """
    Configuration for Quotes GIST retriever.
    """
    # Database connection
    db_host: str = "192.168.3.18"
    db_port: int = 6024
    db_name: str = "langchain"
    db_user: str = "langchain"
    db_password: str = "langchain"
    
    # Table
    table_name: str = "quotes_chunks"
    
    # Embeddings
    embedding_dim: int = 64
    embedding_model: str = "minishlab/M2V_base_output"
    
    # BM25
    bm25_cache_path: Path = Path("quotes_bm25_vocab.msgpack")
    bm25_k1: float = 1.5
    bm25_b: float = 0.75


# =============================================================================
# Text Preprocessing (for BM25)
# =============================================================================

class TextPreprocessor:
    """
    BERT WordPiece tokenizer for BM25.
    
    Uses bert-base-uncased tokenizer to align BM25 vocabulary with
    embedding space. Produces subword tokens like '##ing', '##tion'.
    
    This ensures lexical matching operates on the same token boundaries
    as the dense encoder, improving hybrid retrieval coherence.
    """
    
    _tokenizer = None  # Class-level cache
    
    def __init__(self, vocab_dict: Dict[str, int] = None, tokenizer_name: str = "bert-base-uncased"):
        self.vocab = vocab_dict if vocab_dict else {}
        self.vocab_size = len(self.vocab) if vocab_dict else 0
        self.tokenizer_name = tokenizer_name
        
        # Lazy load tokenizer (shared across instances)
        if TextPreprocessor._tokenizer is None:
            from transformers import AutoTokenizer
            TextPreprocessor._tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    def preprocess(self, text: str) -> List[int]:
        """Tokenize text to vocab IDs using BERT WordPiece."""
        if not isinstance(text, str):
            text = str(text)
        
        tokens = TextPreprocessor._tokenizer.tokenize(text.lower())
        return [self.vocab.get(t, -1) for t in tokens if t in self.vocab]
    
    @staticmethod
    def build_vocab_from_corpus(
        texts: List[str], 
        min_df: int = 2,
        tokenizer_name: str = "bert-base-uncased"
    ) -> Dict[str, int]:
        """Build vocabulary from corpus using BERT tokenizer."""
        from transformers import AutoTokenizer
        from tqdm import tqdm
        
        print(f"  Building vocabulary from {len(texts):,} texts...")
        print(f"  Tokenizer: {tokenizer_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        word_counts = Counter()
        for text in tqdm(texts, desc="  Tokenizing", leave=False):
            tokens = tokenizer.tokenize(str(text).lower())
            word_counts.update(tokens)
        
        # Filter and create vocab (1-indexed for PostgreSQL sparsevec)
        filtered = [(w, c) for w, c in word_counts.items() if c >= min_df]
        vocab = {word: idx + 1 for idx, (word, _) in enumerate(filtered)}
        
        print(f"  Vocabulary: {len(vocab):,} WordPiece tokens (min_df={min_df})")
        return vocab
        return vocab


# =============================================================================
# BM25 Index (for query-time scoring)
# =============================================================================

class SparseBM25Index:
    """
    BM25 index with msgpack serialization.
    
    Stores IDF values and vocabulary for query-time sparse vector generation.
    Document vectors are stored in PostgreSQL as sparsevec.
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.vocab = {}
        self.vocab_size = 0
        self.idf = {}
        self.avgdl = 0
        self.N = 0
    
    def query_to_sparse(self, token_ids: List[int]) -> str:
        """Convert query tokens to sparsevec string."""
        if not token_ids:
            return f"{{}}/{self.vocab_size}"
        
        tf = Counter(token_ids)
        indices, values = [], []
        
        for token_id, freq in tf.items():
            if token_id not in self.idf:
                continue
            weight = self.idf[token_id] * freq
            if weight > 0:
                indices.append(token_id)
                values.append(weight)
        
        if not indices:
            return f"{{}}/{self.vocab_size}"
        
        pairs = [f"{idx}:{val:.4f}" for idx, val in zip(indices, values)]
        return f"{{{','.join(pairs)}}}/{self.vocab_size}"
    
    def save(self, path: Path):
        """Save to msgpack."""
        import msgpack
        
        data = {
            'vocab': self.vocab,
            'vocab_size': self.vocab_size,
            'idf': {str(k): v for k, v in self.idf.items()},
            'avgdl': self.avgdl,
            'N': self.N,
            'k1': self.k1,
            'b': self.b,
        }
        
        with open(path, 'wb') as f:
            f.write(msgpack.packb(data, use_bin_type=True))
    
    def load(self, path: Path):
        """Load from msgpack."""
        import msgpack
        
        with open(path, 'rb') as f:
            data = msgpack.unpackb(f.read(), raw=False, strict_map_key=False)
        
        self.vocab = data['vocab']
        self.vocab_size = data['vocab_size']
        self.idf = {int(k): v for k, v in data['idf'].items()}
        self.avgdl = data['avgdl']
        self.N = data['N']
        self.k1 = data.get('k1', 1.5)
        self.b = data.get('b', 0.75)


# =============================================================================
# Dense Embedder
# =============================================================================

class Model2VecEmbedder:
    """
    Dense embeddings using model2vec with optional PCA reduction.
    """
    
    def __init__(self, model_name: str = "minishlab/M2V_base_output", target_dim: int = 64):
        from model2vec import StaticModel
        
        self.target_dim = target_dim
        self.model = StaticModel.from_pretrained(model_name)
        
        # Check native dimension
        test = self.model.encode(["test"])
        self.native_dim = test.shape[1]
        
        self.pca = None
        self._pca_fitted = False
    
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
        embeddings = self.model.encode(texts[:n])
        
        self.pca = PCA(n_components=self.target_dim)
        self.pca.fit(embeddings)
        self._pca_fitted = True
        
        explained = sum(self.pca.explained_variance_ratio_) * 100
        print(f"  Explained variance: {explained:.1f}%")
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to normalized embeddings."""
        embeddings = self.model.encode(texts)
        
        if self.pca is not None:
            embeddings = self.pca.transform(embeddings)
        elif self.native_dim != self.target_dim:
            embeddings = embeddings[:, :self.target_dim]
        
        # L2 normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.clip(norms, 1e-10, None)
        
        return embeddings.astype(np.float32)


# =============================================================================
# Quotes GIST Retriever
# =============================================================================

class QuotesRetriever(GISTRetriever):
    """
    GIST Retriever for quotes dataset.
    
    Groups by: quote_id
    Reconstructs: Full quote from all chunks
    """
    
    def __init__(self, config: Optional[QuotesConfig] = None):
        self.quotes_config = config or QuotesConfig()
        super().__init__(self.quotes_config)
        
        self.conn = None
        self._embedder = None
        self._bm25_index = None
        self._preprocessor = None
    
    # =========================================================================
    # Connection Management
    # =========================================================================
    
    def connect(self):
        """Open database connection."""
        if self.conn is None or self.conn.closed:
            self.conn = psycopg2.connect(
                host=self.quotes_config.db_host,
                port=self.quotes_config.db_port,
                dbname=self.quotes_config.db_name,
                user=self.quotes_config.db_user,
                password=self.quotes_config.db_password
            )
            self.conn.autocommit = True
    
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
        if self._embedder is None:
            self._embedder = Model2VecEmbedder(
                model_name=self.quotes_config.embedding_model,
                target_dim=self.quotes_config.embedding_dim
            )
        return self._embedder
    
    @property
    def bm25_index(self) -> SparseBM25Index:
        if self._bm25_index is None:
            self._bm25_index = SparseBM25Index(
                k1=self.quotes_config.bm25_k1,
                b=self.quotes_config.bm25_b
            )
            if self.quotes_config.bm25_cache_path.exists():
                self._bm25_index.load(self.quotes_config.bm25_cache_path)
            else:
                raise FileNotFoundError(
                    f"BM25 index not found: {self.quotes_config.bm25_cache_path}. "
                    "Run index building first."
                )
        return self._bm25_index
    
    @property
    def preprocessor(self) -> TextPreprocessor:
        if self._preprocessor is None:
            self._preprocessor = TextPreprocessor(self.bm25_index.vocab)
        return self._preprocessor
    
    # =========================================================================
    # Abstract Method Implementations
    # =========================================================================
    
    def _retrieve_bm25(self, query: str, limit: int) -> List[RetrievedDoc]:
        """Retrieve from BM25."""
        self.connect()
        
        token_ids = self.preprocessor.preprocess(query)
        query_sparse = self.bm25_index.query_to_sparse(token_ids)
        
        with self.conn.cursor() as cur:
            cur.execute(f"""
                SELECT chunk_id, content, quote_id, chunk_idx,
                       (bm25_sparse <#> %s::sparsevec) as neg_score
                FROM {self.quotes_config.table_name}
                ORDER BY bm25_sparse <#> %s::sparsevec
                LIMIT %s
            """, (query_sparse, query_sparse, limit))
            
            results = []
            for rank, row in enumerate(cur.fetchall(), 1):
                doc = RetrievedDoc(
                    doc_id=row[0],
                    content=row[1],
                    metadata={
                        'quote_id': row[2],
                        'chunk_idx': row[3]
                    },
                    bm25_score=-row[4],
                    bm25_rank=rank
                )
                results.append(doc)
        
        return results
    
    def _retrieve_dense(self, query: str, limit: int) -> List[RetrievedDoc]:
        """Retrieve from dense index."""
        self.connect()
        
        query_emb = self.embedder.encode([query])[0]
        
        with self.conn.cursor() as cur:
            cur.execute(f"""
                SELECT chunk_id, content, quote_id, chunk_idx,
                       (embedding <=> %s::vector) as distance
                FROM {self.quotes_config.table_name}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, (query_emb.tolist(), query_emb.tolist(), limit))
            
            results = []
            for rank, row in enumerate(cur.fetchall(), 1):
                doc = RetrievedDoc(
                    doc_id=row[0],
                    content=row[1],
                    metadata={
                        'quote_id': row[2],
                        'chunk_idx': row[3]
                    },
                    dense_score=1.0 - row[4],
                    dense_rank=rank
                )
                results.append(doc)
        
        return results
    
    def _get_bm25_doc_doc_scores(self, doc_ids: List[str]) -> np.ndarray:
        """Compute pairwise BM25 similarity matrix."""
        self.connect()
        n = len(doc_ids)
        sim_matrix = np.zeros((n, n))
        
        with self.conn.cursor() as cur:
            cur.execute(f"""
                SELECT chunk_id, bm25_sparse::text
                FROM {self.quotes_config.table_name}
                WHERE chunk_id = ANY(%s)
            """, (doc_ids,))
            
            sparse_dict = {row[0]: row[1] for row in cur.fetchall()}
        
        with self.conn.cursor() as cur:
            for i, doc_id_i in enumerate(doc_ids):
                vec_i = sparse_dict.get(doc_id_i)
                if not vec_i:
                    continue
                
                cur.execute(f"""
                    SELECT chunk_id, (bm25_sparse <#> %s::sparsevec) as neg_score
                    FROM {self.quotes_config.table_name}
                    WHERE chunk_id = ANY(%s)
                """, (vec_i, doc_ids))
                
                for row in cur.fetchall():
                    j = doc_ids.index(row[0])
                    sim_matrix[i, j] = -row[1]
        
        max_val = sim_matrix.max()
        if max_val > 0:
            sim_matrix = sim_matrix / max_val
        
        np.fill_diagonal(sim_matrix, 1.0)
        return sim_matrix
    
    def _get_dense_embeddings(self, doc_ids: List[str]) -> np.ndarray:
        """Fetch dense embeddings."""
        self.connect()
        
        with self.conn.cursor() as cur:
            cur.execute(f"""
                SELECT chunk_id, embedding::text
                FROM {self.quotes_config.table_name}
                WHERE chunk_id = ANY(%s)
            """, (doc_ids,))
            
            emb_dict = {}
            for row in cur.fetchall():
                vec_str = row[1].strip('[]')
                emb_dict[row[0]] = np.array([float(x) for x in vec_str.split(',')])
        
        embeddings = np.array([emb_dict[doc_id] for doc_id in doc_ids])
        return embeddings
    
    def _get_query_embedding(self, query: str) -> np.ndarray:
        """Encode query."""
        return self.embedder.encode([query])[0]
    
    def _get_bm25_query_scores(self, doc_ids: List[str], query: str) -> np.ndarray:
        """Get BM25 scores for docs against query."""
        self.connect()
        
        token_ids = self.preprocessor.preprocess(query)
        query_sparse = self.bm25_index.query_to_sparse(token_ids)
        
        with self.conn.cursor() as cur:
            cur.execute(f"""
                SELECT chunk_id, (bm25_sparse <#> %s::sparsevec) as neg_score
                FROM {self.quotes_config.table_name}
                WHERE chunk_id = ANY(%s)
            """, (query_sparse, doc_ids))
            
            score_dict = {row[0]: -row[1] for row in cur.fetchall()}
        
        scores = np.array([score_dict.get(doc_id, 0.0) for doc_id in doc_ids])
        return scores
    
    def _get_group_key(self, doc: RetrievedDoc) -> Tuple:
        """
        Extract grouping key: (quote_id,) for quotes.
        """
        return (doc.metadata.get('quote_id'),)
    
    def _fetch_all_chunks_for_group(self, group_key: Tuple) -> List[RetrievedDoc]:
        """
        Fetch ALL chunks for a quote_id group.
        """
        self.connect()
        quote_id = group_key[0]
        
        with self.conn.cursor() as cur:
            cur.execute(f"""
                SELECT chunk_id, content, quote_id, chunk_idx
                FROM {self.quotes_config.table_name}
                WHERE quote_id = %s
                ORDER BY chunk_idx
            """, (quote_id,))
            
            chunks = []
            for row in cur.fetchall():
                doc = RetrievedDoc(
                    doc_id=row[0],
                    content=row[1],
                    metadata={
                        'quote_id': row[2],
                        'chunk_idx': row[3]
                    }
                )
                chunks.append(doc)
        
        return chunks
    
    def _group_key_to_id(self, group_key: Tuple) -> str:
        """
        Convert (quote_id,) to string identifier.
        """
        return group_key[0]
    
    # =========================================================================
    # Index Building
    # =========================================================================
    
    def build_index(
        self,
        chunks: List[Dict],
        reset: bool = False
    ):
        """
        Build hybrid index from quote chunks.
        
        Args:
            chunks: List of dicts with keys:
                    - doc_id: Unique chunk ID (e.g., "quote123_c0")
                    - quote_id: Parent quote ID
                    - chunk_idx: Chunk index within quote
                    - text: Chunk content
            reset: If True, drop existing table
        """
        from tqdm import tqdm
        
        self.connect()
        
        print("=" * 70)
        print("Building Quotes Index")
        print("=" * 70)
        
        chunk_ids = [c['doc_id'] for c in chunks]
        quote_ids = [c['quote_id'] for c in chunks]
        chunk_idxs = [c['chunk_idx'] for c in chunks]
        contents = [c['text'] for c in chunks]
        
        # Step 1: Build BM25
        print("\n[1/4] Building BM25 index...")
        vocab = TextPreprocessor.build_vocab_from_corpus(contents, min_df=2)
        preprocessor = TextPreprocessor(vocab)
        
        N = len(contents)
        df = defaultdict(int)
        for text in tqdm(contents, desc="  Computing DF", leave=False):
            token_ids = preprocessor.preprocess(text)
            for tid in set(token_ids):
                if tid >= 0:
                    df[tid] += 1
        
        idf = {tid: np.log((N - doc_freq + 0.5) / (doc_freq + 0.5) + 1)
               for tid, doc_freq in df.items()}
        
        doc_lengths = [len(preprocessor.preprocess(text)) for text in contents]
        avgdl = np.mean(doc_lengths)
        
        print("  Building sparse vectors...")
        sparse_vectors = []
        for i, text in enumerate(tqdm(contents, desc="  Vectorizing", leave=False)):
            token_ids = preprocessor.preprocess(text)
            tf = Counter(token_ids)
            
            indices, values = [], []
            for tid, freq in tf.items():
                if tid < 0 or tid not in idf:
                    continue
                
                numerator = freq * (self.quotes_config.bm25_k1 + 1)
                denominator = freq + self.quotes_config.bm25_k1 * (
                    1 - self.quotes_config.bm25_b + 
                    self.quotes_config.bm25_b * doc_lengths[i] / avgdl
                )
                weight = idf[tid] * (numerator / denominator)
                
                if weight > 0:
                    indices.append(tid)
                    values.append(weight)
            
            if indices:
                pairs = [f"{idx}:{val:.4f}" for idx, val in zip(indices, values)]
                sparse_vectors.append(f"{{{','.join(pairs)}}}/{len(vocab)}")
            else:
                sparse_vectors.append(f"{{}}/{len(vocab)}")
        
        self._bm25_index = SparseBM25Index(
            k1=self.quotes_config.bm25_k1,
            b=self.quotes_config.bm25_b
        )
        self._bm25_index.vocab = vocab
        self._bm25_index.vocab_size = len(vocab)
        self._bm25_index.idf = idf
        self._bm25_index.avgdl = avgdl
        self._bm25_index.N = N
        self._bm25_index.save(self.quotes_config.bm25_cache_path)
        
        # Step 2: Create embeddings
        print("\n[2/4] Creating dense embeddings...")
        embedder = Model2VecEmbedder(
            model_name=self.quotes_config.embedding_model,
            target_dim=self.quotes_config.embedding_dim
        )
        embedder.fit_pca(contents)
        
        embeddings = []
        batch_size = 256
        for i in tqdm(range(0, len(contents), batch_size), desc="  Embedding"):
            batch = contents[i:i + batch_size]
            embeddings.append(embedder.encode(batch))
        embeddings = np.vstack(embeddings)
        
        # Step 3: Create table
        print("\n[3/4] Creating PostgreSQL table...")
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            
            if reset:
                cur.execute(f"DROP TABLE IF EXISTS {self.quotes_config.table_name} CASCADE")
            
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.quotes_config.table_name} (
                    id SERIAL PRIMARY KEY,
                    chunk_id TEXT UNIQUE NOT NULL,
                    quote_id TEXT NOT NULL,
                    chunk_idx INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    embedding vector({self.quotes_config.embedding_dim}) NOT NULL,
                    bm25_sparse sparsevec({len(vocab)}) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
        
        # Step 4: Insert data
        print("\n[4/4] Inserting data...")
        data = list(zip(
            chunk_ids, quote_ids, chunk_idxs,
            contents,
            [emb.tolist() for emb in embeddings],
            sparse_vectors
        ))
        
        batch_size = 500
        with self.conn.cursor() as cur:
            for i in tqdm(range(0, len(data), batch_size), desc="  Inserting"):
                batch = data[i:i + batch_size]
                execute_values(
                    cur,
                    f"""
                    INSERT INTO {self.quotes_config.table_name}
                    (chunk_id, quote_id, chunk_idx, content, embedding, bm25_sparse)
                    VALUES %s
                    ON CONFLICT (chunk_id) DO UPDATE SET
                        content = EXCLUDED.content,
                        embedding = EXCLUDED.embedding,
                        bm25_sparse = EXCLUDED.bm25_sparse
                    """,
                    batch,
                    template="(%s, %s, %s, %s, %s::vector, %s::sparsevec)"
                )
        
        # Create indexes
        print("  Creating indexes...")
        with self.conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {self.quotes_config.table_name}")
            count = cur.fetchone()[0]
            lists = max(1, int(count ** 0.5))
            
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS {self.quotes_config.table_name}_emb_idx
                ON {self.quotes_config.table_name}
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = {lists})
            """)
            
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS {self.quotes_config.table_name}_quote_idx
                ON {self.quotes_config.table_name} (quote_id)
            """)
        
        print("\n" + "=" * 70)
        print(f"Index built: {len(chunks):,} chunks")
        print(f"  Dense: vector({self.quotes_config.embedding_dim}) with IVFFlat")
        print(f"  Sparse: sparsevec({len(vocab)}) BM25")
        print("=" * 70)
    
    def clear_all(self):
        """Drop all tables."""
        self.connect()
        with self.conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {self.quotes_config.table_name} CASCADE")
            print(f"Dropped table: {self.quotes_config.table_name}")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Quotes GIST Retriever")
    
    # Actions
    parser.add_argument("--search", type=str, help="Search query")
    parser.add_argument("--build", action="store_true", help="Build/rebuild index")
    parser.add_argument("--clear", action="store_true", help="Clear all tables")
    
    # Search options
    parser.add_argument("--top_k", type=int, default=21, help="Number of results")
    parser.add_argument("--save", type=str, help="Save results summary to markdown")
    parser.add_argument("--export", type=str, help="Export full quotes to file")
    parser.add_argument("--chunks", action="store_true", help="Return chunks instead of grouped quotes")
    
    # Build options
    parser.add_argument("--reset", action="store_true", help="Drop existing table before build")
    parser.add_argument("--quotes_file", type=str, help="Path to quotes data file for --build")
    
    # Database connection
    parser.add_argument("--host", type=str, default="192.168.3.18", help="Database host")
    parser.add_argument("--port", type=int, default=6024, help="Database port")
    parser.add_argument("--db", type=str, default="langchain", help="Database name")
    parser.add_argument("--user", type=str, default="langchain", help="Database user")
    parser.add_argument("--password", type=str, default="langchain", help="Database password")
    parser.add_argument("--table", type=str, default="quotes_chunks", help="Table name")
    
    args = parser.parse_args()
    
    config = QuotesConfig(
        db_host=args.host,
        db_port=args.port,
        db_name=args.db,
        db_user=args.user,
        db_password=args.password,
        table_name=args.table
    )
    
    if args.clear:
        confirm = input("Delete all tables? Type 'yes': ")
        if confirm.lower() == "yes":
            with QuotesRetriever(config) as retriever:
                retriever.clear_all()
    
    elif args.build:
        import json
        
        # If quotes_file provided, load from file
        # Otherwise, download from HuggingFace
        if args.quotes_file:
            print(f"Loading quotes from: {args.quotes_file}")
            
            if args.quotes_file.endswith('.json'):
                with open(args.quotes_file, 'r', encoding='utf-8') as f:
                    quotes_data = json.load(f)
            else:
                print("ERROR: Only JSON files supported currently")
                exit(1)
        else:
            # Download from HuggingFace
            print("No quotes file provided. Downloading from HuggingFace (abirate/english_quotes)...")
            
            try:
                from datasets import load_dataset
            except ImportError:
                print("ERROR: 'datasets' package not installed")
                print("Install with: pip install datasets")
                exit(1)
            
            print("Loading dataset from HuggingFace...")
            dataset = load_dataset("abirate/english_quotes", split="train")
            
            print(f"Loaded {len(dataset):,} quotes from HuggingFace")
            
            # Convert to expected format
            quotes_data = []
            for idx, item in enumerate(dataset):
                # Dataset has 'quote', 'author', 'tags' fields
                quote_text = item.get('quote', '')
                author = item.get('author', 'Unknown')
                
                quotes_data.append({
                    'quote_id': f"quote_{idx:06d}",
                    'text': quote_text,
                    'author': author
                })
            
            # Optionally save to file for future use
            if args.quotes_file is None:
                save_path = "quotes_dataset.json"
                print(f"Saving downloaded quotes to {save_path}...")
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(quotes_data, f, indent=2, ensure_ascii=False)
                print(f"Saved to {save_path} (use --quotes_file {save_path} to skip download next time)")
        
        # Apply equidistant chunking to quotes
        print("\nApplying equidistant chunking...")
        from equidistant_chunking import (
            create_paragraph_blocks,
            SlidingAggregator,
            AggregateConfig,
        )
        
        # Compute log median + MAD threshold for quote lengths
        quote_lengths = [len(item['text']) for item in quotes_data if 'text' in item]
        if quote_lengths:
            log_lengths = np.log([max(1, l) for l in quote_lengths])
            log_median = np.median(log_lengths)
            log_mad = np.median(np.abs(log_lengths - log_median))
            max_quote_chars = int(exp(log_median + 2 * log_mad))
            print(f"  Computed max quote threshold: {max_quote_chars} chars (log median + 2*MAD)")
        else:
            max_quote_chars = 500  # Default fallback
        
        # Compute corpus stats for chunking config
        all_texts = [item['text'] for item in quotes_data if 'text' in item]
        
        # Simple config based on corpus
        if all_texts:
            median_len = int(np.median(quote_lengths))
            mad_len = int(np.median(np.abs(np.array(quote_lengths) - median_len)))
            
            config_chunk = AggregateConfig(
                target=median_len,
                tolerance=mad_len * 2,
                ratio=2.0,
                overlap_pct=0.3,
                chars_per_word=5.0
            )
            print(f"  Chunking config: target={config_chunk.target}, tolerance={config_chunk.tolerance}")
        else:
            config_chunk = AggregateConfig(target=200, tolerance=100, ratio=2.0, overlap_pct=0.3)
        
        # Chunk quotes that exceed threshold
        chunk_dicts = []
        for item in quotes_data:
            quote_id = item['quote_id']
            text = item['text']
            
            if len(text) <= max_quote_chars:
                # Small quote - single chunk
                chunk_dicts.append({
                    'doc_id': f"{quote_id}_c0",
                    'quote_id': quote_id,
                    'chunk_idx': 0,
                    'text': text
                })
            else:
                # Large quote - apply equidistant chunking
                blocks = create_paragraph_blocks(text, max_quote_chars)
                agg = SlidingAggregator(config_chunk)
                chunks = []
                
                for block in blocks:
                    for chunk in agg.feed(block):
                        chunks.append(chunk)
                
                for chunk in agg.finish(chunks):
                    chunks.append(chunk)
                
                # Add chunked results
                for idx, chunk_text in enumerate(chunks):
                    chunk_dicts.append({
                        'doc_id': f"{quote_id}_c{idx}",
                        'quote_id': quote_id,
                        'chunk_idx': idx,
                        'text': chunk_text
                    })
        
        print(f"Processed {len(quotes_data):,} quotes → {len(chunk_dicts):,} chunks")
        
        with QuotesRetriever(config) as retriever:
            retriever.build_index(chunk_dicts, reset=args.reset)
    
    elif args.search:
        with QuotesRetriever(config) as retriever:
            if args.chunks:
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
                results = retriever.search(args.search, top_k=args.top_k)
                
                for i, group in enumerate(results, 1):
                    print(f"\n{i}. {group.group_id}")
                    print(f"   Score: {group.final_score:.4f}")
                    print(f"   Chunks: {group.metadata.get('num_chunks', '?')}")
                    preview = group.full_text[:150] + "..." if len(group.full_text) > 150 else group.full_text
                    print(f"   Preview: {preview}")
                
                if args.save:
                    md = format_groups_markdown(results, include_full_text=False)
                    with open(args.save, 'w', encoding='utf-8') as f:
                        f.write(md)
                    print(f"\nSummary saved to {args.save}")
                
                if args.export:
                    with open(args.export, 'w', encoding='utf-8') as f:
                        f.write(f"# Search Results: {args.search}\n\n")
                        f.write(f"**Query:** {args.search}\n")
                        f.write(f"**Results:** {len(results)} quotes\n\n")
                        f.write("---\n\n")
                        
                        for i, group in enumerate(results, 1):
                            f.write(f"## {i}. {group.group_id}\n\n")
                            f.write(f"**Score:** {group.final_score:.4f}\n")
                            f.write(f"**Chunks:** {group.metadata.get('num_chunks', '?')}\n\n")
                            f.write("### Full Quote\n\n")
                            f.write(group.full_text)
                            f.write("\n\n---\n\n")
                    
                    print(f"\nFull quotes exported to {args.export}")
    
    else:
        parser.print_help()