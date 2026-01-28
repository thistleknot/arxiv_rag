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

import re
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set, Any
from collections import Counter, defaultdict

import psycopg2
from psycopg2.extras import execute_values

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
    embedding_dim: int = 64
    embedding_model: str = "minishlab/M2V_base_output"
    use_full_embed: bool = False  # If True, disable model2vec and use full sentence embeddings
    
    # BM25
    bm25_cache_path: Path = Path("bm25_vocab.msgpack")
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    use_lemmatized: bool = False  # If True, use lemmatization for BM25 instead of BERT tokenizer


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


class LemmatizedTextPreprocessor:
    """
    Lemmatized tokenizer for BM25 with improved coverage.
    
    Custom tokenization strategy:
      1. Split on transitions between alpha/numeric/punctuation
         Example: "abc123???" → ["abc", "123", "???"]
                  "s/n 1234-abc12" → ["s", "/", "n", "1234", "-", "abc", "12"]
      2. Lemmatize each token
      3. Map to vocab IDs
    
    This provides better BM25 coverage for technical text, serial numbers,
    mixed alphanumeric strings, and punctuation-heavy content.
    """
    
    _lemmatizer = None  # Class-level cache
    
    def __init__(self, vocab_dict: Dict[str, int] = None):
        self.vocab = vocab_dict if vocab_dict else {}
        self.vocab_size = len(self.vocab) if vocab_dict else 0
        
        # Lazy load lemmatizer (shared across instances)
        if LemmatizedTextPreprocessor._lemmatizer is None:
            import spacy
            try:
                LemmatizedTextPreprocessor._lemmatizer = spacy.load("en_core_web_sm", disable=["parser", "ner"])
            except OSError:
                print("Downloading spacy en_core_web_sm model...")
                import subprocess
                subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
                LemmatizedTextPreprocessor._lemmatizer = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    
    @staticmethod
    def custom_tokenize(text: str) -> List[str]:
        """
        Split text on transitions between character types.
        
        Character types: alphabetic, numeric, punctuation, whitespace
        
        Examples:
          - "abc123???" → ["abc", "123", "???"]
          - "Mrs. Reindeer" → ["Mrs", ".", "Reindeer"]
          - "s/n 1234-abc12" → ["s", "/", "n", "1234", "-", "abc", "12"]
          - "envelope?" → ["envelope", "?"]
        """
        import re
        
        if not text:
            return []
        
        # Pattern: sequences of same character type
        # \w+ matches alphanumeric (but we want to split alpha from numeric)
        # So we use more specific patterns
        pattern = r'[a-zA-Z]+|[0-9]+|[^a-zA-Z0-9\s]+|\s+'
        tokens = re.findall(pattern, text)
        
        # Filter out whitespace-only tokens
        return [t for t in tokens if t.strip()]
    
    def preprocess(self, text: str) -> List[int]:
        """Tokenize text to vocab IDs using custom tokenization + lemmatization."""
        if not isinstance(text, str):
            text = str(text)
        
        # Custom tokenization
        tokens = self.custom_tokenize(text)
        
        # Lemmatize
        doc = LemmatizedTextPreprocessor._lemmatizer(" ".join(tokens))
        lemmas = [token.lemma_.lower() for token in doc]
        
        # Map to vocab IDs
        return [self.vocab.get(t, -1) for t in lemmas if t in self.vocab]
    
    @staticmethod
    def build_vocab_from_corpus(
        texts: List[str], 
        min_df: int = 2
    ) -> Dict[str, int]:
        """Build vocabulary from corpus using lemmatization."""
        from tqdm import tqdm
        import spacy
        
        print(f"  Building lemmatized vocabulary from {len(texts):,} texts...")
        
        # Load spacy
        try:
            nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        except OSError:
            print("  Downloading spacy en_core_web_sm model...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
            nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        
        word_counts = Counter()
        for text in tqdm(texts, desc="  Tokenizing + Lemmatizing", leave=False):
            tokens = LemmatizedTextPreprocessor.custom_tokenize(str(text))
            doc = nlp(" ".join(tokens))
            lemmas = [token.lemma_.lower() for token in doc]
            word_counts.update(lemmas)
        
        # Filter and create vocab (1-indexed for PostgreSQL sparsevec)
        filtered = [(w, c) for w, c in word_counts.items() if c >= min_df]
        vocab = {word: idx + 1 for idx, (word, _) in enumerate(filtered)}
        
        print(f"  Vocabulary: {len(vocab):,} lemmatized tokens (min_df={min_df})")
        return vocab
    
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
        """Convert query tokens to sparsevec string for PostgreSQL."""
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
    Dense embeddings using model2vec with optional PCA reduction,
    or full sentence embeddings via sentence-transformers.
    """
    
    def __init__(self, model_name: str = "minishlab/M2V_base_output", target_dim: int = 64, use_full_embed: bool = False):
        self.target_dim = target_dim
        self.use_full_embed = use_full_embed
        self.model = self._load_model(model_name)
        
        # Check native dimension
        test = self.encode(["test"])
        self.native_dim = test.shape[1]
        
        self.pca = None
        self._pca_fitted = False
    
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
        self._bm25_index = None
        self._preprocessor = None
    
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
        """Lazy-load embedder."""
        if self._embedder is None:
            self._embedder = Model2VecEmbedder(
                model_name=self.pg_config.embedding_model,
                target_dim=self.pg_config.embedding_dim,
                use_full_embed=self.pg_config.use_full_embed
            )
        return self._embedder
    
    @property
    def bm25_index(self) -> SparseBM25Index:
        """Lazy-load BM25 index."""
        if self._bm25_index is None:
            self._bm25_index = SparseBM25Index(
                k1=self.pg_config.bm25_k1,
                b=self.pg_config.bm25_b
            )
            if self.pg_config.bm25_cache_path.exists():
                self._bm25_index.load(self.pg_config.bm25_cache_path)
            else:
                raise FileNotFoundError(
                    f"BM25 index not found: {self.pg_config.bm25_cache_path}. "
                    "Run index building first."
                )
        return self._bm25_index
    
    @property
    def preprocessor(self):
        """Lazy-load preprocessor with BM25 vocab."""
        if self._preprocessor is None:
            if self.pg_config.use_lemmatized:
                self._preprocessor = LemmatizedTextPreprocessor(self.bm25_index.vocab)
            else:
                self._preprocessor = TextPreprocessor(self.bm25_index.vocab)
        return self._preprocessor
    
    # =========================================================================
    # Abstract Method Implementations
    # =========================================================================
    
    def _retrieve_bm25(self, query: str, limit: int) -> List[RetrievedDoc]:
        """Retrieve from BM25 (sparsevec inner product)."""
        self.connect()
        
        # Convert query to sparse vector
        token_ids = self.preprocessor.preprocess(query)
        query_sparse = self.bm25_index.query_to_sparse(token_ids)
        
        with self.conn.cursor() as cur:
            # Use negative inner product operator (<#>) for ordering
            # (pgvector returns negative for inner product to make ORDER BY work)
            cur.execute(f"""
                SELECT chunk_id, content, paper_id, section_idx, chunk_idx,
                       (bm25_sparse <#> %s::sparsevec) as neg_score
                FROM {self.pg_config.table_name}
                ORDER BY bm25_sparse <#> %s::sparsevec
                LIMIT %s
            """, (query_sparse, query_sparse, limit))
            
            results = []
            for rank, row in enumerate(cur.fetchall(), 1):
                doc = RetrievedDoc(
                    doc_id=row[0],
                    content=row[1],
                    metadata={
                        'paper_id': row[2],
                        'section_idx': row[3],
                        'chunk_idx': row[4]
                    },
                    bm25_score=-row[5],  # Negate back to positive
                    bm25_rank=rank
                )
                results.append(doc)
        
        return results
    
    def _retrieve_dense(self, query: str, limit: int) -> List[RetrievedDoc]:
        """Retrieve from dense index (cosine similarity)."""
        self.connect()
        
        # Encode query
        query_emb = self.embedder.encode([query])[0]
        
        with self.conn.cursor() as cur:
            # Cosine distance (lower = more similar)
            cur.execute(f"""
                SELECT chunk_id, content, paper_id, section_idx, chunk_idx,
                       (embedding <=> %s::vector) as distance
                FROM {self.pg_config.table_name}
                ORDER BY embedding <=> %s::vector
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
                        'chunk_idx': row[4]
                    },
                    dense_score=1.0 - row[5],  # Convert distance to similarity
                    dense_rank=rank
                )
                results.append(doc)
        
        return results
    
    def _get_bm25_doc_doc_scores(self, doc_ids: List[str]) -> np.ndarray:
        """
        Compute pairwise BM25 similarity matrix.
        
        Uses sparsevec inner product in PostgreSQL.
        """
        self.connect()
        n = len(doc_ids)
        sim_matrix = np.zeros((n, n))
        
        # Fetch sparse vectors
        with self.conn.cursor() as cur:
            cur.execute(f"""
                SELECT chunk_id, bm25_sparse::text
                FROM {self.pg_config.table_name}
                WHERE chunk_id = ANY(%s)
            """, (doc_ids,))
            
            sparse_dict = {row[0]: row[1] for row in cur.fetchall()}
        
        # Compute pairwise similarities
        with self.conn.cursor() as cur:
            for i, doc_id_i in enumerate(doc_ids):
                vec_i = sparse_dict.get(doc_id_i)
                if not vec_i:
                    continue
                
                # Query all similarities for this doc
                cur.execute(f"""
                    SELECT chunk_id, (bm25_sparse <#> %s::sparsevec) as neg_score
                    FROM {self.pg_config.table_name}
                    WHERE chunk_id = ANY(%s)
                """, (vec_i, doc_ids))
                
                for row in cur.fetchall():
                    j = doc_ids.index(row[0])
                    sim_matrix[i, j] = -row[1]  # Negate to positive
        
        # Normalize to [0, 1]
        max_val = sim_matrix.max()
        if max_val > 0:
            sim_matrix = sim_matrix / max_val
        
        # Ensure diagonal is 1
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
        """Get BM25 scores for docs against query."""
        self.connect()
        
        # Convert query to sparse
        token_ids = self.preprocessor.preprocess(query)
        query_sparse = self.bm25_index.query_to_sparse(token_ids)
        
        with self.conn.cursor() as cur:
            cur.execute(f"""
                SELECT chunk_id, (bm25_sparse <#> %s::sparsevec) as neg_score
                FROM {self.pg_config.table_name}
                WHERE chunk_id = ANY(%s)
            """, (query_sparse, doc_ids))
            
            score_dict = {row[0]: -row[1] for row in cur.fetchall()}
        
        # Preserve order
        scores = np.array([score_dict.get(doc_id, 0.0) for doc_id in doc_ids])
        return scores
    
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
        Build hybrid index from chunks.
        
        Args:
            chunks: List of dicts with keys:
                    - doc_id: Unique chunk ID
                    - paper_id: Parent document ID
                    - section_idx: Section index
                    - chunk_idx: Chunk index within section
                    - text: Chunk content
            reset: If True, drop existing table
        """
        from tqdm import tqdm
        
        self.connect()
        
        print("=" * 70)
        print("Building PGVector Index")
        print("=" * 70)
        
        # Extract data
        chunk_ids = [c['doc_id'] for c in chunks]
        paper_ids = [c['paper_id'] for c in chunks]
        section_idxs = [c['section_idx'] for c in chunks]
        chunk_idxs = [c['chunk_idx'] for c in chunks]
        contents = [c['text'] for c in chunks]
        
        # Step 1: Build BM25 vocabulary and IDF
        print("\n[1/4] Building BM25 index...")
        vocab = TextPreprocessor.build_vocab_from_corpus(contents, min_df=2)
        preprocessor = TextPreprocessor(vocab)
        
        # Compute IDF
        N = len(contents)
        df = defaultdict(int)
        for text in tqdm(contents, desc="  Computing DF", leave=False):
            token_ids = preprocessor.preprocess(text)
            for tid in set(token_ids):
                if tid >= 0:
                    df[tid] += 1
        
        idf = {tid: np.log((N - doc_freq + 0.5) / (doc_freq + 0.5) + 1)
               for tid, doc_freq in df.items()}
        
        # Compute document lengths for BM25
        doc_lengths = []
        for text in contents:
            doc_lengths.append(len(preprocessor.preprocess(text)))
        avgdl = np.mean(doc_lengths)
        
        # Build sparse vectors
        print("  Building sparse vectors...")
        sparse_vectors = []
        for i, text in enumerate(tqdm(contents, desc="  Vectorizing", leave=False)):
            token_ids = preprocessor.preprocess(text)
            tf = Counter(token_ids)
            
            indices, values = [], []
            for tid, freq in tf.items():
                if tid < 0 or tid not in idf:
                    continue
                
                # BM25 weight
                numerator = freq * (self.pg_config.bm25_k1 + 1)
                denominator = freq + self.pg_config.bm25_k1 * (
                    1 - self.pg_config.bm25_b + 
                    self.pg_config.bm25_b * doc_lengths[i] / avgdl
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
        
        # Save BM25 index
        self._bm25_index = SparseBM25Index(
            k1=self.pg_config.bm25_k1,
            b=self.pg_config.bm25_b
        )
        self._bm25_index.vocab = vocab
        self._bm25_index.vocab_size = len(vocab)
        self._bm25_index.idf = idf
        self._bm25_index.avgdl = avgdl
        self._bm25_index.N = N
        self._bm25_index.save(self.pg_config.bm25_cache_path)
        
        # Step 2: Create dense embeddings
        print("\n[2/4] Creating dense embeddings...")
        embedder = Model2VecEmbedder(
            model_name=self.pg_config.embedding_model,
            target_dim=self.pg_config.embedding_dim
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
                cur.execute(f"DROP TABLE IF EXISTS {self.pg_config.table_name} CASCADE")
            
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.pg_config.table_name} (
                    id SERIAL PRIMARY KEY,
                    chunk_id TEXT UNIQUE NOT NULL,
                    paper_id TEXT NOT NULL,
                    section_idx INTEGER NOT NULL,
                    chunk_idx INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    embedding vector({self.pg_config.embedding_dim}) NOT NULL,
                    bm25_sparse sparsevec({len(vocab)}) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
        
        # Step 4: Insert data
        print("\n[4/4] Inserting data...")
        data = list(zip(
            chunk_ids, paper_ids, section_idxs, chunk_idxs,
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
                    INSERT INTO {self.pg_config.table_name}
                    (chunk_id, paper_id, section_idx, chunk_idx, content, embedding, bm25_sparse)
                    VALUES %s
                    ON CONFLICT (chunk_id) DO UPDATE SET
                        content = EXCLUDED.content,
                        embedding = EXCLUDED.embedding,
                        bm25_sparse = EXCLUDED.bm25_sparse
                    """,
                    batch,
                    template="(%s, %s, %s, %s, %s, %s::vector, %s::sparsevec)"
                )
        
        # Create indexes
        print("  Creating indexes...")
        with self.conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {self.pg_config.table_name}")
            count = cur.fetchone()[0]
            lists = max(1, int(count ** 0.5))
            
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS {self.pg_config.table_name}_emb_idx
                ON {self.pg_config.table_name}
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = {lists})
            """)
            
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS {self.pg_config.table_name}_paper_idx
                ON {self.pg_config.table_name} (paper_id)
            """)
        
        print("\n" + "=" * 70)
        print(f"Index built: {len(chunks):,} chunks")
        print(f"  Dense: vector({self.pg_config.embedding_dim}) with IVFFlat")
        print(f"  Sparse: sparsevec({len(vocab)}) BM25")
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
    parser.add_argument("--lemma", action="store_true", help="Use lemmatization for BM25 (better coverage)")
    parser.add_argument("--fullembed", action="store_true", help="Use full sentence embeddings instead of model2vec")
    
    # Database connection args
    parser.add_argument("--host", type=str, default="192.168.3.18", help="Database host")
    parser.add_argument("--port", type=int, default=6024, help="Database port")
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
        use_lemmatized=args.lemma,
        use_full_embed=args.fullembed
    )
    
    if args.clear:
        confirm = input("Delete all tables? Type 'yes': ")
        if confirm.lower() == "yes":
            with PGVectorRetriever(config) as retriever:
                retriever.clear_all()
    
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