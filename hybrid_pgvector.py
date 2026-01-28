"""
Hybrid Retrieval: model2vec dense + BM25 sparse vectors in pgvector + ColBERT late interaction

Preprocessing:
    1. Remove all non-alphanumeric characters
    2. Insert spaces between letters and numbers
    3. Collapse to single spaces
    4. Lowercase
    5. Lemmatize with spaCy

Storage:
    - Dense: vector(64) with IVFFlat + cosine
    - Sparse: sparsevec for BM25

Search Pipeline:
    1. Fibonacci cascade: retrieve → rerank → RRF
    2. Group by metadata (sections/papers)
    3. ColBERT late interaction reranking (optional)
    4. Return top_k groups

Usage:
    python hybrid_pgvector.py --build     # Build index from chunks
    python hybrid_pgvector.py --search "query"  # Search
    python hybrid_pgvector.py --clear     # Clear all tables
"""

import re
import msgpack
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Set
from collections import Counter
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import msgpack
import warnings
from pathlib import Path
import random

import psycopg2
from psycopg2.extras import execute_values
from collections import defaultdict


# =============================================================================
# Cross-Encoder Scorer (MS MARCO)
# =============================================================================

@dataclass
class CrossEncoderConfig:
    """Configuration for cross-encoder reranker."""
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    max_length: int = 512
    device: Optional[str] = None
    batch_size: int = 8


class CrossEncoderScorer:
    """
    Cross-Encoder Reranker using MS MARCO model
    
    Encodes query+document pairs jointly for fine-grained relevance scoring.
    More expensive but more accurate than bi-encoders.
    """
    
    def __init__(self, config: Optional[CrossEncoderConfig] = None):
        self.config = config or CrossEncoderConfig()
        self.available = False
        self.model = None
        
        if self.config.device is None:
            try:
                import torch
                self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = self.config.device
        
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Load cross-encoder model."""
        try:
            from sentence_transformers import CrossEncoder
            
            print(f"Loading Cross-Encoder model ({self.config.model_name})...")
            self.model = CrossEncoder(self.config.model_name, max_length=self.config.max_length, device=self.device)
            self.available = True
            print(f"Cross-Encoder initialized on {self.device}")
            
        except ImportError as e:
            print(f"WARNING: sentence-transformers not installed. Cross-Encoder unavailable. Error: {e}")
            self.available = False
        except Exception as e:
            print(f"WARNING: Cross-Encoder init failed: {e}")
            self.available = False
    
    def score(self, query: str, documents: List[str]) -> np.ndarray:
        """Score query-document pairs with cross-encoder."""
        if not self.available:
            print("WARNING: Cross-Encoder not available, returning zeros")
            return np.zeros(len(documents))
        
        if not documents:
            print("WARNING: No documents to score")
            return np.array([])
        
        try:
            print(f"  Cross-Encoder scoring {len(documents)} documents...")
            
            # Create query-document pairs
            pairs = [[query, doc] for doc in documents]
            
            # Score all pairs
            scores = self.model.predict(pairs, batch_size=self.config.batch_size, show_progress_bar=False)
            
            print(f"  Cross-Encoder scores range: [{np.min(scores):.3f}, {np.max(scores):.3f}]")
            return scores
            
        except Exception as e:
            print(f"ERROR: Cross-Encoder scoring failed: {e}")
            return np.zeros(len(documents))


# =============================================================================
# ColBERT Late Interaction Scorer
# =============================================================================

@dataclass
class ColBERTConfig:
    """Configuration for ColBERT late interaction scorer."""
    model_name: str = "bert-base-uncased"
    query_max_length: int = 32
    doc_max_length: int = 512
    dim: int = 128
    device: Optional[str] = None
    batch_size: int = 8
    normalize_embeddings: bool = True


class ColBERTScorer:
    """
    ColBERT Late Interaction Scorer
    
    Token-level MaxSim scoring: Score(Q,D) = Σᵢ maxⱼ sim(qᵢ, dⱼ)
    """
    
    def __init__(self, config: Optional[ColBERTConfig] = None):
        self.config = config or ColBERTConfig()
        self.available = False
        self.model = None
        self.tokenizer = None
        self.linear = None
        
        if self.config.device is None:
            try:
                import torch
                self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = self.config.device
        
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Load BERT model and projection layer."""
        try:
            import torch
            from transformers import AutoTokenizer, AutoModel
            
            print("Loading ColBERT model (bert-base-uncased)...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            self.model = AutoModel.from_pretrained(self.config.model_name)
            
            hidden_size = self.model.config.hidden_size
            self.linear = torch.nn.Linear(hidden_size, self.config.dim, bias=False)
            torch.nn.init.xavier_uniform_(self.linear.weight)
            
            self.model = self.model.to(self.device)
            self.model.eval()
            self.linear = self.linear.to(self.device)
            
            self.available = True
            print(f"ColBERT initialized on {self.device}")
            
        except ImportError as e:
            print(f"WARNING: transformers/torch not installed. ColBERT unavailable. Error: {e}")
            self.available = False
        except Exception as e:
            print(f"WARNING: ColBERT init failed: {e}")
            self.available = False
    
    def _encode(self, texts: List[str], max_length: int, is_query: bool = False):
        """Encode texts to token embeddings."""
        import torch
        
        if is_query:
            texts = [f"[Q] {t}" for t in texts]
        else:
            texts = [f"[D] {t}" for t in texts]
        
        encoded = self.tokenizer(
            texts, padding="max_length", truncation=True,
            max_length=max_length, return_tensors="pt"
        )
        
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = outputs.last_hidden_state
        
        embeddings = self.linear(embeddings)
        
        if self.config.normalize_embeddings:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
        
        mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size())
        embeddings = embeddings * mask_expanded.float()
        
        return embeddings, attention_mask
    
    def _maxsim(self, query_emb, query_mask, doc_embs, doc_masks):
        """Compute MaxSim scores."""
        import torch
        
        n_docs = doc_embs.size(0)
        query_expanded = query_emb.expand(n_docs, -1, -1)
        sim_matrix = torch.bmm(query_expanded, doc_embs.transpose(1, 2))
        
        doc_mask_expanded = doc_masks.unsqueeze(1).expand(-1, query_emb.size(1), -1)
        sim_matrix = sim_matrix.masked_fill(~doc_mask_expanded.bool(), float('-inf'))
        
        max_sim_per_query_token, _ = sim_matrix.max(dim=-1)
        
        query_mask_expanded = query_mask.expand(n_docs, -1)
        max_sim_per_query_token = max_sim_per_query_token.masked_fill(
            ~query_mask_expanded.bool(), 0.0
        )
        
        scores = max_sim_per_query_token.sum(dim=-1)
        return scores
    
    def score(self, query: str, documents: List[str]) -> np.ndarray:
        """Score documents with ColBERT late interaction."""
        if not self.available:
            print("WARNING: ColBERT not available, returning zeros")
            return np.zeros(len(documents))
        
        if not documents:
            print("WARNING: No documents to score")
            return np.array([])
        
        try:
            print(f"  ColBERT scoring {len(documents)} documents...")
            query_emb, query_mask = self._encode(
                [query], self.config.query_max_length, is_query=True
            )
            
            all_scores = []
            for i in range(0, len(documents), self.config.batch_size):
                batch = documents[i:i + self.config.batch_size]
                doc_embs, doc_masks = self._encode(
                    batch, self.config.doc_max_length, is_query=False
                )
                batch_scores = self._maxsim(query_emb, query_mask, doc_embs, doc_masks)
                all_scores.append(batch_scores.detach().cpu().numpy())
            
            scores = np.concatenate(all_scores)
            print(f"  ColBERT scores range: [{np.min(scores):.3f}, {np.max(scores):.3f}]")
            return scores
            
        except Exception as e:
            print(f"ERROR: ColBERT scoring failed: {e}")
            return np.zeros(len(documents))


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class HybridConfig:
    # Database
    db_host: str = "192.168.3.18"
    db_port: int = 6024
    db_name: str = "langchain"
    db_user: str = "langchain"
    
    # Tokenizer for BM25
    tokenizer_model: str = "bert-base-uncased"
    db_password: str = "langchain"
    
    # Table
    table_name: str = "arxiv_chunks"
    
    # Embeddings
    embedding_dim: int = 64
    
    # BM25
    bm25_cache_path: Path = Path("bm25_vocab.msgpack")
    
    # Search
    rrf_k: int = 60


# =============================================================================
# Text Preprocessing
# =============================================================================

class TextPreprocessor:
    """
    Fast regex-based tokenizer (no transformers dependency).
    Builds vocabulary from corpus - much faster than BERT tokenizer loading.
    """
    
    def __init__(self, vocab_dict: Dict[str, int] = None):
        self.vocab = vocab_dict if vocab_dict else {}
        self.vocab_size = len(self.vocab) if vocab_dict else 0
        self._multi_space = re.compile(r'\s+')
        self._word_pattern = re.compile(r'[a-z0-9]+')
    
    def preprocess(self, text: str) -> List[int]:
        """Tokenize text to token IDs."""
        if not isinstance(text, str):
            text = str(text)
        
        # Normalize and tokenize
        text = self._multi_space.sub(' ', text).strip().lower()
        words = self._word_pattern.findall(text)
        
        # Convert to IDs
        return [self.vocab.get(w, -1) for w in words if w in self.vocab]
    
    def preprocess_batch(self, texts: List[str], batch_size: int = 1000) -> List[List[int]]:
        """Batch tokenization."""
        print(f"  Tokenizing {len(texts):,} texts (regex)...")
        results = []
        for text in tqdm(texts, desc="  Tokenizing", leave=False):
            results.append(self.preprocess(text))
        return results
    
    @staticmethod
    def build_vocab_from_corpus(texts: List[str], min_df: int = 2) -> Dict[str, int]:
        """Build vocabulary from corpus."""
        print(f"  Building vocabulary from {len(texts):,} texts...")
        word_pattern = re.compile(r'[a-z0-9]+')
        multi_space = re.compile(r'\s+')
        
        # Count words
        word_counts = Counter()
        for text in tqdm(texts, desc="  Counting words", leave=False):
            text = multi_space.sub(' ', str(text)).strip().lower()
            words = word_pattern.findall(text)
            word_counts.update(words)
        
        # Filter by min_df and create vocab (1-based indexing for PostgreSQL sparsevec)
        filtered_words = [(word, count) for word, count in word_counts.items() if count >= min_df]
        vocab = {word: idx + 1 for idx, (word, count) in enumerate(filtered_words)}
        print(f"  Vocabulary size: {len(vocab):,} words (min_df={min_df})")
        return vocab


# =============================================================================
# Parallel Preprocessing
# =============================================================================

def _preprocess_chunk(text_with_idx):
    """Worker function for parallel preprocessing."""
    idx, text = text_with_idx
    if not isinstance(text, str):
        text = str(text)
    preprocessor = TextPreprocessor()
    return preprocessor.preprocess(text)

def parallel_preprocess(texts: List[str], n_workers: int = None) -> List[List[str]]:
    """Preprocess texts in parallel using multiprocessing."""
    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)
    
    print(f"  Using {n_workers} workers...")
    
    # Create enumerated list for worker function
    text_with_idx = list(enumerate(texts))
    
    with mp.Pool(n_workers) as pool:
        tokenized = list(tqdm(
            pool.imap(_preprocess_chunk, text_with_idx, chunksize=100),
            total=len(texts),
            desc="  Preprocessing"
        ))
    
    return tokenized


# =============================================================================
# BM25 Inverted Index (Sparse, Msgpack-optimized)
# =============================================================================

class SparseBM25Index:
    """
    Fast BM25 with inverted index (only processes docs sharing query terms).
    
    Key optimizations:
    - Tokenizer-based vocabulary (BERT WordPiece ~30K tokens)
    - No lemmatization (faster tokenization)
    - Inverted index: term_id → [(doc_id, weight), ...]
    - Only non-zero weights stored (sparse)
    - Fast search: Only checks docs with query term overlap
    - Msgpack serialization for compact storage
    
    Typical speedup: 100-1000x vs dense BM25 for sparse corpora.
    """
    
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.vocab_size = None  # Set during fit()
        self.df = None  # Document frequency per term
        self.idf = None  # IDF scores
        self.inverted_index = {}  # term_id → [(doc_id, weight), ...]
        self.doc_index = {}  # doc_id → {term_id: weight}
        self.doc_norms = None  # L2 norms for cosine normalization
        self.doc_lengths = None
        self.avgdl = 0
        self.N = 0
    
    def fit(self, raw_docs: List[str]):
        """Build inverted index with vocabulary from corpus."""
        print(f"Building sparse BM25 inverted index (regex tokenizer)...")
        
        self.N = len(raw_docs)
        
        # Build vocabulary from corpus (min_df=2 to filter rare words)
        vocab_dict = TextPreprocessor.build_vocab_from_corpus(raw_docs, min_df=2)
        self.vocab_size = len(vocab_dict)
        
        # Create preprocessor with this vocab
        preprocessor = TextPreprocessor(vocab_dict)
        
        # Tokenize ALL docs (fast regex)
        print(f"  Tokenizing all {self.N:,} docs (regex)...")
        tokenized_docs = preprocessor.preprocess_batch(raw_docs)
        
        # Compute document frequency (DF)
        print("  Computing document frequencies...")
        self.df = defaultdict(int)
        for token_ids in tqdm(tokenized_docs, desc="  Computing DF", leave=False):
            unique_tokens = set(token_ids)
            for token_id in unique_tokens:
                if token_id >= 0:  # Valid token
                    self.df[token_id] += 1
        
        # Compute IDF
        self.idf = {}
        for token_id, doc_freq in self.df.items():
            self.idf[token_id] = np.log((self.N - doc_freq + 0.5) / (doc_freq + 0.5) + 1)
        
        # Doc lengths
        self.doc_lengths = np.array([len(token_ids) for token_ids in tokenized_docs])
        self.avgdl = self.doc_lengths.mean()
        print(f"  Average doc length: {self.avgdl:.1f} tokens")
        
        # Build inverted index
        self.inverted_index = defaultdict(list)
        self.doc_norms = np.zeros(self.N)
        self.doc_index = {}
        
        for doc_id, token_ids in enumerate(tqdm(tokenized_docs, desc="  Building index", leave=False)):
            if not token_ids:
                continue
            
            tf = Counter(token_ids)
            doc_vec = {}
            
            for token_id, freq in tf.items():
                if token_id < 0 or token_id not in self.idf:
                    continue
                
                idf = self.idf[token_id]
                
                # BM25 weight
                numerator = freq * (self.k1 + 1)
                denominator = freq + self.k1 * (1 - self.b + self.b * self.doc_lengths[doc_id] / self.avgdl)
                weight = idf * (numerator / denominator)
                
                doc_vec[token_id] = weight
                self.inverted_index[token_id].append((doc_id, weight))
            
            self.doc_index[doc_id] = doc_vec
            self.doc_norms[doc_id] = np.sqrt(sum(w**2 for w in doc_vec.values()))
        
        # Save vocab for query-time use
        self.vocab = vocab_dict
        
        print(f"  Index built: {self.N:,} docs, {len(self.inverted_index):,} terms with postings")
    
    def query_to_sparse(self, token_ids: List[int]) -> str:
        """Convert query token IDs to sparse vector (TF-IDF)."""
        if not token_ids:
            return f"{{}}/{self.vocab_size}"
        
        tf = Counter(token_ids)
        
        indices = []
        values = []
        
        for token_id, freq in tf.items():
            if token_id not in self.idf:
                continue
            
            idf = self.idf[token_id]
            weight = idf * freq
            
            if weight > 0:
                indices.append(token_id)
                values.append(weight)
        
        if not indices:
            return f"{{}}/{self.vocab_size}"
        
        pairs = [f"{idx}:{val:.4f}" for idx, val in zip(indices, values)]
        return f"{{{','.join(pairs)}}}/{self.vocab_size}"
    
    def save(self, path: Path):
        """Save to msgpack (compact)."""
        import msgpack
        print(f"  Saving BM25 index to {path}...")
        data = {
            'vocab': self.vocab,  # Save vocab dict for query-time
            'vocab_size': self.vocab_size,
            'df': {str(k): v for k, v in self.df.items()},  # Convert int keys to str
            'idf': {str(k): v for k, v in self.idf.items()},  # Convert int keys to str
            'doc_lengths': self.doc_lengths.tolist(),
            'avgdl': self.avgdl,
            'N': self.N,
            'k1': self.k1,
            'b': self.b,
        }
        with open(path, 'wb') as f:
            packed = msgpack.packb(data, use_bin_type=True)
            f.write(packed)
        print(f"  Saved {len(packed):,} bytes")
    
    def load(self, path: Path):
        """Load from msgpack."""
        import msgpack
        print(f"  Loading BM25 index from {path}...")
        with open(path, 'rb') as f:
            data = msgpack.unpackb(f.read(), raw=False, strict_map_key=False)
        
        self.vocab = data['vocab']  # Load vocab dict for query-time
        self.vocab_size = data['vocab_size']
        self.df = {int(k): v for k, v in data['df'].items()}  # Convert str keys back to int
        self.idf = {int(k): v for k, v in data['idf'].items()}  # Convert str keys back to int
        self.doc_lengths = np.array(data['doc_lengths'])
        self.avgdl = data['avgdl']
        self.N = data['N']
        self.k1 = data.get('k1', 1.5)
        self.b = data.get('b', 0.75)
        print(f"  Loaded: {self.N:,} docs, {len(self.idf):,} terms")
    



# =============================================================================
# Model2Vec Embedder
# =============================================================================

class Model2VecEmbedder:
    """64-dim embeddings from model2vec."""
    
    def __init__(self, target_dim: int = 64):
        from model2vec import StaticModel
        
        self.target_dim = target_dim
        
        print("Loading model2vec...")
        self.model = StaticModel.from_pretrained("minishlab/M2V_base_output")
        
        # Check native dim
        test = self.model.encode(["test"])
        self.native_dim = test.shape[1]
        print(f"  Native dim: {self.native_dim}, target: {target_dim}")
        
        # PCA if needed
        self.pca = None
        self._pca_fitted = False
    
    def fit_pca(self, texts: List[str], sample_size: int = 10000):
        """Fit PCA for dimension reduction."""
        if self.native_dim == self.target_dim:
            return
        
        from sklearn.decomposition import PCA
        
        n = min(len(texts), sample_size)
        if n < self.target_dim:
            print(f"  Not enough samples for PCA ({n} < {self.target_dim}), using truncation")
            self.pca = None
            return
        
        print(f"Fitting PCA on {n} samples...")
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
        
        # L2 normalize for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.clip(norms, 1e-10, None)
        
        return embeddings.astype(np.float32)


# =============================================================================
# Database Operations
# =============================================================================

class HybridPGVector:
    """PostgreSQL operations for hybrid search."""
    
    def __init__(self, config: HybridConfig):
        self.config = config
        self.conn = None
    
    def connect(self):
        self.conn = psycopg2.connect(
            host=self.config.db_host,
            port=self.config.db_port,
            dbname=self.config.db_name,
            user=self.config.db_user,
            password=self.config.db_password
        )
        self.conn.autocommit = True
    
    def close(self):
        if self.conn:
            self.conn.close()
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, *args):
        self.close()
    
    def get_existing_paper_ids(self) -> Set[str]:
        """Get set of paper IDs already in database."""
        with self.conn.cursor() as cur:
            # Check if table exists
            cur.execute(f"""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = '{self.config.table_name}'
                )
            """)
            if not cur.fetchone()[0]:
                return set()
            
            # Get unique paper IDs
            cur.execute(f"SELECT DISTINCT paper_id FROM {self.config.table_name}")
            return {row[0] for row in cur.fetchall()}
    
    def clear_all(self):
        """Drop all document store tables."""
        with self.conn.cursor() as cur:
            # Get all tables
            cur.execute("""
                SELECT tablename FROM pg_tables 
                WHERE schemaname = 'public' 
                AND tablename NOT LIKE 'pg_%'
            """)
            tables = [row[0] for row in cur.fetchall()]
            
            print(f"Found tables: {tables}")
            
            for table in tables:
                cur.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
                print(f"  Dropped: {table}")
            
            print("All tables cleared.")
    
    def create_table(self, embedding_dim: int, vocab_size: int):
        """Create hybrid search table."""
        with self.conn.cursor() as cur:
            # Ensure extensions
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            
            # Create table
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.config.table_name} (
                    id SERIAL PRIMARY KEY,
                    chunk_id TEXT UNIQUE NOT NULL,
                    paper_id TEXT NOT NULL,
                    section_idx INTEGER NOT NULL,
                    chunk_idx INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    embedding vector({embedding_dim}) NOT NULL,
                    bm25_sparse sparsevec({vocab_size}) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            print(f"Created table: {self.config.table_name}")
    
    def create_indexes(self):
        """Create IVFFlat index for dense, no index for sparse (brute force)."""
        with self.conn.cursor() as cur:
            # Count rows for IVFFlat lists parameter
            cur.execute(f"SELECT COUNT(*) FROM {self.config.table_name}")
            count = cur.fetchone()[0]
            
            # IVFFlat lists = sqrt(n), minimum 1
            lists = max(1, int(count ** 0.5))
            
            print(f"Creating IVFFlat index with {lists} lists...")
            
            # Dense: IVFFlat with cosine
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS {self.config.table_name}_embedding_idx
                ON {self.config.table_name}
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = {lists})
            """)
            
            # Sparse: No IVFFlat support, use brute force with btree on paper_id for filtering
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS {self.config.table_name}_paper_idx
                ON {self.config.table_name} (paper_id)
            """)
            
            print("Indexes created (dense: IVFFlat, sparse: brute force).")
    
    def insert_batch(
        self,
        chunk_ids: List[str],
        paper_ids: List[str],
        section_idxs: List[int],
        chunk_idxs: List[int],
        contents: List[str],
        embeddings: np.ndarray,
        sparse_vectors: List[str],
        batch_size: int = 500
    ):
        """Insert batch of documents."""
        data = list(zip(
            chunk_ids, paper_ids, section_idxs, chunk_idxs,
            contents, 
            [emb.tolist() for emb in embeddings],
            sparse_vectors
        ))
        
        with self.conn.cursor() as cur:
            for i in tqdm(range(0, len(data), batch_size), desc="Inserting"):
                batch = data[i:i + batch_size]
                execute_values(
                    cur,
                    f"""
                    INSERT INTO {self.config.table_name} 
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
    
    def search_hybrid(
        self,
        query_embedding: np.ndarray,
        query_sparse: str,
        top_k: int = 20,
        rrf_k: int = 60,
        retrieval_limit: int = None,  # Broad recall pool size
        chunk_pool_size: int = None    # Chunks to return after RRF
    ) -> List[Dict]:
        """
        Hybrid search with Fibonacci cascade and RRF fusion.
        
        Process (Fibonacci Cascade):
        1. Retrieve top_k² chunks from dense (model2vec cosine)
        2. Retrieve top_k² chunks from sparse (BM25)
        3. RRF fusion on top_k² pools
        4. Take one Fibonacci lower (e.g., 377 from 441)
        5. K-means clustering/GIST scoring for diversity
        6. ColBERT rerank to top_k
        7. Cross-encoder rerank to one Fibonacci lower than top_k
        8. Return final results
        
        Typical flow for top_k=21:
        - retrieval_limit = 21² = 441 (broad recall)
        - RRF fusion on 441 chunks
        - Take 377 (one Fib lower) for clustering
        - K-means clustering on 377
        - ColBERT reranks to 21
        - Cross-encoder reranks to 13
        - Return 13 final results
        
        Args:
            query_embedding: Dense embedding vector (model2vec)
            query_sparse: BM25 sparse vector
            query_text: Original query text for ColBERT
            top_k: Target number for ColBERT rerank (default 20)
            rrf_k: RRF parameter (default 60)
            retrieval_limit: Chunks to retrieve (default: top_k²)
            chunk_pool_size: Chunks after RRF (default: one Fib lower than retrieval_limit)
        """
        if retrieval_limit is None:
            retrieval_limit = top_k * 3
        
        if chunk_pool_size is None:
            chunk_pool_size = retrieval_limit  # No filtering by default
        
        with self.conn.cursor() as cur:
            # Dense search (cosine distance, lower is better)
            # Already ranked by model2vec similarity!
            cur.execute(f"""
                SELECT chunk_id, content, paper_id, section_idx, chunk_idx,
                       embedding <=> %s::vector as distance
                FROM {self.config.table_name}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, (query_embedding.tolist(), query_embedding.tolist(), retrieval_limit))
            
            dense_results = {
                row[0]: {
                    "chunk_id": row[0],
                    "content": row[1],
                    "paper_id": row[2],
                    "section_idx": row[3],
                    "chunk_idx": row[4],
                    "dense_score": 1 - row[5],  # Convert distance to similarity
                    "dense_rank": i + 1
                }
                for i, row in enumerate(cur.fetchall())  # Get all retrieval_limit
            }
            
            # Sparse search (inner product, higher is better, use negative for ORDER BY)
            # Already ranked by BM25!
            cur.execute(f"""
                SELECT chunk_id, content, paper_id, section_idx, chunk_idx,
                       (bm25_sparse <#> %s::sparsevec) as neg_score
                FROM {self.config.table_name}
                ORDER BY bm25_sparse <#> %s::sparsevec
                LIMIT %s
            """, (query_sparse, query_sparse, retrieval_limit))
            
            sparse_results = {}
            for i, row in enumerate(list(cur.fetchall())):  # Get all retrieval_limit
                chunk_id = row[0]
                sparse_results[chunk_id] = {
                    "chunk_id": chunk_id,
                    "content": row[1],
                    "paper_id": row[2],
                    "section_idx": row[3],
                    "chunk_idx": row[4],
                    "sparse_score": -row[5],  # Negate back
                    "sparse_rank": i + 1
                }
        
        # Merge and compute RRF
        all_ids = set(dense_results.keys()) | set(sparse_results.keys())
        
        results = []
        for chunk_id in all_ids:
            dense = dense_results.get(chunk_id, {})
            sparse = sparse_results.get(chunk_id, {})
            
            # RRF score
            rrf_score = 0
            if "dense_rank" in dense:
                rrf_score += 1.0 / (rrf_k + dense["dense_rank"])
            if "sparse_rank" in sparse:
                rrf_score += 1.0 / (rrf_k + sparse["sparse_rank"])
            
            # Merge info
            result = {
                "chunk_id": chunk_id,
                "content": dense.get("content") or sparse.get("content"),
                "paper_id": dense.get("paper_id") or sparse.get("paper_id"),
                "section_idx": dense.get("section_idx") or sparse.get("section_idx"),
                "chunk_idx": dense.get("chunk_idx") or sparse.get("chunk_idx"),
                "dense_rank": dense.get("dense_rank"),
                "dense_score": dense.get("dense_score"),
                "sparse_rank": sparse.get("sparse_rank"),
                "sparse_score": sparse.get("sparse_score"),
                "rrf_score": rrf_score
            }
            results.append(result)
        
        # Sort by RRF score
        results = sorted(results, key=lambda x: x["rrf_score"], reverse=True)
        
        # Take one Fibonacci lower than retrieval_limit for clustering
        if chunk_pool_size < len(results):
            chunk_results = results[:chunk_pool_size]
        else:
            chunk_results = results
        
        return chunk_results
    
    def search_sections_rrf(
        self,
        query_embedding: np.ndarray,
        query_sparse: str,
        query_text: str,  # NEW: Original query for ColBERT
        top_k: int = 13,
        rrf_k: int = 60,
        chunk_pool_size: int = 377,
        retrieval_limit: int = None,
        group_by: tuple = ('paper_id', 'section_idx'),
        deduplicate: bool = True,
        similarity_threshold: float = 0.85,
        colbert_rerank: bool = True,  # NEW: Use ColBERT late interaction
        section_pool_multiplier: float = 1.6,  # NEW: Retrieve N sections before ColBERT
        use_clustering: bool = True,  # NEW: Enable k-means clustering
        n_clusters: int = None,  # NEW: Number of clusters (default: phi based on top_k)
        cluster_on: str = 'cosine',  # NEW: 'cosine', 'bm25', or 'both'
        alpha_utility: float = 0.5,  # NEW: Weight for utility score
        alpha_coverage: float = 0.5,  # NEW: Weight for coverage score
        final_k: int = None  # NEW: Final results after cross-encoder
    ) -> List[Dict]:
        """
        RRF search with k-means clustering for diversity (GIST: utility + coverage).
        
        Process:
        1. Retrieve top_k² chunks via RRF
        2. Take one Fibonacci lower for clustering
        3. **K-MEANS CLUSTERING on doc-to-doc similarity matrices**
           - Build cosine similarity matrix from embeddings
           - Build BM25 similarity matrix from inverted index
           - Cluster chunks into semantic groups
        4. **Two-factor scoring: UTILITY + COVERAGE**
           - Utility: Query relevance (RRF rank)
           - Coverage: Distance from cluster centroids (topic diversity)
        5. Group by metadata fields
        6. ColBERT rerank to top_k
        7. Cross-encoder rerank to final_k (one Fib lower than top_k)
        
        Args:
            query_embedding: Dense embedding vector
            query_sparse: BM25 query string
            query_text: Original query text (for ColBERT scoring)
            top_k: Target for ColBERT rerank (default 13)
            rrf_k: RRF parameter (default 60)
            chunk_pool_size: Chunks after RRF for clustering (default 377)
            retrieval_limit: Initial retrieval (top_k², auto-calculated if None)
            group_by: Tuple of fields to group by (default: ('paper_id', 'section_idx'))
            deduplicate: Whether to remove overlapping chunks (default True)
            similarity_threshold: Deduplication threshold (default 0.85)
            colbert_rerank: If True, use ColBERT to rerank sections (default True)
            use_clustering: If True, apply k-means clustering (default True)
            n_clusters: Number of clusters (default: phi ratio of top_k²)
            cluster_on: Similarity matrix to cluster on ('cosine', 'bm25', or 'both')
            alpha_utility: Weight for utility score (default 0.5)
            alpha_coverage: Weight for coverage score (default 0.5)
            
        Returns:
            List of groups with all their chunks, sorted by final score
        """
        # Get top chunks via RRF (large pool for accurate ranking)
        chunk_results = self.search_hybrid(
            query_embedding,
            query_sparse,
            top_k=chunk_pool_size,
            rrf_k=rrf_k
        )
        
        # === K-MEANS CLUSTERING WITH DOC-TO-DOC SIMILARITY ===
        if use_clustering and len(chunk_results) > 1:
            # K-means cross-validation: test Fibonacci values from 5 to top_k²
            if n_clusters is None:
                print(f"  K-means cross-validation (Fibonacci sequence 5 → {top_k**2})...")
                
                # Generate Fibonacci sequence from 5 up to top_k²
                fibonacci = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987]
                max_k = top_k ** 2
                valid_k = [k for k in fibonacci if 5 <= k <= max_k and k < len(chunk_results) // 2]
                
                if not valid_k:
                    # Fallback if no valid k
                    n_clusters = min(int(1.618 * top_k), len(chunk_results) // 2)
                    print(f"    No valid Fibonacci k, using phi*top_k={n_clusters}")
                else:
                    # Will compute similarity matrix first, then test k values
                    n_clusters = None  # Will be set after cross-validation
            else:
                valid_k = None  # User specified n_clusters, skip CV
                print(f"  K-means clustering {len(chunk_results)} chunks into {n_clusters} clusters (user-specified)...")
            
            # Fetch embeddings and BM25 vectors for retrieved chunks
            chunk_ids = [c['chunk_id'] for c in chunk_results]
            
            with self.conn.cursor() as cur:
                # Get embeddings (cast to text, then parse as floats)
                cur.execute(f"""
                    SELECT chunk_id, embedding::text, bm25_sparse
                    FROM {self.config.table_name}
                    WHERE chunk_id = ANY(%s)
                """, (chunk_ids,))
                
                rows = cur.fetchall()
                # Parse vector string format: "[0.1,0.2,0.3]" -> np.array
                embeddings_dict = {
                    row[0]: np.array([float(x) for x in row[1].strip('[]').split(',')])
                    for row in rows
                }
                bm25_dict = {row[0]: row[2] for row in rows}  # Sparse vectors as strings
            
            # Build similarity matrices
            embeddings_matrix = np.array([embeddings_dict[cid] for cid in chunk_ids])
            
            if cluster_on in ['cosine', 'both']:
                # Cosine similarity matrix (n × n)
                from sklearn.metrics.pairwise import cosine_similarity
                cosine_sim_matrix = cosine_similarity(embeddings_matrix)
                print(f"    Built cosine similarity matrix: {cosine_sim_matrix.shape}")
            
            if cluster_on in ['bm25', 'both']:
                # BM25 doc-to-doc similarity matrix using inverted index
                bm25_sim_matrix = self._compute_bm25_similarity_matrix(chunk_ids, bm25_dict)
                print(f"    Built BM25 similarity matrix: {bm25_sim_matrix.shape}")
            
            # Choose clustering matrix
            if cluster_on == 'cosine':
                clustering_matrix = cosine_sim_matrix
            elif cluster_on == 'bm25':
                clustering_matrix = bm25_sim_matrix
            elif cluster_on == 'both':
                # Average both matrices
                clustering_matrix = 0.5 * cosine_sim_matrix + 0.5 * bm25_sim_matrix
            
            # Convert similarity to distance for silhouette score
            distance_matrix = 1 - clustering_matrix
            np.fill_diagonal(distance_matrix, 0)  # Zero out diagonal
            
            # Import clustering tools
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score
            
            # Cross-validation to find optimal k
            if n_clusters is None and valid_k:
                
                best_k = valid_k[0]
                best_score = -1.0
                
                print(f"    Testing k values: {valid_k}")
                for k in valid_k:
                    kmeans_cv = KMeans(n_clusters=k, random_state=42, n_init=10)
                    labels_cv = kmeans_cv.fit_predict(clustering_matrix)
                    
                    # Silhouette score on distance matrix
                    score = silhouette_score(distance_matrix, labels_cv, metric='precomputed')
                    
                    if score > best_score:
                        best_score = score
                        best_k = k
                
                n_clusters = best_k
                print(f"    Best k={n_clusters} (silhouette score: {best_score:.3f})")
            
            # Final k-means clustering with optimal k
            print(f"  K-means clustering {len(chunk_results)} chunks into {n_clusters} clusters...")
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(clustering_matrix)
            
            # Compute centroid distances (coverage scores)
            from sklearn.metrics import pairwise_distances
            centroid_distances = pairwise_distances(clustering_matrix, kmeans.cluster_centers_).min(axis=1)
            
            # Normalize centroid distances to [0, 1]
            max_dist = centroid_distances.max()
            if max_dist > 0:
                coverage_scores = 1 - (centroid_distances / max_dist)  # Higher = closer to centroid
            else:
                coverage_scores = np.ones(len(chunk_results))
            
            # Compute utility scores (query relevance from RRF ranks)
            # Normalize RRF ranks to [0, 1]
            rrf_scores = np.array([c['rrf_score'] for c in chunk_results])
            max_rrf = rrf_scores.max()
            if max_rrf > 0:
                utility_scores = rrf_scores / max_rrf
            else:
                utility_scores = np.ones(len(chunk_results))
            
            # Combined score: α * utility + β * coverage
            combined_scores = alpha_utility * utility_scores + alpha_coverage * coverage_scores
            
            # Update chunk results with clustering info
            for i, chunk in enumerate(chunk_results):
                chunk['cluster_id'] = int(cluster_labels[i])
                chunk['utility_score'] = float(utility_scores[i])
                chunk['coverage_score'] = float(coverage_scores[i])
                chunk['combined_score'] = float(combined_scores[i])
                chunk['rrf_score'] = float(combined_scores[i])  # Replace RRF with combined score
            
            # Re-sort by combined score
            chunk_results.sort(key=lambda x: x['combined_score'], reverse=True)
            
            print(f"    Clustering complete. Score range: [{combined_scores.min():.3f}, {combined_scores.max():.3f}]")
        else:
            # No clustering, use original RRF scores
            for chunk in chunk_results:
                chunk['cluster_id'] = -1
                chunk['utility_score'] = chunk['rrf_score']
                chunk['coverage_score'] = 1.0
                chunk['combined_score'] = chunk['rrf_score']
        
        # Group by specified fields
        from collections import defaultdict
        groups = defaultdict(lambda: {
            **{field: None for field in group_by},
            'rrf_score': 0.0,
            'matched_chunks': []
        })
        
        for chunk in chunk_results:
            # Create grouping key from specified fields
            key = tuple(chunk[field] for field in group_by)
            
            # Populate group metadata
            for field in group_by:
                groups[key][field] = chunk[field]
            
            groups[key]['rrf_score'] += chunk['rrf_score']
            groups[key]['matched_chunks'].append({
                'chunk_idx': chunk['chunk_idx'],
                'content': chunk['content'],
                'rrf_score': chunk['rrf_score']
            })
        
        # Sort groups by RRF score
        ranked_groups = sorted(
            groups.values(),
            key=lambda x: x['rrf_score'],
            reverse=True
        )
        
        # Calculate section pool size using Fibonacci sequence
        fibonacci = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377]
        if colbert_rerank:
            # Find next higher Fibonacci for section pool
            section_pool_size = next((f for f in fibonacci if f > top_k), top_k)
        else:
            section_pool_size = top_k
        
        candidate_groups = ranked_groups[:section_pool_size]
        
        # Fetch ALL chunks for candidate groups
        groups_with_chunks = []
        with self.conn.cursor() as cur:
            for group in candidate_groups:
                # Build WHERE clause dynamically based on group_by fields
                where_conditions = " AND ".join([f"{field} = %s" for field in group_by])
                where_values = tuple(group[field] for field in group_by)
                
                cur.execute(f"""
                    SELECT chunk_idx, content
                    FROM {self.config.table_name}
                    WHERE {where_conditions}
                    ORDER BY chunk_idx
                """, where_values)
                
                all_chunks = [
                    {'chunk_idx': row[0], 'content': row[1]}
                    for row in cur.fetchall()
                ]
                
                # Deduplicate chunks if requested
                if deduplicate:
                    all_chunks = self._reduce_chunk_overlaps(all_chunks, similarity_threshold)
                
                # Combine all chunk content for ColBERT scoring
                combined_content = "\n".join([c['content'] for c in all_chunks])
                
                groups_with_chunks.append({
                    **{field: group[field] for field in group_by},
                    'rrf_score': group['rrf_score'],
                    'num_chunks': len(all_chunks),
                    'chunks': all_chunks,
                    'combined_content': combined_content
                })
        
        # ColBERT late interaction reranking
        if colbert_rerank and len(groups_with_chunks) > top_k:
            print(f"  Applying ColBERT late interaction to {len(groups_with_chunks)} sections...")
            scorer = ColBERTScorer()
            
            if scorer.available:
                # Score all candidate sections
                documents = [g['combined_content'] for g in groups_with_chunks]
                colbert_scores = scorer.score(query_text, documents)
                
                # Use ColBERT scores directly for ranking
                for i, group in enumerate(groups_with_chunks):
                    group['colbert_score'] = float(colbert_scores[i])
                    group['final_score'] = float(colbert_scores[i])
                
                # Rerank by ColBERT score and take top_k
                groups_with_chunks.sort(key=lambda x: x['final_score'], reverse=True)
                groups_with_chunks = groups_with_chunks[:top_k]
                print(f"  ColBERT reranking complete (pool: {len(groups_with_chunks) + (section_pool_size - top_k)} → top {len(groups_with_chunks)})")
            else:
                print(f"  ColBERT unavailable, using RRF scores only")
                for group in groups_with_chunks:
                    group['colbert_score'] = 0.0
                    group['final_score'] = group['rrf_score']
                groups_with_chunks = groups_with_chunks[:top_k]
        else:
            # No ColBERT, use RRF scores directly
            for group in groups_with_chunks:
                group['colbert_score'] = 0.0
                group['final_score'] = group['rrf_score']
            groups_with_chunks = groups_with_chunks[:top_k]
        
        # Cross-Encoder final reranking (top_k → final_k)
        if final_k is None:
            # Auto-calculate: one Fibonacci lower than top_k
            fibonacci = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377]
            final_k = max([f for f in fibonacci if f < top_k], default=top_k)
        
        if final_k < len(groups_with_chunks):
            print(f"  Applying Cross-Encoder to {len(groups_with_chunks)} sections...")
            cross_scorer = CrossEncoderScorer()
            
            if cross_scorer.available:
                # Score with cross-encoder
                documents = [g['combined_content'] for g in groups_with_chunks]
                cross_scores = cross_scorer.score(query_text, documents)
                
                # Use cross-encoder scores for final ranking
                for i, group in enumerate(groups_with_chunks):
                    group['cross_encoder_score'] = float(cross_scores[i])
                    group['final_score'] = float(cross_scores[i])
                
                # Rerank and take final_k
                groups_with_chunks.sort(key=lambda x: x['final_score'], reverse=True)
                groups_with_chunks = groups_with_chunks[:final_k]
                print(f"  Cross-Encoder reranking complete ({top_k} → final {len(groups_with_chunks)})")
            else:
                print(f"  Cross-Encoder unavailable, returning top {final_k} from ColBERT")
                for group in groups_with_chunks:
                    group['cross_encoder_score'] = 0.0
                groups_with_chunks = groups_with_chunks[:final_k]
        else:
            # Not enough sections for cross-encoder cascade
            for group in groups_with_chunks:
                group['cross_encoder_score'] = 0.0
        
        # Clean up combined_content field before returning
        for group in groups_with_chunks:
            group.pop('combined_content', None)
        
        return groups_with_chunks
    
    def _compute_bm25_similarity_matrix(
        self,
        chunk_ids: List[str],
        bm25_dict: Dict[str, str]
    ) -> np.ndarray:
        """
        Compute pairwise BM25 similarity matrix from sparse vectors.
        
        Uses PostgreSQL sparsevec inner product for efficient computation.
        
        Args:
            chunk_ids: List of chunk IDs
            bm25_dict: Dict mapping chunk_id to sparsevec string
            
        Returns:
            n × n similarity matrix
        """
        n = len(chunk_ids)
        sim_matrix = np.zeros((n, n))
        
        # Compute pairwise similarities using PostgreSQL
        with self.conn.cursor() as cur:
            for i in range(n):
                # Get all similarities for chunk i against all others
                query_vec = bm25_dict[chunk_ids[i]]
                
                # Use negative inner product operator (<#>) which PostgreSQL optimizes
                cur.execute(f"""
                    SELECT chunk_id, (bm25_sparse <#> %s::sparsevec) as neg_score
                    FROM {self.config.table_name}
                    WHERE chunk_id = ANY(%s)
                """, (query_vec, chunk_ids))
                
                results = {row[0]: -row[1] for row in cur.fetchall()}  # Negate back to positive
                
                for j, cid in enumerate(chunk_ids):
                    sim_matrix[i, j] = results.get(cid, 0.0)
        
        # Normalize to [0, 1]
        max_score = sim_matrix.max()
        if max_score > 0:
            sim_matrix = sim_matrix / max_score
        
        return sim_matrix
    
    def _reduce_chunk_overlaps(
        self,
        chunks: List[Dict],
        threshold: float = 0.85
    ) -> List[Dict]:
        """Remove semantically overlapping chunks using difflib.
        
        Uses SequenceMatcher to detect highly similar chunks and keeps only
        the first occurrence (preserving chunk_idx order).
        
        Args:
            chunks: List of chunk dicts with 'chunk_idx' and 'content'
            threshold: Similarity ratio threshold (0-1) for considering chunks as duplicates
            
        Returns:
            Deduplicated list of chunks
        """
        from difflib import SequenceMatcher
        
        if not chunks:
            return chunks
        
        # Sort by chunk_idx to preserve document order
        sorted_chunks = sorted(chunks, key=lambda x: x['chunk_idx'])
        
        # Track which chunks to keep
        keep = [True] * len(sorted_chunks)
        
        # Compare each chunk with subsequent chunks
        for i in range(len(sorted_chunks)):
            if not keep[i]:
                continue
                
            for j in range(i + 1, len(sorted_chunks)):
                if not keep[j]:
                    continue
                
                # Calculate similarity ratio
                similarity = SequenceMatcher(
                    None,
                    sorted_chunks[i]['content'],
                    sorted_chunks[j]['content']
                ).ratio()
                
                # If similarity exceeds threshold, mark the later chunk for removal
                if similarity >= threshold:
                    keep[j] = False
        
        # Return only chunks marked to keep
        return [chunk for i, chunk in enumerate(sorted_chunks) if keep[i]]
    
    def format_sections_as_markdown(
        self,
        sections: List[Dict]
    ) -> str:
        """Format retrieved sections as a markdown document.
        
        Args:
            sections: List of section dicts from search_sections_rrf()
            
        Returns:
            Markdown-formatted string with sections organized by arxiv_id and section_idx
        """
        md_lines = []
        md_lines.append("# Retrieved Sections\n")
        
        for section in sections:
            # Section header
            md_lines.append(f"## [{section['paper_id']}] Section {section['section_idx']}\n")
            md_lines.append(f"**RRF Score:** {section['rrf_score']:.4f}  ")
            md_lines.append(f"**Chunks:** {section['num_chunks']}\n")
            
            # Chunks content
            for chunk in section['chunks']:
                md_lines.append(f"### Chunk {chunk['chunk_idx']}\n")
                md_lines.append(f"{chunk['content']}\n")
            
            md_lines.append("---\n")  # Separator between sections
        
        return "\n".join(md_lines)


# =============================================================================
# Main Pipeline
# =============================================================================

def build_index(config: HybridConfig, reset: bool = False):
    """Build complete hybrid index from arxiv chunks.
    
    Args:
        config: Hybrid search configuration
        reset: If True, drop existing table and reload all data.
               If False (default), only process papers not in database.
    """
    from arxiv_chunking_pipeline import chunk_arxiv_papers
    import os
    
    print("=" * 70)
    print("Building Hybrid Index")
    print("=" * 70)
    
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Step 1: Chunk papers
    chunks_msgpack = checkpoint_dir / "chunks.msgpack"
    if chunks_msgpack.exists():
        print("\n[1/5] Loading cached chunks...")
        with open(chunks_msgpack, 'rb') as f:
            chunks_data = msgpack.unpackb(f.read(), raw=False, strict_map_key=False)
        # Convert dicts back to chunk objects
        from dataclasses import dataclass
        @dataclass
        class Chunk:
            doc_id: str
            section_idx: int
            chunk_idx: int
            text: str
        chunks = [Chunk(**c) for c in chunks_data]
        print(f"  Loaded {len(chunks):,} chunks from checkpoint")
    else:
        print("\n[1/5] Chunking papers...")
        papers_dir = r"C:\Users\user\arxiv_id_lists\papers\post_processed"
        chunks = chunk_arxiv_papers(papers_dir)
        print(f"  Generated {len(chunks):,} chunks")
        # Convert to dicts for msgpack
        chunks_data = [
            {'doc_id': c.doc_id, 'section_idx': c.section_idx, 
             'chunk_idx': c.chunk_idx, 'text': c.text}
            for c in chunks
        ]
        with open(chunks_msgpack, 'wb') as f:
            f.write(msgpack.packb(chunks_data, use_bin_type=True))
        print("  Saved checkpoint")
    
    # Step 1.5: Filter chunks for incremental loading
    if not reset:
        with HybridPGVector(config) as db:
            existing_papers = db.get_existing_paper_ids()
        
        if existing_papers:
            original_count = len(chunks)
            chunks = [c for c in chunks if c.doc_id not in existing_papers]
            print(f"\n[Incremental] Skipping {len(existing_papers):,} existing papers")
            print(f"  {original_count:,} chunks → {len(chunks):,} chunks ({len(chunks):,} new)")
            
            if len(chunks) == 0:
                print("\n  No new papers to process. Database is up to date.")
                print("  Use --reset to rebuild entire index.")
                return
        else:
            print("\n[Incremental] No existing data found, processing all papers...")
    else:
        print("\n[Reset Mode] Processing all papers (will drop existing table)...")
    
    # Extract data
    chunk_ids = [f"{c.doc_id}_s{c.section_idx}_c{c.chunk_idx}" for c in chunks]
    paper_ids = [c.doc_id for c in chunks]
    section_idxs = [c.section_idx for c in chunks]
    chunk_idxs = [c.chunk_idx for c in chunks]
    contents = [c.text for c in chunks]
    
    # Step 2: Build BM25 sparse vectors (sample-based vocab, no full preprocessing)
    sparse_msgpack = checkpoint_dir / "sparse_vectors.msgpack"
    bm25_cache = checkpoint_dir / "bm25_index.msgpack"
    
    if sparse_msgpack.exists() and bm25_cache.exists():
        print("\n[2/5] Loading cached BM25 vectors...")
        with open(sparse_msgpack, 'rb') as f:
            sparse_vectors = msgpack.unpackb(f.read(), raw=False, strict_map_key=False)
        bm25_builder = SparseBM25Index()
        bm25_builder.load(bm25_cache)
        print(f"  Loaded {len(sparse_vectors):,} sparse vectors from checkpoint")
    else:
        print("\n[2/4] Building BM25 index (regex tokenizer, corpus vocab)...")
        bm25_builder = SparseBM25Index(k1=1.5, b=0.75)
        bm25_builder.fit(contents)
        
        # Note: doc_index already contains sparse vectors from fit()
        # Just need to convert to sparsevec format
        print("  Converting to sparsevec format...")
        print(f"  Vocab size: {bm25_builder.vocab_size}")
        sparse_vectors = []
        max_seen_idx = -1
        for doc_id in tqdm(range(len(contents)), desc="  Vectorizing", leave=False):
            doc_vec = bm25_builder.doc_index.get(doc_id, {})
            if not doc_vec:
                sparse_vectors.append(f"{{}}/{bm25_builder.vocab_size}")
            else:
                indices = list(doc_vec.keys())
                values = list(doc_vec.values())
                
                # Validation: check if any index out of range [1, vocab_size]
                for idx in indices:
                    if idx < 1 or idx > bm25_builder.vocab_size:
                        print(f"\n  ERROR: Doc {doc_id} has token_id {idx} outside range [1, {bm25_builder.vocab_size}]")
                        print(f"  First 10 indices: {indices[:10]}")
                        raise ValueError(f"Token index {idx} out of bounds for vocab_size {bm25_builder.vocab_size}")
                    max_seen_idx = max(max_seen_idx, idx)
                
                pairs = [f"{idx}:{val:.4f}" for idx, val in zip(indices, values)]
                sparse_vectors.append(f"{{{','.join(pairs)}}}/{bm25_builder.vocab_size}")
        
        print(f"  Max token ID seen: {max_seen_idx}, Vocab size: {bm25_builder.vocab_size}")
        
        bm25_builder.save(bm25_cache)
        with open(sparse_msgpack, 'wb') as f:
            f.write(msgpack.packb(sparse_vectors, use_bin_type=True))
        print("  Saved checkpoint")
    
    # Copy to config path for search
    import shutil
    shutil.copy(bm25_cache, config.bm25_cache_path)
    
    # Step 3: Create dense embeddings
    embeddings_pkl = checkpoint_dir / "embeddings.npy"
    
    if embeddings_pkl.exists():
        print("\n[3/4] Loading cached embeddings...")
        embeddings = np.load(embeddings_pkl)
        print(f"  Loaded embeddings shape {embeddings.shape} from checkpoint")
    else:
        print("\n[3/4] Creating dense embeddings...")
        embedder = Model2VecEmbedder(target_dim=config.embedding_dim)
        embedder.fit_pca(contents)
        
        embeddings = []
        batch_size = 256
        for i in tqdm(range(0, len(contents), batch_size), desc="  Embedding"):
            batch = contents[i:i + batch_size]
            embeddings.append(embedder.encode(batch))
        embeddings = np.vstack(embeddings)
        
        np.save(embeddings_pkl, embeddings)
        print("  Saved checkpoint")
    
    # Step 4: Insert into PostgreSQL
    mode = "drop and reload" if reset else "append new data"
    print(f"\n[4/4] Inserting into PostgreSQL ({mode})...")
    with HybridPGVector(config) as db:
        if reset:
            # Drop existing table
            print(f"  Dropping table {config.table_name} if exists...")
            with db.conn.cursor() as cur:
                cur.execute(f"DROP TABLE IF EXISTS {config.table_name} CASCADE")
            print("  Table dropped.")
        
        # Create table if not exists
        print("  Creating table if not exists...")
        db.create_table(config.embedding_dim, bm25_builder.vocab_size)
        
        # Insert batch with progress
        print(f"  Inserting {len(chunks):,} chunks with embeddings + sparse vectors...")
        db.insert_batch(
            chunk_ids, paper_ids, section_idxs, chunk_idxs,
            contents, embeddings, sparse_vectors
        )
        
        # Create indexes
        print("  Creating indexes...")
        db.create_indexes()
    
    print("\n" + "=" * 70)
    print(f"Index built: {len(chunks):,} chunks")
    print(f"  Dense: vector({config.embedding_dim}) with IVFFlat cosine")
    print(f"  Sparse: sparsevec({bm25_builder.vocab_size}) with IVFFlat inner product")
    print(f"  Vocab: {bm25_builder.vocab_size:,} tokens (tokenizer)")
    print("=" * 70)


def search(
    query: str, 
    config: HybridConfig, 
    top_k: int = 13,
    chunk_pool_size: int = None,  # Auto-calculated from top_k if None
    group_by: tuple = ('paper_id', 'section_idx'),
    rrf_k: int = None,
    deduplicate: bool = True,
    similarity_threshold: float = 0.85,
    return_sections: bool = False
):
    """Search the hybrid index with Fibonacci cascade and return aggregated groups.
    
    Fibonacci Cascade Flow:
    1. Calculate retrieval_limit = top_k² (e.g., 13² = 169)
    2. Calculate rerank_limit = highest_fib ≤ retrieval_limit (e.g., 144)
    3. Retrieve retrieval_limit from dense (model2vec) and sparse (BM25)
    4. GIST: Filter to top rerank_limit from each pool (removes noise)
    5. RRF fusion on cleaned rerank_limit pools
    6. Group by metadata fields, select top_k groups
    
    Args:
        query: Search query string
        config: Hybrid configuration
        top_k: Number of groups to return (default 13)
        chunk_pool_size: Override auto-calculated cascade (not recommended)
        group_by: Tuple of fields to aggregate by (default: ('paper_id', 'section_idx'))
                 Options: ('paper_id',) - entire papers
                          ('paper_id', 'section_idx') - sections within papers
        rrf_k: RRF parameter (uses config.rrf_k if None)
        deduplicate: Whether to deduplicate chunks (default True)
        similarity_threshold: Similarity threshold for dedup (default 0.85)
        return_sections: If True, return sections without printing (default False)
        
    Returns:
        List of groups with chunks
    """
    if not return_sections:
        print(f"Searching: {query}")
        print("=" * 70)
    
    # Use config rrf_k if not specified
    if rrf_k is None:
        rrf_k = config.rrf_k
    
    # Fibonacci cascade calculation
    fibonacci = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597]
    
    if chunk_pool_size is None:
        # Auto-calculate cascade
        retrieval_limit = top_k * top_k  # Retrieve top_k²
        
        # Find one Fibonacci lower than retrieval_limit
        fib_lower = [f for f in fibonacci if f < retrieval_limit]
        chunk_pool_size = fib_lower[-1] if fib_lower else retrieval_limit
        
        # ColBERT reranks to top_k
        colbert_limit = top_k
        
        # Cross-encoder reranks to one Fibonacci lower than top_k
        fib_lower_topk = [f for f in fibonacci if f < top_k]
        final_k = fib_lower_topk[-1] if fib_lower_topk else top_k
        
        if not return_sections:
            print(f"Fibonacci Cascade: retrieve {retrieval_limit} → RRF → {chunk_pool_size} (GIST) → ColBERT {colbert_limit} → Cross-Encoder {final_k}")
    else:
        # Manual override
        retrieval_limit = top_k * top_k
        colbert_limit = top_k
        
        # Find one Fibonacci lower than top_k for final results
        fib_lower_topk = [f for f in fibonacci if f < top_k]
        final_k = fib_lower_topk[-1] if fib_lower_topk else top_k
        
        if not return_sections:
            print(f"Manual Cascade: retrieve {retrieval_limit} → RRF → {chunk_pool_size} (GIST) → ColBERT {colbert_limit} → Cross-Encoder {final_k}")
    
    # Load components
    bm25_builder = SparseBM25Index()
    bm25_builder.load(config.bm25_cache_path)
    
    # Create preprocessor with loaded vocab
    preprocessor = TextPreprocessor(bm25_builder.vocab)
    embedder = Model2VecEmbedder(target_dim=config.embedding_dim)
    
    # Process query
    query_token_ids = preprocessor.preprocess(query)
    query_sparse = bm25_builder.query_to_sparse(query_token_ids)
    query_embedding = embedder.encode([query])[0]
    
    # Search for sections
    with HybridPGVector(config) as db:
        sections = db.search_sections_rrf(
            query_embedding=query_embedding,
            query_sparse=query_sparse,
            query_text=query,  # Pass original query for ColBERT
            top_k=colbert_limit,  # ColBERT reranks to this
            rrf_k=rrf_k,
            chunk_pool_size=chunk_pool_size,  # Post-RRF chunks for clustering
            retrieval_limit=retrieval_limit,  # Initial retrieval (top_k²)
            group_by=group_by,
            deduplicate=deduplicate,
            similarity_threshold=similarity_threshold,
            colbert_rerank=True,  # Enable ColBERT late interaction
            use_clustering=True,  # Enable k-means clustering
            cluster_on='cosine',  # Cluster on cosine similarity matrix
            alpha_utility=0.5,  # Weight for query relevance
            alpha_coverage=0.5,  # Weight for topic diversity
            final_k=final_k  # Cross-encoder returns this many
        )
    
    # Display (unless caller wants raw sections)
    if not return_sections:
        print(f"\nTop {len(sections)} Groups (by {group_by}):\n")
        for i, section in enumerate(sections, 1):
            # Build display label from group_by fields
            group_label = ", ".join([f"{field}={section[field]}" for field in group_by])
            print(f"{i}. {group_label}")
            print(f"   RRF Score: {section['rrf_score']:.4f}")
            print(f"   Chunks: {section['num_chunks']}")
            if section['chunks']:
                print(f"   Preview: {section['chunks'][0]['content'][:150]}...")
            print()
    
    return sections


def clear_all(config: HybridConfig):
    """Clear all tables."""
    with HybridPGVector(config) as db:
        db.clear_all()


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import sys
    import argparse
    
    config = HybridConfig()
    
    parser = argparse.ArgumentParser(description="Hybrid Retrieval with ColBERT late interaction")
    parser.add_argument("--build", action="store_true", help="Build index from chunks")
    parser.add_argument("--search", type=str, metavar="QUERY", help="Search query text")
    parser.add_argument("--clear", action="store_true", help="Clear all tables")
    parser.add_argument("--reset", action="store_true", help="Drop existing table and rebuild (for --build)")
    parser.add_argument("--top_k", type=int, default=13, help="Number of final sections to return (default: 13)")
    parser.add_argument("--save", type=str, metavar="FILE", help="Save results to markdown file")
    
    args = parser.parse_args()
    
    if args.build:
        build_index(config, reset=args.reset)
    elif args.search:
        sections = search(args.search, config, top_k=args.top_k)
        
        if args.save:
            # Save results to markdown file
            with open(args.save, 'w', encoding='utf-8') as f:
                f.write(f"# Search Results: {args.search}\n\n")
                f.write(f"**Top {len(sections)} sections (top_k={args.top_k})**\n\n")
                
                for i, section in enumerate(sections, 1):
                    group_label = f"paper_id={section.get('paper_id', 'N/A')}, section_idx={section.get('section_idx', 'N/A')}"
                    f.write(f"## {i}. {group_label}\n\n")
                    f.write(f"- **RRF Score**: {section['rrf_score']:.4f}\n")
                    f.write(f"- **ColBERT Score**: {section.get('colbert_score', 0.0):.4f}\n")
                    f.write(f"- **Final Score**: {section.get('final_score', section['rrf_score']):.4f}\n")
                    f.write(f"- **Chunks**: {section['num_chunks']}\n\n")
                    
                    f.write("### Content\n\n")
                    for chunk in section['chunks']:
                        f.write(f"{chunk['content']}\n\n")
                    f.write("---\n\n")
            
            print(f"\nResults saved to: {args.save}")
    elif args.clear:
        confirm = input("This will delete ALL tables. Type 'yes' to confirm: ")
        if confirm.lower() == "yes":
            clear_all(config)
        else:
            print("Cancelled.")
    else:
        parser.print_help()
