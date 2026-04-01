"""
PGVector Retriever Module

Query 4 pgvector collections:
1. Layer 1 embeddings (128d HNSW)
2. Layer 1 BM25 sparse (sparsevec 16k, signed hashing, <#> inner product)
3. Layer 2 triplet BM25 (sparsevec 16k, signed hashing, <#> inner product)
4. Layer 2 embeddings (256d HNSW)

BM25: N_BUCKETS=16_000 signed hashing (Weinberger et al. 2009).
No vocab table, no OOV. hash(term) % 16_000 IS the vocabulary.
Stat tables populated by _migrate_bm25_sparsevec.py.
"""

import math
import mmh3
import psycopg2
import json
import numpy as np
from collections import defaultdict
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from pgvector.psycopg2 import register_vector, SparseVector


# ---------------------------------------------------------------------------
# BM25 Signed Hashing — N_BUCKETS=16_000 matches sparsevec(16000) schema
# ---------------------------------------------------------------------------

N_BUCKETS = 16_000


def _hash_token(term: str) -> tuple:
    """
    Returns (bucket, sign). Seed 0=bucket assignment, Seed 1=sign.
    Colliding terms with opposite signs cancel rather than compound
    (unbiased inner product estimator, Weinberger et al. 2009).
    """
    bucket = mmh3.hash(term, seed=0, signed=False) % N_BUCKETS
    sign   = 1.0 if mmh3.hash(term, seed=1, signed=True) >= 0 else -1.0
    return bucket, sign


def _tokenize_bm25(text: str) -> List[str]:
    """Lowercase + whitespace split. Must be identical at index and query time."""
    return text.lower().split()


@dataclass
class PGVectorConfig:
    """PostgreSQL configuration."""
    host: str = 'localhost'
    port: int = 5432
    database: str = 'langchain'
    user: str = 'langchain'
    password: str = 'langchain'


class PGVectorRetriever:
    """Query pgvector collections for ArXiv chunks."""
    
    def __init__(self, config: Optional[PGVectorConfig] = None):
        """
        Initialize retriever with database connection.
        
        Args:
            config: PostgreSQL configuration (uses defaults if None)
        """
        self.config = config or PGVectorConfig()
        self.conn = None
        self.cur = None
        self._connect()
    
    def _connect(self):
        """Establish PostgreSQL connection."""
        try:
            self.conn = psycopg2.connect(
                host=self.config.host,
                port=self.config.port,
                database=self.config.database,
                user=self.config.user,
                password=self.config.password
            )
            self.cur = self.conn.cursor()
            register_vector(self.conn)
            print(f"  [OK] Connected to PostgreSQL {self.config.host}:{self.config.port}/{self.config.database}")
        except Exception as e:
            print(f"  [FAIL] Connection failed: {e}")
            raise
    
    def close(self):
        """Close database connection."""
        if self.cur:
            self.cur.close()
        if self.conn:
            self.conn.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # ========================================================================
    # BM25 SIGNED HASHING HELPERS (sparsevec 16k)
    # ========================================================================

    def _bm25_effective_stats(self, stat_table: str = 'bm25_global_stats') -> Dict:
        """Read corpus stats from stat_table (ANOVA partition model)."""
        try:
            self.cur.execute(
                f"SELECT n_docs, total_tokens FROM {stat_table} WHERE id = 1"
            )
            row = self.cur.fetchone()
            if row:
                n, tok = row
                n = max(n, 1)
                return {'n': n, 'avgdl': max(tok, 1) / n}
        except Exception:
            self.conn.rollback()
        return {'n': 1, 'avgdl': 256}

    def _bm25_batch_df(
        self, buckets: List[int], term_table: str = 'bm25_term_global'
    ) -> Dict[int, int]:
        """Fetch doc_freq for buckets from term_table (one round-trip)."""
        if not buckets:
            return {}
        try:
            self.cur.execute(
                f"SELECT bucket, doc_freq FROM {term_table} WHERE bucket = ANY(%s)",
                (buckets,)
            )
            return {row[0]: row[1] for row in self.cur.fetchall()}
        except Exception:
            self.conn.rollback()
            return {}

    def _bm25_vectorize_query(
        self,
        text: str,
        stat_table: str = 'bm25_global_stats',
        term_table: str = 'bm25_term_global',
    ) -> SparseVector:
        """
        IDF-only query vectorization using signed hashing.
        No TF saturation or length norm — queries are short.
        """
        tokens    = _tokenize_bm25(text)
        stats     = self._bm25_effective_stats(stat_table)
        n         = stats['n']
        bucket_tf: Dict[int, float] = defaultdict(float)
        for term in tokens:
            bucket, sign = _hash_token(term)
            bucket_tf[bucket] += sign
        df_map          = self._bm25_batch_df(list(bucket_tf.keys()), term_table)
        indices, values = [], []
        for bucket, signed_tf in bucket_tf.items():
            df  = df_map.get(bucket, 0)
            idf = math.log((n - df + 0.5) / (df + 0.5) + 1)
            if idf > 0:
                indices.append(bucket)
                values.append(math.copysign(idf, signed_tf))
        return SparseVector(dict(zip(indices, values)), N_BUCKETS)

    def _bm25_vectorize_weighted(
        self,
        term_weights: Dict[str, float],
        stat_table: str = 'bm25_global_stats',
        term_table: str = 'bm25_term_global',
    ) -> SparseVector:
        """
        Weighted IDF vectorization for expansion queries.
        term_weights: {term: weight} — ECDF-weighted TF profile from seed docs.
        Signs from hashing are preserved; weights scale the IDF contribution.
        """
        stats = self._bm25_effective_stats(stat_table)
        n     = stats['n']
        bucket_acc: Dict[int, float] = defaultdict(float)
        for term, weight in term_weights.items():
            bucket, sign = _hash_token(term)
            bucket_acc[bucket] += sign * abs(weight)
        df_map          = self._bm25_batch_df(list(bucket_acc.keys()), term_table)
        indices, values = [], []
        for bucket, signed_w in bucket_acc.items():
            df    = df_map.get(bucket, 0)
            idf   = math.log((n - df + 0.5) / (df + 0.5) + 1)
            score = idf * abs(signed_w)
            if score > 0:
                indices.append(bucket)
                values.append(math.copysign(score, signed_w))
        return SparseVector(dict(zip(indices, values)), N_BUCKETS)

    # ========================================================================
    # UTILITY: Get chunk text and embeddings for seeds
    # ========================================================================
    
    def get_chunk_texts(self, chunk_ids: List[str]) -> List[str]:
        """
        Get chunk text for given chunk IDs.
        
        Args:
            chunk_ids: List of chunk_id strings
        
        Returns:
            List of chunk texts (same order as input)
        """
        if not chunk_ids:
            return []
        
        placeholders = ','.join(['%s'] * len(chunk_ids))
        query = f"""
        SELECT chunk_id, text
        FROM arxiv_chunks
        WHERE chunk_id IN ({placeholders})
        """
        
        self.cur.execute(query, chunk_ids)
        # Build dict to preserve order
        text_map = {row[0]: row[1] for row in self.cur.fetchall()}
        return [text_map.get(cid, '') for cid in chunk_ids]
    
    def get_layer2_embeddings(self, chunk_ids: List[str]) -> np.ndarray:
        """
        Get Layer 2 embeddings (256d) for given chunk IDs.
        
        Args:
            chunk_ids: List of chunk_id strings
        
        Returns:
            numpy array of shape [n_chunks, 256]
        """
        if not chunk_ids:
            return np.empty((0, 256), dtype=np.float32)
        
        placeholders = ','.join(['%s'] * len(chunk_ids))
        query = f"""
        SELECT chunk_id, embedding
        FROM layer2_embeddings_256d
        WHERE chunk_id IN ({placeholders})
        """
        
        self.cur.execute(query, chunk_ids)
        # Build dict to preserve order
        emb_map = {}
        for row in self.cur.fetchall():
            chunk_id = row[0]
            emb_list = row[1]  # pgvector returns as list
            emb_map[chunk_id] = np.array(emb_list, dtype=np.float32)
        
        # Return in order (use zeros for missing)
        embeddings = []
        for cid in chunk_ids:
            if cid in emb_map:
                embeddings.append(emb_map[cid])
            else:
                embeddings.append(np.zeros(256, dtype=np.float32))
        
        return np.array(embeddings)
    
    # ========================================================================
    # LAYER 1: EMBEDDINGS (128d HNSW)
    # ========================================================================
    
    def query_layer1_embeddings_128d(
        self,
        query_embedding: np.ndarray,
        top_k: int = 100,
        ef_search: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Query Layer 1 embeddings (128d) using HNSW index.
        
        Args:
            query_embedding: Query vector (128d)
            top_k: Number of results to return
            ef_search: HNSW search parameter (higher = more accurate but slower)
        
        Returns:
            List of {'chunk_id': str, 'score': float, 'distance': float}
        """
        # Set HNSW search parameter
        self.cur.execute(f"SET hnsw.ef_search = {ef_search};")
        
        # Build query embedding as PostgreSQL array
        emb_str = '[' + ','.join(map(str, query_embedding)) + ']'
        
        # Query: cosine distance (1 - cosine similarity)
        query = f"""
        SELECT chunk_id, 1 - (embedding <=> %s::vector) AS score
        FROM layer1_embeddings_128d
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """
        
        self.cur.execute(query, (emb_str, emb_str, top_k))
        results = []
        
        for row in self.cur.fetchall():
            chunk_id, score = row
            results.append({
                'chunk_id': chunk_id,
                'score': float(score),
                'distance': float(1.0 - score)
            })
        
        return results
    
    # ========================================================================
    # LAYER 1: BM25 SPARSE (sparsevec 16k — <#> inner product)
    # ========================================================================

    def query_layer1_bm25(
        self,
        query_text: str,
        top_k: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Query Layer 1 BM25 via signed-hash sparsevec <#> inner product.

        Requires: arxiv_chunks.bm25_sparse sparsevec(16000), populated by
        _migrate_bm25_sparsevec.py. Uses bm25_global_stats / bm25_term_global.

        <#> returns negative inner product — ORDER BY ASC = highest similarity first.

        Args:
            query_text: Raw query string (tokenized internally via _tokenize_bm25)
            top_k: Number of results
        Returns:
            List of {'chunk_id': str, 'score': float}
        """
        q_sv = self._bm25_vectorize_query(
            query_text,
            stat_table='bm25_global_stats',
            term_table='bm25_term_global',
        )
        self.cur.execute("""
            SELECT chunk_id, -(bm25_sparse <#> %s) AS score
            FROM arxiv_chunks
            ORDER BY bm25_sparse <#> %s ASC
            LIMIT %s
        """, (q_sv, q_sv, top_k))
        return [
            {'chunk_id': row[0], 'score': float(row[1])}
            for row in self.cur.fetchall()
        ]
    
    # ========================================================================
    # LAYER 2: CONCRETE EXPANSION IMPLEMENTATION (pgvector backend)
    # ========================================================================
    
    def _expand_layer2_bm25(
        self,
        seed_docs: List['RetrievedDoc'],
        seed_scores: List[float],
        top_k: int
    ) -> List['RetrievedDoc']:
        """
        Expand via Layer 2 BM25 over triplet corpus (pgvector backend).
        
        Uses ECDF-weighted mean TF profile from seed chunks.
        Queries pgvector layer2_triplet_bm25 table with JSONB sparse vectors.
        
        Args:
            seed_docs: Layer 1 seed chunks
            seed_scores: Layer 1 BM25 scores for ECDF weighting
            top_k: Target expansion count
        
        Returns:
            List of RetrievedDoc from Layer 2 expansion (seeds excluded)
        """
        from collections import Counter
        import numpy as np
        
        # Import RetrievedDoc if not already in scope
        try:
            from gist_retriever import RetrievedDoc, GISTRetriever
        except:
            pass  # Assume it's available
        
        # 1. Compute ECDF weights
        ecdf_w = GISTRetriever._midpoint_ecdf_weights(np.array(seed_scores, dtype=np.float64))
        
        # 2. Get chunk texts for seeds
        seed_ids = [doc.doc_id for doc in seed_docs]
        seed_texts = self.get_chunk_texts(seed_ids)
        
        # 3. Build weighted TF profile
        seed_tfs = []
        valid_weights = []
        
        for idx, text in enumerate(seed_texts):
            if text.strip():
                # Simple tokenization (split on whitespace + lowercasing)
                tokens = text.lower().split()
                tf = Counter(tokens)
                if tf:
                    seed_tfs.append(tf)
                    valid_weights.append(ecdf_w[idx])
        
        if not seed_tfs:
            return []
        
        # 4. Weighted mean TF profile
        vocab = sorted(set().union(*seed_tfs))
        vocab_idx = {term: i for i, term in enumerate(vocab)}
        
        tf_matrix = np.zeros((len(seed_tfs), len(vocab)), dtype=np.float64)
        for i, tf in enumerate(seed_tfs):
            for term, count in tf.items():
                tf_matrix[i, vocab_idx[term]] = count
        
        w = np.array(valid_weights, dtype=np.float64)
        weighted_tf = np.average(tf_matrix, axis=0, weights=w)
        
        # 5. Build weighted term profile {term: weight}
        term_weights = {}
        for j, term in enumerate(vocab):
            weight = weighted_tf[j]
            if weight > 0:
                term_weights[term] = float(weight)

        if not term_weights:
            return []

        # 6. Vectorize with signed hashing -> <#> query against layer2_triplet_bm25
        q_sv = self._bm25_vectorize_weighted(
            term_weights,
            stat_table='bm25_l2_stats',
            term_table='bm25_l2_term_df',
        )
        self.cur.execute("""
            SELECT chunk_id, -(triplet_bm25_vector <#> %s) AS score
            FROM layer2_triplet_bm25
            ORDER BY triplet_bm25_vector <#> %s ASC
            LIMIT %s
        """, (q_sv, q_sv, top_k * 2))
        results = [
            {'chunk_id': row[0], 'score': float(row[1])}
            for row in self.cur.fetchall()
        ]

        # 7. Exclude seeds
        seed_set = set(seed_ids)
        filtered_results = [r for r in results if r['chunk_id'] not in seed_set]
        
        # 8. GIST diversify (NYI - return top scores for now)
        filtered_results = filtered_results[:top_k]
        
        # 9. Convert to RetrievedDoc
        retrieved_docs = []
        for r in filtered_results:
            retrieved_docs.append(RetrievedDoc(
                doc_id=r['chunk_id'],
                content="",  # Fetch later if needed
                metadata={'source': 'layer2_triplet_bm25'},
                bm25_score=r['score'],
                gist_rank=len(retrieved_docs) + 1
            ))
        
        return retrieved_docs
    
    def _expand_layer2_dense(
        self,
        seed_docs: List['RetrievedDoc'],
        seed_scores: List[float],
        top_k: int
    ) -> List['RetrievedDoc']:
        """
        Expand via Layer 2 dense embeddings (256d pgvector backend).
        
        Uses ECDF-weighted centroid from seed embeddings.
        Queries pgvector layer2_embeddings_256d table with HNSW index.
        
        Args:
            seed_docs: Layer 1 seed chunks
            seed_scores: Layer 1 BM25 scores for ECDF weighting
            top_k: Target expansion count
        
        Returns:
            List of RetrievedDoc from Layer 2 expansion (seeds excluded)
        """
        import numpy as np
        
        # Import dependencies
        try:
            from gist_retriever import RetrievedDoc, GISTRetriever
        except:
            pass
        
        # 1. Compute ECDF weights
        ecdf_w = GISTRetriever._midpoint_ecdf_weights(np.array(seed_scores, dtype=np.float64))
        
        # 2. Get embeddings for seed chunks
        seed_ids = [doc.doc_id for doc in seed_docs]
        seed_embeddings = self.get_layer2_embeddings(seed_ids)
        
        if seed_embeddings.shape[0] == 0:
            return []
        
        # 3. Weighted centroid
        query_embedding = np.average(seed_embeddings, axis=0, weights=ecdf_w)
        
        # 4. Query Layer 2 embeddings (pgvector HNSW)
        results = self.query_layer2_embeddings_256d(query_embedding, top_k * 2)
        
        # 5. Exclude seeds
        seed_set = set(seed_ids)
        filtered_results = [r for r in results if r['chunk_id'] not in seed_set]
        
        # 6. GIST diversify (NYI - return top scores for now)
        filtered_results = filtered_results[:top_k]
        
        # 7. Convert to RetrievedDoc
        retrieved_docs = []
        for r in filtered_results:
            retrieved_docs.append(RetrievedDoc(
                doc_id=r['chunk_id'],
                content="",  # Fetch later
                metadata={'source': 'layer2_embeddings_256d'},
                dense_score=r['score'],
                gist_rank=len(retrieved_docs) + 1
            ))
        
        return retrieved_docs
    
    # ========================================================================
    # LAYER 2: TRIPLET BM25 (sparsevec 16k — <#> inner product)
    # ========================================================================

    def query_layer2_triplet_bm25(
        self,
        query_text: str,
        top_k: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Query Layer 2 triplet BM25 via signed-hash sparsevec <#> inner product.

        Requires: layer2_triplet_bm25.triplet_bm25_vector sparsevec(16000), populated
        by _migrate_bm25_sparsevec.py. Uses bm25_l2_stats / bm25_l2_term_df.

        Args:
            query_text: Raw query string (tokenized internally via _tokenize_bm25)
            top_k: Number of results
        Returns:
            List of {'chunk_id': str, 'score': float}
        """
        q_sv = self._bm25_vectorize_query(
            query_text,
            stat_table='bm25_l2_stats',
            term_table='bm25_l2_term_df',
        )
        self.cur.execute("""
            SELECT chunk_id, -(triplet_bm25_vector <#> %s) AS score
            FROM layer2_triplet_bm25
            ORDER BY triplet_bm25_vector <#> %s ASC
            LIMIT %s
        """, (q_sv, q_sv, top_k))
        return [
            {'chunk_id': row[0], 'score': float(row[1])}
            for row in self.cur.fetchall()
        ]
    
    # ========================================================================
    # LAYER 2: EMBEDDINGS (256d HNSW)
    # ========================================================================
    
    def query_layer2_embeddings_256d(
        self,
        query_embedding: np.ndarray,
        top_k: int = 100,
        ef_search: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Query Layer 2 embeddings (256d) using HNSW index.
        
        Args:
            query_embedding: Query vector (256d)
            top_k: Number of results to return
            ef_search: HNSW search parameter
        
        Returns:
            List of {'chunk_id': str, 'score': float, 'distance': float}
        """
        # Set HNSW search parameter
        self.cur.execute(f"SET hnsw.ef_search = {ef_search};")
        
        # Build query embedding as PostgreSQL array
        emb_str = '[' + ','.join(map(str, query_embedding)) + ']'
        
        # Query: cosine distance
        query = f"""
        SELECT chunk_id, 1 - (embedding <=> %s::vector) AS score
        FROM arxiv_chunks
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """
        
        self.cur.execute(query, (emb_str, emb_str, top_k))
        results = []
        
        for row in self.cur.fetchall():
            chunk_id, score = row
            results.append({
                'chunk_id': chunk_id,
                'score': float(score),
                'distance': float(1.0 - score)
            })
        
        return results


def test_pgvector_retriever():
    """Unit test for pgvector retriever (requires populated database)."""
    print("\n" + "="*60)
    print("TESTING PGVECTOR RETRIEVER")
    print("="*60)
    
    try:
        with PGVectorRetriever() as retriever:
            print("\n✓ Connection successful")
            
            # Test 1: Check table existence
            print("\nChecking tables...")
            tables = ['arxiv_chunks', 'layer1_embeddings_128d',
                      'layer2_triplet_bm25', 'bm25_global_stats', 'bm25_term_global']
            
            for table in tables:
                retriever.cur.execute(f"SELECT COUNT(*) FROM {table}")
                count = retriever.cur.fetchone()[0]
                print(f"  {table}: {count:,} rows")
            
            # Test 2: Layer 1 embeddings (mock query)
            print("\nTest Layer 1 embeddings query...")
            query_emb_128 = np.random.rand(128).astype(np.float32)
            results = retriever.query_layer1_embeddings_128d(query_emb_128, top_k=5)
            print(f"  ✓ Returned {len(results)} results")
            if results:
                print(f"  Top result: {results[0]['chunk_id']}, score={results[0]['score']:.4f}")
            
            # Test 3: Layer 1 BM25 (signed hash sparsevec)
            print("\nTest Layer 1 BM25 signed hash query...")
            results = retriever.query_layer1_bm25("contrastive learning self-supervised", top_k=5)
            print(f"  ✓ Returned {len(results)} results")
            if results:
                print(f"  Top result: {results[0]['chunk_id']}, score={results[0]['score']:.4f}")

            # Test 4: Layer 2 triplet BM25 (signed hash sparsevec)
            print("\nTest Layer 2 triplet BM25 query...")
            results = retriever.query_layer2_triplet_bm25("contrastive learning self-supervised", top_k=5)
            print(f"  ✓ Returned {len(results)} results")
            if results:
                print(f"  Top result: {results[0]['chunk_id']}, score={results[0]['score']:.4f}")
            
            # Test 5: Layer 2 embeddings (mock query)
            print("\nTest Layer 2 embeddings query...")
            query_emb_256 = np.random.rand(256).astype(np.float32)
            results = retriever.query_layer2_embeddings_256d(query_emb_256, top_k=5)
            print(f"  ✓ Returned {len(results)} results")
            if results:
                print(f"  Top result: {results[0]['chunk_id']}, score={results[0]['score']:.4f}")
            
            print("\n✓ All tests passed")
    
    except psycopg2.OperationalError as e:
        print(f"\n✗ Database connection failed: {e}")
        print("  (This is expected if ingestion hasn't completed yet)")
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    test_pgvector_retriever()
