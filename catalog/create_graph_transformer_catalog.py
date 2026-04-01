"""
Create graph_transformer_feature_catalog.sqlite3 and populate it with features
extracted from GRAPH_TRANSFORMER_STRATEGY.md.

Organizes features by subsystem:
- bio_tagger_features.sqlite3 → BIO tagger pipeline
- retriever_feature_catalog.sqlite3 → Hybrid retriever (BM25 + dense)
- graph_transformer_feature_catalog.sqlite3 → Graph transformer reranking
"""

import sqlite3
from datetime import datetime


def create_graph_transformer_catalog():
    """Create and populate graph transformer feature catalog."""
    
    db_path = "graph_transformer_feature_catalog.sqlite3"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create tables (same schema as other catalogs)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS features (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL UNIQUE,
        description TEXT,
        status TEXT CHECK(status IN ('TODO', 'IN_PROGRESS', 'VALIDATING', 'DONE', 'FAILED')) DEFAULT 'TODO',
        f1_baseline REAL,
        f1_current REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        validated_by TEXT,
        validation_notes TEXT
    )
    """)
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS claims (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        feature_id INTEGER,
        claim_text TEXT NOT NULL,
        predicted_f1 REAL,
        predicted_improvement REAL,
        confidence_score REAL CHECK(confidence_score >= 0 AND confidence_score <= 1),
        actual_f1 REAL,
        actual_improvement REAL,
        prediction_error REAL,
        claim_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        validated_timestamp TIMESTAMP,
        validation_result TEXT CHECK(validation_result IN ('PENDING', 'CONFIRMED', 'FAILED', 'PARTIAL')),
        FOREIGN KEY (feature_id) REFERENCES features(id)
    )
    """)
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS architectural_decisions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        decision TEXT NOT NULL,
        rationale TEXT,
        before_state TEXT,
        after_state TEXT,
        before_f1 REAL,
        after_f1 REAL,
        measured_impact REAL,
        decision_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        validated_timestamp TIMESTAMP
    )
    """)
    
    conn.commit()
    
    # Add workflow preamble
    workflow = """Workflow Execution Order for Graph Transformer Reranking:

**Context:** Graph transformer reranking is applied AFTER hybrid retrieval to improve precision on candidate chunks. It does NOT operate on the full corpus (too expensive).

1. **Run Hybrid Retrieval**
   - Scripts: query_arxiv.py, query_quotes.py
   - Purpose: 3-layer φ-scaled retrieval (L1 BM25+Dense seeds → L2 ECDF expansion → L3 ColBERT+CrossEncoder)
   - Command: python query_arxiv.py "your query here"
   - Output: ~50-100 candidate chunks with relevance scores

2. **Extract Triplets from Candidates (Optional for baseline)**
   - Script: inference_bio_tagger.py
   - Purpose: Run BIO tagger on candidate chunks only (~25 seconds for 50 chunks vs 11 hours for full corpus)
   - Import: from inference_bio_tagger import BIOTripletExtractor
   - Output: SPO triplets for graph construction

3. **Phase 1: Train Graph Distillation Model (One-time, ~2-3 days)**
   - Script: train_graph_distillation.py (to be created)
   - Purpose: Train lightweight model to predict graph embeddings from text
   - Steps:
     a. Load 250 training chunks from bio_training_250chunks_complete_FIXED.msgpack
     b. Run BIO tagger → extract triplets → build AOKG
     c. Run karateclub.Node2Vec on AOKG → derive node embeddings (128-dim)
     d. Aggregate node embeddings per chunk (mean pooling)
     e. Train DistilBERT + projection layer: text → graph_embedding
     f. MSE loss: predicted vs actual node embeddings
   - Output: graph_distillation_model.pt (fast embedding model)
   - Estimate: ~4-6 hours training time

4. **Phase 2: Derive Graph Embeddings for Full Corpus (Fast, ~18 minutes)**
   - Script: apply_graph_embeddings.py (to be created)
   - Purpose: Apply distilled model to derive graph embeddings for all 161k chunks
   - Command: python apply_graph_embeddings.py --model graph_distillation_model.pt --chunks checkpoints/chunks.msgpack
   - Throughput: ~150 chunks/sec (vs 4.08 chunks/sec with BERT)
   - Output: graph_embeddings.msgpack OR pgvector table with graph embeddings

5. **Phase 3: Build HNSW Index on Graph Embeddings**
   - Script: build_graph_hnsw.py (to be created)
   - Purpose: Create HNSW index for fast graph-based retrieval
   - Command: python build_graph_hnsw.py --embeddings graph_embeddings.msgpack
   - Output: HNSW index in pgvector OR standalone index

6. **Phase 4: Rerank with Graph Embeddings**
   - Script: graph_reranker.py (to be created)
   - Purpose: Combine BM25 + dense + graph signals for final ranking
   - Import: from graph_reranker import GraphReranker
   - Fusion strategies:
     a. Linear combination with learned weights
     b. RRF (Reciprocal Rank Fusion) across 3 signals
     c. Cascade: BM25 → dense → graph
   - Output: Reranked results with improved precision

**Quick Start (after implementation):**
# One-time training (Phase 1)
python train_graph_distillation.py --chunks bio_training_250chunks_complete_FIXED.msgpack --output graph_distillation_model.pt

# One-time full corpus embedding (Phase 2)
python apply_graph_embeddings.py --model graph_distillation_model.pt --chunks checkpoints/chunks.msgpack

# Query with graph reranking (Phase 4)
python query_arxiv.py "machine learning transformers" --rerank graph

**Performance:**
- BIO Tagger (BERT): 4.08 chunks/sec → 11 hours for 161k chunks
- Distilled Model: ~150 chunks/sec → 18 minutes for 161k chunks
- Speedup: 37x faster, 97% time reduction
- Expected improvement: Precision@5 boost via graph semantic distance"""

    cursor.execute("""
    INSERT INTO features (name, description, status, created_at, updated_at)
    VALUES (?, ?, 'DONE', ?, ?)
    """, ("Workflow: Graph Transformer Execution Order", workflow, datetime.now(), datetime.now()))
    
    # Add features from GRAPH_TRANSFORMER_STRATEGY.md
    
    # Feature 1: Node2Vec Graph Embeddings
    feature1 = """Extract graph embeddings from AOKG using karateclub.Node2Vec.

**Purpose:** Derive 128-dim embeddings that capture AOKG topology for the 250 training chunks.

**Implementation:**
- Library: karateclub (Node2Vec class)
- Input: AOKG networkx graph from 250 training chunks
- Parameters: dimensions=128, walk_length=80, walk_number=10
- Output: Dict mapping node_id → 128-dim embedding

**Aggregation:** Mean pooling of node embeddings per chunk
- For each chunk, get all nodes (words) from its triplets
- Average their Node2Vec embeddings
- Result: One 128-dim vector per chunk representing its graph structure

**Validation:**
- Visualize embeddings with t-SNE
- Check that semantically similar chunks cluster together
- Expected: Chunks about similar topics have closer graph embeddings

**Estimate:** 2 hours implementation + validation
**Dependencies:** karateclub, numpy, networkx
**Files:** train_graph_distillation.py (node2vec section)"""

    cursor.execute("""
    INSERT INTO features (name, description, status, created_at, updated_at)
    VALUES (?, ?, 'TODO', ?, ?)
    """, ("Node2Vec Graph Embeddings from AOKG", feature1, datetime.now(), datetime.now()))
    
    # Feature 2: DistilBERT Distillation Model
    feature2 = """Train lightweight model to predict graph embeddings directly from text.

**Architecture:**
- Base: DistilBERT (distilbert-base-uncased)
- Projection: Linear layer mapping 768-dim → 128-dim
- Input: Raw chunk text (tokenized)
- Output: 128-dim graph embedding (without graph construction)

**Training:**
- Data: 250 chunks with known graph embeddings (from Node2Vec)
- Loss: MSE between predicted and actual graph embeddings
- Split: 200 train, 50 validation
- Epochs: 10-20 (early stopping on validation loss)
- Optimizer: AdamW (lr=3e-5)

**Goal:** Learn to map text → graph structure implicitly
- Model learns patterns: "words A, B, C together → this graph topology"
- No explicit graph construction at inference time
- 37x faster than BIO tagger + Node2Vec pipeline

**Validation:**
- Correlation between predicted vs actual embeddings (target: r > 0.7)
- Cosine similarity distribution
- t-SNE plot: predicted vs actual embeddings should overlap

**Estimate:** 4-6 hours training + validation
**Dependencies:** torch, transformers, numpy
**Files:** train_graph_distillation.py (distillation section)"""

    cursor.execute("""
    INSERT INTO features (name, description, status, created_at, updated_at)
    VALUES (?, ?, 'TODO', ?, ?)
    """, ("DistilBERT Graph Embedding Distillation", feature2, datetime.now(), datetime.now()))
    
    # Feature 3: Full Corpus Graph Embedding Derivation
    feature3 = """Apply distilled model to derive graph embeddings for all 161,389 chunks.

**Performance:**
- Throughput: ~150 chunks/sec (vs 4.08 chunks/sec with BERT BIO)
- Full corpus time: ~18 minutes (vs 11 hours with BERT)
- Speedup: 37x faster, 97% time reduction

**Process:**
1. Load distilled model (graph_distillation_model.pt)
2. Load chunks from checkpoints/chunks.msgpack
3. Batch inference (batch_size=64)
4. Store embeddings: either msgpack OR pgvector table

**Storage Options:**
- Option A: msgpack file (graph_embeddings.msgpack)
  - Pros: Simple, portable
  - Cons: Need to load all into memory for HNSW
- Option B: pgvector table (chunks table, add graph_embedding column)
  - Pros: Unified storage with model2vec embeddings
  - Cons: Requires PostgreSQL

**Validation:**
- Benchmark throughput on 1000 chunks first
- Verify embeddings are reasonable (check norms, distributions)
- Spot-check: manually inspect embeddings for known chunks

**Estimate:** 1 hour implementation + 18 minutes inference + 30 min validation
**Dependencies:** torch, transformers, msgpack OR psycopg2
**Files:** apply_graph_embeddings.py"""

    cursor.execute("""
    INSERT INTO features (name, description, status, created_at, updated_at)
    VALUES (?, ?, 'TODO', ?, ?)
    """, ("Full Corpus Graph Embedding Derivation", feature3, datetime.now(), datetime.now()))
    
    # Feature 4: HNSW Index on Graph Embeddings
    feature4 = """Build HNSW index for fast approximate nearest neighbor search on graph embeddings.

**Purpose:** Enable fast graph-based retrieval as alternative/complement to dense retrieval.

**Implementation:**
- Library: pgvector (PostgreSQL extension) OR hnswlib (standalone)
- Index type: HNSW (Hierarchical Navigable Small World)
- Distance metric: Cosine similarity (L2 normalized embeddings)
- Parameters: ef_construction=200, M=16 (same as model2vec)

**pgvector approach:**
```sql
ALTER TABLE chunks ADD COLUMN graph_embedding vector(128);
-- Populate from apply_graph_embeddings.py
CREATE INDEX ON chunks USING hnsw (graph_embedding vector_cosine_ops);
```

**hnswlib approach:**
```python
import hnswlib
index = hnswlib.Index(space='cosine', dim=128)
index.init_index(max_elements=161389, ef_construction=200, M=16)
index.add_items(graph_embeddings, ids)
index.save_index('graph_embeddings.hnsw')
```

**Validation:**
- Query with known chunks, verify similar chunks retrieved
- Compare graph retrieval vs dense retrieval (different signals)
- Measure query latency (target: < 100ms for top-100)

**Estimate:** 2 hours implementation + tuning
**Dependencies:** pgvector OR hnswlib
**Files:** build_graph_hnsw.py"""

    cursor.execute("""
    INSERT INTO features (name, description, status, created_at, updated_at)
    VALUES (?, ?, 'TODO', ?, ?)
    """, ("HNSW Index on Graph Embeddings", feature4, datetime.now(), datetime.now()))
    
    # Feature 5: Graph-Based Reranking
    feature5 = """Rerank hybrid-retrieved candidates using graph embeddings as third signal.

**Pipeline:**
1. Hybrid retrieval: BM25 + model2vec dense → ~100 candidates
2. RRF fusion of BM25 + dense → ~50 top candidates
3. Graph retrieval: Query graph embedding → top-50 by graph similarity
4. Final fusion: Combine BM25 + dense + graph scores

**Fusion Strategies:**

Option A — RRF (Reciprocal Rank Fusion):
```python
score_rrf = 1/(k + rank_bm25) + 1/(k + rank_dense) + 1/(k + rank_graph)
# k=60 (standard RRF constant)
```

Option B — Learned Linear Combination:
```python
score = w1*score_bm25 + w2*score_dense + w3*score_graph
# Learn weights w1, w2, w3 on validation set
```

Option C — Cascade:
- Stage 1: BM25 → top-200
- Stage 2: Dense rerank → top-50
- Stage 3: Graph rerank → final top-5

**Expected Improvement:**
- Graph signal captures semantic distance via AOKG structure
- LCA convergence level (0-3) as discrete metric
- Precision@5 improvement (exact gain TBD, benchmark needed)

**Validation:**
- Benchmark on known query set
- Measure precision@5, recall@10, nDCG@10
- Compare: hybrid-only vs hybrid+graph
- Ablation: BM25 vs dense vs graph contributions

**Estimate:** 3 hours implementation + 2 hours benchmarking
**Dependencies:** Depends on retriever code + HNSW index
**Files:** graph_reranker.py, integrate with query_arxiv.py"""

    cursor.execute("""
    INSERT INTO features (name, description, status, created_at, updated_at)
    VALUES (?, ?, 'TODO', ?, ?)
    """, ("Graph-Based Reranking of Hybrid Results", feature5, datetime.now(), datetime.now()))
    
    # Add architectural decisions
    
    # Decision 1: Node2Vec + Distillation
    decision1_rationale = """The BIO tagger (BERT-based) takes 11 hours to process the full corpus (161,389 chunks at 4.08 chunks/sec). This makes full-corpus graph construction infeasible for retrieval.

Alternative approaches considered:
1. Run BERT BIO on full corpus → 11 hours (rejected: too slow)
2. Graph Transformer trained end-to-end → Requires labeled data + complex implementation (deferred)
3. Graph2Vec instead of Node2Vec → Similar approach, whole-graph embeddings (comparable)
4. Node2Vec + distillation → Train fast model on 250 chunks, apply to full corpus

**Why Node2Vec + Distillation:**
- Proven technique: Node2Vec widely used for graph embeddings
- Simple implementation: karateclub library has clean API
- Fast application: DistilBERT ~150 chunks/sec (37x speedup)
- No labeled data needed: Distillation uses BERT outputs as supervision
- Modular: Can upgrade to Graph Transformer later if needed

**Trade-offs:**
- Pros: 37x speedup, simple implementation, proven technique
- Cons: Requires two-phase training, distillation quality depends on 250-chunk coverage"""

    cursor.execute("""
    INSERT INTO architectural_decisions (decision, rationale, before_state, after_state, decision_timestamp)
    VALUES (?, ?, ?, ?, ?)
    """, (
        "Use Node2Vec + Distillation instead of full BERT inference",
        decision1_rationale,
        "BIO tagger on full corpus: 11 hours (4.08 chunks/sec)",
        "Distilled model: 18 minutes (~150 chunks/sec), 37x speedup",
        datetime.now()
    ))
    
    # Decision 2: Reranker vs Full Corpus Expansion
    decision2_rationale = """Graph-based retrieval can be used in two ways:
1. Recall expansion: Build graph over full corpus, use for initial retrieval
2. Precision reranking: Apply graph only to hybrid-retrieved candidates

**Why Reranker:**
- Hybrid retrieval (BM25 + dense) already provides good recall
- BIO extraction on 161k chunks still takes 18 minutes (with distilled model)
- Graph structure provides semantic distance signal (LCA convergence level)
- Reranking 50-100 candidates takes ~5-10 seconds (acceptable latency)
- Full corpus graph construction + HNSW index is expensive to maintain

**Trade-offs:**
- Pros: Fast query time, leverages hybrid recall, graph adds precision
- Cons: Cannot discover documents missed by BM25+dense (no recall improvement)

**Future:** If distilled model proves highly accurate, can explore full-corpus graph retrieval as alternative ranking signal."""

    cursor.execute("""
    INSERT INTO architectural_decisions (decision, rationale, before_state, after_state, decision_timestamp)
    VALUES (?, ?, ?, ?, ?)
    """, (
        "Use graph embeddings as reranker over hybrid results (not full corpus expansion)",
        decision2_rationale,
        "Graph-based retrieval infeasible due to 11-hour BERT inference",
        "Graph reranking on 50-100 candidates: ~5-10 seconds, precision improvement",
        datetime.now()
    ))
    
    # Decision 3: 250 Chunks for Distillation Training
    decision3_rationale = """Distillation training requires examples with known graph embeddings. Options:
1. Use existing 250-chunk BIO training set (already extracted, validated)
2. Extract new larger set (500-1000 chunks) for better coverage
3. Active learning: Iteratively select diverse chunks

**Why 250 Chunks:**
- Already available: bio_training_250chunks_complete_FIXED.msgpack
- Already validated: BIO tagger trained on this, quality confirmed
- Sufficient diversity: Academic text from arXiv across multiple domains
- Fast to process: ~60 seconds BERT inference + graph construction
- Avoids additional cost: No need to run BIO tagger on more chunks

**Validation Strategy:**
- If distillation quality is poor (correlation r < 0.7), can expand to 500-1000 chunks
- Monitor embedding distribution: Check if 250 chunks cover the full corpus distribution
- Active learning future work: Select chunks that maximize coverage

**Trade-offs:**
- Pros: No additional cost, already validated, sufficient starting point
- Cons: May not cover all graph patterns in full corpus (mitigated by validation)"""

    cursor.execute("""
    INSERT INTO architectural_decisions (decision, rationale, before_state, after_state, decision_timestamp)
    VALUES (?, ?, ?, ?, ?)
    """, (
        "Use existing 250-chunk BIO training set for distillation",
        decision3_rationale,
        "Need examples with graph embeddings for distillation training",
        "Use bio_training_250chunks_complete_FIXED.msgpack (validated, diverse, sufficient)",
        datetime.now()
    ))
    
    conn.commit()
    conn.close()
    
    print("✅ Created graph_transformer_feature_catalog.sqlite3")
    print()
    print("Added features:")
    print("  1. Workflow: Graph Transformer Execution Order (DONE)")
    print("  2. Node2Vec Graph Embeddings from AOKG (TODO)")
    print("  3. DistilBERT Graph Embedding Distillation (TODO)")
    print("  4. Full Corpus Graph Embedding Derivation (TODO)")
    print("  5. HNSW Index on Graph Embeddings (TODO)")
    print("  6. Graph-Based Reranking of Hybrid Results (TODO)")
    print()
    print("Added architectural decisions:")
    print("  1. Node2Vec + Distillation vs full BERT inference")
    print("  2. Reranker vs full corpus expansion")
    print("  3. 250 chunks for distillation training")
    print()
    print("Database organization:")
    print("  - bio_tagger_features.sqlite3 → BIO tagger pipeline")
    print("  - retriever_feature_catalog.sqlite3 → Hybrid retriever (BM25 + dense)")
    print("  - graph_transformer_feature_catalog.sqlite3 → Graph transformer reranking")


if __name__ == "__main__":
    create_graph_transformer_catalog()
