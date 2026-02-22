# Hybrid Retrieval & Knowledge Extraction System

> Auto-generated from feature catalogs on 2026-02-21 17:46.
> Source databases: `retriever_feature_catalog.sqlite3`, `bio_tagger_features.sqlite3`, `graph_transformer_feature_catalog.sqlite3`
> Generator: `generate_readme.py`

**Three subsystems:**
1. **Hybrid Retriever** — BM25 + dense embedding retrieval with GIST pipeline, RRF fusion, ColBERT/CrossEncoder reranking, pgvector backend
2. **BIO Tagger** — BERT-based SPO triplet extraction, Augmented Ontological Knowledge Graph (AOKG), Streamlit demo
3. **Graph Transformer** — Node2Vec distillation for fast graph embeddings, precision reranking over hybrid results

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│              THREE-LAYER φ-SCALED RETRIEVER PIPELINE             │
│                                                                  │
│  Document Sources → Equidistant Chunking → Checkpointing        │
│       ↓                                                          │
│  Dual Indexing: Dense (model2vec 256→64 PCA) + Sparse (BM25)    │
│       ↓                                                          │
│  PostgreSQL + pgvector (HNSW dense + GIN JSONB sparse)           │
│       ↓                                                          │
│  Layer 1: Hybrid Seeds                                           │
│    BM25 pool (top_k²) + Dense pool (top_k²) → RRF               │
│    → prev_fib(top_k²) seeds (e.g. 169+169 → RRF → 144 chunks)   │
│       ↓  seeds + ECDF weights as INPUT to Layer 2                │
│  Layer 2: ECDF-Weighted Dual Expansion                           │
│    Path A: Graph BM25 — ECDF-weighted TF → triplet BM25         │
│            query top_k×2 → filter to top_k per path              │
│    Path B: Qwen3 Dense — ECDF-weighted centroid → cosine         │
│            query top_k×2 → filter to top_k per path              │
│    RRF merge → prev_fib(top_k²) NEW chunks (seeds excluded)     │
│       ↓                                                          │
│  Section Expansion: chunks → all sections from their papers      │
│       ↓                                                          │
│  Layer 3: Dual Reranking                                         │
│    ColBERTv2 Late Interaction + Cross-Encoder (bge-reranker)     │
│    RRF merge → top-k papers → all sections from those papers     │
│       ↓                                                          │
│  [PLANNED] Graph Transformer Reranking via AOKG                  │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                    BIO TAGGER PIPELINE                            │
│                                                                  │
│  Source Chunks (checkpoints/chunks.msgpack — 161,389 chunks)     │
│       ↓                                                          │
│  Stanza Dep Parser (teacher) → Multi-hot BIO Labels              │
│       ↓                                                          │
│  BIOTagger: BERT + 6 Binary Classifiers (B/I-SUBJ/PRED/OBJ)     │
│       ↓                                                          │
│  Optuna Tuning (BoxCox resampling, 21 trials)                    │
│       ↓                                                          │
│  BIOTripletExtractor Inference → SPO Spans                       │
│       ↓                                                          │
│  Post-Inference Stopword Removal (clean_span_tokens)             │
│       ↓                                                          │
│  Cartesian Product Expansion → Atomic SPO Triplets               │
│       ↓                                                          │
│  AOKG: 4-Layer Knowledge Graph                                   │
│    L0=Surface → L1=Lemma → L2=Synset → L3=Hypernym              │
│    LCA convergence level (0-3) = semantic distance metric        │
│       ↓                                                          │
│  Streamlit Demo (3 tabs: Interactive, Eval Browser, Metrics)     │
└──────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Hybrid Retriever (ArXiv Papers)
```bash
# Ingest ArXiv papers
python arxiv_chunking_pipeline.py

# Query
python query_arxiv.py "transformer attention mechanism" --top-k 5
```

### Hybrid Retriever (Quotes)
```bash
# Ingest quotes dataset
python ingest_quotes.py

# Query
python query_quotes.py "knowledge is power" --top-k 5
```

### BIO Tagger
```bash
# Option 1: Interactive menu
run_bio_tagger.bat

# Option 2: Direct Streamlit launch
streamlit run streamlit_bio_demo.py
# → http://localhost:8501

# Training pipeline (if retraining)
python extract_bio_training_data.py --chunks 250 --output training.msgpack
python tune_bio_tagger.py --data training.msgpack --unfreeze-layers 12 --n-trials 21
```

---

## 🔍 Hybrid Retriever — Feature Catalog

*Source: `retriever_feature_catalog.sqlite3` — 19 features, 5 architectural decisions, 8 claims*

| Status | Count |
|--------|-------|
| ✅ DONE | 13 |
| 🔄 IN PROGRESS | 3 |
| ❌ FAILED | 3 |

### Features

#### 1. Frozen BERT Baseline ✅ DONE

Fixed BIO labeling bugs + established baseline with frozen BERT + single linear layer

*Baseline F1: 0.1328 | Current F1: 0.1764 | Notes: Best trial F1=0.1764, but full training collapsed to 0.0805 due to overfitting. Confirms frozen BERT architecture limitation.*

#### 2. Runtime Label Mismatch Bug 🧪 VALIDATING

collate_fn bidirectional substring matching produces 45% different labels than pre-computed

*Baseline F1: 0.1764*

#### 3. Pre-computed Label Sparsity ❌ FAILED

Pre-computed labels in bio_training_250chunks_clean.msgpack only label ~30-40% of tokens

*Baseline F1: 0.1764 | Current F1: 0.0670*

#### 4. Class-Balanced Sampling ❌ FAILED

Iterative token-level class balancing using log/Box-Cox transform

*Baseline F1: 0.1218 | Current F1: 0.1355 | Notes: Class balancing achieved O-token reduction (65.6%→53.6%) but F1 worse (0.1218→0.1355). MORE DATA works better: 160 examples→F1=0.1408. Root cause: Pre-computed labels still incomplete (36.6% coverage). Runtime matcher had 45% wrong labels but higher coverage.*

#### 5. More Training Data ✅ DONE

Three-Layer phi-Retriever: Production implementation for academic paper retrieval.

## Search Process (How It Works)

### Layer 1: Hybrid Seed Retrieval (Facts — BM25-only)
- BM25 keyword search over 161,389 chunks in PostgreSQL/pgvector
- Returns prev_fib(top_k^2) seed chunks as starting points (e.g. top_k=13 -> 169 -> 144 seeds)
- Dense embeddings (M2V) disabled at L1 (too weak for domain queries)

### Layer 2: Dual Expansion (Premises — Graph BM25 + Qwen3 Semantic)
Two independent expansion paths, each GIST-diversified, then RRF-merged:

**GIST (Greedy Independent Set Thresholding):**
- Coverage: distance from k-means centroids via clustering over candidates
- Utility: similarity score via correlation matrix (maximize similarity with query, minimize correlation with other chunks/docs)
- Selection: score[i] = λ*coverage[i] + (1-λ)*utility[i] where λ=0.7
- Each Layer 2 path applies GIST independently, then RRF fuses both paths

**Path A: Graph BM25 Expansion**
1. Extract SPO triplets from seed chunks (via chunk_to_triplets mapping)
2. Concatenate all lemmatized subject/predicate/object tokens as a mega-query
3. BM25 search over the 161K triplet corpus -> top_k×2 candidates (26 for top_k=13)
4. Map triplet hits back to their source chunks (via triplet_to_chunks mapping)
5. GIST select: k-means clustering for coverage + correlation matrix for utility (lambda=0.7)
   - Coverage: KMeans(n_clusters=top_k) over BM25 candidate scores
     → coverage[i] = min_distance_to_centroids[i]
     → Higher distance = more diverse from cluster centers
   - Utility: Correlation matrix over BM25 scores
     → corr_matrix = np.corrcoef(bm25_scores)
     → utility[i] = bm25_score[i] - mean(corr_matrix[i, j!=i])
     → Maximize BM25 relevance, minimize inter-chunk correlation
   - Selection: score[i] = 0.7*coverage[i] + 0.3*utility[i]
     → Select top_k (13) chunks with highest scores
6. Result: diverse, graph-adjacent chunks NOT in original seeds

**Path B: Qwen3 Semantic Expansion**
1. Mean-pool Qwen3 256d embeddings of all seed chunks -> query vector
2. Cosine similarity against ALL 161K chunk embeddings -> top_k×2 candidates (26 for top_k=13)
3. Exclude seed chunks
4. GIST select: k-means clustering for coverage + correlation matrix for utility (lambda=0.7)
   - Coverage: KMeans(n_clusters=top_k) over candidate embeddings
     → coverage[i] = min_distance_to_centroids[i]
     → Higher distance = more diverse from cluster centers
   - Utility: Correlation matrix over cosine similarity scores
     → corr_matrix = np.corrcoef(cosine_scores)
     → utility[i] = cosine_score[i] - mean(corr_matrix[i, j!=i])
     → Maximize cosine relevance, minimize inter-chunk correlation
   - Selection: score[i] = 0.7*coverage[i] + 0.3*utility[i]
     → Select top_k (13) chunks with highest scores
5. Result: semantically similar chunks NOT in original seeds

**Merge: RRF fusion of Path A + Path B -> prev_fib(top_k^2) expansion chunks**
Total after Layer 2: prev_fib(top_k^2) expansion chunks (seeds excluded from output)

### Layer 3: Dual Reranking + Paper Selection (Syllogism)
1. ColBERTv2 scores ALL 157 candidates against query -> full ranking
2. Cross-Encoder scores ALL 157 candidates against query -> full ranking
3. RRF merges both full rankings (k=60)
4. Walk down RRF-sorted list, collect unique paper_ids until top_k papers found
5. Return ALL sections/chunks from those top_k papers (preserving RRF order)
6. Result: all sections from top_k papers (not just best chunk per paper)

## Key Design Decisions
- GIST diversity in L2 prevents redundant expansions
- Graph BM25 finds structurally-adjacent knowledge (same-topic different angles)
- Qwen3 cosine finds semantically-similar knowledge (distant but related)
- ColBERTv2 + Cross-Encoder dual reranking provides robust relevance signal
- Paper-level selection returns complete context (all sections from top papers)

## Files
- three_layer_phi_retriever.py (613 lines)
- query_three_layer.py (321 lines, CLI)

## Dependencies
- PostgreSQL + pgvector (Docker: langchain_postgres)
- checkpoints/: chunks.msgpack, chunk_embeddings_qwen3.msgpack, chunk_to_triplets.msgpack, triplet_to_chunks.msgpack
- triplet_checkpoints_full/stage4_lemmatized.msgpack (161K triplets)
- BM25 index rebuilt at load time from triplet tokens
- Models: colbert-ir/colbertv2.0, cross-encoder/ms-marco-MiniLM-L-6-v2


*Baseline F1: 0.1218 | Current F1: 0.1408 | Notes: VALIDATED 2026-02-09: Query 'agentic memory methods' -> 88 chunks from 13 papers in 166.93s. Breadth 8.5/10, Depth 8/10. Coverage: memory taxonomy, cognitive foundations, implementation details, retrieval mechanisms, learning paradigms, application domains. Missing (expected): benchmarking comparisons, production scaling, security/adversarial, cost modeling - these are literature gaps not retrieval gaps.*

#### 6. Stanza Dependency Parsing for BIO Extraction ✅ DONE

Replace OpenIE with Stanza dependency parsing in extract_bio_training_data.py to preserve multi-word phrases

*Notes: Fixed 1355 BIO violations (99.93% success). Proper B-I-I continuous spans validated.*

#### 7. Stanza Dependency Parsing for Graph Extraction ✅ DONE

Replace OpenIE with Stanza dependency parsing in build_arxiv_graph_sparse.py to preserve multi-word entity nodes

*Notes: Code fixed and validated standalone. Multi-word phrase preservation confirmed.*

#### 8. BIO Sequence Validation Unit Test ✅ DONE

Three-layer phi-retriever pipeline for academic paper retrieval.

Command: python query_three_layer.py "your query" [--top-k N] [-v] [--output results.md]
DB: localhost:5432/langchain/arxiv_chunks
Checkpoints: checkpoints/*.msgpack + triplet_checkpoints_full/stage4_lemmatized.msgpack

Layer 1: BM25 keyword search -> prev_fib(top_k^2) seeds (e.g. 144 for top_k=13; no dense at L1)
Layer 2: Graph BM25 expansion + Qwen3 semantic expansion -> GIST diversified -> RRF merged -> prev_fib(top_k^2) new chunks (seeds excluded from L2 output)
Layer 3: ColBERTv2 + Cross-Encoder dual reranking -> RRF merge -> walk-down paper selection -> ALL sections from top-k papers

Models: Qwen3 256d (L2), colbertv2.0 (L3), ms-marco-MiniLM-L-6-v2 (L3)
Output: All sections from top-k papers (not just best chunk per paper)


*Notes: VALIDATED 2026-02-09: 88 chunks from 13 papers for 'agentic memory methods'. 166.93s runtime. Git commit 7b47574.*

#### 9. get_subtree_text() Function ✅ DONE

Recursively collects dependency subtree to preserve multi-word phrases during extraction

*Notes: Core function for Stanza phrase preservation. Tested and working.*

#### 10. extract_spo_from_sentence() Function ✅ DONE

Extracts S-P-O triplets using Stanza dependency parse instead of OpenIE atomization

*Notes: Replaces OpenIE extraction. Uses get_subtree_text() for full phrases.*

#### 11. BIO Training Data Regeneration ✅ DONE

Regenerated bio_training_250chunks_complete_FIXED.msgpack with Stanza extraction

*Notes: 1103 examples generated. 1355 violations → 1 violation (99.93% fix rate).*

#### 12. Streamlit BIO Tagger Dashboard ✅ DONE

Interactive web dashboard for testing BIO tagger predictions. Allows users to input text and visualize Subject-Predicate-Object extraction with entity highlighting.

*Notes: Implemented in streamlit_bio_demo.py. Loads trained model (bio_tagger_multiclass.pt) and provides interactive testing interface. Shows holdout predictions for reference. Full functionality: text input, BIO prediction, entity extraction, visualization.*

#### 13. Conditional I-PRED Detection and Removal 🔄 IN_PROGRESS

Automatically detect and remove I-PRED class when it has 0 tokens in training data. Root cause: 100% single-token predicates mean I-PRED never exists. Removes 16.7% penalty from macro F1.

*Baseline F1: 0.6715 | Notes: Training started with Optuna. Detection confirmed: I-PRED has 0 tokens, model created with 5 classifiers. Per-class metrics show only 5 labels as expected.*

#### 14. Stopword Filtering in Training Data 🔄 IN_PROGRESS


    ROOT CAUSE FIX: Filter stopwords from BIO training data during data preparation.
    
    PROBLEM IDENTIFIED:
    - Stopwords ("an", "to", etc.) labeled as entities in predictions
    - Training data includes stopwords in entity spans from Stanza extractions
    - No filtering applied during BIO labeling
    
    SOLUTION:
    - Regenerate training data with stopword masking
    - Mask stopwords as 'O' label even if in entity boundaries
    - Retrain model with clean data
    
    COMPLETE PIPELINE (Run in Order):
    
    Step 1: Extract BIO Training Data
    -----------------------------------
    Script: extract_bio_training_data.py
    Purpose: Extract SPO triplets from ArXiv chunks using Stanza, convert to BIO labels
    Input: ArXiv chunks (from pgvector database or msgpack)
    Output: bio_training_250chunks.msgpack (raw training data with stopwords)
    Runtime: ~30-60 minutes (depends on chunk count)
    
    Command:
      python extract_bio_training_data.py --chunks 250
    
    Step 2: Filter Stopwords from Training Data
    --------------------------------------------
    Script: filter_training_stopwords.py
    Purpose: Mask stopwords as 'O' label in BIO sequences
    Input: bio_training_250chunks_complete_FIXED.msgpack (or raw training data)
    Output: bio_training_250chunks_complete_FILTERED.msgpack (clean data)
    Runtime: ~10 seconds
    
    Command:
      python filter_training_stopwords.py
    
    Step 3: Train/Tune BIO Tagger Model
    ------------------------------------
    Script: tune_bio_tagger.py
    Purpose: Train BERT-based BIO tagger with Optuna hyperparameter tuning
    Input: bio_training_250chunks_complete_FILTERED.msgpack (clean training data)
    Output: bio_tagger_atomic.pt (trained model)
    Runtime: ~30-60 minutes (depends on n_trials)
    
    Command:
      python tune_bio_tagger.py \
        --data bio_training_250chunks_complete_FILTERED.msgpack \
        --unfreeze-layers 12 \
        --n-trials 21
    
    Step 4: Test Predictions (Verification)
    ----------------------------------------
    Script: test_predictions.py
    Purpose: Verify stopwords now labeled as 'O' (not entity labels)
    Input: bio_tagger_atomic.pt (trained model)
    Output: Console output with predictions
    Runtime: <5 seconds
    
    Command:
      python test_predictions.py
    
    Expected Result:
      - "an" → O ✅ (not B-OBJ)
      - "to" → O ✅ (not I-OBJ)
    
    Step 5: Launch Streamlit App (Production)
    ------------------------------------------
    Script: streamlit_bio_demo.py
    Launcher: run_bio_tagger.bat app
    Purpose: Interactive BIO tagging interface
    Input: bio_tagger_atomic.pt (trained model)
    Runtime: Continuous (web server)
    
    Command:
      run_bio_tagger.bat app
    
    EXPECTED IMPACT:
    - Improved precision (fewer false positive entity boundaries)
    - Better B-OBJ F1 (currently 0.6197, lowest score)
    - Better I-OBJ F1 (currently 0.5295, second lowest)
    - Cleaner entity boundary predictions
    
    KEY FILES:
    - filter_training_stopwords.py (new script, Phase 48)
    - extract_bio_training_data.py (existing, Phase 44)
    - tune_bio_tagger.py (existing, Phase 45)
    - test_predictions.py (existing, Phase 46)
    - run_bio_tagger.bat (existing, Phase 47)
    
    DATA FILES:
    - Input: bio_training_250chunks_complete_FIXED.msgpack
    - Output: bio_training_250chunks_complete_FILTERED.msgpack
    - Model: bio_tagger_atomic.pt (will be replaced)
    
    NOTES:
    - Data structure uses 'training_data' key (not 'training' or 'eval')
    - No separate eval split in msgpack (splitting done at training time)
    - NLTK stopwords corpus used for filtering
    

*Baseline F1: 0.7467*

#### 15. Graph Transformer Reranking Stage ❌ FAILED

DEPRECATED: Node2Vec/Graph2Vec approach has been replaced by triplet-based expansion.

**DEPRECATION NOTICE (2026-02-08):**
This feature was originally designed to use Node2Vec embeddings for graph-based reranking.
However, the system now uses **semantic triplet extraction** (Stanza dependency parsing) 
for graph expansion instead. See Feature 17 for the active triplet-based implementation.

**Why Deprecated:**
- Node2Vec was never fully implemented (remained as TODO/placeholder)
- Triplet-based approach (BM25 over triplet corpus) provides better semantic matching
- Semantic triplets preserve entity relationships without graph embedding overhead
- Working implementation exists in ThreeLayerPhiRetriever using triplet BM25 expansion

**Migration Path:**
Use triplet-based graph expansion (Feature 17) instead of Node2Vec.

**Original Architecture (NOT USED):**
1. Hybrid Retrieval (BM25 + dense) → ~50-100 candidate chunks
2. BIO Tagger → SPO triplets → AOKG (only on candidates, ~25 seconds)
3. Graph-based reranking using Node2Vec embeddings

*Notes: Depends on bio_tagger_features.sqlite3 pipeline being complete.*

#### 16. Workflow: Hybrid Retriever Execution Order ✅ DONE

Workflow Execution Order for Hybrid Retrieval System:

1. **Chunk Documents**
   - Script: arxiv_chunking_pipeline.py
   - Purpose: Ingest PDFs/text, apply equidistant chunking (log median + 2*MAD), embed with model2vec
   - Command: python arxiv_chunking_pipeline.py --input-dir ./papers/ --collection arxiv
   - Output: checkpoints/chunks.msgpack + PostgreSQL pgvector tables

2. **Build BM25 Vocabulary**
   - Script: build_adaptive_vocab.py
   - Purpose: Extract vocabulary for BM25 lexical matching
   - Command: python build_adaptive_vocab.py --chunks checkpoints/chunks.msgpack --output bm25_vocab.msgpack
   - Output: bm25_vocab.msgpack (or quotes_bm25_vocab.msgpack)

3. **Query Documents**
   - Scripts: query_arxiv.py, query_quotes.py
   - Purpose: 8-stage GIST hybrid retrieval (BM25+dense → diversity → RRF → filters)
   - Command: python query_arxiv.py "query text here"
   - Output: Retrieved sections with relevance scores

4. **Extract Knowledge Graph (Planned - TODO)**
   - Script: (to be created)
   - Purpose: Run BIO tagger on retrieved chunks → build AOKG → graph transformer reranking
   - Command: TBD
   - Output: Reranked results with graph-based semantic distance signal

**Data Sources:**
- ArXiv corpus: checkpoints/chunks.msgpack (161,389 chunks)
- Quotes corpus: quotes_dataset.json (3,500 quotes)
- PostgreSQL: arxiv_retrieval, quotes_retrieval databases

**Quick Start:**
python query_arxiv.py "machine learning transformers"    # Query ArXiv
python query_quotes.py "life wisdom"                     # Query quotes

#### 17. Three-Layer Triplet-Based Retrieval (φ-Scaled) ✅ DONE

Three-Layer phi-Scaled Retrieval with ECDF-Weighted Expansion (ACTIVE)

**Architecture (top_k=13):**

Layer 1: Hybrid Seeds (Facts)
  BM25 pool (top_k^2 = 169 chunks) + Dense pool (top_k^2 = 169 chunks)
  -> Pool 169+169 -> RRF merge -> prev_fib(169) = 144 seed chunks
  Seeds + ECDF weights as INPUT to Layer 2 (seeds EXCLUDED from output)

Layer 2: ECDF-Weighted Dual Expansion (Premises)
  Path A: Graph BM25 -- ECDF-weighted TF -> query top_k x 2 (26) -> GIST select top_k (13) per path
  Path B: Qwen3 Dense -- ECDF-weighted centroid -> query top_k x 2 (26) -> GIST select top_k (13) per path
  GIST (Greedy Independent Set Thresholding): coverage via k-means clustering + utility via correlation matrix (lambda=0.7)
  RRF merge -> prev_fib(top_k^2) = 144 NEW chunks (seeds excluded)
  Section Expansion: 144 chunks -> ALL sections from their papers

Layer 3: Dual Reranking (Syllogism)
  ColBERTv2 Late Interaction + Cross-Encoder (bge-reranker)
  RRF merge -> top_k+1 papers -> drop last paper -> top_k (13) papers with all their sections

**phi-Scaling:**
- top_k = 13
- top_k^2 = 169 (retrieval breadth)
- prev_fib(169) = 144 (phi cascade output)
- Dense index: HNSW (NOT IVFFlat)
- Sparse index: GIN on JSONB (NOT IVFFlat -- IVFFlat is dense-only ANN)

**Key Mechanisms:**
1. Midpoint ECDF weighting: (count_leq + count_lt)/(2n) from L1 BM25 scores
2. Path A term repetition: top-ECDF seeds contribute 3x, bottom 1x
3. Path B weighted centroid: np.average(embeddings, weights=ecdf)
4. L2 oversample 2x per path, GIST select top_k per path (diversity), then RRF merge
   - GIST: k-means clustering (coverage) + correlation matrix (utility), lambda=0.7
5. Seeds excluded during expansion, not concatenated to output


*Notes: LESSON LEARNED (2026-02-13):
Agent incorrectly reported Layer 1 retrieves 441 chunks (21^2). This was the abstract default from gist_retriever.py (top_k=21), NOT production. Production uses top_k=13 (base_gist_retriever.py), so top_k^2 = 169 per pool.

ROOT CAUSE: Conflated abstract class defaults with production config.
- gist_retriever.py search() default: top_k=21 -> top_k^2 = 441 (ABSTRACT ONLY)
- base_gist_retriever.py search() default: top_k=13 -> top_k^2 = 169 (PRODUCTION)

RULE: Always reference the CONCRETE production config (base_gist_retriever.py, top_k=13), never the abstract class defaults (gist_retriever.py, top_k=21).

Also corrected: Production indexes are HNSW (dense) + GIN (JSONB sparse), NOT IVFFlat. IVFFlat is dense-only ANN and cannot index BM25 sparse vectors.

CORRECTED PIPELINE (top_k=13):
  L1: 169 BM25 + 169 Dense -> RRF -> 144 chunks
  L2: top_k x 2 = 26 per expansion path -> RRF -> 144 sections
  L3: RRF -> 13 papers*

#### 18. Layer 2 Query Expansion (ECDF-Weighted Dual Path) ✅ DONE

Dual expansion via BM25 triplet search + Qwen3 embeddings (256d), ECDF-weighted from Layer 1 scores, PostgreSQL pgvector backend.

Path A: Weighted mean TF profile for BM25 (L2 triplet BM25 table, GIN JSONB index).
Path B: Weighted centroid for dense search (L2 embeddings 256d table, HNSW index).

Per-path flow: query top_k x 2 from DB -> exclude seeds -> filter to top_k -> RRF merge both paths.
Index types: HNSW for dense vectors, GIN for JSONB sparse. NOT IVFFlat (IVFFlat is dense-only ANN, cannot index BM25).

*Notes: Restored using pgvector tables: layer2_triplet_bm25 (JSONB sparse, GIN index) and layer2_embeddings_256d (HNSW index).
Concrete implementations in query_modules/pgvector_retriever.py, abstract methods in gist_retriever.py.

CORRECTION (2026-02-12): Production expansion uses top_k x 2 per path (not top_k^2).
query_modules/pgvector_retriever.py lines 321, 385: query_layer2_*(query, top_k * 2).
Abstract default in gist_retriever.py (expansion_count = top_k^2) differs from concrete production path.*

#### 19. 3-Layer Hybrid Retrieval Pipeline (RRF Fusion) ✅ DONE

Core retrieval pipeline architecture using 3-layer cascade with parallel branches and RRF fusion.

**Architecture: 3 Layers × 2 Branches**

**Layer 1: Hybrid Retrieval (Facts)** (169 per branch → RRF → 144)
- Branch A: BM25 lexical retrieval (JSONB dot product on layer1_bm25_sparse.sparse_vector)
- Branch B: Dense semantic retrieval (HNSW on layer1_embeddings_128d.embedding)
- RRF fusion → 144 hybrid seeds (prev_fibonacci(169))
- 0% overlap confirms diverse signal fusion working correctly

**Layer 2: Graph Expansion (Premises)** (currently disabled)
- Branch A: Hybrid seeds passthrough (144)
- Branch B: Graph expansion via Node2Vec (returns [] — disabled)
- RRF fusion → 144 (unchanged since graph is empty)
- TODO: Bootstrap Node2Vec from top 50 hybrid seeds

**Layer 3: Document-Level Selection (Syllogism)** (top_k=13)
- Branch A: Document reconstruction from chunks (groups by paper_id + section_idx)
- Branch B: ColBERT late interaction scoring (token-level MaxSim)
- Final selection → top_k documents

**Key Parameters:**
- top_k = 13
- retrieval_limit = top_k² = 169
- hybrid_seeds = prev_fibonacci(169) = 144
- graph_expansion_limit = retrieval_limit × 2 = 338 (unused)

**Implementation:**
- Pipeline orchestrator: base_gist_retriever.py (BaseGISTRetriever.search)
- PostgreSQL backend: pgvector_retriever.py (PGVectorRetriever)
- RRF fusion: compute_rrf_score() with k=60
- Inheritance chain: ArxivRetriever → BaseGISTRetriever → PGVectorRetriever → GISTRetriever

**Database Schema:**
- arxiv_chunks: main table with bm25_sparse (JSONB, GIN index) + embedding (vector(256), HNSW index)
- layer1_embeddings_128d: reduced-dimension embeddings (vector(128), HNSW index)
- layer1_bm25_sparse: sparse BM25 vectors (JSONB as sparse_vector, GIN index)
- layer2_triplet_bm25: triplet-level BM25 (JSONB as triplet_bm25_vector, GIN index)

**GIST Selection Algorithm:** Used for diversity selection within each Layer 2 expansion path. Each path retrieves top_k×2 candidates, then GIST selects top_k diverse results via greedy scoring.

**GIST (Greedy Independent Set Thresholding):**
- Coverage: distance from k-means centroids via clustering over cartesian product of retrievals with each other (scores)
- Utility: similarity [default] score via correlation matrix maximizing for similarity with y (query) while minimizing correlation with other x vars (chunk/doc scores)
- Selection: score[i] = λ*coverage[i] + (1-λ)*utility[i] where λ=0.7
- Each Layer 2 GIST path processes independently, then results RRF fused together

**GIST Algorithm (Pseudo-code):**
```python
# Input: candidates (top_k×2 per path), query, lambda=0.7
# Output: top_k diverse selections per path

# Step 1: Coverage - k-means clustering over candidate set
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=top_k).fit(candidate_embeddings)
distances = kmeans.transform(candidate_embeddings)  # distance to all centroids
coverage_scores = distances.min(axis=1)  # distance to nearest centroid
# Higher distance = more diverse (farther from cluster center)

# Step 2: Utility - correlation matrix over similarity scores
corr_matrix = np.corrcoef(candidate_scores)  # pairwise correlation
query_sim = [similarity(query, cand) for cand in candidates]
utility_scores = []
for i in range(len(candidates)):
    # Maximize similarity to query, minimize correlation with other candidates
    avg_corr = np.mean([corr_matrix[i][j] for j in range(len(candidates)) if i != j])
    utility_scores[i] = query_sim[i] - avg_corr

# Step 3: Combined scoring and selection
final_scores = lambda * coverage_scores + (1-lambda) * utility_scores
selected_indices = np.argsort(final_scores)[-top_k:]  # top_k highest scores
return [candidates[i] for i in selected_indices]
```

**Key Insight:** GIST balances diversity (coverage via clustering) with relevance (utility via correlation matrix). Lambda=0.7 means 70% weight on diversity, 30% on relevance. RRF then fuses the two GIST-diversified paths. Layer 1 and Layer 3 use pure RRF fusion (no GIST).


*Notes: Production implementation validated via test_per_layer_counts.py against live PostgreSQL (161,389 rows).

VERIFIED COUNTS (2026-02-12):
- L1 BM25: 169 chunks ✓
- L1 Dense: 100 chunks ✓ (capped by hnsw.ef_search=100)
- L1 RRF fusion: 144 hybrid seeds ✓
- L2 Graph: 0 chunks ✓ (disabled)
- L2 RRF fusion: 144 chunks ✓
- L3 Reconstruction: 104 sections, 77 papers ✓
- L3 Final selection: 13 documents ✓

Orphaned methods discovered:
- query_modules/pgvector_retriever.py: _expand_layer2_bm25() and _expand_layer2_dense() never called by orchestrator

Schema notes:
- Layer tables use different column names than query_modules code expects (sparse_vector vs bm25_sparse, triplet_bm25_vector vs bm25_sparse)
- Production bypasses this by querying arxiv_chunks directly
- No layer2_embeddings_256d table exists — 256d embeddings live in arxiv_chunks.embedding*

### Architectural Decisions

**1. Replace OpenIE with Stanza Dependency Parsing**

- **Rationale:** OpenIE atomizes multi-word phrases ("deep learning models" → 3 separate entities), causing 1355 BIO violations. Stanza preserves full phrases via dependency subtree collection.
- **Before:** OpenIE: extract(sentence) → atomized triplets → BIO violations (B-SUBJ B-SUBJ B-SUBJ)
- **After:** Stanza: get_subtree_text(head_word) → complete phrases → proper BIO spans (B-SUBJ I-SUBJ I-SUBJ)

**2. Triplet-based graph expansion as post-hybrid precision stage**

- **Rationale:** Hybrid retrieval (BM25+dense) provides recall. Semantic triplet corpus provides precision signal through BM25 expansion.

**CRITICAL CLARIFICATION (2026-02-08):**
This system uses TRIPLET-BASED graph expansion via BM25 over the triplet corpus,
NOT Node2Vec embeddings. Earlier comments mentioning "Node2Vec" were placeholder TODOs.

**How Triplet Expansion Works:**
1. Extract triplets from seed chunks via Stanza dependency parsing
2. Get triplet lemmatized text (e.g., "model_learn pattern_from data")
3. Use triplet texts as BM25 query over full triplet corpus
4. Map high-scoring triplets back to chunks
5. Exclude seed chunks, return NEW chunks with ECDF gist scores

**Why Triplets (NOT Node2Vec):**
- Preserves semantic relationships explicitly (S-P-O structure)
- BM25 provides interpretable similarity scores
- No graph embedding training/inference overhead
- Working implementation validated in ThreeLayerPhiRetriever
- Better for multi-hop reasoning through triplet chains

**Data Format:**
- Full triplet corpus with lemma_text field (NOT sparse graph format)
- chunk_to_triplets and triplet_to_chunks mappings
- BM25 index built over triplet corpus

**Implementation:** three_layer_phi_retriever.py (_expand_via_graph_bm25)
- **Before:** 3-layer φ-scaled pipeline: L1(BM25+Dense) → L2(ECDF expansion) → L3(ColBERT+CrossEncoder)
- **After:** Proposed 10-stage: ...→CrossEncoder→BIO Extract→AOKG Rerank

**3. Deprecation: Node2Vec/Graph2Vec replaced by Triplet-Based Expansion**

- **Rationale:** ARCHITECTURAL DECISION: Replace Node2Vec with Triplet-Based Graph Expansion

**Decision Date:** 2026-02-08

**Context:**
Original design documents and code comments mentioned using Node2Vec for graph-based
expansion. However, this was never implemented beyond placeholder TODOs. The system
actually uses semantic triplet extraction (Stanza dependency parsing) with BM25 search
over the triplet corpus.

**Why Deprecate Node2Vec:**
1. Never implemented (remained as TODO comments in base_gist_retriever.py)
2. Triplet-based approach provides better semantic matching
3. BM25 over triplet corpus is more interpretable than graph embeddings
4. No training/inference overhead for Node2Vec model
5. Working implementation exists and is validated in ThreeLayerPhiRetriever

**Confusion Source:**
- base_gist_retriever.py line 11: "Graph Expansion from seeds (Node2Vec - placeholder)"
- base_gist_retriever.py line 117: "TODO: Bootstrap Node2Vec from top 50 hybrid seeds"
- These were placeholder comments, not actual implementation intent

**Actual Implementation:**
- Stanza dependency parsing extracts S-P-O triplets (Features 6, 7)
- Triplets stored with lemmatized text in full corpus format
- BM25 search over triplet corpus for expansion
- Working in three_layer_phi_retriever.py (_expand_via_graph_bm25)
- Validated in query_three_layer.py CLI tool

**Migration Impact:**
- Feature 15 marked as DEPRECATED
- Feature 17 documents active triplet-based approach
- AD2 updated to clarify triplet-based expansion
- Code comments should be updated to remove Node2Vec references

**Lessons Learned:**
- Placeholder comments can be misleading if not kept current
- Document ACTUAL implementation, not TODO intentions
- Clear deprecation notices prevent future confusion

- **Before:** Node2Vec mentioned in comments as placeholder/TODO for graph expansion
- **After:** Triplet-based BM25 expansion documented as active implementation. Node2Vec deprecated.

**4. Three-Layer phi-Retriever: Final Architecture with Dual Expansion + Dual Reranking**

- **Rationale:** COMPLETE SEARCH PROCESS DOCUMENTATION (2026-02-09)

The retriever uses a 3-layer cascade to find relevant academic papers:

LAYER 1 (Seeds): BM25 keyword search over 161K chunks. Returns 13 seeds.
Dense embeddings disabled at L1 (M2V too weak for domain-specific queries).

LAYER 2 (Expansion): Two independent paths expand from seeds:
  Path A - Graph BM25: Extract SPO triplets from seeds -> concatenate lemmas as mega-query -> BM25 over 161K triplet corpus -> map hits back to chunks -> GIST select for diversity
  Path B - Qwen3 Semantic: Mean-pool seed embeddings -> cosine similarity vs all 161K embeddings -> GIST select for diversity
  Both paths GIST-diversified (lambda=0.7), then RRF-merged -> 144 expansions.

WHY TWO PATHS: Graph BM25 finds structurally-adjacent knowledge (papers discussing same concepts through different lenses). Qwen3 finds semantically-similar knowledge (distant papers with related themes). Together they provide both local graph connectivity AND global semantic coverage.

LAYER 3 (Reranking + Paper Selection):
  ColBERTv2 + Cross-Encoder BOTH score ALL 157 candidates. RRF merges their full rankings.
  Walk down RRF-sorted list collecting unique paper_ids until top_k papers identified.
  Return ALL chunks/sections from those papers (not just best chunk).

WHY ALL SECTIONS: A paper's value comes from its complete context. Returning only the best-matching chunk loses surrounding methodology, related work, and implementation details that make the retrieval useful for research.

VALIDATED RESULT: 'agentic memory methods' -> 88 chunks from 13 papers. Breadth 8.5/10, Depth 8/10.
Coverage gaps (benchmarking, production scaling, security) are LITERATURE GAPS not retrieval gaps - these topics exist in different ArXiv neighborhoods than agentic memory research papers.

- **Before:** Layer 3 used GIST-before-reranking, limited to 1 chunk per paper, only 9 papers retrieved
- **After:** Layer 3 uses full-ranking RRF over ColBERTv2 + Cross-Encoder, paper walk-down selection, returns ALL sections from top-k papers. 88 chunks from 13 papers.

**5. PostgreSQL pgvector backend for Layer 2 vector search**

- **Rationale:** Hybrid storage architecture: mappings (chunk_to_triplets, triplet_to_chunks) remain in msgpack files for fast loading, while vectors stored in pgvector tables for scalable similarity search. Enables ECDF-weighted dual expansion without loading full corpus into memory.
- **Before:** File-based msgpack with in-memory BM25Okapi index and in-memory cosine similarity over qwen3_embeddings
- **After:** Mappings in msgpack files (fast loading), vectors in layer2_triplet_bm25 (JSONB sparse) and layer2_embeddings_256d (256d HNSW) tables. Query methods: query_layer2_triplet_bm25() and query_layer2_embeddings_256d().

### Prediction Accuracy

| Feature | Claim | Predicted F1 | Actual F1 | Error | Result |
|---------|-------|-------------|-----------|-------|--------|
| Runtime Label Mismatch Bug | Runtime matcher produces 45% label mismatches vs p... | — | — | — | CONFIRMED |
| Runtime Label Mismatch Bug | Empirical mismatch rates: Ex1=50%, Ex2=70.4%, Ex3=... | — | 0.4500 | — | CONFIRMED |
| Pre-computed Label Sparsity | Pre-computed labels would improve training (expect... | 0.4000 | 0.0670 | 0.3330 | FAILED |
| Pre-computed Label Sparsity | Label sparsity analysis: Ex1=32% coverage (11/34),... | — | 0.3000 | — | CONFIRMED |
| Class-Balanced Sampling | Class balancing will reduce O-token dominance and ... | 0.1800 | 0.1355 | 0.0445 | FAILED |
| More Training Data | 160 training examples will improve F1 vs 40 exampl... | 0.1500 | 0.1408 | 0.0092 | CONFIRMED |

---

## 🏷️ BIO Tagger — Feature Catalog

*Source: `bio_tagger_features.sqlite3` — 14 features, 4 architectural decisions, 1 claims*

| Status | Count |
|--------|-------|
| ✅ DONE | 13 |
| 🔄 IN PROGRESS | 1 |

### Features

#### 1. Class Balancing via Log-Transform Resampling 🔄 IN_PROGRESS

Implement log-transform resampling strategy to address catastrophic class imbalance (I-OBJ F1=0.88 vs B-tags F1=0.00-0.21). Log-transforms class counts, scales to max, and resamples per-epoch: I-SUBJ 2.19x, B-SUBJ 1.73x, B-OBJ 1.55x, B-PRED 1.44x, I-OBJ 1.0x. Epoch size increases from 1,103 to 1,665 samples (1.51x larger).

*Baseline F1: 0.6715*

#### 2. KeyError(9) Bug Fix - Label Iteration Scope ✅ DONE

Fixed critical bug in resampling implementation where code iterated through ALL 40+ label positions in multi-hot vectors instead of only first 6 BIO entity labels (B-SUBJ, I-SUBJ, B-PRED, I-PRED, B-OBJ, I-OBJ). Changed from enumerate(token_labels) to explicit range(6) iteration with bounds checking in both objective() and create_balanced_epoch_sampler() functions. Immediate crash prevented.

*Notes: Bug fixed in two locations: objective() line ~645 and create_balanced_epoch_sampler() line ~558. No KeyError on startup. Ready for testing.*

#### 3. Stanza Knowledge Distillation ✅ DONE

Extract BIO training labels from Stanza dependency parser (teacher model). Script: extract_bio_training_data.py. CLI: python extract_bio_training_data.py --chunks 250 --output <file>.msgpack. Parses sentences with Stanza dep parser, extracts nsubj/dobj/root triples, converts to multi-hot BIO labels per BERT token. Output msgpack keys: training_data, stats, tokenizer_name, label_names, architecture.

*Notes: Primary training data source. 250 chunks → ~1375 sentences.*

#### 4. BIOTagger Model Architecture ✅ DONE

BERT-base-uncased + 6 independent binary classifiers (nn.Linear(768,1) + sigmoid). Script: train_bio_tagger.py, class BIOTagger (line 93). Labels: B-SUBJ, I-SUBJ, B-PRED, I-PRED, B-OBJ, I-OBJ. forward() returns (probs, logits) tuple — probs shape [batch, seq_len, 6]. Model artifact: bio_tagger_best.pt (direct state_dict, 200+ keys).

*Notes: 6 independent heads allow multi-label tagging per token.*

#### 5. Optuna Hyperparameter Tuning ✅ DONE

Box-Cox resampled training with Optuna. Script: tune_bio_tagger.py, class BoxCoxBIODataset (line 44). CLI: python tune_bio_tagger.py --data <file>.msgpack --unfreeze-layers 12 --n-trials 21. Hyperparameter space: LR (1e-6 to 1e-3), dropout (0-0.5), O-weight (0.001-0.1), batch (4/8/16), optimizer (AdamW/Adam/SGD). Entity-density-based sample weighting. Best model saved to bio_tagger_best.pt.

*Notes: 21 trials on bio_training_250chunks_complete_FIXED.msgpack. All 12 BERT layers unfrozen.*

#### 6. BIOTripletExtractor Inference ✅ DONE

Production inference class for SPO triplet extraction. Script: inference_bio_tagger.py, class BIOTripletExtractor (line 123). Methods: extract_triplets(text, threshold) → list of (subj, pred, obj). Internal: extract_spans() parses BIO probabilities into Span objects, reconstruct_triplets() groups spans into S-P-O triples. Model loads bio_tagger_best.pt as direct state_dict.

*Notes: 4.08 chunks/sec on CUDA. 94.5% triplet extraction rate.*

#### 7. Post-Inference Stopword Removal ✅ DONE

Strip stopwords from extracted spans AFTER model inference (model trains on full boundaries). Script: inference_bio_tagger.py. SPAN_STOPWORDS (line ~71): articles (a/an/the), prepositions (to/of/in/on/at/by/for/with/from/into/through), conjunctions (and/or/but/nor), copulas (is/are/was/were/be/been/being), relative pronouns (that/which/who), pronouns (this/these/those/it/its), auxiliaries (has/have/had). clean_span_tokens(tokens, label_type): strips all stopwords from ALL span types including PRED. BUG FIXED: old code had fallback 'return cleaned if cleaned else tokens' which silently returned original tokens if model fired on a bare copula (is/are/has etc). Fix: removed fallback so copulas and auxiliaries are now stripped from PRED spans and can return empty list. Can return empty list for any label type — which are then skipped during triplet reconstruction. Stopwords covered for predicates: is, are, was, were, be, been, being, has, have, had, to, of, in, on, at, by, for, with, from.

*Notes: Fixed: removed fallback for all types. All span types now consistently strip stopwords and can return empty. Paired with training-time copula construction fix (see feature 14).*

#### 8. Cartesian Product Triplet Expansion ✅ DONE

Expand multi-word spans into atomic word-level triplets via cartesian product. Script: streamlit_bio_demo.py, function expand_cartesian_triplets() (line 177). Process: flatten all words per role → deduplicate → sorted set → itertools.product(S, P, O). Example: SUBJ=['humans'], PRED=['possess'], OBJ=['ability','tools'] → [('humans','possess','ability'), ('humans','possess','tools')]. Each atomic triplet becomes an edge in the knowledge graph.

*Notes: Produces atomic SPO triples suitable for graph construction.*

#### 9. Atomic Relation Graph ✅ DONE

NetworkX DiGraph visualization of SPO triplets. Script: streamlit_bio_demo.py, function render_triplet_graph() (line 195). SUBJ nodes (red, left column) → OBJ nodes (blue, right column), edges labeled by PRED. Multiple predicates between same S-O pair concatenated as comma-separated edge label. Shell layout: subjects on left column, objects on right column, centered vertically.

*Notes: Tab 1 visualization in Streamlit demo.*

#### 10. Augmented Ontological Knowledge Graph (AOKG) ✅ DONE

4-layer ANN-style knowledge graph with WordNet resolution. Script: streamlit_bio_demo.py. Layer 0 (bottom): Surface words — raw SPO edges from cartesian expansion. Layer 1: Lemmas — morphy-reduced via wn.morphy(), deduplicated. Layer 2: Synsets — first WordNet synset name, deduplicated. Layer 3 (top): Hypernyms — one level up via synset.hypernyms(), deduplicated. Functions: _resolve_word(word, role) (line 320) resolves L0→L3, render_layered_graph(S, P, O) (line 341) builds 4-layer visualization. Vertical dashed edges connect each node upward to its parent. Multiple words sharing the same lemma/synset/hypernym CONVERGE at that layer. LCA convergence level (0-3) serves as semantic distance metric for reranking.

*Notes: Named AOKG. LCA distance: 0=same word, 1=same lemma, 2=same synset, 3=same hypernym.*

#### 11. Streamlit BIO Tagger Demo ✅ DONE

3-tab interactive demo application. Script: streamlit_bio_demo.py (1064 lines). Tab 1 — Interactive Testing: paste text → extract triplets → view atomic graph + AOKG. Tab 2 — Eval Set Browser: browse evaluation samples from bio_training_250chunks_complete_FIXED.msgpack. Tab 3 — Eval Metrics: aggregate precision/recall/F1 across eval set. Launch: run_bio_tagger.bat app  OR  streamlit run streamlit_bio_demo.py. URL: http://localhost:8501.

*Notes: Menu-driven via run_bio_tagger.bat (4 options: app, test, integration, check).*

#### 12. Full Corpus Inference Benchmark ✅ DONE

Benchmark BIO inference throughput on full chunk corpus. Script: benchmark_full_corpus_inference.py. Data source: checkpoints/chunks.msgpack (161,389 chunks as list of dicts). Schema: [doc_id, paper_id, section_idx, chunk_idx, text]. Results (100 chunks): 4.08 chunks/sec, 22.26 sentences/sec, 21.04 triplets/sec, 94.5% extraction rate, 5.5 avg sentences/chunk, 5.2 avg triplets/chunk. Full corpus estimate: ~11 hours.

*Notes: Strategic pivot: too expensive for full corpus → use as reranker on hybrid-retrieved results only.*

#### 13. Workflow: BIO Tagger Execution Order ✅ DONE

Workflow Execution Order for BIO Tagger System:

1. **Extract Training Data**
   - Script: extract_bio_training_data.py
   - Purpose: Use Stanza to extract BIO labels from chunks via knowledge distillation
   - Command: python extract_bio_training_data.py --chunks 250 --output bio_training_250chunks.msgpack
   - Output: bio_training_250chunks_complete_FIXED.msgpack

2. **Tune Hyperparameters (Optional)**
   - Script: tune_bio_tagger.py
   - Purpose: Optuna-based hyperparameter optimization with BoxCox resampling
   - Command: python tune_bio_tagger.py --data bio_training_250chunks_complete_FIXED.msgpack --unfreeze-layers 12 --n-trials 21
   - Output: best_hyperparams.json, bio_tagger_best.pt

3. **Train Model**
   - Script: train_bio_tagger.py (or via tune_bio_tagger.py)
   - Purpose: Train 6 binary classifiers on BIO labels
   - Output: bio_tagger_best.pt (direct state_dict)

4. **Run Inference**
   - Script: inference_bio_tagger.py
   - Purpose: Extract SPO triplets from text using trained model
   - Import: from inference_bio_tagger import BIOTripletExtractor

5. **Launch Demo Application**
   - Script: streamlit_bio_demo.py
   - Purpose: 3-tab interactive demo (Testing, Eval Browser, Metrics)
   - Command: run_bio_tagger.bat app  OR  streamlit run streamlit_bio_demo.py
   - URL: http://localhost:8501

6. **Benchmark Performance (Optional)**
   - Script: benchmark_full_corpus_inference.py
   - Purpose: Measure throughput on full corpus
   - Command: python benchmark_full_corpus_inference.py --max-chunks 100
   - Data source: checkpoints/chunks.msgpack (161,389 chunks)

**Quick Start:**
run_bio_tagger.bat          # Interactive menu
run_bio_tagger.bat app      # Launch Streamlit
run_bio_tagger.bat test     # Run unit tests
run_bio_tagger.bat check    # System health check

#### 14. Copula Construction Fix (extract_spo_from_sentence) ✅ DONE

Training-time fix in extract_bio_training_data.py: extract_spo_from_sentence() now detects UD copula constructions via Stanza dependency parse and labels them correctly. Detection: when ROOT token is NOUN or ADJ and has a 'cop' dependency child, the nominal/adjectival ROOT is labelled B-PRED instead of the copula verb. Example: 'The bible is the word of God' — Stanza parse gives ROOT=word (NOUN), cop=is. Old behaviour: 'is' labelled B-PRED -> stripped by SPAN_STOPWORDS -> empty predicate -> triplet dropped. Fixed behaviour: 'word' labelled B-PRED -> triplet (bible, word, god). Implementation detail: cop_children = [ch for ch in token.children if ch.deprel == 'cop'] check on the ROOT token; if present, ROOT is used as predicate head and its nominal modifiers get I-PRED. Affects: predicate span boundary in training data. Requires retraining bio_tagger_atomic.pt to take effect. Works in tandem with clean_span_tokens PRED fix (feature 7) as defence-in-depth: training fix teaches correct predicate; inference fix handles edge cases if model misfires on a copula.

*Notes: Retraining required and in progress (2026-02-19). bio_training_data.msgpack (1,168 examples) generated from fixed extractor. bio_tagger_atomic.pt predates this fix; new model will replace it.*

### Architectural Decisions

**1. Post-inference stopword removal instead of training-time filtering**

- **Rationale:** Model should learn full BIO span boundaries including stopwords, then strip at output. Training on cleaned data caused boundary confusion. SPAN_STOPWORDS applied in clean_span_tokens() after span extraction.
- **Before:** Attempted stopword filtering in training data (extract_bio_training_data.py)
- **After:** Stopwords stripped post-inference in inference_bio_tagger.py:88

**2. Cartesian product expansion for atomic triplets**

- **Rationale:** Multi-word BIO spans like OBJ=['extraordinary','ability','create','utilize','tools'] need decomposition into atomic word-level triplets for graph construction. itertools.product(S,P,O) produces all combinations.
- **Before:** Multi-word spans as single graph nodes
- **After:** Atomic word-level nodes via expand_cartesian_triplets() in streamlit_bio_demo.py:177

**3. 4-layer AOKG with WordNet resolution for semantic distance**

- **Rationale:** LCA convergence level (0-3) provides a discrete semantic distance metric. Two words converging at L1 (same lemma) are closer than at L3 (same hypernym). Enables graph-based reranking signal complementary to embedding similarity.
- **Before:** Flat collapsed synset/hypernym graph
- **After:** 4-layer ANN-style graph: L0=surface, L1=lemma, L2=synset, L3=hypernym

**4. Use AOKG as precision reranker, not full corpus expansion**

- **Rationale:** Full corpus inference at 4.08 chunks/sec × 161k chunks = ~11 hours. Too expensive for recall expansion. Instead, use hybrid retrieval (BM25+dense) for recall, then apply AOKG graph on retrieved results only for precision reranking.
- **Before:** Plan to build AOKG over entire corpus for graph-based recall
- **After:** Strategic pivot: AOKG as reranker over hybrid-retrieved results only

---

## 🔀 Graph Transformer — Feature Catalog

*Source: `graph_transformer_feature_catalog.sqlite3` — 6 features, 2 architectural decisions, 0 claims*

| Status | Count |
|--------|-------|
| ✅ DONE | 1 |
| 📋 TODO | 5 |

### Features

#### 1. Workflow: Graph Transformer Execution Order ✅ DONE

Workflow Execution Order for Graph Transformer Reranking:

**Context:** Graph transformer reranking is applied AFTER hybrid retrieval to improve precision on candidate chunks. It does NOT operate on the full corpus (too expensive).

1. **Run Hybrid Retrieval**
   - Scripts: query_arxiv.py, query_quotes.py
   - Purpose: 8-stage GIST retrieval (BM25 + dense → diversity → RRF → filters)
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
- Expected improvement: Precision@5 boost via graph semantic distance

#### 2. Node2Vec Graph Embeddings from AOKG 📋 TODO

Extract graph embeddings from AOKG using karateclub.Node2Vec.

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
**Files:** train_graph_distillation.py (node2vec section)

#### 3. DistilBERT Graph Embedding Distillation 📋 TODO

Train lightweight model to predict graph embeddings directly from text.

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
**Files:** train_graph_distillation.py (distillation section)

#### 4. Full Corpus Graph Embedding Derivation 📋 TODO

Apply distilled model to derive graph embeddings for all 161,389 chunks.

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
**Files:** apply_graph_embeddings.py

#### 5. HNSW Index on Graph Embeddings 📋 TODO

Build HNSW index for fast approximate nearest neighbor search on graph embeddings.

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
**Files:** build_graph_hnsw.py

#### 6. Graph-Based Reranking of Hybrid Results 📋 TODO

Rerank hybrid-retrieved candidates using graph embeddings as third signal.

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
**Files:** graph_reranker.py, integrate with query_arxiv.py

### Architectural Decisions

**1. Use BERT BIO Tagger directly on full corpus (Node2Vec + Distillation rejected)**

- **Rationale:** ACTUAL IMPLEMENTATION: BERT-based BIO tagger was run directly on the full corpus (161,389 chunks). Initial estimate of 11 hours at 4.08 chunks/sec was pessimistic - actual runtime was less than 3 hours. Alternative approaches considered: 1. Node2Vec + DistilBERT distillation → Rejected: Unnecessary complexity, BERT direct is fast enough 2. Graph Transformer trained end-to-end → Deferred: Requires labeled data + complex implementation 3. Graph2Vec embeddings → Superseded by direct BERT extraction
- **Before:** Considered distillation to avoid "11-hour" BERT inference
- **After:** BERT direct on full corpus: <3 hours actual runtime, triplets extracted and stored in arxiv_graph_sparse.msgpack

**2. Full corpus graph construction completed (not just reranker)**

- **Rationale:** With <3 hour extraction time, full corpus graph construction became feasible. The msgpack artifacts (arxiv_graph_sparse.msgpack) contain triplets for all 161k chunks. Triplets extracted via build_arxiv_graph_sparse.py using BERT BIO tagger. Stored in msgpack format for fast loading. Graph can be used for both expansion (Layer 2 triplet BM25) and reranking. No HNSW index needed - BM25 over triplet corpus provides retrieval.
- **Before:** Planned graph-based reranking only on hybrid-retrieved candidates
- **After:** Full corpus graph available, used in Layer 2 triplet BM25 expansion

---

## 📁 Key Files

### Hybrid Retriever
| File | Purpose |
|------|---------|
| `gist_retriever.py` | Core GIST pipeline with 3-layer φ-scaled retrieval |
| `base_gist_retriever.py` | Abstract base class for retrievers |
| `arxiv_retriever.py` | ArXiv-specific retriever implementation |
| `quotes_retriever.py` | Quotes-specific retriever implementation |
| `pgvector_retriever.py` | PostgreSQL/pgvector backend |
| `arxiv_chunking_pipeline.py` | Document ingestion and chunking |
| `equidistant_chunking.py` | Log median + 2*MAD chunking algorithm |
| `build_adaptive_vocab.py` | BM25 vocabulary builder |
| `query_arxiv.py` | ArXiv query CLI |
| `query_quotes.py` | Quotes query CLI |
| `feature_catalog.py` | Feature catalog management functions |
| `retriever_feature_catalog.sqlite3` | Retriever feature tracking database |

### BIO Tagger
| File | Purpose |
|------|---------|
| `train_bio_tagger.py` | BIOTagger model definition (6 binary classifiers) |
| `tune_bio_tagger.py` | Optuna hyperparameter tuning with BoxCox resampling |
| `extract_bio_training_data.py` | Stanza knowledge distillation → BIO labels |
| `inference_bio_tagger.py` | BIOTripletExtractor: production inference + stopword removal |
| `streamlit_bio_demo.py` | 3-tab Streamlit demo (Interactive, Eval, Metrics) |
| `benchmark_full_corpus_inference.py` | Throughput benchmark on full chunk corpus |
| `run_bio_tagger.bat` | Windows launcher (app / test / integration / check) |
| `bio_tagger_best.pt` | Trained model weights (direct state_dict) |
| `bio_tagger_features.sqlite3` | BIO tagger feature tracking database |
| `checkpoints/chunks.msgpack` | Source corpus: 161,389 chunks |
| `bio_training_250chunks_complete_FIXED.msgpack` | Training/eval data (250 chunks) |

### Graph Transformer
| File | Purpose |
|------|---------|
| `GRAPH_TRANSFORMER_STRATEGY.md` | Comprehensive technical strategy (350 lines) |
| `graph_transformer_feature_catalog.sqlite3` | Graph transformer feature tracking database |
| `train_graph_distillation.py` | **(TODO)** Node2Vec + DistilBERT distillation training |
| `apply_graph_embeddings.py` | **(TODO)** Fast embedding derivation for full corpus |
| `build_graph_hnsw.py` | **(TODO)** HNSW index on graph embeddings |
| `graph_reranker.py` | **(TODO)** Graph-based reranking integration |

---

## 📊 Performance

### Hybrid Retriever
| Metric | ArXiv (172k chunks) | Quotes (3.5k chunks) |
|--------|--------------------|--------------------|
| Ingest time | ~15 min | ~30 sec |
| Query latency | < 2 sec | < 1 sec |
| Storage | ~2 GB pgvector | ~50 MB pgvector |

### BIO Tagger Inference (CUDA)
| Metric | Value |
|--------|-------|
| Throughput | 4.08 chunks/sec |
| Sentences/sec | 22.26 |
| Triplets/sec | 21.04 |
| Extraction rate | 94.5% |
| Avg sentences/chunk | 5.5 |
| Avg triplets/chunk | 5.2 |
| Full corpus (161k) estimate | ~11 hours |

---

## 🧠 AOKG — Augmented Ontological Knowledge Graph

4-layer ANN-style knowledge graph built from extracted SPO triplets using WordNet resolution.

```
Layer 3 (top):  Hypernyms     ─── entity.n.01 ── physical_entity.n.01
                                      │                  │
Layer 2:        Synsets        ─── dog.n.01 ──────── cat.n.01
                                      │                  │
Layer 1:        Lemmas         ─── dog ──────────── cat
                                      │                  │
Layer 0 (base): Surface Words  ─── dogs ─────────── cats
```

### Semantic Distance via LCA
| Convergence Level | Meaning | Example |
|-------------------|---------|---------|
| 0 | Same surface word | "dogs" vs "dogs" |
| 1 | Same lemma | "dogs" vs "dog" |
| 2 | Same synset | "dog" vs "hound" |
| 3 | Same hypernym | "dog" vs "cat" (both → animal) |

### Resolution Pipeline
1. **L0 → L1:** `nltk.corpus.wordnet.morphy(word)` for lemma reduction
2. **L1 → L2:** First WordNet synset via `wn.synsets(lemma)`
3. **L2 → L3:** One-level hypernym via `synset.hypernyms()`
4. **Convergence:** Multiple words sharing a parent node MERGE at that layer

### Use Case
Precision reranker on hybrid-retrieved results. AOKG is NOT applied to the full corpus (~11 hours too expensive). Instead:
1. Hybrid retriever fetches candidate chunks (BM25 + dense)
2. BIO tagger extracts SPO triplets from retrieved chunks only
3. AOKG graph built on-the-fly for retrieved results
4. LCA convergence level provides reranking signal

---

## ⚙️ Configuration

### PostgreSQL (for Hybrid Retriever)
```python
# pgvector_retriever.py — PGVectorConfig dataclass
host = "localhost"
port = 5432
dbname = "arxiv_retrieval"  # or "quotes_retrieval"
user = "postgres"
embedding_dim = 64          # model2vec PCA-reduced
```

### BIO Tagger
```python
# Key files and paths
model_path = "bio_tagger_best.pt"           # Trained model
training_data = "bio_training_250chunks_complete_FIXED.msgpack"
source_chunks = "checkpoints/chunks.msgpack"  # 161,389 chunks

# Model architecture (train_bio_tagger.py)
bert_model = "bert-base-uncased"
num_labels = 6  # B-SUBJ, I-SUBJ, B-PRED, I-PRED, B-OBJ, I-OBJ
```

### GIST Pipeline Parameters
```python
bm25_candidates = 100       # Initial BM25 pool size
dense_candidates = 100      # Initial dense pool size
gist_k = 30                 # GIST diversity selection per pool
rrf_k = 60                  # RRF fusion constant
final_top_k = 5             # Final sections returned
```

---

## 📝 Feature Catalog Management

Both subsystems track features, claims, and architectural decisions in SQLite databases using `feature_catalog.py`.

```bash
# List features
python -c "from feature_catalog import list_features; list_features('feature_catalog.sqlite3')"
python -c "from feature_catalog import list_features; list_features('bio_tagger_features.sqlite3')"

# Regenerate this README
python generate_readme.py

# Populate missing features (INSERT OR IGNORE)
python populate_feature_dbs.py
```

### Database Schema (shared by both DBs)
| Table | Columns | Purpose |
|-------|---------|---------|
| `features` | name, description, status, f1_baseline, f1_current | Track implementation status |
| `claims` | claim_text, predicted_f1, actual_f1, prediction_error | Prediction accountability |
| `architectural_decisions` | decision, rationale, before/after state | Design rationale history |

---

## 📄 License

MIT

---

*Generated 2026-02-21 17:46 by `generate_readme.py`*
