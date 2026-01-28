# GIST Hybrid Retrieval System

**Production-ready GIST retrieval system with PostgreSQL/pgvector backend supporting multiple document types (ArXiv papers, quotes).**

## 🎯 System Overview

### Architecture
```
Document Sources
  ├─ ArXiv Papers (Markdown) → Section-based grouping
  └─ Quotes Dataset (HuggingFace) → Quote-level grouping
    ↓
[Equidistant Chunking] → Log median + 2*MAD threshold
    ↓
[Checkpointing] → msgpack cache (incremental)
    ↓
[Dual Indexing]
    ├─ Dense: model2vec (256→64 PCA)
    └─ Sparse: BM25 (BERT WordPiece tokenizer)
    ↓
[PostgreSQL + pgvector]
    ├─ vector(64) IVFFlat + cosine
    └─ sparsevec(vocab_size)
    ↓
[GIST Pipeline] → 7-Stage Retrieval
  1. Parallel Retrieval (BM25 + Dense) at CHUNK level
  2. GIST Selection on BM25 pool (diversity)
  3. GIST Selection on Dense pool (diversity)
  4. RRF Fusion of chunks
  5. GROUP chunks + RECONSTRUCT full text
  6. ColBERT Late Interaction on FULL TEXT
  7. Cross-Encoder Reranking on FULL TEXT
```

### Key Features

#### ✅ GIST Retrieval Pipeline
**7-stage cascading pipeline for optimal relevance-diversity tradeoff**:
1. **Parallel Retrieval**: BM25 + Dense at chunk level (top_k²)
2. **GIST Selection BM25**: Greedy diversity on BM25 pool (λ=0.7)
3. **GIST Selection Dense**: Greedy diversity on dense pool (λ=0.7)
4. **RRF Fusion**: Combine pools with reciprocal rank fusion
5. **Group Reconstruction**: Group chunks by parent ID, fetch ALL chunks, reconstruct full text
6. **ColBERT Reranking**: Late interaction on FULL TEXT (token-level MaxSim)
7. **Cross-Encoder Reranking**: Final scoring on FULL TEXT

**Key Innovation**: Chunking for retrieval, reconstruction before neural reranking
- Retrieve granular chunks for precision
- Reconstruct complete context for neural models
- GIST ensures diversity (vs MMR pure relevance)

#### ✅ Multiple Document Types
**Unified GIST interface with document-specific implementations**:
- **ArXiv Papers**: Groups by `(paper_id, section_idx)` → reconstructs full sections
- **Quotes**: Groups by `quote_id` → reconstructs full quotes from chunks

#### ✅ BERT WordPiece Tokenization
**Aligned lexical and semantic spaces**:
- Uses `bert-base-uncased` tokenizer for BM25
- Produces subword tokens (`##ing`, `##tion`)
- Vocabulary aligned with dense embedding space
- **Benefits**: Improved hybrid retrieval coherence over regex tokenization
- **ArXiv**: 114,523 tokens | **Quotes**: 5,445 tokens

#### ✅ Equidistant Chunking
**Robust threshold-based chunking with log statistics**:
- **Method**: Compute log median + MAD * 2 on document lengths
- **ArXiv**: Threshold on section content → large sections split
- **Quotes**: Threshold on quote text → long quotes split
- **SlidingAggregator**: Character-based overlap control for splits
- **Result**: Preserves small items whole, chunks large items consistently

#### ✅ Neural Reranking
**Two-stage reranking on reconstructed full text**:
- **ColBERT**: Token-level late interaction (MaxSim)
  - Model: `bert-base-uncased` with linear projection
  - Scoring: Σᵢ maxⱼ dot(qᵢ, dⱼ) per query token
  - Fast GPU batch processing
- **Cross-Encoder**: Full attention query-document pairs
  - Model: `cross-encoder/ms-marco-MiniLM-L-6-v2`
  - Direct relevance scoring
  - Final ranking authority

#### ✅ Efficient Data Checkpointing
**msgpack-based caching** for fast restart/resume:
```
checkpoints/
├── chunks.msgpack          # Processed chunks
├── bm25_index.msgpack      # BM25 vocab + IDF + avgdl
├── embeddings.npy          # Dense vectors (64-dim)
└── sparse_vectors.msgpack  # Sparse BM25 vectors
```

**Pattern**: 
1. Check if checkpoint exists → load
2. Else: compute → save checkpoint
3. Next run: instant reload from checkpoint

#### ✅ Incremental Loading
**Smart checkpoint management**: Only processes documents not already in PostgreSQL
- Query existing `paper_id` or `quote_id` from database
- Filter chunks to skip processed documents
- Spot-add new documents without full rebuild
- **Performance**: <5s vs 36min for no-op runs (288x speedup)

---

## 📊 Performance Metrics

### ArXiv Dataset

| Metric | Value | Notes |
|--------|-------|-------|
| Papers | 1,607 | Post-processed markdown |
| Chunks | 172,272 | After equidistant chunking |
| Avg chunks/paper | 107 | Variable (depends on section sizes) |
| Embedding dim | 64 | PCA from 256 |
| Sparse vocab | 114,523 | BERT WordPiece tokens |
| Database size | ~500 MB | Including indexes |
| **Build time (full)** | 36 min | All papers |
| **Build time (incremental)** | <5 s | No new papers |
| **Search time** | <500 ms | Full GIST pipeline |

### Quotes Dataset

| Metric | Value | Notes |
|--------|-------|-------|
| Quotes | 2,508 | HuggingFace abirate/english_quotes |
| Chunks | 3,516 | After equidistant chunking |
| Quotes > threshold | 405 | Exceeded 229 char threshold |
| Chunking threshold | 229 chars | log median + 2*MAD |
| Chunk target | 98 chars | SlidingAggregator target |
| Embedding dim | 64 | PCA from 256 |
| Sparse vocab | 5,445 | BERT WordPiece tokens |
| **Build time** | ~30 s | Full index creation |
| **Search time** | <500 ms | Full GIST pipeline |

### GIST Pipeline Stages (Example: top_k=21)

| Stage | Input | Output | Operation |
|-------|-------|--------|-----------|
| 1. Retrieve | Query | 625+625 chunks | Parallel BM25 + Dense |
| 2. GIST BM25 | 625 chunks | 21 chunks | Greedy diversity (λ=0.7) |
| 3. GIST Dense | 625 chunks | 21 chunks | Greedy diversity (λ=0.7) |
| 4. RRF | 42 chunks | 20 chunks | Reciprocal rank fusion |
| 5. Group | 20 chunks | 13 groups | Reconstruct full text |
| 6. ColBERT | 13 groups | 5 groups | Token MaxSim on full text |
| 7. Cross-Encoder | 5 groups | 3 groups | Attention scoring |

---

## 🚀 Quick Start

### Installation

```bash
# Core dependencies
pip install psycopg2-binary pgvector numpy tqdm model2vec msgpack scipy

# Neural reranking (optional but recommended)
pip install torch sentence-transformers transformers

# For quotes dataset
pip install datasets

# Verify PostgreSQL connection
python test_pg_connection.py
```

### ArXiv Papers

#### 1. Build Index (Incremental - Default)
```bash
python pgvector_retriever.py --build
```

**What happens:**
- Loads chunks from checkpoint (fast)
- Queries PostgreSQL for existing papers
- **Skips papers already in database**
- Processes ONLY new papers

#### 2. Search ArXiv
```bash
python pgvector_retriever.py --search "transformer attention mechanism" --top_k 5
```

**Output:**
```
GIST Pipeline: retrieve 625 chunks → GIST 21 → group 13 → ColBERT 5 → Cross-Encoder 3

1. [2410_05258] Section 1
   Score: 8.4521
   Chunks: 8
   Preview: Transformer has garnered significant research...
```

#### 3. Clear ArXiv Database
```bash
python pgvector_retriever.py --clear
```

### Quotes Dataset

#### 1. Build Quotes Index
```bash
python quotes_retriever.py --build --quotes_file quotes_dataset.json --reset
```

**What happens:**
- Downloads `abirate/english_quotes` from HuggingFace (if not exists)
- Applies equidistant chunking (log median + 2*MAD)
- Creates PostgreSQL table with BERT WordPiece BM25

#### 2. Search Quotes
```bash
python quotes_retriever.py --search "wisdom and knowledge" --top_k 5
```

**Output:**
```
1. quote_001475
   Score: 5.9874
   Chunks: 1
   Preview: "Knowledge speaks, but wisdom listens"

2. quote_001561
   Score: 1.0183
   Chunks: 1
   Preview: "Wisdom comes from experience..."
```

---

## 📁 File Structure

### Core Modules

```
gist_retriever.py                 # Abstract GIST pipeline base class
├── GISTRetriever                 # 7-stage pipeline implementation
├── GIST Selection Algorithm      # Greedy diversity with λ tradeoff
├── ColBERTScorer                 # Token-level late interaction
├── CrossEncoderScorer            # Full attention reranking
└── RetrievedGroup                # Grouped result container

pgvector_retriever.py             # ArXiv papers implementation
├── TextPreprocessor              # BERT WordPiece tokenizer
├── SparseBM25Index               # BM25 with msgpack cache
├── Model2VecEmbedder             # PCA compression (256→64)
├── PGVectorRetriever             # PostgreSQL GIST backend
└── Group by: (paper_id, section_idx)

quotes_retriever.py               # Quotes dataset implementation
├── TextPreprocessor              # BERT WordPiece tokenizer
├── SparseBM25Index               # BM25 with msgpack cache
├── Model2VecEmbedder             # PCA compression (256→64)
├── QuotesRetriever               # PostgreSQL GIST backend
└── Group by: (quote_id,)

equidistant_chunking.py           # Robust threshold-based chunking
├── SlidingAggregator             # Character-based overlap control
├── create_paragraph_blocks       # Extract paragraphs from markdown
└── Log median + MAD * 2          # Threshold calculation
```

### Legacy/Archive Scripts

```
archive/
├── hybrid_pgvector.py            # Original ArXiv implementation
├── arxiv_chunking_pipeline.py   # Section-based chunking
├── build_arxiv_rrf_index.py     # Legacy build script
└── rrf_retriever.py              # Pre-GIST RRF implementation
```

### Supporting Scripts

```
test_pg_connection.py             # Verify database connection
check_pg_extensions.py            # Verify pgvector support
```

### Data Files

```
papers/post_processed/            # ArXiv: 1,607 markdown files
quotes_dataset.json               # Quotes: 2,508 entries
checkpoints/                      # msgpack cache
  ├── chunks.msgpack
  ├── bm25_index.msgpack
  ├── embeddings.npy
  └── sparse_vectors.msgpack
bm25_vocab.msgpack                # ArXiv BM25 vocab (for search)
quotes_bm25_vocab.msgpack         # Quotes BM25 vocab (for search)
```

---

## 🔧 Configuration

### PostgreSQL Connection

```python
# ArXiv
@dataclass
class PGVectorConfig(GISTConfig):
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "langchain"
    db_user: str = "langchain"
    db_password: str = "langchain"
    table_name: str = "arxiv_chunks"
    
    # GIST pipeline
    fibonacci_sequence: List[int] = [3, 5, 8, 13, 21, 34, 55, 89]
    gist_lambda: float = 0.7        # Utility vs diversity tradeoff
    rrf_k: int = 60                 # RRF constant
    use_colbert: bool = True
    use_cross_encoder: bool = True
    
    # Embeddings
    embedding_dim: int = 64
    embedding_model: str = "minishlab/M2V_base_output"
    
    # BM25
    bm25_k1: float = 1.5
    bm25_b: float = 0.75

# Quotes
@dataclass
class QuotesConfig(GISTConfig):
    db_host: str = "192.168.3.18"
    db_port: int = 6024
    table_name: str = "quotes_chunks"
    # ... (same GIST/embedding/BM25 settings)
```

### GIST Pipeline Parameters

```python
# λ (lambda): Utility vs Diversity Tradeoff
gist_lambda = 0.7
# - 1.0: Pure utility (no diversity)
# - 0.7: Slight preference for relevance (default)
# - 0.5: Equal utility and diversity
# - 0.0: Pure diversity (ignore relevance)

# Fibonacci Cascade
fibonacci_sequence = [3, 5, 8, 13, 21, 34, 55, 89]
# Used to progressively reduce candidates through pipeline stages
# Example: top_k=21 → retrieve 625, GIST 21, group 13, ColBERT 5, CE 3

# RRF Constant
rrf_k = 60  # Standard RRF constant for rank fusion
```

### BM25 Parameters

```python
# BERT WordPiece Tokenization
tokenizer: "bert-base-uncased"
min_df: 2                        # Min document frequency

# Scoring
k1: 1.5                          # Term saturation
b: 0.75                          # Length normalization
```

### Embedding Parameters

```python
model: "minishlab/M2V_base_output"
native_dim: 256
target_dim: 64                   # PCA reduction
batch_size: 256
explained_variance: ~74%         # After PCA
```

### Chunking Parameters

```python
# Equidistant Chunking (Log Threshold)
# Computed per corpus:
threshold = exp(log_median + mad_multiplier * log_mad)

# ArXiv example:
#   log_median = 6.8, log_mad = 0.5, mad_multiplier = 2
#   threshold = exp(6.8 + 2*0.5) = exp(7.8) ≈ 2440 chars

# Quotes example:
#   Computed threshold: 229 chars
#   Target chunk size: 98 chars
#   Tolerance: 76 chars

# SlidingAggregator settings:
target_chars: 98
tolerance: 76
overlap_chars: 0              # For below-target paragraphs
```

---

## 🎓 Technical Details

### Incremental Loading Pattern

**Core Logic:**
```python
# 1. Query existing papers
with HybridPGVector(config) as db:
    existing_papers = db.get_existing_paper_ids()  # Set[str]

# 2. Filter chunks
if not reset:
    chunks = [c for c in chunks if c.doc_id not in existing_papers]

# 3. Early exit if nothing to do
if len(chunks) == 0:
    print("Database is up to date.")
    return
```

**Database Query:**
```sql
SELECT DISTINCT paper_id FROM arxiv_hybrid_index;
```

**Benefits:**
- Avoid redundant computation
- Fast continuation (<5s for no-op)
- Safe: Uses `ON CONFLICT` for upserts

### Chunking Strategy Comparison

| Feature | Equidistant | Section-Level |
|---------|-------------|---------------|
| **Atomic unit** | Paragraphs | Sections |
| **Overlap control** | 0% below-target | N/A |
| **Semantic** | Partial | Full |
| **Use case** | General RAG | ArXiv papers |
| **Config** | Sample-based | Corpus stats |

### msgpack Checkpointing

**Why msgpack?**
- Faster than pickle for structured data
- Smaller file size (4.3 MB vs ~8 MB)
- Cross-platform compatibility

**Pattern:**
```python
# Save
with open('checkpoint.msgpack', 'wb') as f:
    # Convert int keys to strings (msgpack limitation)
    data = {str(k): v for k, v in data.items()}
    msgpack.pack(data, f)

# Load
with open('checkpoint.msgpack', 'rb') as f:
    data = msgpack.unpack(f)
    # Convert back to ints
    data = {int(k): v for k, v in data.items()}
```

### PostgreSQL Schema

```sql
CREATE TABLE arxiv_hybrid_index (
    id SERIAL PRIMARY KEY,
    chunk_id TEXT UNIQUE,           -- For ON CONFLICT upserts
    paper_id TEXT,                   -- For incremental filtering
    section_idx INTEGER,
    chunk_idx INTEGER,
    content TEXT,
    embedding vector(64),            -- Dense vector
    bm25_sparse sparsevec(114523)   -- Sparse vector (1-based indexing)
);

-- Indexes
CREATE INDEX idx_embedding ON arxiv_hybrid_index 
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 415);

CREATE INDEX idx_paper_id ON arxiv_hybrid_index (paper_id);
-- Note: Sparse uses brute force (sparsevec_ip_ops not supported by IVFFlat)
```

### RRF Fusion Algorithm

```python
def rrf_score(dense_rank, sparse_rank, k=60):
    """Reciprocal Rank Fusion."""
    return 1/(k + dense_rank) + 1/(k + sparse_rank)

# Example:
# Dense rank 5, Sparse rank 10:
#   1/(60+5) + 1/(60+10) = 0.0154 + 0.0143 = 0.0297
```

---

## 🔍 Usage Patterns

### Daily Workflow

```bash
# Check for new papers
python hybrid_pgvector.py --build

# If no new papers: exits in <5s
# If new papers: processes only new ones (~30s per 10 papers)
```

### Adding New Papers

```bash
# 1. Add markdown files to papers/post_processed/
cp new_paper.md papers/post_processed/

# 2. Delete chunks checkpoint to force re-chunking
rm checkpoints\chunks.pkl

# 3. Run incremental build
python hybrid_pgvector.py --build

# Result: Only new papers processed
```

### Parameter Changes

```bash
# If you modified BM25 (k1, b) or embedding_dim:
python hybrid_pgvector.py --build --reset

# Full rebuild (~36 minutes)
```

### Research Query

```bash
# Search with default settings (top 13 sections)
python hybrid_pgvector.py --search "attention mechanism"

# Returns:
# - Section-level results (not individual chunks)
# - RRF scores (fusion of dense + sparse)
# - Preview of first chunk per section
```

---

## 🐛 Troubleshooting

### Issue: "Database is up to date" but want to rebuild
**Solution:**
```bash
python hybrid_pgvector.py --build --reset
```

### Issue: Import takes too long
**Cause:** Removed transformers dependency  
**Status:** ✅ Fixed (regex tokenizer, instant startup)

### Issue: sparsevec index error
**Cause:** sparsevec_ip_ops not supported by IVFFlat  
**Status:** ✅ Fixed (removed sparse index, using brute force)

### Issue: Token ID out of bounds
**Cause:** PostgreSQL sparsevec uses 1-based indexing  
**Status:** ✅ Fixed (vocab uses idx+1)

### Issue: Msgpack int key error
**Cause:** msgpack strict_map_key=True doesn't allow int keys  
**Status:** ✅ Fixed (convert to string in save, back to int in load)

### Issue: Search returns no results
**Cause:** BM25 vocab file missing  
**Solution:** Run `--build` (will copy bm25_vocab.pkl)

### Issue: PostgreSQL connection error
**Cause:** Server not accessible  
**Solution:** Check host, port, firewall, verify with `test_pg_connection.py`

---

## 📊 Database Stats

### Current Data
```
Papers: 1,607
Chunks: 172,272
Avg chunks/paper: 107
Embedding dimension: 64
Sparse vocabulary: 114,523 tokens
Database size: ~500 MB (including indexes)
```

### Index Stats
```
Dense (IVFFlat):
  - Lists: 415
  - Distance: Cosine
  - Build time: ~2 minutes

Sparse (Brute Force):
  - No index (sparsevec_ip_ops not supported)
  - Query time: <300ms (acceptable)

Paper ID (btree):
  - For filtering by paper_id
  - Used in incremental loading
```

---

## 🔬 Advanced Topics

### Custom Chunking Strategy

```python
from arxiv_chunking_pipeline import ArxivChunkingPipeline

# Initialize with custom config
pipeline = ArxivChunkingPipeline(
    papers_dir="path/to/papers",
    target_paragraphs=3,          # Chunk size control
    min_section_chars=500,        # Section merging threshold
)

chunks = pipeline.process()

# Access chunk metadata
for chunk in chunks:
    print(f"Paper: {chunk.doc_id}")
    print(f"Section: {chunk.section_title}")
    print(f"Position: {chunk.section_idx}/{chunk.chunk_idx}")
```

### Direct PostgreSQL Access

```python
from hybrid_pgvector import HybridPGVector, HybridConfig

config = HybridConfig()

with HybridPGVector(config) as db:
    # Get existing papers
    papers = db.get_existing_paper_ids()
    
    # Custom query
    with db.conn.cursor() as cur:
        cur.execute("""
            SELECT paper_id, COUNT(*) as chunks
            FROM arxiv_hybrid_index
            GROUP BY paper_id
            ORDER BY chunks DESC
            LIMIT 10
        """)
        for row in cur.fetchall():
            print(f"{row[0]}: {row[1]} chunks")
```

### Modify RRF Parameters

```python
# In search() function
sections = db.search_sections_rrf(
    query_embedding,
    query_sparse,
    top_sections=20,      # Return top 20 instead of 13
    rrf_k=100            # Change RRF constant
)
```

---

## � Requirements

### Functional Requirements

| ID | Requirement | Implementation | Status |
|----|-------------|----------------|--------|
| **FR-01** | Multiple document types support | ArXiv + Quotes retrievers via abstract GISTRetriever | ✅ |
| **FR-02** | BERT WordPiece tokenization for BM25 | TextPreprocessor with bert-base-uncased | ✅ |
| **FR-03** | Equidistant chunking with log statistics | Log median + MAD * 2 threshold | ✅ |
| **FR-04** | Chunk reconstruction for neural reranking | _group_and_reconstruct() fetches all chunks | ✅ |
| **FR-05** | GIST diversity selection | gist_select() with λ=0.7 tradeoff | ✅ |
| **FR-06** | ColBERT late interaction scoring | Token-level MaxSim on full text | ✅ |
| **FR-07** | Cross-encoder final reranking | ms-marco-MiniLM-L-6-v2 on full text | ✅ |
| **FR-08** | Incremental index building | Skip existing paper_id/quote_id | ✅ |
| **FR-09** | msgpack checkpointing | Cache chunks, embeddings, BM25 | ✅ |
| **FR-10** | HuggingFace dataset integration | Auto-download abirate/english_quotes | ✅ |

### Non-Functional Requirements

| ID | Requirement | Target | Actual | Status |
|----|-------------|--------|--------|--------|
| **NFR-01** | Search latency (full GIST pipeline) | <1s | <500ms | ✅ |
| **NFR-02** | Incremental build latency (no-op) | <10s | <5s | ✅ |
| **NFR-03** | PCA explained variance | >70% | 74.1% | ✅ |
| **NFR-04** | Chunking threshold robustness | MAD-based | log median + 2*MAD | ✅ |
| **NFR-05** | Token alignment (BM25 vs embedding) | Shared vocab | BERT tokenizer both | ✅ |
| **NFR-06** | Database index efficiency | <500ms query | IVFFlat cosine | ✅ |
| **NFR-07** | Checkpoint load time | <10s | <2s (msgpack) | ✅ |
| **NFR-08** | GPU memory usage (ColBERT+CE) | <4GB | <2GB typical | ✅ |

### Technical Requirements

| ID | Requirement | Implementation | Status |
|----|-------------|----------------|--------|
| **TR-01** | PostgreSQL 16+ with pgvector 0.8.0+ | IVFFlat + sparsevec support | ✅ |
| **TR-02** | Python 3.10+ | Type hints, dataclasses | ✅ |
| **TR-03** | model2vec for dense embeddings | M2V_base_output 256→64 PCA | ✅ |
| **TR-04** | transformers for tokenization | AutoTokenizer bert-base-uncased | ✅ |
| **TR-05** | sentence-transformers for reranking | CrossEncoder + BERT base | ✅ |
| **TR-06** | Abstract base class for extensibility | GISTRetriever ABC | ✅ |
| **TR-07** | Fibonacci cascade for stage sizing | get_fibonacci_lower() | ✅ |
| **TR-08** | Reciprocal rank fusion | compute_rrf_score(k=60) | ✅ |

### Data Requirements

| ID | Requirement | ArXiv | Quotes | Status |
|----|-------------|-------|--------|--------|
| **DR-01** | Document count | 1,607 papers | 2,508 quotes | ✅ |
| **DR-02** | Chunk count after equidistant split | 172,272 | 3,516 | ✅ |
| **DR-03** | BM25 vocabulary size | 114,523 tokens | 5,445 tokens | ✅ |
| **DR-04** | Embedding dimension | 64 (PCA) | 64 (PCA) | ✅ |
| **DR-05** | Grouping key | (paper_id, section_idx) | (quote_id,) | ✅ |
| **DR-06** | Database table schema | chunk_id, paper_id, section_idx, chunk_idx, content, embedding, bm25_sparse | chunk_id, quote_id, chunk_idx, content, embedding, bm25_sparse | ✅ |

### Algorithm Requirements

| ID | Algorithm | Formula | Implementation | Status |
|----|-----------|---------|----------------|--------|
| **AR-01** | GIST Selection | score(d) = λ*utility(d) - (1-λ)*max_sim(d,S) | gist_select() | ✅ |
| **AR-02** | RRF Fusion | 1/(k+rank₁) + 1/(k+rank₂) | compute_rrf_score() | ✅ |
| **AR-03** | BM25 Scoring | IDF(q) * (f(q,D) * (k₁+1)) / (f(q,D) + k₁*(1-b+b*|D|/avgdl)) | SparseBM25Index | ✅ |
| **AR-04** | ColBERT Scoring | Σᵢ maxⱼ cosine(qᵢ, dⱼ) | ColBERTScorer.score() | ✅ |
| **AR-05** | Log Threshold | exp(log_median + mad_multiplier * log_mad) | equidistant_chunking.py | ✅ |
| **AR-06** | PCA Reduction | fit(256) → transform(64) | Model2VecEmbedder | ✅ |

---

## 📚 References

### Papers & Algorithms

- **GIST/MMR**: Carbonell & Goldstein (1998) - Maximal Marginal Relevance
- **BM25**: Robertson & Zaragoza (2009) - Okapi BM25
- **RRF**: Cormack et al. (2009) - Reciprocal Rank Fusion
- **ColBERT**: Khattab & Zaharia (2020) - Late Interaction Retrieval
- **model2vec**: Minish Lab (2024) - Static Embedding Distillation
- **Cross-Encoder**: Reimers & Gurevych (2019) - Sentence-BERT

### Components Used

- **model2vec**: https://github.com/MinishLab/model2vec
- **pgvector**: https://github.com/pgvector/pgvector
- **sentence-transformers**: https://www.sbert.net/
- **transformers**: https://huggingface.co/transformers/

### Datasets

- **ArXiv papers**: https://arxiv.org (1,607 markdown files)
- **Quotes**: HuggingFace abirate/english_quotes (2,508 entries)

---

## 📝 License & Credits

**Author**: Created for multi-domain GIST retrieval research  
**Status**: Production-ready (January 2026)  
**Database**: PostgreSQL 16 + pgvector 0.8.0  
**Python**: 3.10+

---

## 🎉 Quick Reference

### Most Common Commands

```bash
# ArXiv: Daily check for new papers
python pgvector_retriever.py --build

# ArXiv: Search
python pgvector_retriever.py --search "your query" --top_k 5

# ArXiv: Full rebuild
python pgvector_retriever.py --build --reset

# Quotes: Build index
python quotes_retriever.py --build --quotes_file quotes_dataset.json --reset

# Quotes: Search
python quotes_retriever.py --search "wisdom" --top_k 5
```

### Key Numbers

**ArXiv:**
- **172,272 chunks** from 1,607 papers
- **<5s** incremental check (vs 36min full rebuild)
- **<500ms** GIST pipeline query
- **114,523** BERT WordPiece vocab

**Quotes:**
- **3,516 chunks** from 2,508 quotes
- **405 quotes** exceeded 229 char threshold
- **5,445** BERT WordPiece vocab
- **~30s** full index build

**GIST Pipeline:**
- **7 stages**: Retrieve → GIST(BM25) → GIST(Dense) → RRF → Group → ColBERT → Cross-Encoder
- **λ=0.7**: Slight preference for relevance over diversity
- **k=60**: RRF constant for rank fusion

---

**System Status**: ✅ PRODUCTION READY  
**Last Updated**: January 27, 2026  
**Primary Database**: 192.168.3.18:6024 (langchain/langchain)  
**Features**: GIST retrieval, BERT tokenization, equidistant chunking, neural reranking

1. **Chunk-level incremental loading**
   - Current: Paper-level tracking
   - Proposed: Track individual chunk_id
   - Benefit: Finer-grained updates

2. **Timestamp-based updates**
   - Track file modification times
   - Auto-detect changed papers
   - Requires: Add mtime column to schema

3. **Dry-run mode**
   ```bash
   python hybrid_pgvector.py --build --dry-run
   ```
   - Show what would be processed
   - Estimated time/cost

4. **Sparse vector indexing**
   - Current: Brute force
   - Waiting: pgvector support for sparsevec IVFFlat
   - Tracking: https://github.com/pgvector/pgvector/issues

5. **API wrapper**
   ```python
   from fastapi import FastAPI
   app = FastAPI()
   
   @app.get("/search")
   def search_endpoint(query: str, top_k: int = 13):
       results = search(query, config, top_sections=top_k)
       return results
   ```

---

## 🧪 Testing

### Unit Tests

```bash
# Test PostgreSQL connection
python test_pg_connection.py

# Test extensions
python check_pg_extensions.py

# Verify incremental loading
python hybrid_pgvector.py --build  # Should exit quickly if up-to-date
```

### Integration Tests

```bash
# Full pipeline test (use small sample)
python -c "
from arxiv_chunking_pipeline import chunk_arxiv_papers
chunks = chunk_arxiv_papers('papers/post_processed', max_papers=10)
print(f'Generated {len(chunks)} chunks from 10 papers')
"
```

### Performance Benchmarks

```python
import time
from hybrid_pgvector import search, HybridConfig

config = HybridConfig()

queries = [
    "transformer attention mechanism",
    "diffusion models",
    "reinforcement learning"
]

for query in queries:
    start = time.time()
    results = search(query, config)
    elapsed = time.time() - start
    print(f"{query}: {elapsed*1000:.1f}ms")
```

---

## 📚 References

### Components Used

- **model2vec**: https://github.com/MinishLab/model2vec
- **pgvector**: https://github.com/pgvector/pgvector
- **BM25**: Okapi BM25 algorithm (Robertson & Zaragoza, 2009)
- **RRF**: Reciprocal Rank Fusion (Cormack et al., 2009)

### Related Papers

- ArXiv papers sourced from: https://arxiv.org
- Chunking strategy inspired by: LangChain RecursiveCharacterTextSplitter
- Equidistant chunking: Based on sliding window aggregation

---

## 📝 License & Credits

**Author**: Created for ArXiv paper retrieval research  
**Status**: Production-ready (January 2026)  
**Database**: PostgreSQL 16 + pgvector 0.8.0  
**Python**: 3.10+

---

## 🎉 Quick Reference

### Most Common Commands

```bash
# Daily check for new papers (fast)
python hybrid_pgvector.py --build

# Search
python hybrid_pgvector.py --search "your query"

# Full rebuild (rare)
python hybrid_pgvector.py --build --reset

# Clear everything (testing)
python hybrid_pgvector.py --clear
```

### Key Numbers

- **172,272 chunks** from 1,607 papers
- **<5s** incremental check (vs 36min full rebuild)
- **<500ms** hybrid search query
- **64-dim** dense vectors (PCA from 256)
- **114,523** sparse vocabulary size
- **288x speedup** for no-op incremental runs

---

**System Status**: ✅ PRODUCTION READY  
**Last Updated**: January 25, 2026  
**Database**: 192.168.3.18:6024 (langchain/langchain)
