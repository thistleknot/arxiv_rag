"""
Generate README.md from the consolidated feature catalog.

    feature_catalog_master.sqlite3  → all subsystems (retriever, BIO tagger)

Usage:
    python catalog/generate_readme.py              # writes README.md at repo root
    python catalog/generate_readme.py --output out.md
    python catalog/generate_readme.py --dry-run    # print to stdout only
"""

import sqlite3
import argparse
from datetime import datetime
from pathlib import Path


# ═══════════════════════════════════════════════════════════════
#  Database Queries
# ═══════════════════════════════════════════════════════════════

def query_features(db_path: str, category: str | None = None) -> list[dict]:
    """Fetch features from the master catalog, optionally filtered by category."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    if category:
        rows = conn.execute("""
            SELECT id, name, category, description, definition, files, status, source,
                   created_at, updated_at
            FROM features WHERE category = ? ORDER BY id
        """, (category,)).fetchall()
    else:
        rows = conn.execute("""
            SELECT id, name, category, description, definition, files, status, source,
                   created_at, updated_at
            FROM features ORDER BY id
        """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def query_decisions(db_path: str) -> list[dict]:
    """Fetch architectural decisions if the table exists."""
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT id, decision, rationale, before_state, after_state,
                   before_f1, after_f1, measured_impact, decision_timestamp
            FROM architectural_decisions ORDER BY id
        """).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception:
        return []


def query_claims(db_path: str) -> list[dict]:
    """Fetch prediction claims if the table exists."""
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT c.id, c.claim_text, c.predicted_f1, c.actual_f1,
                   c.prediction_error, c.validation_result, c.confidence_score,
                   f.name as feature_name
            FROM claims c
            LEFT JOIN features f ON c.feature_id = f.id
            ORDER BY c.id
        """).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception:
        return []


# ═══════════════════════════════════════════════════════════════
#  Status Badge
# ═══════════════════════════════════════════════════════════════

STATUS_EMOJI = {
    'DONE': '✅',   'done': '✅',
    'IN_PROGRESS': '🔄',   'in_progress': '🔄',   'wip': '🔄',
    'VALIDATING': '🧪',   'validating': '🧪',
    'TODO': '📋',   'todo': '📋',   'planned': '📋',
    'FAILED': '❌',   'failed': '❌',
}

def status_badge(s: str) -> str:
    key = (s or '').upper()
    return f"{STATUS_EMOJI.get(s, STATUS_EMOJI.get(key, '❓'))} {s}"


# ═══════════════════════════════════════════════════════════════
#  Section Generators
# ═══════════════════════════════════════════════════════════════

def generate_header() -> str:
    return f"""# Hybrid Retrieval & Knowledge Extraction System

> Auto-generated from feature catalogs on {datetime.now().strftime('%Y-%m-%d %H:%M')}.
> Source database: `feature_catalog_master.sqlite3` (1,756 features)
> Generator: `generate_readme.py`

**Two subsystems:**
1. **Hybrid Retriever** — 3-layer φ-scaled retrieval: L1 BM25+GIST → RRF → L2 BM25-triplet+dense → RRF → L3 ColBERT reranking, pgvector backend
2. **BIO Tagger** — BERT-based SPO triplet extraction, Augmented Ontological Knowledge Graph (AOKG), Streamlit demo

---
"""


def generate_architecture() -> str:
    return """## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│              THREE-LAYER φ-SCALED RETRIEVER PIPELINE             │
│                                                                  │
│  Document Sources → Equidistant Chunking → Checkpointing        │
│       ↓                                                          │
│  Dual Indexing: Dense (model2vec 256→64 PCA) + Sparse (BM25)    │
│       ↓                                                          │
│  PostgreSQL + pgvector (HNSW dense; BM25 exact seq-scan)         │
│       ↓                                                          │
│  Layer 1: Hybrid Seeds                                           │
│    BM25 pool (top_k²) + Dense pool (top_k²) → RRF               │
│    → prev_fib(top_k²) seeds (e.g. 169+169 → RRF → 144 chunks)   │
│       ↓  seeds + ECDF weights as INPUT to Layer 2                │
│  Layer 2: ECDF-Weighted Dual Expansion                           │
│    Path A: BM25 triplets (layer2_triplet_bm25, sparsevec 16k)    │
│            signed-hash, exact seq-scan, ECDF-weighted query      │
│    Path B: Qwen3 Dense — ECDF-weighted centroid → cosine         │
│            HNSW approx ANN on 256-dim model2vec embeddings       │
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
"""


def generate_quick_start() -> str:
    return """## 🚀 Quick Start

### Hybrid Retriever (ArXiv Papers)
```bash
# Ingest ArXiv papers (runs chunking + BM25 + dense indexing)
python ingestion/arxiv_chunking_pipeline.py

# Query — full 3-layer pipeline
python arxiv_retriever.py --search "transformer attention mechanism" --top-k 5

# Save results to markdown
python arxiv_retriever.py --search "attention mechanism" --top-k 13 --save results.md
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
"""


def generate_feature_section(title: str, db_path: str, category: str | None = None) -> str:
    """Generate a feature catalog section from the master catalog database."""
    features = query_features(db_path, category=category)
    decisions = query_decisions(db_path)
    claims = query_claims(db_path)

    cat_label = f" (category: {category})" if category else ""
    lines = [f"## {title}\n"]
    lines.append(f"*Source: `{Path(db_path).name}`{cat_label} — {len(features)} features*\n")

    # Summary table
    done = sum(1 for f in features if f['status'] in ('DONE', 'done'))
    todo = sum(1 for f in features if f['status'] in ('TODO', 'todo', 'planned'))
    wip  = sum(1 for f in features if f['status'] in ('IN_PROGRESS', 'in_progress', 'VALIDATING', 'validating', 'wip'))
    failed = sum(1 for f in features if f['status'] in ('FAILED', 'failed'))

    lines.append(f"| Status | Count |")
    lines.append(f"|--------|-------|")
    if done:   lines.append(f"| ✅ DONE | {done} |")
    if wip:    lines.append(f"| 🔄 IN PROGRESS | {wip} |")
    if todo:   lines.append(f"| 📋 TODO | {todo} |")
    if failed: lines.append(f"| ❌ FAILED | {failed} |")
    lines.append("")

    # Feature details
    lines.append("### Features\n")
    for f in features:
        lines.append(f"#### {f['id']}. {f['name']} {status_badge(f['status'])}\n")
        if f.get('description'):
            lines.append(f"{f['description']}\n")
        if f.get('definition'):
            lines.append(f"*Definition: {f['definition']}*\n")
        if f.get('category'):
            lines.append(f"*Category: `{f['category']}`*\n")

    # Architectural decisions
    if decisions:
        lines.append("### Architectural Decisions\n")
        for d in decisions:
            lines.append(f"**{d['id']}. {d['decision']}**\n")
            if d['rationale']:
                lines.append(f"- **Rationale:** {d['rationale']}")
            if d['before_state']:
                lines.append(f"- **Before:** {d['before_state']}")
            if d['after_state']:
                lines.append(f"- **After:** {d['after_state']}")
            if d['before_f1'] is not None and d['after_f1'] is not None:
                delta = d['after_f1'] - d['before_f1']
                lines.append(f"- **Impact:** F1 {d['before_f1']:.4f} → {d['after_f1']:.4f} ({delta:+.4f})")
            lines.append("")

    # Claims (prediction tracking)
    validated_claims = [c for c in claims if c['validation_result'] and c['validation_result'] != 'PENDING']
    if validated_claims:
        lines.append("### Prediction Accuracy\n")
        lines.append("| Feature | Claim | Predicted F1 | Actual F1 | Error | Result |")
        lines.append("|---------|-------|-------------|-----------|-------|--------|")
        for c in validated_claims:
            pred = f"{c['predicted_f1']:.4f}" if c['predicted_f1'] is not None else "—"
            actual = f"{c['actual_f1']:.4f}" if c['actual_f1'] is not None else "—"
            err = f"{c['prediction_error']:.4f}" if c['prediction_error'] is not None else "—"
            result = c['validation_result'] or "—"
            feat = c['feature_name'] or "—"
            claim_short = (c['claim_text'][:50] + "...") if len(c['claim_text'] or '') > 50 else (c['claim_text'] or "—")
            lines.append(f"| {feat} | {claim_short} | {pred} | {actual} | {err} | {result} |")
        lines.append("")

    lines.append("---\n")
    return "\n".join(lines)


def generate_file_structure() -> str:
    return """## 📁 Key Files

### Hybrid Retriever
| File | Purpose |
|------|---------|
| `arxiv_retriever.py` | CLI entry point — full 3-layer pipeline (L1→L2→ColBERT) |
| `retrieval/base_gist_retriever.py` | Abstract base: search(), RRF, L2 expansion, ColBERT |
| `retrieval/pgvector_retriever.py` | PostgreSQL/pgvector backend (BM25 + dense) |
| `retrieval/ecdf_rrf_retriever.py` | ECDF-weighted L2 expansion logic |
| `ingestion/arxiv_chunking_pipeline.py` | Document ingestion and chunking |
| `ingestion/ingest_layer2_new_triplets.py` | L2 BM25 triplet table ingestion (sparsevec 16k) |
| `catalog/feature_catalog_master.sqlite3` | Consolidated feature tracking (1,756 features) |

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
| `bio_tagger_features.sqlite3` | → consolidated into `catalog/feature_catalog_master.sqlite3` |
| `checkpoints/chunks.msgpack` | Source corpus: 161,389 chunks |
| `bio_training_250chunks_complete_FIXED.msgpack` | Training/eval data (250 chunks) |



---
"""


def generate_performance() -> str:
    return """## 📊 Performance

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
"""


def generate_aokg_section() -> str:
    return """## 🧠 AOKG — Augmented Ontological Knowledge Graph

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
"""


def generate_config_section() -> str:
    return """## ⚙️ Configuration

### PostgreSQL (for Hybrid Retriever)
```python
# pgvector_retriever.py — PGVectorConfig dataclass
host = "localhost"
port = 5432
dbname = "langchain"        # Docker: langchain_postgres
user = "langchain"
embedding_dim = 256         # model2vec qwen3_static_embeddings
bm25_k1 = 1.5
bm25_b  = 0.75
n_buckets = 16000           # signed-hash sparsevec dimension
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
"""


def generate_footer() -> str:
    return f"""## 📝 Feature Catalog Management

Both subsystems track features, claims, and architectural decisions in SQLite databases using `feature_catalog.py`.

```bash
# List all features
python -c "import sqlite3; c=sqlite3.connect('feature_catalog_master.sqlite3'); [print(r) for r in c.execute('SELECT id,name,status FROM features').fetchall()[:20]]"

# Regenerate this README
python catalog/generate_readme.py

# Search features (FTS5)
python -c "import sqlite3; c=sqlite3.connect('feature_catalog_master.sqlite3'); [print(r) for r in c.execute(\"SELECT name,status FROM features_fts JOIN features ON features.id=features_fts.rowid WHERE features_fts MATCH 'bm25'\").fetchall()]"
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

*Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} by `generate_readme.py`*
"""


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

def generate_readme() -> str:
    """Build complete README from the consolidated master catalog."""
    # Master DB is one level up from the catalog/ directory
    MASTER_DB = str(Path(__file__).parent.parent / 'feature_catalog_master.sqlite3')

    sections = [
        generate_header(),
        generate_architecture(),
        generate_quick_start(),
        generate_feature_section("🔍 Hybrid Retriever — Feature Catalog", MASTER_DB, category='retrieval'),
        generate_feature_section("🏷️ BIO Tagger — Feature Catalog", MASTER_DB, category='training'),
        generate_feature_section("📋 All Features — Master Catalog", MASTER_DB),
        generate_file_structure(),
        generate_performance(),
        generate_aokg_section(),
        generate_config_section(),
        generate_footer(),
    ]
    return "\n".join(sections)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate README from feature catalogs')
    parser.add_argument('--output', default='README.md', help='Output file (default: README.md)')
    parser.add_argument('--dry-run', action='store_true', help='Print to stdout only')
    args = parser.parse_args()

    readme_content = generate_readme()

    if args.dry_run:
        print(readme_content)
    else:
        Path(args.output).write_text(readme_content, encoding='utf-8')
        print(f"[OK] README written to {args.output} ({len(readme_content)} chars, "
              f"{readme_content.count(chr(10))} lines)")
