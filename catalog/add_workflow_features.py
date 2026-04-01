"""
Add workflow preamble features to both feature catalog databases.
Documents the expected execution order for each subsystem.
"""

import sqlite3
from datetime import datetime


def ensure_tables(db_path):
    """Create tables if they don't exist."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
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
    
    conn.commit()
    conn.close()


def add_workflow_feature(db_path, name, description):
    """Add workflow preamble feature using INSERT OR IGNORE."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
        INSERT OR IGNORE INTO features (name, description, status, created_at, updated_at)
        VALUES (?, ?, 'DONE', ?, ?)
        """, (name, description, datetime.now(), datetime.now()))
        
        if cursor.rowcount > 0:
            conn.commit()
            print(f"✅ Added workflow feature to {db_path}")
        else:
            print(f"⏭️  Workflow feature already exists in {db_path}")
    except Exception as e:
        print(f"❌ Error adding to {db_path}: {e}")
    finally:
        conn.close()


def main():
    # BIO Tagger workflow
    bio_workflow = """Workflow Execution Order for BIO Tagger System:

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
run_bio_tagger.bat check    # System health check"""

    # Hybrid Retriever workflow
    retriever_workflow = """Workflow Execution Order for Hybrid Retrieval System:

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
   - Purpose: 3-layer φ-scaled retrieval (L1 BM25+Dense seeds → L2 ECDF expansion → L3 ColBERT+CrossEncoder)
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
python query_quotes.py "life wisdom"                     # Query quotes"""

    # Add workflow features to both databases
    print("=" * 60)
    print("Adding workflow preamble features to feature catalogs")
    print("=" * 60)
    print()
    
    ensure_tables("bio_tagger_features.sqlite3")
    ensure_tables("feature_catalog.sqlite3")
    
    add_workflow_feature(
        "bio_tagger_features.sqlite3",
        "Workflow: BIO Tagger Execution Order",
        bio_workflow
    )
    
    add_workflow_feature(
        "feature_catalog.sqlite3",
        "Workflow: Hybrid Retriever Execution Order",
        retriever_workflow
    )
    
    print()
    print("=" * 60)
    print("✅ Workflow features added to both databases")
    print("=" * 60)
    print()
    print("Regenerate README with:")
    print("  python generate_readme.py")


if __name__ == "__main__":
    main()
