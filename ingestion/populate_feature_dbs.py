"""
Populate feature catalog databases with missing features.
INSERT OR IGNORE only — never deletes or overwrites existing rows.

Two databases:
    bio_tagger_features.sqlite3  — BIO tagger pipeline (BERT training + Streamlit app)
    feature_catalog.sqlite3      — Hybrid retriever (BM25/embeddings) + graph transformer reranking
"""

import sqlite3
from datetime import datetime


def ensure_tables(db_path: str):
    """Create tables if they don't exist (idempotent)."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""
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
    c.execute("""
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
    c.execute("""
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
    conn.close()


def add_feature(db_path, name, description, status='DONE', f1_baseline=None,
                f1_current=None, validation_notes=None):
    """INSERT OR IGNORE a single feature. Returns feature id."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""
    INSERT OR IGNORE INTO features (name, description, status, f1_baseline, f1_current, validation_notes)
    VALUES (?, ?, ?, ?, ?, ?)
    """, (name, description, status, f1_baseline, f1_current, validation_notes))
    conn.commit()
    if c.rowcount > 0:
        fid = c.lastrowid
        print(f"  ✅ Added: {name} (id={fid})")
    else:
        c.execute("SELECT id FROM features WHERE name = ?", (name,))
        fid = c.fetchone()[0]
        print(f"  ⏭️  Exists: {name} (id={fid})")
    conn.close()
    return fid


def add_decision(db_path, decision, rationale, before_state=None, after_state=None):
    """INSERT an architectural decision (no unique constraint, so always inserts).
    Check for duplicates by decision text first."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT id FROM architectural_decisions WHERE decision = ?", (decision,))
    existing = c.fetchone()
    if existing:
        print(f"  ⏭️  Decision exists: {decision[:60]}...")
        conn.close()
        return existing[0]
    c.execute("""
    INSERT INTO architectural_decisions (decision, rationale, before_state, after_state)
    VALUES (?, ?, ?, ?)
    """, (decision, rationale, before_state, after_state))
    conn.commit()
    did = c.lastrowid
    print(f"  ✅ Decision added: {decision[:60]}... (id={did})")
    conn.close()
    return did


# ═══════════════════════════════════════════════════════════════
#  BIO TAGGER FEATURES  →  bio_tagger_features.sqlite3
# ═══════════════════════════════════════════════════════════════

BIO_DB = 'bio_tagger_features.sqlite3'

BIO_FEATURES = [
    # ── Stage 1: Training Data Extraction ──
    dict(
        name="Stanza Knowledge Distillation",
        description=(
            "Extract BIO training labels from Stanza dependency parser (teacher model). "
            "Script: extract_bio_training_data.py. "
            "CLI: python extract_bio_training_data.py --chunks 250 --output <file>.msgpack. "
            "Parses sentences with Stanza dep parser, extracts nsubj/dobj/root triples, "
            "converts to multi-hot BIO labels per BERT token. "
            "Output msgpack keys: training_data, stats, tokenizer_name, label_names, architecture."
        ),
        status='DONE',
        validation_notes="Primary training data source. 250 chunks → ~1375 sentences."
    ),
    # ── Stage 2: Model Architecture ──
    dict(
        name="BIOTagger Model Architecture",
        description=(
            "BERT-base-uncased + 6 independent binary classifiers (nn.Linear(768,1) + sigmoid). "
            "Script: train_bio_tagger.py, class BIOTagger (line 93). "
            "Labels: B-SUBJ, I-SUBJ, B-PRED, I-PRED, B-OBJ, I-OBJ. "
            "forward() returns (probs, logits) tuple — probs shape [batch, seq_len, 6]. "
            "Model artifact: bio_tagger_best.pt (direct state_dict, 200+ keys)."
        ),
        status='DONE',
        validation_notes="6 independent heads allow multi-label tagging per token."
    ),
    # ── Stage 3: Optuna Hyperparameter Tuning ──
    dict(
        name="Optuna Hyperparameter Tuning",
        description=(
            "Box-Cox resampled training with Optuna. "
            "Script: tune_bio_tagger.py, class BoxCoxBIODataset (line 44). "
            "CLI: python tune_bio_tagger.py --data <file>.msgpack --unfreeze-layers 12 --n-trials 21. "
            "Hyperparameter space: LR (1e-6 to 1e-3), dropout (0-0.5), O-weight (0.001-0.1), "
            "batch (4/8/16), optimizer (AdamW/Adam/SGD). "
            "Entity-density-based sample weighting. Best model saved to bio_tagger_best.pt."
        ),
        status='DONE',
        validation_notes="21 trials on bio_training_250chunks_complete_FIXED.msgpack. All 12 BERT layers unfrozen."
    ),
    # ── Stage 4: Inference Engine ──
    dict(
        name="BIOTripletExtractor Inference",
        description=(
            "Production inference class for SPO triplet extraction. "
            "Script: inference_bio_tagger.py, class BIOTripletExtractor (line 123). "
            "Methods: extract_triplets(text, threshold) → list of (subj, pred, obj). "
            "Internal: extract_spans() parses BIO probabilities into Span objects, "
            "reconstruct_triplets() groups spans into S-P-O triples. "
            "Model loads bio_tagger_best.pt as direct state_dict."
        ),
        status='DONE',
        validation_notes="4.08 chunks/sec on CUDA. 94.5% triplet extraction rate."
    ),
    # ── Post-inference: Stopword Removal ──
    dict(
        name="Post-Inference Stopword Removal",
        description=(
            "Strip stopwords from extracted spans AFTER model inference (model learns full boundaries). "
            "Script: inference_bio_tagger.py. "
            "SPAN_STOPWORDS set (line 71): articles (a/an/the), prepositions (to/of/in/on/at/by/for/with/from/into/through), "
            "conjunctions (and/or/but/nor), copulas (is/are/was/were/be/been/being), "
            "relative pronouns (that/which/who), pronouns (this/these/those/it/its). "
            "clean_span_tokens(tokens, label_type) (line 88): "
            "strips all stopwords from SUBJ/OBJ and PRED spans. "
            "Can return empty list if span is entirely stopwords, which are then skipped during triplet reconstruction."
        ),
        status='DONE',
        validation_notes="Post-inference design: model trains on full spans, cleanup happens at output."
    ),
    # ── Post-inference: Cartesian Product Expansion ──
    dict(
        name="Cartesian Product Triplet Expansion",
        description=(
            "Expand multi-word spans into atomic word-level triplets via cartesian product. "
            "Script: streamlit_bio_demo.py, function expand_cartesian_triplets() (line 177). "
            "Process: flatten all words per role → deduplicate → sorted set → itertools.product(S, P, O). "
            "Example: SUBJ=['humans'], PRED=['possess'], OBJ=['ability','tools'] "
            "→ [('humans','possess','ability'), ('humans','possess','tools')]. "
            "Each atomic triplet becomes an edge in the knowledge graph."
        ),
        status='DONE',
        validation_notes="Produces atomic SPO triples suitable for graph construction."
    ),
    # ── Post-inference: Relation Graph ──
    dict(
        name="Atomic Relation Graph",
        description=(
            "NetworkX DiGraph visualization of SPO triplets. "
            "Script: streamlit_bio_demo.py, function render_triplet_graph() (line 195). "
            "SUBJ nodes (red, left column) → OBJ nodes (blue, right column), edges labeled by PRED. "
            "Multiple predicates between same S-O pair concatenated as comma-separated edge label. "
            "Shell layout: subjects on left column, objects on right column, centered vertically."
        ),
        status='DONE',
        validation_notes="Tab 1 visualization in Streamlit demo."
    ),
    # ── Post-inference: AOKG 4-Layer Graph ──
    dict(
        name="Augmented Ontological Knowledge Graph (AOKG)",
        description=(
            "4-layer ANN-style knowledge graph with WordNet resolution. "
            "Script: streamlit_bio_demo.py. "
            "Layer 0 (bottom): Surface words — raw SPO edges from cartesian expansion. "
            "Layer 1: Lemmas — morphy-reduced via wn.morphy(), deduplicated. "
            "Layer 2: Synsets — first WordNet synset name, deduplicated. "
            "Layer 3 (top): Hypernyms — one level up via synset.hypernyms(), deduplicated. "
            "Functions: _resolve_word(word, role) (line 320) resolves L0→L3, "
            "render_layered_graph(S, P, O) (line 341) builds 4-layer visualization. "
            "Vertical dashed edges connect each node upward to its parent. "
            "Multiple words sharing the same lemma/synset/hypernym CONVERGE at that layer. "
            "LCA convergence level (0-3) serves as semantic distance metric for reranking."
        ),
        status='DONE',
        validation_notes="Named AOKG. LCA distance: 0=same word, 1=same lemma, 2=same synset, 3=same hypernym."
    ),
    # ── Streamlit Demo App ──
    dict(
        name="Streamlit BIO Tagger Demo",
        description=(
            "3-tab interactive demo application. "
            "Script: streamlit_bio_demo.py (1064 lines). "
            "Tab 1 — Interactive Testing: paste text → extract triplets → view atomic graph + AOKG. "
            "Tab 2 — Eval Set Browser: browse evaluation samples from bio_training_250chunks_complete_FIXED.msgpack. "
            "Tab 3 — Eval Metrics: aggregate precision/recall/F1 across eval set. "
            "Launch: run_bio_tagger.bat app  OR  streamlit run streamlit_bio_demo.py. "
            "URL: http://localhost:8501."
        ),
        status='DONE',
        validation_notes="Menu-driven via run_bio_tagger.bat (4 options: app, test, integration, check)."
    ),
    # ── Corpus Benchmark ──
    dict(
        name="Full Corpus Inference Benchmark",
        description=(
            "Benchmark BIO inference throughput on full chunk corpus. "
            "Script: benchmark_full_corpus_inference.py. "
            "Data source: checkpoints/chunks.msgpack (161,389 chunks as list of dicts). "
            "Schema: [doc_id, paper_id, section_idx, chunk_idx, text]. "
            "Results (100 chunks): 4.08 chunks/sec, 22.26 sentences/sec, 21.04 triplets/sec, "
            "94.5% extraction rate, 5.5 avg sentences/chunk, 5.2 avg triplets/chunk. "
            "Full corpus estimate: ~11 hours."
        ),
        status='DONE',
        validation_notes="Strategic pivot: too expensive for full corpus → use as reranker on hybrid-retrieved results only."
    ),
]

BIO_DECISIONS = [
    dict(
        decision="Post-inference stopword removal instead of training-time filtering",
        rationale=(
            "Model should learn full BIO span boundaries including stopwords, "
            "then strip at output. Training on cleaned data caused boundary confusion. "
            "SPAN_STOPWORDS applied in clean_span_tokens() after span extraction."
        ),
        before_state="Attempted stopword filtering in training data (extract_bio_training_data.py)",
        after_state="Stopwords stripped post-inference in inference_bio_tagger.py:88"
    ),
    dict(
        decision="Cartesian product expansion for atomic triplets",
        rationale=(
            "Multi-word BIO spans like OBJ=['extraordinary','ability','create','utilize','tools'] "
            "need decomposition into atomic word-level triplets for graph construction. "
            "itertools.product(S,P,O) produces all combinations."
        ),
        before_state="Multi-word spans as single graph nodes",
        after_state="Atomic word-level nodes via expand_cartesian_triplets() in streamlit_bio_demo.py:177"
    ),
    dict(
        decision="4-layer AOKG with WordNet resolution for semantic distance",
        rationale=(
            "LCA convergence level (0-3) provides a discrete semantic distance metric. "
            "Two words converging at L1 (same lemma) are closer than at L3 (same hypernym). "
            "Enables graph-based reranking signal complementary to embedding similarity."
        ),
        before_state="Flat collapsed synset/hypernym graph",
        after_state="4-layer ANN-style graph: L0=surface, L1=lemma, L2=synset, L3=hypernym"
    ),
    dict(
        decision="Use AOKG as precision reranker, not full corpus expansion",
        rationale=(
            "Full corpus inference at 4.08 chunks/sec × 161k chunks = ~11 hours. "
            "Too expensive for recall expansion. Instead, use hybrid retrieval (BM25+dense) "
            "for recall, then apply AOKG graph on retrieved results only for precision reranking."
        ),
        before_state="Plan to build AOKG over entire corpus for graph-based recall",
        after_state="Strategic pivot: AOKG as reranker over hybrid-retrieved results only"
    ),
]


# ═══════════════════════════════════════════════════════════════
#  HYBRID RETRIEVER FEATURES  →  feature_catalog.sqlite3
# ═══════════════════════════════════════════════════════════════

RETRIEVER_DB = 'feature_catalog.sqlite3'

RETRIEVER_FEATURES = [
    # ── Future: Graph Transformer Reranking ──
    dict(
        name="Graph Transformer Reranking Stage",
        description=(
            "Expand-and-rerank stage AFTER hybrid retrieval (BM25+dense). "
            "Pipeline: hybrid retrieve → BIO extract on retrieved chunks → build AOKG subgraph → "
            "graph embeddings (node2vec/graph2vec/graph transformer) → rerank by graph similarity. "
            "Requires: (1) train BIO model (done, bio_tagger_best.pt), "
            "(2) apply model to derive graph for retrieved chunks at query time, "
            "(3) compute graph embeddings, (4) rerank. "
            "Acceleration ideas: model2vec-style approximation for fast graph embeddings, "
            "WordPiece tokenization of graph nodes → generalizable sparse vectors."
        ),
        status='TODO',
        validation_notes="Depends on bio_tagger_features.sqlite3 pipeline being complete."
    ),
]

RETRIEVER_DECISIONS = [
    dict(
        decision="Graph transformer reranking as post-hybrid precision stage",
        rationale=(
            "Hybrid retrieval (BM25+dense) provides recall. AOKG graph provides precision signal. "
            "Graph embeddings could be accelerated via WordPiece tokenization of graph nodes, "
            "producing a fixed-vocabulary sparse vector analogous to BM25 but over ontological structure. "
            "This avoids the 11-hour full corpus inference cost by only processing retrieved chunks."
        ),
        before_state="3-layer φ-scaled pipeline: L1(BM25+Dense) → L2(ECDF expansion) → L3(ColBERT+CrossEncoder)",
        after_state="Proposed: ...→L3 reranking→BIO Extract→AOKG Rerank"
    ),
]


def populate_bio_db():
    """Add missing BIO tagger features to bio_tagger_features.sqlite3."""
    print(f"\n{'='*60}")
    print(f"  Populating {BIO_DB}")
    print(f"{'='*60}")
    ensure_tables(BIO_DB)

    for feat in BIO_FEATURES:
        add_feature(BIO_DB, **feat)

    print(f"\n  Architectural decisions:")
    for dec in BIO_DECISIONS:
        add_decision(BIO_DB, **dec)


def populate_retriever_db():
    """Add missing retriever features to feature_catalog.sqlite3."""
    print(f"\n{'='*60}")
    print(f"  Populating {RETRIEVER_DB}")
    print(f"{'='*60}")
    ensure_tables(RETRIEVER_DB)

    for feat in RETRIEVER_FEATURES:
        add_feature(RETRIEVER_DB, **feat)

    print(f"\n  Architectural decisions:")
    for dec in RETRIEVER_DECISIONS:
        add_decision(RETRIEVER_DB, **dec)


def show_summary():
    """Print row counts for both databases."""
    for db_path, label in [(BIO_DB, "BIO Tagger"), (RETRIEVER_DB, "Hybrid Retriever")]:
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM features")
        feat_count = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM claims")
        claim_count = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM architectural_decisions")
        dec_count = c.fetchone()[0]
        conn.close()
        print(f"\n  {label} ({db_path}):")
        print(f"    Features: {feat_count}")
        print(f"    Claims: {claim_count}")
        print(f"    Decisions: {dec_count}")


if __name__ == '__main__':
    populate_bio_db()
    populate_retriever_db()
    print(f"\n{'='*60}")
    print("  Summary")
    print(f"{'='*60}")
    show_summary()
