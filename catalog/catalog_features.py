"""
Catalog all BIO tagging and Stanza extraction features in SQLite3 feature catalog.
"""
import sqlite3
import os
from datetime import datetime

DB_PATH = r'c:\Users\user\arxiv_id_lists\feature_catalog.sqlite3'

def init_feature_catalog():
    """Initialize the feature catalog database with all tables."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create features table
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
    
    # Create claims table
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
    
    # Create architectural_decisions table
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
    conn.close()
    print(f"✅ Feature catalog initialized: {DB_PATH}")

def add_stanza_features():
    """Add all Stanza-related features to the catalog."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    features = [
        {
            'name': 'Stanza Dependency Parsing for BIO Extraction',
            'description': 'Replace OpenIE with Stanza dependency parsing in extract_bio_training_data.py to preserve multi-word phrases',
            'status': 'DONE',
            'f1_baseline': None,
            'f1_current': None,
            'validation_notes': 'Fixed 1355 BIO violations (99.93% success). Proper B-I-I continuous spans validated.'
        },
        {
            'name': 'Stanza Dependency Parsing for Graph Extraction',
            'description': 'Replace OpenIE with Stanza dependency parsing in build_arxiv_graph_sparse.py to preserve multi-word entity nodes',
            'status': 'DONE',
            'f1_baseline': None,
            'f1_current': None,
            'validation_notes': 'Code fixed and validated standalone. Multi-word phrase preservation confirmed.'
        },
        {
            'name': 'BIO Sequence Validation Unit Test',
            'description': 'Created validate_bio_sequences.py to check for consecutive B-X B-X violations',
            'status': 'DONE',
            'f1_baseline': None,
            'f1_current': None,
            'validation_notes': 'Validates BIO sequences, distinguishes atomic vs compound approaches.'
        },
        {
            'name': 'get_subtree_text() Function',
            'description': 'Recursively collects dependency subtree to preserve multi-word phrases during extraction',
            'status': 'DONE',
            'f1_baseline': None,
            'f1_current': None,
            'validation_notes': 'Core function for Stanza phrase preservation. Tested and working.'
        },
        {
            'name': 'extract_spo_from_sentence() Function',
            'description': 'Extracts S-P-O triplets using Stanza dependency parse instead of OpenIE atomization',
            'status': 'DONE',
            'f1_baseline': None,
            'f1_current': None,
            'validation_notes': 'Replaces OpenIE extraction. Uses get_subtree_text() for full phrases.'
        },
        {
            'name': 'BIO Training Data Regeneration',
            'description': 'Regenerated bio_training_250chunks_complete_FIXED.msgpack with Stanza extraction',
            'status': 'DONE',
            'f1_baseline': None,
            'f1_current': None,
            'validation_notes': '1103 examples generated. 1355 violations → 1 violation (99.93% fix rate).'
        }
    ]
    
    for feature in features:
        try:
            cursor.execute("""
            INSERT INTO features (name, description, status, f1_baseline, f1_current, validation_notes)
            VALUES (?, ?, ?, ?, ?, ?)
            """, (feature['name'], feature['description'], feature['status'], 
                  feature['f1_baseline'], feature['f1_current'], feature['validation_notes']))
            print(f"✅ Added feature: {feature['name']}")
        except sqlite3.IntegrityError:
            print(f"⚠️  Feature already exists: {feature['name']}")
    
    conn.commit()
    conn.close()

def add_architectural_decision():
    """Add the architectural decision to replace OpenIE with Stanza."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
    INSERT INTO architectural_decisions 
    (decision, rationale, before_state, after_state, measured_impact)
    VALUES (?, ?, ?, ?, ?)
    """, (
        'Replace OpenIE with Stanza Dependency Parsing',
        'OpenIE atomizes multi-word phrases ("deep learning models" → 3 separate entities), causing 1355 BIO violations. Stanza preserves full phrases via dependency subtree collection.',
        'OpenIE: extract(sentence) → atomized triplets → BIO violations (B-SUBJ B-SUBJ B-SUBJ)',
        'Stanza: get_subtree_text(head_word) → complete phrases → proper BIO spans (B-SUBJ I-SUBJ I-SUBJ)',
        '1355 violations → 1 violation (99.93% fix rate)'
    ))
    
    conn.commit()
    conn.close()
    print("✅ Added architectural decision")

def list_all_features():
    """List all features in the catalog."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
    SELECT id, name, status, validation_notes, updated_at
    FROM features
    ORDER BY updated_at DESC
    """)
    
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        print("No features in catalog yet")
        return
    
    print("\n📋 Feature Catalog:")
    print("-" * 100)
    for row in rows:
        id, name, status, notes, updated = row
        print(f"\n[{id}] {name}")
        print(f"    Status: {status}")
        print(f"    Updated: {updated}")
        if notes:
            print(f"    Notes: {notes[:100]}...")
    print()

if __name__ == '__main__':
    print("Initializing feature catalog...")
    init_feature_catalog()
    
    print("\nAdding Stanza features...")
    add_stanza_features()
    
    print("\nAdding architectural decision...")
    add_architectural_decision()
    
    print("\nCurrent feature catalog:")
    list_all_features()
    
    print("\n✅ Feature cataloging complete!")
