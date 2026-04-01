import sqlite3
from datetime import datetime

db_path = 'feature_catalog.sqlite3'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Add Bug #1: Runtime Label Mismatch (MEASURED)
print("➕ Adding Bug #1: Runtime Label Mismatch...")
cursor.execute("""
INSERT INTO features (name, description, f1_baseline, status)
VALUES (?, ?, ?, 'VALIDATING')
""", (
    "Runtime Label Mismatch Bug",
    "collate_fn bidirectional substring matching produces 45% different labels than pre-computed",
    0.1764  # baseline with runtime matcher
))
bug1_id = cursor.lastrowid

# Log claim with MEASURED results
cursor.execute("""
INSERT INTO claims (feature_id, claim_text, predicted_f1, actual_f1, 
                   confidence_score, validation_result)
VALUES (?, ?, ?, ?, ?, ?)
""", (
    bug1_id,
    "Runtime matcher produces 45% label mismatches vs pre-computed labels",
    None,  # No prediction made, this was empirically discovered
    None,  # Actual F1 with this bug is 0.1764 (baseline)
    1.0,   # Fully validated empirically
    'CONFIRMED'
))

# Log actual mismatch rates from test
cursor.execute("""
INSERT INTO claims (feature_id, claim_text, actual_f1, confidence_score, validation_result)
VALUES (?, ?, ?, ?, ?)
""", (
    bug1_id,
    "Empirical mismatch rates: Ex1=50%, Ex2=70.4%, Ex3=67.6%, Ex4=52%, Ex5=34.4% → Average 55% match (45% mismatch)",
    0.45,  # Mismatch rate as F1-like metric
    1.0,
    'CONFIRMED'
))

print(f"✅ Bug #1 logged (Feature ID {bug1_id})")

# Add Bug #2: Pre-computed Label Sparsity (MEASURED)
print("➕ Adding Bug #2: Pre-computed Label Sparsity...")
cursor.execute("""
INSERT INTO features (name, description, f1_baseline, f1_current, status)
VALUES (?, ?, ?, ?, 'FAILED')
""", (
    "Pre-computed Label Sparsity",
    "Pre-computed labels in bio_training_250chunks_clean.msgpack only label ~30-40% of tokens",
    0.1764,  # baseline with runtime matcher
    0.067    # actual F1 with sparse pre-computed labels
))
bug2_id = cursor.lastrowid

# Log claim with MEASURED results
cursor.execute("""
INSERT INTO claims (feature_id, claim_text, predicted_f1, actual_f1, 
                   prediction_error, confidence_score, validation_result)
VALUES (?, ?, ?, ?, ?, ?, ?)
""", (
    bug2_id,
    "Pre-computed labels would improve training (expected improvement ~0.25+ over baseline)",
    0.4,    # Optimistic prediction: might reach 0.40-0.45
    0.067,  # ACTUAL: F1 collapsed to 0.067
    0.333,  # Error: |0.067 - 0.4| = 0.333 (prediction was very wrong)
    0.0,    # Confidence 0 - prediction completely failed
    'FAILED'
))

# Log coverage analysis
cursor.execute("""
INSERT INTO claims (feature_id, claim_text, actual_f1, confidence_score, validation_result)
VALUES (?, ?, ?, ?, ?)
""", (
    bug2_id,
    "Label sparsity analysis: Ex1=32% coverage (11/34), Ex2=23% (12/52), Ex3=23% (8/35), Ex4=21% (10/48), Ex5=43% (13/30)",
    0.30,  # Average ~30% coverage
    1.0,
    'CONFIRMED'
))

print(f"✅ Bug #2 logged (Feature ID {bug2_id})")

conn.commit()
conn.close()

print("\n📊 Feature Catalog Updated:")
print("  ✅ Bug #1: Runtime mismatch (45%) - CONFIRMED EMPIRICALLY")
print("  ✅ Bug #2: Pre-computed sparse (30% coverage) - F1=0.067 CONFIRMED")
print("\n⚠️  ROOT CAUSE: Pre-computed labels only label subset of entities")
print("   Runtime matcher was BETTER due to over-labeling (high recall)")
print("\n🔧 NEXT: Fix pre-computed label generation for COMPLETE coverage")
