"""
Feature Catalog Management
Tracks features, predictions, and actual results for empirical accountability.
"""

import sqlite3
from datetime import datetime

DB_PATH = 'feature_catalog.sqlite3'

def init_feature_catalog(db_path=DB_PATH):
    """Initialize the feature catalog database with all tables."""
    conn = sqlite3.connect(db_path)
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
    print(f"✅ Feature catalog initialized: {db_path}")

def add_feature(name, description, f1_baseline=None, db_path=DB_PATH):
    """Add a new feature to track."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
        INSERT INTO features (name, description, f1_baseline, status)
        VALUES (?, ?, ?, 'TODO')
        """, (name, description, f1_baseline))
        
        conn.commit()
        feature_id = cursor.lastrowid
        print(f"✅ Feature added: {name} (ID: {feature_id})")
        return feature_id
    except sqlite3.IntegrityError:
        print(f"⚠️ Feature already exists: {name}")
        cursor.execute("SELECT id FROM features WHERE name = ?", (name,))
        return cursor.fetchone()[0]
    finally:
        conn.close()

def log_claim(feature_id, claim_text, predicted_f1=None, 
              predicted_improvement=None, confidence_score=0.5,
              db_path=DB_PATH):
    """
    Log a prediction claim BEFORE implementation.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
    INSERT INTO claims (feature_id, claim_text, predicted_f1, predicted_improvement, 
                       confidence_score, validation_result)
    VALUES (?, ?, ?, ?, ?, 'PENDING')
    """, (feature_id, claim_text, predicted_f1, predicted_improvement, confidence_score))
    
    conn.commit()
    claim_id = cursor.lastrowid
    conn.close()
    
    print(f"📝 Claim logged (ID: {claim_id}): {claim_text}")
    return claim_id

def update_feature_status(feature_id, status, f1_current=None, 
                         validation_notes=None, db_path=DB_PATH):
    """Update feature status and optionally record measured F1."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    updates = ["status = ?", "updated_at = CURRENT_TIMESTAMP"]
    params = [status]
    
    if f1_current is not None:
        updates.append("f1_current = ?")
        params.append(f1_current)
    
    if validation_notes:
        updates.append("validation_notes = ?")
        params.append(validation_notes)
    
    params.append(feature_id)
    
    cursor.execute(f"""
    UPDATE features 
    SET {', '.join(updates)}
    WHERE id = ?
    """, params)
    
    conn.commit()
    conn.close()
    print(f"✅ Feature {feature_id} updated: status={status}, f1={f1_current}")

def validate_claim(claim_id, actual_f1, db_path=DB_PATH):
    """
    Record actual measured F1 and calculate prediction error.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get predicted values
    cursor.execute("""
    SELECT predicted_f1, predicted_improvement 
    FROM claims WHERE id = ?
    """, (claim_id,))
    
    row = cursor.fetchone()
    if not row:
        print(f"❌ Claim {claim_id} not found")
        conn.close()
        return
    
    predicted_f1, predicted_improvement = row
    
    # Calculate error
    if predicted_f1 is not None:
        prediction_error = abs(actual_f1 - predicted_f1)
    else:
        prediction_error = None
    
    # Determine validation result
    if prediction_error is not None:
        if prediction_error < 0.05:
            validation_result = 'CONFIRMED'
        elif prediction_error < 0.15:
            validation_result = 'PARTIAL'
        else:
            validation_result = 'FAILED'
    else:
        validation_result = 'CONFIRMED'
    
    # Update claim
    cursor.execute("""
    UPDATE claims 
    SET actual_f1 = ?,
        prediction_error = ?,
        validated_timestamp = CURRENT_TIMESTAMP,
        validation_result = ?
    WHERE id = ?
    """, (actual_f1, prediction_error, validation_result, claim_id))
    
    conn.commit()
    conn.close()
    
    if prediction_error is not None:
        print(f"✅ Claim validated: predicted={predicted_f1:.4f}, actual={actual_f1:.4f}, error={prediction_error:.4f} ({validation_result})")
    else:
        print(f"✅ Claim validated: actual={actual_f1:.4f} ({validation_result})")

def get_prediction_accuracy(db_path=DB_PATH):
    """Get historical prediction accuracy to calibrate confidence."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
    SELECT 
        COUNT(*) as total_claims,
        AVG(prediction_error) as avg_error,
        AVG(CASE WHEN validation_result = 'CONFIRMED' THEN 1 ELSE 0 END) as confirmed_rate,
        AVG(CASE WHEN validation_result = 'FAILED' THEN 1 ELSE 0 END) as failed_rate
    FROM claims
    WHERE validation_result != 'PENDING'
    """)
    
    row = cursor.fetchone()
    conn.close()
    
    if row[0] == 0:
        print("⚠️ No validated claims yet - no historical accuracy data")
        return None
    
    total, avg_error, confirmed_rate, failed_rate = row
    print(f"\n📊 Historical Prediction Accuracy:")
    print(f"  Total validated claims: {total}")
    print(f"  Average error: {avg_error:.4f}" if avg_error else "  Average error: N/A")
    print(f"  Confirmed rate: {confirmed_rate*100:.1f}%")
    print(f"  Failed rate: {failed_rate*100:.1f}%")
    
    return {
        'total': total,
        'avg_error': avg_error,
        'confirmed_rate': confirmed_rate,
        'failed_rate': failed_rate
    }

def list_features(db_path=DB_PATH):
    """List all features with current status."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
    SELECT id, name, status, f1_baseline, f1_current, 
           updated_at, validation_notes
    FROM features
    ORDER BY updated_at DESC
    """)
    
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        print("No features tracked yet")
        return
    
    print("\n📋 Feature Catalog:")
    print("-" * 80)
    for row in rows:
        id, name, status, f1_base, f1_curr, updated, notes = row
        print(f"[{id}] {name}")
        print(f"    Status: {status}")
        if f1_base is not None:
            print(f"    Baseline F1: {f1_base:.4f}")
        if f1_curr is not None:
            improvement = f1_curr - (f1_base or 0)
            print(f"    Current F1: {f1_curr:.4f} ({improvement:+.4f})")
        print(f"    Updated: {updated}")
        if notes:
            print(f"    Notes: {notes}")
        print()

def list_claims(feature_id, db_path=DB_PATH):
    """List all claims for a specific feature."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
    SELECT id, claim_text, predicted_f1, actual_f1, 
           prediction_error, validation_result, claim_timestamp
    FROM claims
    WHERE feature_id = ?
    ORDER BY claim_timestamp DESC
    """, (feature_id,))
    
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        print(f"No claims for feature {feature_id}")
        return
    
    print(f"\n📝 Claims for Feature {feature_id}:")
    print("-" * 80)
    for row in rows:
        id, claim, pred_f1, actual_f1, error, result, timestamp = row
        print(f"[{id}] {claim}")
        if pred_f1 is not None:
            print(f"    Predicted F1: {pred_f1:.4f}")
        if actual_f1 is not None:
            print(f"    Actual F1: {actual_f1:.4f}")
        if error is not None:
            print(f"    Error: {error:.4f}")
        print(f"    Result: {result}")
        print(f"    Timestamp: {timestamp}")
        print()


if __name__ == '__main__':
    # Initialize database
    init_feature_catalog()
    
    # Record the frozen BERT baseline we just completed
    print("\n=== Recording Frozen BERT Baseline ===")
    baseline_id = add_feature(
        name="Frozen BERT Baseline",
        description="Fixed BIO labeling bugs + established baseline with frozen BERT + single linear layer",
        f1_baseline=0.1328  # First 250-chunk attempt after fixing labels
    )
    
    # This is what we just measured
    update_feature_status(
        baseline_id, 
        'DONE',
        f1_current=0.1764,  # Best trial result
        validation_notes="Best trial F1=0.1764, but full training collapsed to 0.0805 due to overfitting. Confirms frozen BERT architecture limitation."
    )
    
    # Check historical accuracy before making any new predictions
    print("\n=== Checking Historical Prediction Accuracy ===")
    accuracy = get_prediction_accuracy()
    
    # Display current state
    print("\n=== Current Feature Catalog ===")
    list_features()
    
    print("\n" + "="*80)
    print("✅ Database initialized and baseline recorded")
    print("="*80)
    print("\nNext steps:")
    print("1. Before proposing any architecture changes, check historical accuracy")
    print("2. Log prediction with appropriate confidence score")
    print("3. Implement changes")
    print("4. Measure actual F1")
    print("5. Validate claim against measured result")
