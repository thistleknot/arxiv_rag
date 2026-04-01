"""
Add Streamlit Dashboard Feature to Catalog
"""
import sqlite3
from pathlib import Path

def add_dashboard_feature():
    """Add Streamlit BIO demo dashboard to feature catalog."""
    db_path = Path('feature_catalog.sqlite3')
    
    if not db_path.exists():
        print(f"❌ Database not found: {db_path}")
        return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check if already exists
    cursor.execute("SELECT id FROM features WHERE name = 'Streamlit BIO Tagger Dashboard'")
    if cursor.fetchone():
        print("⚠️ Feature already exists: Streamlit BIO Tagger Dashboard")
        conn.close()
        return
    
    # Add the feature
    try:
        cursor.execute("""
        INSERT INTO features (name, description, status, validation_notes)
        VALUES (?, ?, ?, ?)
        """, (
            'Streamlit BIO Tagger Dashboard',
            'Interactive web dashboard for testing BIO tagger predictions. Allows users to input text and visualize Subject-Predicate-Object extraction with entity highlighting.',
            'DONE',
            'Implemented in streamlit_bio_demo.py. Loads trained model (bio_tagger_multiclass.pt) and provides interactive testing interface. Shows holdout predictions for reference. Full functionality: text input, BIO prediction, entity extraction, visualization.'
        ))
        
        conn.commit()
        feature_id = cursor.lastrowid
        
        print(f"✅ Added feature: Streamlit BIO Tagger Dashboard (ID: {feature_id})")
        
        # List all features now
        cursor.execute("SELECT id, name, status FROM features ORDER BY id")
        rows = cursor.fetchall()
        print(f"\n📋 Feature Catalog: {len(rows)} total features")
        for id, name, status in rows:
            print(f"  [{id}] {name} - {status}")
        
    except sqlite3.IntegrityError as e:
        print(f"❌ Error adding feature: {e}")
    
    conn.close()

if __name__ == '__main__':
    print("Adding Streamlit dashboard to feature catalog...")
    add_dashboard_feature()
    print("\n✅ Dashboard feature cataloging complete!")
