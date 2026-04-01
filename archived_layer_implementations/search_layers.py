"""Search for all features mentioning layers or architecture."""
import sqlite3

db_path = r'c:\Users\user\arxiv_id_lists\feature_catalog.sqlite3'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

print("=== ALL FEATURES (to find preamble/diagram) ===\n")
cursor.execute("SELECT id, name FROM features ORDER BY id")
all_feats = cursor.fetchall()
for feat_id, name in all_feats:
    print(f"[{feat_id}] {name}")

print("\n\n=== SEARCHING FOR LAYER/ARCHITECTURE DESCRIPTIONS ===\n")
cursor.execute("SELECT id, name, description FROM features WHERE description LIKE '%layer 1%' OR description LIKE '%layer 2%' OR description LIKE '%layer 3%' OR description LIKE '%architecture%'")
layer_features = cursor.fetchall()
for feat in layer_features:
    print(f"\n{'='*80}")
    print(f"[{feat[0]}] {feat[1]}")
    print(f"{'='*80}")
    print(feat[2])
    print()

conn.close()
