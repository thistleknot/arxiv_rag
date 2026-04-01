"""Quick script to read feature catalog preamble."""
import sqlite3

db_path = r'c:\Users\user\arxiv_id_lists\feature_catalog.sqlite3'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# List all tables
print("=== TABLES ===")
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()
for table in tables:
    print(f"  {table[0]}")

print("\n=== FEATURES TABLE SCHEMA ===")
cursor.execute("PRAGMA table_info(features)")
schema = cursor.fetchall()
for col in schema:
    print(f"  {col[1]}: {col[2]}")

print("\n=== FIRST FEW FEATURES ===")
cursor.execute("SELECT id, name, description FROM features LIMIT 5")
features = cursor.fetchall()
for feat in features:
    print(f"\n[{feat[0]}] {feat[1]}")
    print(f"  {feat[2]}")

# Check for preamble
print("\n=== CHECKING FOR PREAMBLE ===")
cursor.execute("SELECT name FROM features WHERE name LIKE '%preamble%' OR name LIKE '%diagram%' OR name LIKE '%architecture%'")
preambles = cursor.fetchall()
if preambles:
    print("Found preamble-related features:")
    for p in preambles:
        print(f"  {p[0]}")
        cursor.execute("SELECT description FROM features WHERE name = ?", (p[0],))
        desc = cursor.fetchone()
        if desc:
            print(f"  Description: {desc[0][:500]}")
else:
    print("No preamble features found by name. Checking all features...")
    cursor.execute("SELECT id, name, description FROM features")
    all_features = cursor.fetchall()
    for feat in all_features:
        if any(keyword in str(feat).lower() for keyword in ['diagram', 'architecture', 'preamble', 'layer', 'gist']):
            print(f"\n[{feat[0]}] {feat[1]}")
            print(f"  {feat[2][:500]}")

conn.close()
