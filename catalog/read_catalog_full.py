"""Read feature 12 and architectural decisions."""
import sqlite3

db_path = r'c:\Users\user\arxiv_id_lists\feature_catalog.sqlite3'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

print("=== FEATURE 12: Hierarchical Paper→Section Retrieval (FULL) ===\n")
cursor.execute("SELECT * FROM features WHERE id = 12")
feat = cursor.fetchone()
if feat:
    col_names = [desc[0] for desc in cursor.description]
    for col_name, val in zip(col_names, feat):
        if val:
            print(f"{col_name}: {val}\n")

print("\n\n=== ARCHITECTURAL DECISIONS ===\n")
cursor.execute("PRAGMA table_info(architectural_decisions)")
schema = cursor.fetchall()
print("Schema:")
for col in schema:
    print(f"  {col[1]}: {col[2]}")

print("\n\nArchitectural Decisions:")
cursor.execute("SELECT id, decision, rationale FROM architectural_decisions")
decisions = cursor.fetchall()
for dec in decisions:
    print(f"\n[{dec[0]}] {dec[1]}")
    print(f"  Rationale: {dec[2]}")

# Check for GIST-related decisions
print("\n\n=== GIST-RELATED FEATURES ===\n")
cursor.execute("SELECT id, name, description FROM features WHERE name LIKE '%GIST%' OR description LIKE '%GIST%'")
gist_features = cursor.fetchall()
for feat in gist_features:
    print(f"\n[{feat[0]}] {feat[1]}")
    print(f"  {feat[2][:800]}")

conn.close()
