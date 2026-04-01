import sqlite3

conn = sqlite3.connect('feature_catalog.sqlite3')
cursor = conn.cursor()

# List all features
cursor.execute("SELECT name, description, files FROM features")
rows = cursor.fetchall()

print(f"Found {len(rows)} features\n")
for name, desc, files in rows:
    print(f"Name: {name}")
    print(f"Desc: {desc[:100] if desc else 'None'}...")
    print(f"Files: {files[:100] if files else 'None'}...")
    print("---\n")

conn.close()
