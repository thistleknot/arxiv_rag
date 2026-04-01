"""Inspect the master feature catalog."""
import sqlite3

conn = sqlite3.connect('feature_catalog_master.sqlite3')
conn.row_factory = sqlite3.Row

tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
print('Tables:', [t['name'] for t in tables])

rows = conn.execute('SELECT id, name, category, status FROM features ORDER BY id').fetchall()
print(f'Total features: {len(rows)}')
for r in rows:
    print(f'  [{r["id"]}] {r["category"]:20s} {r["status"]:12s} {r["name"]}')

conn.close()
