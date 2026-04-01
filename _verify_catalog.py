import sqlite3, sys
sys.stdout.reconfigure(encoding='utf-8')
conn = sqlite3.connect('feature_catalog_master.sqlite3')
rows = conn.execute(
    "SELECT id, name, status, source FROM features WHERE source IN ('commit bfff7f2','commit 8ce44c7') ORDER BY id"
).fetchall()
for r in rows:
    print(r)
conn.close()
