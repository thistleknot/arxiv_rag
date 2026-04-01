import sqlite3

conn = sqlite3.connect(r'C:\Users\user\arxiv_id_lists\feature_catalog_master.sqlite3')
cur = conn.cursor()

cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
print('Tables:', [r[0] for r in cur.fetchall()])

# Get column names for features
cur.execute("PRAGMA table_info(features)")
cols = [r[1] for r in cur.fetchall()]
print('Columns:', cols)

# Find retrieval/layer architecture features
cur.execute("""
SELECT id, name, category, description, definition, status
FROM features
WHERE lower(name) LIKE '%retriev%' 
   OR lower(name) LIKE '%layer%'
   OR lower(category) LIKE '%retriev%'
   OR lower(name) LIKE '%gist%'
   OR lower(name) LIKE '%top_k%'
   OR lower(name) LIKE '%l1%'
   OR lower(name) LIKE '%l2%'
   OR lower(name) LIKE '%l3%'
   OR lower(name) LIKE '%bm25%'
   OR lower(name) LIKE '%embed%'
   OR lower(name) LIKE '%ecdf%'
   OR lower(name) LIKE '%rrf%'
   OR lower(name) LIKE '%fib%'
   OR lower(name) LIKE '%section%'
   OR lower(name) LIKE '%paper%'
ORDER BY id
LIMIT 100
""")
rows = cur.fetchall()
print(f'\nFound {len(rows)} retrieval features:')
for r in rows:
    desc = str(r[3] or '').replace('\n', ' ')[:120]
    defn = str(r[4] or '').replace('\n', ' ')[:120]
    print(f'  [{r[0]}] {r[1]} | cat={r[2]} | {r[5]}')
    if desc:
        print(f'    DESC: {desc}')
    if defn:
        print(f'    DEFN: {defn}')

conn.close()
