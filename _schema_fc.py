import sqlite3

conn = sqlite3.connect('feature_catalog_master.sqlite3')

keywords = ['gist', 'collinear', 'bm25 arm', 'l2 arm', 'utility vec',
            'centroid norm', 'mean penalty', 'l2 seeds', 'all seeds',
            'layer2 expansion', 'gist select']

clauses = " OR ".join(
    f"lower(name||coalesce(description,'')||coalesce(definition,'')) LIKE '%{k}%'"
    for k in keywords
)
rows = conn.execute(
    f"SELECT id, name, status FROM features WHERE {clauses} ORDER BY name"
).fetchall()

print(f"Found {len(rows)} related entries:")
for r in rows:
    print(f"  [{r[0]}] {r[1]} | {r[2]}")

conn.close()
