import sqlite3

conn = sqlite3.connect('feature_catalog_master.sqlite3')

print("=== Pipeline-related features ===")
rows = conn.execute("""
    SELECT id, name, category, status, substr(coalesce(description,''),1,100)
    FROM features
    WHERE lower(name||' '||coalesce(category,'')||' '||coalesce(description,'')) LIKE '%layer%'
       OR lower(name) LIKE '%gist%' OR lower(name) LIKE '%rrf%'
       OR lower(name) LIKE '%ecdf%' OR lower(name) LIKE '%retriev%'
       OR lower(category) LIKE '%retriev%' OR lower(category) LIKE '%pipeline%'
    ORDER BY category, name
""").fetchall()
for r in rows:
    print(f"  [{r[0]}] {r[2]} | {r[1]} | {r[3]} | {r[4]}")
print(f"\nTotal hits: {len(rows)}")

print("\n=== All categories ===")
for c in conn.execute("SELECT category, COUNT(*) FROM features GROUP BY category ORDER BY category").fetchall():
    print(f"  {c[0]}: {c[1]}")

conn.close()
