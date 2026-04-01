import sqlite3
conn = sqlite3.connect("reasoning/triplet_cache.sqlite3")
tabs = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
for t in tabs:
    cnt = conn.execute(f"SELECT COUNT(*) FROM [{t[0]}]").fetchone()[0]
    print(f"{t[0]}: {cnt} rows")
conn.close()
