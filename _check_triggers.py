import sqlite3
conn = sqlite3.connect("feature_catalog_master.sqlite3")
triggers = conn.execute("SELECT name FROM sqlite_master WHERE type='trigger'").fetchall()
print("Triggers:", [r[0] for r in triggers])
conn.close()
