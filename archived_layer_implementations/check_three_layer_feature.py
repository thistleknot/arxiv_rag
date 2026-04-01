import sqlite3

conn = sqlite3.connect('retriever_feature_catalog.sqlite3')
cur = conn.cursor()

# Check for three-layer, phi, Node2Vec, or graph-related features
cur.execute('''
    SELECT id, name, status, description 
    FROM features 
    WHERE name LIKE "%three%" 
       OR name LIKE "%layer%" 
       OR name LIKE "%phi%" 
       OR name LIKE "%node2vec%"
       OR name LIKE "%graph2vec%"
       OR name LIKE "%Graph Transformer%"
    ORDER BY id
''')

rows = cur.fetchall()
print(f'Found {len(rows)} matching features:\n')

for r in rows:
    print(f'ID: {r[0]}')
    print(f'Name: {r[1]}')
    print(f'Status: {r[2]}')
    print(f'Description: {r[3][:300] if r[3] else "(no description)"}...')
    print('-' * 80)

conn.close()
