import psycopg2
conn = psycopg2.connect('postgresql://langchain:langchain@localhost:5432/langchain')
cur = conn.cursor()
cur.execute("""
    SELECT a.attname, a.atttypmod
    FROM pg_attribute a
    JOIN pg_class c ON a.attrelid = c.oid
    WHERE c.relname = 'arxiv_chunks' AND a.attname = 'embedding'
""")
row = cur.fetchone()
print('atttypmod:', row)
# atttypmod for vector = n_dims + 4 (pgvector encoding)
if row and row[1]:
    print('dim:', row[1] - 4)

# also check via actual data
cur.execute("SELECT embedding FROM arxiv_chunks LIMIT 1")
v = cur.fetchone()
if v and v[0]:
    print('actual len:', len(v[0]))
conn.close()
