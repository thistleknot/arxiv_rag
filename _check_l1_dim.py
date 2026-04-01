import psycopg2
from pgvector.psycopg2 import register_vector

conn = psycopg2.connect(host='localhost', port=5432, dbname='langchain',
                        user='langchain', password='langchain')
register_vector(conn)
cur = conn.cursor()

cur.execute("SELECT vector_dims(embedding) FROM layer1_embeddings_128d LIMIT 1")
dim = cur.fetchone()[0]
print(f"layer1_embeddings_128d actual dim: {dim}")

# Also check the indexdef to see if there's an HNSW or IVFFlat index
cur.execute("""
    SELECT indexname, indexdef FROM pg_indexes
    WHERE tablename = 'layer1_embeddings_128d'
""")
for row in cur.fetchall():
    print(f"  index: {row[0]}: {row[1][:120]}")

# Check if there are any vocab/term_id tables for lemmatized BM25
cur.execute("""
    SELECT table_name FROM information_schema.tables
    WHERE table_schema = 'public'
    AND (table_name LIKE '%bm25%' OR table_name LIKE '%term%' OR table_name LIKE '%vocab%' OR table_name LIKE '%lemma%')
    ORDER BY table_name
""")
print("BM25/term/vocab/lemma tables:", [r[0] for r in cur.fetchall()])

conn.close()
