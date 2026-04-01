import psycopg2
conn = psycopg2.connect(dbname='langchain', user='langchain', password='langchain', host='localhost', port=5432)
cur = conn.cursor()
cur.execute('DROP INDEX IF EXISTS layer2_triplet_bm25_vector_idx')
conn.commit()
cur.execute("SELECT indexname FROM pg_indexes WHERE tablename='layer2_triplet_bm25'")
print('Remaining indexes:', cur.fetchall())
conn.close()
print('Done')
