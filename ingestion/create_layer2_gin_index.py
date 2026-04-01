"""Create GIN index for layer2_triplet_bm25."""
import psycopg2

conn = psycopg2.connect(
    dbname='langchain',
    user='langchain',
    password='langchain',
    host='localhost',
    port=5432
)

cursor = conn.cursor()

print("Creating GIN index on layer2_triplet_bm25.triplet_bm25_vector...")
print("(This may take 2-3 minutes for 161,389 rows)")
print()

cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_layer2_triplet_bm25_gin
    ON layer2_triplet_bm25
    USING gin (triplet_bm25_vector)
""")

conn.commit()

# Get index size
cursor.execute("""
    SELECT pg_size_pretty(pg_relation_size('idx_layer2_triplet_bm25_gin'))
""")
size = cursor.fetchone()[0]

print(f"✅ Index created successfully!")
print(f"   Index size: {size}")

# Get total table size now
cursor.execute("""
    SELECT pg_size_pretty(pg_total_relation_size('layer2_triplet_bm25'))
""")
total_size = cursor.fetchone()[0]
print(f"   Total table size: {total_size}")

cursor.close()
conn.close()
