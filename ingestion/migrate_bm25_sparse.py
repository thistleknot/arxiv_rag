"""
Migration: Add bm25_sparse JSONB column to arxiv_chunks and populate it.

The _retrieve_bm25() query uses JSONB operators (->> and ?), 
so the column type must be JSONB, not sparsevec.
"""
import psycopg2
import msgpack
from collections import Counter
from tqdm import tqdm

DB = dict(host='localhost', port=5432, dbname='langchain', user='langchain', password='langchain')
TABLE = 'arxiv_chunks'
VOCAB_PATH = 'bm25_vocab.msgpack'
BATCH_SIZE = 500

def main():
    # Load BM25 vocab
    print("Loading BM25 vocab...")
    with open(VOCAB_PATH, 'rb') as f:
        bm25_data = msgpack.unpack(f, raw=False)
    
    vocab = bm25_data['vocab']       # token -> id
    idf = bm25_data['idf']           # id -> idf value
    avgdl = bm25_data['avgdl']
    k1 = bm25_data.get('k1', 1.2)
    b = bm25_data.get('b', 0.75)
    
    # idf keys might be strings from msgpack
    idf = {int(k): v for k, v in idf.items()}
    
    print(f"  Vocab size: {len(vocab)}, avgdl: {avgdl:.1f}, k1: {k1}, b: {b}")
    
    # Load tokenizer
    print("Loading BERT tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    conn = psycopg2.connect(**DB)
    conn.autocommit = False
    cur = conn.cursor()
    
    # Check if column exists
    cur.execute("""
        SELECT column_name FROM information_schema.columns 
        WHERE table_name=%s AND column_name='bm25_sparse'
    """, (TABLE,))
    
    if cur.fetchone():
        print("bm25_sparse column already exists. Repopulating...")
    else:
        print("Adding bm25_sparse JSONB column...")
        cur.execute(f"ALTER TABLE {TABLE} ADD COLUMN bm25_sparse JSONB DEFAULT '{{}}'::jsonb")
        conn.commit()
        print("  Column added.")
    
    # Fetch all chunks
    print("Fetching all chunk content...")
    cur.execute(f"SELECT chunk_id, content FROM {TABLE}")
    rows = cur.fetchall()
    print(f"  {len(rows)} chunks to process.")
    
    # Build sparse vectors
    print("Computing BM25 sparse vectors...")
    updates = []
    for chunk_id, content in tqdm(rows, desc="  Vectorizing"):
        tokens = tokenizer.tokenize(content.lower())
        token_ids = [vocab.get(t, -1) for t in tokens if t in vocab]
        
        dl = len(token_ids)
        tf = Counter(token_ids)
        
        sparse = {}
        for tid, freq in tf.items():
            if tid < 0 or tid not in idf:
                continue
            numerator = freq * (k1 + 1)
            denominator = freq + k1 * (1 - b + b * dl / avgdl)
            weight = idf[tid] * (numerator / denominator)
            if weight > 0:
                sparse[str(tid)] = round(weight, 4)
        
        import json
        updates.append((json.dumps(sparse), chunk_id))
    
    # Batch update
    print("Writing to database...")

    for i in tqdm(range(0, len(updates), BATCH_SIZE), desc="  Updating"):
        batch = updates[i:i + BATCH_SIZE]
        args_str = ",".join(
            cur.mogrify("(%s::jsonb, %s)", (json_str, cid)).decode()
            for json_str, cid in batch
        )
        cur.execute(f"""
            UPDATE {TABLE} AS t SET bm25_sparse = v.sparse
            FROM (VALUES {args_str}) AS v(sparse, cid)
            WHERE t.chunk_id = v.cid
        """)
        conn.commit()
    
    # Create GIN index for ? operator
    print("Creating GIN index on bm25_sparse...")
    cur.execute(f"""
        CREATE INDEX IF NOT EXISTS {TABLE}_bm25_gin_idx 
        ON {TABLE} USING gin (bm25_sparse)
    """)
    conn.commit()
    
    # Verify
    cur.execute(f"SELECT COUNT(*) FROM {TABLE} WHERE bm25_sparse != '{{}}'::jsonb")
    populated = cur.fetchone()[0]
    print(f"\nDone. {populated}/{len(rows)} chunks have non-empty BM25 vectors.")
    
    cur.close()
    conn.close()

if __name__ == '__main__':
    import psycopg2.extras
    main()
