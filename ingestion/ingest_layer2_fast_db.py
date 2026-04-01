"""
LAYER 2: TRIPLET BM25 SPARSE VECTORS - FAST IMPLEMENTATION (DB-DIRECT)

Fixes vocabulary mismatch by using BERT tokenization consistently.
Loads data directly from PostgreSQL for simplicity.
Optimized to process 161K chunks in ~5-10 minutes.
"""

import msgpack
import psycopg2
from psycopg2.extras import execute_batch
from collections import Counter, defaultdict
import numpy as np
from tqdm import tqdm
import math
from transformers import BertTokenizer


def load_bert_vocab(vocab_path='checkpoints/chunk_bm25_sparse.msgpack'):
    """Load BERT vocabulary from chunk_bm25_sparse.msgpack"""
    print("Loading BERT vocabulary...")
    with open(vocab_path, 'rb') as f:
        data = msgpack.unpack(f, strict_map_key=False)
    
    vocab = data['vocab']  # token_str → token_id
    
    print(f"  ✓ BERT Vocabulary: {len(vocab):,} tokens")
    return vocab


def load_data_from_db():
    """Load chunk IDs and aggregated triplet texts from database"""
    print("\nLoading data from PostgreSQL...")
    
    conn = psycopg2.connect(
        dbname='langchain',
        user='langchain',
        password='langchain',
        host='localhost',
        port=5432
    )
    cursor = conn.cursor()
    
    # Load chunk_id and triplet_ids from arxiv_chunks
    print("  Loading chunks and triplet mappings...")
    cursor.execute("""
        SELECT chunk_id, triplet_ids
        FROM arxiv_chunks
        WHERE triplet_ids IS NOT NULL AND triplet_ids != '{}'
        ORDER BY chunk_id
    """)
    
    chunks_with_triplets = cursor.fetchall()
    print(f"    ✓ Loaded {len(chunks_with_triplets):,} chunks with triplets")
    
    # Load triplet texts
    print("  Loading triplet texts...")
    cursor.execute("SELECT triplet_id, triplet_text FROM arxiv_triplets")
    
    triplet_dict = {row[0]: row[1] for row in cursor.fetchall()}
    print(f"    ✓ Loaded {len(triplet_dict):,} triplet texts")
    
    cursor.close()
    conn.close()
    
    # Aggregate triplets to chunk level
    print("\n  Aggregating triplets to chunks...")
    chunk_aggregated = []
    
    for chunk_id, triplet_ids in tqdm(chunks_with_triplets, desc="    Processing"):
        # Concatenate all triplet texts for this chunk
        triplet_texts = [triplet_dict.get(tid, '') for tid in triplet_ids]
        combined_text = ' '.join([t for t in triplet_texts if t])
        
        if combined_text:
            chunk_aggregated.append((chunk_id, combined_text))
    
    print(f"    ✓ Aggregated: {len(chunk_aggregated):,} chunks")
    
    return chunk_aggregated


def compute_sparse_vectors_fast(chunk_aggregated, vocab, tokenizer):
    """
    Fast TF-IDF computation with BERT tokenization.
    
    Optimization: Compute IDF once, then iterate through documents computing TF.
    Avoids per-document BM25 scoring overhead.
    """
    print("\nComputing sparse vectors (fast TF-IDF with BERT)...")
    
    # Step 1: Tokenize all documents
    print("  Step 1: BERT tokenization...")
    all_docs_tokens = []
    token_to_id_mappings = []
    
    for chunk_id, text in tqdm(chunk_aggregated, desc="    Tokenizing"):
        # Use BERT tokenizer for subword tokenization
        bert_tokens = tokenizer.tokenize(text.lower())
        
        # Filter to vocabulary
        valid_tokens = [t for t in bert_tokens if t in vocab]
        
        # Store tokens and mapping
        all_docs_tokens.append(valid_tokens)
        token_to_id_mappings.append({token: vocab[token] for token in valid_tokens})
    
    print(f"    ✓ Tokenized {len(all_docs_tokens):,} documents")
    
    # Step 2: Compute document frequency (DF) for IDF calculation
    print("  Step 2: Computing document frequencies...")
    doc_freq = defaultdict(int)
    total_docs = len(all_docs_tokens)
    
    for doc_tokens in tqdm(all_docs_tokens, desc="    Counting"):
        unique_tokens = set(doc_tokens)
        for token in unique_tokens:
            doc_freq[token] += 1
    
    print(f"    ✓ Total unique tokens: {len(doc_freq):,}")
    
    # Step 3: Compute IDF
    print("  Step 3: Computing IDF...")
    idf = {}
    for token, df in doc_freq.items():
        # BM25 IDF formula: log((N - df + 0.5) / (df + 0.5) + 1)
        idf[token] = math.log((total_docs - df + 0.5) / (df + 0.5) + 1.0)
    
    print(f"    ✓ Computed IDF for {len(idf):,} tokens")
    
    # Step 4: Compute TF and combine with IDF
    print("  Step 4: Computing TF-IDF sparse vectors...")
    sparse_vectors = []
    
    # BM25 parameters
    k1 = 1.5
    b = 0.75
    avgdl = np.mean([len(doc) for doc in all_docs_tokens])
    
    for i, ((chunk_id, _), doc_tokens, token_mapping) in enumerate(
        tqdm(
            zip(chunk_aggregated, all_docs_tokens, token_to_id_mappings),
            total=len(chunk_aggregated),
            desc="    Computing"
        )
    ):
        # Compute term frequencies
        tf_counter = Counter(doc_tokens)
        doc_len = len(doc_tokens)
        
        sparse_dict = {}
        
        for token, tf in tf_counter.items():
            # Get BERT token ID
            bert_id = token_mapping.get(token)
            if bert_id is None:
                continue
            
            # BM25 scoring formula
            idf_score = idf.get(token, 0)
            
            # TF component with document length normalization
            tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / avgdl)))
            
            # Combined score
            score = idf_score * tf_norm
            
            if score > 0:
                sparse_dict[str(bert_id)] = float(score)
        
        sparse_vectors.append((chunk_id, sparse_dict))
    
    print(f"    ✓ Computed {len(sparse_vectors):,} sparse vectors")
    
    return sparse_vectors


def insert_to_database(sparse_vectors):
    """Insert sparse vectors into layer2_triplet_bm25 table"""
    print("\nInserting into PostgreSQL...")
    
    conn = psycopg2.connect(
        dbname='langchain',
        user='langchain',
        password='langchain',
        host='localhost',
        port=5432
    )
    cursor = conn.cursor()
    
    # Clear existing data
    print("  Clearing existing Layer 2 data...")
    cursor.execute("DELETE FROM layer2_triplet_bm25")
    conn.commit()
    print(f"    ✓ Cleared table")
    
    # Prepare data for batch insert
    print("  Preparing batch insert...")
    insert_data = [
        (chunk_id, msgpack.packb(sparse_dict))
        for chunk_id, sparse_dict in sparse_vectors
    ]
    
    # Batch insert
    print(f"  Inserting {len(insert_data):,} rows...")
    execute_batch(
        cursor,
        "INSERT INTO layer2_triplet_bm25 (chunk_id, triplet_bm25_vector) VALUES (%s, %s)",
        insert_data,
        page_size=1000
    )
    conn.commit()
    
    print(f"    ✓ Inserted {len(insert_data):,} rows")
    
    # Verify
    cursor.execute("SELECT COUNT(*) FROM layer2_triplet_bm25")
    count = cursor.fetchone()[0]
    print(f"    ✓ Verification: {count:,} rows in table")
    
    cursor.close()
    conn.close()


def verify_vocabulary_consistency():
    """Verify that token IDs are in BERT range"""
    print("\nVerifying BERT vocabulary consistency...")
    
    conn = psycopg2.connect(
        dbname='langchain',
        user='langchain',
        password='langchain',
        host='localhost',
        port=5432
    )
    cursor = conn.cursor()
    
    # Sample 100 token IDs from database
    cursor.execute("""
        SELECT chunk_id, jsonb_object_keys(triplet_bm25_vector) AS token_id
        FROM layer2_triplet_bm25
        LIMIT 100
    """)
    
    samples = cursor.fetchall()
    token_ids = [int(row[1]) for row in samples]
    
    # Check ID ranges
    bert_range_ids = [tid for tid in token_ids if 0 <= tid <= 30521]
    
    print(f"\nToken ID Analysis (sample of {len(token_ids)}):")
    print(f"  BERT range (0-30521): {len(bert_range_ids)} / {len(token_ids)} ({100*len(bert_range_ids)/len(token_ids):.1f}%)")
    print(f"  Min ID: {min(token_ids)}")
    print(f"  Max ID: {max(token_ids)}")
    print(f"  Sample IDs: {sorted(set(token_ids))[:10]}")
    
    if len(bert_range_ids) == len(token_ids):
        print(f"\n✓ BERT vocabulary consistency: YES")
        print("  All token IDs are in BERT range (0-30521)")
    else:
        print(f"\n✗ BERT vocabulary consistency: NO")
        print(f"  Found {len(token_ids) - len(bert_range_ids)} IDs outside BERT range")
    
    cursor.close()
    conn.close()


def main():
    print("=" * 60)
    print("LAYER 2: TRIPLET BM25 - FAST BERT IMPLEMENTATION")
    print("=" * 60)
    print("\nOptimized for speed: ~5-10 minutes for 161K chunks")
    print("Uses direct TF-IDF computation with BERT tokenization")
    print()
    
    # Load vocabulary
    vocab = load_bert_vocab()
    
    # Initialize BERT tokenizer
    print("\nInitializing BERT tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    print("  ✓ BERT tokenizer loaded")
    
    # Load data from database
    chunk_aggregated = load_data_from_db()
    
    # Compute sparse vectors (FAST)
    sparse_vectors = compute_sparse_vectors_fast(chunk_aggregated, vocab, tokenizer)
    
    # Insert to database
    insert_to_database(sparse_vectors)
    
    # Verify
    verify_vocabulary_consistency()
    
    print("\n" + "=" * 60)
    print("✓ LAYER 2 INGESTION COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
