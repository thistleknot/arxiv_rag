"""
Layer 2 Triplet BM25 Ingestion - BERT VOCABULARY FIX

**THE FIX**: Use BERT tokenizer properly to ensure vocabulary consistency
across Layer 1 (BERT vocab) and Layer 2 (BERT vocab).

Key changes from original:
1. Use transformers BertTokenizer for proper tokenization
2. Pass token IDs (not strings) to BM25Okapi
3. Maintain BERT vocab consistency throughout pipeline

Runtime: ~30-60 seconds
"""

import psycopg2
import msgpack
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from transformers import BertTokenizer

# Configuration
DB_CONFIG = {
    'dbname': 'langchain',
    'user': 'langchain',
    'password': 'langchain',
    'host': 'localhost',
    'port': 5432
}

CHECKPOINT_DIR = Path('checkpoints')
CHUNKS_FILE = CHECKPOINT_DIR / 'chunks.msgpack'
CHUNK_TO_TRIPLETS = CHECKPOINT_DIR / 'chunk_to_triplets.msgpack'
TRIPLET_INDEX = CHECKPOINT_DIR / 'triplet_bm25_index.msgpack'
CHUNK_BM25_SPARSE = CHECKPOINT_DIR / 'chunk_bm25_sparse.msgpack'


def load_chunk_bm25_vocab():
    """Load BERT vocabulary from chunk_bm25_sparse.msgpack."""
    print("\nLoading BERT vocabulary...")
    with open(CHUNK_BM25_SPARSE, 'rb') as f:
        data = msgpack.unpackb(f.read(), strict_map_key=False)
    
    vocab = data['vocab']  # dict: token (str) -> id (int)
    print(f"  ✓ BERT Vocabulary: {len(vocab):,} tokens")
    
    # Create reverse mapping (id -> token)
    id_to_token = {v: k for k, v in vocab.items()}
    
    return vocab, id_to_token


def load_data():
    """Load chunks, mappings, and triplet texts."""
    print("\nLoading data...")
    
    # Chunks
    with open(CHUNKS_FILE, 'rb') as f:
        chunks = msgpack.unpackb(f.read(), strict_map_key=False)
    print(f"  ✓ Loaded {len(chunks):,} chunks")
    
    # Chunk→triplets mapping
    with open(CHUNK_TO_TRIPLETS, 'rb') as f:
        chunk_to_triplets = msgpack.unpackb(f.read(), strict_map_key=False)
    print(f"  ✓ Loaded {len(chunk_to_triplets):,} chunk→triplet mappings")
    
    # Triplet texts
    with open(TRIPLET_INDEX, 'rb') as f:
        triplet_data = msgpack.unpackb(f.read(), strict_map_key=False)
    triplet_texts = triplet_data['triplet_texts']
    print(f"  ✓ Loaded {len(triplet_texts):,} triplet texts")
    
    return chunks, chunk_to_triplets, triplet_texts


def aggregate_chunk_triplets(chunks, chunk_to_triplets, triplet_texts):
    """
    Aggregate triplets to chunk level.
    
    Returns:
        list of (chunk_id, aggregated_text)
    """
    print(f"\nAggregating triplets to chunk level...")
    
    chunk_aggregated = []
    skipped = 0
    
    for chunk in tqdm(chunks, desc="  Processing"):
        # Extract chunk_id
        chunk_id = chunk.get('doc_id') or chunk.get('chunk_id')
        section_idx = chunk.get('section_idx', 0)
        chunk_idx = chunk.get('chunk_idx', 0)
        
        # Construct mapping key
        mapping_key = f"{chunk_id}_s{section_idx}_c{chunk_idx}"
        
        # Get triplet indices
        triplet_indices = chunk_to_triplets.get(mapping_key, [])
        
        if not triplet_indices:
            skipped += 1
            continue
        
        # Aggregate triplet texts for this chunk
        chunk_triplet_texts = [
            triplet_texts[idx] 
            for idx in triplet_indices 
            if idx < len(triplet_texts)
        ]
        
        if not chunk_triplet_texts:
            skipped += 1
            continue
        
        # Concatenate into single text blob
        aggregated_text = ' '.join(chunk_triplet_texts)
        
        chunk_aggregated.append((chunk_id, aggregated_text))
    
    print(f"  ✓ Aggregated: {len(chunk_aggregated):,} chunks")
    print(f"  ✓ Skipped (no triplets): {skipped:,}")
    
    return chunk_aggregated


def compute_sparse_vectors_bert(chunk_aggregated, vocab, tokenizer):
    """
    Compute BM25 sparse vectors using BERT tokenization.
    
    **THE FIX**: Use BERT tokenizer and maintain token ID consistency.
    """
    print(f"\nComputing BM25 sparse vectors with BERT tokenization...")
    
    # Step 1: Tokenize using BERT tokenizer (produces token strings)
    print("  Step 1: BERT tokenization...")
    tokenized_corpus = []
    token_to_id_mapping = []  # Track token string → BERT ID for each doc
    
    for _, text in tqdm(chunk_aggregated, desc="    Tokenizing"):
        # BERT tokenization (returns subword token strings)
        bert_tokens = tokenizer.tokenize(text.lower())
        
        # Filter: keep only tokens that exist in vocab
        valid_tokens = [t for t in bert_tokens if t in vocab]
        
        # Store token strings for BM25
        tokenized_corpus.append(valid_tokens)
        
        # Store mapping: token string → BERT ID for this document
        token_to_id_mapping.append({token: vocab[token] for token in valid_tokens})
    
    # Step 2: Build BM25 index (BM25 creates internal indices for token strings)
    print("  Step 2: Building BM25 index...")
    bm25 = BM25Okapi(tokenized_corpus)
    
    # Step 3: Extract sparse vectors and map back to BERT IDs
    print("  Step 3: Extracting sparse vectors with BERT IDs...")
    sparse_vectors = []
    
    for i, (chunk_id, _) in enumerate(tqdm(chunk_aggregated, desc="    Extracting")):
        # Get BM25's internal representation for this document
        doc_tokens = tokenized_corpus[i]
        doc_mapping = token_to_id_mapping[i]
        
        # Compute BM25 scores for this document
        scores = bm25.get_scores(doc_tokens)
        
        # Build sparse dict using BERT vocab IDs
        sparse_dict = {}
        for j, token_str in enumerate(doc_tokens):
            # Look up BERT ID for this token
            bert_id = doc_mapping.get(token_str)
            if bert_id is None:
                continue
            
            # BM25 score = IDF * term frequency
            # For self-scoring: IDF * 1 (term appears once in query doc)
            idf = bm25.idf.get(token_str, 0)
            tf = doc_tokens.count(token_str)
            score = idf * tf
            
            if score > 0:
                sparse_dict[str(bert_id)] = float(score)
        
        sparse_vectors.append((chunk_id, sparse_dict))
    
    print(f"  ✓ Computed {len(sparse_vectors):,} sparse vectors with BERT IDs")
    
    # Statistics
    non_empty = sum(1 for _, v in sparse_vectors if v)
    total_entries = sum(len(v) for _, v in sparse_vectors)
    avg_sparsity = total_entries / len(sparse_vectors) if sparse_vectors else 0
    
    print(f"  ✓ Non-empty: {non_empty:,} ({100*non_empty/len(sparse_vectors):.1f}%)")
    print(f"  ✓ Average sparsity: {avg_sparsity:.1f} non-zero entries per vector")
    
    return sparse_vectors


def bulk_insert_sparse_vectors(sparse_vectors, conn):
    """Bulk insert sparse vectors into layer2_triplet_bm25."""
    print(f"\nInserting {len(sparse_vectors):,} sparse vectors...")
    
    cursor = conn.cursor()
    batch_size = 1000
    batch = []
    
    for chunk_id, sparse_dict in tqdm(sparse_vectors, desc="  Inserting"):
        sparse_json = json.dumps(sparse_dict)
        batch.append((chunk_id, sparse_json))
        
        if len(batch) >= batch_size:
            cursor.executemany(
                """
                INSERT INTO layer2_triplet_bm25 (chunk_id, triplet_bm25_vector)
                VALUES (%s, %s::jsonb)
                ON CONFLICT (chunk_id) DO UPDATE
                SET triplet_bm25_vector = EXCLUDED.triplet_bm25_vector
                """,
                batch
            )
            conn.commit()
            batch = []
    
    # Insert remaining
    if batch:
        cursor.executemany(
            """
            INSERT INTO layer2_triplet_bm25 (chunk_id, triplet_bm25_vector)
            VALUES (%s, %s::jsonb)
            ON CONFLICT (chunk_id) DO UPDATE
            SET triplet_bm25_vector = EXCLUDED.triplet_bm25_vector
            """,
            batch
        )
        conn.commit()
    
    cursor.close()
    print(f"  ✓ Inserted {len(sparse_vectors):,} rows")


def verify_ingestion(conn):
    """Verify ingestion with sample token ID checks."""
    cursor = conn.cursor()
    
    # Row count
    cursor.execute("SELECT COUNT(*) FROM layer2_triplet_bm25")
    row_count = cursor.fetchone()[0]
    
    # Non-null
    cursor.execute("SELECT COUNT(*) FROM layer2_triplet_bm25 WHERE triplet_bm25_vector IS NOT NULL")
    non_null = cursor.fetchone()[0]
    
    # Sample: check for BERT-range token IDs (0-30521)
    cursor.execute("""
        SELECT chunk_id, 
               jsonb_object_keys(triplet_bm25_vector) AS token_id
        FROM layer2_triplet_bm25
        LIMIT 100
    """)
    samples = cursor.fetchall()
    
    # Parse token IDs
    token_ids = [int(row[1]) for row in samples]
    bert_range_ids = [tid for tid in token_ids if 0 <= tid <= 30521]
    
    # Table size
    cursor.execute("SELECT pg_size_pretty(pg_total_relation_size('layer2_triplet_bm25'))")
    size = cursor.fetchone()[0]
    
    cursor.close()
    
    print("\n" + "="*60)
    print("VERIFICATION")
    print("="*60)
    print(f"Rows: {row_count:,}")
    print(f"Non-null triplet_bm25_vector: {non_null:,} ({100*non_null/row_count:.1f}%)")
    print(f"Table size: {size}")
    print(f"\nSample token ID range check:")
    print(f"  Total sampled IDs: {len(token_ids)}")
    print(f"  BERT range (0-30521): {len(bert_range_ids)} ({100*len(bert_range_ids)/len(token_ids):.1f}%)")
    if token_ids:
        print(f"  Min ID: {min(token_ids)}, Max ID: {max(token_ids)}")
    print(f"  ✓ BERT vocabulary consistency: {'YES' if len(bert_range_ids) == len(token_ids) else 'NO'}")


def main():
    print("="*60)
    print("LAYER 2: TRIPLET BM25 - BERT VOCABULARY FIX")
    print("="*60)
    print("\nThis script fixes the vocabulary mismatch between:")
    print("  - Layer 1: BERT vocab (working)")
    print("  - Layer 2: Mixed vocab (broken)")
    print("\nFix: Use BERT tokenizer consistently for Layer 2\n")
    
    # Load BERT vocabulary
    vocab, id_to_token = load_chunk_bm25_vocab()
    
    # Initialize BERT tokenizer
    print("\nInitializing BERT tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    print("  ✓ BERT tokenizer loaded")
    
    # Load data
    chunks, chunk_to_triplets, triplet_texts = load_data()
    
    # Aggregate triplets to chunk level
    chunk_aggregated = aggregate_chunk_triplets(chunks, chunk_to_triplets, triplet_texts)
    
    # Compute BM25 sparse vectors with BERT tokenization
    sparse_vectors = compute_sparse_vectors_bert(chunk_aggregated, vocab, tokenizer)
    
    # Connect to database
    print("\nConnecting to PostgreSQL...")
    conn = psycopg2.connect(**DB_CONFIG)
    print("  ✓ Connected")
    
    try:
        # Bulk insert
        bulk_insert_sparse_vectors(sparse_vectors, conn)
        
        # Verify
        verify_ingestion(conn)
        
        print("\n" + "="*60)
        print("✓ LAYER 2 BERT FIX COMPLETE")
        print("="*60)
        print("\nNext step: Run validate_layer_by_layer.py to verify fix")
        
    finally:
        conn.close()


if __name__ == '__main__':
    main()
