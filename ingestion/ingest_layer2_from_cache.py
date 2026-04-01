"""
Layer 2 Triplet BM25 Re-Ingestion (Using Cached Mappings)

Efficient approach using cached chunk→triplet mappings:
1. Load chunk_to_triplets.msgpack (chunk_id → [triplet_ids])
2. Load triplet_bm25_index.msgpack (triplet texts)
3. Aggregate triplets by chunk with BERT tokenization
4. Compute BM25 scores using direct IDF/TF computation (O(n × m))
5. Ingest into layer2_triplet_bm25

Fixes:
- Uses cached mappings (no re-derivation needed)
- BERT WordPiece vocabulary (token IDs 0-30521)
- Direct TF-IDF computation (avoids O(n²) bug)
"""

import msgpack
import psycopg2
from psycopg2.extras import Json
from transformers import BertTokenizer
from rank_bm25 import BM25Okapi
import numpy as np
from tqdm import tqdm
from collections import defaultdict

# ============= CONFIGURATION =============
DB_CONFIG = {
    'dbname': 'langchain',
    'user': 'langchain',
    'password': 'langchain',
    'host': 'localhost',
    'port': 5432
}

CHECKPOINT_DIR = 'checkpoints'
CHUNK_TO_TRIPLETS = f'{CHECKPOINT_DIR}/chunk_to_triplets.msgpack'
TRIPLET_INDEX = f'{CHECKPOINT_DIR}/triplet_bm25_index.msgpack'

# ============= DATA LOADING =============

def load_chunk_to_triplets():
    """Load chunk_id → [triplet_ids] mapping from checkpoint."""
    print("Loading chunk→triplet mapping...")
    with open(CHUNK_TO_TRIPLETS, 'rb') as f:
        data = msgpack.load(f)
    print(f"  ✓ Loaded {len(data):,} chunk→triplet mappings from cache")
    return data

def get_valid_chunk_ids():
    """Get valid chunk_ids from arxiv_chunks table."""
    print("Loading valid chunk_ids from arxiv_chunks...")
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    
    cursor.execute("SELECT chunk_id FROM arxiv_chunks ORDER BY chunk_id")
    chunk_ids = [row[0] for row in cursor.fetchall()]
    
    cursor.close()
    conn.close()
    
    print(f"  ✓ Loaded {len(chunk_ids):,} valid chunk_ids")
    return chunk_ids

def load_triplet_texts():
    """Load triplet texts from checkpoint."""
    print("Loading triplet texts...")
    with open(TRIPLET_INDEX, 'rb') as f:
        data = msgpack.load(f)
    
    triplet_texts = data['triplet_texts']
    print(f"  ✓ Loaded {len(triplet_texts):,} triplet texts")
    return triplet_texts

# ============= TOKENIZATION =============

def init_bert_tokenizer():
    """Initialize BERT WordPiece tokenizer."""
    print("Initializing BERT tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    print(f"  ✓ Vocab size: {tokenizer.vocab_size:,} (IDs 0-{tokenizer.vocab_size-1})")
    return tokenizer

def tokenize_triplets(triplet_texts, chunk_to_triplets_cache, valid_chunk_ids, tokenizer):
    """
    Tokenize aggregated triplets for each chunk using valid chunk_ids.
    
    Args:
        triplet_texts: list of triplet text strings
        chunk_to_triplets_cache: dict from msgpack (has doubled format keys)
        valid_chunk_ids: list of correct chunk_ids from arxiv_chunks
        tokenizer: BERT tokenizer
    
    Returns:
        dict: chunk_id → [token_ids]
    """
    print("\nTokenizing aggregated triplets per chunk...")
    chunk_tokens = {}
    not_found = 0
    
    for chunk_id in tqdm(valid_chunk_ids, desc="Tokenizing"):
        # Try direct lookup first
        if chunk_id in chunk_to_triplets_cache:
            triplet_ids = chunk_to_triplets_cache[chunk_id]
        else:
            # Cache file has doubled format: "paperid_sX_cY_sX_cY"
            # arxiv_chunks has single format: "paperid_sX_cY"
            # Transform: "1301_3781_s0_c0" → "1301_3781_s0_c0_s0_c0"
            parts = chunk_id.split('_')
            if len(parts) >= 4:  # paperid_sX_cY format
                paper_part = '_'.join(parts[:-2])  # e.g., "1301_3781"
                section_chunk = '_'.join(parts[-2:])  # e.g., "s0_c0"
                doubled_key = f"{paper_part}_{section_chunk}_{section_chunk}"
                
                if doubled_key in chunk_to_triplets_cache:
                    triplet_ids = chunk_to_triplets_cache[doubled_key]
                else:
                    not_found += 1
                    chunk_tokens[chunk_id] = []
                    continue
            else:
                not_found += 1
                chunk_tokens[chunk_id] = []
                continue
        
        # Aggregate all triplets for this chunk
        aggregated_text = ' '.join([triplet_texts[tid] for tid in triplet_ids])
        
        # Tokenize with BERT (no [CLS]/[SEP], just token IDs)
        tokens = tokenizer.encode(
            aggregated_text,
            add_special_tokens=False,
            truncation=True,
            max_length=512
        )
        
        chunk_tokens[chunk_id] = tokens
    
    print(f"  ✓ Tokenized {len(chunk_tokens):,} chunks")
    if not_found > 0:
        print(f"  ⚠ {not_found:,} chunks not found in cache (empty vectors)")
    return chunk_tokens

# ============= BM25 COMPUTATION =============

def build_bm25_index(chunk_tokens):
    """
    Build BM25 index from chunk tokens.
    
    Returns:
        BM25Okapi: BM25 index
        list: chunk_ids in corpus order
    """
    print("\nBuilding BM25 index...")
    
    # Create corpus (list of token lists) and maintain chunk_id order
    chunk_ids = list(chunk_tokens.keys())
    corpus = [chunk_tokens[cid] for cid in chunk_ids]
    
    # Build BM25 index
    bm25 = BM25Okapi(corpus)
    
    print(f"  ✓ Built BM25 index for {len(corpus):,} documents")
    print(f"  ✓ Average document length: {bm25.avgdl:.1f} tokens")
    
    return bm25, chunk_ids

def compute_bm25_vectors(bm25, chunk_tokens, chunk_ids):
    """
    Compute BM25 sparse vectors for all chunks using direct IDF/TF computation.
    
    Avoids O(n²) bug by computing per-document scores directly.
    
    Returns:
        dict: chunk_id → {token_id: score}
    """
    print("\nComputing BM25 sparse vectors...")
    
    # Pre-compute IDF dictionary for all tokens in corpus
    # bm25.idf is a defaultdict mapping token_id → IDF score
    
    vectors = {}
    
    for chunk_id in tqdm(chunk_ids, desc="Computing"):
        doc_tokens = chunk_tokens[chunk_id]
        
        if not doc_tokens:
            vectors[chunk_id] = {}
            continue
        
        # Compute BM25 score for each unique token in document
        sparse_vector = {}
        
        for token in set(doc_tokens):
            # Get IDF for this token
            idf = bm25.idf.get(token, 0)
            
            if idf > 0:
                # Compute TF component
                tf = doc_tokens.count(token)
                
                # BM25 formula: IDF(token) × TF(token, doc)
                # (rank_bm25 handles k1, b internally via get_scores)
                # For sparse storage, we just need the weighted term frequency
                score = idf * tf
                
                sparse_vector[int(token)] = float(score)
        
        vectors[chunk_id] = sparse_vector
    
    print(f"  ✓ Computed {len(vectors):,} sparse vectors")
    
    # Statistics
    non_empty = sum(1 for v in vectors.values() if v)
    avg_nnz = np.mean([len(v) for v in vectors.values() if v]) if non_empty > 0 else 0
    
    print(f"  ✓ Non-empty vectors: {non_empty:,}/{len(vectors):,}")
    print(f"  ✓ Average non-zero elements: {avg_nnz:.1f}")
    
    return vectors

# ============= DATABASE INGESTION =============

def ingest_to_layer2(vectors):
    """
    Ingest BM25 sparse vectors into layer2_triplet_bm25 table.
    
    Strategy: UPSERT (INSERT ... ON CONFLICT UPDATE)
    """
    print("\n" + "="*60)
    print("INGESTING INTO LAYER 2 TRIPLET BM25")
    print("="*60)
    
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    
    try:
        # Prepare batch insert with UPSERT
        insert_sql = """
            INSERT INTO layer2_triplet_bm25 (chunk_id, triplet_bm25_vector)
            VALUES (%s, %s)
            ON CONFLICT (chunk_id) DO UPDATE
            SET triplet_bm25_vector = EXCLUDED.triplet_bm25_vector,
                created_at = NOW()
        """
        
        # Batch insert for performance
        batch = []
        batch_size = 1000
        
        for chunk_id, sparse_vector in tqdm(vectors.items(), desc="Ingesting"):
            # Convert sparse vector to JSONB format using psycopg2's Json adapter
            batch.append((chunk_id, Json(sparse_vector)))
            
            if len(batch) >= batch_size:
                cursor.executemany(insert_sql, batch)
                conn.commit()
                batch = []
        
        # Insert remaining
        if batch:
            cursor.executemany(insert_sql, batch)
            conn.commit()
        
        print(f"\n✅ LAYER 2 INGESTION COMPLETE")
        print(f"   Inserted/Updated: {len(vectors):,} rows")
        
        # Verify
        cursor.execute("SELECT COUNT(*) FROM layer2_triplet_bm25")
        count = cursor.fetchone()[0]
        print(f"   Total rows in layer2_triplet_bm25: {count:,}")
        
    except Exception as e:
        conn.rollback()
        print(f"\n❌ ERROR during ingestion: {e}")
        raise
    finally:
        cursor.close()
        conn.close()

# ============= MAIN PIPELINE =============

def main():
    print("="*60)
    print("LAYER 2 TRIPLET BM25 RE-INGESTION (FROM CACHE)")
    print("="*60)
    print()
    
    # Step 1: Get valid chunk_ids from database
    valid_chunk_ids = get_valid_chunk_ids()
    
    # Step 2: Load cached data
    chunk_to_triplets_cache = load_chunk_to_triplets()
    triplet_texts = load_triplet_texts()
    
    # Step 3: Initialize BERT tokenizer
    tokenizer = init_bert_tokenizer()
    
    # Step 4: Tokenize aggregated triplets per chunk (using valid chunk_ids)
    chunk_tokens = tokenize_triplets(triplet_texts, chunk_to_triplets_cache, valid_chunk_ids, tokenizer)
    
    # Step 5: Build BM25 index
    bm25, chunk_ids = build_bm25_index(chunk_tokens)
    
    # Step 6: Compute BM25 sparse vectors
    vectors = compute_bm25_vectors(bm25, chunk_tokens, chunk_ids)
    
    # Step 7: Ingest into database
    ingest_to_layer2(vectors)
    
    print("\n" + "="*60)
    print("✅ ALL DONE - Layer 2 now uses BERT vocabulary!")
    print("="*60)
    print()
    print("Next step: Validate with:")
    print("  python validate_layer_by_layer.py")

if __name__ == '__main__':
    main()
