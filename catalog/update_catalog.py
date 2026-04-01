"""Add Layer 2 Triplet BM25 feature to catalog with corrected understanding."""

import sqlite3
from datetime import datetime

db_path = 'feature_catalog.sqlite3'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Check if this feature already exists
cursor.execute("SELECT id FROM features WHERE name LIKE '%Layer 2%Triplet%BM25%'")
existing = cursor.fetchone()

DESCRIPTION = """Layer 2: Triplet BM25 Sparse Vector Ingestion

Complete BM25 sparse vector extraction for aggregated semantic triplets.

**Correct Architecture** (O(n) complexity):
1. Load existing vocabulary from chunk_bm25_sparse.msgpack (don't rebuild)
2. Aggregate all triplet texts per chunk into one concatenated text
3. Build BM25 index ONCE on all aggregated texts (161k documents)
4. Extract sparse vectors from BM25 internal structures (doc_freqs[i] + idf)
5. DO NOT call bm25.get_scores() per chunk (that's O(n²))

**Critical Mistake to Avoid**:
❌ DO NOT call bm25.get_scores(tokens) per chunk during ingestion!
  - get_scores() computes query similarity against entire corpus → O(n) per call
  - Calling per chunk → O(n²) total (30+ seconds PER chunk, 1344 hours!)
  - For ingestion: Extract from doc_freqs directly → O(n) total (~30 seconds)

**Complexity**: O(n), ~30 seconds total (same as layer1_bm25_sparse)
**Database**: 161,389 rows in layer2_triplet_bm25 table with GIN index
**Implementation**: ingest_layer2_triplet_bm25_final.py
"""

if existing:
    print(f"⚠️ Feature already exists with ID {existing[0]}, updating...")
    feature_id = existing[0]
    cursor.execute("""
        UPDATE features 
        SET description = ?,
            updated_at = ?,
            status = 'DONE',
            validation_notes = 'Corrected: O(n) approach using BM25 internal structures, NOT per-chunk scoring'
        WHERE id = ?
    """, (DESCRIPTION, datetime.now().isoformat(), feature_id))
else:
    print("➕ Adding new Layer 2 Triplet BM25 Ingestion feature...")
    cursor.execute("""
        INSERT INTO features (name, description, status, created_at, updated_at, validation_notes)
        VALUES (?, ?, 'DONE', ?, ?, ?)
    """, (
        'Layer 2: Triplet BM25 Sparse Vector Ingestion',
        DESCRIPTION,
        datetime.now().isoformat(),
        datetime.now().isoformat(),
        'Runtime verified: ~30 seconds O(n) complexity. Correct: build BM25 once, extract from internal structures.'
    ))
    feature_id = cursor.lastrowid

conn.commit()

# Also add architectural decision documenting the correction
cursor.execute("""
    INSERT INTO architectural_decisions (decision, rationale, before_state, after_state, decision_timestamp)
    VALUES (?, ?, ?, ?, ?)
""", (
    'Layer 2 Triplet BM25: Aggregate-then-Index vs Per-Document Scoring',
    """Problem: Initially attempted to score each chunk against corpus using bm25.get_scores(), resulting in O(n²) complexity (30+ seconds PER chunk).

Root Cause: Misunderstood BM25 API - get_scores() is for query-time similarity, NOT ingestion-time vector extraction.

Correct Approach:
1. Aggregate triplet texts to chunk level (concatenate)
2. Build BM25 index ONCE on all aggregated texts
3. Extract sparse vectors from bm25.doc_freqs[i] + bm25.idf (internal structures)
4. Complexity: O(n) total, ~30 seconds

Key Insight: For ingestion, extract document representation from BM25 internals. For query, use get_scores() to find similar docs.""",
    """Failed attempts:
- precompute_triplet_bm25_lookup.py (O(n²) pre-computation, hung)
- ingest_layer2_triplet_bm25_correct.py (29.99s/it × 161k = 1344 hours)""",
    """Correct implementation:
- ingest_layer2_triplet_bm25_final.py
- Aggregate triplets → build BM25 once → extract from internal structures
- Runtime: ~30 seconds (O(n) complexity)""",
    datetime.now().isoformat()
))

conn.commit()
conn.close()

print("\n✅ Feature catalog updated!")
print(f"   Feature ID: {feature_id}")
print("   Status: DONE")
print("   Architectural decision logged")
print("\n📄 Documentation updated:")
print("   - GIST_THREE_LAYER_ARCHITECTURE.md (Layer 2 Triplet BM25 section)")
print("   - feature_catalog.sqlite3 (new feature + architectural decision)")
