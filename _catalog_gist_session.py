"""Insert session GIST changes into feature_catalog_master.sqlite3."""
import sqlite3
from datetime import datetime

DB = "feature_catalog_master.sqlite3"
NOW = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

entries = [
    {
        "name": "GIST mean collinearity penalty",
        "category": "retrieval",
        "description": (
            "gist_select uses mean pairwise correlation with already-selected docs "
            "instead of max. mean_corr(d,S) = sum(sim(d,s) for s in S) / |S|. "
            "Prevents over-pruning when a candidate is similar to only one outlier "
            "in the selected set; gives unbiased VIF pressure proportional to "
            "true redundancy across all selected docs."
        ),
        "definition": "mean_corr(d, S) = (1/|S|) * sum_{s in S} cos(d, s)",
        "files": "retrieval/gist_retriever.py",
        "status": "done",
        "source": "commit bfff7f2",
    },
    {
        "name": "GIST explicit utility vector BM25 arm",
        "category": "retrieval",
        "description": (
            "L2 BM25 arm utility vector y uses explicit query-doc TF cosine instead of "
            "mixing RRF scores with BM25 scores. "
            "q_norm = normalised query term-weight vector over shared vocab; "
            "y_scores = tf_norm @ q_norm. "
            "Ensures y is a principled correlation(x_i, y) proxy on the same cosine scale "
            "as collinearity S, making GIST numerically coherent."
        ),
        "definition": "y_scores = tf_norm @ q_norm  (TF cosine with query term-weight vector)",
        "files": "retrieval/pgvector_retriever.py",
        "status": "done",
        "source": "commit bfff7f2",
    },
    {
        "name": "GIST explicit utility vector dense arm",
        "category": "retrieval",
        "description": (
            "L2 dense arm utility vector y uses explicit embedding cosine with seed centroid "
            "instead of mixing dense_score proxy. "
            "centroid_norm = normalised mean of seed embeddings; "
            "y_scores = emb_norm @ centroid_norm. "
            "Ensures utility y is on the same cosine scale as collinearity S."
        ),
        "definition": "y_scores = emb_norm @ centroid_norm  (embedding cosine to seed centroid)",
        "files": "retrieval/pgvector_retriever.py",
        "status": "done",
        "source": "commit bfff7f2",
    },
    {
        "name": "BM25 arm ALL seeds in GIST pool",
        "category": "retrieval",
        "description": (
            "BM25 L2 arm includes all hybrid seeds in the GIST candidate pool regardless of "
            "whether they have a precomputed bm25_vec_map entry. "
            "Previously seeds were silently dropped if missing from the DB vector map, "
            "causing the arm to return ~144 instead of ~288 candidates. "
            "Fix: seeds forwarded directly; TF coverage matrix built over actual query vocab "
            "from the seeds+new-candidates pool."
        ),
        "definition": "pool = seeds + bm25_new_candidates (no DB vector filter on seeds)",
        "files": "retrieval/pgvector_retriever.py",
        "status": "done",
        "source": "commit 8ce44c7",
    },
]

conn = sqlite3.connect(DB)
cur = conn.cursor()

inserted = 0
skipped = 0
for e in entries:
    cur.execute("SELECT id FROM features WHERE name = ?", (e["name"],))
    row = cur.fetchone()
    if row:
        print(f"  SKIP (exists id={row[0]}): {e['name']}")
        skipped += 1
        continue
    cur.execute(
        """INSERT INTO features (name, category, description, definition, files,
             status, source, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            e["name"],
            e["category"],
            e["description"],
            e["definition"],
            e["files"],
            e["status"],
            e["source"],
            NOW,
            NOW,
        ),
    )
    print(f"  INSERT: {e['name']}")
    inserted += 1

conn.commit()
conn.close()
print(f"\nDone — inserted={inserted} skipped={skipped}")
