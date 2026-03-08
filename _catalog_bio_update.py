"""
Add BIO tagger v14→v15 extraction improvement features to feature_catalog_master.sqlite3.

Session work:
- Fixed find_phrase_fuzzy: punctuation strip, gap 2→4, long-phrase first-to-last fallback
- Fixed long-phrase char reconstruction via word-index counting
- Fixed UnicodeEncodeError on Windows (checkmark -> [OK])
- Result: 50/52 complete S+P+O rows (up from 46) in 10-chunk test batch
"""

import sqlite3
from datetime import datetime

DB = 'feature_catalog_master.sqlite3'


def add_feature(conn, name, category, description, definition, files, status, source=None):
    cur = conn.cursor()
    cur.execute("SELECT id FROM features WHERE name = ?", (name,))
    row = cur.fetchone()
    if row:
        cur.execute("""
            UPDATE features SET category=?, description=?, definition=?, files=?, status=?, source=?,
            updated_at=CURRENT_TIMESTAMP WHERE name=?
        """, (category, description, definition, files, status, source, name))
        print(f"  [UPDATE] {name}")
        return row[0]
    else:
        cur.execute("""
            INSERT INTO features (name, category, description, definition, files, status, source)
            VALUES (?,?,?,?,?,?,?)
        """, (name, category, description, definition, files, status, source))
        fid = cur.lastrowid
        print(f"  [INSERT id={fid}] {name}")
        return fid


def main():
    conn = sqlite3.connect(DB)

    # ── Feature 1: find_phrase_fuzzy punctuation strip ───────────────────────
    add_feature(
        conn,
        name="find_phrase_fuzzy: punctuation-only token strip",
        category="extraction",
        description=(
            "Strip punctuation-only tokens from phrase_words before fuzzy matching "
            "so objects like '- starting at 0 089' or 'First , a token set' are "
            "recoverable after leading punctuation is removed by normalize_for_matching."
        ),
        definition=(
            "After splitting phrase_norm, filter: "
            "phrase_words = [w for w in phrase_words if re.search(r'\\w', w)]"
        ),
        files="extraction/extract_bio_atomic_clean.py",
        status="done",
        source="session:v14->v15 BIO fix",
    )

    # ── Feature 2: find_phrase_fuzzy max_token_gap 2→4 ───────────────────────
    add_feature(
        conn,
        name="find_phrase_fuzzy: max_token_gap 2→4",
        category="extraction",
        description=(
            "Increased default max_token_gap from 2 to 4 to allow fuzzy matching "
            "to stitch together tokens with wider gaps, handling ccomp/xcomp objects "
            "whose subtree text contains punctuation tokens separated by pruned branches."
        ),
        definition="max_token_gap default changed from 2 to 4 in find_phrase_fuzzy signature",
        files="extraction/extract_bio_atomic_clean.py",
        status="done",
        source="session:v14->v15 BIO fix",
    )

    # ── Feature 3: long-phrase first-to-last span fallback ───────────────────
    add_feature(
        conn,
        name="find_phrase_fuzzy: first-to-last span for non-contiguous ccomp objects",
        category="extraction",
        description=(
            "For long phrases (>8 words) that cannot be found as contiguous substrings "
            "because branch pruning (skip_deps) creates gaps, fall back to spanning from "
            "first phrase word position to last phrase word position in the sentence. "
            "Fixes rows where ccomp/xcomp objects like "
            "'that , under broad conditions , black-box ... separated' appear non-contiguously."
        ),
        definition=(
            "When len(phrase_words) > 8 and exact substring match fails: "
            "find first_sent_idx (first occurrence of phrase_words[0]) and "
            "last_sent_idx (last occurrence of phrase_words[-1]), "
            "return char span from sent_words[first_sent_idx] to end of sent_words[last_sent_idx]."
        ),
        files="extraction/extract_bio_atomic_clean.py",
        status="done",
        source="session:v14->v15 BIO fix",
    )

    # ── Feature 4: long-phrase word-index char reconstruction ────────────────
    add_feature(
        conn,
        name="find_phrase_fuzzy: word-index counting for long-phrase char reconstruction",
        category="extraction",
        description=(
            "Replaced fragile forward character search for long-phrase start_char with "
            "word-index counting: count words in prefix of sentence_norm before the matched "
            "substring, map that count to sequential word scan in sentence_lower to find "
            "the exact character offset. Prevents spurious matches on short common words "
            "like 's', 'a', 'by' that appear multiple times."
        ),
        definition=(
            "prefix = sentence_norm[:pos]; "
            "prefix_word_count = len(prefix.split()) if prefix.strip() else 0; "
            "scan sentence_lower word-by-word counting to prefix_word_count to get start_char."
        ),
        files="extraction/extract_bio_atomic_clean.py",
        status="done",
        source="session:v14->v15 BIO fix",
    )

    # ── Feature 5: UnicodeEncodeError fix for Windows console ────────────────
    add_feature(
        conn,
        name="extraction: replace checkmark unicode with [OK] for Windows cp1252",
        category="extraction",
        description=(
            "Replaced U+2713 (✓) checkmark characters in print() calls with '[OK]' "
            "to prevent UnicodeEncodeError on Windows when stdout is redirected to a file "
            "or terminal with cp1252 encoding (which cannot encode U+2713)."
        ),
        definition="Lines 1021, 1037 in extract_bio_atomic_clean.py: print('[OK] ...') instead of print('✓ ...')",
        files="extraction/extract_bio_atomic_clean.py",
        status="done",
        source="session:v14->v15 BIO fix",
    )

    # ── Feature 6: v15 extraction result ─────────────────────────────────────
    add_feature(
        conn,
        name="BIO extraction v15: 50/52 complete S+P+O rows",
        category="training",
        description=(
            "Re-ran BIO extraction on 10-chunk test batch (seed=42) with all v15 fixes applied. "
            "Result: 50 of 52 examples have complete Subject+Predicate+Object BIO labels "
            "(up from 46 in v14). The 2 remaining incomplete rows (20, 29) are linguistically "
            "unresolvable: row 20 has expletive 'it' (expl deprel, not nsubj), row 29 is "
            "imperative with understood-you subject."
        ),
        definition=(
            "Command: python extraction/extract_bio_atomic_clean.py "
            "--chunks 10 --input checkpoints/chunks.msgpack "
            "--output data/bio_training_test10_v15.msgpack --seed 42"
        ),
        files="data/bio_training_test10_v15.msgpack, extraction/extract_bio_atomic_clean.py",
        status="done",
        source="session:v14->v15 BIO fix",
    )

    conn.commit()
    conn.close()
    print("\n[OK] Catalog updated.")


if __name__ == "__main__":
    main()
