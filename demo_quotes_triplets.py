"""
Quick end-to-end demo: english_quotes → BIO tagger → cleaned triplets.

Shows S-P-O extractions AFTER cleanup so user can spot-check quality.
Keeps it in English (no arxiv jargon) so correctness is obvious.

Usage:
    python demo_quotes_triplets.py [--n 50] [--threshold 0.4585]
"""

import sys
import re
import json
import argparse
import msgpack
from pathlib import Path
from collections import defaultdict

# ── sys.path ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / 'training'))

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--n',         type=int,   default=60,      help='# quotes to process')
parser.add_argument('--threshold', type=float, default=0.4585,  help='BIO threshold')
parser.add_argument('--source',    choices=['hf', 'builtin'], default='hf',
                    help='hf = HuggingFace english_quotes, builtin = hardcoded fallback')
parser.add_argument('--output',    type=str, default=None,
                    help='Save msgpack to this path (e.g. checkpoints/quotes_triplets.msgpack)')
args = parser.parse_args()

# ── Load quotes ───────────────────────────────────────────────────────────────

BUILTIN_QUOTES = [
    {"quote": "The only way to do great work is to love what you do.", "author": "Steve Jobs", "tags": ["love", "work"]},
    {"quote": "In three words I can sum up everything I've learned about life: it goes on.", "author": "Robert Frost", "tags": ["life"]},
    {"quote": "If you want to live a happy life, tie it to a goal, not to people or things.", "author": "Albert Einstein", "tags": ["happiness", "goals"]},
    {"quote": "Never let the fear of striking out keep you from playing the game.", "author": "Babe Ruth", "tags": ["fear", "courage"]},
    {"quote": "Money and success don't change people; they merely amplify what is already there.", "author": "Will Smith", "tags": ["success"]},
    {"quote": "Your time is limited, so don't waste it living someone else's life.", "author": "Steve Jobs", "tags": ["time", "life"]},
    {"quote": "Not all those who wander are lost.", "author": "J.R.R. Tolkien", "tags": ["wandering"]},
    {"quote": "It does not matter how slowly you go as long as you do not stop.", "author": "Confucius", "tags": ["perseverance"]},
    {"quote": "We accept the love we think we deserve.", "author": "Stephen Chbosky", "tags": ["love"]},
    {"quote": "Life is what happens to us while we are making other plans.", "author": "Allen Saunders", "tags": ["life"]},
    {"quote": "The greatest glory in living lies not in never falling, but in rising every time we fall.", "author": "Nelson Mandela", "tags": ["life", "perseverance"]},
    {"quote": "The way to get started is to quit talking and begin doing.", "author": "Walt Disney", "tags": ["action"]},
    {"quote": "If life were predictable it would cease to be life, and be without flavor.", "author": "Eleanor Roosevelt", "tags": ["life"]},
    {"quote": "If you look at what you have in life, you'll always have more. If you look at what you don't have in life, you'll never have enough.", "author": "Oprah Winfrey", "tags": ["gratitude"]},
    {"quote": "If you set your goals ridiculously high and it's a failure, you will fail above everyone else's success.", "author": "James Cameron", "tags": ["goals"]},
    {"quote": "Life is not measured by the number of breaths we take, but by the moments that take our breath away.", "author": "Maya Angelou", "tags": ["life"]},
    {"quote": "We must learn to live together as brothers or perish together as fools.", "author": "Martin Luther King Jr.", "tags": ["humanity"]},
    {"quote": "The secret of getting ahead is getting started.", "author": "Mark Twain", "tags": ["motivation"]},
    {"quote": "It always seems impossible until it's done.", "author": "Nelson Mandela", "tags": ["perseverance"]},
    {"quote": "Do not go where the path may lead, go instead where there is no path and leave a trail.", "author": "Ralph Waldo Emerson", "tags": ["leadership"]},
    {"quote": "You will face many defeats in life, but never let yourself be defeated.", "author": "Maya Angelou", "tags": ["perseverance"]},
    {"quote": "The greatest pleasure in life is doing what people say you cannot do.", "author": "Walter Bagehot", "tags": ["pleasure", "achievement"]},
    {"quote": "In the end, it's not the years in your life that count. It's the life in your years.", "author": "Abraham Lincoln", "tags": ["life"]},
    {"quote": "Never bend your head. Always hold it high. Look the world straight in the eye.", "author": "Helen Keller", "tags": ["courage"]},
    {"quote": "When you reach the end of your rope, tie a knot in it and hang on.", "author": "Franklin D. Roosevelt", "tags": ["perseverance"]},
    {"quote": "Always remember that you are absolutely unique. Just like everyone else.", "author": "Margaret Mead", "tags": ["humor"]},
    {"quote": "Don't judge each day by the harvest you reap but by the seeds that you plant.", "author": "Robert Louis Stevenson", "tags": ["wisdom"]},
    {"quote": "The future belongs to those who believe in the beauty of their dreams.", "author": "Eleanor Roosevelt", "tags": ["dreams"]},
    {"quote": "Tell me and I forget. Teach me and I remember. Involve me and I learn.", "author": "Benjamin Franklin", "tags": ["learning"]},
    {"quote": "When one door of happiness closes, another opens; but often we look so long at the closed door that we do not see the one which has been opened for us.", "author": "Helen Keller", "tags": ["happiness"]},
]


def load_quotes_hf(n: int):
    try:
        from datasets import load_dataset
        print("Loading english_quotes from HuggingFace...")
        ds = load_dataset("Abirate/english_quotes", split="train")
        quotes = []
        for row in ds:
            q = (row.get("quote") or "").strip()
            if q:
                quotes.append({
                    "quote":  q,
                    "author": row.get("author", "Unknown"),
                    "tags":   row.get("tags", []) or [],
                })
        print(f"  Loaded {len(quotes)} quotes from HuggingFace")
        return quotes[:n]
    except Exception as e:
        print(f"  HuggingFace load failed ({e}), using builtin quotes")
        return BUILTIN_QUOTES[:n]


def load_quotes_builtin(n: int):
    return BUILTIN_QUOTES[:n]


# ── Cleanup helpers ───────────────────────────────────────────────────────────

# Personal pronouns as subjects → no real entity, drop the whole triplet
_SUBJ_PRONOUNS = {
    'i', 'you', 'we', 'they', 'he', 'she', 'one', 'it',
    'our', 'their', 'your', 'its', 'my', 'his', 'her',
}

# Pronouns as objects → anaphoric/empty, null the object (reduce to unary)
_OBJ_PRONOUNS = {'he', 'she', 'him', 'her', 'them', 'it', 'its', 'you'}

# Predicates with no relational content → drop the whole triplet
_DEAD_PREDICATES = {
    'would', 'would be', 'would not', 'would not be',
    'not', 'no', 'must', 'cannot', 'can not', 'will not', 'is not', 'are not',
}

# BERT splits contractions into fragments: "they're" → "they" "'" "re"
# The apostrophe gets dropped by BIO tagging, leaving bare "re", "s", "ve" etc.
# These are meaningless as standalone predicates or subjects.
_CONTRACTION_FRAGMENTS = {'s', 're', 've', 'll', 't', 'd', 'm', 'nt', 'n', 'er'}


def clean_triplet(t: dict) -> dict | None:
    """
    Post-extraction cleanup:
      - Guard None values
      - Strip surrounding punctuation / whitespace
      - Remove ##-wordpiece artifacts
      - Drop triplets with pronoun subjects or dead predicates
      - Null out pronoun objects or subordinate-clause objects (reduce to unary)
      - Require min token length after cleaning
    Returns None to signal 'discard this triplet'.
    """
    subj = (t.get("subject")   or "").strip()
    pred = (t.get("predicate") or "").strip()
    obj  = (t.get("object")    or "").strip()

    # If any field STARTS with ## the span boundary landed mid-word.
    # We cannot reconstruct the full word, so discard the triplet entirely.
    if subj.startswith("##") or pred.startswith("##"):
        return None
    if obj.startswith("##"):
        obj = ''  # null the object; subject+predicate may still be valid

    # Fix BERT wordpiece artifacts  (e.g. "g ##lor ##ify" → "glorify")
    # This only fires for ## that appear mid-field (start-## already handled above)
    subj = re.sub(r'\s*##\s*', '', subj)
    pred = re.sub(r'\s*##\s*', '', pred)
    obj  = re.sub(r'\s*##\s*', '', obj)

    # Strip leading/trailing punctuation left by stopword removal
    for ch in ',-;:.\'\"()[]':
        subj = subj.strip(ch)
        pred = pred.strip(ch)
        obj  = obj.strip(ch)

    subj = subj.strip()
    pred = pred.strip()
    obj  = obj.strip()

    # Drop if subject or predicate is a bare contraction fragment
    # ("they're" → "re", "here's" → "s", "older" ##er → "er", etc.)
    if subj.lower() in _CONTRACTION_FRAGMENTS or pred.lower() in _CONTRACTION_FRAGMENTS:
        return None

    # Drop if subject is a bare pronoun (no real entity)
    if subj.lower() in _SUBJ_PRONOUNS:
        return None

    # Drop if predicate is vacuous (no relational content)
    if pred.lower() in _DEAD_PREDICATES:
        return None

    # Null out object if it is a bare pronoun or a subordinate clause
    # ("like there's nobody watching", "like you'll never be hurt", etc.)
    first_obj_word = obj.lower().split()[0] if obj else ''
    if first_obj_word in _OBJ_PRONOUNS or obj.lower().startswith('like '):
        obj = ''

    # Discard if any required field is empty or trivially short
    if len(subj) < 2 or len(pred) < 2:
        return None

    # Mark unary when object was removed or never present
    relation_type = "unary" if not obj else "binary"

    return {
        "subject":       subj,
        "predicate":     pred,
        "object":        obj if obj else None,
        "relation_type": relation_type,
        "arity":         1 if not obj else 2,
    }


def dedup_triplets(triplets: list[dict]) -> list[dict]:
    """Remove exact-duplicate (subj, pred, obj) triples."""
    seen = set()
    out = []
    for t in triplets:
        key = (t["subject"], t["predicate"], t.get("object") or "")
        if key not in seen:
            seen.add(key)
            out.append(t)
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Load quotes
    if args.source == 'hf':
        quotes = load_quotes_hf(args.n)
    else:
        quotes = load_quotes_builtin(args.n)

    print(f"\nProcessing {len(quotes)} quotes with BIO tagger (threshold={args.threshold})...\n")

    # Load BIO tagger
    MODEL_PATH = str(ROOT / 'bio_tagger_atomic.pt')
    from inference_bio_tagger import BIOTripletExtractor
    extractor = BIOTripletExtractor(MODEL_PATH)

    # Extract triplets
    results = []
    total_raw = 0
    total_clean = 0
    unary_count = 0

    for row in quotes:
        quote  = row["quote"]
        author = row["author"]
        tags   = row.get("tags", [])

        # Strip non-ASCII characters (curly quotes etc. cause â€œ prefix artifacts)
        clean_quote = re.sub(r'[^\x00-\x7F]+', ' ', quote).strip()
        raw_triplets = extractor.extract_triplets(clean_quote, threshold=args.threshold)
        total_raw += len(raw_triplets)

        cleaned = []
        for t in raw_triplets:
            c = clean_triplet(t)
            if c is not None:
                cleaned.append(c)
                if c["arity"] == 1:
                    unary_count += 1
        cleaned = dedup_triplets(cleaned)
        total_clean += len(cleaned)

        if cleaned:
            results.append({
                "quote":    quote,
                "author":   author,
                "tags":     tags,
                "triplets": cleaned,
            })

    # ── Print spot-check output ───────────────────────────────────────────────
    print("=" * 72)
    print(f"EXTRACTION RESULTS  ({len(quotes)} quotes, threshold={args.threshold})")
    print(f"  Raw triplets:    {total_raw}")
    print(f"  After cleanup:   {total_clean}")
    print(f"  Unary P(S):      {unary_count}  (predicate with subject only)")
    print(f"  Drop rate:       {100*(total_raw-total_clean)/max(1,total_raw):.1f}%")
    print(f"  Quotes with >=1 triplet: {len(results)}/{len(quotes)}")
    print("=" * 72)
    print()

    # Show every extracted record
    for i, r in enumerate(results):
        q_safe = r['quote'].encode('ascii', errors='replace').decode('ascii')
        a_safe = r['author'].encode('ascii', errors='replace').decode('ascii')
        print(f"[{i+1:02d}] \"{q_safe[:80]}{'...' if len(r['quote'])>80 else ''}\"")
        print(f"     -- {a_safe}   tags={r['tags']}")
        for t in r["triplets"]:
            s = t["subject"]
            p = t["predicate"]
            o = t.get("object") or ""
            arity = t["arity"]
            rtype = t["relation_type"]
            if arity == 1:
                print(f"       P({s})  unary: predicate = '{p}'")
            else:
                print(f"       {s}  |  {p}  |  {o}")
        print()

    # ── Summary stats ─────────────────────────────────────────────────────────
    print("-" * 72)
    # Predicate frequency
    pred_counts = defaultdict(int)
    for r in results:
        for t in r["triplets"]:
            pred_counts[t["predicate"]] += 1
    top_preds = sorted(pred_counts.items(), key=lambda x: -x[1])[:15]
    print("Top predicates:")
    for p, c in top_preds:
        print(f"  {c:4d}x  {p}")
    print()

    # Arity distribution
    arity1 = sum(1 for r in results for t in r["triplets"] if t["arity"] == 1)
    arity2 = sum(1 for r in results for t in r["triplets"] if t["arity"] == 2)
    print(f"Arity:  unary={arity1}  binary={arity2}")
    print()

    # Quote: unary predicate discussion
    print("-" * 72)
    print("NOTE on unary predicates:")
    print("  Your quote 'we should be removed' illustrates a case where the")
    print("  cleanup pipeline drops the object ('removed' is tagged as predicate,")
    print("  but 'we' is subject and there is no object span).")

    # ── Optional msgpack save (for graph pipeline) ────────────────────────────
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Serialize in normalize_entities-compatible format:
        #   [{doc_id, author, tags, triplets: [{subject, predicate, object}]}]
        records = []
        for idx, r in enumerate(results):
            # Flatten triplet dicts to plain {subject, predicate, object}
            trips = [
                {"subject": t["subject"],
                 "predicate": t["predicate"],
                 "object": t.get("object") or ""}
                for t in r["triplets"]
            ]
            records.append({
                "doc_id":  f"{r['author']}_{idx}",
                "author":  r["author"],
                "tags":    r["tags"],
                "quote":   r["quote"],
                "triplets": trips,
            })
        with open(out_path, "wb") as f:
            f.write(msgpack.packb(records, use_bin_type=True))
        print(f"\nSaved {len(records)} records ({total_clean} triplets) -> {out_path}")
    print("  In FOL: Should_Be_Removed(we)  P(S) unary predicate.")
    print("  The BIO tagger emits this as a binary with an empty object,")
    print("  which the cleanup marks arity=1 / relation_type='unary'.")
    print("  Whether to keep or discard unary extractions is a design choice.")


if __name__ == "__main__":
    main()
