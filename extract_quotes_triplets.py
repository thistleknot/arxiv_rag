"""
Apply quotes_bio_tagger.pt to all quotes and extract clean triplets.

Core Thesis:
    The arxiv-trained BIO tagger produced noisy triplets on quote text.
    We trained quotes_bio_tagger.pt (F1=0.85) specifically on quotes.
    This script re-runs inference with the domain-specific model, producing
    clean (subject, predicate, object) triplets for graph construction.

Workflow:
    Load(english_quotes) → 2,508 quotes
        → SentenceSplit → ~25k sentences
        → BIOTripletExtractor(quotes_bio_tagger.pt).extract_triplets_batch()
        → filter: discard if no triplets extracted
        → Save → checkpoints/quotes_triplets.msgpack

Output format (matches normalize_entities.py expectation):
    list of {
        'doc_id':  str,                        # "quote_{i}"
        'text':    str,                        # full quote text
        'author':  str,
        'tags':    list[str],
        'triplets': [
            {'subject': str, 'predicate': str, 'object': str|None, ...},
            ...
        ]
    }

Usage:
    python extract_quotes_triplets.py
    python extract_quotes_triplets.py --threshold 0.45 --batch-size 32
    python extract_quotes_triplets.py --output checkpoints/quotes_triplets.msgpack
"""

import sys
import re
import argparse
import msgpack
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn

# ── sys.path ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent

# ── Label schema (mirrors train_bert_bio_optuna.py exactly) ──────────────────
LABEL_TO_ID = {'O': 0, 'B-SUBJ': 1, 'I-SUBJ': 2, 'B-PRED': 3, 'I-PRED': 4, 'B-OBJ': 5, 'I-OBJ': 6}
ID_TO_LABEL  = {v: k for k, v in LABEL_TO_ID.items()}

SPAN_STOPWORDS = frozenset({
    'a', 'an', 'the', 'to', 'of', 'in', 'on', 'at', 'by', 'for', 'with', 'from',
    'and', 'or', 'but', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'has', 'have', 'had', 'do', 'does', 'did',
    'that', 'which', 'who', 'this', 'these', 'those', 'it', 'its',
    # pointer / approximator words — no entity identity as boundary tokens
    'like', 'unlike', 'about', 'under', 'over', 'upon', 'above', 'below',
    'beyond', 'among', 'across', 'toward', 'towards', 'without', 'within',
    'throughout', 'per', 'around', 'along',
    # subordinating conjunctions and auxiliary verbs
    'because', 'although', 'though', 'while', 'whereas', 'than',
    'will', 'would', 'could', 'should',
    'have', 'has', 'had',
    'do', 'does', 'did',
    # temporal/conditional subordinating conjunctions
    'before', 'after',
    # relative adverbs — signal subordinate clause membership
    'wherever', 'whenever', 'however', 'whatever', 'whichever',
    # epistemic and manner adverbs with no referential entity identity
    'perhaps', 'maybe', 'probably', 'somehow', 'otherwise',
    # politeness/imperative marker
    'please',
    # comparative/superlative adjectives not valid as standalone entity concepts
    'better', 'worse', 'greater', 'lesser', 'higher', 'lower',
    'larger', 'smaller', 'longer', 'shorter', 'stronger', 'weaker',
    # missing modals
    'can', 'cannot',
    # direct implication and filler words
    'must',                    # modal of necessity
    'doing', 'going',          # gerund continuations
    'get', 'got',              # 'to get' phrase components
    # temporal and frequency adverbs
    'never', 'ever', 'always', 'sometimes', 'often', 'seldom', 'rarely',
    'once', 'twice', 'again', 'soon', 'ahead', 'almost', 'nearly', 'quite', 'rather',
    # attribute adjectives with no standalone entity identity
    'next',                    # "next person" → strips to "person"
    # phrasal-verb directional particles — tag complement, not entity
    'together', 'apart',       # "come together", "fall apart" → bare verb only
})
PRONOUN_HEADS = frozenset({
    'i', 'me', 'my', 'we', 'our', 'us', 'you', 'your',
    'he', 'him', 'his', 'she', 'her', 'it', 'its', 'they', 'them', 'their',
    # reflexive pronouns
    'myself', 'yourself', 'himself', 'herself', 'itself',
    'ourselves', 'yourselves', 'themselves',
})

PREDICATE_STOPWORDS: frozenset = frozenset({
    # bare copula — no relational semantic content
    'be', 'is', 'are', 'was', 'were', 'been', 'being',
    # bare auxiliaries
    'have', 'has', 'had',
    'do', 'does', 'did',
    # bare modals
    'will', 'would', 'could', 'should', 'shall', 'might', 'may', 'must', 'can', 'cannot',
    # BERT contraction stems and their reassembled forms
    "aren", "aren't", "isn", "isn't", "wasn", "wasn't", "weren", "weren't",
    "haven", "haven't", "hasn", "hasn't", "hadn", "hadn't",
    "don", "don't", "doesn", "doesn't", "didn", "didn't",
    "won", "won't", "couldn", "couldn't", "can't", "shouldn", "shouldn't",
    "wouldn", "wouldn't", "mustn", "mustn't",
})


def _is_content_predicate(text: str) -> bool:
    """Reject bare auxiliary/modal predicates with no relational semantic content.

    Single-word auxiliaries and modals (is, are, must, can, …) and their
    BERT contraction stems (aren, isn, …) are excluded.  Multi-word phrases
    containing at least one non-auxiliary word (make use of, should feel)
    are retained — the main verb carries the relation.
    """
    if not text:
        return False
    words = text.split()
    return not all(w in PREDICATE_STOPWORDS for w in words)


# ── Model (mirrors BIOTaggerMultiClass from train_bert_bio_optuna.py) ─────────

class _BIOTaggerMultiClass(nn.Module):
    """Inline replica of BIOTaggerMultiClass from train_bert_bio_optuna.py.

    Defined here to avoid importing the training script (which calls
    parse_args() at import time and would clobber our own argparse state).
    Architecture must stay bit-for-bit identical to the training definition:
        BERT-base-uncased → Dropout → Linear(768, 7)
    """
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        from transformers import BertModel
        self.bert       = BertModel.from_pretrained('bert-base-uncased')
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 7)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(self.dropout(out.last_hidden_state))  # (B, T, 7)


# ── Span + triplet helpers  ───────────────────────────────────────────────────

def _join_wordpieces(tokens: List[str]) -> str:
    """Merge BERT ## continuation tokens into surface form.

    If the span boundary falls on a ## token (B- fired on a subword-
    continuation), there is no prior word to attach to — strip the ##
    prefix and treat it as a standalone partial token.  This avoids
    '##we', '##€' etc. leaking into entity keys as verbatim strings.

    Also reassembles contractions that BERT splits into separate tokens:
        aren ' t  →  aren't
        they ' re →  they're
    """
    words: List[str] = []
    for tok in tokens:
        if tok.startswith('##') and words:
            words[-1] += tok[2:]
        elif tok.startswith('##'):
            words.append(tok[2:])   # orphan ##-token: strip prefix, keep fragment
        else:
            words.append(tok)
    text = ' '.join(words)
    # Merge negation contractions: word ' t  →  word n't  (aren't, don't, won't…)
    text = re.sub(r"(\w+)\s+'\s+t\b", r"\1n't", text)
    # Merge other contractions: word ' re/ve/ll/d/m  →  word're / word've …
    text = re.sub(r"(\w+)\s+'\s+(re|ve|ll|d|m)\b", r"\1'\2", text)
    return text


_LY_NON_ADVERB_EXT: frozenset = frozenset({
    'family', 'holy', 'only', 'early', 'daily', 'rally', 'belly',
    'ally', 'bully', 'jelly', 'poly', 'folly', 'tally', 'valley',
    'melancholy', 'anomaly',
})


def _is_adverb_token_ext(tok: str) -> bool:
    """True if tok is a manner/degree adverb (-ly suffix) with no entity identity."""
    return len(tok) > 3 and tok.endswith('ly') and tok not in _LY_NON_ADVERB_EXT


def _clean_span(text: str) -> str:
    """Strip BERT possessive artifacts and leading/trailing functional tokens from a span."""
    import unicodedata
    # Normalize BERT-tokenized possessives: "person ' s body" -> "body" (keep possessum)
    m = re.search(r"^.*\s+'\s+s\s+(.+)$", text)
    if m:
        text = m.group(1).strip()
    else:
        text = re.sub(r"^'\s+\S+\s*", "", text).strip()  # "' re capable" -> "capable"
        text = re.sub(r"^'\s*", "", text).strip()          # remaining leading '
    text = ' '.join(text.split())
    words = text.split()
    _is_punct = lambda w: all(unicodedata.category(c).startswith('P') for c in w)
    # Strip leading/trailing stopwords FIRST so a leading article or possessive
    # ("your life in fear", "the life in fear") can't shield the hollow head.
    while words and (words[0].lower()  in SPAN_STOPWORDS or _is_punct(words[0]) or _is_adverb_token_ext(words[0].lower())): words = words[1:]
    while words and (words[-1].lower() in SPAN_STOPWORDS or _is_punct(words[-1]) or _is_adverb_token_ext(words[-1].lower())): words = words[:-1]
    # Strip hollow-head prepositional phrases AFTER stopwords are removed:
    # "life in fear" -> "fear", "way in X" -> "X".  These arise when the BIO
    # tagger includes a semantically vacuous head noun that mirrors the predicate
    # ("live [your] life in fear" -> predicate=live, object="[your] life in fear").
    text = re.sub(
        r'^(?:life|way|time|role|part|place|state|form|manner)\s+in\s+',
        '', ' '.join(words)
    ).strip()
    return text


def _is_valid_entity(text: str) -> bool:
    words = text.split()
    if not words:
        return False
    # ANY pronoun anywhere in span → clause fragment, not a noun phrase
    if any(w.lower() in PRONOUN_HEADS for w in words):
        return False
    if "'" in words:
        return False
    # Clause-fragment guard: spans with ≥50% function-word tokens are verb/prep phrases
    if len(words) > 1 and sum(1 for w in words if w.lower() in SPAN_STOPWORDS) * 2 >= len(words):
        return False
    return any(w.lower() not in SPAN_STOPWORDS for w in words)


def _atomize_span(text: str) -> List[str]:
    """Split a cleaned span on coordinators into atomic sub-spans.

    Spans with 'and' / 'or' / ',' signal multiple co-referenced entities
    that belong as separate KG nodes.  Each sub-span is re-cleaned;
    empty or pronoun-headed results are dropped.
        'hatred and war' -> ['hatred', 'war']
        'love and kindness' -> ['love', 'kindness']
        'love' -> ['love']
    """
    import re
    parts = re.split(r'\s+and\s+|\s+or\s+|,\s*|\s+is\s+|\s+are\s+|\s+was\s+|\s+were\s+|\s+of\s+', text, flags=re.IGNORECASE)
    result = []
    for part in parts:
        cleaned = _clean_span(part.strip())
        if cleaned and _is_valid_entity(cleaned):
            result.append(cleaned)
    return result or [text]


def _extract_spans(
    tokens:   List[str],
    label_ids: List[int],
    probs:    np.ndarray,
    threshold: float,
) -> Dict[str, List[Tuple[int, int, str]]]:
    """Extract (start, end, text) spans per type from argmax label IDs."""
    SPECIAL = {'[CLS]', '[SEP]', '[PAD]'}
    spans: Dict[str, List] = {'SUBJ': [], 'PRED': [], 'OBJ': []}
    i = 0
    while i < len(tokens):
        if tokens[i] in SPECIAL:
            i += 1
            continue
        lid   = label_ids[i]
        lname = ID_TO_LABEL.get(lid, 'O')
        if lname.startswith('B-') and probs[i, lid] >= threshold:
            typ   = lname[2:]       # SUBJ / PRED / OBJ
            toks  = [tokens[i]]
            start = i
            i    += 1
            while i < len(tokens) and tokens[i] not in SPECIAL:
                nlid  = label_ids[i]
                nlname = ID_TO_LABEL.get(nlid, 'O')
                if nlname == f'I-{typ}' and probs[i, nlid] >= threshold:
                    toks.append(tokens[i])
                    i += 1
                else:
                    break
            cleaned = _clean_span(_join_wordpieces(toks))
            if cleaned:
                spans[typ].append((start, i, cleaned))
        else:
            i += 1
    return spans


# Verb → nominal echo forms.  When the BIO tagger produces a triplet like
# (don, live, life in fear) the object head "life" mirrors the predicate "live"
# — a morphological artifact, not a distinct concept.  Strip the echo noun
# (and any following bare preposition) so "life in fear" → "fear".
# Standard lemmatisers treat "live" and "life" as different lemmas, so this
# mapping is intentional rather than redundant.
_PRED_ECHO: Dict[str, set] = {
    'live':    {'life', 'living'},
    'die':     {'death', 'dying'},
    'love':    {'love', 'loving'},
    'hate':    {'hate', 'hatred'},
    'fight':   {'fight', 'fighting'},
    'work':    {'work', 'working'},
    'run':     {'running'},
    'play':    {'play', 'playing'},
    'suffer':  {'suffering'},
    'speak':   {'speech', 'speaking'},
    'grow':    {'growth', 'growing'},
    'think':   {'thought', 'thinking'},
    'feel':    {'feeling', 'feelings'},
    'move':    {'movement', 'motion', 'moving'},
    'change':  {'change', 'changing'},
    'act':     {'action', 'acting'},
    'build':   {'building'},
    'create':  {'creation'},
    'make':    {'making'},
}
_ECHO_PREPS = frozenset({'in', 'of', 'on', 'at', 'for', 'with', 'by', 'to',
                         'into', 'through', 'about', 'against'})


def _strip_pred_echo(pred: str, obj: str) -> str:
    """Remove leading nominal echo of *pred* from *obj*.

    'live' + 'life in fear'  → 'fear'
    'live' + 'life of crime' → 'crime'
    'think' + 'thoughts on X' → 'X'
    Returns the original obj string if no echo is detected or stripping
    would produce an empty result.
    """
    if not pred or not obj:
        return obj
    echo_nouns = _PRED_ECHO.get(pred.strip().lower(), set())
    if not echo_nouns:
        return obj
    words = obj.strip().split()
    if not words or words[0].lower() not in echo_nouns:
        return obj
    rest = words[1:]
    if rest and rest[0].lower() in _ECHO_PREPS:
        rest = rest[1:]
    return ' '.join(rest) if rest else obj  # never collapse to empty


def _reconstruct_triplets(
    spans: Dict[str, List[Tuple[int, int, str]]]
) -> List[Dict]:
    """Build (subject, predicate, object) using position heuristics.

    Subjects and objects are atomized — spans containing coordinators
    ('and', 'or', ',') are split into individual entities before assembly.
    The cartesian product (subject_atoms x object_atoms) is emitted per
    predicate, so 'hatred and war' as an object yields two separate triplets.
    """
    triplets = []
    for ps, pe, pt in spans['PRED']:
        if not _is_content_predicate(pt):   # skip bare auxiliary/modal predicates
            continue
        before_subj = [(s, e, t) for s, e, t in spans['SUBJ'] if e <= ps]
        if not before_subj:
            continue
        _, _, subj_text = max(before_subj, key=lambda x: x[1])
        if not _is_valid_entity(subj_text):
            continue
        subj_atoms = _atomize_span(subj_text)
        after_obj = [(s, e, t) for s, e, t in spans['OBJ'] if s >= pe]
        obj_atoms: List[Optional[str]] = [None]
        if after_obj:
            _, _, cand = min(after_obj, key=lambda x: x[0])
            if _is_valid_entity(cand):
                obj_atoms = _atomize_span(cand)  # type: ignore[assignment]
        for s_atom in subj_atoms:
            for o_atom in obj_atoms:
                # Remove nominal echo of the predicate verb from the object:
                # "live" + "life in fear" → "fear"  (see _PRED_ECHO / _strip_pred_echo)
                if o_atom is not None:
                    o_atom = _strip_pred_echo(pt, o_atom)
                triplets.append({
                    'subject':       s_atom,
                    'predicate':     pt,
                    'object':        o_atom,
                    'arity':         'unary' if o_atom is None else 'binary',
                    'relation_type': 'predicate',
                })
    return triplets


def _run_batch(
    model:     _BIOTaggerMultiClass,
    tokenizer,
    sentences: List[str],
    device:    torch.device,
    threshold: float,
) -> List[List[Dict]]:
    """Tokenize + infer + reconstruct a batch of sentences."""
    enc = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors='pt',
    )
    input_ids      = enc['input_ids'].to(device)
    attention_mask = enc['attention_mask'].to(device)
    with torch.no_grad():
        logits = model(input_ids, attention_mask)           # (B, T, 7)
        probs  = torch.softmax(logits, dim=-1).cpu().numpy()  # (B, T, 7)
        preds  = logits.argmax(dim=-1).cpu().numpy()          # (B, T)
    results = []
    for b in range(input_ids.size(0)):
        seq_len = int(attention_mask[b].sum().item())
        tokens  = tokenizer.convert_ids_to_tokens(input_ids[b].cpu())[:seq_len]
        spans   = _extract_spans(tokens, preds[b, :seq_len], probs[b, :seq_len], threshold)
        results.append(_reconstruct_triplets(spans))
    return results

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='Extract triplets from quotes using domain-specific BIO tagger')
parser.add_argument('--model', type=str, default='quotes_bio_tagger.pt',
                    help='Path to trained BIO tagger (default: quotes_bio_tagger.pt)')
parser.add_argument('--output', type=str, default='checkpoints/quotes_triplets.msgpack',
                    help='Output msgpack path')
parser.add_argument('--n', type=int, default=2508,
                    help='Number of quotes to process (default: all)')
parser.add_argument('--threshold', type=float, default=0.5,
                    help='BIO extraction threshold (default: 0.5)')
parser.add_argument('--batch-size', type=int, default=16,
                    help='Sentences per GPU batch (default: 16)')
parser.add_argument('--min-length', type=int, default=10,
                    help='Minimum sentence character length (default: 10)')
args = parser.parse_args()


# ── Load quotes ───────────────────────────────────────────────────────────────

def _fix_encoding(text: str) -> str:
    """Fix mojibake from smart-quote UTF-8 bytes misread as Latin-1."""
    try:
        import ftfy
        return ftfy.fix_text(text)
    except ImportError:
        pass
    return (text
            .replace('\u2019', "'").replace('\u2018', "'")
            .replace('\u201c', '"').replace('\u201d', '"')
            .replace('\u2013', '-').replace('\u2014', '-'))


def load_quotes(n: int) -> list:
    """Load english_quotes dataset from HuggingFace."""
    try:
        from datasets import load_dataset
        print(f"Loading english_quotes from HuggingFace (n={n})...")
        ds = load_dataset("Abirate/english_quotes", split="train")

        quotes = []
        for row in ds:
            q = _fix_encoding((row.get("quote") or "").strip())
            if q and len(q) >= args.min_length:
                quotes.append({
                    "quote":  q,
                    "author": row.get("author", "Unknown"),
                    "tags":   row.get("tags", []) or [],
                })

        print(f"  Loaded {len(quotes)} valid quotes")
        return quotes[:n]

    except Exception as e:
        print(f"  ERROR loading from HuggingFace: {e}")
        sys.exit(1)


def split_into_sentences(text: str, min_len: int) -> list[str]:
    """Split a quote into sentences at sentence-ending punctuation."""
    sents = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in sents if len(s.strip()) >= min_len]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    model_path = ROOT / args.model
    if not model_path.exists():
        print(f"ERROR: model not found at {model_path}")
        print("  Run: python training/train_bert_bio_optuna.py --output quotes_bio_tagger.pt ...")
        sys.exit(1)

    from transformers import BertTokenizerFast
    print(f"Loading BIOTaggerMultiClass: {model_path}")
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    model     = _BIOTaggerMultiClass()
    model.load_state_dict(torch.load(str(model_path), map_location=device))
    model.to(device).eval()
    print(f"  Loaded: {sum(p.numel() for p in model.parameters()):,} params  device={device}")

    # Load quotes
    quotes = load_quotes(args.n)

    # Flatten quotes → sentences, track provenance
    records = []
    sentence_meta = []   # (quote_index, sentence_text)

    for i, q in enumerate(quotes):
        # Split CamelCase runs from section-header concatenation, e.g.
        # "PersonallyNothing" (from "...Personally\nNothing...") → "Personally Nothing"
        qt = re.sub(r'([a-z])([A-Z])', r'\1 \2', q['quote'])
        sents = split_into_sentences(qt, args.min_length)
        if not sents:
            # Treat the whole quote as one sentence if split yields nothing
            sents = [qt]
        for s in sents:
            sentence_meta.append((i, s))

    print(f"  {len(quotes)} quotes -> {len(sentence_meta)} sentences")

    # Batch inference
    batch_size = args.batch_size
    all_sentence_triplets: list[list[dict]] = []

    print(f"Extracting triplets (threshold={args.threshold}, batch={batch_size})...")
    for start in tqdm(range(0, len(sentence_meta), batch_size),
                      desc="  BIO inference", unit="batch"):
        batch_sents = [s for _, s in sentence_meta[start:start + batch_size]]
        try:
            batch_results = _run_batch(model, tokenizer, batch_sents, device, args.threshold)
        except Exception as e:
            print(f"\n  [WARN] Batch failed ({e}), falling back to single inference")
            batch_results = []
            for sent in batch_sents:
                try:
                    batch_results.append(_run_batch(model, tokenizer, [sent], device, args.threshold)[0])
                except Exception as e2:
                    print(f"  [WARN] Single inference failed: {e2}")
                    batch_results.append([])
        all_sentence_triplets.extend(batch_results)

    # Aggregate per-quote: collapse sentence triplets back to quote level
    quote_triplets: list[list[dict]] = [[] for _ in quotes]
    for (quote_idx, _sent_text), triplets in zip(sentence_meta, all_sentence_triplets):
        quote_triplets[quote_idx].extend(triplets)

    # Build output records
    output_records = []
    skipped = 0
    for i, q in enumerate(quotes):
        triplets = quote_triplets[i]
        if not triplets:
            skipped += 1
            continue

        # Deduplicate across sentences within the same quote
        seen = set()
        deduped = []
        for t in triplets:
            key = (t.get('subject', ''), t.get('predicate', ''), t.get('object'))
            if key not in seen:
                seen.add(key)
                deduped.append(t)

        output_records.append({
            'doc_id':   f"quote_{i}",
            'text':     q['quote'],
            'author':   q['author'],
            'tags':     q['tags'],
            'triplets': deduped,
        })

    print(f"\nResults:")
    print(f"  Quotes with triplets: {len(output_records)} / {len(quotes)}")
    print(f"  Quotes with no triplets: {skipped}")
    total_triplets = sum(len(r['triplets']) for r in output_records)
    print(f"  Total triplets: {total_triplets}")
    print(f"  Avg triplets/quote: {total_triplets / max(1, len(output_records)):.1f}")

    # Save
    print(f"\nSaving to {output_path}...")
    with open(output_path, 'wb') as f:
        msgpack.pack(output_records, f, use_bin_type=True)

    print(f"Done. {len(output_records)} quote records saved.")
    print(f"\nNext steps:")
    print(f"  python graph/normalize_entities.py --msgpack {args.output}")
    print(f"  python graph/build_kg_dataset.py")
    print(f"  python graph/train_gat.py")


if __name__ == '__main__':
    main()
