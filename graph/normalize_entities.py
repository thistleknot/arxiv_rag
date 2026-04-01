"""
Entity and predicate normalization for KG construction.

Core Thesis:
    BIO triplet spans are surface-form noisy — "transformer", "the transformer",
    "transformer model" are the same concept. This script collapses near-duplicate
    mentions into canonical nodes via embedding similarity clustering, producing a
    clean vocabulary for the graph.

Workflow (FOL):
    Load(bio_triplets_full_corpus.msgpack)
        → ExtractSpans(records) → {subject_spans, predicate_spans, object_spans}
        → Dedup(spans) → unique_spans
        → Embed(unique_spans) → embeddings  [static model, CPU, fast]
        → Cluster(embeddings, threshold=0.92) → {span → canonical_id}
        → Save(entity_index.json, entity_embeddings.npy, predicate_index.json, predicate_embeddings.npy)

Node vocabulary:
    Entities  = subject and object spans (same type — both are "things")
    Predicates = relation spans (separate embedding space, used as edge_attr)

Dependencies:
    pip install model2vec  (or sentence-transformers as fallback)
    pip install scikit-learn msgpack numpy tqdm

Usage:
    python graph/normalize_entities.py
    python graph/normalize_entities.py --threshold 0.90 --no-cluster
"""

import argparse
import json
import pickle
import numpy as np
import msgpack
from pathlib import Path
from collections import Counter
from functools import lru_cache
from tqdm import tqdm

import nltk
try:
    from nltk.stem import WordNetLemmatizer as _WNL
    _wnl = _WNL()
except Exception:
    _wnl = None

try:
    from nltk.corpus import wordnet as _wn
    _wn_available = True
    _ = _wn.synsets("test", pos=_wn.NOUN)   # warm up / verify corpus loaded
except Exception:
    _wn = None
    _wn_available = False

# ── Paths (absolute so the script is CWD-independent) ─────────────────────────
_ROOT          = Path(__file__).parent.parent   # workspace root
MSGPACK_PATH   = _ROOT / "checkpoints/quotes_triplets.msgpack"
OUT_DIR        = Path(__file__).parent          # .../graph/
ENTITY_IDX     = OUT_DIR / "entity_index.json"
ENTITY_EMB     = OUT_DIR / "entity_embeddings.npy"
PREDICATE_IDX  = OUT_DIR / "predicate_index.json"
PREDICATE_EMB  = OUT_DIR / "predicate_embeddings.npy"
TRIPLET_MAP         = OUT_DIR / "triplet_map.pkl"   # chunk_id → list of (eid_subj, eid_pred, eid_obj)
ENTITY_SYNSET_FEATS = OUT_DIR / "entity_synset_feats.npy"   # (N_entities, 6) auxiliary WN features

# ── Config ────────────────────────────────────────────────────────────────────
EMBEDDER_PATH      = str(_ROOT / "model2vec_jina")
EMBED_BATCH        = 512
COSINE_THRESHOLD   = 0.92   # spans above this → same canonical node
MIN_SPAN_LEN       = 3      # chars; "a" / "i" / "go" / "mr" noise filtered
MAX_ENTITY_LEN     = 60     # chars; prevents full quote-sentences becoming entity nodes
MAX_ENTITY_WORDS   = 4      # tokens; noun phrases only — no clause fragments
MAX_PREDICATE_LEN  = 80     # chars; prevents clause-length predicate spans
MAX_PREDICATE_WORDS = 5     # tokens; allows "make use of", "is made of"

PREDICATE_STOPWORDS: frozenset = frozenset({
    # bare copula
    'be', 'is', 'are', 'was', 'were', 'been', 'being',
    # bare auxiliaries
    'have', 'has', 'had',
    'do', 'does', 'did',
    # bare modals
    'will', 'would', 'could', 'should', 'shall', 'might', 'may', 'must', 'can', 'cannot',
    # contraction stems and forms (BERT artifacts)
    "aren", "aren't", "isn", "isn't", "wasn", "wasn't", "weren", "weren't",
    "haven", "haven't", "hasn", "hasn't", "hadn", "hadn't",
    "don", "don't", "doesn", "doesn't", "didn", "didn't",
    "won", "won't", "couldn", "couldn't", "can't", "shouldn", "shouldn't",
    "wouldn", "wouldn't", "mustn", "mustn't",
})

# Tokens that are invalid as *sole* content of an entity span.
# Any span whose every token is in this set is rejected.
ENTITY_STOPWORDS: frozenset = frozenset({
    # articles / determiners
    "a", "an", "the", "this", "that", "these", "those",
    # personal / possessive pronouns
    "i", "me", "my", "we", "our", "you", "your",
    "he", "him", "his", "she", "her", "it", "its", "they", "them", "their",
    # to-be forms
    "is", "are", "was", "were", "be", "been", "being",
    # coordinating conjunctions / discourse markers
    "and", "or", "but", "so", "yet", "nor", "for",
    # common prepositions (single-token spans only — multi-token fine)
    "in", "on", "at", "to", "of", "by", "as", "up", "out", "if", "into",
    "under", "over", "upon", "above", "below", "beyond", "among", "across",
    "toward", "towards", "without", "within", "throughout", "per", "around", "along",
    # pointer / approximator modifiers — no entity identity as boundary tokens
    "like", "unlike", "about",
    # quantifiers / comparatives that carry no entity identity
    "more", "less", "most", "least", "very", "much",
    "many", "some", "any", "all", "both", "each", "few",
    "another", "other", "such",
    # common adverbs with no referential content
    "just", "also", "even", "still", "back", "now", "then",
    "here", "there", "not", "no", "only", "well", "too",
    # generic adjectives that slip through ("normal" in "fact we think normal")
    "normal", "good", "great", "new", "old", "long", "right", "big", "same",
    # pronominal / universal referents that become high-degree hub nodes
    "one", "someone", "anyone", "everyone", "no one", "nobody", "somebody",
    "anybody", "nothing", "something", "anything", "everything",
    # interrogative / relative pronouns as standalone objects
    "what", "which", "who", "whom", "whose", "where", "when", "why", "how",
    # subordinating conjunctions / clause-boundary tokens
    "because", "although", "though", "while", "whereas",
    "than", "since", "unless", "until",
    # modal / auxiliary verbs — prevent clause-fragment entities
    "will", "would", "could", "should", "shall", "might",
    "have", "has", "had",
    "do", "does", "did",
    # contraction stems — "don't"→"don", "won't"→"won", etc. (function-word fragments)
    "don", "won", "shouldn", "wouldn", "couldn", "isn", "aren",
    # preposition missing from original list
    "from",
    # abbreviated titles caught as 2-char span noise
    "mr", "mrs", "ms", "dr", "fr",
    # missing preposition ("without" already present but not "with")
    "with",
    # temporal/conditional subordinating conjunctions missing from prior list
    "before", "after",
    # relative adverbs — signal subordinate clause membership
    "wherever", "whenever", "however", "whatever", "whichever",
    # epistemic and manner adverbs with no referential entity identity
    "perhaps", "maybe", "probably", "somehow", "otherwise",
    # politeness/imperative markers
    "please",
    # lone predicate-adjective complements that slip through as objects
    "proud", "free", "alone", "able",
    # comparative/superlative adjectives that are not standalone entity concepts
    "better", "worse", "greater", "lesser", "higher", "lower",
    "larger", "smaller", "longer", "shorter", "stronger", "weaker",
    # missing modals
    "can", "cannot",
    # direct implication and filler words
    "must",                    # modal of necessity (implies rather than asserts)
    "doing", "going",          # gerund continuations with no referential identity
    "get", "got",              # 'to get' phrase components
    # temporal and frequency adverbs with no referential entity identity
    "never", "ever", "always", "sometimes", "often", "seldom", "rarely",
    "once", "twice", "again", "soon", "then", "yet",
    # degree adverbs that combine with adjectives
    "almost", "nearly", "quite", "rather", "enough", "ahead",
    # attribute adjectives with no standalone entity identity
    "next",                    # "next person" → strips to "person"
    # phrasal-verb directional particles — strip to expose bare verb
    "together", "apart",       # "come together", "fall apart" → bare verb only
})


# ── Lemmatization ────────────────────────────────────────────────────────────

# -ly words that look like adverbs but are nouns / proper names → do not strip
_LY_NON_ADVERB: frozenset = frozenset({
    'family', 'holy', 'only', 'early', 'daily', 'rally', 'belly',
    'ally', 'bully', 'jelly', 'poly', 'folly', 'tally', 'valley',
    'melancholy', 'anomaly',
})


def _is_adverb_token(tok: str) -> bool:
    """True if tok is a manner/degree adverb (ends -ly) with no entity identity."""
    return len(tok) > 3 and tok.endswith('ly') and tok not in _LY_NON_ADVERB


@lru_cache(maxsize=32768)
def _lemma_token(tok: str, pos: str) -> str:
    """Lemmatize a single lowercase token. Cached for performance.

    Gerund/present-participle fix: when called for an entity span (pos='n'),
    tokens ending in -ing are first attempted as verbs so gerunds collapse to
    their base verb form before the noun lemmatizer is tried.
      'loving'    -> 'love'     (gerund -> base verb)
      'believing' -> 'believe'
      'ceiling'   -> 'ceiling'  (unchanged: verb lemma == surface form)
    """
    if _wnl is None:
        return tok
    if pos == 'n' and tok.endswith('ing') and len(tok) > 4:
        v = _wnl.lemmatize(tok, pos='v')
        if v != tok:
            return v
    return _wnl.lemmatize(tok, pos=pos)


def _lemmatize_span(text: str, pos: str = 'n') -> str:
    """
    Lemmatize each token in a (already lowercased) span.

    pos='n'  for entities (noun-phrase subjects / objects)
    pos='v'  for predicates (verb-phrase relations)

    For entity spans (pos='n') additional normalization is applied:
      - Manner/degree adverb tokens (ending -ly) are stripped entirely;
        they carry no entity identity.
            'loving strongly' -> 'love'  (adverb stripped, gerund lemmatized)
            'deeply held'     -> 'held'  ('deeply' dropped)
      - Gerund tokens are lemmatized as verbs first via _lemma_token.

    Keeps multi-word spans intact; only inflections are stripped:
      'friendships' -> 'friendship'   (pos='n')
      'believes'    -> 'believe'      (pos='v')
      'true love'   -> 'true love'    (no change for base forms)

    IMPORTANT: function words are skipped. WNL.morphy maps 'as'→'a',
    'was'→'wa', 'has'→'ha', 'does'→'doe' when pos='n' because it
    finds a spurious noun plural match in WordNet. Gate on ENTITY_STOPWORDS
    to avoid corrupting spans containing these tokens.
    """
    if _wnl is None:
        return text
    tokens = []
    for tok in text.split():
        if tok in ENTITY_STOPWORDS:
            tokens.append(tok)
            continue
        if pos == 'n' and _is_adverb_token(tok):
            continue                         # strip manner adverbs from entity spans
        tokens.append(_lemma_token(tok, pos))
    return ' '.join(tokens)


# ── WordNet auxiliary features ────────────────────────────────────────────────

_WN_MAX_OFFSET: float = 1e8   # WordNet offsets fit in [0, ~1e8]; normalize to [0, 1]

# WordNet lexicographer file categories for nouns (26 noun.* files, sorted alphabetically).
# synset.lexname() → e.g. 'noun.feeling', 'noun.state', 'noun.act'
# Replaces the former 4-dim constant POS one-hot (all entities were forced pos='n' → [1,0,0,0]
# for every node, adding zero discriminative signal). A single normalized lexname dim
# encodes semantic type (noun.feeling=person/emotion domain, noun.artifact=physical objects, …)
# that varies meaningfully across the entity vocabulary.
_NOUN_LEXNAMES: list = [
    'noun.Tops', 'noun.act', 'noun.animal', 'noun.artifact', 'noun.attribute',
    'noun.body', 'noun.cognition', 'noun.communication', 'noun.event', 'noun.feeling',
    'noun.food', 'noun.group', 'noun.location', 'noun.motive', 'noun.object',
    'noun.person', 'noun.phenomenon', 'noun.plant', 'noun.possession', 'noun.process',
    'noun.quantity', 'noun.relation', 'noun.shape', 'noun.state', 'noun.substance',
    'noun.time',
]
_LEXNAME_TO_IDX: dict = {name: i for i, name in enumerate(_NOUN_LEXNAMES)}
_N_LEXNAMES: int = len(_NOUN_LEXNAMES)  # 26


def _get_wn_features(canonical: str, pos: str = 'n') -> np.ndarray:
    """
    3-dim float32 auxiliary feature vector for a canonical entity span:
        [lexname_norm,          WN lexicographer-file category (0 = no WN hit)
         synset_offset_norm,    0.0 if OOV / proper noun
         hypernym_offset_norm]  0.0 if no hypernym

    Uses the HEAD WORD of the span (last token) for WordNet lookup —
    rightmost token is the syntactic head of an English NP.
    All dims stay zero when: wordnet unavailable, no synsets found
    (proper nouns and domain-specific OOV terms fall through gracefully).
    """
    feats = np.zeros(3, dtype=np.float32)

    if not _wn_available:
        return feats

    # Head word: last token of the span (syntactic head of English NP)
    head = canonical.split()[-1] if canonical else canonical

    wn_pos = (_wn.NOUN if pos == 'n' else
              _wn.VERB if pos == 'v' else
              _wn.ADJ  if pos == 'j' else _wn.ADV)
    synsets = _wn.synsets(head, pos=wn_pos)
    if not synsets:
        return feats               # proper noun / OOV → all dims stay 0

    syn = synsets[0]
    lexname = syn.lexname() if hasattr(syn, 'lexname') else ''
    lex_idx = _LEXNAME_TO_IDX.get(lexname, -1)
    if lex_idx >= 0:
        feats[0] = lex_idx / (_N_LEXNAMES - 1)   # normalize to [0, 1]

    feats[1] = syn.offset() / _WN_MAX_OFFSET
    hyps = syn.hypernyms()
    if hyps:
        feats[2] = hyps[0].offset() / _WN_MAX_OFFSET

    return feats


def compute_synset_features(n_nodes: int, id_to_canonical: dict,
                             pos: str = 'n') -> np.ndarray:
    """
    Build (n_nodes, 3) float32 auxiliary feature matrix from WordNet.

    Args:
        n_nodes         total node count (= max_id + 1)
        id_to_canonical {node_id: canonical_surface_form (lowercased lemma)}
        pos             'n' for entity nodes, 'v' for predicate nodes
    """
    feats = np.zeros((n_nodes, 3), dtype=np.float32)
    for nid, canon in tqdm(id_to_canonical.items(), desc="  WN features",
                           total=len(id_to_canonical), leave=False):
        feats[nid] = _get_wn_features(canon, pos)
    return feats


# ── Embedder ──────────────────────────────────────────────────────────────────

def load_embedder(path: str):
    """Load static embedder. model2vec preferred (fast, CPU), fallback sentence-transformers."""
    try:
        from model2vec import StaticModel
        print(f"  Loading model2vec from {path}...")
        return StaticModel.from_pretrained(path)
    except (ImportError, Exception) as e:
        print(f"  model2vec unavailable ({e}), trying sentence-transformers...")
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(path)


def embed_texts(model, texts: list, batch_size: int = EMBED_BATCH) -> np.ndarray:
    """Embed list of texts in batches → float32 (N, dim)."""
    results = []
    for i in tqdm(range(0, len(texts), batch_size), desc="  Embedding", leave=False):
        batch = texts[i : i + batch_size]
        emb   = model.encode(batch, show_progress_bar=False)
        results.append(np.array(emb, dtype=np.float32))
    return np.vstack(results)


# ── Span extraction ─────────────────────────────────────────────────────────

_PRONOUN_HEADS: frozenset = frozenset({
    "i", "me", "my", "we", "our", "us",
    "you", "your",
    "he", "him", "his", "she", "her",
    "it", "its", "they", "them", "their",
    # reflexive pronouns
    "myself", "yourself", "himself", "herself", "itself",
    "ourselves", "yourselves", "themselves",
})

def _is_content_predicate(pred: str) -> bool:
    """Reject bare auxiliary/modal predicates with no relational semantic content.

    Bare copulas (is, are, was), bare modals (must, can, should), and BERT
    contraction stems (aren, isn, don…) are filtered.  Multi-word phrases
    that contain at least one non-auxiliary word are retained.
    """
    if not pred:
        return False
    words = pred.split()
    return not all(w in PREDICATE_STOPWORDS for w in words)


def _atomize_span(text: str) -> list:
    """Split a cleaned/lemmatized span on coordinators into atomic sub-spans.

    After splitting on 'and' / 'or' / ',' each piece has its leading/trailing
    ENTITY_STOPWORDS stripped — pointer words like 'like', 'of', 'under' that
    survive inside compound spans are removed at the boundary of atoms.
    Returns at least [text] when no split applies.
        'hatred and war' -> ['hatred', 'war']
        'love and kindness' -> ['love', 'kindness']
        'love' -> ['love']
    """
    import re
    parts = re.split(r'\s+and\s+|\s+or\s+|,\s*|\s+is\s+|\s+are\s+|\s+was\s+|\s+were\s+|\s+of\s+', text, flags=re.IGNORECASE)
    result = []
    for part in parts:
        words = part.strip().split()
        # Strip ENTITY_STOPWORDS FIRST so a leading article/possessive
        # ("your life in fear") cannot shield the hollow-head noun from
        # the regex below.
        while words and words[0] in ENTITY_STOPWORDS:
            words = words[1:]
        while words and words[-1] in ENTITY_STOPWORDS:
            words = words[:-1]
        # Strip hollow-head prepositional phrases AFTER stopwords are removed:
        # "life in fear" -> "fear", "way in X" -> "X"
        part = re.sub(
            r'^(?:life|way|time|role|part|place|state|form|manner)\s+in\s+',
            '', ' '.join(words)
        ).strip()
        cleaned = part
        if cleaned and len(cleaned) >= MIN_SPAN_LEN:
            words_final = cleaned.split()
            if len(words_final) > 1:
                # Multi-token atom: each remaining token is its own semantic
                # predicate / entity (e.g. "clever person" → ["clever", "person"])
                for w in words_final:
                    if len(w) >= MIN_SPAN_LEN:
                        result.append(w)
            else:
                result.append(cleaned)
    return result or [text]


def _is_content_span(text: str) -> bool:
    """Return True iff span is a genuine content entity.

    Checks (in order):
      1. Any token that is a personal/possessive pronoun → False.
         Rejects clause fragments like 'bottle up your anger', 'before she
         is left' that contain pronouns anywhere in the span, not just at
         the head position.
      2. No token may be a BERT ## continuation token.
      3. No apostrophe token (possessive/contraction artifact).
      4. Clause-fragment ratio: if >50% of tokens are ENTITY_STOPWORDS the
         span is a verb/prep phrase, not a noun phrase.  Catches 'mean to
         be human' (2/4 stop), 'thing are good with' (3/4 stop), etc.
      5. At least one non-stopword token must remain.
    Applied only to subject/object spans — predicates are exempt.
    """
    tokens = text.split()
    if not tokens:
        return False
    # ANY pronoun anywhere in span → clause fragment, not a noun phrase
    if any(tok in _PRONOUN_HEADS for tok in tokens):
        return False
    if any(tok.startswith('##') for tok in tokens):
        return False
    # Reject leftover BERT contraction/possessive fragments not cleaned by _strip_possessive
    # (e.g. "shouldn ' t", "others ' opinion", "saying ' time")
    if "'" in tokens:
        return False
    # Clause-fragment guard: spans with ≥50% function-word tokens are verb/prep phrases
    if len(tokens) > 1 and sum(1 for t in tokens if t in ENTITY_STOPWORDS) * 2 >= len(tokens):
        return False
    return any(tok not in ENTITY_STOPWORDS for tok in tokens)


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


def _strip_possessive(text: str) -> str:
    """
    Normalize BERT-tokenized possessives and leading contraction fragments.

    BERT tokenizes ``person's`` as ['person', "'", 's'], which
    ``_join_wordpieces`` reconstructs as ``person ' s body``.
    This function reduces genitive possession to the possessum
    (the thing possessed): ``person ' s body`` -> ``body``.
    Leading contraction fragments such as ``' re capable`` or
    ``' s self`` are stripped to their content portion.
    Remaining bare apostrophe tokens are cleaned.
    """
    import re
    # Genitive: "X ' s Y" -> keep only Y (the possessum)
    m = re.search(r"^.*\s+'\s+s\s+(.+)$", text)
    if m:
        return m.group(1).strip()
    # Leading contraction fragment: "' re capable" -> "capable"
    text = re.sub(r"^'\s+\S+\s*", "", text).strip()
    # Remaining leading apostrophe
    text = re.sub(r"^'\s*", "", text).strip()
    return ' '.join(text.split())


def extract_spans(data: list) -> tuple[Counter, Counter]:
    """
    Extract entity and predicate surface forms from triplet records.

    Args:
        data: list of {doc_id, triplets: [{subject, predicate, object, ...}]}

    Returns:
        entity_counts    Counter({span: frequency})
        predicate_counts Counter({span: frequency})
    """
    entities   = Counter()
    predicates = Counter()

    for record in tqdm(data, desc="  Extracting spans"):
        for t in record.get("triplets", []):
            subj_raw = _lemmatize_span(_strip_possessive(_fix_encoding((t.get("subject")   or "").strip().lower())), 'n')
            pred     = _lemmatize_span(_strip_possessive(_fix_encoding((t.get("predicate") or "").strip().lower())), 'v')
            obj_raw  = _lemmatize_span(_strip_possessive(_fix_encoding((t.get("object")    or "").strip().lower())), 'n')
            for subj in _atomize_span(subj_raw):
                if (MIN_SPAN_LEN <= len(subj) <= MAX_ENTITY_LEN
                        and len(subj.split()) <= MAX_ENTITY_WORDS
                        and _is_content_span(subj)):             entities[subj]   += 1
            if (MIN_SPAN_LEN <= len(pred) <= MAX_PREDICATE_LEN
                    and len(pred.split()) <= MAX_PREDICATE_WORDS
                    and _is_content_predicate(pred)):   predicates[pred] += 1
            for obj in _atomize_span(obj_raw):
                if (MIN_SPAN_LEN <= len(obj)  <= MAX_ENTITY_LEN
                        and len(obj.split())  <= MAX_ENTITY_WORDS
                        and _is_content_span(obj)):              entities[obj]    += 1

    return entities, predicates


# ── Clustering ────────────────────────────────────────────────────────────────

def cluster_greedy(spans_counter: Counter, embs_norm: np.ndarray,
                   threshold: float) -> tuple[dict, dict, np.ndarray]:
    """
    Greedy embedding cluster: process spans by descending frequency.
    High-frequency spans become cluster seeds; rare spans join the nearest
    existing cluster if cosine-sim >= threshold.

    Complexity: O(N * C * dim) where C = number of clusters (C << N when threshold is high).
    At threshold=0.92, English entity spans cluster aggressively (5-10x reduction typical).

    Pre-allocates centroid buffer (N, dim) to avoid O(N²) memory from np.vstack.
    Only the live slice [:n_clusters] is used for dot-products.

    Returns:
        span_to_canonical  {span: canonical_span}
        canonical_to_id    {canonical_span: node_id}
        centroid_embs      float32 (n_nodes, dim)
    """
    spans   = list(spans_counter.keys())
    order   = sorted(range(len(spans)), key=lambda i: -spans_counter[spans[i]])

    N, dim         = embs_norm.shape
    centroid_buf   = np.empty((N, dim), dtype=np.float32)   # pre-allocated
    n_clusters     = 0
    members: dict  = {}   # cid → [span, ...]
    span2clust     = {}

    for idx in tqdm(order, desc="  Clustering", miniters=1000):
        span = spans[idx]
        emb  = embs_norm[idx]   # (dim,)

        if n_clusters == 0:
            centroid_buf[0] = emb
            n_clusters = 1
            members[0] = [span]
            span2clust[span] = 0
            continue

        sims     = centroid_buf[:n_clusters] @ emb   # (n_clusters,) — view, no copy
        best_idx = int(np.argmax(sims))

        if sims[best_idx] >= threshold:
            members[best_idx].append(span)
            span2clust[span] = best_idx
            # Running-mean centroid update (stays normalized)
            new_c = centroid_buf[best_idx] + emb
            new_c /= (np.linalg.norm(new_c) + 1e-9)
            centroid_buf[best_idx] = new_c
        else:
            cid = n_clusters
            centroid_buf[cid] = emb
            n_clusters += 1
            members[cid] = [span]
            span2clust[span] = cid

    # Canonical = most-frequent member per cluster
    span_to_canonical = {}
    canonical_to_id   = {}
    for cid in range(n_clusters):
        mbrs  = members[cid]
        canon = max(mbrs, key=lambda s: spans_counter[s])
        canonical_to_id[canon] = cid
        for m in mbrs:
            span_to_canonical[m] = canon

    centroid_arr = centroid_buf[:n_clusters].copy()
    return span_to_canonical, canonical_to_id, centroid_arr


def build_index(spans_counter: Counter, embedder, threshold: float,
                do_cluster: bool) -> tuple[dict, np.ndarray]:
    """
    Full pipeline for one span vocabulary (entities or predicates).

    Returns:
        span_to_id   {span: node_id}
        embeddings   float32 (n_nodes, dim)
    """
    spans = list(spans_counter.keys())
    print(f"    Unique spans: {len(spans):,}")

    print(f"    Embedding...")
    embs      = embed_texts(embedder, spans)
    norms     = np.linalg.norm(embs, axis=1, keepdims=True)
    embs_norm = embs / (norms + 1e-9)

    if do_cluster:
        print(f"    Clustering (threshold={threshold})...")
        span_to_canonical, canonical_to_id, centroid_embs = cluster_greedy(
            spans_counter, embs_norm, threshold
        )
        span_to_id = {s: canonical_to_id[c] for s, c in span_to_canonical.items()}
        n_nodes    = len(canonical_to_id)
        print(f"    Collapsed {len(spans):,} -> {n_nodes:,} canonical nodes "
              f"({100*(1-n_nodes/len(spans)):.1f}% reduction)")
        return span_to_id, centroid_embs
    else:
        # No clustering: each unique lowercase span is its own node
        span_to_id = {s: i for i, s in enumerate(spans)}
        print(f"    No clustering: {len(spans):,} nodes")
        return span_to_id, embs_norm


# ── Triplet map ───────────────────────────────────────────────────────────────

def build_triplet_map(data: list, entity_to_id: dict,
                      predicate_to_id: dict) -> dict:
    """
    Map each chunk_id to its list of (entity_id_subj, predicate_id, entity_id_obj) triples.
    Spans not in the vocabulary (too short, filtered) are skipped.

    Returns:
        {chunk_id: [(eid_subj, pid, eid_obj), ...]}
    """
    triplet_map = {}
    skipped = 0

    for record in tqdm(data, desc="  Building triplet map"):
        chunk_id = record.get("doc_id") or record.get("chunk_id", "")
        triples  = []
        for t in record.get("triplets", []):
            subj_raw = _lemmatize_span((t.get("subject")    or "").strip().lower(), 'n')
            pred     = _lemmatize_span((t.get("predicate")  or "").strip().lower(), 'v')
            obj_raw  = _lemmatize_span((t.get("object")     or "").strip().lower(), 'n')
            pid = predicate_to_id.get(pred)
            if pid is None:
                skipped += 1
                continue
            for s_atom in _atomize_span(subj_raw):
                eid_s = entity_to_id.get(s_atom)
                if eid_s is None:
                    skipped += 1
                    continue
                for o_atom in (_atomize_span(obj_raw) if obj_raw else []):
                    eid_o = entity_to_id.get(o_atom)
                    if eid_o is not None:
                        triples.append((eid_s, pid, eid_o))
                    else:
                        skipped += 1
        if triples:
            triplet_map[chunk_id] = triples

    print(f"  Mapped {sum(len(v) for v in triplet_map.values()):,} triplets, "
          f"skipped {skipped:,} (filtered spans)")
    return triplet_map


# ── Main ──────────────────────────────────────────────────────────────────────

def main(threshold: float = COSINE_THRESHOLD, do_cluster: bool = True,
         msgpack_override: str | None = None):
    OUT_DIR.mkdir(exist_ok=True)

    src = Path(msgpack_override) if msgpack_override else MSGPACK_PATH
    assert src.exists(), (
        f"Inference output not found: {src}\n"
        "Run apply_bio_corpus.py first, or pass --msgpack <path>."
    )

    print(f"Loading {src}...")
    with open(src, "rb") as f:
        data = msgpack.unpackb(f.read(), raw=False)
    print(f"  Records: {len(data):,}")

    print("Extracting spans...")
    entity_counts, predicate_counts = extract_spans(data)
    print(f"  Entities unique: {len(entity_counts):,}, "
          f"Predicates unique: {len(predicate_counts):,}")
    print(f"  Top entities:   {entity_counts.most_common(5)}")
    print(f"  Top predicates: {predicate_counts.most_common(5)}")

    print("Loading embedder...")
    embedder = load_embedder(EMBEDDER_PATH)

    print("Building entity vocabulary...")
    entity_to_id, entity_embs = build_index(entity_counts, embedder, threshold, do_cluster)

    print("Building predicate vocabulary...")
    predicate_to_id, predicate_embs = build_index(predicate_counts, embedder, threshold, do_cluster)

    # Canonical span per entity node-id (most-frequent member = canonical by cluster_greedy)
    id_to_canonical_e: dict = {}
    for span, nid in entity_to_id.items():
        if nid not in id_to_canonical_e or entity_counts[span] > entity_counts[id_to_canonical_e[nid]]:
            id_to_canonical_e[nid] = span

    print("Computing entity WordNet auxiliary features...")
    entity_synset_feats = compute_synset_features(len(entity_embs), id_to_canonical_e, pos='n')
    n_nonzero = int((entity_synset_feats[:, 1] > 0).sum())  # col 1 = synset_offset_norm
    print(f"  WN synset coverage: {n_nonzero:,}/{len(entity_embs):,} entities "
          f"({100*n_nonzero/max(1, len(entity_embs)):.1f}%)")

    print("Building triplet map...")
    triplet_map = build_triplet_map(data, entity_to_id, predicate_to_id)

    # Persist
    print("Saving...")
    with open(ENTITY_IDX, "w") as f:
        json.dump({"span_to_id": entity_to_id, "n_nodes": len(entity_embs)}, f)
    np.save(ENTITY_EMB, entity_embs)
    np.save(ENTITY_SYNSET_FEATS, entity_synset_feats)

    with open(PREDICATE_IDX, "w") as f:
        json.dump({"span_to_id": predicate_to_id, "n_nodes": len(predicate_embs)}, f)
    np.save(PREDICATE_EMB, predicate_embs)

    with open(TRIPLET_MAP, "wb") as f:
        pickle.dump(triplet_map, f)

    print("\nDone")
    print(f"  Entity nodes:    {len(entity_embs):,}  -> {ENTITY_IDX}")
    print(f"  Predicate nodes: {len(predicate_embs):,} -> {PREDICATE_IDX}")
    print(f"  Triplet map:     {len(triplet_map):,} chunks -> {TRIPLET_MAP}")
    print(f"  Emb shapes: entity={entity_embs.shape}, predicate={predicate_embs.shape}")
    print(f"  WN feats:   entity={entity_synset_feats.shape} -> {ENTITY_SYNSET_FEATS}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=COSINE_THRESHOLD,
                        help="Cosine similarity threshold for entity clustering (default 0.92)")
    parser.add_argument("--no-cluster", action="store_true",
                        help="Skip clustering: each unique span is its own node")
    parser.add_argument("--msgpack", type=str, default=None,
                        help="Override input msgpack path (default: checkpoints/bio_triplets_full_corpus.msgpack)")
    args = parser.parse_args()
    main(threshold=args.threshold, do_cluster=not args.no_cluster, msgpack_override=args.msgpack)
