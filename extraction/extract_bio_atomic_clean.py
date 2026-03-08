"""Extract BIO-tagged training data using Stanza + atomic decomposition pipeline.

Pipeline (based on kg_extractor.py reference + dependency parsing paper):
0. Load pre-chunked texts from checkpoints/chunks.msgpack  
1. Sample N chunks randomly
2. Break into sentences using Stanza
3. Extract SPO triplets using dependency parsing
4. Apply atomic decomposition pipeline:
   - Fix encoding artifacts
   - Stopword removal  
   - POS filtering (exclude DT/IN/RB/MD)
   - Lemmatization
   - Deduplication
   - Atomic token decomposition (cartesian product)
   - Position conflict resolution
   - Auxiliary verb filtering
5. Generate BIO labels (B-SUBJ, B-PRED, B-OBJ only - no I-tags needed since atomic)

Usage:
    python extract_bio_atomic_clean.py --chunks 250 --output bio_training_atomic_clean.msgpack
"""

import argparse
import msgpack
import numpy as np
import random
import re
import stanza
import cleantext
import wordninja
import string
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import BertTokenizer
from scipy import stats
import nltk

# Download NLTK data if needed
try:
    stop_words = set(stopwords.words('english'))
except:
    nltk.download('stopwords')
    nltk.download('wordnet')
    stop_words = set(stopwords.words('english'))

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# POS tags to EXCLUDE (negative inference)
OTHER_POS = {"CC", "DT", "EX", "IN", "LS", "PDT", "POS", "RB", "RBR", "RBS", "RP", "SYM", "TO", "UH", "WDT", "WRB"}
ENTITY_EXCLUDE_POS = OTHER_POS | {'MD'}  # Entities (subject/object) - no modals
PREDICATE_EXCLUDE_POS = OTHER_POS  # Predicates - keep modals

# Auxiliary/light verbs to exclude from predicates.
# Only true grammatical function words — NOT content verbs.
# Rule: a token is functional if its removal loses NO propositional content.
#   is/are/was/were  → copulas: grammatical plumbing; complement promoted as PRED
#   have/has/had     → possession aux or perfect tense aux
#   do/does/did      → do-support; no independent meaning
#   modals           → epistemic operators (can/could/may/…)
#   let/make/get/take → light verbs (borderline; kept for safety)
# NOT included: contain, include, comprise — these carry real semantic content
#   ("A contains B", "A includes B") and should be labelled as PRED spans.
AUXILIARY_VERBS = {'be', 'am', 'is', 'are', 'was', 'were', 'been', 'being',
                   'have', 'has', 'had', 'having',
                   'do', 'does', 'did', 'doing',
                   'can', 'could', 'may', 'might', 'must', 'shall', 'should', 'will', 'would',
                   'let', 'make', 'get', 'take'}

parser = argparse.ArgumentParser(description='Extract BIO-tagged training data using Stanza + atomic decomposition')
parser.add_argument('--chunks', type=int, default=250, help='Number of chunks to sample')
parser.add_argument('--input', type=str, default=r'checkpoints\chunks.msgpack',
                   help='Input msgpack file with pre-chunked texts')
parser.add_argument('--output', type=str, default='bio_training_atomic_clean.msgpack',
                   help='Output file path')
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
args = parser.parse_args()

# Set random seed
random.seed(args.seed)
np.random.seed(args.seed)

print(f"\nLoading Stanza pipeline (tokenize, pos, lemma, depparse)...")
nlp = stanza.Pipeline('en', processors='tokenize,pos,lemma,depparse', use_gpu=False, download_method=None)
print("[OK] Stanza loaded")

print(f"Loading BERT tokenizer...")
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
print("[OK] BERT tokenizer loaded\n")


# ============================================================================
# Preprocessing Functions (from kg_extractor.py reference)
# ============================================================================

def fix_encoding_artifacts(text: str) -> str:
    """Fix common encoding corruption before extraction."""
    replacements = {
        'â€™': "'",
        'â€œ': '"',
        'â€': '"',
        'â€"': '—',
        'â€"': '–',
        'Â': '',
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    return text

def remove_stopwords(text: str) -> str:
    """Remove stop words from text, preserving word order."""
    if not text:
        return ''
    words = text.split()
    filtered = [w for w in words if w.lower() not in stop_words]
    return ' '.join(filtered)

def detect_space_density_outliers(texts: List[str], percentile_low=2.5, percentile_high=97.5) -> Set[int]:
    """Detect texts with outlier space density using log-transformed median + MAD.
    
    Args:
        texts: List of text strings to analyze
        percentile_low: Lower percentile for outlier threshold (default 2.5)
        percentile_high: Upper percentile for outlier threshold (default 97.5)
    
    Returns:
        Set of indices where wordninja should be applied
    """
    if not texts:
        return set()
    
    # Compute space density: (count of spaces) / (total length)
    densities = []
    for text in texts:
        if len(text) > 0:
            space_count = text.count(' ')
            density = space_count / len(text)
            # Add small epsilon to avoid log(0)
            densities.append(np.log(density + 1e-10))
        else:
            densities.append(np.nan)
    
    # Filter out NaN values for stats computation
    valid_densities = [d for d in densities if not np.isnan(d)]
    if not valid_densities:
        return set()
    
    # Median + MAD approach (robust to outliers)
    median = np.median(valid_densities)
    mad = np.median(np.abs(np.array(valid_densities) - median))
    # MAD → std conversion factor: 1.4826 (for normal distribution)
    std_estimate = 1.4826 * mad
    
    # Transform percentiles to z-scores (approximate for normal)
    from scipy import stats
    z_low = stats.norm.ppf(percentile_low / 100.0)
    z_high = stats.norm.ppf(percentile_high / 100.0)
    
    # Boundaries in log space
    lower_bound = median + z_low * std_estimate
    upper_bound = median + z_high * std_estimate
    
    # Find outliers
    outlier_indices = set()
    for i, log_density in enumerate(densities):
        if np.isnan(log_density):
            continue
        if log_density < lower_bound or log_density > upper_bound:
            outlier_indices.add(i)
    
    return outlier_indices

def should_apply_wordninja(text: str, all_texts: List[str], idx: int, outlier_cache: Set[int] = None) -> bool:
    """Check if wordninja should be applied to this text based on space density.
    
    Args:
        text: The text to check
        all_texts: All texts in batch (for statistical context)
        idx: Index of current text in all_texts
        outlier_cache: Pre-computed outlier indices (optional)
    
    Returns:
        True if wordninja should be applied
    """
    if outlier_cache is not None:
        return idx in outlier_cache
    
    # Compute on-the-fly if no cache
    outliers = detect_space_density_outliers(all_texts)
    return idx in outliers

def filter_pos_stanza(tokens: List, role: str = 'entity') -> List:
    """Filter tokens by POS tags using Stanza output.
    
    Args:
        tokens: List of Stanza Token objects
        role: 'entity' (subject/object) or 'predicate'
    
    Returns:
        Filtered list of token texts
    """
    exclude_set = ENTITY_EXCLUDE_POS if role == 'entity' else PREDICATE_EXCLUDE_POS
    return [t.text for t in tokens if t.xpos not in exclude_set]

def lemmatize_tokens_stanza(tokens: List) -> str:
    """Lemmatize tokens using Stanza lemmas."""
    return ' '.join([t.lemma.lower() for t in tokens if t.lemma])

def deduplicate_phrase(text: str) -> str:
    """Remove duplicate words within phrase while preserving order."""
    if not text:
        return ''
    words = text.split()
    return ' '.join(dict.fromkeys(words))

def is_valid_extraction(text: str) -> bool:
    """Filter out invalid extractions."""
    if not text or len(text) < 2:
        return False
    # Check if mostly punctuation
    alpha_count = sum(c.isalpha() for c in text)
    if alpha_count < len(text) * 0.5:
        return False
    # Single-word fragments must be 3+ chars
    words = text.split()
    if len(words) == 1 and len(text) < 3:
        return False
    return True

def is_valid_predicate(text: str) -> bool:
    """Check if predicate is semantic (not auxiliary/light verb)."""
    if not is_valid_extraction(text):
        return False
    tokens = text.lower().split()
    lemmas = [lemmatizer.lemmatize(token, pos='v') for token in tokens]
    return not any(lemma in AUXILIARY_VERBS for lemma in lemmas)


# ============================================================================
# Dependency Parsing SPO Extraction (from reference paper approach)
# ============================================================================

def get_subtree_text(head_word, sent, skip_deps=None) -> str:
    """Get all text tokens in the subtree rooted at head_word (for complete phrases).

    skip_deps: set of dependency relation prefixes to prune (e.g. {'relcl', 'acl'}).
    Use for noun-phrase heads to avoid non-contiguous relative/adjectival clauses.
    """
    if skip_deps is None:
        skip_deps = set()
    subtree_words = [head_word]

    # Recursively collect all descendants, pruning skip_deps branches
    def collect_descendants(word_id):
        for w in sent.words:
            if w.head == word_id:
                # deprel may be 'acl:relcl' — compare the base relation
                base_dep = w.deprel.split(':')[0]
                if base_dep in skip_deps:
                    continue
                subtree_words.append(w)
                collect_descendants(w.id)

    collect_descendants(head_word.id)

    # Sort by position and join
    sorted_words = sorted(subtree_words, key=lambda x: x.id)
    return ' '.join([w.text for w in sorted_words])


def extract_spo_from_sentence(sent) -> List[Dict]:
    """Extract Subject-Predicate-Object triplets from Stanza sentence using dependency parse.

    Handles two construction types following Universal Dependencies conventions:

    Regular verbs:
        subject→nsubj of VERB ROOT, object→obj/obl of VERB ROOT
        e.g. "Transformers learn representations"
             ROOT=learn(VERB), nsubj=Transformers, obj=representations
             → (Transformers, learn, representations)

    Copula constructions (UD style):
        NOUN/PROPN/ADJ is ROOT; the copula verb (is/are/was) is a 'cop' child.
        The nominal/adjectival ROOT head IS the semantic predicate — not the copula.
        e.g. "The sky is blue"
             ROOT=blue(ADJ), cop=is, nsubj=sky
             → (sky, blue, ?)
        e.g. "Alice is a teacher"
             ROOT=teacher(NOUN), cop=is, nsubj=Alice
             → (Alice, teacher, ?)

    Args:
        sent: Stanza Sentence object with dependency parse

    Returns:
        List of triplets as dicts: {'subject': str, 'predicate': str, 'object': str}
    """
    triplets = []

    # Regular verbs: VERB/AUX at ROOT or conj, but skip pure copulas (deprel='cop').
    # In UD, copula verbs have deprel='cop' and are NOT the ROOT, so this guard is
    # belt-and-suspenders — it prevents edge cases where the parser assigns AUX as ROOT.
    verbs = []
    for word in sent.words:
        if word.upos in ['VERB', 'AUX'] and (word.head == 0 or word.deprel in ['ROOT', 'conj']):
            if word.deprel != 'cop':
                verbs.append(word)

    # Copula constructions (UD style): NOUN/PROPN/ADJ is ROOT, 'is/are/was' is cop child.
    # The head nominal/adjective carries the semantic content and becomes the PRED span.
    copular_predicates = []
    for word in sent.words:
        if word.upos in ['NOUN', 'PROPN', 'ADJ'] and (word.head == 0 or word.deprel in ['ROOT', 'conj']):
            has_cop = any(w.head == word.id and w.deprel == 'cop' for w in sent.words)
            if has_cop:
                copular_predicates.append(word)

    if not verbs and not copular_predicates:
        return []

    # -------------------------------------------------------------------------
    # Process regular verbs
    # -------------------------------------------------------------------------
    for verb in verbs:
        subject_head = None
        object_head = None

        for word in sent.words:
            if word.head == verb.id:
                if word.deprel in ['nsubj', 'nsubj:pass', 'nsubj:outer', 'csubj']:
                    subject_head = word
                elif word.deprel in ['obj', 'dobj', 'iobj', 'ccomp', 'xcomp']:
                    object_head = word  # direct object — never overwrite with obl
                elif word.deprel == 'obl':
                    # Allow last-obl-wins when no direct obj, but never overwrite obj/dobj/iobj
                    if object_head is None or object_head.deprel == 'obl':
                        object_head = word

        # Fallback: phrasal complement (e.g. "focus on X", "try to achieve X")
        # advcl/xcomp children indicate the verb has a clausal complement; use
        # the inner verb's direct object as the outer object rather than the
        # inner verb itself, so we capture the nominal (e.g. "knowledge") not
        # the gerund ("updating").
        if object_head is None:
            for word in sent.words:
                if word.head == verb.id and word.deprel in ['advcl', 'xcomp']:
                    inner_obj = next(
                        (w for w in sent.words
                         if w.head == word.id and w.deprel in ['obj', 'dobj', 'obl']),
                        None
                    )
                    object_head = inner_obj if inner_obj is not None else word
                    break

        subject_text = get_subtree_text(subject_head, sent, skip_deps={'relcl', 'acl', 'appos'}) if subject_head else '?'
        predicate_text = verb.text
        object_text = get_subtree_text(object_head, sent, skip_deps={'relcl', 'acl', 'appos'}) if object_head else '?'

        if predicate_text and (subject_text != '?' or object_text != '?'):
            triplets.append({
                'subject': subject_text if subject_text != '?' else '?',
                'predicate': predicate_text,
                'object': object_text if object_text != '?' else '?',
            })

    # -------------------------------------------------------------------------
    # Process copula constructions — cop verb becomes predicate, nominal/adj
    # head becomes object (e.g. 'is cumbersome' → P='may be', O='cumbersome';
    # 'is a challenge' → P='is', O='a multi-step reasoning challenge')
    # -------------------------------------------------------------------------
    for pred_nominal in copular_predicates:
        subject_head = None

        for word in sent.words:
            if word.head == pred_nominal.id:
                if word.deprel in ['nsubj', 'nsubj:pass', 'nsubj:outer', 'csubj']:
                    subject_head = word

        # Build predicate from cop + any aux (e.g. 'may' + 'be' → 'may be')
        cop_word = next((w for w in sent.words if w.head == pred_nominal.id and w.deprel == 'cop'), None)
        aux_words = [w for w in sent.words if w.head == pred_nominal.id and w.deprel == 'aux']

        if cop_word:
            pred_parts = [w.text for w in sorted(aux_words + [cop_word], key=lambda x: x.id)]
            predicate_text = ' '.join(pred_parts)
            # Nominal/adj head is the object; skip clausal subjects, coordinations,
            # and adverbial clauses so only the immediate nominal/adj phrase is kept.
            object_text = get_subtree_text(pred_nominal, sent,
                                           skip_deps={'relcl', 'acl', 'nsubj', 'csubj',
                                                      'cop', 'aux', 'conj', 'cc',
                                                      'advcl', 'parataxis'})
        else:
            # Fallback: no cop found — use nominal text as predicate (original behaviour)
            predicate_text = pred_nominal.text
            object_text = '?'

        subject_text = get_subtree_text(subject_head, sent, skip_deps={'relcl', 'acl'}) if subject_head else '?'

        if subject_text != '?' or object_text != '?':
            triplets.append({
                'subject': subject_text if subject_text != '?' else '?',
                'predicate': predicate_text,
                'object': object_text if object_text != '?' else '?',
            })

    return triplets


# ============================================================================
# Atomic Decomposition Pipeline
# ============================================================================

def apply_atomic_pipeline(triplets: List[Dict]) -> List[Dict]:
    """Apply cleaning pipeline to raw triplets (NO cartesian product).
    
    Pipeline steps:
    1. Fix encoding artifacts
    2. Apply cleantext
    3. Apply wordninja ONLY to outlier space density (median+MAD)
    4. Remove stopwords
    5. POS filtering (position-aware)
    6. Lowercase raw text (no lemmatization)
    7. Deduplication
    8. Validation (empty strings, auxiliary verbs)
    
    Args:
        triplets: List of raw triplet dicts
    
    Returns:
        List of cleaned triplets (multi-word spans preserved)
    """
    if not triplets:
        return []
    
    # Collect all text components for space density outlier detection
    all_components = []
    component_map = []  # (triplet_idx, role)
    for i, triplet in enumerate(triplets):
        for role in ['subject', 'predicate', 'object']:
            all_components.append(triplet[role])
            component_map.append((i, role))
    
    # Detect outliers once for entire batch
    outlier_indices = detect_space_density_outliers(all_components)
    
    processed = []
    
    for triplet_idx, triplet in enumerate(triplets):
        # Skip invalid triplets
        if triplet['subject'] == '?' and triplet['object'] == '?':
            continue
        
        # Step 1-2: Fix encoding + cleantext
        subj = fix_encoding_artifacts(triplet['subject'])
        pred = fix_encoding_artifacts(triplet['predicate'])
        obj = fix_encoding_artifacts(triplet['object'])
        
        subj = cleantext.clean(subj, lower=False, no_emoji=True)
        pred = cleantext.clean(pred, lower=False, no_emoji=True)
        obj = cleantext.clean(obj, lower=False, no_emoji=True)
        
        # Step 3: Selective wordninja (only for space density outliers)
        for comp_idx, (t_idx, role) in enumerate(component_map):
            if t_idx == triplet_idx:
                if comp_idx in outlier_indices:
                    if role == 'subject':
                        subj = ' '.join(wordninja.split(subj))
                    elif role == 'predicate':
                        pred = ' '.join(wordninja.split(pred))
                    elif role == 'object':
                        obj = ' '.join(wordninja.split(obj))
        
        # Step 4: Remove stopwords
        subj = remove_stopwords(subj)
        pred = remove_stopwords(pred)
        obj = remove_stopwords(obj)
        
        # Step 5-6: POS filter + lowercase raw text using Stanza
        # Process each component separately
        for idx, (component, role) in enumerate([(subj, 'entity'), (pred, 'predicate'), (obj, 'entity')]):
            if not component or component == '?':
                continue
                
            # Parse with Stanza for POS filtering
            doc = nlp(component)
            if not doc.sentences:
                continue
                
            sent = doc.sentences[0]
            
            # Filter by POS
            filtered_tokens = [t for t in sent.words if t.xpos not in (ENTITY_EXCLUDE_POS if role == 'entity' else PREDICATE_EXCLUDE_POS)]
            
            if not filtered_tokens:
                continue
            
            # Keep raw text (lowercase only, no lemmatization)
            raw_text = ' '.join([t.text.lower() for t in filtered_tokens])
            
            # Step 7: Deduplicate
            raw_text = deduplicate_phrase(raw_text)
            
            # Update component
            if idx == 0:
                subj = raw_text
            elif idx == 1:
                pred = raw_text
            else:
                obj = raw_text
        
        # Step 8: Validation (NO cartesian product — keep multi-word triplets)
        if not is_valid_extraction(pred) or not is_valid_predicate(pred):
            continue
        
        if subj != '?' and not is_valid_extraction(subj):
            continue
        if obj != '?' and not is_valid_extraction(obj):
            continue
        
        # Validate predicate doesn't contain only auxiliary verbs
        pred_tokens = pred.split()
        if all(token in AUXILIARY_VERBS for token in pred_tokens):
            continue
        
        # Store cleaned triplet (multi-word spans preserved)
        processed.append({
            'subject': subj if subj else '?',
            'predicate': pred,
            'object': obj if obj else '?'
        })
    
    return processed


# ============================================================================
# BIO Label Generation (Multi-hot for overlapping triplets)
# ============================================================================

def create_bio_labels(sentence: str, triplets: List[Dict]) -> Tuple[List[str], List[List[int]]]:
    """Create BIO labels using character-span to token-span mapping.
    
    Multi-word spans preserved from Stanza parsing:
      Triplet: ("deep learning models", "tend", "inductive bias")
      Sentence: "deep learning models tend to exhibit inductive bias"
      
      Labels:
        deep → B-SUBJ (begin subject span)
        learning → I-SUBJ (inside subject span)
        models → I-SUBJ (inside subject span)
        tend → B-PRED
        to → O
        exhibit → O
        inductive → B-OBJ (begin object span)
        bias → I-OBJ (inside object span)
    
    Algorithm:
    1. Find character positions of each SPO phrase in sentence
    2. Map character spans to BERT token indices
    3. First token in span → B-tag, continuation tokens → I-tag
    
    Args:
        sentence: Original sentence text
        triplets: List of triplet dicts with multi-word 'subject', 'predicate', 'object'
    
    Returns:
        (bert_tokens, labels) where labels is [n_bert_tokens, 6] with columns:
        [B-SUBJ, I-SUBJ, B-PRED, I-PRED, B-OBJ, I-OBJ]
    """
    # Tokenize with BERT (without special tokens)
    bert_encoding = bert_tokenizer(sentence, add_special_tokens=False)
    bert_tokens = bert_tokenizer.convert_ids_to_tokens(bert_encoding['input_ids'])
    
    # Manually build offset mapping
    sentence_lower = sentence.lower()
    offset_mapping = []
    char_pos = 0
    
    for tok in bert_tokens:
        clean_tok = tok.replace('##', '')
        
        if not tok.startswith('##'):
            # Fresh token - find it in sentence
            next_pos = sentence_lower.find(clean_tok, char_pos)
            if next_pos != -1:
                char_pos = next_pos
        
        # Record this token's character span
        tok_start = char_pos
        tok_end = char_pos + len(clean_tok)
        offset_mapping.append((tok_start, tok_end))
        
        if not tok.startswith('##'):
            char_pos = tok_end
    
    # Initialize labels: [n_bert_tokens, 6]
    labels = np.zeros((len(bert_tokens), 6), dtype=np.int32)
    
    # ========================================================================
    # STEP 1: Find triplet components using fuzzy token-level matching
    # ========================================================================
    
    def normalize_for_matching(text: str) -> str:
        """Normalize text for fuzzy matching (handle Stanza normalization differences).
        
        Handles:
        - Hyphen spacing: "fine-grained" vs "fine - grained"
        - Punctuation spacing: "controllability," vs "controllability ,"
        - Case differences
        - Remove punctuation entirely for matching (cleaner word-level matching)
        """
        import re
        normalized = text.lower()
        # Normalize hyphen spacing
        normalized = normalized.replace(' - ', '-')
        # Remove punctuation (except hyphens which are part of words)
        # This handles "controllability,which" → "controllability which"
        normalized = re.sub(r'[^\w\s-]', ' ', normalized)
        # Collapse multiple spaces
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        return normalized
    
    def find_phrase_fuzzy(phrase: str, sentence_norm: str, max_token_gap: int = 4) -> List[Tuple[int, int]]:
        """Find phrase in sentence using fuzzy token-level matching.
        
        Args:
            phrase: Text to find (e.g. "fine - grained controllability")
            sentence_norm: Normalized sentence text
            max_token_gap: Maximum number of tokens allowed between matches (for punctuation)
        
        Returns:
            List of (start_char, end_char) spans where phrase is found
        """
        if not phrase or phrase == '?':
            return []
        
        # Normalize phrase
        phrase_norm = normalize_for_matching(phrase)
        
        # Split into words (filter empty strings)
        # Strip leading/trailing punctuation-only tokens (e.g. standalone "-" from
        # ccomp objects like "- starting at 0 089") which can't be found in the
        # sentence as standalone tokens and cause word_positions to have an empty list.
        phrase_words = [w for w in phrase_norm.split() if w]
        # Remove tokens that are purely punctuation / non-word characters
        import re as _re
        phrase_words = [w for w in phrase_words if _re.search(r'\w', w)]
        if not phrase_words:
            return []
        
        # For very long phrases (likely complex objects), use exact substring match first
        if len(phrase_words) > 8:
            # Try exact match in normalised sentence (both sides stripped of punctuation)
            pos = sentence_norm.find(phrase_norm)
            if pos != -1:
                # phrase_norm found as continuous substring in sentence_norm.
                # Map back to sentence_lower by finding the first and last
                # phrase words sequentially (phrase IS contiguous in the original).
                sentence_words_exact = sentence_norm.split()
                # Find how many words precede 'pos' in sentence_norm
                prefix = sentence_norm[:pos]
                prefix_word_count = len(prefix.split()) if prefix.strip() else 0
                if prefix and not prefix.endswith(' '):
                    # pos is mid-word — do plain word search instead
                    prefix_word_count = len(prefix.split()) - 1
                
                orig_pos = 0
                start_char = None
                end_char = None
                for wi, w in enumerate(sentence_words_exact):
                    found_at = sentence_lower.find(w, orig_pos)
                    if found_at == -1:
                        continue
                    if wi == prefix_word_count:
                        start_char = found_at
                    if wi == prefix_word_count + len(phrase_words) - 1:
                        end_char = found_at + len(w)
                        break
                    orig_pos = found_at + len(w)
                if start_char is not None and end_char is not None:
                    return [(start_char, end_char)]
            # Exact fails (non-contiguous ccomp/xcomp clause): find first and last
            # word positions in sentence_lower and span the full range between them.
            # This correctly handles ccomp objects where pruned branches create
            # non-contiguous word sequences (e.g., "that X Y , Z" where tokens
            # between Y and Z appear elsewhere in the sentence).
            sentence_words_for_long = sentence_norm.split()
            # Find position of first phrase word in sentence
            first_phrase_word = phrase_words[0]
            last_phrase_word = phrase_words[-1]
            
            first_sent_idx = None
            for si, sw in enumerate(sentence_words_for_long):
                if sw == first_phrase_word or (len(first_phrase_word) >= 4 and first_phrase_word in sw) or (len(sw) >= 4 and sw in first_phrase_word):
                    first_sent_idx = si
                    break
            
            if first_sent_idx is not None:
                last_sent_idx = None
                # Search for last phrase word from the end of sentence
                for si in range(len(sentence_words_for_long) - 1, first_sent_idx - 1, -1):
                    sw = sentence_words_for_long[si]
                    if sw == last_phrase_word or (len(last_phrase_word) >= 4 and last_phrase_word in sw) or (len(sw) >= 4 and sw in last_phrase_word):
                        last_sent_idx = si
                        break
                
                if last_sent_idx is not None and last_sent_idx >= first_sent_idx:
                    # Map word indices to char positions in sentence_lower
                    orig_pos = 0
                    start_char = None
                    end_char = None
                    for wi, w in enumerate(sentence_words_for_long):
                        found_at = sentence_lower.find(w, orig_pos)
                        if found_at == -1:
                            continue
                        if wi == first_sent_idx:
                            start_char = found_at
                        if wi == last_sent_idx:
                            end_char = found_at + len(w)
                            break
                        orig_pos = found_at + len(w)
                    if start_char is not None and end_char is not None:
                        return [(start_char, end_char)]
            # Fall through to token-level matching as final fallback
        
        # Token-level matching: find each word in sentence
        sentence_words = sentence_norm.split()
        
        # Find all positions of each phrase word in sentence
        word_positions = []  # List of lists: for each phrase word, list of sentence positions
        for phrase_word in phrase_words:
            positions = []
            for sent_idx, sent_word in enumerate(sentence_words):
                # Exact match always OK.
                # Substring match only when the SHORTER of the two words is >= 4 chars,
                # preventing short words ('be','al','in','of','to','a') from matching
                # INSIDE longer words ('cumbersome','optimal','introduced','granularity').
                if phrase_word == sent_word:
                    positions.append(sent_idx)
                elif len(phrase_word) >= 4 and phrase_word in sent_word:
                    positions.append(sent_idx)
                elif len(sent_word) >= 4 and sent_word in phrase_word:
                    positions.append(sent_idx)
            word_positions.append(positions)
        
        # Check if all phrase words found
        if any(not pos_list for pos_list in word_positions):
            # Fallback for compound words split by parsing but contiguous in text (e.g. "LLM2 Vec" vs "LLM2Vec")
            phrase_joined = "".join(phrase_words)
            for idx, sent_word in enumerate(sentence_words):
                if phrase_joined == sent_word or (len(phrase_joined) >= 4 and phrase_joined in sent_word) or (len(sent_word) >= 4 and sent_word in phrase_joined):
                    # Found as a single word or partial match
                    orig_pos = 0
                    start_char = None
                    end_char = None
                    for w_idx, w in enumerate(sentence_words):
                        found_at = sentence_lower.find(w, orig_pos)
                        if found_at == -1:
                            continue
                        if w_idx == idx:
                            start_char = found_at
                            end_char = found_at + len(w)
                            break
                        orig_pos = found_at + len(w)
                    if start_char is not None and end_char is not None:
                        return [(start_char, end_char)]
                        
            return []  # Some phrase word not found
        
        # Try to find sequential matches within max_token_gap
        matches = []
        for first_pos in word_positions[0]:
            # Try to match all subsequent words
            current_pos = first_pos
            matched = True
            
            for i in range(1, len(phrase_words)):
                # Look for next phrase word within max_token_gap
                found_next = False
                for candidate_pos in word_positions[i]:
                    if 0 < candidate_pos - current_pos <= max_token_gap + 1:
                        current_pos = candidate_pos
                        found_next = True
                        break
                
                if not found_next:
                    matched = False
                    break
            
            if matched:
                # Convert word positions to character spans.
                # IMPORTANT: offset_mapping was built from sentence_lower (original lowercased),
                # so char positions must be in sentence_lower coordinates — NOT sentence_norm
                # coordinates. Normalized text removes punctuation, shifting char offsets and
                # causing wrong tokens to be labeled (e.g. "2023 ) ," tagged as SUBJ instead of
                # "updating enormous knowledge in").
                start_word_idx = first_pos
                end_word_idx = current_pos

                # Sequential forward search in original lowercased sentence
                orig_pos = 0
                start_char = None
                end_char = None

                for idx, word in enumerate(sentence_words):
                    found_at = sentence_lower.find(word, orig_pos)
                    if found_at == -1:
                        # Word not findable from current position — skip position advance
                        # (handles punctuation-embedded words like 'al' in 'al.,')
                        continue

                    if idx == start_word_idx:
                        start_char = found_at
                    if idx == end_word_idx:
                        end_char = found_at + len(word)
                        break

                    orig_pos = found_at + len(word)

                if start_char is not None and end_char is not None:
                    matches.append((start_char, end_char))

        return matches
    
    # Normalize sentence once
    sentence_norm = normalize_for_matching(sentence_lower)
    
    # Find phrases in sentence (using raw multi-word triplets)
    subject_spans = []
    predicate_spans = []
    object_spans = []
    
    for triplet in triplets:
        if triplet['subject'] != '?':
            subject_spans.extend(find_phrase_fuzzy(triplet['subject'], sentence_norm))
        if triplet['predicate'] != '?':
            predicate_spans.extend(find_phrase_fuzzy(triplet['predicate'], sentence_norm))
        if triplet['object'] != '?':
            object_spans.extend(find_phrase_fuzzy(triplet['object'], sentence_norm))
    
    # Deduplicate spans (same start/end)
    subject_spans = list(set(subject_spans))
    predicate_spans = list(set(predicate_spans))
    object_spans = list(set(object_spans))
    
    # Collect all entity spans
    entity_spans = []  # List of (char_start, char_end, entity_type_idx)
    # entity_type_idx: 0=SUBJ, 1=PRED, 2=OBJ
    
    for start, end in subject_spans:
        entity_spans.append((start, end, 0))  # 0 = SUBJ
    for start, end in predicate_spans:
        entity_spans.append((start, end, 1))  # 1 = PRED
    for start, end in object_spans:
        entity_spans.append((start, end, 2))  # 2 = OBJ
    
    # Map entity spans to token indices
    for entity_start, entity_end, entity_type in entity_spans:
        first_token_idx = None
        
        for tok_idx, (tok_start, tok_end) in enumerate(offset_mapping):
            # Check if token overlaps with entity span
            if tok_start < entity_end and tok_end > entity_start:
                if first_token_idx is None:
                    # First token covering this entity → B-tag
                    first_token_idx = tok_idx
                    b_col = entity_type * 2  # 0, 2, 4 for SUBJ, PRED, OBJ
                    labels[tok_idx, b_col] = 1
                else:
                    # Continuation token → I-tag
                    i_col = entity_type * 2 + 1  # 1, 3, 5 for SUBJ, PRED, OBJ
                    labels[tok_idx, i_col] = 1
    
    return bert_tokens, labels.tolist()


# ============================================================================
# Triplet Quality Cleanup
# ============================================================================

def _strip_leading_citation(text: str) -> str:
    """Strip leading citation fragments (e.g. '84 ] , ' or '2023 ) , ')."""
    # Number/year followed by bracket/paren + optional comma/semicolon
    text = re.sub(r'^\s*\d+\s*[\)\]]\s*[,;]?\s*', '', text)
    # Opening bracket/paren followed by number + optional closing bracket
    text = re.sub(r'^\s*[\(\[]\s*\d+\s*[\)\]]?\s*[,;]?\s*', '', text)
    return text.strip()


def _strip_trailing_citation(text: str) -> str:
    """Strip trailing citation brackets, e.g. '... training [ 84 ]' or '... training [ 84 ] ,'."""
    # Trailing [ numbers ] or ( numbers ) — standalone numeric citations,
    # optionally followed by whitespace/punctuation like ' ,' or ';' or '.'
    text = re.sub(r'\s*[\[\(]\s*[\d,\s;]+\s*[\]\)][\s.,;:!]*$', '', text)
    return text.strip()


def _has_citation_contamination(text: str) -> bool:
    """Return True if text contains unstrippable citation noise."""
    # Reversed 'et al.' as 'al et'
    if re.search(r'\bal\s+et\b', text, re.IGNORECASE):
        return True
    # Year followed by ) or ] (partial parenthetical)
    if re.search(r'\b\d{4}\s*[\)\]]', text):
        return True
    # Opening bracket/paren that encloses ONLY digits (standalone citation like [ 84 ] or (12))
    # Does NOT match content parentheticals like (72 to 67) which contain words
    if re.search(r'[\(\[]\s*[\d,\s;]+\s*[\)\]]', text):
        return True
    return False


def clean_triplets(triplets: List[Dict]) -> List[Dict]:
    """Clean extracted triplets to improve quality.

    Fixes:
    - Remove trailing punctuation from predicates
    - Strip prepositional prefixes ("in investigate" → "investigate")
    - Strip leading citation fragments from subjects/objects
    - Reject spans with unstrippable citation contamination
    - Filter lone punctuation subjects/objects

    Args:
        triplets: Raw triplets from Stanza extraction

    Returns:
        Cleaned triplets
    """
    cleaned = []

    # Common prepositions that get prepended to predicates
    PREP_PREFIXES = {'in', 'on', 'at', 'to', 'for', 'with', 'by', 'from', 'of'}

    for triplet in triplets:
        # ── Predicate ──────────────────────────────────────────────────────
        pred = triplet['predicate'].strip()

        # Remove trailing punctuation
        while pred and pred[-1] in string.punctuation:
            pred = pred[:-1]

        # Strip prepositional prefix (handles "in investigate" → "investigate")
        pred_words = pred.split()
        if len(pred_words) > 1 and pred_words[0].lower() in PREP_PREFIXES:
            pred = ' '.join(pred_words[1:])

        # Citation contamination in predicate → invalid
        if _has_citation_contamination(pred):
            pred = '?'

        # ── Subject ────────────────────────────────────────────────────────
        subj = triplet['subject'].strip()

        # Strip leading citation fragment ("84 ] , " or "2023 ) , ")
        subj = _strip_leading_citation(subj)

        # Remove embedded complete parenthetical citations
        subj = re.sub(r'\s*\([^)]*\d{4}[^)]*\)', '', subj)

        # Remove leading/trailing punctuation except hyphens
        subj = subj.strip(string.punctuation.replace('-', ''))

        # Reject if too short or still citation-contaminated
        if len(subj) <= 1 or subj in string.punctuation:
            subj = '?'
        elif _has_citation_contamination(subj):
            subj = '?'

        # ── Object ─────────────────────────────────────────────────────────
        obj = triplet['object'].strip()

        # Strip leading citation fragment
        obj = _strip_leading_citation(obj)

        # Strip trailing numeric citation brackets (e.g. '... training [ 84 ]')
        obj = _strip_trailing_citation(obj)

        # Remove embedded complete parenthetical citations
        obj = re.sub(r'\s*\([^)]*\d{4}[^)]*\)', '', obj)

        # Remove leading/trailing punctuation except hyphens
        obj = obj.strip(string.punctuation.replace('-', ''))

        # Reject if too short or citation-contaminated
        if len(obj) <= 1 or obj in string.punctuation:
            obj = '?'
        elif _has_citation_contamination(obj):
            obj = '?'

        # Keep triplet if predicate is valid
        if pred and pred != '?' and pred not in string.punctuation:
            cleaned.append({
                'subject': subj,
                'predicate': pred,
                'object': obj
            })

    return cleaned


# ============================================================================
# Main Pipeline
# ============================================================================

print(f"Loading chunks from {args.input}...")
with open(args.input, 'rb') as f:
    chunks = msgpack.unpackb(f.read(), raw=False)

print(f"Loaded {len(chunks)} chunks")
print(f"Sampling {args.chunks} chunks...")

# Sample chunks
sampled_indices = random.sample(range(len(chunks)), min(args.chunks, len(chunks)))
sampled_chunks = [chunks[i] for i in sampled_indices]

print(f"\nProcessing {len(sampled_chunks)} chunks...")

training_data = []

for chunk in tqdm(sampled_chunks, desc="Extracting triplets"):
    chunk_text = chunk['text']
    chunk_id = chunk['chunk_idx']
    
    # Parse with Stanza to get sentences
    doc = nlp(chunk_text)
    
    # Process each sentence
    for sent in doc.sentences:
        sent_text = sent.text
        
        # Extract raw SPO triplets using dependency parse
        raw_triplets = extract_spo_from_sentence(sent)
        
        if not raw_triplets:
            continue
        
        # Clean triplets (remove trailing punctuation, citations, etc.)
        raw_triplets = clean_triplets(raw_triplets)
        
        if not raw_triplets:
            continue
        
        # Generate BIO labels from cleaned RAW triplets
        # This ensures we match the actual text in the sentence
        tokens, labels = create_bio_labels(sent_text, raw_triplets)
        
        if not tokens:
            continue
        
        # Apply atomic decomposition pipeline for cleaned storage metadata
        atomic_triplets = apply_atomic_pipeline(raw_triplets)
        
        # Convert labels from [n_tokens, 6] to dict format for training
        label_names = ['B-SUBJ', 'I-SUBJ', 'B-PRED', 'I-PRED', 'B-OBJ', 'I-OBJ']
        labels_dict = {
            label_names[i]: [token_labels[i] for token_labels in labels]
            for i in range(6)
        }
        
        # Store training example with both raw and cleaned triplets
        training_data.append({
            'chunk_id': chunk_id,
            'sentence': sent_text,
            'tokens': tokens,
            'labels': labels_dict,  # Dict format: {'B-SUBJ': [...], 'B-PRED': [...], ...}
            'triplets': raw_triplets,  # Original triplets used for labeling
            'triplets_cleaned': atomic_triplets  # Cleaned version for reference
        })

print(f"\n[OK] Extracted {len(training_data)} training examples")

# Save to msgpack
output_data = {
    'training_data': training_data,
    'label_names': ['B-SUBJ', 'I-SUBJ', 'B-PRED', 'I-PRED', 'B-OBJ', 'I-OBJ'],
    'stats': {
        'n_examples': len(training_data),
        'n_chunks': len(sampled_chunks),
        'n_triplets': sum(len(ex['triplets']) for ex in training_data)
    }
}

with open(args.output, 'wb') as f:
    f.write(msgpack.packb(output_data, use_bin_type=True))

print(f"[OK] Saved to {args.output}")

# Print sample
print(f"\nSample training examples:")
for i, example in enumerate(training_data[:5], 1):
    print(f"\n{i}. Sentence: {example['sentence']}")
    print(f"   Triplets: {example['triplets']}")
    print(f"   Tokens: {example['tokens']}")
    print(f"   Labels: {example['labels']}")
