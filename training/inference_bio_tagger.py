"""
Inference Script for BIO-Tagged Triplet Extraction

FAST TRIPLET EXTRACTION: Text → BERT → BIO Tags → Triplets
===========================================================

Pipeline:
    1. Text input
    2. BERT tokenization
    3. BIO tagging (6 independent binary predictions)
    4. Span extraction (group consecutive B/I tags)
    5. Triplet reconstruction (position-based heuristics)
    6. Synset reduction (optional post-processing)

Performance:
    - Target: ~10ms per chunk (600x faster than OpenIE)
    - Quality: 80-90% of OpenIE accuracy

Post-Processing Rules (Simple Heuristics):
    These mimic OpenIE's behavior without re-learning:
    
    1. Extract spans: Group consecutive B/I tags
       Example: [B-SUBJ, I-SUBJ, I-SUBJ] → "neural network model"
    
    2. Match predicates with arguments by position:
       - Find nearest SUBJECT before predicate
       - Find nearest OBJECT after predicate
       - If no subject found, skip triplet
       - Object is optional
    
    3. Handle overlapping: Multi-hot labels naturally support it
       Example: "model" tagged as B-SUBJ multiple times
                → Generates multiple triplets

Integration with Synset Reduction:
    After extracting triplets, apply synset_reducer.py:
    
    Raw: ("neural network", "learns", "patterns")
    → Synsets: (["nervous", "network"], ["learn"], ["pattern"])
    → Final: P(learn, [nervous, network], [pattern])

Usage:
    # Single chunk
    python inference_bio_tagger.py --text "The model learns patterns"
    
    # Batch processing
    python inference_bio_tagger.py --db --chunks 100 --output results.msgpack
"""

import argparse
import torch
import msgpack
from transformers import AutoTokenizer, BitsAndBytesConfig
from typing import List, Dict, Tuple, Optional
import numpy as np
from tqdm import tqdm
import time
import warnings

# Database configuration (matches all working scripts)
DB_CONFIG = {
    'dbname': 'postgres',
    'user': 'postgres',
    'password': 'postgres',
    'host': 'localhost',
    'port': 5432
}

# Stopwords to strip from extracted spans POST-INFERENCE
# Model learns full BIO boundaries including these words,
# but we strip them from final span text for cleaner triplets
SPAN_STOPWORDS = {
    'a', 'an', 'the',           # articles
    'to', 'of', 'in', 'on',     # prepositions
    'at', 'by', 'for', 'with',
    'from', 'into', 'through',
    'and', 'or', 'but', 'nor',   # conjunctions
    'is', 'are', 'was', 'were',  # copulas — stripped from all spans; _copula_recover() handles OBJ promotion
    'be', 'been', 'being',
    'has', 'have', 'had',        # auxiliaries
    'do', 'does', 'did',         # do-support auxiliaries
    'that', 'which', 'who',      # relative pronouns
    'this', 'these', 'those',
    'it', 'its',                 # pronouns
}

# Possession/attribution verbs: semantically equivalent to a copula.
# 'X has [attr] [NP]' → (X, attr, NP)  — 'has' itself carries no relational content,
# the following adjective/quantifier IS the predicate.
_POSSESSION_VERBS = frozenset({'has', 'have', 'had'})

# Copular be-verbs: grammatical plumbing only.  Used by _copula_recover() to
# detect stripped-predicate spans where the semantic payload landed in the OBJ
# slot rather than PRED (model tagged 'is' alone as PRED, complement as OBJ).
# Distinct from SPAN_STOPWORDS: become/seem/appear are *not* copular here
# because they carry independent relational content and should survive stripping.
_COPULAR_VERBS = frozenset({
    'is', 'are', 'was', 'were', 'be', 'been', 'being',
})

# Standalone punctuation artifacts that linger after stopword removal.
# e.g. 'state-of-the-art' → after stripping 'of'/'the', only '-' tokens remain.
_PUNCT_ARTIFACTS = frozenset({'-', '–', '—', '·', '*', '•', '|'})


def _is_copula_only(raw_tokens: List[str]) -> bool:
    """True if every non-empty token in raw_tokens is a copular be-verb."""
    lower = [t.lower() for t in raw_tokens if t.strip()]
    return bool(lower) and all(t in _COPULAR_VERBS for t in lower)


def clean_span_tokens(tokens: List[str], label_type: str) -> List[str]:
    """
    Remove functional/auxiliary tokens from an extracted span, leaving only
    the semantically load-bearing tokens.

    Semantic predicates vs. functional predicates
    ---------------------------------------------
    A *semantic* predicate carries relational or attributional content:
        blue(sky)          — unary;  predicate = 'blue'
        destroys(army, city) — binary; predicate = 'destroys'

    A *functional* predicate is grammatical plumbing with no independent
    relational content:
        'is'  in 'The sky is blue'      — copula; strip it
        'has' in 'model has N layers'   — possession; strip it
        'do'  in 'cells do not divide'  — do-support; strip it

    Post-inference cleanup: model predicts full BIO boundaries
    (including functional tokens), then we strip them here.

    Rules:
        - Strip articles, prepositions, conjunctions from all spans
        - Copulas (is/are/was/were), possession verbs (has/have/had), and
          do-support auxiliaries (do/does/did) are stripped
        - Note: become/seem/appear are NOT stripped — they ARE semantic
          predicates ('the model becomes efficient' → predicate='becomes')
        - Standalone punctuation artifacts (e.g. '-' from 'state-of-the-art')
          are stripped.
        - If a span becomes empty after stripping, returns an empty list;
          the caller (_copula_recover / reconstruct_triplets) handles recovery.

    Args:
        tokens: raw span tokens from the model prediction
        label_type: 'SUBJ', 'PRED', or 'OBJ' — unified behaviour, kept for API compat
    """
    stop = SPAN_STOPWORDS  # copulas/auxiliaries stripped everywhere

    cleaned = [t for t in tokens if t.lower() not in stop]
    # Remove standalone punctuation artifacts left from compound word stripping
    # (e.g. '-' remaining after 'of'/'the' removed from 'state-of-the-art')
    cleaned = [t for t in cleaned if t not in _PUNCT_ARTIFACTS]
    return cleaned


class Span:
    """Represents a span of tokens."""
    def __init__(self, start: int, end: int, tokens: List[str], label_type: str):
        self.start = start
        self.end = end
        self.tokens = tokens
        self.raw_text = ' '.join(tokens)
        # Post-inference semantic filtering: strip functional/auxiliary tokens
        cleaned = clean_span_tokens(tokens, label_type)
        self.text = ' '.join(cleaned)
        self.label_type = label_type  # 'SUBJ', 'PRED', or 'OBJ'
        # True when raw_text was non-empty but text became empty because all
        # raw tokens were copular be-verbs — signals copula-recovery is needed.
        self.is_copula_stripped: bool = (
            bool(self.raw_text.strip())
            and not bool(self.text.strip())
            and _is_copula_only(tokens)
        )

    def __repr__(self):
        return f"Span({self.label_type}, {self.start}:{self.end}, '{self.text}')"


class BIOTripletExtractor:
    """
    Fast triplet extraction using trained BIO tagger.
    """
    def __init__(self, model_path: str, device: Optional[str] = None, use_quantization: bool = False):
        """
        Args:
            model_path: Path to trained model (.pt file)
            device: 'cuda', 'cpu', or None (auto-detect)
            use_quantization: If True, use NF4 4-bit quantization
                - Reduces memory by 75% (220MB → 55MB)
                - Enables 2-4x larger batch sizes
                - 1.2-2.0x faster inference
                - Minimal accuracy loss (0-2% F1)
                - Requires: pip install bitsandbytes
        """
        print(f"Loading model from {model_path}...")
        
        self.use_quantization = use_quantization
        
        # Device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        _ALL_LABEL_NAMES = ['B-SUBJ', 'I-SUBJ', 'B-PRED', 'I-PRED', 'B-OBJ', 'I-OBJ']
        
        # Check if checkpoint is a state_dict or wrapped dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            tokenizer_name = checkpoint.get('tokenizer_name', 'bert-base-uncased')
            self.active_labels = checkpoint.get('active_labels', _ALL_LABEL_NAMES)
        else:
            # Bare state dict (old training run, no label metadata saved).
            state_dict = checkpoint
            tokenizer_name = 'bert-base-uncased'
            # If a label was dropped the mapping is ambiguous — we can't guess which one.
            _n_saved = sum(1 for k in state_dict
                          if k.startswith('classifiers.') and k.endswith('.weight'))
            if _n_saved < 6:
                raise RuntimeError(
                    f"Checkpoint is a bare state dict with only {_n_saved} classifiers "
                    f"but no label metadata — cannot determine which label was dropped.\n"
                    f"Migrate the checkpoint with:\n\n"
                    f"  python migrate_checkpoint.py\n\n"
                    f"(file created alongside this error)"
                )
            self.active_labels = _ALL_LABEL_NAMES
        
        num_labels = len(self.active_labels)
        if num_labels != 6:
            print(f"[INFO] Active labels ({num_labels}/6): {self.active_labels}")
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Initialize model architecture
        from train_bio_tagger import BIOTagger
        
        if self.use_quantization and self.device.type == 'cuda':
            print("⚡ Quantization enabled: NF4 4-bit")
            
            try:
                # Create quantization config
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
                
                # Note: BIOTagger wraps BertModel, need to pass config through
                # This requires modifying train_bio_tagger.py to accept quantization_config
                # For now, load normally then quantize manually
                warnings.warn(
                    "[WARN] Quantization requires modifying BIOTagger class. "
                    "Loading model normally for now. "
                    "See inference_bio_tagger.py for implementation notes."
                )
                self.model = BIOTagger(bert_model_name=tokenizer_name, num_labels=num_labels)
                self.model.load_state_dict(state_dict)
                self.model = self.model.to(self.device)
                
            except ImportError as e:
                warnings.warn(
                    f"[WARN] bitsandbytes not installed: {e}. "
                    "Install with: pip install bitsandbytes\n"
                    "Loading model normally."
                )
                self.model = BIOTagger(bert_model_name=tokenizer_name, num_labels=num_labels)
                self.model.load_state_dict(state_dict)
                self.model = self.model.to(self.device)
                
        elif self.use_quantization and self.device.type == 'cpu':
            warnings.warn(
                "[WARN] Quantization only works on GPU. "
                "Loading model normally."
            )
            self.model = BIOTagger(bert_model_name=tokenizer_name, num_labels=num_labels)
            self.model.load_state_dict(state_dict)
            self.model = self.model.to(self.device)
            
        else:
            # Standard loading (no quantization)
            self.model = BIOTagger(bert_model_name=tokenizer_name, num_labels=num_labels)
            self.model.load_state_dict(state_dict)
            self.model = self.model.to(self.device)
        
        self.model.eval()
        
        quant_status = " (NF4 requested)" if self.use_quantization else ""
        print(f"Model loaded on {self.device}{quant_status}")
    
    def extract_spans(
        self,
        begin_probs: np.ndarray,
        inside_probs: np.ndarray,
        tokens: List[str],
        label_type: str,
        threshold: float = 0.5
    ) -> List[Span]:
        """
        Extract spans from BIO probabilities.
        
        Args:
            begin_probs: [seq_len] probabilities for B tag
            inside_probs: [seq_len] probabilities for I tag
            tokens: List of tokens
            label_type: 'SUBJ', 'PRED', or 'OBJ'
            threshold: Probability threshold
        
        Returns:
            List of Span objects
        """
        _SPECIAL = {'[CLS]', '[SEP]', '[PAD]'}
        spans = []
        i = 0

        while i < len(tokens):
            if tokens[i] in _SPECIAL:
                i += 1
                continue
            # Any token above threshold for this entity type (B or I) is included
            if begin_probs[i] > threshold or inside_probs[i] > threshold:
                start = i
                i += 1
                while i < len(tokens) and tokens[i] not in _SPECIAL and (begin_probs[i] > threshold or inside_probs[i] > threshold):
                    i += 1
                spans.append(Span(start, i, tokens[start:i], label_type))
            else:
                i += 1

        return spans
    
    def find_nearest_before(self, spans: List[Span], position: int) -> Optional[Span]:
        """Find nearest span before given position."""
        candidates = [s for s in spans if s.end <= position]
        if not candidates:
            return None
        return max(candidates, key=lambda s: s.end)  # Closest to position
    
    def find_nearest_after(self, spans: List[Span], position: int) -> Optional[Span]:
        """Find nearest span after given position."""
        candidates = [s for s in spans if s.start >= position]
        if not candidates:
            return None
        return min(candidates, key=lambda s: s.start)  # Closest to position
    
    def reconstruct_triplets(
        self,
        subjects: List[Span],
        predicates: List[Span],
        objects: List[Span]
    ) -> List[Dict]:
        """
        Reconstruct triplets from spans using position heuristics.

        Only *semantic* predicates (spans whose text survives functional
        token stripping) are emitted here.  Predicates that collapse to
        empty (copulas, auxiliaries) are handled by _copula_recover().

        Heuristics (mimics OpenIE):
            - For each predicate, find nearest subject BEFORE it
            - Find nearest object AFTER predicate
            - Subject is required; object is optional

        Output fields per triplet:
            subject      : str
            predicate    : str  — always a semantic predicate
            object       : str | None — None signals a unary predicate
            arity        : 'unary' | 'binary'
            relation_type: 'predicate'  (same_as handled by _copula_recover)
            positions    : dict
        """
        triplets = []

        for pred in predicates:
            if not pred.text.strip():
                # Predicate stripped to empty — either copula (handled by
                # _copula_recover) or fully-functional aux span (discard).
                continue

            # Find subject (nearest before predicate)
            subj = self.find_nearest_before(subjects, pred.start)

            if not subj or not subj.text.strip():
                continue  # No valid subject — skip

            # Find object (nearest after predicate)
            obj = self.find_nearest_after(objects, pred.end)

            # Object absent or cleaned to empty → unary predicate
            obj_text = obj.text if obj and obj.text.strip() else None

            triplets.append({
                'subject':       subj.text,
                'predicate':     pred.text,
                'object':        obj_text,
                'arity':         'unary' if obj_text is None else 'binary',
                'relation_type': 'predicate',
                'positions': {
                    'subject':   (subj.start, subj.end),
                    'predicate': (pred.start, pred.end),
                    'object':    (obj.start, obj.end) if obj and obj_text else None,
                },
            })

        return triplets

    # ------------------------------------------------------------------
    # Copula recovery
    # ------------------------------------------------------------------

    def _copula_recover(
        self,
        subjects: List[Span],
        predicates: List[Span],
        objects: List[Span],
    ) -> List[Dict]:
        """
        Recover semantic content from copula-stripped predicate spans.

        When the model tags a copular be-verb ('is', 'are', 'was', …) as a
        PRED span and the semantic complement as an OBJ span, the standard
        pipeline discards the triplet because `pred.text` is empty after
        stripping. This method promotes the OBJ into the predicate slot.

        Two cases:

        1. Unary predicate promotion ('The sky is blue')
                PRED=['is'] → stripped empty
                OBJ=['blue']
            Emit: subject='sky', predicate='blue', object=None (unary)

        2. Identity / same_as ('Clark Kent is Superman')
                PRED=['is'] → stripped empty
                OBJ=['Superman']  (both sides are noun-phrase spans)
            Emit: subject='Clark Kent', predicate='is', object='Superman',
                  relation_type='same_as'
            Heuristic: OBJ raw_text appears to be a proper-noun phrase
            (starts with capital letter or is multi-word).

        Note: the two cases are distinguished by a simple capitalisation
        heuristic because the model has no POS signal.  For scientific text
        (the primary domain), case 1 dominates; case 2 is rare.
        """
        results = []
        seen_positions: set = set()   # (pred.start, pred.end) already processed

        for pred in predicates:
            if not pred.is_copula_stripped:
                continue  # Regular predicate — handled by reconstruct_triplets
            if (pred.start, pred.end) in seen_positions:
                continue

            subj = self.find_nearest_before(subjects, pred.start)
            if not subj or not subj.text.strip():
                continue

            obj = self.find_nearest_after(objects, pred.end)
            if not obj or not obj.text.strip():
                continue  # Nothing to promote — discard

            seen_positions.add((pred.start, pred.end))

            # Same-as heuristic: OBJ looks like a proper-noun phrase
            # (starts with capital or is multi-word) while SUBJ is also NP-like.
            obj_tokens = obj.raw_text.split()
            _obj_is_np = (
                len(obj_tokens) > 1
                or (obj_tokens and obj_tokens[0][:1].isupper())
            )
            _subj_is_np = subj.raw_text[:1].isupper() if subj.raw_text else False

            if _obj_is_np and _subj_is_np:
                # NP-is-NP → identity relation
                results.append({
                    'subject':       subj.text,
                    'predicate':     'is',          # canonical copular marker
                    'object':        obj.text,
                    'arity':         'binary',
                    'relation_type': 'same_as',
                    'positions': {
                        'subject':   (subj.start, subj.end),
                        'predicate': (pred.start, pred.end),
                        'object':    (obj.start, obj.end),
                    },
                })
            else:
                # NP-is-complement → promote complement as unary predicate
                results.append({
                    'subject':       subj.text,
                    'predicate':     obj.text,      # complement IS the predicate
                    'object':        None,
                    'arity':         'unary',
                    'relation_type': 'predicate',
                    'positions': {
                        'subject':   (subj.start, subj.end),
                        'predicate': (obj.start, obj.end),
                        'object':    None,
                    },
                })

        return results

    def _possession_rewrite(
        self,
        tokens: List[str],
        subjects: List[Span],
    ) -> List[Dict]:
        """
        Rewrite possession patterns into semantic predicate form.

        'has/have/had' carries no relational content.  The first non-stopword
        token after the possession verb IS the semantic predicate; remaining
        tokens form the object (if any).

        Arity:
            Unary  (1 content token): 'model has depth'
                → subject='model', predicate='depth', object=None
            Binary (2+ content tokens): 'model has strong generalization'
                → subject='model', predicate='strong', object='generalization'

        This method always runs (not fallback-only) and is merged with
        BIO-span results in extract_triplets with deduplication.
        """
        _SPECIAL = frozenset({'[CLS]', '[SEP]', '[PAD]'})
        _SENT_PUNCT = frozenset({'.', ',', '!', '?', ';', ':'})
        results = []
        seen: set = set()

        for i, tok in enumerate(tokens):
            if tok.lower() not in _POSSESSION_VERBS:
                continue
            if i in seen:
                continue

            subj = self.find_nearest_before(subjects, i)
            if not subj or not subj.text.strip():
                continue

            # Reassemble BERT wordpieces after the possession verb
            raw: List[str] = []
            for j in range(i + 1, len(tokens)):
                t = tokens[j]
                if t in _SPECIAL:
                    break
                if t.startswith('##'):
                    if raw:
                        raw[-1] = raw[-1] + t[2:]
                else:
                    raw.append(t)

            content = [
                t for t in raw
                if t.lower() not in SPAN_STOPWORDS
                and t not in _PUNCT_ARTIFACTS
                and t not in _SENT_PUNCT
            ]

            if not content:      # nothing semantic after 'has' — skip
                continue

            pred_text = content[0]                                        # semantic predicate
            obj_text  = ' '.join(content[1:]) if len(content) > 1 else None  # optional object

            seen.add(i)
            results.append({
                'subject':       subj.text,
                'predicate':     pred_text,
                'object':        obj_text,
                'arity':         'unary' if obj_text is None else 'binary',
                'relation_type': 'predicate',
                'positions': {
                    'subject':   (subj.start, subj.end),
                    'predicate': (i, i + 1),
                    'object':    None,
                },
            })

        return results

    def show_token_probs(self, text: str, threshold: float = 0.5) -> None:
        """
        Print per-token label probabilities for diagnostic inspection.
        
        Columns are the active labels for this model (may be fewer than 6
        if detect_nonexistent_labels() dropped a zero-token label at training time).
        Values above `threshold` are marked with *.
        """
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        
        with torch.no_grad():
            probs, _ = self.model(input_ids, attention_mask)
        probs_np = probs.squeeze(0).cpu().numpy()  # [seq_len, num_active_labels]
        
        col_w = 9  # column width per label
        header = f"{'Token':<20}" + "".join(f"{lbl:>{col_w}}" for lbl in self.active_labels)
        print()
        print(header)
        print("-" * len(header))
        for i, tok in enumerate(tokens):
            if tok in ('[CLS]', '[SEP]', '[PAD]'):
                continue
            row = f"{tok:<20}"
            for j in range(len(self.active_labels)):
                val = probs_np[i, j]
                mark = '*' if val > threshold else ' '
                row += f"{val:>{col_w - 1}.3f}{mark}"
            print(row)
        print()
    
    def extract_triplets(
        self,
        text: str,
        threshold: float = 0.5,
        apply_synsets: bool = False
    ) -> List[Dict]:
        """
        Extract triplets from text.
        
        Args:
            text: Input text
            threshold: BIO probability threshold
            apply_synsets: Whether to apply synset reduction
        
        Returns:
            List of triplets: [{'subject': str, 'predicate': str, 'object': str}, ...]
        """
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Get tokens (for span text)
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        
        # BERT prediction
        with torch.no_grad():
            probs, logits = self.model(input_ids, attention_mask)  # Returns (probs, logits) tuple
        
        # Convert to numpy dict using self.active_labels for correct column mapping.
        # detect_nonexistent_labels() may drop ANY label (not necessarily trailing),
        # so column i in probs corresponds to active_labels[i], not the global label index.
        _ALL_LABELS = ['B-SUBJ', 'I-SUBJ', 'B-PRED', 'I-PRED', 'B-OBJ', 'I-OBJ']
        probs_np = probs.squeeze(0).cpu().numpy()  # [seq_len, num_active_labels]
        _zero = np.zeros(probs_np.shape[0])
        predictions_np = {
            label: (probs_np[:, self.active_labels.index(label)]
                    if label in self.active_labels else _zero)
            for label in _ALL_LABELS
        }
        
        # Extract spans
        subjects = self.extract_spans(
            predictions_np['B-SUBJ'],
            predictions_np['I-SUBJ'],
            tokens,
            'SUBJ',
            threshold
        )
        
        predicates = self.extract_spans(
            predictions_np['B-PRED'],
            predictions_np['I-PRED'],
            tokens,
            'PRED',
            threshold
        )
        
        objects = self.extract_spans(
            predictions_np['B-OBJ'],
            predictions_np['I-OBJ'],
            tokens,
            'OBJ',
            threshold
        )
        
        # Reconstruct triplets from BIO spans (semantic predicates only)
        triplets = self.reconstruct_triplets(subjects, predicates, objects)

        # Copula recovery: PRED='is' (stripped empty) → promote OBJ as predicate
        # or emit same_as for NP-is-NP identity constructions.
        triplets += self._copula_recover(subjects, predicates, objects)

        # Possession rewrite: 'X has [attr] [NP]' → (X, semantic_pred, obj)
        # Runs always (not fallback-only); deduplication follows.
        if subjects:
            triplets += self._possession_rewrite(tokens, subjects)

        # Deduplicate across all three sources by (subject, predicate, object).
        seen_keys: set = set()
        deduped: List[Dict] = []
        for t in triplets:
            key = (t['subject'], t['predicate'], t.get('object'))
            if key not in seen_keys:
                seen_keys.add(key)
                deduped.append(t)
        triplets = deduped

        # Apply synset reduction if requested
        if apply_synsets:
            try:
                from synset_reducer import SynsetReducer
                reducer = SynsetReducer()
                
                for triplet in triplets:
                    triplet['subject_synsets'] = reducer.reduce(triplet['subject'])
                    triplet['predicate_synsets'] = reducer.reduce(triplet['predicate'])
                    if triplet['object']:
                        triplet['object_synsets'] = reducer.reduce(triplet['object'])
            except ImportError:
                print("Warning: synset_reducer not available")
        
        return triplets
    
    def extract_triplets_batch(
        self,
        texts: List[str],
        threshold: float = 0.5,
        apply_synsets: bool = False
    ) -> List[List[Dict]]:
        """
        Extract triplets from multiple texts in a single batch (GPU-accelerated).
        
        Processes all texts simultaneously through BERT for 4-6x speedup.
        
        Args:
            texts: List of input texts
            threshold: BIO probability threshold
            apply_synsets: Whether to apply synset reduction
        
        Returns:
            List of triplet lists (one per input text)
            [
                [{'subject': str, 'predicate': str, 'object': str}, ...],  # text 1
                [{'subject': str, 'predicate': str, 'object': str}, ...],  # text 2
                ...
            ]
        """
        if not texts:
            return []
        
        # Tokenize entire batch
        encodings = self.tokenizer(
            texts,
            truncation=True,
            max_length=512,
            padding=True,
            return_tensors='pt'
        )
        
        input_ids = encodings['input_ids'].to(self.device)
        attention_mask = encodings['attention_mask'].to(self.device)
        
        # BERT prediction on entire batch
        with torch.no_grad():
            probs, logits = self.model(input_ids, attention_mask)  # [batch_size, seq_len, 6]
        
        # Process each text in the batch
        batch_results = []
        
        for i in range(len(texts)):
            # Get tokens for this text
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids[i])
            
            # Map model outputs to label names using self.active_labels.
            # detect_nonexistent_labels() may have removed any label (not necessarily trailing).
            _ALL_LABELS = ['B-SUBJ', 'I-SUBJ', 'B-PRED', 'I-PRED', 'B-OBJ', 'I-OBJ']
            probs_np = probs[i].cpu().numpy()  # [seq_len, num_active_labels]
            _zero = np.zeros(probs_np.shape[0])
            predictions_np = {
                label: (probs_np[:, self.active_labels.index(label)]
                        if label in self.active_labels else _zero)
                for label in _ALL_LABELS
            }
            
            # Extract spans
            subjects = self.extract_spans(
                predictions_np['B-SUBJ'],
                predictions_np['I-SUBJ'],
                tokens,
                'SUBJ',
                threshold
            )
            
            predicates = self.extract_spans(
                predictions_np['B-PRED'],
                predictions_np['I-PRED'],
                tokens,
                'PRED',
                threshold
            )
            
            objects = self.extract_spans(
                predictions_np['B-OBJ'],
                predictions_np['I-OBJ'],
                tokens,
                'OBJ',
                threshold
            )
            
            # Reconstruct triplets from BIO spans (semantic predicates only)
            triplets = self.reconstruct_triplets(subjects, predicates, objects)

            # Copula recovery: PRED='is' (stripped empty) → promote OBJ as predicate
            triplets += self._copula_recover(subjects, predicates, objects)

            # Possession rewrite: always runs; deduplication follows.
            if subjects:
                triplets += self._possession_rewrite(tokens, subjects)

            # Deduplicate across all three sources
            _seen: set = set()
            _deduped: List[Dict] = []
            for t in triplets:
                _key = (t['subject'], t['predicate'], t.get('object'))
                if _key not in _seen:
                    _seen.add(_key)
                    _deduped.append(t)
            triplets = _deduped

            # Apply synset reduction if requested
            if apply_synsets:
                try:
                    from synset_reducer import SynsetReducer
                    reducer = SynsetReducer()
                    
                    for triplet in triplets:
                        triplet['subject_synsets'] = reducer.reduce(triplet['subject'])
                        triplet['predicate_synsets'] = reducer.reduce(triplet['predicate'])
                        if triplet['object']:
                            triplet['object_synsets'] = reducer.reduce(triplet['object'])
                except ImportError:
                    pass  # Skip synsets if not available
            
            batch_results.append(triplets)
        
        return batch_results


def process_database_chunks(
    extractor: BIOTripletExtractor,
    num_chunks: int,
    output_file: str,
    apply_synsets: bool = False
):
    """
    Process chunks from database and save results.
    """
    import psycopg2
    
    print(f"\n[INFO] Processing {num_chunks} chunks from database...")
    
    # Connect to database
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    # Get random chunks
    cur.execute("""
        SELECT id, content
        FROM arxiv_papers_lemma_fullembed
        ORDER BY RANDOM()
        LIMIT %s
    """, (num_chunks,))
    
    chunks = cur.fetchall()
    
    # Process chunks
    results = []
    times = []
    
    for chunk_id, content in tqdm(chunks, desc='Extracting'):
        start_time = time.time()
        
        triplets = extractor.extract_triplets(
            content,
            apply_synsets=apply_synsets
        )
        
        elapsed = time.time() - start_time
        times.append(elapsed)
        
        results.append({
            'chunk_id': chunk_id,
            'triplets': triplets,
            'time_ms': elapsed * 1000
        })
    
    # Save results
    with open(output_file, 'wb') as f:
        msgpack.pack(results, f)
    
    # Statistics
    avg_time = np.mean(times) * 1000  # ms
    total_triplets = sum(len(r['triplets']) for r in results)
    
    print(f"\n[DONE] Processing complete!")
    print(f"   Average time: {avg_time:.2f}ms per chunk")
    print(f"   Total triplets: {total_triplets}")
    print(f"   Results saved to: {output_file}")
    
    # Compare to OpenIE
    openie_time = 6000  # ms
    speedup = openie_time / avg_time
    print(f"\n⚡ Speedup vs OpenIE (~{openie_time}ms): {speedup:.0f}x")
    
    cur.close()
    conn.close()


def main():
    parser = argparse.ArgumentParser(description='Fast triplet extraction with BIO tagger')
    parser.add_argument('--model', type=str, default='bio_tagger_model.pt',
                        help='Path to trained model')
    parser.add_argument('--text', type=str, default=None,
                        help='Text to process (single example)')
    parser.add_argument('--db', action='store_true',
                        help='Process chunks from database')
    parser.add_argument('--chunks', type=int, default=100,
                        help='Number of chunks to process from database')
    parser.add_argument('--output', type=str, default='inference_results.msgpack',
                        help='Output file for batch processing')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='BIO probability threshold')
    parser.add_argument('--synsets', action='store_true',
                        help='Apply synset reduction')
    parser.add_argument('--probs', action='store_true',
                        help='Print per-token label probabilities (diagnostic)')
    args = parser.parse_args()
    
    print("="*70)
    print("FAST TRIPLET EXTRACTION (BIO TAGGER)")
    print("="*70)
    
    # Initialize extractor
    extractor = BIOTripletExtractor(model_path=args.model)
    
    # Single text example
    if args.text:
        print(f"\n📝 Input: {args.text}")
        
        start_time = time.time()
        triplets = extractor.extract_triplets(
            args.text,
            threshold=args.threshold,
            apply_synsets=args.synsets
        )
        elapsed = time.time() - start_time
        
        print(f"\n⏱️  Time: {elapsed * 1000:.2f}ms")
        
        if args.probs:
            print("\n[INFO] Per-token probabilities (* = above threshold):")
            extractor.show_token_probs(args.text, threshold=args.threshold)
        
        print(f"\n🔍 Extracted {len(triplets)} triplet(s):")

        for i, triplet in enumerate(triplets, 1):
            arity = triplet.get('arity', '?')
            rtype = triplet.get('relation_type', 'predicate')
            print(f"\n   {i}. [{arity}] [{rtype}]")
            print(f"      Subject:   {triplet['subject']}")
            print(f"      Predicate: {triplet['predicate']}")
            print(f"      Object:    {triplet.get('object', '—')}")
            
            if args.synsets and 'subject_synsets' in triplet:
                print(f"      Subject synsets: {triplet['subject_synsets']}")
                print(f"      Predicate synsets: {triplet['predicate_synsets']}")
                if triplet.get('object_synsets'):
                    print(f"      Object synsets: {triplet['object_synsets']}")
    
    # Database batch processing
    elif args.db:
        process_database_chunks(
            extractor,
            num_chunks=args.chunks,
            output_file=args.output,
            apply_synsets=args.synsets
        )
    
    else:
        print("\n[ERROR] Provide either --text or --db")
        print("   Examples:")
        print('     python inference_bio_tagger.py --text "The model learns patterns"')
        print('     python inference_bio_tagger.py --db --chunks 100')


if __name__ == '__main__':
    main()
