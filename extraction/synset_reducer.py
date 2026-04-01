"""
Synset-based vocabulary reduction for triplet elements.

Pipeline:
  Raw text → Lemmatize → Most common synset → Lemma of synset → Wordpiece tokens
  
Example:
  "was eating" → "eat" → synset(consume.v.01) → "consume" → [13104]
  "fresh apple" → "apple" → synset(apple.n.01) → "apple" → [6207]
"""

import nltk
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from transformers import AutoTokenizer
from collections import defaultdict
import re

# Download required NLTK data
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')


try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


class SynsetReducer:
    """
    Reduces raw text to synset-based lemmatized wordpiece tokens.
    
    This compresses vocabulary by mapping words to their most common
    semantic concepts (synsets) before tokenization.
    """
    
    def __init__(self, tokenizer_name='bert-base-uncased'):
        from nltk.corpus import stopwords
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.cache = {}  # Cache synset reductions
    
    def _get_wordnet_pos(self, treebank_tag):
        """Convert Penn Treebank POS tag to WordNet POS"""
        if treebank_tag.startswith('J'):
            return wn.ADJ
        elif treebank_tag.startswith('V'):
            return wn.VERB
        elif treebank_tag.startswith('N'):
            return wn.NOUN
        elif treebank_tag.startswith('R'):
            return wn.ADV
        else:
            return wn.NOUN  # Default
    
    def _lemmatize_with_pos(self, word, pos_tag):
        """Lemmatize word with POS tag"""
        wn_pos = self._get_wordnet_pos(pos_tag)
        return self.lemmatizer.lemmatize(word.lower(), pos=wn_pos)
    
    def _get_most_common_synset(self, word, pos_tag):
        """
        Get most common synset for word (by usage frequency).
        
        Handles phrases by removing stop words first.
        Example: 'the model' → 'model' → synset
        
        Returns synset or None if no synsets found.
        """
        # Remove stop words from phrase before synset lookup
        tokens = word.lower().split()
        clean_tokens = [t for t in tokens if t not in self.stop_words and len(t) > 1]
        
        if not clean_tokens:
            return None
        
        # Use first non-stop word
        clean_word = clean_tokens[0]
        
        wn_pos = self._get_wordnet_pos(pos_tag)
        synsets = wn.synsets(clean_word, pos=wn_pos)
        
        if not synsets:
            return None
        
        # WordNet orders synsets by frequency/commonality
        # First synset is most common (DETERMINISTIC)
        return synsets[0]
    
    def reduce(self, text):
        """
        Reduce text to synset-based lemma tokens.
        
        Args:
            text: Raw text (can be multi-word phrase)
        
        Returns:
            list of token IDs (from BERT tokenizer)
        """
        if text in self.cache:
            return self.cache[text]
        
        # Clean text
        text = text.strip()
        if not text:
            return []
        
        # Tokenize and POS tag
        words = nltk.word_tokenize(text)
        pos_tags = nltk.pos_tag(words)
        
        reduced_words = []
        
        for word, pos in pos_tags:
            # Skip punctuation and very short words
            if not re.match(r'^[a-zA-Z]+$', word) or len(word) <= 1:
                continue
            
            # Step 1: Lemmatize
            lemma = self._lemmatize_with_pos(word, pos)
            
            # Step 2: Get most common synset
            synset = self._get_most_common_synset(lemma, pos)
            
            if synset:
                # Step 3: Get lemma name from synset (canonical form)
                synset_lemma = synset.lemmas()[0].name()
                # Remove underscores (multi-word synsets)
                synset_lemma = synset_lemma.replace('_', ' ')
                reduced_words.append(synset_lemma)
            else:
                # No synset found, use lemma
                reduced_words.append(lemma)
        
        # Join and convert to wordpiece tokens
        reduced_text = ' '.join(reduced_words)
        token_ids = self.tokenizer.encode(
            reduced_text,
            add_special_tokens=False,
            max_length=32,  # Reasonable max for single entity
            truncation=True
        )
        
        self.cache[text] = token_ids
        return token_ids
    
    def reduce_triplet(self, subject, predicate, obj=None):
        """
        Reduce full triplet to token IDs.
        
        Returns:
            dict with 'subject', 'predicate', 'object' (object may be None)
        """
        result = {
            'subject': self.reduce(subject) if subject else [],
            'predicate': self.reduce(predicate) if predicate else [],
            'object': self.reduce(obj) if obj else []
        }
        return result
    
    def tokens_to_text(self, token_ids):
        """Convert token IDs back to text (for debugging)"""
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)


# Example usage
if __name__ == "__main__":
    reducer = SynsetReducer()
    
    print("="*70)
    print("SYNSET REDUCTION EXAMPLES")
    print("="*70)
    
    examples = [
        "was eating",
        "fresh apple",
        "neural network",
        "transformer model",
        "self-attention mechanism",
        "processes sequences",
        "John Mayer",
        "the red car"
    ]
    
    for text in examples:
        tokens = reducer.reduce(text)
        reduced = reducer.tokens_to_text(tokens)
        print(f"\nOriginal: {text}")
        print(f"Reduced:  {reduced}")
        print(f"Tokens:   {tokens[:10]}{'...' if len(tokens) > 10 else ''}")
    
    print("\n" + "="*70)
    print("TRIPLET REDUCTION EXAMPLE")
    print("="*70)
    
    triplets = [
        ("transformer model", "uses", "self-attention mechanism"),
        ("the model", "was processing", "input sequences"),
        ("John", "ate", None),  # Partial triplet
    ]
    
    for s, p, o in triplets:
        reduced = reducer.reduce_triplet(s, p, o)
        print(f"\nOriginal: P({s}, {o if o else ''})")
        print(f"Subject:   {reducer.tokens_to_text(reduced['subject'])} {reduced['subject']}")
        print(f"Predicate: {reducer.tokens_to_text(reduced['predicate'])} {reduced['predicate']}")
        if reduced['object']:
            print(f"Object:    {reducer.tokens_to_text(reduced['object'])} {reduced['object']}")
