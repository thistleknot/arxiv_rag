"""Debug clean_triplets for row 9."""
import sys
import stanza
import re
import string

sys.path.insert(0, 'extraction')

# Load only what we need from the module without triggering the full pipeline
import importlib.util

# Patch the module to avoid running the main pipeline
import types

# Read the source and exec only the functions we need
src = open('extraction/extract_bio_atomic_clean.py', encoding='utf-8').read()

# Create a fake module namespace
ns = {
    '__name__': 'extract_bio_atomic_clean',
    '__file__': 'extraction/extract_bio_atomic_clean.py',
    '__builtins__': __builtins__,
}
# Execute only up to the main pipeline section (stop before the top-level code)
stop_marker = "\n# ============================================================================\n# Main Pipeline"
src_trunc = src[:src.index(stop_marker)]

exec(compile(src_trunc, 'extract_bio_atomic_clean.py', 'exec'), ns)

extract_spo = ns['extract_spo_from_sentence']
clean_t = ns['clean_triplets']

# Load stanza
print("Loading Stanza...")
nlp = stanza.Pipeline('en', processors='tokenize,mwt,pos,lemma,depparse', verbose=False)

sent_text = "However, perhaps due to the inductive bias of gradient-based training [84], deep learning models tend to diffuse information across the entire representation vector."
doc = nlp(sent_text)

for sent in doc.sentences:
    raw = extract_spo(sent)
    print("Raw triplets:", raw)
    cleaned = clean_t(raw)
    print("Cleaned triplets:", cleaned)
