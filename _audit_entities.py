import json, re

with open('graph/entity_index.json') as f:
    data = json.load(f)

spans = sorted(data['span_to_id'].keys())
FUNC = {'a','an','the','is','are','was','were','be','been','of','in','on','to','at',
        'by','and','or','but','for','with','from','into','your','my','his','her',
        'its','our','their','you','he','she','it','they','we','me','him','them',
        'us','not','no','so','if','as','than','that','this','these','those',
        'because','although','though','when','while','than','what','do',
        'have','has','had','will','would','could','should','may','might'}

BAD_FIRST = FUNC | {'because','although','though','if','when','as','than','with',
                    'for','from','by'}

bad = []
for s in spans:
    words = s.split()
    if len(s) <= 2:
        bad.append((s, 'too short'))
    elif s.strip("'") == '':
        bad.append((s, 'only apostrophes'))
    elif words[0] in BAD_FIRST:
        bad.append((s, f'starts with func word: {words[0]}'))
    elif words[-1] in FUNC and words[-1] not in {'time','place','love','life','man','way'}:
        if all(w in FUNC for w in words):
            bad.append((s, 'all function words'))
    # possessive not lemmatized
    if "'" in s and len(s) > 2:
        bad.append((s, 'contains apostrophe/possessive'))

# dedupe keeping first reason
seen = {}
for s, reason in bad:
    if s not in seen:
        seen[s] = reason

print(f'Total entities: {len(spans)}')
print(f'\nProblematic entities ({len(seen)}):')
for s, r in sorted(seen.items(), key=lambda x: x[1]):
    print(f'  {repr(s):45s}  <- {r}')
