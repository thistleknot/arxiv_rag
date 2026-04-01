"""Audit dropped IDs: find which have 'Validation Failed' or sentinel utilities."""
import pandas as pd, json, ast

raw   = pd.read_csv(r'papers\post_processed\arxiv_data_with_analysis.csv', encoding_errors='replace')
clean = pd.read_csv(r'papers\post_processed\arxiv_data_with_analysis_cleaned.csv', encoding_errors='replace')

def strip_id(s):
    return str(s).strip('"').strip("'").strip()

raw['_id']   = raw['arxiv_id'].apply(strip_id)
clean['_id'] = clean['arxiv_id'].apply(strip_id)

dropped_ids = set(raw['_id']) - set(clean['_id'])
dropped = raw[raw['_id'].isin(dropped_ids)].copy()

SENTINELS = ['Validation Failed', 'Invalid Utility', 'None', '[]', 'null']

def classify(u):
    if pd.isna(u) or str(u).strip() in ('', '[]', 'null', 'nan'): return 'empty'
    s = str(u)
    for sentinel in SENTINELS:
        if sentinel.lower() in s.lower():
            return 'sentinel'
    # Try parse
    for p in (json.loads, ast.literal_eval):
        try:
            v = p(s)
            if isinstance(v, list) and len(v) == 0: return 'empty'
            return 'valid'
        except: pass
    return 'valid'  # raw string

dropped['class'] = dropped['utility'].apply(classify)
print('Classification of dropped rows:')
print(dropped['class'].value_counts())

# Sentinel / empty = need re-extraction
needs_reextract = dropped[dropped['class'].isin(['sentinel','empty'])][['_id','title','abstract']].drop_duplicates('_id')
valid_dropped   = dropped[dropped['class'] == 'valid'].drop_duplicates('_id')

print(f'\nNeeds re-extraction (sentinel/empty utility): {len(needs_reextract)}')
print(f'Dropped but had valid utility (filtered for other reason): {len(valid_dropped)}')

print('\nSample sentinel utilities:')
for _, r in dropped[dropped['class']=='sentinel'].head(6).iterrows():
    print(f"  {r['_id']} | {repr(str(r['utility'])[:100])}")

# Write re-extraction gap file
needs_reextract.rename(columns={'_id':'arxiv_id'}).to_csv('_reextract_gaps.csv', index=False)
print(f'\nWritten _reextract_gaps.csv ({len(needs_reextract)} IDs)')

# Also check the valid-but-dropped ones -- why were they removed?
print('\nSample valid-utility-but-still-dropped rows:')
for _, r in valid_dropped.head(6).iterrows():
    print(f"  {r['_id']} | is_complete={r.get('is_complete')} | utility={repr(str(r['utility'])[:80])}")
