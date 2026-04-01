import pandas as pd, json, ast

raw   = pd.read_csv(r'papers\post_processed\arxiv_data_with_analysis.csv', encoding_errors='replace')
clean = pd.read_csv(r'papers\post_processed\arxiv_data_with_analysis_cleaned.csv', encoding_errors='replace')

def strip_id(s):
    return str(s).strip('"').strip().strip("'").strip()

raw['_id']   = raw['arxiv_id'].apply(strip_id)
clean['_id'] = clean['arxiv_id'].apply(strip_id)

dropped = raw[~raw['_id'].isin(set(clean['_id']))].copy()
print('Dropped rows:', len(dropped))
print('Have non-null abstract:', dropped['abstract'].notna().sum())
print('Have non-null title:', dropped['title'].notna().sum())

def why_bad(u):
    if pd.isna(u): return 'null'
    s = str(u).strip()
    if s in ('', '[]', 'null', 'nan'): return 'empty'
    for p in (json.loads, ast.literal_eval):
        try:
            r = p(s)
            if isinstance(r, list) and len(r) > 0: return 'parseable'
            return 'empty_list'
        except:
            pass
    return 'bad_json'

dropped['why'] = dropped['utility'].apply(why_bad)
print('\nUtility parse status on dropped:')
print(dropped['why'].value_counts())

print('\nSample bad_json utilities (truncated?):')
for _, r in dropped[dropped['why'] == 'bad_json'].head(6).iterrows():
    print(f"  {r['_id']} | {repr(str(r['utility'])[:120])}")

# Save the re-extraction targets with their existing abstracts
targets = dropped[['_id', 'title', 'abstract']].rename(columns={'_id': 'arxiv_id'})
targets.to_csv('_reextract_targets.csv', index=False)
print(f'\nSaved {len(targets)} targets to _reextract_targets.csv')
