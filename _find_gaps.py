import pandas as pd, json, ast

raw   = pd.read_csv(r'papers\post_processed\arxiv_data_with_analysis.csv', encoding_errors='replace')
clean = pd.read_csv(r'papers\post_processed\arxiv_data_with_analysis_cleaned.csv', encoding_errors='replace')

print('Raw rows:', len(raw))
print('Clean rows:', len(clean))

def strip_id(s):
    return str(s).strip('"').strip("'").strip()

raw['_id'] = raw['arxiv_id'].apply(strip_id)
clean['_id'] = clean['arxiv_id'].apply(strip_id)

raw_ids   = set(raw['_id'])
clean_ids = set(clean['_id'])
dropped   = raw_ids - clean_ids
print(f'\nIDs in raw but NOT in clean: {len(dropped)}')

dropped_rows = raw[raw['_id'].isin(dropped)].copy()

def is_empty(u):
    if pd.isna(u): return True
    s = str(u).strip()
    if len(s) <= 4: return True
    for p in (json.loads, ast.literal_eval):
        try:
            v = p(s)
            return isinstance(v, list) and len(v) == 0
        except: pass
    return False

dropped_rows['empty_utility'] = dropped_rows['utility'].apply(is_empty)
print('empty utility:', dropped_rows['empty_utility'].sum())
print('non-empty utility:', (~dropped_rows['empty_utility']).sum())

if 'is_complete' in dropped_rows.columns:
    print('\nis_complete on dropped:')
    print(dropped_rows['is_complete'].value_counts())

print('\nSample dropped utility:')
for _, r in dropped_rows.head(8).iterrows():
    print(f"  {r['_id']} | empty={r['empty_utility']} | {repr(str(r.get('utility',''))[:80])}")

# Save the gap list
gap = dropped_rows[dropped_rows['empty_utility'] == True][['arxiv_id','title','abstract']]
print(f'\nTrue gaps (empty utility): {len(gap)}')
gap.to_csv('_utility_gaps.csv', index=False)
print('Written to _utility_gaps.csv')
