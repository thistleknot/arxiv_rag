"""
Find out WHY 557 rows were dropped from cleaned CSV.
All 557 have parseable utility, title, abstract in raw -- so no re-extraction needed.
"""
import pandas as pd, json, ast

raw   = pd.read_csv(r'papers\post_processed\arxiv_data_with_analysis.csv', encoding_errors='replace')
clean = pd.read_csv(r'papers\post_processed\arxiv_data_with_analysis_cleaned.csv', encoding_errors='replace')

def strip_id(s):
    return str(s).strip('"').strip().strip("'").strip()

raw['_id']   = raw['arxiv_id'].apply(strip_id)
clean['_id'] = clean['arxiv_id'].apply(strip_id)

print('=== Raw ===')
print(f'  Total rows:       {len(raw)}')
print(f'  Unique arxiv_ids: {raw["_id"].nunique()}')
print(f'  Duplicate IDs:    {len(raw) - raw["_id"].nunique()}')

print('\n=== Clean ===')
print(f'  Total rows:       {len(clean)}')
print(f'  Unique arxiv_ids: {clean["_id"].nunique()}')

# Papers in raw but not in clean AT ALL (unique IDs)
raw_only = set(raw['_id']) - set(clean['_id'])
both     = set(raw['_id']) & set(clean['_id'])
print(f'\n=== Coverage ===')
print(f'  Paper IDs only in raw (missing from clean): {len(raw_only)}')
print(f'  Paper IDs in both:                         {len(both)}')

# For the raw_only papers -- do they have valid utility?
raw_only_rows = raw[raw['_id'].isin(raw_only)].drop_duplicates('_id')
print(f'\n=== The {len(raw_only)} papers completely absent from clean ===')
print(f'  Have abstract: {raw_only_rows["abstract"].notna().sum()}')
print(f'  Have title:    {raw_only_rows["title"].notna().sum()}')

def parse_utility(u):
    if pd.isna(u): return None
    s = str(u).strip()
    for p in (json.loads, ast.literal_eval):
        try:
            v = p(s)
            if isinstance(v, list) and len(v) > 0:
                return v
        except: pass
    return None

raw_only_rows = raw_only_rows.copy()
raw_only_rows['parsed_utility'] = raw_only_rows['utility'].apply(parse_utility)
has_utility = raw_only_rows['parsed_utility'].notna().sum()
print(f'  Have valid parseable utility: {has_utility}')
print(f'  Need re-extraction: {len(raw_only_rows) - has_utility}')

print('\nSample of the missing papers (they already have utility!):')
for _, r in raw_only_rows.head(5).iterrows():
    pu = r['parsed_utility']
    print(f"  {r['_id']} | {r['title'][:60]}")
    if pu:
        print(f"    utility[0]: {pu[0][:80]}")

# Write them out -- ready to merge directly, no LLM needed
missing_complete = raw_only_rows[raw_only_rows['parsed_utility'].notna()][
    ['_id','title','abstract','utility','barriers','thesis','is_complete']
].rename(columns={'_id':'arxiv_id'})
missing_complete.to_csv('_missing_papers.csv', index=False)
print(f'\nWritten {len(missing_complete)} complete rows to _missing_papers.csv (no re-extraction needed)')

need_llm = raw_only_rows[raw_only_rows['parsed_utility'].isna()][
    ['_id','title','abstract']
].rename(columns={'_id':'arxiv_id'})
if len(need_llm):
    need_llm.to_csv('_need_llm_extraction.csv', index=False)
    print(f'Written {len(need_llm)} rows needing LLM extraction to _need_llm_extraction.csv')
