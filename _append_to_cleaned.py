"""Append valid, deduped new rows from arxiv_data_with_analysis.csv to _cleaned.csv."""
import pandas as pd, json
from pathlib import Path

RAW = Path(r'C:\Users\user\arxiv_id_lists\papers\post_processed\arxiv_data_with_analysis.csv')
CLEAN = Path(r'C:\Users\user\arxiv_id_lists\papers\post_processed\arxiv_data_with_analysis_cleaned.csv')
BACKUP = CLEAN.with_suffix('.csv.bak')

raw = pd.read_csv(RAW, encoding_errors='replace')
clean = pd.read_csv(CLEAN, encoding_errors='replace')
norm = lambda s: str(s).strip().strip('"').strip()
raw['_id'] = raw['arxiv_id'].apply(norm)
clean['_id'] = clean['arxiv_id'].apply(norm)

SENTINELS = ['validation failed', 'invalid utility']
def is_bad(u):
    if pd.isna(u): return True
    s = str(u).strip()
    if s in ('', '[]', 'null', 'nan'): return True
    sl = s.lower()
    if any(sen in sl for sen in SENTINELS): return True
    if s.startswith('['):
        try:
            v = json.loads(s)
            if isinstance(v, list) and len(v) == 0: return True
        except Exception: pass
    return False

new_ids = set(raw['_id']) - set(clean['_id'])
new_rows = raw[raw['_id'].isin(new_ids)].copy()
new_rows = new_rows[~new_rows['utility'].apply(is_bad)]
new_rows = new_rows.drop_duplicates(subset=['_id'], keep='first')
new_rows = new_rows.drop(columns=['_id'])

# Align columns to clean's schema
clean_cols = [c for c in clean.columns if c != '_id']
new_rows = new_rows[[c for c in clean_cols if c in new_rows.columns]]

print(f'About to append {len(new_rows)} unique valid new rows to cleaned CSV')
print(f'Backup: {BACKUP}')

# Backup, then append
import shutil
shutil.copy2(CLEAN, BACKUP)
new_rows.to_csv(CLEAN, mode='a', header=False, index=False)

# Verify
clean2 = pd.read_csv(CLEAN, encoding_errors='replace')
print(f'Cleaned CSV: {len(clean)} -> {len(clean2)} rows  (delta={len(clean2)-len(clean)})')
appended = clean2.tail(len(new_rows))[['arxiv_id','title']].head(30)
print(appended.to_string(index=False))
