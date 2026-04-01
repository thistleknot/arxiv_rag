import os, pandas as pd

post = 'papers/post_processed'

existing = set()
for f in os.listdir(post):
    if not f.endswith('.md'):
        continue
    stem = os.path.splitext(f)[0]
    parts = stem.split('_', 1)
    if len(parts) == 2:
        existing.add(parts[0] + '.' + parts[1])

clean = pd.read_csv(post + '/arxiv_data_with_analysis_cleaned.csv', encoding_errors='replace')
ids = set(clean['arxiv_id'].astype(str).str.strip().str.strip('"'))

missing = sorted(ids - existing)
print(f'CSV:          {len(ids)}')
print(f'Existing .md: {len(existing)}')
print(f'Missing .md:  {len(missing)}')
print('Sample missing:', missing[:8])
