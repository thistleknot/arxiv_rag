"""
Gap analysis: arxiv IDs in llm.txt vs arxiv_data_with_analysis.csv
"""
import re
import pandas as pd

LLM_TXT = r"C:\Users\user\Documents\wiki\data science\llm\llm.txt"
CSV_PATH = r"C:\Users\user\arxiv_id_lists\papers\post_processed\arxiv_data_with_analysis.csv"

# 1. Extract IDs from llm.txt (URLs like https://arxiv.org/abs/2601.00366)
with open(LLM_TXT, encoding="utf-8") as f:
    content = f.read()

# Match both URL-embedded IDs and bare IDs
url_ids = re.findall(r'arxiv\.org/abs/(\d{4}\.\d{4,5})', content)
bare_ids = re.findall(r'(?<![/\w])(\d{4}\.\d{4,5})(?!\d)', content)
llm_ids = sorted(set(url_ids + bare_ids))
print(f"Total unique arxiv IDs in llm.txt: {len(llm_ids)}")

# 2. Load CSV and clean IDs
df = pd.read_csv(CSV_PATH)
print(f"\nCSV shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"\nNon-null counts:")
print(df.count())
print(f"\nis_complete value counts:")
print(df['is_complete'].value_counts(dropna=False))

# Normalize arxiv_id column (strip triple quotes)
df['arxiv_id_clean'] = (
    df['arxiv_id'].astype(str)
    .str.replace(r'["\s]', '', regex=True)
)
csv_ids = set(df['arxiv_id_clean'].tolist())
print(f"\nUnique IDs in CSV: {len(csv_ids)}")
print(f"Sample CSV IDs: {list(csv_ids)[:5]}")

# 3. Find gap
gap_ids = [i for i in llm_ids if i not in csv_ids]
print(f"\n=== GAP: {len(gap_ids)} IDs in llm.txt NOT in CSV ===")
for i in gap_ids:
    print(f"  {i}")

# Also: IDs in CSV with missing fields
if 'utility' in df.columns:
    incomplete = df[df['is_complete'].astype(str).str.strip() != 'True']
    print(f"\n=== {len(incomplete)} rows in CSV with is_complete != True ===")
    print(incomplete[['arxiv_id_clean', 'utility', 'barriers', 'thesis']].head(10).to_string())
