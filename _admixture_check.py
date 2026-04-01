"""Quick check: which gap papers exist in CSV, what fields look like."""
import pandas as pd

CSV = "papers/post_processed/arxiv_data_with_analysis_cleaned.csv"
df = pd.read_csv(CSV)
df["arxiv_id"] = df["arxiv_id"].astype(str).str.strip('"')

GAP_IDS = ["2306.15595", "2307.03172"]

ssm = df["title"].str.contains(
    "Mamba|state space|SSM|S4|Hyena|linear recurrence", case=False, na=False
)
flash = df["title"].str.contains(
    "FlashAttention|Flash Attention|IO-aware", case=False, na=False
)

print("=== Gap paper presence ===")
for pid in GAP_IDS:
    row = df[df["arxiv_id"] == pid]
    if len(row):
        r = row.iloc[0]
        print(f"\n{pid}: FOUND  {r['title']}")
        print(f"  utility:  {str(r['utility'])[:160]}")
        print(f"  thesis:   {str(r['thesis'])[:160]}")
        print(f"  barriers: {str(r['barriers'])[:160]}")
        print(f"  abstract: {str(r.get('abstract','n/a'))[:160]}")
    else:
        print(f"\n{pid}: NOT IN CSV")

print("\n=== SSM/Mamba papers in CSV ===")
print(df[ssm][["arxiv_id", "title"]].to_string(index=False))

print("\n=== FlashAttention papers in CSV ===")
print(df[flash][["arxiv_id", "title"]].to_string(index=False))

print(f"\nCSV columns: {df.columns.tolist()}")
print(f"Total rows:  {len(df)}")

# Sample barriers/thesis to see what they look like
print("\n=== Sample barriers (first 5 non-null) ===")
for _, r in df[df["barriers"].notna() & (df["barriers"] != "")].head(5).iterrows():
    print(f"  [{r['arxiv_id']}] {str(r['barriers'])[:200]}")
