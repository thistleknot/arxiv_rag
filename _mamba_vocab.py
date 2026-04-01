import pandas as pd

df = pd.read_csv("papers/post_processed/arxiv_data_with_analysis_cleaned.csv")
df["arxiv_id"] = df["arxiv_id"].astype(str).str.strip('"')

for pid in ["2312.00752", "2405.21060", "2406.07522", "2410.07145"]:
    row = df[df["arxiv_id"] == pid]
    if len(row):
        r = row.iloc[0]
        print(f"[{pid}] {r['title']}")
        print(f"  utility:  {str(r['utility'])[:200]}")
        print(f"  thesis:   {str(r['thesis'])[:160]}")
        print(f"  abstract: {str(r['abstract'])[:200]}")
        print()
