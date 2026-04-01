import pathlib, pandas as pd

post = pathlib.Path("papers/post_processed")
df = pd.read_csv(str(post / "arxiv_data_with_analysis_cleaned.csv"), encoding_errors="replace")
ids = [str(x).strip().strip('"') for x in df["arxiv_id"]]
md_stems = set(f.stem for f in post.glob("*.md"))

miss = [a for a in ids if a.replace(".", "_") not in md_stems]
print(f"CSV IDs: {len(ids)}  MDs: {len(md_stems)}  Missing: {len(miss)}")
if miss:
    for m in miss[:20]:
        print(f"  {m}")
else:
    print("ALL COMPLETE")
