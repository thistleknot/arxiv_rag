import csv

targets = ['2505.22101','2507.07957','2409.12294','2409.05591','2405.14831','2401.18059']
seen = set()

with open(r'papers\post_processed\arxiv_data_with_analysis.csv', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        aid = row['arxiv_id'].strip().strip('"')
        if aid in targets and aid not in seen:
            seen.add(aid)
            print(f"=== {aid} ===")
            print(f"Title: {row['title']}")
            print(f"Abstract: {row['abstract'][:600]}")
            print(f"Thesis: {row['thesis']}")
            print()

print(f"Found: {len(seen)} / {len(targets)}")
