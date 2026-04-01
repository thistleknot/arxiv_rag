import csv

targets = {
    '2504.15965','2504.19413','2505.22101','2507.07957','2502.14802',
    '2501.07278','2409.05591','2508.16153','2407.09450','2306.07174',
    '2405.14831','2502.06049','2401.18059','2309.02427','2306.03901',
    '2409.12294','2410.08133','2601.00671','2308.15022','2310.05029',
    '2404.16130','2408.09955','2412.15605'
}

with open(r'C:\Users\user\arxiv_id_lists\papers\post_processed\arxiv_data_with_analysis.csv', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    found = []
    for row in reader:
        aid = row['arxiv_id'].strip().strip('"')
        if aid in targets:
            found.append(row)

print(f"Found {len(found)} of {len(targets)} targets\n")
for row in found:
    print(f"=== {row['arxiv_id'].strip().strip(chr(34))} ===")
    print(f"Title: {row['title']}")
    print(f"Utility: {row['utility']}\n")
