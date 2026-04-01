import json

data = json.load(open('_admixture_results.json'))

id_keys = [
    ('2307.03172', 'rank_2307_03172', 'LostMid'),
    ('2306.15595', 'rank_2306_15595', 'RoPE/PI'),
    ('2312.00752', 'rank_2312_00752', 'Mamba'),
    ('2406.07522', 'rank_2406_07522', 'Samba'),
    ('2410.07145', 'rank_2410_07145', 'StfMamba'),
    ('2405.21060', 'rank_2405_21060', 'Mamba-2'),
]

top_configs = ['baseline_utility','all_fields','blend_title60_util40','title_utility','abstract_only','blend_util40_abst40_tit20']

print(f"{'Config':<28}", end='')
for _, _, name in id_keys:
    print(f"{name:>12}", end='')
print(f"{'R@10':>6}{'R@22':>6}{'R@50':>6}{'MRR':>8}")
print('-'*120)

for cfg in top_configs:
    row = next(r for r in data if r['name'] == cfg)
    print(f"{cfg:<28}", end='')
    for _, key, _ in id_keys:
        rank = row[key]
        flag = '<' if isinstance(rank, int) and rank <= 22 else ' '
        print(f"{str(rank)+flag:>12}", end='')
    print(f"{row['recall@10']:>6.2f}{row['recall@22']:>6.2f}{row['recall@50']:>6.2f}{row['mrr']:>8.3f}")
