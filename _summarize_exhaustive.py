import pandas as pd

f = '_admixture_exhaustive_results.csv'
df = pd.read_csv(f)

cols = [
    'name','mode','recall@10','recall@22','recall@50','mrr',
    '2307.03172','2306.15595','2312.00752','2406.07522','2410.07145','2405.21060'
]

want = [
    'concat_utility',
    'concat_title_utility',
    'concat_title_abstract_utility',
    'blend_t0.4_a0.3_u0.3',
    'concat_title_abstract',
]

sub = df[df['name'].isin(want)][cols].copy()
sub = sub.sort_values('name')
print(sub.to_string(index=False))

best_r22 = df.sort_values(['recall@22','mrr','recall@50'], ascending=[False,False,False]).head(5)
print('\nTop5 by R@22 then MRR:\n')
print(best_r22[cols].to_string(index=False))

best_mrr = df.sort_values('mrr', ascending=False).head(5)
print('\nTop5 by MRR:\n')
print(best_mrr[cols].to_string(index=False))
