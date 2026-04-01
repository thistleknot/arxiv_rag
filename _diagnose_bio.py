"""Diagnose BIO label failures for rows 29,31,32,35,44."""
import sys
import re
import msgpack
sys.path.insert(0, '.')

TARGET_ROWS = [29, 31, 32, 35, 44]

with open(r'c:\Users\user\arxiv_id_lists\data\bio_training_test10_v14.msgpack', 'rb') as f:
    data = msgpack.unpackb(f.read(), raw=False)

lines = []
for idx in TARGET_ROWS:
    ex = data[idx]
    sentence = ex.get('sentence', '')
    triplets = ex.get('triplets', [])
    lines.append(f"=== Row {idx} ===")
    lines.append(f"  SENTENCE: {sentence}")
    for i, t in enumerate(triplets):
        lines.append(f"  TRIPLET {i}:")
        lines.append(f"    subject  : {repr(t.get('subject',''))}")
        lines.append(f"    predicate: {repr(t.get('predicate',''))}")
        lines.append(f"    object   : {repr(t.get('object',''))}")
    lines.append("")

output = '\n'.join(lines)
print(output)
with open(r'c:\Users\user\arxiv_id_lists\_diagnose_bio_out.txt', 'w', encoding='utf-8') as f:
    f.write(output)
