import msgpack, sys, os

path = r'c:\Users\user\arxiv_id_lists\data\bio_training_test10_v15.msgpack'
out  = r'c:\Users\user\AppData\Local\Temp\_diag4_out.txt'

with open(path, 'rb') as f:
    data = msgpack.unpackb(f.read(), raw=False)

print("TYPE:", type(data), "LEN:", len(data))
if isinstance(data, dict):
    print("KEYS (first 5):", list(data.keys())[:5])
    data = data.get('training_data', data.get('examples', data.get('data', list(data.values())[0])))
print("EXAMPLES len:", len(data))

lines = []
for idx in [20, 29]:
    ex = data[idx]
    s  = ex.get('sentence', '')
    ts = ex.get('triplets', [])
    lines.append(f"=== Row {idx} ===")
    lines.append(f"SENT: {s}")
    for i, t in enumerate(ts):
        lines.append(f"  T{i}: subj={repr(t.get('subject','?'))}")
        lines.append(f"       pred={repr(t.get('predicate','?'))}")
        lines.append(f"       obj ={repr(t.get('object','?'))}")
    lines.append("")

text = '\n'.join(lines)
with open(out, 'w', encoding='utf-8') as f:
    f.write(text)

print("DONE wrote", out)
