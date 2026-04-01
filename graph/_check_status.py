import os, importlib, json, sys
from pathlib import Path

ROOT  = Path(r"c:\Users\user\arxiv_id_lists")
os.chdir(ROOT)

out = {}
chk   = ROOT / "checkpoints/bio_triplets_checkpoint.msgpack"
final = ROOT / "checkpoints/bio_triplets_full_corpus.msgpack"
out["checkpoint_exists"] = chk.exists()
out["checkpoint_mb"]     = round(chk.stat().st_size / 1024 / 1024, 1) if chk.exists() else 0
out["final_exists"]      = final.exists()
out["final_mb"]          = round(final.stat().st_size / 1024 / 1024, 1) if final.exists() else 0

for pkg in ["torch", "torch_geometric", "sklearn", "networkx", "model2vec", "msgpack"]:
    try:
        m = importlib.import_module(pkg)
        out[pkg] = getattr(m, "__version__", "ok")
    except ImportError:
        out[pkg] = "MISSING"

if out.get("torch_geometric") != "MISSING":
    try:
        from torch_geometric.nn import GATv2Conv  # noqa
        out["GATv2Conv"] = "ok"
    except Exception as e:
        out["GATv2Conv"] = str(e)

result_path = ROOT / "graph/_status.json"
result_path.write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
