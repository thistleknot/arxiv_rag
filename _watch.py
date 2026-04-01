import json, csv, time
from pathlib import Path

state = Path('_extractor_state.json')
csvp  = Path('papers/post_processed/arxiv_data_with_analysis.csv')

for i in range(120):
    processed = len(json.loads(state.read_text(encoding='utf-8-sig')).get('processed', []))
    rows = sum(1 for _ in csv.reader(csvp.open(encoding='utf-8', errors='replace'))) - 1
    ts = time.strftime("%H:%M:%S")
    print(f"{ts}  processed={processed}/283  csv_rows={rows}", flush=True)
    time.sleep(30)
