# Graph pipeline launcher — run from workspace root after inference completes.
#
# Prerequisites:
#   checkpoints/bio_triplets_full_corpus.msgpack   <- produced by apply_bio_corpus.py
#
# Steps:
#   1. normalize_entities.py   — cluster spans, embed, write graph/*.json/npy/pkl
#   2. build_kg_dataset.py     — build PyG Data object, write graph/kg_dataset.pt
#   3. train_gat.py --smoke    — 5-epoch sanity check on 10K edges
#   4. train_gat.py            — full 20-epoch training on Quadro RTX 5000
#   5. graph_retriever.py      — end-to-end retrieval test
#
# Run: powershell -ExecutionPolicy Bypass -File run_graph_pipeline.ps1
#      Add -SkipSmoke to skip step 3.
#      Add -SkipTrain to skip steps 3/4 (use existing gat_model.pt).

param(
    [switch]$SkipSmoke,
    [switch]$SkipTrain
)

$PYTHON = "c:\users\user\py310\scripts\python.exe"
$ROOT   = $PSScriptRoot   # workspace root (where this script lives)

Set-Location $ROOT

function Run-Step($label, $cmd) {
    Write-Host "`n=== $label ===" -ForegroundColor Cyan
    & $cmd[0] $cmd[1..($cmd.Length-1)]
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED (exit $LASTEXITCODE)" -ForegroundColor Red
        exit $LASTEXITCODE
    }
    Write-Host "OK" -ForegroundColor Green
}

# Guard
if (-not (Test-Path "checkpoints\bio_triplets_full_corpus.msgpack")) {
    Write-Host "ERROR: checkpoints\bio_triplets_full_corpus.msgpack not found." -ForegroundColor Red
    Write-Host "Wait for apply_bio_corpus.py to finish, then rerun this script."
    exit 1
}

Run-Step "Step 1: Normalize entities" @($PYTHON, "graph\normalize_entities.py")

Run-Step "Step 2: Build PyG dataset" @($PYTHON, "graph\build_kg_dataset.py")

if (-not $SkipTrain) {
    if (-not $SkipSmoke) {
        Run-Step "Step 3: Smoke-test training (5 epochs, 10K edges)" @($PYTHON, "graph\train_gat.py", "--smoke-test")
    }
    Run-Step "Step 4: Full GATv2 training (20 epochs)" @($PYTHON, "graph\train_gat.py")
}

Run-Step "Step 5: Retrieval test" @($PYTHON, "graph\graph_retriever.py", "attention mechanism in transformers")

Write-Host "`n=== Graph pipeline complete ===" -ForegroundColor Green
