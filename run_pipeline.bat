@echo off
setlocal enabledelayedexpansion

set "INPUT_FILE=%~f1"
set "INPUT_DIR=%~dp1"
set "INPUT_STEM=%~n1"
set "INPUT_EXT=%~x1"
set "SCRIPT_DIR=%~dp0"

if "%INPUT_FILE%"=="" (
    echo Usage: run_pipeline.bat ^<input_file^>
    echo Supported: .pdf  .md
    exit /b 1
)

echo =========================================
echo  Document VLM Pipeline
echo  Input: %INPUT_FILE%
echo =========================================
echo.

pushd "%INPUT_DIR%"

rem ── Phase 1: Docling extraction ──────────────────────────────────────────────
if /i "%INPUT_EXT%"==".pdf" (
    echo [1/5] Extracting document with docling...
    docling "%INPUT_STEM%.pdf" --to md --output . --pdf-backend dlparse_v4
    if errorlevel 1 (
        popd
        echo ERROR: docling extraction failed.
        exit /b 1
    )
    echo.
) else if /i "%INPUT_EXT%"==".md" (
    echo [1/5] Skipping docling ^(input is already Markdown^).
    echo.
) else (
    popd
    echo ERROR: Unsupported file type: %INPUT_EXT%
    echo        Supported: .pdf  .md
    exit /b 1
)

rem ── Phase 2: Extract base64 images to CSV ────────────────────────────────────
echo [2/5] Extracting base64 images to CSV...
python "%SCRIPT_DIR%post_process_md-csv.py" "%INPUT_STEM%.md"
if errorlevel 1 (
    popd
    echo ERROR: base64 extraction failed.
    exit /b 1
)
echo.

rem ── Phase 3: VLM descriptions via copilot-proxy ──────────────────────────────
echo [3/5] Generating VLM descriptions ^(qwen3-vl:2b via Ollama^)...
python "%SCRIPT_DIR%vlm_describe.py" "%INPUT_STEM%.csv"
if errorlevel 1 (
    popd
    echo ERROR: VLM description generation failed.
    exit /b 1
)
echo.

rem ── Phase 4: Reinsert descriptions into document ─────────────────────────────
echo [4/5] Reinserting descriptions into document...
python "%SCRIPT_DIR%reinsert_descriptions.py" "%INPUT_STEM%.md" "%INPUT_STEM%.csv"
if errorlevel 1 (
    popd
    echo ERROR: Reinsert failed.
    exit /b 1
)
echo.

rem ── Phase 5: Extract core methods as pseudocode ─────────────────────────────
echo [5/5] Extracting core methods as pseudocode...
python "%SCRIPT_DIR%extract_methods.py" "%INPUT_STEM%.md"
if errorlevel 1 (
    popd
    echo ERROR: Methods extraction failed.
    exit /b 1
)
echo.

popd

echo =========================================
echo  Pipeline complete
echo  Output: %INPUT_DIR%%INPUT_STEM%.md
echo  Methods: %INPUT_DIR%%INPUT_STEM%_methods.md
echo =========================================
