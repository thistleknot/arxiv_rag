@echo off
REM Model2Vec Qwen3 Embedding - Quick Start Script

echo ============================================================
echo MODEL2VEC QWEN3 CHUNK EMBEDDINGS
echo ============================================================
echo.

if "%1"=="" goto menu
if /i "%1"=="full" goto full
if /i "%1"=="test" goto test
if /i "%1"=="catalog" goto catalog
if /i "%1"=="check" goto check
goto usage

:menu
echo What would you like to do?
echo.
echo [1] Embed Full Corpus (161k chunks, ~20 min)
echo [2] Test Run (1000 chunks, ~10 sec)
echo [3] Update Feature Catalog
echo [4] Check Output
echo [5] Exit
echo.
set /p choice="Enter choice (1-5): "

if "%choice%"=="1" goto full
if "%choice%"=="2" goto test
if "%choice%"=="3" goto catalog
if "%choice%"=="4" goto check
if "%choice%"=="5" goto end
echo Invalid choice!
goto menu

:full
echo.
echo ============================================================
echo Embedding Full Corpus...
echo ============================================================
echo.
echo Chunks:      161,389
echo Model:       Model2Vec Qwen3 (256d)
echo Output:      checkpoints/chunk_embeddings_qwen3.msgpack
echo Est. time:   ~20 minutes
echo.
echo Progress will be shown. Press Ctrl+C to stop.
echo.
c:\users\user\py310\scripts\python.exe embed_chunks_qwen3.py
echo.
pause
goto end

:test
echo.
echo ============================================================
echo Test Run (1000 chunks)...
echo ============================================================
echo.
c:\users\user\py310\scripts\python.exe embed_chunks_qwen3.py --max-chunks 1000
echo.
pause
goto end

:catalog
echo.
echo ============================================================
echo Updating Feature Catalog...
echo ============================================================
echo.
c:\users\user\py310\scripts\python.exe update_catalog_qwen3.py
echo.
pause
goto end

:check
echo.
echo ============================================================
echo Checking Output...
echo ============================================================
echo.
c:\users\user\py310\scripts\python.exe -c "import msgpack; data = msgpack.load(open('checkpoints/chunk_embeddings_qwen3.msgpack', 'rb'), raw=False); print(f'Shape: {data[\"shape\"]}'); print(f'Metadata: {data[\"metadata\"]}')"
echo.
pause
goto end

:usage
echo.
echo Usage: run_qwen3_embed.bat [command]
echo.
echo Commands:
echo   full     - Embed full corpus (~20 min)
echo   test     - Test run (1000 chunks)
echo   catalog  - Update feature catalog
echo   check    - Check output file
echo   (no cmd) - Show interactive menu
echo.
echo Examples:
echo   run_qwen3_embed.bat
echo   run_qwen3_embed.bat full
echo   run_qwen3_embed.bat test
echo.
goto end

:end
