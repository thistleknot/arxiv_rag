@echo off
REM Test different batch sizes to find optimal GPU utilization

echo ============================================================
echo BATCH SIZE OPTIMIZATION TEST
echo ============================================================
echo.
echo Testing batch sizes: 32, 64, 128
echo Each test: 100 chunks (~5-10 seconds each)
echo.

echo [1/3] Testing batch_size=32 (baseline - known working)
echo --------------------------------------------------------
c:\users\user\py310\scripts\python.exe build_triplet_bm25_batched.py --chunks checkpoints/chunks.msgpack --bio-model bio_tagger_best.pt --output-dir test_batch32 --max-chunks 100 --batch-size 32
echo.
echo Baseline complete. Press any key to test batch_size=64...
pause >nul

echo.
echo [2/3] Testing batch_size=64 (2x batches)
echo --------------------------------------------------------
c:\users\user\py310\scripts\python.exe build_triplet_bm25_batched.py --chunks checkpoints/chunks.msgpack --bio-model bio_tagger_best.pt --output-dir test_batch64 --max-chunks 100 --batch-size 64
echo.
if errorlevel 1 (
    echo ❌ batch_size=64 FAILED - GPU out of memory
    echo Optimal batch size: 32
    goto summary
)
echo ✅ batch_size=64 SUCCESS - GPU can handle 2x batches
echo Press any key to test batch_size=128...
pause >nul

echo.
echo [3/3] Testing batch_size=128 (4x batches)
echo --------------------------------------------------------
c:\users\user\py310\scripts\python.exe build_triplet_bm25_batched.py --chunks checkpoints/chunks.msgpack --bio-model bio_tagger_best.pt --output-dir test_batch128 --max-chunks 100 --batch-size 128
echo.
if errorlevel 1 (
    echo ❌ batch_size=128 FAILED - GPU out of memory
    echo Optimal batch size: 64
    goto summary
)
echo ✅ batch_size=128 SUCCESS - GPU can handle 4x batches!

:summary
echo.
echo ============================================================
echo SUMMARY
echo ============================================================
echo.
echo Check the outputs above to see:
echo   - Which batch sizes worked
echo   - Throughput (chunks/sec) for each
echo   - GPU memory usage
echo.
echo Recommendation:
echo   Use the largest batch size that worked successfully
echo.
pause
