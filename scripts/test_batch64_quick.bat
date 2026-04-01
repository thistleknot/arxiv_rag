@echo off
REM Quick test: Find optimal batch size before full corpus run

echo ============================================================
echo QUICK BATCH SIZE TEST
echo ============================================================
echo.
echo This will test batch_size=64 on 100 chunks (~10 seconds)
echo If it works, you'll save 40+ minutes on the full corpus run
echo.
echo Current performance:
echo   batch_size=32:  22 chunks/sec  -^> 2.2 hours full corpus
echo.
echo Expected if successful:
echo   batch_size=64:  35-40 chunks/sec  -^> 1.5 hours full corpus
echo   SAVINGS: 42 minutes!
echo.
pause

echo.
echo Running test...
echo.
c:\users\user\py310\scripts\python.exe build_triplet_bm25_batched.py --chunks checkpoints/chunks.msgpack --bio-model bio_tagger_best.pt --output-dir test_batch64 --max-chunks 100 --batch-size 64

if errorlevel 1 (
    echo.
    echo ========================================
    echo RESULT: batch_size=64 FAILED
    echo ========================================
    echo.
    echo GPU out of memory. Your GPU can handle:
    echo   ✅ batch_size=32 (tested in Phase 72)
    echo   ❌ batch_size=64 (too large)
    echo.
    echo RECOMMENDATION:
    echo   Run full corpus with batch_size=32 (2.2 hours)
    echo.
    echo   OR implement quantization to enable batch_size=64
    echo   (See OPTIMIZATION_GUIDE.md for quantization setup)
    echo.
) else (
    echo.
    echo ========================================
    echo RESULT: batch_size=64 SUCCESS!
    echo ========================================
    echo.
    echo Your GPU can handle 2x larger batches!
    echo.
    echo RECOMMENDATION:
    echo   Run full corpus with batch_size=64
    echo.
    echo   Command:
    echo   python build_triplet_bm25_batched.py \
    echo     --chunks checkpoints/chunks.msgpack \
    echo     --bio-model bio_tagger_best.pt \
    echo     --output-dir triplet_checkpoints_full \
    echo     --batch-size 64
    echo.
    echo   Expected runtime: 1.5 hours (saves 42 minutes!)
    echo.
    echo   OPTIONAL: Test batch_size=128 for even more speed
    echo   (Run test_batch_sizes.bat to test 128)
    echo.
)

pause
