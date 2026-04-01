@echo off
REM Build complete BM25 triplet index for full ArXiv corpus

echo ============================================================
echo BUILD FULL CORPUS BM25 TRIPLET INDEX
echo ============================================================
echo.
echo Configuration:
echo   Chunks:      161,389 chunks (checkpoints/chunks.msgpack)
echo   BIO Model:   bio_tagger_best.pt
echo   Batch Size:  64 (tested and working)
echo   Output Dir:  triplet_checkpoints_full
echo.
echo Expected Runtime: ~2.2 hours (based on measured performance)
echo.
echo Pipeline Stages:
echo   1. BERT triplet extraction (batched, GPU)
echo   2. Lowercase + punctuation cleaning
echo   3. Stopword removal + tokenization
echo   4. Lemmatization
echo   5. Synset expansion (WordNet)
echo   6. Hypernym expansion (WordNet)
echo   7. BM25 index construction
echo.
echo All stages are checkpointed - resumable if interrupted
echo.
pause

echo.
echo ============================================================
echo STARTING FULL CORPUS BUILD
echo ============================================================
echo.
echo Started at: %date% %time%
echo.

c:\users\user\py310\scripts\python.exe build_triplet_bm25_batched.py --chunks checkpoints/chunks.msgpack --bio-model bio_tagger_best.pt --output-dir triplet_checkpoints_full --batch-size 64

if errorlevel 1 (
    echo.
    echo ========================================
    echo ❌ BUILD FAILED
    echo ========================================
    echo.
    echo Check the error output above.
    echo Pipeline is checkpointed - you can resume by running this script again.
    echo.
) else (
    echo.
    echo ============================================================
    echo ✅ BUILD COMPLETE!
    echo ============================================================
    echo.
    echo Finished at: %date% %time%
    echo.
    echo Output files:
    echo   triplet_checkpoints_full\stage1_raw_triplets.msgpack
    echo   triplet_checkpoints_full\stage2_cleaned.msgpack
    echo   triplet_checkpoints_full\stage3_tokenized.msgpack
    echo   triplet_checkpoints_full\stage4_lemmatized.msgpack
    echo   triplet_checkpoints_full\stage5_with_synsets.msgpack
    echo   triplet_checkpoints_full\stage6_with_hypernyms.msgpack
    echo   triplet_checkpoints_full\stage7_bm25_index.pkl  ^<--- FINAL INDEX
    echo.
    echo Next Steps:
    echo   1. Integrate BM25 index with RRF retriever
    echo   2. Test graph expansion logic
    echo   3. Run benchmarks
    echo.
)

pause
