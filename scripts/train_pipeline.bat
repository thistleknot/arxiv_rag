@echo off
REM ============================================================================
REM BIO Tagger Training Pipeline
REM ============================================================================
REM Step 1: Extract 1500 chunks with atomic decomposition
REM Step 2: Train with full dataset using best hyperparameters
REM ============================================================================

echo ======================================================================
echo BIO TAGGER TRAINING PIPELINE
echo ======================================================================
echo.
echo [STEP 1/2] Extracting 1500 chunks...
echo   Output: bio_training_1500chunks_atomic.msgpack
echo.

c:\users\user\py310\scripts\python.exe extract_bio_atomic_clean.py --chunks 1500 --output bio_training_1500chunks_atomic.msgpack

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ❌ Extraction failed! Exiting...
    exit /b 1
)

echo.
echo ✅ Extraction complete!
echo.
echo [STEP 2/2] Training on full dataset...
echo   Using best hyperparameters from tuning
echo   Data: bio_training_1500chunks_atomic.msgpack
echo.

c:\users\user\py310\scripts\python.exe train_bio_tagger.py --data bio_training_1500chunks_atomic.msgpack

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ❌ Training failed! Exiting...
    exit /b 1
)

echo.
echo ======================================================================
echo ✅ PIPELINE COMPLETE!
echo ======================================================================
echo   Model saved: bio_tagger_best.pt
echo   Results saved: training_results.msgpack
echo ======================================================================
echo.

pause
