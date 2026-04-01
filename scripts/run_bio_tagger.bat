@echo off
REM BIO Tagger System - Quick Start Script

echo ============================================================
echo BIO TAGGER SYSTEM
echo ============================================================
echo.

if "%1"=="" goto menu
if /i "%1"=="app" goto app
if /i "%1"=="test" goto test
if /i "%1"=="check" goto check
if /i "%1"=="integration" goto integration
goto usage

:menu
echo What would you like to do?
echo.
echo [1] Run Streamlit App
echo [2] Run Unit Tests
echo [3] Run Integration Test
echo [4] Run System Health Check
echo [5] Exit
echo.
set /p choice="Enter choice (1-5): "

if "%choice%"=="1" goto app
if "%choice%"=="2" goto test
if "%choice%"=="3" goto integration
if "%choice%"=="4" goto check
if "%choice%"=="5" goto end
echo Invalid choice!
goto menu

:app
echo.
echo ============================================================
echo Starting Streamlit BIO Tagger App...
echo ============================================================
echo.
echo App will open at: http://localhost:8501
echo Press Ctrl+C to stop the server
echo.
c:\users\user\py310\scripts\streamlit.exe run streamlit_bio_demo.py
goto end

:test
echo.
echo ============================================================
echo Running Unit Tests...
echo ============================================================
echo.
c:\users\user\py310\scripts\python.exe test_bio_tagger.py
echo.
pause
goto end

:integration
echo.
echo ============================================================
echo Running Integration Test...
echo ============================================================
echo.
c:\users\user\py310\scripts\python.exe test_predictions.py
echo.
pause
goto end

:check
echo.
echo ============================================================
echo Running System Health Check...
echo ============================================================
echo.
c:\users\user\py310\scripts\python.exe check_system.py
echo.
pause
goto end

:usage
echo.
echo Usage: run_bio_tagger.bat [command]
echo.
echo Commands:
echo   app          - Start Streamlit app
echo   test         - Run unit tests
echo   integration  - Run integration test
echo   check        - Run system health check
echo   (no command) - Show interactive menu
echo.
echo Examples:
echo   run_bio_tagger.bat
echo   run_bio_tagger.bat app
echo   run_bio_tagger.bat test
echo.
goto end

:end
