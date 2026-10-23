@echo off
REM ============================================================
REM CBIR System - Deployment Script for Windows
REM ============================================================
REM
REM This script starts the CBIR system:
REM 1. Checks MongoDB connection
REM 2. Starts the FastAPI backend
REM 3. Opens the web interface
REM
REM Prerequisites:
REM - Python 3.11+ installed
REM - MongoDB installed and running
REM - Virtual environment activated (optional)
REM
REM Usage:
REM   Double-click start.bat or run from command line:
REM   > start.bat
REM
REM ============================================================

echo.
echo ============================================================
echo CBIR System - MIRFLICKR-1M Image Search Engine
echo ============================================================
echo.

REM Set environment variables
set DATA_DIR=%~dp0data
set MONGO_URI=mongodb://localhost:27017

echo [1/4] Checking prerequisites...
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.11+
    pause
    exit /b 1
)
echo   [OK] Python found

REM Check MongoDB (try to connect)
mongod --version >nul 2>&1
if errorlevel 1 (
    echo   [WARN] MongoDB not found in PATH
    echo   [WARN] Make sure MongoDB is installed and running
) else (
    echo   [OK] MongoDB found
)

REM Check data directory
echo   Checking data directory: %DATA_DIR%
if not exist "%DATA_DIR%" (
    echo   [ERROR] Data directory not found: %DATA_DIR%
    echo   [ERROR] Please create data folder or download from MIRFLICKR
    echo.
    pause
    exit /b 1
)
if not exist "%DATA_DIR%\features" (
    echo   [WARN] Features directory not found: %DATA_DIR%\features
    echo   [WARN] You may need to run: python scripts/parse_descriptors.py
    echo.
    pause
    REM Continue anyway - maybe just testing
) else (
    echo   [OK] Data directory found
)

echo.
echo [2/4] Checking dependencies...
echo.

REM Check if in backend directory or need to cd
if exist "backend\requirements.txt" (
    cd backend
) else if exist "requirements.txt" (
    REM Already in backend directory
) else (
    echo   [ERROR] Cannot find requirements.txt
    echo   [ERROR] Please run this script from the project root directory
    pause
    exit /b 1
)

REM Install dependencies if needed
pip show fastapi >nul 2>&1
if errorlevel 1 (
    echo   Installing dependencies...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo   [ERROR] Failed to install dependencies
        pause
        exit /b 1
    )
) else (
    echo   [OK] Dependencies already installed
)

echo.
echo [3/4] Starting backend server...
echo.

REM Start backend in background
start "CBIR Backend" python main.py

REM Wait for server to start
echo   Waiting for server to start...
timeout /t 5 /nobreak >nul

echo.
echo [4/4] Opening web interface...
echo.

REM Open browser
start http://localhost:8000

echo.
echo ============================================================
echo CBIR System is running!
echo ============================================================
echo.
echo   Backend API:  http://localhost:8000
echo   API Docs:     http://localhost:8000/docs
echo   Frontend:     Open frontend/index.html in your browser
echo.
echo   Press Ctrl+C in the backend window to stop the server
echo.
echo ============================================================
echo.

REM Keep the window open
pause
