@echo off
setlocal
set "ROOT=%~dp0"
set "DATA_DIR=%ROOT%data"
set "MONGO_URI=mongodb://localhost:27017"
set "MONGO_DB=cbir"
set "TORCH_HOME=%DATA_DIR%\models\torch"
set "PYTHON=%ROOT%venv\Scripts\python.exe"

pushd "%ROOT%"
echo.
echo ============================================================
echo CBIR System - MIRFLICKR-1M Image Search Engine
echo ============================================================
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Install Python 3.11 or newer and add it to PATH.
    goto :fail
)
python -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python 3.11 or newer is required.
    goto :fail
)

for %%D in (thumbs1-33 thumbs34-66 thumbs67-100 tags license features_edgehistogram features_homogeneoustexture) do (
    if not exist "%DATA_DIR%\mirflickr\raw\%%D\" (
        echo [ERROR] Required raw data folder is missing:
        echo         %DATA_DIR%\mirflickr\raw\%%D
        echo Download and extract all MIRFLICKR archives as described in README.md.
        goto :fail
    )
)

if not exist "%PYTHON%" (
    echo [1/5] Creating Python virtual environment...
    python -m venv "%ROOT%venv"
    if errorlevel 1 (
        echo [ERROR] Could not create the virtual environment.
        goto :fail
    )
)

"%PYTHON%" -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] The project virtual environment is not using Python 3.11 or newer.
    goto :fail
)

echo [2/5] Checking Python dependencies...
"%PYTHON%" -c "import fastapi, numpy, pymongo, faiss, torch, torchvision" >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies. This may take several minutes on first run...
    "%PYTHON%" -m pip install -r "%ROOT%backend\requirements.txt"
    if errorlevel 1 (
        echo [ERROR] Dependency installation failed.
        goto :fail
    )
)

echo [3/5] Checking MongoDB...
"%PYTHON%" -c "from pymongo import MongoClient; MongoClient('mongodb://localhost:27017', serverSelectionTimeoutMS=2000).admin.command('ping')" >nul 2>&1
if errorlevel 1 (
    where mongod >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] MongoDB is not running and mongod was not found in PATH.
        echo Install MongoDB Community Server or start your MongoDB service, then retry.
        goto :fail
    )
    if not exist "%DATA_DIR%\mongo\" mkdir "%DATA_DIR%\mongo"
    echo Starting a local MongoDB instance...
    start "CBIR MongoDB" /min mongod --dbpath "%DATA_DIR%\mongo" --bind_ip 127.0.0.1 --port 27017
    for /l %%I in (1,1,30) do (
        "%PYTHON%" -c "from pymongo import MongoClient; MongoClient('mongodb://localhost:27017', serverSelectionTimeoutMS=1000).admin.command('ping')" >nul 2>&1
        if not errorlevel 1 goto :mongo_ready
        timeout /t 2 /nobreak >nul
    )
    echo [ERROR] MongoDB did not become available.
    goto :fail
)

:mongo_ready
echo [4/5] Creating missing features, indexes, metadata, and MongoDB records...
"%PYTHON%" "%ROOT%backend\scripts\prepare_data.py"
if errorlevel 1 (
    echo [ERROR] Data preparation failed. Review the error above.
    goto :fail
)

echo [5/5] Starting the backend...
start "CBIR Backend" /D "%ROOT%backend" "%PYTHON%" -m uvicorn main:app --host 0.0.0.0 --port 8000
for /l %%I in (1,1,30) do (
    "%PYTHON%" -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health', timeout=2)" >nul 2>&1
    if not errorlevel 1 goto :backend_ready
    timeout /t 2 /nobreak >nul
)
echo [ERROR] Backend did not become available at http://localhost:8000.
goto :fail

:backend_ready
echo Opening the web interface...
start "" "http://localhost:8000/"
echo.
echo CBIR is running: http://localhost:8000/
echo API docs: http://localhost:8000/docs
echo Close the CBIR Backend window to stop the API.
echo.
popd
pause
exit /b 0

:fail
popd
pause
exit /b 1
