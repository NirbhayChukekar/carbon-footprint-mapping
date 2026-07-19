@echo off
title CarbonLens Backend Server
echo ===================================================
echo   Starting CarbonLens Premium Backend Server...
echo ===================================================
cd /d "%~dp0"
if not exist "venv\Scripts\activate.bat" (
    echo Error: venv folder not found. Please ensure this file is placed inside the carbon-api directory.
    pause
    exit /b
)
echo Activating virtual environment...
call .\venv\Scripts\activate.bat
echo Starting FastAPI application with Uvicorn on http://127.0.0.1:8000...
python -m uvicorn run:app --host 127.0.0.1 --port 8000 --reload
pause
