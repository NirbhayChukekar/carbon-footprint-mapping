@echo off
title CarbonLens App
echo ===================================================
echo   Opening CarbonLens in your default browser...
echo ===================================================
cd /d "%~dp0"
start index.html
echo Opened index.html successfully!
timeout /t 3 >nul
exit
