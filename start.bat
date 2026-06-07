@echo off
chcp 936 >nul 2>&1
title Personal Knowledge Base

set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"

echo.
echo  ========================================
echo         Personal Knowledge Base
echo  ========================================
echo.

:: Check Python is available
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo  [ERROR] Python not found in PATH.
    echo  Please install Python 3.10+ and add it to PATH.
    pause
    exit /b 1
)

:: Check .env exists
if not exist "%ROOT%\.env" (
    echo  [WARNING] .env file not found.
    echo  Please copy .env.example to .env and configure your API key.
    echo.
)

:: Kill any existing processes on ports 8080 and 8001
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8080.*LISTENING" 2^>nul') do taskkill /F /PID %%a >nul 2>&1
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8001.*LISTENING" 2^>nul') do taskkill /F /PID %%a >nul 2>&1

echo  [1/2] Starting web server on :8080 ...
cd /d "%ROOT%"
start /b python -m http.server 8080 --directory web

timeout /t 2 /nobreak >nul

echo  [2/2] Starting backend on :8001 ...
echo.
echo  ========================================
echo   Press Ctrl+C to stop all services.
echo   Browser will open shortly...
echo  ========================================
echo.

timeout /t 4 /nobreak >nul
start "" "http://localhost:8080"

:: Run backend in foreground (Ctrl+C stops it; background HTTP server shares the console)
python backend\main_siliconflow_rag.py
