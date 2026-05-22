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
echo  Starting services...
echo  The browser will open automatically.
echo.

start "Personal Knowledge Base" /d "%ROOT%" cmd /k "start /b python -m http.server 8080 --directory web & python backend\main_siliconflow_rag.py"

timeout /t 6 /nobreak >nul
start http://localhost:8080
