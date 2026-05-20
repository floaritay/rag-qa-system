@echo off
chcp 65001 >nul 2>&1
title 智能课程助手 - 启动中...

echo.
echo  ╔══════════════════════════════════════╗
echo  ║        智能课程助手 · 一键启动        ║
echo  ╚══════════════════════════════════════╝
echo.

:: 获取脚本所在目录
set "ROOT=%~dp0"

:: 启动后端
echo [1/3] 启动后端服务 (端口 8001)...
start "课程助手-后端" cmd /k "cd /d "%ROOT%" && python backend\main_siliconflow_rag.py"

:: 等待后端启动
echo [2/3] 等待后端初始化...
timeout /t 5 /nobreak >nul

:: 启动前端
echo [3/3] 启动前端服务 (端口 8080)...
start "课程助手-前端" cmd /k "cd /d "%ROOT%" && python -m http.server 8080 --directory web"

:: 等待前端启动
timeout /t 2 /nobreak >nul

:: 打开浏览器
echo.
echo  正在打开浏览器...
start http://localhost:8080

echo.
echo  ══════════════════════════════════════
echo   后端: http://localhost:8001
echo   前端: http://localhost:8080
echo   关闭此窗口不影响服务运行
echo  ══════════════════════════════════════
echo.
pause
