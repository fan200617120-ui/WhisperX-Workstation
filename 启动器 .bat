@echo off
chcp 65001 >nul
title 语音字幕工作站 凡哥制作
color 0B

cls

echo ========================================
echo        轻舟 AI・LightShip AI
echo ========================================
echo.
echo       whisperX语音字幕工作站
echo.
echo ========================================
echo       轻舟渡万境，一智载千寻!
echo ========================================
echo.

set "PROJECT_ROOT=%~dp0"
set "PROJECT_ROOT=%PROJECT_ROOT:~0,-1%"
set "PYTHON_DIR=%PROJECT_ROOT%\python_embeded"
set "PYTHON_EXE=%PYTHON_DIR%\python.exe"
set "INDEX_SCRIPT=%PROJECT_ROOT%\Index_Public_release.py"

if not exist "%PYTHON_EXE%" (
    echo [错误] 未找到嵌入版 Python，请确保 python_embeded 目录完整。
    pause
    exit /b 1
)

if not exist "%INDEX_SCRIPT%" (
    echo [错误] 未找到主界面脚本 Index_Public_release.py，请重新下载完整包。
    pause
    exit /b 1
)

echo [信息] 正在启动主界面...
"%PYTHON_EXE%" "%INDEX_SCRIPT%"
pause
