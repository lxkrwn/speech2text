@echo off
chcp 65001 >nul
cd /d "%~dp0"
"gigaam-env\Scripts\python.exe" transcribe.py
pause
