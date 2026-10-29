@echo off
cd /d "%~dp0.."
set "PYTHONPATH=%cd%"
call .venv\Scripts\activate.bat
uvicorn app.main:app --app-dir backend --port 8000
