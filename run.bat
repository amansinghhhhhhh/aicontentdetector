@echo off
setlocal
cd /d "%~dp0"

echo ============================================
echo   AI Content Detector - local launcher
echo ============================================

if not exist ".venv\Scripts\python.exe" (
  echo [1/4] First run: creating venv + installing packages, this takes a while...
  python -m venv .venv
  call .venv\Scripts\activate.bat
  python -m pip install -q --upgrade pip
  python -m pip install -q -r backend\requirements.txt
  python -m pip install -q torch --index-url https://download.pytorch.org/whl/cpu
  python -m pip install -q -r ml\requirements.txt
) else (
  call .venv\Scripts\activate.bat
  echo [1/4] Environment ready.
)

if not exist "models\detector-onnx\model_fp32.onnx" (
  echo [2/4] Exporting ONNX model - takes a few minutes, one time only...
  python -m ml.export_onnx
) else (
  echo [2/4] ONNX model ready.
)

if not exist "frontend\node_modules" (
  echo [3/4] Installing frontend packages...
  pushd frontend
  call npm ci
  popd
) else (
  echo [3/4] Frontend packages ready.
)

echo [4/4] Starting app...
start "AI Detector - Backend (port 8000)" cmd /k ""%~dp0backend\start_backend.bat""
timeout /t 3 /nobreak >nul
start "AI Detector - Frontend (port 5173)" cmd /k ""%~dp0frontend\start_frontend.bat""
timeout /t 6 /nobreak >nul
start "" http://localhost:5173

echo.
echo   App:  http://localhost:5173
echo   API:  http://localhost:8000/api/health
echo   Band karne ke liye: dono command windows band karein (X dabayein).
pause
