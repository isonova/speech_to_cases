@echo off
REM cleanup.cmd - Clean Docker & ML caches, rebuild image, warm light models
REM Run this script from an elevated (Administrator) CMD prompt.

echo.
echo ============================
echo ML Environment Cleanup Script
echo ============================
echo.

REM --- CONFIG (edit if your paths differ) ---
SET HF_CACHE=C:\Users\isode\hf_cache
SET PROJECT_DIR=%cd%
REM Project export (uploaded file) for traceability:
SET PROJECT_EXPORT=file:///mnt/data/AIPRM-export-chatgpt-thread_Call-center-case-segmentation_2025-11-20T23_27_47.292Z.md

echo Project dir: %PROJECT_DIR%
echo HuggingFace cache dir: %HF_CACHE%
echo Project export: %PROJECT_EXPORT%
echo.

pause

echo.
echo 1) Prune Docker build cache, dangling images, containers, volumes
echo (This will remove unused images, builders and volumes.)
docker builder prune -af
docker container prune -f
docker image prune -af
docker volume prune -f
docker system prune -af

echo.
echo 2) Remove and recreate HuggingFace cache (fast model re-download)
if exist "%HF_CACHE%" (
    echo Removing %HF_CACHE% ...
    rmdir /S /Q "%HF_CACHE%"
) else (
    echo %HF_CACHE% not found, creating...
)
mkdir "%HF_CACHE%"

echo.
echo 3) Remove user-level HuggingFace & pip caches (if present)
if exist "%USERPROFILE%\.cache\huggingface" (
    rmdir /S /Q "%USERPROFILE%\.cache\huggingface"
)
if exist "%USERPROFILE%\.cache\pip" (
    rmdir /S /Q "%USERPROFILE%\.cache\pip"
)
if exist "%LOCALAPPDATA%\Temp\pip-*" (
    rmdir /S /Q "%LOCALAPPDATA%\Temp\pip-*"
)

echo.
echo 4) Remove Python __pycache__ folders under the project
echo (This will recursively remove __pycache__ dirs from the current project directory.)
for /f "delims=" %%d in ('dir /s /b /ad __pycache__ 2^>nul') do (
    echo Removing %%d
    rd /s /q "%%d"
)

echo.
echo 5) Optional: Shutdown WSL2 to apply .wslconfig changes (if you use WSL)
echo If you changed %USERPROFILE%\.wslconfig you should run:
echo     wsl --shutdown
echo (This script will run it automatically if WSL exists.)
wsl --shutdown >nul 2>&1 || echo "WSL not present or failed to shutdown (ok)."

echo.
echo 6) Rebuild Docker image (no-cache so it pulls fresh layers)
echo (This will rebuild whisper-pipeline; expect it to take a few minutes.)
docker build --no-cache -t whisper-pipeline .

echo.
echo 7) Warm lightweight models (optional but recommended)
echo Run these commands after the build to download small CPU models (Whisper tiny/base, MiniLM, DistilBART).
echo You can uncomment the lines below or run them manually.

REM docker run --rm -v "%PROJECT_DIR%:/app" -v "%HF_CACHE%:/root/.cache/huggingface" whisper-pipeline python summarize_cases.py cases.json
REM docker run --rm -v "%PROJECT_DIR%:/app" -v "%HF_CACHE%:/root/.cache/huggingface" whisper-pipeline python pipeline.py sample_call.wav

echo.
echo DONE.
echo - You successfully pruned Docker cache, cleared HF & Python caches, removed __pycache__.
echo - Rebuilt image: whisper-pipeline
echo - To warm models, run the commented docker run lines above (uncomment then run) or run them manually.
echo.
echo Project export (traceability): %PROJECT_EXPORT%
echo.

pause
exit /b 0
