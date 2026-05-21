@echo off
setlocal

title Research Swarm
cd /d "%~dp0"

echo.
echo  ================================================================
echo   Research Swarm  ^|  LangGraph + LlamaIndex + Streamlit
echo  ================================================================
echo.

:: ---------- .env check -----------------------------------------------------
if not exist ".env" (
    if exist ".env.example" (
        echo  Copying .env.example -^> .env ...
        copy /y ".env.example" ".env" >nul
        echo  Done.  Open .env, fill in your API keys, then run this again.
    ) else (
        echo  ERROR: .env file missing. Create one from .env.example.
    )
    echo.
    pause
    exit /b 1
)
echo  [OK] .env found

:: ---------- poetry check ----------------------------------------------------
where poetry >nul 2>&1
if errorlevel 1 (
    echo  ERROR: poetry not found.
    echo  Install it from https://python-poetry.org/docs/#installation
    echo.
    pause
    exit /b 1
)
echo  [OK] poetry found

:: ---------- install deps ----------------------------------------------------
echo  [..] Installing dependencies ...
call poetry install --quiet
if errorlevel 1 (
    echo  ERROR: poetry install failed.
    pause
    exit /b 1
)
echo  [OK] Dependencies ready
echo.

:: ---------- Ollama (optional) -----------------------------------------------
where ollama >nul 2>&1
if not errorlevel 1 (
    echo  [..] Ollama found - starting backend in a new window ...
    start "Ollama Backend" cmd /k "ollama serve"
    timeout /t 3 /nobreak >nul
    echo  [OK] Ollama window opened
) else (
    echo  [--] Ollama not installed - skipping local backend
)
echo.

:: ---------- Streamlit -------------------------------------------------------
echo  ================================================================
echo   Open your browser at  http://localhost:8501
echo   Press Ctrl+C here to stop the app
echo  ================================================================
echo.

call poetry run streamlit run app.py --server.port 8501 --server.headless false

echo.
echo  App stopped.  Close the Ollama window too if it was opened.
pause
endlocal
