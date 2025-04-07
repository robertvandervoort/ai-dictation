@echo off
setlocal enabledelayedexpansion

echo == AI Dictation Launcher (Windows) ==

REM Check if Python is installed
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Error: Python not found. Please install Python 3.8 or newer.
    pause
    exit /b 1
)

REM Check Python version
for /f "tokens=2" %%I in ('python --version 2^>^&1') do set PYTHON_VERSION=%%I
echo Found Python %PYTHON_VERSION%

REM Set up virtual environment path
set VENV_DIR=venv

REM Check if venv exists
if not exist %VENV_DIR% (
    echo Creating virtual environment...
    python -m venv %VENV_DIR%
    if !ERRORLEVEL! neq 0 (
        echo Error: Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo Virtual environment created successfully.
)

REM Activate venv
echo Activating virtual environment...
call %VENV_DIR%\Scripts\activate

REM Update pip to latest version FIRST
echo Updating pip to the latest version...
python -m pip install --upgrade pip
if !ERRORLEVEL! neq 0 (
    echo Warning: Failed to update pip. Continuing with existing version.
)

REM Install build tools that might be needed SECOND
echo Installing build tools...
pip install setuptools wheel cython
if !ERRORLEVEL! neq 0 (
    echo Warning: Failed to install build tools. Some packages might fail to install.
)

REM Install all dependencies from requirements.txt
echo Installing requirements...
pip install -r requirements.txt --no-cache-dir
if !ERRORLEVEL! neq 0 (
    echo Warning: Some packages failed to install. Trying with --no-build-isolation flag...
    pip install -r requirements.txt --no-cache-dir --no-build-isolation
    if !ERRORLEVEL! neq 0 (
        echo Error: Failed to install requirements.
        pause
        exit /b 1
    )
)
echo Requirements installed successfully.

REM Check if ffmpeg is available for Whisper
where ffmpeg >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Warning: ffmpeg not found. Some audio processing functionality may be limited.
    echo You can install ffmpeg from https://ffmpeg.org/download.html
) else (
    echo ffmpeg is available.
)

REM Run the application
echo Starting AI Dictation application...
streamlit run dictate.py

REM Deactivate venv on exit
call %VENV_DIR%\Scripts\deactivate

pause
exit /b 0 