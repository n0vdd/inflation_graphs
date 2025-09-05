@echo off
REM Brazil Inflation Analysis Tool - Run Script
REM This script activates the uv-managed virtual environment and runs the analysis

echo.
echo ========================================
echo  Brazil Inflation Analysis Tool
echo ========================================
echo.

REM Check if we're in the right directory
if not exist "main.py" (
    echo Error: main.py not found. Please run this script from the project root directory.
    pause
    exit /b 1
)

REM Run using uv (recommended - manages dependencies automatically)
echo Running with uv (recommended method)...
uv run python main.py %*

REM Alternative: Run with direct Python if uv-managed venv is available
REM .venv\Scripts\python main.py %*

echo.
echo Analysis complete!
pause