# Brazil Inflation Analysis Tool - PowerShell Run Script
# This script provides multiple ways to run the analysis tool

Write-Host ""
Write-Host "========================================"
Write-Host " Brazil Inflation Analysis Tool"
Write-Host "========================================"
Write-Host ""

# Check if we're in the right directory
if (-not (Test-Path "main.py")) {
    Write-Host "Error: main.py not found. Please run this script from the project root directory." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Function to run with uv (recommended)
function Run-WithUv {
    Write-Host "Running with uv (recommended method)..." -ForegroundColor Green
    & uv run python main.py $args
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Analysis completed successfully!" -ForegroundColor Green
    } else {
        Write-Host "Analysis failed with exit code: $LASTEXITCODE" -ForegroundColor Red
    }
}

# Function to run with virtual environment Python
function Run-WithVenv {
    if (Test-Path ".venv\Scripts\python.exe") {
        Write-Host "Running with virtual environment Python..." -ForegroundColor Yellow
        & .\.venv\Scripts\python main.py $args
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "Analysis completed successfully!" -ForegroundColor Green
        } else {
            Write-Host "Analysis failed with exit code: $LASTEXITCODE" -ForegroundColor Red
        }
    } else {
        Write-Host "Virtual environment not found. Please run 'uv sync' first." -ForegroundColor Red
        return $false
    }
}

# Main execution
try {
    # Try uv first (recommended)
    Run-WithUv $args
}
catch {
    Write-Host "UV failed, trying virtual environment..." -ForegroundColor Yellow
    if (-not (Run-WithVenv $args)) {
        Write-Host "Both methods failed. Please check your installation." -ForegroundColor Red
        exit 1
    }
}

Write-Host ""
Read-Host "Press Enter to exit"