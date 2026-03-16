# PowerShell script to start Agentic ML Platform services
# This starts both FastAPI and Celery worker

Write-Host "Starting Agentic ML Platform Services..." -ForegroundColor Green
Write-Host ""

# Check if Redis is running
Write-Host "Checking Redis connection..." -ForegroundColor Yellow
try {
    $redis = redis-cli ping 2>$null
    if ($redis -ne "PONG") {
        throw "Redis not responding"
    }
    Write-Host "Redis is running." -ForegroundColor Green
} catch {
    Write-Host "WARNING: Redis does not appear to be running!" -ForegroundColor Red
    Write-Host "Please start Redis before continuing." -ForegroundColor Red
    Write-Host "You can use: redis-server" -ForegroundColor Yellow
    Write-Host "Or start Redis via Docker: docker run -d -p 6379:6379 redis:alpine" -ForegroundColor Yellow
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Host ""

# Navigate to backend directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

# Activate virtual environment
& .\.venv311\Scripts\Activate.ps1

# Start Celery worker in a new window
Write-Host "Starting Celery worker..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$scriptDir'; & .\.venv311\Scripts\Activate.ps1; celery -A app.core.celery_app worker --loglevel=info --pool=solo"

# Wait a moment for Celery to start
Start-Sleep -Seconds 3

# Start FastAPI server
Write-Host "Starting FastAPI server..." -ForegroundColor Yellow
Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "FastAPI: http://localhost:8000" -ForegroundColor Cyan
Write-Host "API Docs: http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
