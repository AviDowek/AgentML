@echo off
REM Start script for Agentic ML Platform services
REM This starts both FastAPI and Celery worker

echo Starting Agentic ML Platform Services...
echo.
echo Make sure Redis is running: docker run -d -p 6379:6379 redis:alpine
echo.

REM Activate virtual environment
call .venv311\Scripts\activate.bat

REM Start Celery worker in background
echo Starting Celery worker...
start "Celery Worker" cmd /k ".venv311\Scripts\celery.exe -A app.core.celery_app worker --loglevel=info --pool=solo"

REM Wait a moment for Celery to start
timeout /t 3 /nobreak >nul

REM Start FastAPI server
echo Starting FastAPI server...
echo.
echo ============================================
echo FastAPI: http://localhost:8000
echo API Docs: http://localhost:8000/docs
echo ============================================
echo.
.venv311\Scripts\uvicorn.exe app.main:app --reload --host 0.0.0.0 --port 8000
