# Runbook - Local Development

## Prerequisites

- Docker & Docker Compose
- Node.js >= 18 (for frontend)
- Python 3.11+ (for backend)
- PostgreSQL 14+ (or use Docker)
- Redis 6+ (or use Docker)

## Quick Start (Docker)

Start all services in development mode:

```powershell
docker-compose -f docker-compose.local.yml up --build
```

## Manual Setup

### 1. Database & Redis

Start PostgreSQL and Redis using Docker:

```powershell
docker-compose up postgres redis -d
```

Or install locally and ensure they're running.

### 2. Backend Setup

```powershell
cd backend

# Create virtual environment
python -m venv .venv

# Activate (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Activate (Windows cmd)
.\.venv\Scripts\activate.bat

# Activate (macOS/Linux)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp .env.example .env
# Edit .env with your settings (API keys, database URL, etc.)

# Run database migrations
alembic upgrade head

# Start backend server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Celery Worker

In a separate terminal (with venv activated):

```powershell
cd backend
celery -A app.tasks.celery_app worker --loglevel=info --pool=solo
```

Note: Use `--pool=solo` on Windows.

### 4. Frontend Setup

```powershell
cd frontend
npm install
npm run dev
```

## Access Points

| Service | URL |
|---------|-----|
| Frontend | http://localhost:5173 |
| Backend API | http://localhost:8000 |
| API Docs (Swagger) | http://localhost:8000/docs |
| API Docs (ReDoc) | http://localhost:8000/redoc |

## Environment Variables

See `backend/.env.example` for all required settings:

- `DATABASE_URL` - PostgreSQL connection string
- `REDIS_URL` - Redis connection string
- `LLM_API_KEY` - OpenAI API key (or compatible)
- `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` - For cloud training (optional)

## Common Tasks

### Reset Database

```powershell
cd backend
alembic downgrade base
alembic upgrade head
```

### Run Tests

```powershell
# Backend
cd backend
pytest

# Frontend
cd frontend
npm test
```

### Check Celery Task Status

```powershell
celery -A app.tasks.celery_app inspect active
```

## Troubleshooting

### Port already in use

```powershell
# Find process using port 8000
netstat -ano | findstr :8000

# Kill the process (replace PID)
taskkill /PID <pid> /F
```

### Database connection failed

1. Check PostgreSQL is running
2. Verify DATABASE_URL in .env
3. Ensure database exists: `createdb agentic_ml`

### Celery worker not processing tasks

1. Check Redis is running
2. Verify CELERY_BROKER_URL matches REDIS_URL
3. Restart the worker

### LLM API errors

1. Verify LLM_API_KEY is set correctly
2. Check API quota/billing status
3. Try a different model if rate limited
