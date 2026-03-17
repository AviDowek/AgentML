# AgentML

An intelligent ML engineering platform that helps you build, train, and deploy machine learning models for tabular data. Describe your prediction task in natural language, connect your data, and let AI-powered agents design and run experiments automatically.

## Features

- **Natural Language Problem Description**: Describe your ML problem in plain English
- **AI-Powered Pipeline**: 6-step wizard with Plan Critic validation (Schema Analysis -> Target Recommendation -> Feature Engineering -> Plan Generation -> Plan Validation -> Execution)
- **Multi-Table Data Architecture**: Automatically join multiple data sources with intelligent feature engineering
- **Automated Training**: AutoGluon-based AutoML with cloud support via Modal.com
- **Auto-Improve Iterations**: Automatically iterate on experiments with overfitting protection
- **Model Lifecycle**: Promote models through draft -> candidate -> shadow -> production stages
- **Three-Tier Validation**: Train/Validation/Holdout scoring with holdout as canonical final score
- **Feature Leakage Detection**: Automatic detection of suspicious columns that could cause data leakage
- **Risk-Adjusted Scoring**: Penalties for overfitting, leakage, and "too good to be true" metrics
- **Promotion Guardrails**: Critical risks require explicit override with documented justification
- **Real-Time Collaboration**: Share projects with team members

## Architecture

| Layer | Technology |
|-------|-----------|
| Frontend | React + TypeScript (Vite) |
| Backend | Python + FastAPI |
| Database | PostgreSQL |
| Task Queue | Celery + Redis |
| AutoML | AutoGluon |
| Cloud Training | Modal.com (optional) |
| LLM | OpenAI API compatible (GPT-4, Gemini, etc.) |

## Getting API Keys

AgentML requires an LLM API key to power its AI agents. You'll need at least one:

| Provider | How to get a key |
|----------|-----------------|
| **OpenAI** | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) |
| **Google Gemini** | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) |
| **Any OpenAI-compatible provider** | Set `LLM_API_BASE_URL` to your provider's endpoint |

For cloud training (optional):
- **Modal.com**: Sign up at [modal.com](https://modal.com) and generate tokens under Settings

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Node.js 18+ (for local frontend development)
- Python 3.11+ (for local backend development)

### Running with Docker (Recommended)

1. Clone the repository:
   ```bash
   git clone https://github.com/AviDowek/AgentML.git
   cd AgentML
   ```

2. Copy and configure environment variables:
   ```bash
   cp backend/.env.example backend/.env
   ```
   Open `backend/.env` and set at minimum:
   - `LLM_API_KEY` — your OpenAI (or compatible) API key
   - `SECRET_KEY` — generate one with: `python -c "import secrets; print(secrets.token_urlsafe(32))"`

3. Start all services:
   ```bash
   docker-compose up --build
   ```

4. Open the app:
   - **Frontend**: http://localhost:5173
   - **Backend API**: http://localhost:8000
   - **API Docs**: http://localhost:8000/docs

### Running Locally (Development)

#### Backend

```bash
cd backend
python -m venv .venv

# Windows
.\.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env  # Then edit .env with your API keys

# Start PostgreSQL and Redis (Docker required)
docker-compose up postgres redis -d

# Run database migrations
alembic upgrade head

# Start the backend server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

#### Frontend

```bash
cd frontend
npm install
npm run dev
```

#### Celery Worker (for background training tasks)

```bash
cd backend
celery -A app.tasks.celery_app worker --loglevel=info --pool=solo
```

## How It Works

1. **Create a project** and upload your tabular data (CSV, Excel, Parquet)
2. **Describe your prediction task** in plain English (e.g., "predict whether a customer will churn")
3. **AI agents analyze your data** — they inspect schemas, recommend targets, engineer features, and design an experiment plan
4. **A Plan Critic agent reviews the plan** for issues like data leakage or unrealistic targets
5. **AutoGluon trains models** locally or on Modal.com cloud GPUs
6. **Results agents interpret performance**, flag overfitting, and suggest improvements
7. **Auto-improve** iterates automatically, with each iteration building on prior learnings
8. **Promote your best model** through draft -> candidate -> shadow -> production stages

## Configuration

All configuration is via environment variables. See [`backend/.env.example`](backend/.env.example) for the full list.

| Variable | Required | Description |
|----------|----------|-------------|
| `DATABASE_URL` | Yes | PostgreSQL connection string |
| `REDIS_URL` | Yes | Redis connection string |
| `LLM_API_KEY` | Yes | OpenAI API key (or compatible provider) |
| `SECRET_KEY` | Yes | JWT signing key (generate a random one) |
| `LLM_API_BASE_URL` | No | LLM API endpoint (defaults to OpenAI) |
| `MODAL_TOKEN_ID` | No | Modal.com token for cloud training |
| `MODAL_TOKEN_SECRET` | No | Modal.com secret for cloud training |
| `API_KEY_ENCRYPTION_KEY` | Recommended | Fernet key for encrypting stored API keys |
| `GOOGLE_CLIENT_ID` | No | Google OAuth client ID |
| `SMTP_HOST` | No | SMTP server for email notifications |

## Project Structure

```
AgentML/
├── backend/
│   ├── app/
│   │   ├── api/                  # REST API endpoints
│   │   ├── core/                 # Config, database, security
│   │   ├── models/               # SQLAlchemy ORM models
│   │   ├── schemas/              # Pydantic request/response schemas
│   │   ├── services/             # Business logic
│   │   │   ├── agents/           # 21 LLM agent implementations
│   │   │   │   ├── base.py       # BaseAgent abstract class
│   │   │   │   ├── registry.py   # Agent type -> class mapping
│   │   │   │   ├── setup/        # Setup pipeline (6 agents)
│   │   │   │   ├── results/      # Results pipeline (2 agents)
│   │   │   │   ├── data_architect/  # Multi-table pipeline (4 agents)
│   │   │   │   ├── improvement/  # Auto-improve pipeline (6 agents)
│   │   │   │   └── standalone/   # Independent agents (3 agents)
│   │   │   ├── llm/              # LLM client & prompt templates
│   │   │   ├── agent_executor.py # Pipeline orchestration
│   │   │   └── risk_scoring.py   # Overfitting & leakage detection
│   │   └── tasks/                # Celery background tasks
│   ├── alembic/                  # Database migrations
│   └── tests/
├── frontend/
│   ├── src/
│   │   ├── components/           # Reusable UI components
│   │   ├── pages/                # Route pages
│   │   ├── contexts/             # React context (auth, etc.)
│   │   └── api/                  # API client
│   └── package.json
├── docker-compose.yml
├── ARCHITECTURE.md
├── AGENTS.md
└── README.md
```

## Deploying to Production

### Railway (Recommended)

Railway handles PostgreSQL, Redis, HTTPS, and auto-deploy from GitHub.

1. **Create a Railway project** at [railway.app](https://railway.app)

2. **Add services** from the Railway dashboard:
   - **PostgreSQL** — add from the "New" menu (plugin)
   - **Redis** — add from the "New" menu (plugin)
   - **Backend** — "New" > "GitHub Repo" > select `AgentML`, set root directory to `backend`
   - **Celery Worker** — "New" > "GitHub Repo" > select `AgentML`, set root directory to `backend`, override start command:
     ```
     celery -A app.core.celery_app worker --loglevel=info --concurrency=4 -Q celery
     ```
   - **Frontend** — "New" > "GitHub Repo" > select `AgentML`, set root directory to `frontend`

3. **Set environment variables** on the Backend service:
   ```
   DATABASE_URL        → ${{Postgres.DATABASE_URL}}  (Railway provides this)
   REDIS_URL           → ${{Redis.REDIS_URL}}        (Railway provides this)
   CELERY_BROKER_URL   → ${{Redis.REDIS_URL}}
   CELERY_RESULT_BACKEND → ${{Redis.REDIS_URL}}
   SECRET_KEY          → (generate: python -c "import secrets; print(secrets.token_urlsafe(32))")
   LLM_API_KEY         → (your OpenAI key)
   CORS_ORIGINS        → ["https://your-frontend.up.railway.app"]
   FRONTEND_URL        → https://your-frontend.up.railway.app
   API_KEY_ENCRYPTION_KEY → (generate: python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())")
   DEBUG               → false
   ```
   Copy the same DB/Redis vars to the Celery Worker service.

4. **Set build args** on the Frontend service:
   ```
   VITE_API_URL → https://your-backend.up.railway.app
   ```

5. **Deploy** — Railway auto-deploys on push to `main`.

### Self-Hosted (Docker Compose)

For VPS or bare-metal deployment:

```bash
# 1. Clone and configure
git clone https://github.com/AviDowek/AgentML.git
cd AgentML
cp .env.example .env
cp backend/.env.example backend/.env

# 2. Edit both .env files with production values
#    - Set POSTGRES_PASSWORD, REDIS_PASSWORD in root .env
#    - Set SECRET_KEY, LLM_API_KEY, API_KEY_ENCRYPTION_KEY in backend/.env
#    - Set CORS_ORIGINS to your domain

# 3. Start everything
docker-compose -f docker-compose.prod.yml up --build -d
```

The app will be available on port 80 (frontend) and 8000 (backend API).
For HTTPS, put nginx or Caddy in front as a reverse proxy.

## Testing

```bash
# Backend tests
cd backend
pytest

# Frontend tests
cd frontend
npm test
```

## Documentation

- [Architecture](ARCHITECTURE.md) — System design, database schema, and API endpoints
- [Agent Guidelines](AGENTS.md) — How the 21 LLM agents work and how to add new ones
- [Capabilities Guide](capabilities.md) — Complete feature documentation with examples
- [Runbook](RUNBOOK.md) — Operational procedures and troubleshooting
- [Testing](testing.md) — Test procedures and verification steps
- [Contributing](CONTRIBUTING.md) — How to contribute
- [Security](SECURITY.md) — Security policy and responsible disclosure

## License

MIT License — see [LICENSE](LICENSE) for details.
