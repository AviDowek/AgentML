# Contributing to AgentML

Thank you for your interest in contributing to AgentML! This guide will help you get started.

## Development Setup

### Prerequisites

- Python 3.11+
- Node.js 18+
- Docker and Docker Compose
- Git

### Local Environment

```bash
# Clone the repository
git clone https://github.com/your-username/AgentML.git
cd AgentML

# Backend
cd backend
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env       # Fill in your API keys

# Frontend
cd ../frontend
npm install

# Start infrastructure
docker-compose up -d postgres redis

# Run migrations
cd ../backend
alembic upgrade head
```

### Running Tests

```bash
# Backend
cd backend
pytest

# Frontend
cd frontend
npm test
```

## How to Contribute

### Reporting Issues

- Search existing issues before creating a new one
- Include steps to reproduce, expected behavior, and actual behavior
- Include your environment details (OS, Python version, Node version)

### Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Make your changes
4. Run tests and ensure they pass
5. Commit with a clear message
6. Push to your fork and open a pull request

### PR Guidelines

- Keep PRs focused on a single change
- Include tests for new functionality
- Update documentation if your change affects user-facing behavior
- Follow the existing code style

## Code Style

### Python (Backend)

- Format with [Black](https://github.com/psf/black) (line length 88)
- Lint with [Ruff](https://github.com/astral-sh/ruff)
- Type hints encouraged on public APIs

### TypeScript (Frontend)

- Lint with ESLint (config in `eslint.config.js`)
- Use TypeScript strict mode

## Project Structure

See [ARCHITECTURE.md](ARCHITECTURE.md) for a detailed overview of the codebase.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
