# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability, please report it responsibly:

1. **Do not** open a public issue
2. Email the maintainers with a description of the vulnerability
3. Include steps to reproduce if possible
4. Allow reasonable time for a fix before public disclosure

## Security Considerations

### API Keys

AgentML connects to external LLM providers (OpenAI, Google, etc.) and optionally to Modal.com for cloud training. These credentials are sensitive:

- **Never commit `.env` files** to version control
- Store API keys only in `backend/.env` (which is gitignored)
- Rotate keys if you suspect they have been exposed
- Use the minimum required API key permissions

### Default Credentials

The `docker-compose.yml` ships with default PostgreSQL credentials (`postgres:postgres`) for local development convenience. **For any non-local deployment:**

- Change `POSTGRES_PASSWORD` via environment variable
- Configure Redis authentication
- Set a strong `SECRET_KEY` for JWT signing
- Use TLS for all external connections

### Authentication

- JWT tokens are used for session management
- Google OAuth is supported for SSO
- Tokens expire after 24 hours by default

### Data Privacy

- Uploaded datasets are stored locally in the `uploads/` directory
- Model artifacts are stored in `artifacts/`
- Both directories are gitignored and should not be committed
- Consider encrypting sensitive datasets at rest in production

## Supported Versions

| Version | Supported |
|---------|-----------|
| Latest  | Yes       |
