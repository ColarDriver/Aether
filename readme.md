# Aether Harness

Aether Harness is a lightweight agent runtime framework focused on provider abstraction, middleware pipelines, tool calling, and subagent orchestration.

## Project Layout

- `backend/harness/aether`: core runtime, providers, middleware, tools, and tests
- `docs/agent-engine`: design notes and enhancement docs
- `docker`: container deployment files

## Quick Start (Local)

1. Create and activate a Python 3.12 virtual environment.
2. Install the package in editable mode.
3. Configure environment variables.

```bash
cd backend/harness/aether
python3.12 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
cp .env.example .env
```

## Environment Variables (Example)

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `OPENAI_MODEL_NAME`
- `ANTHROPIC_API_KEY`

## Docker Compose

Ready-to-use deployment files are provided in `docker/` (`Dockerfile` + `docker-compose.yaml`).

```bash
cd docker
docker compose up -d
```

Open a shell in the running container:

```bash
docker compose exec aether-harness sh
```

Run test profile in container:

```bash
docker compose --profile tests run --rm aether-tests
```
