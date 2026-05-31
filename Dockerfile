FROM node:22-bookworm-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    UV_PYTHON=3.13 \
    UV_LINK_MODE=copy

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN uv python install 3.13

COPY .python-version pyproject.toml uv.lock ./

COPY koja_diffuser ./koja_diffuser

RUN uv sync --locked

COPY frontend ./frontend

EXPOSE 7860

CMD ["bash", "-lc", "\
    set -euo pipefail; \
    node frontend/server.js & \
    node_pid=$!; \
    uv run uvicorn koja_diffuser.runtime.api:app --host 127.0.0.1 --port 8000 & \
    api_pid=$!; \
    trap 'kill $node_pid $api_pid 2>/dev/null || true' INT TERM EXIT; \
    wait -n $node_pid $api_pid; \
    status=$?; \
    kill $node_pid $api_pid 2>/dev/null || true; \
    wait 2>/dev/null || true; \
    exit $status \
    "]