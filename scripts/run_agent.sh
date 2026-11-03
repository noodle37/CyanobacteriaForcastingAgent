#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
set -a
# shellcheck disable=SC1091
source "$ROOT/.env"
set +a
exec "$ROOT/.venv-agent/bin/python" -m uvicorn agent.app:app --host "${AGENT_HOST:-127.0.0.1}" --port "${AGENT_PORT:-8000}"
