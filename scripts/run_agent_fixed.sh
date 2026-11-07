#!/usr/bin/env bash
# 以固定流水线模式启动 Agent:不配置语言模型,流程为
# ingest → select_model → forecast → baseline → report。
# 用于 GIFT-Eval 这类需要快速、确定性重复的批量评测场景。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
set -a
# shellcheck disable=SC1091
source "$ROOT/.env"
set +a
# 注意:agent.settings 会用 load_dotenv 重新载入 .env,而 load_dotenv 默认不覆盖
# 已存在的变量。因此这里必须置空,而不是 unset——unset 之后键不存在,会被 .env 填回。
export LLM_BASE_URL="" LLM_API_KEY="" LLM_MODEL=""
PORT="${1:-8001}"
echo "固定流水线模式启动 Agent: http://127.0.0.1:${PORT}"
exec "$ROOT/.venv-agent/bin/python" -m uvicorn agent.app:app --host 127.0.0.1 --port "$PORT"
