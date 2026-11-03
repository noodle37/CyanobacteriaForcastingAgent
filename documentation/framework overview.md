# Framework overview

This repository is a local time-series agent. A FastAPI process decides what to do. A separate MCP process owns the forecasting models. The agent never imports Chronos or TimesFM. It calls them as tools.

Two virtual environments keep those stacks apart:

| Environment | Requirements | Process |
| --- | --- | --- |
| `.venv-agent` | `requirements-agent.txt` | `scripts/run_agent.sh` → `agent.app:app` on `127.0.0.1:8000` |
| `.venv-tsfm` | `requirements-tsfm.txt` | `scripts/run_tsfm.sh` → `python -m tsfm_server` on `127.0.0.1:8765` |

Both processes read the repo-root `.env`. Paths and ports can be overridden there. `.env.example` lists the variables. Checkpoints such as `CHRONOS2_CHECKPOINT` and `TIMESFM3_CHECKPOINT` are expanded only when `configs/models.yaml` is loaded.

## Request path

```
HTTP client
    → agent.app  (FastAPI)
        → agent.graph.build_graph
            → ReAct agent          if LLM_BASE_URL, LLM_API_KEY, and LLM_MODEL are all set
            → fixed pipeline       otherwise
        → agent.mcp_client.MCPToolClient
            → tsfm_server MCP tools
                → ModelRuntime
                    → one adapter
```

`agent.settings` builds the process-wide `LLM` client, loads `Policy` from `configs/permissions.yaml` (or `AGENT_POLICY`), and sets `TSFM_MCP_URL` (default `http://127.0.0.1:8765/mcp`).

## HTTP API

`agent/app.py` compiles one graph at import time and keeps it for the life of the process.

| Method | Path | Behavior |
| --- | --- | --- |
| `GET` | `/health` | Returns `status`, whether the LLM is configured, and the MCP URL. |
| `POST` | `/v1/runs` | Runs one task to completion, or stops on an approval interrupt. |
| `POST` | `/v1/runs/stream` | Same run, streamed as server-sent events. Ends with `data: [DONE]`. |
| `POST` | `/v1/runs/{thread_id}/resume` | Continues an interrupted run with `{"approved": true\|false}`. |

`RunRequest` fields:

- `user_request`: natural-language task.
- `series`: the history, as a list of numbers.
- `horizon`: steps to forecast. Default `3`. Policy allows `1`–`64`.
- `freq`: pandas-style frequency. Default `"D"`. Used for season length (`D`→7, `H`→24, `W`→1, `M`→12).
- `actual`: optional future values. If present, the agent can score the forecast.
- `model_name`: optional model id hint.
- `thread_id`: optional. Omitted ids are random. The checkpointer is `MemorySaver`, so memory lives in this process only. A new id starts a new transcript.

The public response is built by `agent/present.py`. It includes `model_name`, `model_reason`, the working `series`, `forecast`, `baseline`, `metrics`, `baseline_metrics`, `report`, `prepared`, `tool_trace`, and `errors`. On a finished ReAct run, `report` is the last assistant message that did not call a tool. If that text is missing, `agent/workers.py` fills in a template from the ledger. While a run is paused for approval, `status` is `interrupt` and `report` is left empty rather than claiming that forecasting failed.

## Two graphs

`agent/graph.py` `build_graph` chooses the graph once, from whether a chat model exists.

### ReAct graph

Used when the three LLM environment variables are set. `agent/react_graph.py` calls `langchain.agents.create_agent`.

- System prompt: `prompts/react.md`.
- Tools: the list returned by `agent/tools.py` `make_tools`.
- State: `ReactLedger` in `agent/state.py`.
- Checkpointer: in-memory, keyed by `thread_id`.

One assistant message may contain several tool calls. Those calls run together and do not see each other's results. The model is called again after all of them return. Dependent steps (analyze, then rewrite `series`, then forecast, then score, then edit code) are later turns.

`run_config` sets LangGraph `recursion_limit` to `max_react_steps * 2 + 4`. With the default `max_react_steps: 12`, that limit is 28. This stops a tool loop. It is not the token window.

Before each model call, two LangChain middlewares run. Thresholds are fractions of `context_window_tokens` (default `32768`). That number must match the deployed chat model. The code does not read a provider context-length error and then compact.

| Middleware | When | What it does |
| --- | --- | --- |
| `SummarizationMiddleware` | About 75% of the window | Summarizes older messages with the same chat model and writes the summary into the checkpoint. The most recent ~30% of tokens stay verbatim. A tool call is not split from its tool result. |
| `ContextEditingMiddleware` / `ClearToolUsesEdit` | About 60% of the window | On the request sent to the model, replaces older tool results with a placeholder and keeps the latest 3. The checkpoint copy of those results is not deleted by this edit. |

The first user message is built by `task_message`. It sends the request, horizon, frequency, length, the first 8 points, the last 16 points, `actual`, and `model_hint`. The full series stays in graph state. `python` and `forecast` read that state. `forecast` uses it when the model omits `values`.

### Fixed pipeline

Used when the LLM is not configured. `agent/workers.py` supplies five nodes on a `TaskLedger`. Prompts `prompts/select_model.md` and `prompts/report.md` are unused, because the LLM client is disabled. Model choice is then: the request hint if it is allowed, otherwise `chronos2`, otherwise the first allowed id. Phrases such as "只要基线" route straight to `seasonal_naive`.

```
ingest → select_model → forecast → baseline → report
```

`ingest` rejects a bad series or horizon and jumps to `report`. `select_model` calls `list_models` once. `forecast` calls the chosen model once. `baseline` always calls `seasonal_naive` unless the run already failed. `report` calls `evaluate` when `actual` is present, then writes the template report.

If `require_approval_models` lists a model, `forecast` calls LangGraph `interrupt` before the MCP call. The HTTP run returns `status: interrupt`. Resume with `approved: true` runs the forecast. `approved: false` records "预测调用未获批准" and does not call the model. The same interrupt exists on the ReAct `forecast` tool. The default policy list is empty, so approval is off until configured.

## Agent modules

| Module | Role |
| --- | --- |
| `agent/app.py` | FastAPI routes, SSE, resume. |
| `agent/settings.py` | `.env`, policy path, prompt loader. |
| `agent/graph.py` | Chooses ReAct or the fixed pipeline. Builds the first human message and the recursion limit. |
| `agent/react_graph.py` | `create_agent` plus the two context middlewares. |
| `agent/workers.py` | Fixed-pipeline nodes and the fallback report template. `season_of` maps `freq` to a seasonal period. |
| `agent/tools.py` | ReAct tools. Each tool checks the policy allow-list, calls MCP or the Python sandbox, and returns a LangGraph `Command` that updates `ReactLedger`. |
| `agent/hooks.py` | `Policy`, series checks, and the forecast context tail. |
| `agent/state.py` | `TaskLedger` for the pipeline, `ReactLedger` for ReAct. `tool_trace` and `errors` append. Other fields replace. |
| `agent/llm.py` | One-shot Chat Completions client for the unused pipeline prompts, and `as_chat_model()`. |
| `agent/chat_model.py` | `OpenAICompatibleChat`. Implements `bind_tools` so `create_agent` can run a tool loop against any OpenAI-compatible endpoint. Temperature is 0. |
| `agent/mcp_client.py` | `MultiServerMCPClient` over streamable HTTP. `call(name, args)` looks the tool up by name and requires a JSON object back. |
| `agent/present.py` | Maps graph state to the HTTP body. |
| `agent/python_exec.py` | Sandbox used by the `python` tool. |
| `agent/python_worker.py` | Subprocess entry. Reads a JSON job on stdin and writes the result file given on `argv`. |

### ReAct tools

All five names must also appear in `allowed_tools`. The TSFM server exposes only `list_models`, `forecast`, `evaluate`, and `reload_models`. `python` runs in the agent process.

| Tool | Effect |
| --- | --- |
| `list_models` | MCP `list_models`. Stores the cards on `catalog`. |
| `forecast` | Checks model allow-list, horizon, and numeric range, then cuts the series to the context tail and calls MCP `forecast`. `seasonal_naive` is stored as `baseline`. Any other model is stored as `forecast` and becomes `model_name`. |
| `evaluate` | Scores `forecast` or `baseline` against `actual` through MCP `evaluate`. |
| `reload_models` | Asks the TSFM process to reimport adapters and drop cached weights. |
| `python` | Runs model-written code in a child interpreter. |

`forecast` arguments are `model`, optional `horizon`, optional `freq`, and optional `values`. Omitted `values` means the current ledger series, including any series written back by `python`.

`python` injects `series`, `horizon`, `freq`, `actual`, `forecast`, `baseline`, `metrics`, and `result`. Assigning `series` or `horizon` updates the ledger when the new values still pass `check_request`. `read_file` can read `tsfm_server/`, `shared/`, `prompts/`, and `configs/`. `write_file` can write only `tsfm_server/` and `shared/`, and only `.py`, `.yaml`, `.yml`, `.json`, `.md`, and `.txt`. After a successful write, the tool calls `reload_models` itself. Imports are limited to a small analysis set (`math`, `statistics`, `json`, `numpy`, `pandas`, and a few others). `os`, `subprocess`, and `open` are not in that set. The child is killed when `python_timeout_s` (default 20) elapses.

A single tool message sent back to the model is cut at 6000 characters. That cut is not the context-window policy. The middlewares above are.

### Policy checks

`authorize_forecast` refuses a tool or model outside the allow-lists. `check_request` requires at least 2 finite numbers, length at most `max_series_length` (2048), absolute value at most `max_abs_value`, and a horizon inside `horizon_min`–`horizon_max`.

`compress_series` then keeps the tail. The length is the minimum of the series length, the model's `context_length` from its card, and `context_tail` (512). Chronos is registered with `context_length: 8192`, so the agent still sends at most 512 points. The MCP server itself accepts up to 8192 values. The 512-point cap is agent policy, not the foundation model's native window.

## TSFM server

`tsfm_server/server.py` is a FastMCP server on streamable HTTP, path `/mcp`, stateless. `scripts/run_tsfm.sh` forces `HF_HUB_OFFLINE=1` unless already set, and selects the GPU with `CUDA_VISIBLE_DEVICES`.

| Module | Role |
| --- | --- |
| `tsfm_server/settings.py` | Host, port, model-config path, default checkpoint environment variables. |
| `tsfm_server/registry.py` | Loads `configs/models.yaml` into `ModelSpec`. `${VAR}` in checkpoints is expanded at load time. `card()` is what `list_models` returns. It does not include the checkpoint path. |
| `tsfm_server/runtime.py` | `ModelRuntime`. One GPU adapter is resident at a time. Switching GPU models calls `unload()` on the previous one. CPU adapters stay cached and do not evict the GPU model. `forecast` holds a process-wide lock. |
| `tsfm_server/reload.py` | `importlib.reload` on `base`, `registry`, each adapter, the adapter package, and `runtime`, then builds a new `ModelRuntime`. On failure the server keeps the previous runtime. |
| `tsfm_server/base.py` | `BaseTSFMModel`. Adapters implement `forecast` and return `point`, `q10`, `q90`, `model`, `elapsed_ms`, `freq`. |
| `tsfm_server/adapters/__init__.py` | Maps adapter names to classes: `chronos2`, `timesfm3`, `seasonal_naive`. |
| `tsfm_server/adapters/chronos2.py` | Lazy `Chronos2Pipeline`. Quantile levels 0.1, 0.5, 0.9. The point forecast is the pipeline mean. Falls back to CPU if CUDA is requested but unavailable. |
| `tsfm_server/adapters/timesfm3.py` | Lazy TimesFM-3 PyTorch pipeline, using the local checkpoint. |
| `tsfm_server/adapters/seasonal_naive.py` | Repeats the last season. Default season is 7, from the YAML `season` field. Bands come from residual quantiles. No checkpoint. |

`shared/metrics.py` is imported by the TSFM process, not by the agent. `evaluate_forecast` returns `mae`, `mase`, `coverage`, and `crps_proxy`. `mase` is `None` when the history is shorter than one season. `crps_proxy` is the average pinball loss at quantiles 0.1, 0.5, and 0.9. Without intervals, only the median pinball is used.

## Configuration files

| File | Contents |
| --- | --- |
| `configs/models.yaml` | Registered forecasters, device, checkpoint, context length, quantiles. |
| `configs/permissions.yaml` | Tool and model allow-lists, series and horizon bounds, approval list, ReAct step cap, chat context window, Python timeout and path roots. |
| `prompts/react.md` | System prompt for the tool loop. |
| `prompts/select_model.md` | JSON model-choice prompt. Only the fixed pipeline would use it, and only if an LLM were enabled on that path. The current switch does not do that. |
| `prompts/report.md` | Report prompt. Same situation as `select_model.md`. |
| `samples/meichengmatou_tail128.json` | Input for `scripts/smoke_agent.py` and `scripts/smoke_tsfm.py`. |

## Tests

Tests do not load Chronos or TimesFM weights.

| File | Covers |
| --- | --- |
| `tests/test_pipeline.py` | Fixed pipeline routing, context tail of 8 in the test policy, and approval interrupt. Tools are scripted. |
| `tests/test_react.py` | Several tool calls in one turn, a later forecast seeing a rewritten series, source edits triggering `reload_models`, and approve/deny. |
| `tests/test_context.py` | `ClearToolUsesEdit` keeps the latest tool results. `SummarizationMiddleware` replaces older messages once the configured window is crossed. |
| `tests/test_python_exec.py` | Series rewrite, blocked imports, write jail, subprocess round-trip, timeout. |
| `tests/test_chat_model.py` | One completion can carry two tool calls. |
| `tests/test_reload.py` | A freshly imported runtime can forecast with `seasonal_naive`. |
| `tests/test_metrics.py` | MAE, MASE, coverage, CRPS proxy. |

Run them with `.venv-agent/bin/python -m unittest discover tests`.
