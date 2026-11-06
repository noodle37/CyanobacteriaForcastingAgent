# GIFT-Eval Integration — Progress Report

**Date:** 2026-11-06
**Status:** Environment, benchmark plumbing and the end-to-end evaluation path are complete and
verified. Agent-mode (ReAct) evaluation is blocked on an external LLM quota limit.
**Companion document:** `documentation/gift-eval.md` (operational guide, in Chinese).

---

## 1. Objective and scope

Integrate `tsfm-agent` into the GIFT-Eval benchmark so that the agent can be evaluated under the
benchmark's own task splits and metric suite, and so that the agent's contribution can be separated
from the contribution of the underlying forecasting models.

Scope boundary: **no training or fine-tuning of any time-series foundation model is involved.**
The models already deployed in `tsfm_server` are used as frozen tools. The benchmark serves here as
an evaluation environment for the agent, not as a model-development track.

---

## 2. Completed work

### 2.1 Evaluation environment

A dedicated environment `.venv-gift` was created, isolated from both `.venv-agent` (agent service)
and `.venv-tsfm` (model service).

| Component | Version | Reason |
| --- | --- | --- |
| gluonts | 0.15.1 | Matches the benchmark's own `pyproject.toml` pin |
| datasets | 2.17.1 | Required by `gift_eval.data` for `load_from_disk` |
| numpy | 1.26.4 | Matches the benchmark's `numpy~=1.26.0` constraint |
| pandas | 2.3.3 | Dataset property lookups and result assembly |

The environment is built from the conda interpreter with `--system-site-packages` so that the
existing PyTorch installation is inherited rather than downloaded a second time. The gluonts pin
matters in practice: the return structure of `evaluate_model` and the iteration semantics of
`TestData` differ across gluonts releases, and the benchmark's own code is written against 0.15.x.

### 2.2 Benchmark source and data

- The benchmark source is vendored at `third_party/GIFT-Eval/` (8 MB, including `LICENSE.txt`).
  `run_eval.py` injects its `src` directory into the module search path, so no package install is
  needed.
- The dataset lives on HuggingFace as `Salesforce/GiftEval`. It was downloaded to
  `/home/jxy/data/gift_eval`. The published layout already matches the `save_to_disk` format that
  `datasets.load_from_disk` expects — each dataset directory contains `data-*.arrow`,
  `state.json` and `dataset_info.json` — so no format conversion is required.
- Network note: `github.com` and `huggingface.co` were intermittently unreachable from this host
  during the work. The working channels are `gh-proxy.com` for source archives and `hf-mirror.com`
  for datasets; both acquisition commands are recorded verbatim in `documentation/gift-eval.md`.

### 2.3 Predictor adapter — `evaluation/gift_eval/agent_predictor.py`

Wraps the project's HTTP `/v1/runs` endpoint as a `gluonts.model.Predictor`. Per test instance it:

1. extracts the history and fills missing values with `LastValueImputation`;
2. truncates the context to the configured maximum and records whether truncation occurred;
3. posts to the agent — **the request body never contains `actual`**; ground truth exists only on
   the GluonTS label side, so the agent cannot see the future values it will be scored against;
4. expands the returned point forecast and 10%/90% quantiles into nine quantile levels;
5. assembles a `QuantileForecast` whose `start_date` is `forecast_start(entry)`.

A step-count guard was added after a failure was observed: if the agent returns fewer steps than
the task requires, the adapter raises an explicit error; if it returns more, the forecast is
trimmed. Without this, `BatchForecast.__getitem__` stacks every forecast array in a batch and a
single length mismatch fails the whole batch.

### 2.4 Evaluation runner — `evaluation/gift_eval/run_eval.py`

Follows the official protocol:

- multivariate datasets are split with `to_univariate=True`, as the official notebooks do;
- seasonality is taken from `get_seasonality(dataset.freq)`;
- `evaluate_model` is invoked with the official parameters — `axis=None`,
  `mask_invalid_label=True`, `allow_nan_forecast=False`, `batch_size=512`;
- the metric set is the official one (MSE, MAE, MASE, MAPE, sMAPE, MSIS, RMSE, NRMSE, ND, weighted
  sum quantile loss over nine levels).

Output is `all_results.csv` whose first 15 columns are identical to the official submission schema
(names and order), followed by run-metadata columns: series count, truncation count, mean context
length, mean used length, mean latency, agent-selected model, prediction length, season length and
notes.

Sampling strategies are `head`, `stride` (default) and `random`. The default is `stride` because
`head` can systematically select degenerate instances — in `car_parts_with_missing` the first
several series have entirely missing label windows, which makes every metric undefined.

Failures are isolated per task: an exception is recorded as a failed row with the reason in
`notes`, and the remaining tasks continue.

### 2.5 Fixed-pipeline launcher — `scripts/run_agent_fixed.sh`

Starts the agent with the language-model variables blanked, giving the deterministic path
`ingest → select_model → forecast → baseline → report` at roughly 0.05 s per series, which is the
mode used for bulk evaluation.

Implementation detail worth recording: `agent/settings.py` loads the repository `.env` via
`load_dotenv`, and `load_dotenv` does not override variables that already exist. The three
language-model variables must therefore be set to **empty strings** rather than `unset` — after
`unset` the keys no longer exist and `.env` repopulates them, silently putting the agent back on
the ReAct path.

### 2.6 Verified test results

The fixed-pipeline path produces complete metric sets across three frequencies and a
missing-value-heavy dataset:

| Task configuration | Series | MASE | WQL | MSIS |
| --- | --- | --- | --- | --- |
| `covid_deaths/D/short` | 10 | 65.2660 | 0.0487 | 2032.32 |
| `car_parts/M/short` | 10 | 1.6845 | 1.4280 | 19.53 |
| `m4_weekly/W/short` | 10 | 2.4787 | 0.0282 | 19.43 |
| `hierarchical_sales/D/short` | 10 | 0.9055 | 0.6452 | 7.55 |

For calibration: on `covid_deaths/D/short` the official leaderboard reports MASE 46.91 for
`Seasonal_Naive`, 32.54 for `Chronos-2` and 31.86 for `TimesFM-3`. Values in the tens for this
task are normal — the series are cumulative epidemic counts that jump from a flat zero history.
Our numbers sit in the same regime on a 10-series sample and are not directly comparable to the
official full-dataset figures.

The project's existing 35 unit tests still pass unchanged.

### 2.7 Documentation

- `documentation/gift-eval.md` — operational guide in Chinese: environment setup, mirror-based
  acquisition, launch modes, command reference, adapter behaviour, and a full list of known
  deviations from the official protocol.
- `documentation/gift-eval-progress.md` — this report.

---

## 3. What is still missing

### 3.1 Blocking item

**Agent-mode evaluation cannot run until the LLM provider quota is restored.** A request against
the ReAct instance returns `PermissionDeniedError: 403 - Free quota exhausted`; the agent then
finishes in `partial` status without returning a forecast. This is an account-side limit that
requires either adding funds or disabling the free-tier-only mode. The adapter, the failure
handling and the runner are already in place, so the run can be resumed directly once quota is
available. The fixed-pipeline path is unaffected and remains usable for bulk work.

### 3.2 Deviations from the official protocol that should be closed

1. **Quantile levels.** The agent's external contract exposes only a point forecast and the 10%
   and 90% quantiles, while the benchmark's weighted quantile loss needs nine levels. The
   remaining seven levels are currently obtained by piecewise-linear interpolation, which is an
   approximation and makes `mean_weighted_sum_quantile_loss` and `MSIS` systematically deviate from
   official values. The fix is to have the model adapters emit the nine levels directly — both
   Chronos-2 and TimesFM-3 support arbitrary quantile levels natively.
2. **Context length.** The agent policy caps the input at 2048 points (`context_tail` is 512), while
   many benchmark datasets have far longer histories. Truncation changes MASE and quantile loss.
   The result rows record `n_context_truncated` and `mean_used_len`; comparability with official
   rows requires either confirming a row is untruncated or relaxing the cap for evaluation runs.
3. **Horizon cap.** `horizon_max` is 64. The benchmark's medium and long terms are 10× and 15× the
   base prediction length (300 and 450 steps for daily data), so **only short-term tasks are
   currently reachable**. Lifting this requires a policy change and a latency check at long
   horizons.

### 3.3 Work not yet started

4. **No full 97-task sweep.** Only four datasets have been downloaded so far; the remainder is
   roughly 1.6 GB from the mirror. The fixed pipeline costs about 0.05 s per series, so the sweep
   is dominated by download and by the length of the longest tasks rather than by compute.
5. **Sampling design is undecided.** Open decisions: how many series per task, whether to use
   equal-interval or random sampling, and whether to cover all 97 task configurations at a reduced
   series count.
6. **No reference-baseline row.** The leaderboard normalises every model by the `Seasonal_Naive`
   row for the same task. Computing a local seasonal-naive row using the official
   `get_seasonality` season length would make our numbers interpretable on the same scale. This was
   deliberately left out at this stage so that no separate model-evaluation track is reintroduced;
   it should be decided together with the sampling design.
7. **No agent-mode cost accounting.** Per-series model calls, tool calls, wall time and consumed
   context length are not yet recorded. Without them, the central question — whether the agent's
   additional accuracy justifies its overhead — cannot be answered.
8. **No leaderboard submission package.** Not required for internal use. If submission is ever
   wanted, it needs `results/<name>/all_results.csv` and `config.json` with `model_type` set to
   `agentic`, plus `model_link`, `code_link`, `org` and `testdata_leakage`.

---

## 4. Technical notes and risks

- **Horizon is not a hard contract in ReAct mode.** The agent's `forecast` tool allows overriding
  `horizon`, so the model may choose a different number of steps than requested. This is now
  detected and reported, but the durable fix is to make the requested horizon binding inside the
  tool or the prompt.
- **The project's known metrics/forecast binding issue does not affect the benchmark numbers.** The
  harness scores the forecast object it receives from the agent, so a stale metric inside the
  agent's own report text is not part of the metric computation. It does mean the agent's
  narrative output can still be internally inconsistent.
- **Series rewriting through the `python` tool is legitimate in this setting.** The agent only ever
  sees the context, so transforming the history is not leakage. The leakage risk recorded
  elsewhere in this project — `actual` being injected into the sandbox — is structurally eliminated
  here, because the benchmark request never carries `actual`.

---

## 5. Recommended order of work

1. Restore the LLM quota, then re-run agent mode on a small sample (3–5 series) to measure
   per-series latency, tool-call counts and accuracy against the fixed pipeline on identical
   series.
2. Implement nine-quantile output in the model adapters to close deviation 3.2.1.
3. Settle the sampling design and run the fixed pipeline over the selected task configurations.
4. Add agent-mode cost accounting to the runner.
5. On the basis of steps 1–4, decide whether the agent layer earns its overhead, and whether
   coverage should be extended to the medium and long terms.

---

## 6. How to reproduce the verified state

```bash
cd /home/jxy/tsfm-agent
export GIFT_EVAL=/home/jxy/data/gift_eval

# 固定流水线模式的 Agent(已在 8001 端口运行)
./scripts/run_agent_fixed.sh 8001

# 评测
.venv-gift/bin/python -m evaluation.gift_eval.run_eval \
    --dataset covid_deaths --dataset car_parts_with_missing \
    --dataset m4_weekly --dataset hierarchical_sales/D \
    --term short --limit-series 10 \
    --agent-url http://127.0.0.1:8001 \
    --out evaluation/results/gift_eval
```

Expected output: four completed task rows with eleven metrics each, and an `all_results.csv` whose
first 15 columns match the official submission schema.
