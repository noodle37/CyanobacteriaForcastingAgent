"""把 tsfm-agent 包装成 GluonTS 预测器。

GIFT-Eval 的评测对象是一个「把上下文序列映射到预测分布」的函数,通过 GluonTS 的
Predictor 接口接入。本模块把本项目的 HTTP 接口包装成该接口:对测试集中的每个实例,
把历史序列提交给 Agent,取回点预测与分位数,组装成 QuantileForecast。

评测请求不携带 actual。真值只存在于 GluonTS 的标签侧,Agent 在评分前看不到未来值,
这与项目内部「开发信号与评分信号必须分离」的要求一致。
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Iterable, Iterator, List, Optional, Sequence

import numpy as np
from gluonts.dataset.util import forecast_start
from gluonts.model import Forecast, Predictor
from gluonts.model.forecast import QuantileForecast
from gluonts.transform.feature import LastValueImputation

# 基准的加权分位数损失需要 0.1 至 0.9 共九档。
DEFAULT_QUANTILE_LEVELS: tuple[float, ...] = (
    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
)

DEFAULT_USER_REQUEST = "预测该时间序列未来 {horizon} 步的取值,并给出分位数区间"


@dataclass
class SeriesTrace:
    """单条序列的评测记录,用于事后核查上下文长度、耗时与失败原因。"""

    item_id: str
    context_len: int
    used_len: int
    truncated: bool
    elapsed_s: float
    status: str
    model_name: Optional[str] = None
    error: Optional[str] = None


class AgentRequestError(RuntimeError):
    """Agent 未返回可用预测时抛出,附带响应里的状态与错误信息。"""


class AgentPredictor(Predictor):
    """通过本项目的 HTTP 接口产生预测的 GluonTS 预测器。

    参数
    ----
    prediction_length : 预测步数,即 GIFT-Eval 的任务预测长度。
    freq : 序列频率,原样传给 Agent。
    agent_url : Agent 服务地址,评测请求发往 ``{agent_url}/v1/runs``。
    quantile_levels : 需要的分位水平。
    max_context : 提交给 Agent 的最大上下文点数。Agent 策略上限为 2048,
        超过部分会被截断,截断情况记入 traces。
    timeout_s : 单次请求超时秒数。ReAct 模式单次运行可达数分钟。
    """

    def __init__(
        self,
        prediction_length: int,
        freq: str,
        agent_url: str = "http://127.0.0.1:8000",
        quantile_levels: Optional[Sequence[float]] = None,
        max_context: int = 2048,
        timeout_s: float = 900.0,
        user_request: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> None:
        self.prediction_length = int(prediction_length)
        self.freq = freq
        self.agent_url = agent_url.rstrip("/")
        self.quantile_levels = list(quantile_levels or DEFAULT_QUANTILE_LEVELS)
        self.max_context = int(max_context)
        self.timeout_s = float(timeout_s)
        self.user_request = user_request or DEFAULT_USER_REQUEST
        self.model_name = model_name
        self.imputation = LastValueImputation()
        self.traces: List[SeriesTrace] = []

    # -- GluonTS 接口 ---------------------------------------------------------

    def predict(self, dataset: Iterable[dict], **kwargs) -> Iterator[Forecast]:
        for entry in dataset:
            yield self._forecast_one(entry)

    # -- 单条序列 -------------------------------------------------------------

    def _forecast_one(self, entry: dict) -> Forecast:
        raw = np.asarray(entry["target"], dtype=float)
        context_len = int(raw.size)
        values = self.imputation(raw).astype(float)
        used = values[-self.max_context :] if self.max_context > 0 else values
        truncated = int(used.size) != context_len

        started = time.perf_counter()
        try:
            response = self._post(used.tolist())
        except Exception as exc:  # 失败信息写入 traces 后继续抛出,由上层决定是否终止
            self.traces.append(
                SeriesTrace(
                    item_id=str(entry.get("item_id")),
                    context_len=context_len,
                    used_len=int(used.size),
                    truncated=truncated,
                    elapsed_s=time.perf_counter() - started,
                    status="failed",
                    error=str(exc),
                )
            )
            raise
        elapsed = time.perf_counter() - started

        payload = response.get("forecast") or {}
        point = payload.get("point")
        if not point:
            message = (
                f"Agent 未返回预测:status={response.get('status')} "
                f"errors={response.get('errors')} "
                f"model={response.get('model_name')}"
            )
            self.traces.append(
                SeriesTrace(
                    item_id=str(entry.get("item_id")),
                    context_len=context_len,
                    used_len=int(used.size),
                    truncated=truncated,
                    elapsed_s=elapsed,
                    status=str(response.get("status") or "unknown"),
                    model_name=response.get("model_name"),
                    error=message,
                )
            )
            raise AgentRequestError(message)

        quantiles = expand_quantiles(
            point=list(point),
            q10=payload.get("q10"),
            q90=payload.get("q90"),
            levels=self.quantile_levels,
        )
        self.traces.append(
            SeriesTrace(
                item_id=str(entry.get("item_id")),
                context_len=context_len,
                used_len=int(used.size),
                truncated=truncated,
                elapsed_s=elapsed,
                status=str(response.get("status") or "unknown"),
                model_name=response.get("model_name"),
            )
        )

        # Agent 返回的步数必须与任务要求一致:短了无法评分,长了按任务长度截取。
        returned = len(quantiles[self.quantile_levels[0]])
        if returned < self.prediction_length:
            raise AgentRequestError(
                f"Agent 返回 {returned} 步,少于任务要求的 {self.prediction_length} 步"
            )
        if returned > self.prediction_length:
            quantiles = {
                level: values[: self.prediction_length]
                for level, values in quantiles.items()
            }

        arrays = np.vstack([np.asarray(quantiles[level], dtype=float) for level in self.quantile_levels])
        return QuantileForecast(
            forecast_arrays=arrays,
            start_date=forecast_start(entry),
            forecast_keys=[str(level) for level in self.quantile_levels],
            item_id=entry.get("item_id"),
        )

    # -- HTTP -----------------------------------------------------------------

    def _post(self, series: List[float]) -> dict:
        body = json.dumps(
            {
                "user_request": self.user_request.format(horizon=self.prediction_length),
                "series": series,
                "horizon": self.prediction_length,
                "freq": self.freq,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.agent_url}/v1/runs",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise AgentRequestError(f"HTTP {exc.code}: {detail}") from exc


def expand_quantiles(
    point: List[float],
    q10: Optional[List[float]],
    q90: Optional[List[float]],
    levels: Sequence[float],
) -> dict:
    """把 Agent 返回的三个分位扩展到需要的分位水平。

    Agent 目前的对外契约只有点预测与 10%、90% 两个分位。九档分位里其余各档
    用分段线性插值得到,属于近似做法,会在评测报告中标注。
    """
    center = np.asarray(point, dtype=float)
    lower = np.asarray(q10, dtype=float) if q10 else center.copy()
    upper = np.asarray(q90, dtype=float) if q90 else center.copy()
    out: dict = {}
    for level in levels:
        level = float(level)
        if level <= 0.5:
            ratio = (level - 0.1) / 0.4
            values = lower + ratio * (center - lower)
        else:
            ratio = (level - 0.5) / 0.4
            values = center + ratio * (upper - center)
        out[level] = values.tolist()
    return out
