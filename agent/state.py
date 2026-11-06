from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langgraph.graph.message import add_messages


def keep_last(left: Any, right: Any) -> Any:
    """同一个超步里多个工具写同一个键时，保留后写的那份。

    LangGraph 按分支完成顺序成对调用 reducer。同一轮并行写的多个值最终
    落到同一个分支上，所以 `forecast` 和 `prepared` 这类成对出现的键不会
    互相串位。

    没有 reducer 的键在并行写时会抛 InvalidUpdateError，这是本项目之前
    「同一轮并行调两个模型就崩」的直接原因。
    """
    return right


class TaskLedger(TypedDict, total=False):
    run_id: str
    user_request: str
    series: list[float]
    horizon: int
    freq: str
    actual: list[float] | None
    model_name: str
    model_reason: str
    model_card: dict
    catalog: list[dict]
    forecast: dict | None
    baseline: dict | None
    metrics: dict | None
    baseline_metrics: dict | None
    report: str
    prepared: dict
    route: str
    tool_trace: Annotated[list[dict], operator.add]
    errors: Annotated[list[str], operator.add]


class ReactLedger(TypedDict, total=False):
    """ReAct 图的状态。messages 走模型，其余字段给工具读写。

    工具会写的键一律带 reducer，同一轮的并行调用因此不会互相覆盖。
    `forecasts` 和 `evaluations` 是候选台账：每一次预测和评分都留痕，
    多模型对比的结果不会因为后一次调用而丢失。
    """

    messages: Annotated[list, add_messages]
    run_id: str
    user_request: str
    series: Annotated[list[float], keep_last]
    horizon: Annotated[int, keep_last]
    freq: Annotated[str, keep_last]
    actual: list[float] | None
    model_name: Annotated[str, keep_last]
    model_reason: Annotated[str, keep_last]
    forecast: Annotated[dict | None, keep_last]
    baseline: Annotated[dict | None, keep_last]
    metrics: Annotated[dict | None, keep_last]
    baseline_metrics: Annotated[dict | None, keep_last]
    report: str
    prepared: Annotated[dict, keep_last]
    catalog: Annotated[list[dict], keep_last]
    tool_trace: Annotated[list[dict], operator.add]
    errors: Annotated[list[str], operator.add]
    forecasts: Annotated[list[dict], operator.add]
    evaluations: Annotated[list[dict], operator.add]
