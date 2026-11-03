from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages
from langgraph.managed import RemainingSteps


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
    """ReAct 图的状态。messages 走模型，其余字段给工具读写。"""

    messages: Annotated[list, add_messages]
    remaining_steps: RemainingSteps
    run_id: str
    user_request: str
    series: list[float]
    horizon: int
    freq: str
    actual: list[float] | None
    model_name: str
    model_reason: str
    forecast: dict | None
    baseline: dict | None
    metrics: dict | None
    baseline_metrics: dict | None
    report: str
    prepared: dict
    catalog: list[dict]
    tool_trace: Annotated[list[dict], operator.add]
    errors: Annotated[list[str], operator.add]
