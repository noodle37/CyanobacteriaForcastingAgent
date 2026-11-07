"""在 GIFT-Eval 基准的任务切分与指标体系下评测本项目的 Agent。

用法示例(数据目录需先按文档准备):

    GIFT_EVAL=/home/jxy/data/gift_eval \\
    .venv-gift/bin/python -m evaluation.gift_eval.run_eval \\
        --dataset covid_deaths --term short --limit-series 8

产出 `all_results.csv`,列名与官方提交格式一致,另附若干本次评测的附加信息列。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
GIFT_SRC = ROOT / "third_party" / "GIFT-Eval" / "src"
if GIFT_SRC.is_dir():
    sys.path.insert(0, str(GIFT_SRC))

from gluonts.ev.metrics import (  # noqa: E402
    MAE,
    MAPE,
    MASE,
    MSE,
    MSIS,
    ND,
    NRMSE,
    RMSE,
    SMAPE,
    MeanWeightedSumQuantileLoss,
)
from gluonts.model import evaluate_model  # noqa: E402
from gluonts.time_feature import get_seasonality  # noqa: E402

from evaluation.gift_eval.agent_predictor import (  # noqa: E402
    DEFAULT_QUANTILE_LEVELS,
    AgentPredictor,
    SeriesTrace,
)

# 官方提交文件的列顺序。
OFFICIAL_COLUMNS = [
    "dataset",
    "model",
    "eval_metrics/MSE[mean]",
    "eval_metrics/MSE[0.5]",
    "eval_metrics/MAE[0.5]",
    "eval_metrics/MASE[0.5]",
    "eval_metrics/MAPE[0.5]",
    "eval_metrics/sMAPE[0.5]",
    "eval_metrics/MSIS",
    "eval_metrics/RMSE[mean]",
    "eval_metrics/NRMSE[mean]",
    "eval_metrics/ND[0.5]",
    "eval_metrics/mean_weighted_sum_quantile_loss",
    "domain",
    "num_variates",
]

EXTRA_COLUMNS = [
    "n_series",
    "n_context_truncated",
    "mean_context_len",
    "mean_used_len",
    "mean_elapsed_s",
    "agent_model",
    "prediction_length",
    "season_length",
    "notes",
]

# 官方 notebook 使用的名称美化规则。
PRETTY_NAMES = {
    "saugeenday": "saugeen",
    "temperature_rain_with_missing": "temperature_rain",
    "kdd_cup_2018_with_missing": "kdd_cup_2018",
    "car_parts_with_missing": "car_parts",
}


def build_metrics():
    """官方 notebook 使用的指标集合,顺序与列名对应。"""
    return [
        MSE(forecast_type="mean"),
        MSE(forecast_type=0.5),
        MAE(),
        MASE(),
        MAPE(),
        SMAPE(),
        MSIS(),
        RMSE(),
        NRMSE(),
        ND(),
        MeanWeightedSumQuantileLoss(
            quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        ),
    ]


def metric_value(result, name: str):
    """兼容 evaluate_model 返回字典或 DataFrame 两种形态。"""
    if hasattr(result, "iloc"):
        return result[name].iloc[0]
    value = result[name]
    if isinstance(value, (tuple, list, np.ndarray)):
        return value[0]
    return value


class LimitedTestData:
    """按指定策略抽取若干条序列的测试集包装。

    GluonTS 的 evaluate_model 需要 test_data 同时具备 input 与 label 两个属性,
    这里保留同样的结构,只截断序列条数,便于快速验证链路。

    抽样策略:
    head    取前 limit 条。实现简单,但可能系统性地取到退化样本——某些数据集
            (如 car_parts_with_missing)前若干条序列的标签窗口整体缺测,指标会算成 nan。
    stride  在全部实例上等距抽样,覆盖面更均匀,默认采用。
    random  固定随机种子抽取,可复现。
    """

    def __init__(self, test_data, limit: Optional[int] = None, strategy: str = "stride") -> None:
        pairs = list(test_data)
        if limit is None or limit >= len(pairs):
            selected = pairs
        elif strategy == "head":
            selected = pairs[:limit]
        elif strategy == "random":
            rng = np.random.default_rng(20241105)
            index = np.sort(rng.choice(len(pairs), size=limit, replace=False))
            selected = [pairs[i] for i in index]
        else:  # stride
            index = np.linspace(0, len(pairs) - 1, num=limit).round().astype(int)
            selected = [pairs[i] for i in sorted(set(index.tolist()))]
        self._pairs = selected
        self._input = [pair[0] for pair in self._pairs]
        self._label = [pair[1] for pair in self._pairs]

    @property
    def input(self) -> List[dict]:
        return self._input

    @property
    def label(self) -> List[dict]:
        return self._label

    def __len__(self) -> int:
        return len(self._pairs)


def load_dataset_properties() -> dict:
    path = ROOT / "third_party" / "GIFT-Eval" / "notebooks" / "dataset_properties.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def split_dataset_name(name: str) -> tuple[str, Optional[str]]:
    """把数据集名拆成 (主名, 频率目录)。频率目录可能为空。"""
    if "/" in name:
        key, freq = name.split("/", 1)
        return key, freq
    return name, None


def config_name(name: str, freq: str, term: str, properties: dict) -> str:
    key, freq_dir = split_dataset_name(name)
    key = key.lower()
    key = PRETTY_NAMES.get(key, key)
    if freq_dir is None:
        freq_dir = (properties.get(key) or {}).get("frequency", freq)
    return f"{key}/{freq_dir}/{term}"


def evaluate_task(
    name: str,
    term: str,
    args: argparse.Namespace,
    properties: dict,
) -> Optional[dict]:
    """评测一个任务配置,返回一行结果。"""
    from gift_eval.data import Dataset

    plain = Dataset(name=name, term=term, to_univariate=False)
    to_univariate = plain.target_dim != 1
    dataset = Dataset(name=name, term=term, to_univariate=to_univariate)

    season_length = get_seasonality(dataset.freq)
    if dataset.prediction_length > args.max_horizon:
        print(
            f"[跳过] {name}/{term}:预测长度 {dataset.prediction_length} 超过 Agent 策略上限 {args.max_horizon}",
            flush=True,
        )
        return None

    predictor = AgentPredictor(
        prediction_length=dataset.prediction_length,
        freq=dataset.freq,
        agent_url=args.agent_url,
        quantile_levels=DEFAULT_QUANTILE_LEVELS,
        max_context=args.max_context,
        timeout_s=args.timeout_s,
        user_request=args.user_request,
    )

    test_data = LimitedTestData(dataset.test_data, limit=args.limit_series, strategy=args.sample_strategy)
    started = time.perf_counter()
    try:
        result = evaluate_model(
            predictor,
            test_data=test_data,
            metrics=build_metrics(),
            batch_size=args.batch_size,
            axis=None,
            mask_invalid_label=True,
            allow_nan_forecast=False,
            seasonality=season_length,
        )
    except Exception as exc:
        elapsed = time.perf_counter() - started
        key, _freq_dir = split_dataset_name(name)
        key = PRETTY_NAMES.get(key.lower(), key.lower())
        meta = properties.get(key) or {}
        row = {
            "dataset": config_name(name, dataset.freq, term, properties),
            "model": args.model_name,
            "domain": meta.get("domain", ""),
            "num_variates": meta.get("num_variates", dataset.target_dim),
            "n_series": len(test_data),
            "notes": f"评测失败: {type(exc).__name__}: {exc}",
        }
        for column in OFFICIAL_COLUMNS:
            row.setdefault(column, "")
        for column in EXTRA_COLUMNS:
            row.setdefault(column, "")
        print(f"[失败] {row['dataset']} {row['notes']}", flush=True)
        return row
    elapsed = time.perf_counter() - started

    traces: List[SeriesTrace] = predictor.traces
    key, _freq_dir = split_dataset_name(name)
    key = PRETTY_NAMES.get(key.lower(), key.lower())
    meta = properties.get(key) or {}
    domain = meta.get("domain", "")
    num_variates = meta.get("num_variates", dataset.target_dim)

    truncated = sum(1 for trace in traces if trace.truncated)
    row = {
        "dataset": config_name(name, dataset.freq, term, properties),
        "model": args.model_name,
        "domain": domain,
        "num_variates": num_variates,
        "n_series": len(traces),
        "n_context_truncated": truncated,
        "mean_context_len": round(float(np.mean([t.context_len for t in traces])), 2) if traces else 0,
        "mean_used_len": round(float(np.mean([t.used_len for t in traces])), 2) if traces else 0,
        "mean_elapsed_s": round(float(np.mean([t.elapsed_s for t in traces])), 3) if traces else 0,
        "agent_model": traces[-1].model_name if traces else "",
        "prediction_length": dataset.prediction_length,
        "season_length": season_length,
        "notes": "",
    }
    for column in OFFICIAL_COLUMNS:
        if column in ("dataset", "model", "domain", "num_variates"):
            continue
        row[column] = metric_value(result, column.replace("eval_metrics/", ""))
    row["notes"] = f"wall={elapsed:.1f}s"
    if not np.isfinite(float(row["eval_metrics/MASE[0.5]"])):
        row["notes"] += "; 指标无定义(所抽序列的标签窗口整体缺测)"
        print(f"[提示] {row['dataset']} 指标为 nan,通常是所抽序列的标签窗口整体缺测", flush=True)
    print(
        f"[完成] {row['dataset']} 序列数={row['n_series']} "
        f"MASE={row['eval_metrics/MASE[0.5]']:.4f} "
        f"WQL={row['eval_metrics/mean_weighted_sum_quantile_loss']:.4f} "
        f"用时={elapsed:.1f}s",
        flush=True,
    )
    return row


def write_rows(rows: Sequence[dict], out_dir: Path, model_name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "all_results.csv"
    columns = OFFICIAL_COLUMNS + EXTRA_COLUMNS
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"结果已写入 {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="在 GIFT-Eval 上评测 tsfm-agent")
    parser.add_argument("--dataset", action="append", required=True, help="数据集名,可重复;如 covid_deaths、hierarchical_sales/D")
    parser.add_argument("--term", default="short", choices=["short", "medium", "long"], help="预测长度档位")
    parser.add_argument("--limit-series", type=int, default=None, help="每个任务最多评测多少条序列")
    parser.add_argument(
        "--sample-strategy",
        default="stride",
        choices=["head", "stride", "random"],
        help="序列抽样策略,默认等距抽样",
    )
    parser.add_argument("--agent-url", default="http://127.0.0.1:8000", help="Agent 服务地址")
    parser.add_argument("--model-name", default="tsfm-agent", help="结果中记录的名字")
    parser.add_argument("--max-context", type=int, default=2048, help="提交给 Agent 的最大上下文点数")
    parser.add_argument("--max-horizon", type=int, default=64, help="Agent 策略允许的最大预测步数")
    parser.add_argument("--timeout-s", type=float, default=900.0, help="单次请求超时秒数")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--out", default=str(ROOT / "evaluation" / "results" / "gift_eval"), help="结果目录")
    parser.add_argument(
        "--user-request",
        default="预测该时间序列未来 {horizon} 步的取值,并给出分位数区间",
        help="提交给 Agent 的任务说明,可用 {horizon} 占位",
    )
    args = parser.parse_args()

    if not os.getenv("GIFT_EVAL"):
        raise SystemExit(
            "未设置 GIFT_EVAL 环境变量。它应指向 GIFT-Eval 数据集目录(其中每个子目录是一个数据集)。"
        )
    properties = load_dataset_properties()
    rows: List[dict] = []
    for name in args.dataset:
        row = evaluate_task(name, args.term, args, properties)
        if row is not None:
            rows.append(row)
    if not rows:
        raise SystemExit("没有可评测的任务,检查 --dataset 与 --term 是否匹配。")
    write_rows(rows, Path(args.out), args.model_name)


if __name__ == "__main__":
    main()
