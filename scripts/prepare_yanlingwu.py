"""把严陵坞原始逐小时观测整成 Agent 能吃的一维序列。

原始 CSV 是宽表：obs_time 加 18 个指标列，2008 年起混合 1 小时与 4 小时采样，
2023 年前后采样制度还变过。Agent 的 HTTP 接口只接受一维数值序列，所以这一步负责：
选目标列、按日聚合、补齐日历网格、线性插值缺测、裁到上限长度。

默认口径与 chronos-cyano/scripts/prepare_data.py 保持一致：目标是日最大蓝藻浓度
cyano_ugl，协变量是水温、浊度、总磷、总氮的日均值。

用法：
    .venv-agent/bin/python scripts/prepare_yanlingwu.py
    .venv-agent/bin/python scripts/prepare_yanlingwu.py --target total_chla_ugl --tail 2048
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = Path("/home/jxy/data/yanlingwu_observations.csv")

# 与旧工程一致的目标与协变量口径。target 取日最大值，协变量取日均值。
TARGETS = {
    "cyano_ugl": "max",
    "total_chla_ugl": "max",
    "green_algae_ugl": "max",
    "water_temp_c": "mean",
    "turbidity_ntu": "mean",
    "tp_mgl": "mean",
    "tn_mgl": "mean",
    "do_mgl": "mean",
    "ph": "mean",
}


def load_daily(raw_path: Path, target: str, agg: str) -> pd.DataFrame:
    df = pd.read_csv(raw_path, encoding="utf-8-sig")
    df["obs_time"] = pd.to_datetime(df["obs_time"])
    df = df.sort_values("obs_time").drop_duplicates("obs_time", keep="last")
    df["date"] = df["obs_time"].dt.floor("D")

    daily = df.groupby("date")[target].agg(agg).to_frame(target)

    # 补齐完整日历网格，缺测用线性插值，与旧工程 to_regular_daily 同一口径。
    grid = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(grid)
    daily[target] = daily[target].interpolate(method="linear", limit_direction="both")
    daily = daily.dropna(subset=[target])
    daily.index.name = "timestamp"
    return daily


def to_sample(daily: pd.DataFrame, item_id: str, tail: int, freq: str) -> dict:
    series = daily.tail(tail)
    return {
        "item_id": item_id,
        "freq": freq,
        "timestamps": [ts.strftime("%Y-%m-%d %H:%M:%S") for ts in series.index],
        "values": [round(float(v), 6) for v in series[series.columns[0]].to_numpy()],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="严陵坞原始观测 → Agent 输入序列")
    parser.add_argument("--raw", default=str(DEFAULT_RAW), help="原始观测 CSV")
    parser.add_argument("--target", default="cyano_ugl", choices=sorted(TARGETS), help="目标指标列")
    parser.add_argument("--agg", default=None, choices=["max", "mean", "sum"], help="日聚合方式，默认按目标列的常用口径")
    parser.add_argument("--tail", type=int, default=128, help="取末尾多少个日点，默认 128")
    parser.add_argument("--freq", default="D", help="pandas 频率字符串")
    parser.add_argument("--item-id", default="yanlingwu", help="序列名")
    parser.add_argument("--output", default=str(ROOT / "samples" / "yanlingwu_daily_tail128.json"))
    args = parser.parse_args()

    agg = args.agg or TARGETS[args.target]
    daily = load_daily(Path(args.raw), args.target, agg)
    sample = to_sample(daily, args.item_id, args.tail, args.freq)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(sample, ensure_ascii=False, indent=2), encoding="utf-8")

    values = sample["values"]
    print(f"源文件      : {args.raw}")
    print(f"目标列      : {args.target}（日{agg}）")
    print(f"日频区间    : {sample['timestamps'][0]} → {sample['timestamps'][-1]}")
    print(f"点数        : {len(values)}  范围 [{min(values):.3f}, {max(values):.3f}]")
    print(f"写出        : {out}")


if __name__ == "__main__":
    main()
