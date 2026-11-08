"""统一对比 RSI / 纯TSFM(Chronos-2) / 传统基线在 2026-09 的日最高值预测。

任务: H1=预测t+1日最高值; H3=预测t+1~t+3逐日最高值(仅3天全实测窗口评分)。
主指标:
  MAE / RMSE                         常规误差
  W-MAE / W-RMSE = Σw|e|/Σw, w_t=1+α·y_t/max(y)   暴发期加权, 强迫模型关注高值
  PRE = |max(y)-max(ŷ)| / max(y)     最强峰值相对误差
各指标在每组(总体/逐站)内用该组 max(y) 归一化权重。

输出 data/replay/2026_max_{h1,h3}_comparison.csv 与 2026_max_comparison.md。

Usage:
    python scripts/compare_systems.py [--alpha 2.0]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "replay"

TASKS = {
    "h1": {
        "rsi": "2026_rsi_max_details.csv",
        "tsfm_chronos2": "2026_chronos2_max_ctx28_details.csv",
        "lstm": "2026_lstm_max_h1_ctx28_details.csv",
        "cnn_lstm": "2026_cnn_lstm_max_h1_ctx28_details.csv",
        "gru": "2026_gru_max_h1_ctx28_details.csv",
        "transformer": "2026_transformer_max_h1_ctx28_details.csv",
    },
    "h3": {
        "rsi": "2026_rsi_max_h3_details.csv",
        "tsfm_chronos2": "2026_chronos2_max_ctx28_h3_details.csv",
        "lstm": "2026_lstm_max_h3_ctx28_details.csv",
        "cnn_lstm": "2026_cnn_lstm_max_h3_ctx28_details.csv",
        "gru": "2026_gru_max_h3_ctx28_details.csv",
        "transformer": "2026_transformer_max_h3_ctx28_details.csv",
    },
}
NOTES = {
    "rsi": "TSFM + LLM选模型(H3锁定chronos2)/天气修正(带上限)/复盘memory (蓝藻+天气)",
    "tsfm_chronos2": "Chronos-2 固定28天context, 零样本 (仅蓝藻历史)",
    "lstm": "LSTM 有监督, 2024~2026.8训练 (蓝藻+天气历史)",
    "cnn_lstm": "CNN-LSTM 有监督, 2024~2026.8训练 (蓝藻+天气历史)",
    "gru": "GRU 有监督, 2024~2026.8训练 (蓝藻+天气历史)",
    "transformer": "Transformer(均值池化) 有监督, 2024~2026.8训练 (蓝藻+天气历史)",
}


def metrics(g: pd.DataFrame, alpha: float) -> dict:
    y = g["actual"].to_numpy(dtype=float)
    yhat = g["point"].to_numpy(dtype=float)
    e = yhat - y
    w = 1.0 + alpha * y / y.max()
    return {"n": len(g), "mae": round(np.abs(e).mean(), 3), "rmse": round(np.sqrt((e ** 2).mean()), 3),
            "wmae": round((w * np.abs(e)).sum() / w.sum(), 3),
            "wrmse": round(np.sqrt((w * e ** 2).sum() / w.sum()), 3),
            "pre": round(abs(y.max() - yhat.max()) / y.max(), 3)}


def load_task(task: str) -> dict[str, pd.DataFrame]:
    frames = {}
    for name, fname in TASKS[task].items():
        p = OUT / fname
        if not p.exists():
            sys.exit(f"缺少 {p}，请先运行对应回放")
        frames[name] = pd.read_csv(p).query("scored == True").copy()
    key = ["station", "date", "origin", "h"] if task == "h3" else ["station", "date"]
    base = set(map(tuple, frames["rsi"][key].to_numpy()))
    for name, g in frames.items():
        got = set(map(tuple, g[key].to_numpy()))
        assert got == base, f"{task}/{name} 评分样本集合不一致: {len(got)} vs {len(base)}"
    print(f"{task} 对齐验证 PASS: {len(base)} 个评分样本 × {len(frames)} 系统")
    return frames


def table(frames: dict[str, pd.DataFrame], alpha: float) -> pd.DataFrame:
    rows = []
    for name, g in frames.items():
        rows.append({"system": name, "station": "overall", **metrics(g, alpha)})
        for st, gs in g.groupby("station"):
            rows.append({"system": name, "station": st, **metrics(gs, alpha)})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=2.0, help="暴发期权重系数 w=1+α·y/max(y)")
    args = ap.parse_args()

    all_tables, lines = {}, ["# 2026-09 日最高值预测对比: RSI vs 纯TSFM(Chronos-2) vs 传统基线", "",
                             "目标: 日最高蓝藻浓度 (cyano_ugl_max)。H1=预测未来1天; H3=预测未来3天逐日值"
                             "(仅3天全实测窗口评分, 每个窗口3行)。",
                             f"加权指标: w_t = 1 + {args.alpha}·y_t/max(y) (组内归一化), W-MAE=Σw|e|/Σw, "
                             f"W-RMSE=√(Σw·e²/Σw); PRE=|max(y)-max(ŷ)|/max(y)。", ""]

    for task in TASKS:
        frames = load_task(task)
        df = table(frames, args.alpha)
        all_tables[task] = df
        df.to_csv(OUT / f"2026_max_{task}_comparison.csv", index=False)

        overall = df[df["station"] == "overall"].set_index("system").loc[list(TASKS[task])]
        lines += [f"## {'H1: 预测未来1天' if task == 'h1' else 'H3: 预测未来3天逐日值'}", "",
                  "| 系统 | 样本数 | MAE | RMSE | W-MAE | W-RMSE | PRE |",
                  "|---|---|---|---|---|---|---|"]
        for name, r in overall.iterrows():
            lines.append(f"| {name} | {r['n']} | {r['mae']:.3f} | {r['rmse']:.3f} | {r['wmae']:.3f} | {r['wrmse']:.3f} | {r['pre']:.3f} |")

        if task == "h3":
            hrows = []
            for name, g in frames.items():
                for h, gh in g.groupby("h"):
                    hrows.append({"system": name, "h": int(h), **metrics(gh, args.alpha)})
            hdf = pd.DataFrame(hrows)
            hdf.to_csv(OUT / "2026_max_h3_by_horizon.csv", index=False)
            lines += ["", "**H3 分 horizon（逐日）**", "",
                      "| 系统 | h | MAE | RMSE | W-MAE | W-RMSE |", "|---|---|---|---|---|---|"]
            for _, r in hdf.iterrows():
                lines.append(f"| {r['system']} | t+{r['h']} | {r['mae']:.3f} | {r['rmse']:.3f} | {r['wmae']:.3f} | {r['wrmse']:.3f} |")

        stations = sorted(df[df["station"] != "overall"]["station"].unique())
        lines += ["", "**逐站 MAE / RMSE / PRE**", "",
                  "| 系统 | " + " | ".join(stations) + " |", "|---|" + "---|" * len(stations)]
        for name in TASKS[task]:
            cells = []
            for st in stations:
                r = df[(df["system"] == name) & (df["station"] == st)].iloc[0]
                cells.append(f"{r['mae']:.3f} / {r['rmse']:.3f} / {r['pre']:.3f}")
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        lines.append("")

        seed_files = [ROOT / "data" / "baselines" / f"seed_metrics_{arch}_max_{task}_ctx28.csv"
                      for arch in TASKS[task] if arch not in ("rsi", "tsfm_chronos2")]
        seed_files = [p for p in seed_files if p.exists()]
        if seed_files:
            sm = pd.concat([pd.read_csv(p) for p in seed_files])
            sm["arch"] = sm["arch"].replace({"transformer_meanpool": "transformer",
                                             "transformer_lastpool": "transformer_lastpool"})
            per_seed = sm.groupby(["arch", "seed"]).apply(
                lambda g: pd.Series({"mae": (g["mae"] * g["days"]).sum() / g["days"].sum(),
                                     "best_epoch": g["best_epoch"].mean()}), include_groups=False).reset_index()
            agg = per_seed.groupby("arch").agg(mae_mean=("mae", "mean"), mae_std=("mae", "std")).round(3)
            epochs = sm.groupby("arch")["best_epoch"].agg(["min", "max"])
            lines += ["**传统基线种子稳定性（每种子总体MAE，3种子）**", "",
                      "| 架构 | MAE mean±std | best_epoch范围 |", "|---|---|---|"]
            for arch, r in agg.iterrows():
                lines.append(f"| {arch} | {r['mae_mean']:.3f}±{r['mae_std']:.3f} | {epochs.loc[arch, 'min']:.0f}~{epochs.loc[arch, 'max']:.0f} |")
            lines.append("")

    lines += ["## 口径说明", ""]
    lines += [f"- **{k}**: {v}" for k, v in NOTES.items()]
    lines += ["- 传统基线每站独立训练, 输入28天[蓝藻日最高值,Tmax,降水,辐射,缺测mask], 天气只用≤origin; "
              "训练=2024全年+2025/2026年5~8月(标签仅实测日,H3要求3天全实测), 验证=2025-09早停, 3种子分位数平均。",
              "- 覆盖率类区间指标不作为主指标（用户关注点预测精度与峰值），仍保留在各 details CSV 中。",
              "- 信息集三档: 纯TSFM仅蓝藻历史(零样本); 传统基线多了天气历史与2024~2026监督训练; "
              "RSI在TSFM之上叠加天气修正与跨日memory。",
              f"- 加权指标权重 w=1+{args.alpha}·y/max(y) 在总体/逐站组内分别归一化, α 可用 --alpha 调整。"]
    (OUT / "2026_max_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    for task, df in all_tables.items():
        print(f"\n== {task} ==")
        print(df[df["station"] == "overall"].to_string(index=False))
    print(f"\n已写出 {OUT / '2026_max_comparison.md'} 及各 comparison CSV")


if __name__ == "__main__":
    main()