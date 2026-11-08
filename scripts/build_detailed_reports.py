"""生成 2026-09 H1/H3 详细对比 Markdown。

H3 的 RSI 版本固定为: 未来天气 + 放宽护栏 + cap 0.10
  data/replay/2026_rsi_max_h3_fwx_relax_cap0.1_details.csv
H1 的 RSI 版本固定为: 在线逐站门控 + 一步偏差修正 (dev-selected)
  data/replay/2026_rsi_v1_1_gate_bias_devsel_details.csv

对照仅包含 CNN-LSTM / GRU / LSTM / Transformer, 按用户要求不写原始 TSFM。

输出:
  data/replay/2026_h3_detailed_comparison.md
  data/replay/2026_h1_detailed_comparison.md

Usage:
    python scripts/build_detailed_reports.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "replay"
ALPHA = 2.0
STATIONS = ["jiangjunyan", "meichengmatou", "yanlingwu", "yanziling"]
STATION_CN = {
    "jiangjunyan": "将军岩",
    "meichengmatou": "美称码头",
    "yanlingwu": "严陵坞",
    "yanziling": "严子陵",
}

H3_SYSTEMS = [
    ("RSI-H3（未来天气+放宽护栏）", "2026_rsi_max_h3_fwx_relax_cap0.1_details.csv"),
    ("CNN-LSTM", "2026_cnn_lstm_max_h3_ctx28_details.csv"),
    ("GRU", "2026_gru_max_h3_ctx28_details.csv"),
    ("LSTM", "2026_lstm_max_h3_ctx28_details.csv"),
    ("Transformer", "2026_transformer_max_h3_ctx28_details.csv"),
]
H1_SYSTEMS = [
    ("RSI-H1 v1.1（门控+偏差修正）", "2026_rsi_v1_1_gate_bias_devsel_details.csv"),
    ("CNN-LSTM", "2026_cnn_lstm_max_h1_ctx28_details.csv"),
    ("GRU", "2026_gru_max_h1_ctx28_details.csv"),
    ("LSTM", "2026_lstm_max_h1_ctx28_details.csv"),
    ("Transformer", "2026_transformer_max_h1_ctx28_details.csv"),
]


def fmt(x, nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)) or pd.isna(x):
        return "—"
    return f"{x:.{nd}f}"


def pct(x) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{100 * x:.1f}%"


def load(fname: str, h3: bool) -> pd.DataFrame:
    d = pd.read_csv(OUT / fname)
    if "scored" in d.columns:
        d = d[d["scored"] == True].copy()
    d["date"] = pd.to_datetime(d["date"])
    d["origin"] = pd.to_datetime(d["origin"])
    if "covered" not in d.columns:
        d["covered"] = np.nan
    return d


def metrics(g: pd.DataFrame) -> dict:
    y = g["actual"].to_numpy(dtype=float)
    p = g["point"].to_numpy(dtype=float)
    e = p - y
    w = 1.0 + ALPHA * y / y.max()
    return {
        "n": len(g),
        "mae": float(np.abs(e).mean()),
        "rmse": float(np.sqrt((e ** 2).mean())),
        "wmae": float((w * np.abs(e)).sum() / w.sum()),
        "wrmse": float(np.sqrt((w * e ** 2).sum() / w.sum())),
        "pre": float(abs(y.max() - p.max()) / y.max()),
        "coverage": float(g["covered"].mean()) if g["covered"].notna().any() else np.nan,
    }


def metric_table(frames: dict[str, pd.DataFrame], title: str, note: str = "") -> list[str]:
    lines = [f"### {title}", ""]
    if note:
        lines += [note, ""]
    lines += ["| 方法 | 样本数 | MAE | RMSE | W-MAE | W-RMSE | PRE | 覆盖率 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    rows = [(name, metrics(d)) for name, d in frames.items()]
    best = {}
    for k in ("mae", "rmse", "wmae", "wrmse", "pre"):
        best[k] = min(m[k] for _, m in rows)
    covs = [m["coverage"] for _, m in rows if np.isfinite(m["coverage"])]
    best_cov = max(covs) if covs else np.nan

    def cell(k: str, v: float) -> str:
        return f"**{fmt(v)}**" if np.isfinite(v) and abs(v - best[k]) < 1e-12 else fmt(v)

    for name, m in sorted(rows, key=lambda x: x[1]["mae"]):
        cov = pct(m["coverage"])
        if np.isfinite(m["coverage"]) and np.isfinite(best_cov) and abs(m["coverage"] - best_cov) < 1e-12:
            cov = f"**{cov}**"
        lines.append(f"| {name} | {m['n']} | {cell('mae', m['mae'])} | {cell('rmse', m['rmse'])} | "
                     f"{cell('wmae', m['wmae'])} | {cell('wrmse', m['wrmse'])} | {cell('pre', m['pre'])} | {cov} |")
    lines.append("")
    return lines


def rank_summary(metric_by_group: pd.DataFrame, group_label: str) -> list[str]:
    """metric_by_group: index=group, columns=system, values=lower-is-better metric."""
    ranks = metric_by_group.rank(axis=1, method="min")
    lines = [f"| 方法 | {group_label}最优/并列次数 | 平均排名 | 平均误差 |",
             "|---|---:|---:|---:|"]
    n = len(metric_by_group)
    for col in metric_by_group.columns:
        wins = int((metric_by_group[col] <= metric_by_group.min(axis=1) + 1e-12).sum())
        lines.append(f"| {col} | {wins}/{n} | {ranks[col].mean():.2f} | {metric_by_group[col].mean():.3f} |")
    lines.append("")
    return lines


def md_header(title: str, intro: list[str]) -> list[str]:
    return [f"# {title}", ""] + intro + [""]


def build_h3() -> Path:
    frames = {name: load(fname, h3=True) for name, fname in H3_SYSTEMS}
    key = ["station", "origin", "date", "h"]
    base = set(map(tuple, frames[H3_SYSTEMS[0][0]][key].to_numpy()))
    for name, d in frames.items():
        got = set(map(tuple, d[key].to_numpy()))
        assert got == base, f"{name} 样本集合不一致"

    lines = md_header("2026-09 H3 详细对比：预测未来 3 天逐日最高值", [
        "目标变量：`cyano_ugl_max`（日最高蓝藻浓度）。H3 = 在 origin=t 预测 t+1、t+2、t+3 三天的逐日最高值；"
        "只有三天全部有实测值的窗口才评分。",
        "",
        "**样本口径**：4 个测站，103 个三天全实测窗口，共 309 行（每窗口 3 行）。",
        "",
        "**RSI-H3 版本**：未来天气 + 放宽护栏 + cap 0.10（MAE 2.298）。"
        "未来天气为逐日实测/再分析值（非预报存档），只注入 RSI 的融合环节；"
        "放宽护栏允许 point 超出原始 q10/q90，但仍限制在原始点预测 ±0.10×区间宽内，"
        "且修正后区间宽度不得小于原始宽度的 50%。**这是上界实验，不是可直接上线的预报系统。**",
        "",
        "**对照方法**：CNN-LSTM、GRU、LSTM、Transformer（均值池化）。"
        "四者均使用 2024 全年 + 2025/2026 年 5–8 月训练，输入为蓝藻与历史天气，不使用未来天气。"
        "按用户要求，本文不包含原始 TSFM（Chronos-2）。",
        "",
        "**指标定义**：MAE、RMSE；W-MAE = Σw|e|/Σw，W-RMSE = √(Σw·e²/Σw)，"
        "其中 w = 1 + 2·y/max(y)，权重在总体/逐站/逐 horizon 组内分别用该组 max(y) 归一化；"
        "PRE = |max(y) − max(ŷ)| / max(y)。覆盖率 = 实测值落入 [q10, q90] 的比例。",
        "",
        "> 说明：表中每个“窗口”对应一个 origin；d1/d2/d3 分别为 t+1/t+2/t+3。"
        "预测值保留 3 位小数；绝对误差表中每行加粗的是当天 5 个方法中误差最小的值（并列时都加粗）。",
    ])

    lines += ["## 1. 总体指标", ""]
    lines += metric_table(frames, "全部测站（309 行）")
    overall_cmp = pd.DataFrame({
        name: [metrics(d)["mae"]] for name, d in frames.items()
    }, index=["MAE"])
    best = overall_cmp.iloc[0].idxmin()
    lines += [f"**总体 MAE 最优：{best}（{fmt(overall_cmp.iloc[0][best])}）。**", ""]

    lines += ["## 2. 逐测站指标", ""]
    for st in STATIONS:
        sub = {name: d[d["station"] == st] for name, d in frames.items()}
        lines += metric_table(sub, f"{STATION_CN[st]}（{st}，{metrics(sub[H3_SYSTEMS[0][0]])['n']} 行）")

    lines += ["## 3. 逐 horizon 指标", ""]
    for h in (1, 2, 3):
        sub = {name: d[d["h"] == h] for name, d in frames.items()}
        lines += metric_table(sub, f"t+{h}（{metrics(sub[H3_SYSTEMS[0][0]])['n']} 行）")

    # 每个窗口的 3 日 MAE + 最优方法
    win = {}
    for name, d in frames.items():
        win[name] = d.assign(ae=(d["point"] - d["actual"]).abs()).groupby(["station", "origin"], as_index=False)["ae"].mean()
    wm = win[H3_SYSTEMS[0][0]][["station", "origin"]].copy()
    for name, d in win.items():
        wm = wm.merge(d.rename(columns={"ae": name}), on=["station", "origin"], how="left")
    sys_names = [n for n, _ in H3_SYSTEMS]
    wm["best"] = wm[sys_names].idxmin(axis=1)
    wm["best_mae"] = wm[sys_names].min(axis=1)

    lines += ["## 4. 每个窗口的 3 日平均绝对误差与最优方法", ""]
    lines += ["### 4.1 总体最优次数与平均排名", ""]
    lines += rank_summary(wm.set_index(["station", "origin"])[sys_names], "窗口")
    lines += ["### 4.2 逐测站统计", ""]
    for st in STATIONS:
        g = wm[wm["station"] == st].set_index("origin")[sys_names]
        lines += [f"#### {STATION_CN[st]}（{st}）", ""]
        lines += rank_summary(g, "窗口")

    # 每个窗口的 3 日 MAE 明细
    targets_all = frames[H3_SYSTEMS[0][0]].pivot(index=["station", "origin"], columns="h", values="date")
    targets_all = targets_all.apply(lambda r: f"{r[1].strftime('%m-%d')}~{r[3].strftime('%m-%d')}", axis=1)
    lines += ["### 4.3 逐窗口 3 日 MAE 明细", ""]
    for st in STATIONS:
        g = wm[wm["station"] == st].sort_values("origin")
        lines += [f"#### {STATION_CN[st]}（{st}）", ""]
        head = ["窗口起点", "目标日期"] + sys_names + ["窗口最优"]
        lines += ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
        for _, r in g.iterrows():
            row = [pd.Timestamp(r["origin"]).strftime("%m-%d"),
                   targets_all.loc[(st, r["origin"])]]
            for name in sys_names:
                v = r[name]
                row.append(f"**{fmt(v)}**" if abs(v - r["best_mae"]) < 1e-12 else fmt(v))
            row.append(str(r["best"]))
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    # 每个窗口的 3 日预测宽表 + 绝对误差宽表
    lines += ["## 5. 逐窗口三天预测明细", ""]
    for st in STATIONS:
        sub_frames = {name: frames[name][frames[name]["station"] == st].copy() for name, _ in H3_SYSTEMS}
        origins = sorted(sub_frames[H3_SYSTEMS[0][0]]["origin"].unique())
        lines += [f"### 5.{STATIONS.index(st) + 1} {STATION_CN[st]}（{st}）", "",
                  f"{STATION_CN[st]} 共有 {len(origins)} 个三天全实测窗口。", ""]

        actual = sub_frames[H3_SYSTEMS[0][0]].pivot(index="origin", columns="h", values="actual").sort_index()
        dates = sub_frames[H3_SYSTEMS[0][0]].pivot(index="origin", columns="h", values="date").sort_index()
        targets = dates.apply(lambda r: f"{r[1].strftime('%m-%d')}~{r[3].strftime('%m-%d')}", axis=1)

        pred = {}
        for name, _ in H3_SYSTEMS:
            p = sub_frames[name].pivot(index="origin", columns="h", values="point").sort_index()
            pred[name] = p

        head = ["窗口起点", "目标日期", "实际 d1", "实际 d2", "实际 d3"]
        for name, _ in H3_SYSTEMS:
            head += [f"{name} d1", f"{name} d2", f"{name} d3"]
        lines += ["| " + " | ".join(head) + " |",
                  "|" + "|".join(["---"] * len(head)) + "|"]
        for origin in origins:
            row = [origin.strftime("%m-%d"), targets.loc[origin],
                   fmt(actual.loc[origin, 1]), fmt(actual.loc[origin, 2]), fmt(actual.loc[origin, 3])]
            for name, _ in H3_SYSTEMS:
                row += [fmt(pred[name].loc[origin, 1]), fmt(pred[name].loc[origin, 2]), fmt(pred[name].loc[origin, 3])]
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

        # 绝对误差宽表, 每天加粗最小
        lines += [f"#### {STATION_CN[st]} 逐日绝对误差", ""]
        ehead = ["窗口起点"]
        for name, _ in H3_SYSTEMS:
            ehead += [f"{name} e1", f"{name} e2", f"{name} e3"]
        lines += ["| " + " | ".join(ehead) + " |",
                  "|" + "|".join(["---"] * len(ehead)) + "|"]
        for origin in origins:
            row = [origin.strftime("%m-%d")]
            vals = {}
            for name, _ in H3_SYSTEMS:
                es = []
                for h in (1, 2, 3):
                    es.append(abs(pred[name].loc[origin, h] - actual.loc[origin, h]))
                vals[name] = es
            for h_idx in range(3):
                best_val = min(vals[name][h_idx] for name, _ in H3_SYSTEMS)
                for name, _ in H3_SYSTEMS:
                    v = vals[name][h_idx]
                    row.append(f"**{fmt(v)}**" if abs(v - best_val) < 1e-12 else fmt(v))
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

        # RSI 最终区间明细
        lines += [f"#### {STATION_CN[st]} RSI-H3 每日 point / q10 / q90 / 覆盖率", "",
                  "| 窗口起点 | 目标日期 | h | point | q10 | q90 | 实际 | 覆盖 |",
                  "|---|---|---:|---:|---:|---:|---:|---|"]
        rsi = sub_frames[H3_SYSTEMS[0][0]].sort_values(["origin", "h"])
        for _, r in rsi.iterrows():
            lines.append(f"| {pd.Timestamp(r['origin']).strftime('%m-%d')} | "
                         f"{pd.Timestamp(r['date']).strftime('%m-%d')} | {int(r['h'])} | "
                         f"{fmt(r['point'])} | {fmt(r['q10'])} | {fmt(r['q90'])} | "
                         f"{fmt(r['actual'])} | {'是' if bool(r['covered']) else '否'} |")
        lines.append("")

    lines += ["## 6. 复现与文件", "",
              "```bash",
              "# 生成 H3 RSI 版本（未来天气 + 放宽护栏 + cap 0.10）",
              "python scripts/replay_rsi.py --year 2026 --month 9 --target max --horizon 3 \\",
              "  --future-weather --relax-h3-guard --h3-point-cap 0.1",
              "",
              "# 生成本文档",
              "python scripts/build_detailed_reports.py",
              "```",
              "",
              "| 文件 | 内容 |",
              "|---|---|",
              "| `2026_rsi_max_h3_fwx_relax_cap0.1_details.csv` | RSI-H3 逐行预测（309 行） |",
              "| `2026_cnn_lstm_max_h3_ctx28_details.csv` | CNN-LSTM 逐行预测 |",
              "| `2026_gru_max_h3_ctx28_details.csv` | GRU 逐行预测 |",
              "| `2026_lstm_max_h3_ctx28_details.csv` | LSTM 逐行预测 |",
              "| `2026_transformer_max_h3_ctx28_details.csv` | Transformer（均值池化）逐行预测 |",
              "| `2026_h3_detailed_comparison.md` | 本文件 |",
              ""]

    path = OUT / "2026_h3_detailed_comparison.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def build_h1() -> Path:
    frames = {name: load(fname, h3=False) for name, fname in H1_SYSTEMS}
    key = ["station", "date"]
    base = set(map(tuple, frames[H1_SYSTEMS[0][0]][key].to_numpy()))
    for name, d in frames.items():
        got = set(map(tuple, d[key].to_numpy()))
        assert got == base, f"{name} 样本集合不一致"

    lines = md_header("2026-09 H1 详细对比：预测未来 1 天日最高值", [
        "目标变量：`cyano_ugl_max`（日最高蓝藻浓度）。H1 = 在 origin=t 预测 t+1 一天的最高值；"
        "只有 t+1 有实测值的站-日才评分。",
        "",
        "**样本口径**：4 个测站，113 个已观测站-日。",
        "",
        "**RSI-H1 版本**：在线逐站门控 + 一步偏差修正（dev-selected），2026-09 MAE 1.689。"
        "策略完全因果：每个 origin 只用此前已实现的误差决定用 v1 还是 raw，再用最近 k 次带符号误差均值做修正；"
        "超参数仅用 2025-09 dev 选择（margin=−0.5, k=1, λ=0.1），未使用 2026 测试月调参。"
        "该版本目前仍是候选：dev 增益只有 0.020，且 dev 内部选参不稳定；"
        "canonical H1 仍为 RSI v1（MAE 1.747）。本文按“效果最好的 H1 RSI 版本”展示 v1.1。",
        "",
        "**对照方法**：CNN-LSTM、GRU、LSTM、Transformer（均值池化）。"
        "四者均使用 2024 全年 + 2025/2026 年 5–8 月训练，输入为蓝藻与历史天气。"
        "按用户要求，本文不包含原始 TSFM（Chronos-2）。",
        "",
        "**指标定义**：MAE、RMSE；W-MAE = Σw|e|/Σw，W-RMSE = √(Σw·e²/Σw)，"
        "其中 w = 1 + 2·y/max(y)，权重在总体/逐站组内分别用该组 max(y) 归一化；"
        "PRE = |max(y) − max(ŷ)| / max(y)。覆盖率 = 实测值落入 [q10, q90] 的比例；"
        "RSI-H1 v1.1 是点预测修正，不输出区间，因此覆盖率记为 —。",
        "",
        "> 说明：表中每行对应一个站-日。绝对误差表中每行加粗的是当天 5 个方法中误差最小的值（并列时都加粗）。",
    ])

    lines += ["## 1. 总体指标", ""]
    lines += metric_table(frames, "全部测站（113 个站-日）")
    overall_cmp = pd.DataFrame({name: [metrics(d)["mae"]] for name, d in frames.items()}, index=["MAE"])
    best = overall_cmp.iloc[0].idxmin()
    lines += [f"**总体 MAE 最优：{best}（{fmt(overall_cmp.iloc[0][best])}）。**", ""]

    lines += ["## 2. 逐测站指标", ""]
    for st in STATIONS:
        sub = {name: d[d["station"] == st] for name, d in frames.items()}
        lines += metric_table(sub, f"{STATION_CN[st]}（{st}，{metrics(sub[H1_SYSTEMS[0][0]])['n']} 个站-日）")

    sys_names = [n for n, _ in H1_SYSTEMS]
    err = frames[H1_SYSTEMS[0][0]][["station", "date"]].copy()
    for name, d in frames.items():
        err = err.merge(d[["station", "date", "point", "actual"]].rename(columns={
            "point": f"{name}_point", "actual": f"{name}_actual"}), on=["station", "date"], how="left")
        err[f"{name}_err"] = (err[f"{name}_point"] - err[f"{name}_actual"]).abs()

    lines += ["## 3. 逐日最优次数与平均排名", ""]
    err["best"] = err[[f"{n}_err" for n in sys_names]].idxmin(axis=1).str.replace("_err", "", regex=False)
    lines += ["### 3.1 总体", ""]
    lines += rank_summary(err.set_index(["station", "date"])[[f"{n}_err" for n in sys_names]].rename(
        columns={f"{n}_err": n for n in sys_names}), "站-日")
    lines += ["### 3.2 逐测站", ""]
    for st in STATIONS:
        g = err[err["station"] == st].set_index("date")[[f"{n}_err" for n in sys_names]].rename(
            columns={f"{n}_err": n for n in sys_names})
        lines += [f"#### {STATION_CN[st]}（{st}）", ""]
        lines += rank_summary(g, "站-日")

    lines += ["## 4. 逐日预测与绝对误差明细", ""]
    for st in STATIONS:
        g = err[err["station"] == st].sort_values("date")
        lines += [f"### 4.{STATIONS.index(st) + 1} {STATION_CN[st]}（{st}，{len(g)} 个站-日）", ""]
        head = ["日期", "实际值"]
        for name, _ in H1_SYSTEMS:
            head += [f"{name} 预测", f"{name} 误差"]
        lines += ["| " + " | ".join(head) + " |",
                  "|" + "|".join(["---"] * len(head)) + "|"]
        for _, r in g.iterrows():
            row = [pd.Timestamp(r["date"]).strftime("%m-%d"), fmt(r[f"{H1_SYSTEMS[0][0]}_actual"])]
            best_val = min(abs(r[f"{n}_err"]) for n in sys_names)
            for name, _ in H1_SYSTEMS:
                v = abs(r[f"{name}_err"])
                row += [fmt(r[f"{name}_point"]), f"**{fmt(v)}**" if abs(v - best_val) < 1e-12 else fmt(v)]
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    lines += ["## 5. 复现与文件", "",
              "```bash",
              "# H1 RSI v1.1（在线门控 + 偏差修正, dev 选参 + 2026 迁移）",
              "python scripts/eval_online_gate_bias.py",
              "",
              "# 生成本文档",
              "python scripts/build_detailed_reports.py",
              "```",
              "",
              "| 文件 | 内容 |",
              "|---|---|",
              "| `2026_rsi_v1_1_gate_bias_devsel_details.csv` | RSI-H1 v1.1 逐行预测（113 行） |",
              "| `2026_cnn_lstm_max_h1_ctx28_details.csv` | CNN-LSTM 逐行预测 |",
              "| `2026_gru_max_h1_ctx28_details.csv` | GRU 逐行预测 |",
              "| `2026_lstm_max_h1_ctx28_details.csv` | LSTM 逐行预测 |",
              "| `2026_transformer_max_h1_ctx28_details.csv` | Transformer（均值池化）逐行预测 |",
              "| `2026_h1_detailed_comparison.md` | 本文件 |",
              ""]

    path = OUT / "2026_h1_detailed_comparison.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    p3 = build_h3()
    p1 = build_h1()
    print("已写出:")
    print(" ", p3, f"({p3.stat().st_size / 1024:.1f} KB)")
    print(" ", p1, f"({p1.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
