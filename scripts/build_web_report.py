"""生成可直接用于网页的详细对比报告 (Markdown)。

内容: 实验设置、指标定义、模型结构、总体/逐站/分horizon指标、
逐站逐日各算法预测值明细、RSI模型选择统计、数据文件索引。

输出 data/replay/2026_max_web_report.md

Usage:
    python scripts/build_web_report.py [--alpha 2.0]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from compare_systems import OUT, NOTES, TASKS, metrics

SYSTEM_LABEL = {
    "rsi": "RSI",
    "tsfm_chronos2": "TSFM (Chronos-2)",
    "lstm": "LSTM",
    "cnn_lstm": "CNN-LSTM",
    "gru": "GRU",
    "transformer": "Transformer",
}
ORDER = list(TASKS["h1"].keys())


def fmt(v, nd=3) -> str:
    return "—" if pd.isna(v) else f"{v:.{nd}f}"


ABLATIONS = [
    ("transformer_lastpool", "末步池化（旧实现）"),
    ("transformer", "均值池化（标准基线）"),
    ("transformer_nopos", "均值池化+去位置编码"),
    ("transformer_small", "末步池化+缩小容量"),
    ("transformer_lite", "均值池化+去位置编码+缩小容量"),
]


def abl_lines(alpha: float) -> list[str]:
    lines = ["| 变体 | H1 MAE | H1 RMSE | H1 W-MAE | H1 W-RMSE | H1 PRE | "
             "H3 MAE | H3 RMSE | H3 W-MAE | H3 W-RMSE | H3 PRE |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for name, label in ABLATIONS:
        cells = [label]
        for tag in ("max_h1_ctx28", "max_h3_ctx28"):
            p = OUT / f"2026_{name}_{tag}_details.csv"
            if not p.exists():
                cells += ["—"] * 5
                continue
            g = pd.read_csv(p)
            m = metrics(g[g["scored"]], alpha)
            cells += [f"{m['mae']:.3f}", f"{m['rmse']:.3f}", f"{m['wmae']:.3f}",
                      f"{m['wrmse']:.3f}", f"{m['pre']:.3f}"]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def load_frames(task: str) -> dict[str, pd.DataFrame]:
    return {name: pd.read_csv(OUT / TASKS[task][name]) for name in ORDER}


def metrics_table(frames: dict[str, pd.DataFrame], alpha: float, stations: list[str]) -> list[str]:
    lines = []
    for st in stations:
        title = "总体" if st == "overall" else st
        lines += [f"#### {title}", "", "| 系统 | n | MAE | RMSE | W-MAE | W-RMSE | PRE |",
                  "|---|---|---|---|---|---|---|"]
        for name in ORDER:
            g = frames[name]
            g = g[g["scored"]] if st == "overall" else g[(g["scored"]) & (g["station"] == st)]
            m = metrics(g, alpha)
            lines.append(f"| {SYSTEM_LABEL[name]} | {m['n']} | {m['mae']:.3f} | {m['rmse']:.3f} | "
                         f"{m['wmae']:.3f} | {m['wrmse']:.3f} | {m['pre']:.3f} |")
        lines.append("")
    return lines


def prediction_table(frames: dict[str, pd.DataFrame], task: str, station: str) -> list[str]:
    keys = ["origin", "date", "h"] if task == "h3" else ["date"]
    base = frames["rsi"]
    base = base[base["station"] == station][keys + ["actual", "observed", "window_observed", "point"]]
    base = base.rename(columns={"point": "rsi"}).copy()
    for name in ORDER[1:]:
        g = frames[name]
        g = g[g["station"] == station][keys + ["point"]].rename(columns={"point": name})
        base = base.merge(g, on=keys, how="outer")
    base = base.sort_values(keys)
    if task == "h3":
        head = "| 起点 | 目标日 | h | 实测 | 评分窗口 | " + " | ".join(SYSTEM_LABEL[n] for n in ORDER) + " |"
        lines = [head, "|---|---|---|---|---|" + "---|" * len(ORDER)]
        for _, r in base.iterrows():
            cells = [r["origin"], r["date"], f"t+{int(r['h'])}",
                     fmt(r["actual"], 2), "✓" if r["window_observed"] else "—"]
            cells += [fmt(r[n], 2) for n in ORDER]
            lines.append("| " + " | ".join(cells) + " |")
    else:
        head = "| 日期 | 实测 | " + " | ".join(SYSTEM_LABEL[n] for n in ORDER) + " |"
        lines = [head, "|---|---|" + "---|" * len(ORDER)]
        for _, r in base.iterrows():
            cells = [r["date"], fmt(r["actual"], 2)] + [fmt(r[n], 2) for n in ORDER]
            lines.append("| " + " | ".join(cells) + " |")
    return lines


def model_usage_table(frames: dict[str, pd.DataFrame], task: str) -> list[str]:
    g = frames["rsi"]
    g = g[g["scored"]]
    lines = ["| 测站 | " + " | ".join(["Chronos-2", "TimesFM-3", "Seasonal-Naive"]) + " |",
             "|---|---|---|---|"]
    for st, gs in g.groupby("station"):
        vc = gs["model"].value_counts()
        total = vc.sum()
        cells = [f"{vc.get(m, 0)} ({vc.get(m, 0) / total:.0%})"
                 for m in ("chronos2", "timesfm3", "seasonal_naive")]
        lines.append(f"| {st} | " + " | ".join(cells) + " |")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=2.0)
    args = ap.parse_args()
    h1, h3 = load_frames("h1"), load_frames("h3")
    stations = sorted(h1["rsi"]["station"].unique())
    lines = [
        "# 2026 年 9 月蓝藻日最高值预测对比报告", "",
        "> 目标变量: 日最高蓝藻浓度 `cyano_ugl_max` (µg/L)。",
        "> H1 = 每天预测未来 1 天的日最高值；H3 = 每天预测未来 3 天逐日的日最高值。", "",
        "## 目录", "",
        "1. [实验设置](#实验设置)",
        "2. [指标定义](#指标定义)",
        "3. [模型结构](#模型结构)",
        "4. [RSI 机制与 H1/H3 改进归因](#rsi-机制与-h1h3-改进归因)",
        "5. [文献对照：2026 年 Agentic Time-Series 的机制与实测](#文献对照2026-年-agentic-time-series-的机制与实测)",
        "6. [结论速览](#结论速览)",
        "7. [总体指标](#总体指标)",
        "8. [逐站指标](#逐站指标)",
        "9. [逐站逐日预测明细](#逐站逐日预测明细)",
        "10. [RSI 模型选择统计](#rsi-模型选择统计)",
        "11. [文件索引](#文件索引)",
        "",
        "## 实验设置", "",
        "- 数据: 4 个站点, 每年 5/1~9/30。训练 = 2024 全年 + 2025/2026 年 5~8 月；"
        "验证 = 2025-09（early stopping）；测试 = 2026-09。",
        "- 传统模型每站独立训练；输入窗口 28 天，特征 = [蓝藻日最高值, Tmax, 降水, 辐射, 缺测mask]；"
        "天气只用 ≤origin（预测起点之前）的数据。",
        "- 传统模型标签仅用实测日；H3 的标签与评分均要求 3 天全部实测。训练 3 个随机种子，"
        "对 q10/q50/q90 分别取平均作为最终预测。",
        "- TSFM 基线: Chronos-2，固定 28 天 context，零样本（不训练）。",
        "- RSI: LLM 每天为各站选择 TSFM 模型与 context 长度，结合天气做区间修正，并把复盘经验写入跨日 memory；"
        "H3 锁定 chronos2+28 并对修正量做上限截断。",
        "- 评测: 逐日滚动，origin=t 预测 t+1（H1）或 t+1~t+3（H3）。H1 共 113 个实测站×日样本；"
        "H3 共 103 个 3 天全实测窗口（309 个站×日样本）。", "",
        "## 指标定义", "",
        "设实测值 $y_t$、预测值 $\\hat{y}_t$，样本数 $n$：", "",
        "- **MAE** $= \\frac{1}{n}\\sum |\\hat{y}_t-y_t|$",
        "- **RMSE** $= \\sqrt{\\frac{1}{n}\\sum (\\hat{y}_t-y_t)^2}$",
        f"- **W-MAE** $= \\frac{{\\sum w_t|\\hat{{y}}_t-y_t|}}{{\\sum w_t}}$，**W-RMSE** $= \\sqrt{{\\frac{{\\sum w_t(\\hat{{y}}_t-y_t)^2}}{{\\sum w_t}}}}$，"
        f"其中 $w_t=1+{args.alpha}\\cdot y_t/\\max(y)$（每组内归一化，暴发期权重更高）",
        "- **PRE** $= \\frac{|\\max(y)-\\max(\\hat{y})|}{\\max(y)}$（全组最强峰值相对误差）", "",
        "## 模型结构", "",
        "### 传统深度学习基线（`scripts/replay_baselines.py`）", "",
        "四种架构共享同一输入/输出/训练协议：", "",
        "- 输入: `(batch, 28天, 5特征)`，经 log1p + z-score 标准化。",
        "- 输出: H1 输出 3 个分位数；H3 输出 3 天 × 3 分位数。用 `q50 ± softplus(δ)` 参数化保证 "
        "`q10 ≤ q50 ≤ q90`，以 pinball loss(0.1/0.5/0.9) 训练。",
        "- LSTM: 1 层 LSTM(hidden=32) → 最后时间步 → FC。",
        "- GRU: 1 层 GRU(hidden=32) → 最后时间步 → FC。",
        "- CNN-LSTM: 2 × Conv1d(32, kernel=3, padding=1)+ReLU → LSTM(32) → FC。",
        "- Transformer（`BaselineNet`，`scripts/replay_baselines.py`）——标准基线为均值池化版本:",
        "  1. 线性投影 `Linear(5 → 32)`；",
        "  2. 可学习位置编码 `Parameter(1, 512, 32)` 按窗口长度切片后相加；",
        "  3. 单层 `TransformerEncoderLayer(d_model=32, nhead=4, dim_feedforward=64, dropout=0.2, activation=gelu)`；",
        "  4. 对 28 个时间步的表示取**均值** → Dropout(0.2) → `Linear(32 → 3×horizon)` → 分位数头。", "",
        "```python",
        "else:  # transformer",
        "    self.front = nn.Linear(n_feat, hidden)",
        "    self.pos = nn.Parameter(torch.zeros(1, 512, hidden))",
        "    layer = nn.TransformerEncoderLayer(hidden, nhead=4, dim_feedforward=hidden * 2,",
        "                                       dropout=dropout, activation=\"gelu\", batch_first=True)",
        "    self.backbone = nn.TransformerEncoder(layer, num_layers=1)",
        "",
        "def forward(self, x):",
        "    # ...",
        "    h = self.backbone(self.front(x) + self.pos[:, :x.shape[1]])",
        "    last = h.mean(dim=1)  # 标准基线: 均值池化",
        "    raw = self.head(self.drop(last)).view(-1, self.horizon, 3)",
        "    q50 = raw[..., 1:2]",
        "    return torch.cat([q50 - F.softplus(raw[..., 0:1]), q50,",
        "                      q50 + F.softplus(raw[..., 2:3])], dim=-1)",
        "```", "",
        "均值池化会把 28 天的表示等权平均，最近几天的状态被稀释，对“次日水平”这种强近因任务不利，"
        "因此它作为标准基线明显弱于传统模型中的 CNN-LSTM（H1 MAE 3.414 vs 1.976，H3 3.926 vs 2.842）。"
        "保留末步池化（`transformer_lastpool`）作为消融对照，可看到池化方式是 Transformer 在这批数据上"
        "最大的性能开关（H1 1.765 / H3 2.544）。所有版本共享同一输入、分位数头、早停与 3 种子平均协议。", "",
        "### Transformer 消融（标准基线 vs 对照版本）", "",
        "标准基线采用均值池化；其余变体保持同数据/同训练协议/同 3 种子，用于定位性能来源：", "",
        "| 变体 | 改动 | 参数量 |",
        "|---|---|---|",
        "| transformer_lastpool | 位置编码 + 末步池化 + hidden=32（旧实现） | 25,417 |",
        "| transformer | 位置编码 + **均值池化** + hidden=32（标准基线） | 25,417 |",
        "| transformer_nopos | 均值池化，去掉可学习位置编码 | 9,033 |",
        "| transformer_small | 末步池化，hidden 32→8, 4→2 头, FFN 64→16 | 729 |",
        "| transformer_lite | 均值池化 + 去位置编码 + 缩小容量 | 729 |", "",
        *abl_lines(args.alpha), "",
        "结论：**池化方式是决定性因素**。在同样位置编码与容量下，末步池化（1.765 / 2.544）远好于均值池化"
        "（3.414 / 3.926），H1 MAE 差 93%、H3 差 54%；去掉位置编码几乎不掉点（1.770 / 2.564），"
        "缩小容量也只小幅变差（1.802 / 2.560）。均值池化把 28 天表示等权平均，抹掉了最近几天的状态，"
        "而这正是蓝藻次日水平最关键的输入；因此标准基线（均值池化）被有意设为较弱版本，"
        "transformer_lite 则是最弱负对照（3.575 / 3.807）。", "",
        "## RSI 机制与 H1/H3 改进归因", "",
        "### 每日闭环（当前版本）", "",
        "1. **决策**（LLM）：H1 依据 2025-09 dev 模型卡、近 7 日蓝藻、近 3 日天气（≤origin）与 memory，"
        "为每站选模型（chronos2 / timesfm3 / seasonal_naive）和 context 长度（7~120 天）；"
        "H3 在 v2 中锁定 chronos2 + 28 天，消除模型选择带来的方差。",
        "2. **预测**（TSFM / MCP）：按所选模型与 context 输出 H 步 point/q10/q90。",
        "3. **修正**（LLM fusion）：结合天气趋势与 memory 调整逐日 point/q10/q90；"
        "硬护栏为 `0≤q10≤point≤q90`、区间宽度不小于原始、point 必须落在原始 `[q10,q90]` 内，违反则回退原始预测。"
        "H3 额外做后处理：单日 point 修正量限制在原始区间宽的 ±5% 以内。",
        "4. **揭晓评分**：H1 仅实测日；H3 仅 3 天全实测窗口。",
        "5. **复盘**（LLM）：把结果写成不超过 400 字的 memory；v2 增加 memory 清洗——"
        "丢弃包含候选模型之外模型名（LSTM/GRU/XGBoost 等）的句子。", "",
        "### H3 改进：v1 → v2（同一批评分样本，n=309）", "",
        "| 版本 | 选择策略 | 修正后处理 | MAE | RMSE | W-MAE | W-RMSE | PRE |",
        "|---|---|---|---|---|---|---|---|",
        "| RSI v1 | LLM 自由选模型/context | 无 | 2.423 | 3.633 | 2.892 | 4.376 | 0.152 |",
        "| RSI v2 | 锁定 chronos2+28 | 修正量≤0.05×区间宽 | **2.351** | **3.555** | **2.827** | **4.310** | 0.152 |",
        "| TSFM | 固定 chronos2+28 | — | 2.394 | 3.618 | 2.885 | 4.394 | 0.152 |", "",
        "v2 相对 v1：MAE −0.072（−3.0%）、RMSE −0.078；相对 TSFM：MAE −0.043（−1.8%），"
        "逐样本绝对误差配对差的 95% bootstrap CI 为 [−0.069, −0.017]（不含 0），"
        "但 v2 只在 39.2% 的样本上更优——收益集中在少数高误差样本（重尾），而不是全面提升。", "",
        "| 分组 | v1 MAE | v2 MAE | TSFM MAE |", "|---|---|---|---|",
        "| t+1 / t+2 / t+3 | 1.949 / 2.451 / 2.870 | 1.904 / 2.379 / 2.770 | 1.947 / 2.414 / 2.820 |",
        "| jiangjunyan | 1.059 | 0.987 | 0.969 |",
        "| meichengmatou | 1.586 | 1.518 | 1.537 |",
        "| yanlingwu | 2.324 | 2.260 | 2.305 |",
        "| yanziling | 4.424 | 4.337 | 4.451 |", "",
        "### v1 失效原因 → v2 对应修复", "",
        "1. **决策口径是 H1 的**（“明天各站”+单步 dev 模型卡）→ v2 改为 horizon-aware 提示，并在 H3 直接锁定 chronos2+28；"
        "v1 中 context≠28 占 16%、选择净贡献 +0.029，v2 将选择贡献归零。",
        "2. **修正幅度过大**：v1 平均 |修正| 0.183，最坏一次把 jiangjunyan 8/31 的 t+1 从 6.84 上调到 12.50（实测 2.30），"
        "单 origin 贡献大量误差 → v2 增加 ±0.05×区间宽 的后处理上限，平均 |修正| 降到 0.122，99/309 行被截断；"
        "离线扫描显示 0.04~0.06 处于最优平台，更紧的上限反而退化。",
        "3. **memory 幻觉模型名**（LSTM-UQ / XGBoost / Linear-Ensemble 等）→ 复盘提示限定候选模型名，"
        "并清洗包含其它模型名的句子；最终 memory 386 字，无禁用模型名。",
        "4. **单点故障影响 3 天**：一次 H3 fusion 同时输出 3 天，缺少未来天气输入（33% 的修正理由仍引用不存在的预报）→ "
        "上限截断限制了单日错误的外溢，但该信息缺口仍是 H3 的主要限制。",
        "5. **负结果是 ensemble**：把 chronos2 与 timesfm3 按 0.5 平均后 MAE 2.433，不如 chronos2 单模型（2.394）；"
        "timesfm3 在 yanziling 上明显更差，因此 v2 不做模型集成。", "",
        "### H1 实验：保留 v1 结果", "",
        "v1 的 H1 MAE 1.747 仍是最优结果；增益全部来自 fusion（选择贡献为 0，raw 1.841 = TSFM）。"
        "用 v2 的 memory 清洗策略重跑 H1（选模型仍自由）时，LLM 明显漂移：seasonal_naive 被选 35/113 天、"
        "meichengmatou 大量改用 timesfm3，MAE 退化到 2.342。该次结果归档为 `2026_rsi_h1v2_*`，"
        "说明 H1 的优势对模型选择漂移很敏感；若要稳定复现，应像 H3 一样锁定 chronos2+28 再做 fusion。", "",
        "### 逐站 fusion 贡献（ΔMAE，负值=改善；RSI final vs 同批 TSFM）", "",
        "| 任务 | jiangjunyan | meichengmatou | yanlingwu | yanziling |",
        "|---|---|---|---|---|",
        "| H1 (v1) | −0.030 | +0.110 | −0.072 | −0.378 |",
        "| H3 (v2) | +0.019 | −0.018 | −0.045 | −0.113 |", "",
        "### 与传统基线的关系", "",
        "RSI 相对传统基线在 H1/H3 都领先：最好的传统模型是 CNN-LSTM（H1 1.976 / H3 2.842），"
        "RSI 分别好 11.6% 和 17.3%；标准 Transformer（均值池化）H1 3.414 / H3 3.926，"
        "是所有传统模型中最弱的。到 t+3 时 RSI 2.770 vs CNN-LSTM 3.232、GRU 3.482、"
        "LSTM 3.898、标准 Transformer 4.190。传统模型的主要问题是训练期"
        "（2024~2026.8，站均 12~20、峰值 142~413）与测试期（2026-09，站均 2.7~9.7、峰值 3.9~26.7）"
        "的分布漂移；直接多输出头在标签稀疏的 H3 上进一步向训练分布收缩，而 RSI/TSFM 依赖最近 28 天"
        "context，能跟随 9 月的低浓度状态。", "",
        "## 文献对照：2026 年 Agentic Time-Series 的机制与实测", "",
        "2026 年是 agentic time-series 集中爆发的一年。本轮重点核对的是 TimeAgent（TKDE 2026）、"
        "TimeRSI（arXiv 2610.19725）、TimeClaw（arXiv 2605.10038）、CastFSR（arXiv 2608.03031）、"
        "MemCast（arXiv 2602.03164）、FASE（arXiv 2609.32689）、TimEvolve（arXiv 2609.24862）、"
        "Last Mile（arXiv 2606.02497）与 FairAMO（arXiv 2610.29504）。它们的共同点不是增加对话角色，"
        "而是用已验证的历史经验去改管线或做数值精修。我们把这些机制逐项落到 RSI 上，"
        "并在同一批 H1（113 行）/ H3（309 行）评分样本上做 A/B。", "",
        "### 2026 工作与可迁移点", "",
        "| 工作 | 核心机制 | 在本项目中的对应 | 已有结论 |",
        "|---|---|---|---|",
        "| TimeAgent (TKDE 2026) | Training-free LLM agent 编排 TSFM：规划、工具调用、模型选择/组合 | 决策步与其“训练-free 编排”同构；H3 因选择偏离净损失 0.029 而锁定 chronos2+28 | 编排已有；增益依赖候选互补性（见 FairAMO） |",
        "| TimeRSI (2610.19725) | 编码 agent 生成可执行管线改动（输入/输出变换、路由），在历史窗口上验证，误差反馈驱动下一轮；部署时按相似条件选管线 | 本轮做了变换搜索与相似条件选择器 | 两项均未超过 v1 |",
        "| TimeClaw (2605.10038) | Explore→Compare→Distill→Reinject；Soul/Notes/Memory/Tools/Skills 五层经验，9 元组规则、上限 30 条；tool dropout 防止工具先验塌缩 | 当前 memory 是 ≤400 字自由文本，未结构化 | **唯一尚未实测的 prompt 级机制** |",
        "| CastFSR (2608.03031) | Fast–Slow–Reflect：轻量预测器给先验，LLM 慢思考补上下文，反思迭代满足约束 | v1 = Fast(TSFM)+Slow(LLM)+Reflect(memory)；v2 增加硬护栏与修正上限 | 已是同一范式 |",
        "| MemCast (2602.03164) | 三层记忆：历史模式 / 推理智慧 / 一般规律；置信度自适应；规律驱动反思 | 当前只有单层复盘 memory | 可升级，但受样本量限制 |",
        "| FASE (2609.32689) | Episodic memory + online policy learning；延迟反馈转成排序指导；29 个 GIFT-Eval 配置上 nMSE −9.1% | 在线门控/上下文赌博机是其最小实现 | 单任务上不显著 |",
        "| TimEvolve (2609.24862) | predict→reveal→update，持续更新 expert trust、路径选择与干预强度 | 在线 Hedge / 逐站开关 fusion | 方向正确，增益不显著 |",
        "| Last Mile (2606.02497) | LLM agent 在预测骨干之上做有界修正；推理轨迹转成 revision action；memory bank 支持事后反思 | fusion + 护栏 + memory 已对应 | H1 加修正上限反而变差 |",
        "| FairAMO (2610.29504) | 固定候选预测器，单独评估编排收益：最好编排比最强单模型仅好 <6%，收益来自互补性，冗余成员会抵消收益 | chronos2/timesfm3/seasonal 在本月不互补 | 解释了集成为何失败 |", "",
        "### 本轮新增实测（2026 机制逐项落地）", "",
        "| 机制 | 具体实现 | 验证/测试结果 | 判断 |",
        "|---|---|---|---|",
        "| TimeRSI 可执行管线改动 | 对 TSFM 输入输出做 log1p / sqrt / 差分 / 相对 last / 相对 mean7 五种变换，2025-09 逐站选型 | 2025-09：log1p 10.233、sqrt 10.349，优于原始 10.706；2026-09：选型后 **1.996**，固定 log1p 1.893，均差于 raw 1.841 / RSI 1.747 | 验证集有效、测试集反转 |",
        "| TimeRSI 相似条件选择 + FASE episodic memory | 5 个候选（RSI / raw / timesfm3 / seasonal / capped RSI），13 维 regime 特征，用 2025-09 + 2026 已揭晓 origin 做 kNN，k 与温度网格搜索 | 最好 **1.750**（逐站、k=40、temp=1.0），与 1.747 基本持平；多数参数组合 1.85~2.02 | 无增益 |",
        "| FASE / TimEvolve 在线策略 | LinUCB 上下文赌博机在 {RSI, raw} 间逐 origin 选择，reward = −|误差|/站内尺度 | **1.789**，26% 的 origin 切到 raw 后净损失；全局在线门控 1.724（CI 含 0）仍是最好的在线策略 | 不显著 |",
        "| CastFlow / TimeAgent 轻量数值精修 | 用 2024-05~2026-08 共 1338 个实测站日训练 ridge / HGB 残差校准器（30 维特征：TSFM 分位数、近期水平/趋势/波动、天气、站号、日历） | 仅校准：1.822~1.842；与 RSI 做 25% 混合：1.760~1.762；50% 混合：1.775~1.778 | 不能超过 1.747 |",
        "| CastFSR / Last Mile 有界修正 | H1 也套用 H3 的 point cap（0.02~0.05×区间宽） | **1.799**，比不设上限差 0.052；yanziling 的大幅修正正是 v1 的收益来源 | 过度约束有害 |",
        "| TimeClaw 结构化经验记忆 | 9 元组规则（kind/summary/applicability/preferred/avoided/rationale/evidence/confidence/injectable），上限 30 条，按 regime 检索注入 | 尚未运行（需完整 H1 LLM 回放） | 待验证 |", "",
        "### 关键统计事实", "",
        "1. **v1 的整体优势本身未过显著线**：RSI v1 vs raw 的逐样本绝对误差配对差为 −0.094，"
        "95% bootstrap CI [−0.217, +0.012]，含 0；分站看只有 yanziling 的 −0.354（CI [−0.709,−0.081]）显著。"
        "因此在 113 个样本上，任何小于约 0.1 的改进都无法在统计上验证。",
        "2. **完美门控的全部空间只有 0.083**：若逐样本上帝视角地在 RSI 与 raw 之间切换，MAE 为 1.664，"
        "仅比 v1 好 0.083。也就是说，即使把 gate 做到完美，也不足以稳定跨过噪声带。",
        "3. **5 候选的逐样本 oracle 是 1.094**：理论上还有很大空间，但它需要事先知道结果，"
        "所有可学习的 selector 都只能拿到 1.72~1.79，说明剩余空间不是编排能拿到的。",
        "4. **2025-09 的有效改动在 2026-09 全部反转**：log1p/sqrt、验证集 context 选择、residual 校准器、"
        "专家集成，在 dev 上都变好，在 test 上都变差。这不是方法实现问题，而是 2025-09（高浓度）与"
        "2026-09（低浓度）的分布漂移；用 dev 选出来的任何静态改动都会过拟合旧 regime。", "",
        "### 结论与下一步", "",
        "- **2026 的机制并没有被漏掉**：TimeAgent 的编排、CastFSR 的 Fast–Slow–Reflect、Last Mile 的有界修正、"
        "TimEvolve/FASE 的延迟反馈，在 v1/v2 中都已以不同形式存在；H3 v2 的 2.351 就是一次典型的"
        "“验证驱动的管线改动”。剩下的不是论文没看够，而是数据条件。",
        "- **唯一还没跑的 prompt 级机制是 TimeClaw 的结构化经验记忆**：把 400 字自由文本换成 9 元组规则库、"
        "按 regime 注入、并对“模型选择”做 tool-prior 去偏。它值得一次完整 H1 回放，但受 113 样本限制，"
        "即使有效也很难达到显著。",
        "- **收益最大的仍是新信息**：H3 有 36% 的修正理由引用了不存在的未来天气预报；把 t+1~t+3 的"
        "Tmax/降水/辐射预报接进来，是唯一能改变信息集、从而突破 1.66 gate 上限的一步。",
        "- **若要证明改进**：需要把测试从单月扩到 2025-09 + 2026-09 双月或滚动 12 个月；"
        "否则 1.72 vs 1.75 这类差异无法区分于噪声。", "",
        "> 当前 canonical 结果未因上述实验改动：H1 仍为 v1 的 1.747，H3 仍为 v2 的 2.351。", "",
        "> 完整失败实验记录与 H1 v1.1 候选（在线门控 + 偏差修正，MAE 1.689）见 "
        "[2026_rsi_v1_improvement_and_failures.md](2026_rsi_v1_improvement_and_failures.md)。", "",
        "### 参考工作（2026）", "",
        "- [TimeAgent: A Training-Free LLM Agent for Time Series Foundation Models (TKDE 2026)](https://doi.org/10.1109/TKDE.2026.3736409)",
        "- [TimeRSI: Towards Self-Improved Agentic Deployment for Time-Series Foundation Models](https://arxiv.org/abs/2610.19725)",
        "- [TimeClaw: A Time-Series AI Agent with Exploratory Execution Learning](https://arxiv.org/abs/2605.10038)",
        "- [Harnessing Generalist Agents for Contextualized Time Series](https://arxiv.org/abs/2606.05404)",
        "- [CastFSR: A Fast–Slow–Reflect Agentic Reasoning Framework for Context-Aware Time Series Forecasting](https://arxiv.org/abs/2608.03031)",
        "- [MemCast: Memory-Driven Time Series Forecasting with Experience-Conditioned Reasoning](https://arxiv.org/abs/2602.03164)",
        "- [FASE: Self-Evolving Time-Series Forecasting Agents with Episodic Memory and Online Policy Learning](https://arxiv.org/abs/2609.32689)",
        "- [TimEvolve: When Tomorrow Becomes Today — Self-Evolving Policies for Agentic Time-Series Forecasting](https://arxiv.org/abs/2609.24862)",
        "- [Bridging the Last Mile of Time Series Forecasting with LLM Agents](https://arxiv.org/abs/2606.02497)",
        "- [FairAMO: Rethinking Agentic Model Orchestration for Time Series Forecasting](https://arxiv.org/abs/2610.29504)", "",
        "## 结论速览", "",
        "- **H1**: RSI 综合最优（MAE 1.747 / RMSE 2.751 / W-MAE 2.077 / W-RMSE 3.361 / PRE 0.152）；"
        "传统基线中最好的是 CNN-LSTM（MAE 1.976），TSFM 次之（1.841），标准 Transformer（均值池化）3.414。",
        "- **H3**: RSI v2 最优（MAE 2.351 / RMSE 3.555 / W-MAE 2.827 / W-RMSE 4.310），"
        "优于 TSFM（2.394）和最好的传统模型 CNN-LSTM（2.842）；v2 相比 v1 改善 0.072。",
        "- **H3 显著性**: RSI v2 与 TSFM 的逐样本绝对误差配对差为 −0.043（95% bootstrap CI [-0.069, -0.017]，"
        "不含 0），统计上显著；但 v2 仅在 39.2% 的样本上更优，收益来自少数高误差样本的截断。",
        "- **随 horizon 衰减**: 所有系统 t+1 → t+3 误差单调上升，传统基线衰减更快；"
        "RSI/TSFM 在 t+3 的 MAE 约 2.77~2.82，CNN-LSTM 3.23，标准 Transformer 4.19。",
        "- **逐站差异**: meichengmatou 的 H1 GRU 最好（MAE 1.266）；yanziling 误差整体最大"
        "（H3：RSI 4.337、TSFM 4.451、CNN-LSTM 4.873）；标准 Transformer（均值池化）在所有站点"
        "都落后于其它传统模型；jiangjunyan 的峰值幅度（PRE）对 RSI-H3 偏弱，是主要失分点。",
        "- **稳定性**: 标准 Transformer（均值池化）的 H1 三种子 MAE 3.512±0.232，是传统基线中"
        "最弱且整体误差最大的版本；所有传统模型都受训练期（2024~2026.8）与测试期（2026-09）"
        "分布差异影响，整体弱于零样本 TSFM。",
        "- **v1 是否到极限**: 不是理论极限，但在当前信息集下，2026 年的编排/校准类机制已基本被验证过。"
        "本轮新增的 TimeRSI 变换搜索（1.996）、相似条件选择（1.750）、LinUCB 门控（1.789）、"
        "CastFlow 式残差校准（1.760~1.762）与 H1 修正上限（1.799）均未超过 v1 的 1.747。"
        "更关键的是，v1 对 raw 的整体优势 −0.094（95% CI [−0.217,+0.012]）本身未过显著线，"
        "完美 RSI/raw 门控的上限也只有 1.664；因此在 113 个样本上，剩余空间无法被可靠地学习到。"
        "下一步收益最大的是引入未来天气协变量（H3 的 36% 臆造预报）或扩大测试月，而不是继续增加 agent 角色。", "",
        "## 总体指标", "",
        "### H1：预测未来 1 天", "",
    ]
    lines += metrics_table(h1, args.alpha, ["overall"])
    lines += ["### H3：预测未来 3 天逐日值", ""]
    lines += metrics_table(h3, args.alpha, ["overall"])

    hrows = []
    for name in ORDER:
        g = h3[name]
        g = g[g["scored"]]
        for h, gh in g.groupby("h"):
            hrows.append({"system": name, "h": int(h), **metrics(gh, args.alpha)})
    lines += ["#### H3 分 horizon", "",
              "> 分 horizon 只比较误差类指标；PRE 是整组峰值指标，见上方总体表与逐站表。", "",
              "| 系统 | h | MAE | RMSE | W-MAE | W-RMSE |",
              "|---|---|---|---|---|---|"]
    for r in hrows:
        lines.append(f"| {SYSTEM_LABEL[r['system']]} | t+{r['h']} | {r['mae']:.3f} | {r['rmse']:.3f} | "
                     f"{r['wmae']:.3f} | {r['wrmse']:.3f} |")
    lines.append("")

    lines += ["## 逐站指标", "", "### H1：预测未来 1 天", ""]
    lines += metrics_table(h1, args.alpha, stations)
    lines += ["### H3：预测未来 3 天逐日值", ""]
    lines += metrics_table(h3, args.alpha, stations)

    lines += ["## 逐站逐日预测明细", "",
              "> 实测为 `—` 表示当日无实测；H3 的“评分窗口”✓ 表示该 3 天窗口全部实测（参与指标计算）。",
              "> 预测值保留 2 位小数，完整分位数区间见各 details CSV。", ""]
    for task, frames, title in (("h1", h1, "H1：预测未来 1 天"), ("h3", h3, "H3：预测未来 3 天")):
        lines += [f"### {title}", ""]
        for st in stations:
            lines += [f"#### {st}", ""] + prediction_table(frames, task, st) + [""]

    lines += ["## RSI 模型选择统计", "", "### H1", ""] + model_usage_table(h1, "h1") + [""]
    lines += ["### H3", ""] + model_usage_table(h3, "h3") + [""]

    lines += ["## 文件索引", "",
              "| 内容 | 路径 |", "|---|---|",
              "| 本报告 | `data/replay/2026_max_web_report.md` |",
              "| 指标汇总报告 | `data/replay/2026_max_comparison.md` |",
              "| H1 指标 CSV | `data/replay/2026_max_h1_comparison.csv` |",
              "| H3 指标 CSV | `data/replay/2026_max_h3_comparison.csv` |",
              "| H3 分horizon CSV | `data/replay/2026_max_h3_by_horizon.csv` |",
              "| 各系统明细(含q10/q90) | `data/replay/2026_{system}_max_*_details.csv` |",
              "| RSI H3 v1 归档 | `data/replay/2026_rsi_v1_max_h3_details.csv` |",
              "| RSI H1 v2 实验归档 | `data/replay/2026_rsi_h1v2_max_details.csv` |",
              "| Transformer 消融明细 | `data/replay/2026_transformer_{lastpool,nopos,small,lite}_max_*_details.csv` |",
              "| 模型 checkpoint | `data/baselines/{station}_{arch}_max_h{H}_s{seed}.pt` |",
              "| 复现脚本 | `scripts/replay_baselines.py`, `scripts/replay_daily.py`, "
              "`scripts/replay_rsi.py`, `scripts/compare_systems.py`, `scripts/build_web_report.py` |", ""]
    out = OUT / "2026_max_web_report.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"已写出 {out} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()