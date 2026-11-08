"""RSI 逐日回放: LLM 统筹 4 站, 每天 3 次调用形成自我改进闭环。

每日流程 (origin=t, 预测 t+1..t+H, 天气只用 ≤t):
  1. 决策: LLM 按近期形态/天气趋势/记忆为每站选 TSFM 模型与 context 长度
  2. 预测: 经 MCP 调用各站选中的模型 (chronos2/timesfm3/seasonal_naive)
  3. 修正: LLM 结合天气趋势逐日调整 point/q10/q90 (考核区间宽度, 防无脑加宽)
  4. 揭晓真值评分 (H=1 仅实测日; H=3 仅3天全实测窗口)
  5. 复盘: LLM 提炼经验写入跨天 memory, 明日生效

断点续跑: 状态存 data/replay/rsi_{year}{suffix}_state.json; --fresh 重新开始。
--target max 预测日最高值, --horizon 3 预测未来3天逐日值。

Usage:
    python scripts/replay_rsi.py
    python scripts/replay_rsi.py --target max --horizon 1 --fresh
    python scripts/replay_rsi.py --target max --horizon 3 --fresh
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.llm import parse_json_object
from agent.mcp_client import MCPToolClient
from agent.settings import LLM, TSFM_MCP_URL
from scripts.replay_daily import STATIONS, load_station

OUT = ROOT / "data" / "replay"
H3_POINT_CAP = 0.05  # H3修正量上限 = 该日原始区间宽 × 0.05
BANNED_MODEL = re.compile(r"LSTM|XGBoost|GBDT|GRU|Transformer|Linear-|Tree-|Ensemble")
MODELS = {
    "chronos2": "Chronos-2: 通用时序基础模型, 趋势延续与分位数较可靠",
    "timesfm3": "TimesFM-3: 另一基础模型, 部分站点精度相当",
    "seasonal_naive": "7天季节naive: 平稳期覆盖率尚可, 爆发/消退期严重滞后",
}

def dev_cards(target: str = "mean") -> str:
    suffix = "_max" if target == "max" else ""
    lines = []
    for m in MODELS:
        p = OUT / f"2025_{m}_dev2025{suffix}_summary.csv"
        if p.exists():
            s = pd.read_csv(p)
            lines.append(m + ": " + "; ".join(f"{r['station']} 覆盖{r['coverage']:.0%}/MAE{r['mae']:.1f}"
                                              for _, r in s.iterrows()))
    return "\n".join(lines)


def baseline_note(target: str) -> str:
    p = OUT / f"2026_chronos2_{target}_ctx28_summary.csv"
    if p.exists():
        s = pd.read_csv(p)
        cov = (s["coverage"] * s["days"]).sum() / s["days"].sum()
        return f"参考基线: 2026年9月固定 chronos2+28天context 覆盖率{cov:.1%}。"
    return ""


def decision_sys(target: str, horizon: int = 1) -> str:
    tdesc = "日最高值(max)" if target == "max" else "日均值(mean)"
    when = "明天各站" if horizon == 1 else f"未来{horizon}天各站逐日"
    lock = ("" if horizon == 1 else
            "注意: H3回测中偏离 chronos2+28 的模型/context 选择净损失精度, 因此H3固定 chronos2+28, "
            "你只需给出 outlook 和理由, model/context_days 会被系统覆盖。")
    return """你是统筹4个蓝藻监测站逐日预测的调度agent。""" + when + tdesc + """的预测由你选模型和context长度。
""" + lock + """
可选模型: """ + json.dumps(MODELS, ensure_ascii=False) + """
以下是2025年9月dev回测的模型卡(逐站覆盖率/MAE, """ + tdesc + """), 选模型必须以此为证据:
""" + dev_cards(target) + """
""" + baseline_note(target) + """默认选 chronos2+28天, 只有模型卡和近期形态给出强证据时才偏离; seasonal_naive在dev上全面最差, 基本不应选。
依据: 近期序列形态、天气趋势(持续升温无雨→蓝藻易爆发, 降雨→稀释冲刷下降)、memory。只输出JSON:
{"stations": {"站名": {"model": "chronos2|timesfm3|seasonal_naive", "context_days": 整数7-120, "outlook": "rising|falling|stable", "reason": "一句话"}}}"""

FUSION_SYS = """你是蓝藻预测修正专家。TSFM已给出明天各站的点预测与q10/q90区间, 你结合天气趋势做最后修正。
规则: 持续升温+无雨+强辐射→藻华爆发风险, 可上调point和q90; 降雨(尤其>10mm)→下调point和q10/q90; 平稳天气不要动, 原样返回。
方向一致性(最重要): 实际高于区间=低估→同类天气应上调; 实际低于区间=高估→才下调。禁止反向操作。
硬护栏(违反则你的修正被丢弃): 修正后区间宽度不得小于原始宽度; point必须落在原始[q10,q90]内; 0 ≤ q10 ≤ point ≤ q90。
只输出JSON: {"stations": {"站名": {"point": 数值, "q10": 数值, "q90": 数值, "reason": "一句话"}}}"""

FUSION_SYS_H3 = """你是蓝藻预测修正专家。TSFM已给出未来3天各站逐日的点预测与q10/q90区间, 你结合天气趋势逐日做最后修正。
规则: 持续升温+无雨+强辐射→藻华爆发风险, 可上调point和q90; 降雨(尤其>10mm)→下调point和q10/q90; 平稳天气不要动, 原样返回。
方向一致性(最重要): 实际高于区间=低估→同类天气应上调; 实际低于区间=高估→才下调。禁止反向操作。
硬护栏(违反则该日修正被丢弃): 修正后区间宽度不得小于原始宽度; point必须落在原始[q10,q90]内; 0 ≤ q10 ≤ point ≤ q90。
只输出JSON: {"stations": {"站名": {"days": [{"point": 数值, "q10": 数值, "q90": 数值}, {"point": 数值, "q10": 数值, "q90": 数值}, {"point": 数值, "q10": 数值, "q90": 数值}], "reason": "一句话"}}}
days数组按 t+1,t+2,t+3 顺序, 长度必须为3。"""

REFLECT_SYS = """你是预测复盘agent。今天各站的预测已揭晓真值。请提炼经验更新 memory(供明天决策与修正参考)。
要求: 只写定性经验——哪站适合哪个模型(参照dev模型卡)、哪类天气信号导致低估还是高估、明日同类天气区间该上调还是下调。
方向必须一致: 实际>区间上界=低估→明日同类天气应上调; 实际<区间下界=高估→应下调。
禁止发明数值阈值规则(如"Tmax≥34℃则下调20%"), 数值修正由预测环节按天气信号执行, 你只负责判断方向与模型适配。
只允许提及候选模型 chronos2 / timesfm3 / seasonal_naive, 禁止出现其它模型名。
只输出JSON: {"memory": "更新后的完整经验文本, 保留仍有效的旧经验, 删除被证伪的, 不超过400字"}"""


def sanitize_memory(text: str) -> str:
    """丢弃提及非候选模型的句子, 避免复盘幻觉污染次日决策。"""
    parts = [p for p in re.split(r"[；;。\n]", text) if p.strip() and not BANNED_MODEL.search(p)]
    return "；".join(parts)[:1200]


def ask_json(system: str, user: str) -> dict | None:
    for _ in range(2):
        try:
            return parse_json_object(LLM.complete(system, user, json_mode=True))
        except Exception:
            continue
    return None


def station_brief(st: str, d: pd.DataFrame, w: pd.DataFrame, origin: pd.Timestamp,
                  horizon: int = 1, future_weather: bool = False) -> str:
    hist = d.loc[:origin, "ctx"].dropna().tail(7)
    vals = ", ".join(f"{dt.strftime('%m-%d')}={v:.2f}" for dt, v in hist.items())
    wx = w.loc[origin - pd.Timedelta(days=2): origin]
    wtext = "; ".join(f"{dt.strftime('%m-%d')} Tmax={r['temperature_2m_max']:.1f} 雨={r['precipitation_sum']:.1f}mm 辐射={r['shortwave_radiation_sum']:.1f}"
                      for dt, r in wx.iterrows())
    text = f"[{st}] 近7日蓝藻: {vals}\n  近3日天气(≤预测起点): {wtext}"
    if future_weather and horizon > 1:
        ftext = future_weather_brief(w, origin, horizon)
        if ftext:
            text += f"\n  未来{horizon}天天气(逐日): {ftext}"
    return text


def future_weather_brief(w: pd.DataFrame, origin: pd.Timestamp, horizon: int) -> str:
    parts = []
    for h in range(1, horizon + 1):
        td = origin + pd.Timedelta(days=h)
        if td in w.index:
            r = w.loc[td]
            parts.append(f"t+{h}({td.strftime('%m-%d')}) Tmax={r['temperature_2m_max']:.1f} "
                         f"雨={r['precipitation_sum']:.1f}mm 辐射={r['shortwave_radiation_sum']:.1f}")
    return "; ".join(parts)


async def run_day(client, date, stations, data, weather, memory, yesterday, no_fusion=False,
                  target="mean", horizon=1, h3_point_cap=H3_POINT_CAP, future_weather=False,
                  relax_h3_guard=False):
    origin = date - pd.Timedelta(days=1)
    tdesc = "日最高值(max)" if target == "max" else "日均值(mean)"
    briefs = "\n".join(station_brief(st, data[st], weather[st], origin, horizon, future_weather)
                       for st in stations)
    dec_user = f"预测目标日: {date.date()} 起未来{horizon}天 ({tdesc}, 起点 {origin.date()})\nmemory: {memory or '(无, 第一天)'}\n昨日结果: {yesterday or '(无)'}\n{briefs}"
    dec = ask_json(decision_sys(target, horizon), dec_user) or {}
    choices = {}
    for st in stations:
        c = (dec.get("stations") or {}).get(st) or {}
        model = c.get("model") if c.get("model") in MODELS else "chronos2"
        try:
            ctx = int(np.clip(int(c.get("context_days", 28)), 7, 120))
        except (TypeError, ValueError):
            ctx = 28
        choices[st] = {"model": model, "context_days": ctx, "outlook": c.get("outlook", "stable"),
                       "reason": str(c.get("reason", ""))[:200]}
    if horizon > 1:
        for st in stations:
            choices[st]["model"] = "chronos2"
            choices[st]["context_days"] = 28
            choices[st]["reason"] = "H3锁定chronos2+28(回测:选择偏离净损失0.029) | " + choices[st]["reason"]

    raw = {}
    for st in stations:
        hist = data[st].loc[:origin, "ctx"].dropna().tail(choices[st]["context_days"])
        r = await client.call("forecast", {"model": choices[st]["model"], "values": [float(v) for v in hist],
                                           "horizon": horizon, "freq": "D"})
        raw[st] = [{"point": r["point"][h], "q10": r["q10"][h], "q90": r["q90"][h]} for h in range(horizon)]

    def raw_text(st):
        return "; ".join(f"d{h+1} point={p['point']:.2f} q10={p['q10']:.2f} q90={p['q90']:.2f}"
                         for h, p in enumerate(raw[st]))
    fus_user = (f"目标日 {date.date()} 起未来{horizon}天 ({tdesc})\nmemory: {memory or '(无)'}\n"
                + (f"未来{horizon}天逐日天气已在各站给出, 请按 t+1/t+2/t+3 对应天气信号修正; "
                   "禁止臆造未给出的天气预报。\n" if future_weather and horizon > 1 else "")
                + ("允许point超出原始q10/q90(最多±0.1×原始区间宽), 但请同步放宽q10/q90使区间仍包含point, "
                   "且区间宽度不得小于原始宽度的50%。\n" if relax_h3_guard and horizon > 1 else "")
                + "\n".join(f"[{st}] 决策outlook={choices[st]['outlook']}; TSFM({choices[st]['model']}): "
                            f"{raw_text(st)}\n"
                            + station_brief(st, data[st], weather[st], origin, horizon, future_weather)
                            for st in stations))
    fus = {} if no_fusion else (ask_json(FUSION_SYS_H3 if horizon > 1 else FUSION_SYS, fus_user) or {})
    final = {}
    for st in stations:
        f = (fus.get("stations") or {}).get(st) or {}
        days_f = f.get("days") if horizon > 1 else [f]
        final[st] = []
        for h in range(horizon):
            if no_fusion:
                final[st].append({**raw[st][h], "reason": "(融合关闭, 用原始TSFM)",
                                  "adjusted": False, "post_cap": False})
                continue
            try:
                fd = days_f[h] if isinstance(days_f, list) else {}
                point, q10, q90 = float(fd["point"]), float(fd["q10"]), float(fd["q90"])
                if relax_h3_guard:
                    q10 = max(0.0, q10)
                    q90 = max(q90, point, q10)
                    point = max(point, q10)
                if not (0 <= q10 <= point <= q90):
                    raise ValueError("区间顺序")
                raw_w = raw[st][h]["q90"] - raw[st][h]["q10"]
                if relax_h3_guard:
                    if q90 - q10 < 0.5 * raw_w - 1e-6:
                        raise ValueError("区间收缩超过50%")
                else:
                    if q90 - q10 < raw_w - 1e-6:
                        raise ValueError("区间收缩")
                    if not (raw[st][h]["q10"] <= point <= raw[st][h]["q90"]):
                        raise ValueError("point超出原始区间")
                capped = False
                if horizon > 1:
                    rp, rq10, rq90 = raw[st][h]["point"], raw[st][h]["q10"], raw[st][h]["q90"]
                    cap = max(h3_point_cap * (rq90 - rq10), 1e-6)
                    if relax_h3_guard:
                        lo, hi = max(0.0, rp - cap), rp + cap
                    else:
                        lo, hi = max(rq10, rp - cap), min(rq90, rp + cap)
                    clipped = float(np.clip(point, lo, hi))
                    capped = abs(clipped - point) > 1e-9
                    point = float(np.clip(clipped, q10, q90))
                reason = str(f.get("reason", ""))[:200]
                if capped:
                    reason += f" | 后处理: H3修正量限制在±{h3_point_cap:.2f}×区间宽"
                final[st].append({"point": point, "q10": q10, "q90": q90,
                                  "reason": reason, "adjusted": True, "post_cap": capped})
            except (KeyError, TypeError, ValueError, AssertionError, IndexError) as exc:
                final[st].append({**raw[st][h],
                                  "reason": f"(违反护栏或输出无效: {type(exc).__name__}:{exc}, 用原始TSFM)",
                                  "adjusted": False, "post_cap": False})

    rows, revealed = [], []
    for st in stations:
        tdates = [date + pd.Timedelta(days=h) for h in range(horizon)]
        lab_obs = [bool(data[st].loc[td, "n_obs"] > 0 and np.isfinite(data[st].loc[td, "actual"])) for td in tdates]
        window_obs = all(lab_obs)
        for h, td in enumerate(tdates):
            row = data[st].loc[td]
            obs = lab_obs[h]
            scored = window_obs if horizon > 1 else obs
            p = final[st][h]
            rows.append({"station": st, "date": str(td.date()), "origin": str(origin.date()), "h": h + 1,
                         "model": choices[st]["model"], "target": target,
                         "context_len": choices[st]["context_days"],
                         "actual": round(float(row["actual"]), 3) if obs else None,
                         "point": round(p["point"], 3), "q10": round(p["q10"], 3), "q90": round(p["q90"], 3),
                         "observed": obs, "window_observed": window_obs,
                         "covered": bool(obs and p["q10"] <= row["actual"] <= p["q90"]) if obs else None,
                         "abs_err": round(abs(p["point"] - row["actual"]), 3) if obs else None,
                         "scored": scored,
                         "raw_point": round(raw[st][h]["point"], 3), "raw_q10": round(raw[st][h]["q10"], 3),
                         "raw_q90": round(raw[st][h]["q90"], 3), "adjusted": p["adjusted"],
                         "post_cap": p.get("post_cap", False),
                         "decision_reason": choices[st]["reason"], "fusion_reason": p["reason"],
                         "future_weather": bool(future_weather)})
            if obs:
                wx = weather[st].loc[td]
                revealed.append(f"[{st}] t+{h+1}({td.strftime('%m-%d')}) 预测point={p['point']:.2f} "
                                f"区间[{p['q10']:.2f},{p['q90']:.2f}] 实际={row['actual']:.2f} "
                                f"{'覆盖' if rows[-1]['covered'] else '未覆盖'}; "
                                f"当日天气 Tmax={wx['temperature_2m_max']:.1f} 雨={wx['precipitation_sum']:.1f}mm")
    ref_user = f"目标日 {date.date()} 起未来{horizon}天揭晓:\n" + "\n".join(revealed) + f"\n\n旧memory: {memory or '(无)'}"
    ref = ask_json(REFLECT_SYS, ref_user) or {}
    new_memory = str(ref.get("memory", memory))[:1200] if ref.get("memory") else memory
    if new_memory:
        new_memory = sanitize_memory(new_memory)
    return rows, new_memory, "; ".join(f"{st}:{choices[st]['model']}/{choices[st]['context_days']}d" for st in stations)


def report(rows: list[dict], year: int, suffix: str = "") -> None:
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"{year}_rsi{suffix}_details.csv", index=False)
    obs = df[df["scored"] if "scored" in df.columns else df["observed"]].copy()
    obs["width"] = obs["q90"] - obs["q10"]
    lines = []
    for st, g in obs.groupby("station"):
        peaks = g.nlargest(3, "actual")
        mx = g.loc[g["actual"].idxmax()]
        half = len(g) // 2
        lines.append({"station": st, "days": len(g), "coverage": round(g["covered"].mean(), 3),
                      "mae": round(g["abs_err"].mean(), 3), "width": round(g["width"].mean(), 2),
                      "coverage_1st_half": round(g.iloc[:half]["covered"].mean(), 2),
                      "coverage_2nd_half": round(g.iloc[half:]["covered"].mean(), 2),
                      "peak3": f"{int(peaks['covered'].sum())}/3",
                      "max_day_covered": bool(mx["covered"]),
                      "models": g["model"].value_counts().to_dict()})
    summary = pd.DataFrame(lines)
    summary.to_csv(OUT / f"{year}_rsi{suffix}_summary.csv", index=False)
    print(summary.to_string(index=False))
    half = len(obs) // 2
    print(f"\n总体: 覆盖率 {obs['covered'].mean():.1%}, MAE {obs['abs_err'].mean():.3f}, 平均区间宽 {obs['width'].mean():.2f}")
    print(f"前半月覆盖率 {obs.iloc[:half]['covered'].mean():.1%} → 后半月 {obs.iloc[half:]['covered'].mean():.1%}")
    print(f"模型使用: {obs['model'].value_counts().to_dict()}")
    print(f"区间被LLM调整比例: {obs['adjusted'].mean():.1%}")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2026)
    ap.add_argument("--month", type=int, default=9)
    ap.add_argument("--target", choices=["mean", "max"], default="mean")
    ap.add_argument("--horizon", type=int, choices=[1, 3], default=1)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--no-fusion", action="store_true", help="关闭LLM区间修正, 只做模型选择+复盘")
    ap.add_argument("--h3-point-cap", type=float, default=H3_POINT_CAP,
                    help="H3每日修正量上限(相对原始区间宽), 默认0.05")
    ap.add_argument("--future-weather", action="store_true",
                    help="实验: H3融合注入t+1..t+H未来天气, 仅RSI可见; 其他模型不使用")
    ap.add_argument("--relax-h3-guard", action="store_true",
                    help="实验: 放宽H3护栏, 允许point超出原始q10/q90(仍受cap限制, 区间宽度≥原始50%)")
    ap.add_argument("--limit", type=int, default=0, help="最多运行N个新日期(调试用), 0=全部")
    args = ap.parse_args()
    if not LLM.enabled:
        sys.exit("LLM 未配置")
    OUT.mkdir(parents=True, exist_ok=True)
    suffix = ("_nofusion" if args.no_fusion else "") + (f"_{args.target}" if args.target != "mean" else "") \
        + (f"_h{args.horizon}" if args.horizon > 1 else "") + ("_fwx" if args.future_weather else "") \
        + ("_relax" if args.relax_h3_guard else "") \
        + (f"_cap{args.h3_point_cap:g}" if abs(args.h3_point_cap - H3_POINT_CAP) > 1e-9 else "")
    state_path = OUT / f"rsi_{args.year}{suffix}_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() and not args.fresh else {"done": [], "rows": [], "memory": ""}

    data = {st: load_station(st, args.year, args.target) for st in STATIONS}
    weather = {st: pd.read_csv(ROOT / "data" / "weather" / f"{st}_daily.csv", parse_dates=["date"]).set_index("date")
               for st in STATIONS}
    all_dates = sorted(set().union(*[set(d.loc[f"{args.year}-{args.month:02d}"].dropna(subset=["ctx"]).index) for d in data.values()]))
    client = MCPToolClient(TSFM_MCP_URL)
    ran = 0
    for ts in all_dates:
        date = ts.strftime("%Y-%m-%d")
        if date in state["done"]:
            continue
        if args.limit and ran >= args.limit:
            break
        stations = [st for st in STATIONS if ts in data[st].index and np.isfinite(data[st].loc[ts, "ctx"])
                    and ts + pd.Timedelta(days=args.horizon - 1) in data[st].index]
        if not stations:
            continue
        yesterday = None
        if state["rows"]:
            prev = [r for r in state["rows"] if r["date"] == (ts - pd.Timedelta(days=1)).strftime("%Y-%m-%d") and r["observed"]]
            yesterday = "; ".join(f"{r['station']} {'中' if r['covered'] else '漏'}(err={r['abs_err']})" for r in prev)
        rows, state["memory"], used = await run_day(client, ts, stations, data, weather, state["memory"], yesterday,
                                                  args.no_fusion, args.target, args.horizon, args.h3_point_cap,
                                                  args.future_weather, args.relax_h3_guard)
        state["rows"] += rows
        state["done"].append(date)
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=1))
        ran += 1
        obs = [r for r in rows if r["scored"]]
        cov = f"{sum(r['covered'] for r in obs)}/{len(obs)}" if obs else "无实测"
        print(f"{date} 覆盖{cov} | {used}", flush=True)
    report(state["rows"], args.year, suffix)


if __name__ == "__main__":
    asyncio.run(main())