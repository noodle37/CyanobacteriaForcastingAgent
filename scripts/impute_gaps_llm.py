"""用 LLM 估测蓝藻日序列的缺测段, 替代盲目线性插值; 并提供遮蔽验证。

LLM 依据: 缺口前后实测值 + 缺口期天气(气温/降雨/辐射) + 历史同期形态。
原始 data/daily/ 永不覆盖; LLM 结果写入 data/llm_imputed/ 独立副本。

Usage:
    python scripts/impute_gaps_llm.py --validate     # 遮蔽验证: linear vs LLM (需 .env 配 LLM)
    python scripts/impute_gaps_llm.py                # 对全部真实缺口段估测 (需 .env 配 LLM)
    python scripts/impute_gaps_llm.py --validate --linear-only   # 无 LLM 时跑 linear 基线
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

from agent.llm import LLMClient, parse_json_object

load_dotenv(ROOT / ".env")
LLM = LLMClient(
    base_url=os.getenv("LLM_BASE_URL", ""),
    api_key=os.getenv("LLM_API_KEY", ""),
    model=os.getenv("LLM_MODEL", ""),
    timeout=float(os.getenv("LLM_TIMEOUT_S", "120")),
)

STATIONS = {
    "jiangjunyan": "将军岩小微站(建德大洋)",
    "meichengmatou": "梅城码头",
    "yanlingwu": "严陵坞BBE",
    "yanziling": "严子陵藻类剖面浮标站",
}
YEARS = (2024, 2025, 2026)
OUT = ROOT / "data" / "llm_imputed"

SYSTEM = """你是富春江—新安江流域的水生态专家。任务: 根据缺测段前后的实测蓝藻浓度、缺测期间的逐日天气、以及该站历史同期形态, 估测缺测段每一天的蓝藻浓度(μg/L)。
领域知识: 蓝藻生长受水温驱动(持续高温促进增殖, 最适25-35℃); 降雨通过稀释/冲刷/降温/降光照抑制蓝藻, 效果滞后1-3天; 藻华爆发与消退是非线性的, 不能简单线性内插; 秋季(9月)整体趋于消退。
要求: 估值非负, 与前后实测值量级衔接。缺测段前后实测值接近且期间无显著天气事件时, 估测应平稳、接近前后实测均值, 不要臆造波动; 只有存在天气事件(降雨/持续高温/寒潮)或前后值显示明确爆发/消退趋势时, 才估测显著变化。缺测越长、天气越复杂, confidence 应越低。只输出JSON: {"values": [逐日数值...], "confidence": "high|medium|low", "reasoning": "简述依据"}。values 长度必须等于缺测天数。"""


def load_station_year(station: str, year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = pd.read_csv(ROOT / "data" / "daily" / f"{station}_{year}_daily.csv", parse_dates=["date"])
    w = pd.read_csv(ROOT / "data" / "weather" / f"{station}_daily.csv", parse_dates=["date"])
    return d.set_index("date"), w.set_index("date")


def gap_segments(d: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    m = d["interpolated"].astype(bool)
    grp = (m != m.shift()).cumsum()
    return [(g["date"].min(), g["date"].max()) for _, g in d[m].assign(date=d.index[m]).groupby(grp[m])]


def fmt_series(s: pd.Series, w: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> str:
    lines = []
    for date, v in s.loc[start:end].items():
        wx = w.loc[date]
        lines.append(
            f"{date.strftime('%m-%d')} 蓝藻={v:.2f} 最高温={wx['temperature_2m_max']:.1f} "
            f"降雨={wx['precipitation_sum']:.1f}mm 辐射={wx['shortwave_radiation_sum']:.1f}"
        )
    return "\n".join(lines)


def build_context(station: str, year: int, start: pd.Timestamp, end: pd.Timestamp,
                  d: pd.DataFrame, w: pd.DataFrame, masked: bool) -> str:
    s = d["cyano_ugl"]
    obs = d.loc[d["n_obs"] > 0, "cyano_ugl"]
    before = obs.loc[: start - pd.Timedelta(days=1)].tail(7)
    after = obs.loc[end + pd.Timedelta(days=1):].head(7)
    parts = [
        f"站点: {STATIONS[station]}; 缺测段: {start.strftime('%Y-%m-%d')} 至 {end.strftime('%Y-%m-%d')} 共 {(end - start).days + 1} 天。",
        "\n[缺口前最近实测(日)]\n" + fmt_series(before, w, before.index[0], before.index[-1]),
        "\n[缺口后最近实测(日)]\n" + fmt_series(after, w, after.index[0], after.index[-1]),
        "\n[缺口期间逐日天气]\n" + "\n".join(
            f"{date.strftime('%m-%d')} 最高温={r['temperature_2m_max']:.1f} 均温={r['temperature_2m_mean']:.1f} "
            f"降雨={r['precipitation_sum']:.1f}mm 辐射={r['shortwave_radiation_sum']:.1f}"
            for date, r in w.loc[start:end].iterrows()
        ),
    ]
    if year - 1 in YEARS:
        dp, wp = load_station_year(station, year - 1)
        pstart, pend = start - pd.DateOffset(years=1), end - pd.DateOffset(years=1)
        ref = dp["cyano_ugl"].loc[pstart - pd.Timedelta(days=7): pend + pd.Timedelta(days=7)].dropna()
        if len(ref):
            parts.append(f"\n[去年({year-1})同日期段±7天的蓝藻形态, 供参考]\n"
                         + fmt_series(ref, wp, ref.index[0], ref.index[-1]))
    if masked:
        parts.append("\n注意: 这是一段被人为遮盖的实测数据, 请给出你的最佳估测。")
    return "\n".join(parts)


def llm_impute(context: str, n_days: int) -> dict:
    text = LLM.complete(SYSTEM, context, json_mode=True)
    obj = parse_json_object(text)
    values = [float(v) for v in obj["values"]]
    if len(values) != n_days:
        raise ValueError(f"LLM 返回 {len(values)} 个值, 期望 {n_days}")
    if any(v < 0 for v in values):
        raise ValueError("LLM 返回负值")
    return {"values": [round(v, 3) for v in values],
            "confidence": obj.get("confidence", "medium"),
            "reasoning": str(obj.get("reasoning", ""))[:500]}


def linear_impute(d: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> list[float]:
    s = d["cyano_ugl"].copy()
    s.loc[start:end] = np.nan
    return s.interpolate(limit_area="inside").loc[start:end].round(3).tolist()


def validate(linear_only: bool) -> None:
    rows = []
    for station in STATIONS:
        for year in (2025, 2026):
            d, w = load_station_year(station, year)
            for month in (6, 7, 8):
                days = d.loc[f"{year}-{month:02d}"]
                days = days[days["n_obs"] > 0]
                win = None
                for i in range(len(days) - 4):
                    cand = days.iloc[i: i + 5]
                    if (cand.index.diff().dropna() == pd.Timedelta(days=1)).all():
                        win = cand
                        break
                if win is None:
                    continue
                start, end = win.index[0], win.index[-1]
                truth = win["cyano_ugl"].to_numpy()
                lin = np.array(linear_impute(d, start, end))
                row = {"station": station, "year": year, "window": f"{start.strftime('%m-%d')}..{end.strftime('%m-%d')}",
                       "true_mean": round(float(truth.mean()), 2),
                       "mae_linear": round(float(np.abs(lin - truth).mean()), 3)}
                if not linear_only:
                    dm = d.copy()
                    dm.loc[start:end, "cyano_ugl"] = np.nan
                    dm.loc[start:end, "n_obs"] = 0
                    est = llm_impute(build_context(station, year, start, end, dm, w, masked=True), 5)
                    row["mae_llm"] = round(float(np.abs(np.array(est["values"]) - truth).mean()), 3)
                    row["confidence"] = est["confidence"]
                rows.append(row)
                print(rows[-1])
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "validation.csv", index=False)
    print("\n== 汇总 ==")
    print(df.groupby("station")[["mae_linear"] + (["mae_llm"] if "mae_llm" in df else [])].mean().round(3).to_string())
    if "mae_llm" in df:
        wins = int((df["mae_llm"] < df["mae_linear"]).sum())
        print(f"LLM 优于 linear: {wins}/{len(df)} 窗口; 整体 MAE linear={df['mae_linear'].mean():.3f} llm={df['mae_llm'].mean():.3f}")


def impute_all(years: tuple[int, ...] = YEARS) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for station in STATIONS:
        for year in years:
            d, w = load_station_year(station, year)
            segs = gap_segments(d)
            if not segs:
                continue
            log_path = OUT / f"{station}_{year}_reasoning.json"
            done = json.loads(log_path.read_text()) if log_path.exists() else {}
            out = d[["cyano_ugl", "n_obs", "interpolated"]].copy()
            out = out.rename(columns={"cyano_ugl": "cyano_linear"})
            out["cyano_llm"] = out["cyano_linear"]
            for start, end in segs:
                key = f"{start.date()}..{end.date()}"
                n = (end - start).days + 1
                if key not in done:
                    done[key] = llm_impute(build_context(station, year, start, end, d, w, masked=False), n)
                    log_path.write_text(json.dumps(done, ensure_ascii=False, indent=1))
                    print(f"{station} {year} {key}: {done[key]['confidence']} | {done[key]['reasoning'][:80]}")
                out.loc[start:end, "cyano_llm"] = done[key]["values"]
            out.round(3).to_csv(OUT / f"{station}_{year}_daily.csv")
    print(f"\nLLM 估测副本与推理日志 -> {OUT.relative_to(ROOT)}/ (原始 data/daily/ 未改动)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true", help="遮蔽验证而非真实缺口估测")
    ap.add_argument("--linear-only", action="store_true", help="验证时只跑 linear 基线")
    ap.add_argument("--years", type=int, nargs="+", default=list(YEARS), help="只估测指定年份")
    args = ap.parse_args()
    if not args.linear_only and not LLM.enabled:
        sys.exit("LLM 未配置: 请在 .env 设置 LLM_BASE_URL/LLM_API_KEY/LLM_MODEL, 或用 --validate --linear-only 跑基线")
    if args.validate:
        validate(args.linear_only)
    else:
        impute_all(tuple(args.years))


if __name__ == "__main__":
    main()