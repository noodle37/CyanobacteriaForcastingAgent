"""逐日滚动回放: 每天 origin=t 预测 t+1 (horizon=1), 揭晓真值, 评分区间覆盖。

评分只计 n_obs>0 的实测日; 缺测日的 context 用 LLM 估测副本(2026)或线性插值。
输出 data/replay/ 明细CSV + 汇总表(覆盖率/MAE/峰值日捕捉)。

Usage:
    python scripts/replay_daily.py --model chronos2 --target mean
    python scripts/replay_daily.py --model chronos2 --target max --context-days 28
    python scripts/replay_daily.py --model seasonal_naive
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.mcp_client import MCPToolClient
from agent.settings import TSFM_MCP_URL

STATIONS = ("jiangjunyan", "meichengmatou", "yanlingwu", "yanziling")
OUT = ROOT / "data" / "replay"


def load_station(station: str, year: int, target: str) -> pd.DataFrame:
    d = pd.read_csv(ROOT / "data" / "daily" / f"{station}_{year}_daily.csv", parse_dates=["date"]).set_index("date")
    ctx = d["cyano_ugl"].copy()
    llm_path = ROOT / "data" / "llm_imputed" / f"{station}_{year}_daily.csv"
    if llm_path.exists():
        alt = pd.read_csv(llm_path, parse_dates=["date"]).set_index("date")
        ctx = alt["cyano_llm"].combine_first(ctx)
    if target == "max":
        actual = d["cyano_ugl_max"].where(d["n_obs"] > 0)
        ctx = d["cyano_ugl_max"].combine_first(ctx)  # 缺测日的max用均值估测近似
    else:
        actual = d["cyano_ugl"].where(d["n_obs"] > 0)
    return pd.DataFrame({"ctx": ctx, "actual": actual, "n_obs": d["n_obs"]})


async def replay_station(client: MCPToolClient, station: str, year: int, month: int,
                         model: str, target: str, context_days: int, horizon: int = 1) -> list[dict]:
    d = load_station(station, year, target)
    targets = d.loc[f"{year}-{month:02d}"].dropna(subset=["ctx"])
    rows = []
    for date, row in targets.iterrows():
        origin = date - pd.Timedelta(days=1)
        hist = d.loc[:origin, "ctx"].dropna()
        if context_days > 0:
            hist = hist.tail(context_days)
        if len(hist) < 2:
            continue
        tdates = [date + pd.Timedelta(days=h) for h in range(horizon)]
        if any(td not in d.index for td in tdates):
            continue
        r = await client.call("forecast", {"model": model, "values": [float(v) for v in hist],
                                           "horizon": horizon, "freq": "D"})
        lab_obs = [bool(d.loc[td, "n_obs"] > 0) and np.isfinite(d.loc[td, "actual"]) for td in tdates]
        window_obs = all(lab_obs)
        for h, td in enumerate(tdates):
            trow = d.loc[td]
            point, q10, q90 = r["point"][h], r["q10"][h], r["q90"][h]
            obs = lab_obs[h]
            scored = window_obs if horizon > 1 else obs
            rows.append({
                "station": station, "date": str(td.date()), "origin": str(origin.date()), "h": h + 1,
                "model": model, "target": target,
                "context_len": len(hist), "actual": round(float(trow["actual"]), 3) if obs else None,
                "point": round(point, 3), "q10": round(q10, 3), "q90": round(q90, 3),
                "observed": obs, "window_observed": window_obs,
                "covered": bool(obs and q10 <= trow["actual"] <= q90) if obs else None,
                "abs_err": round(abs(point - trow["actual"]), 3) if obs else None,
                "scored": scored,
            })
    return rows


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    obs = df[df["observed"]]
    rows = []
    for (station, model, target), g in obs.groupby(["station", "model", "target"]):
        peaks = g.nlargest(3, "actual")
        actual_max = g.loc[g["actual"].idxmax()]
        pred_max = g.loc[g["point"].idxmax()]
        rows.append({
            "station": station, "model": model, "target": target,
            "days": len(g), "coverage": round(g["covered"].mean(), 3),
            "mae": round(g["abs_err"].mean(), 3),
            "peak3_covered": f"{int(peaks['covered'].sum())}/3",
            "max_day": actual_max["date"], "max_actual": actual_max["actual"],
            "max_day_point": actual_max["point"], "max_day_covered": bool(actual_max["covered"]),
            "pred_max": pred_max["point"], "pred_max_day": pred_max["date"],
        })
    return pd.DataFrame(rows)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stations", nargs="+", default=list(STATIONS))
    ap.add_argument("--year", type=int, default=2026)
    ap.add_argument("--month", type=int, default=9)
    ap.add_argument("--model", default="chronos2")
    ap.add_argument("--target", choices=["mean", "max"], default="mean")
    ap.add_argument("--context-days", type=int, default=0, help="0=用全部可用历史")
    ap.add_argument("--horizon", type=int, choices=[1, 3], default=1)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    client = MCPToolClient(TSFM_MCP_URL)
    rows = []
    for station in args.stations:
        rows += await replay_station(client, station, args.year, args.month, args.model, args.target,
                                     args.context_days, args.horizon)
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"{args.model}_{args.target}_ctx{args.context_days}"
    if args.horizon > 1 and not args.tag:
        tag += f"_h{args.horizon}"
    df.to_csv(OUT / f"{args.year}_{tag}_details.csv", index=False)
    summary = summarize(df[df["scored"]])
    summary.to_csv(OUT / f"{args.year}_{tag}_summary.csv", index=False)
    print(summary.to_string(index=False))
    obs = df[df["observed"]]
    print(f"\n总体: {len(obs)} 个实测日, 次日区间覆盖率 {obs['covered'].mean():.1%}, MAE {obs['abs_err'].mean():.3f}")


if __name__ == "__main__":
    asyncio.run(main())