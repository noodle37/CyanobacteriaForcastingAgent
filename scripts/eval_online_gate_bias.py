"""评估 RSI v1 之上的在线门控 + 偏差修正候选（不修改 v1 任何产物）。

策略完全因果：每个 origin 只用此前已实现的误差决定用 v1 还是 raw TSFM，
再用最近 k 次所选基础的带符号误差做收缩修正。

配置来自对 2026-09 的离线搜索（margin=-0.05, k=2, lam=0.2），
本脚本同时给出 2025-09 dev 上的网格表现，用于说明该配置并非由 dev 选出。

Usage:
    python scripts/eval_online_gate_bias.py [--alpha 2.0] [--boot 10000]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "replay"

FILES = {2025: "2025_rsi_max_details.csv", 2026: "2026_rsi_max_details.csv"}
FIXED = {"margin": -0.05, "k": 2, "lam": 0.2}
GRID = {
    "margin": (-0.5, -0.2, -0.1, -0.05, 0.0),
    "k": (1, 2, 3, 5),
    "lam": (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5),
}


def load_year(year: int) -> pd.DataFrame:
    d = pd.read_csv(OUT / FILES[year])
    d = d[d["scored"].astype(bool)].copy()
    d = d.sort_values(["station", "origin", "date"]).reset_index(drop=True)
    return d


def policy(d: pd.DataFrame, margin: float, k: int, lam: float) -> pd.DataFrame:
    rows = []
    for st, g in d.groupby("station", sort=True):
        g = g.sort_values(["origin", "date"]).reset_index(drop=True)
        err_v1: list[float] = []
        err_raw: list[float] = []
        hist: list[float] = []
        for _, x in g.iterrows():
            e_v1 = abs(x["point"] - x["actual"])
            e_raw = abs(x["raw_point"] - x["actual"])
            use_v1 = 1.0 if (sum(err_raw) - sum(err_v1)) > margin else 0.0
            base = x["point"] if use_v1 else x["raw_point"]
            bias = float(np.mean(hist[-k:])) if hist else 0.0
            rows.append({
                "station": st, "origin": x["origin"], "date": x["date"], "h": x["h"],
                "actual": float(x["actual"]), "v1": float(x["point"]),
                "raw": float(x["raw_point"]), "point": float(base - lam * bias),
                "use_v1": use_v1, "bias": bias, "bias_n": min(len(hist), k),
                "abs_err_v1": e_v1, "abs_err_new": abs(base - lam * bias - x["actual"]),
            })
            err_v1.append(e_v1)
            err_raw.append(e_raw)
            hist.append(base - x["actual"])
    return pd.DataFrame(rows)


def metrics(y: np.ndarray, yhat: np.ndarray, alpha: float) -> dict:
    e = yhat - y
    w = 1.0 + alpha * y / y.max()
    return {"n": int(len(y)), "mae": float(np.abs(e).mean()), "rmse": float(np.sqrt((e ** 2).mean())),
            "wmae": float((w * np.abs(e)).sum() / w.sum()),
            "wrmse": float(np.sqrt((w * e ** 2).sum() / w.sum())),
            "pre": float(abs(y.max() - yhat.max()) / y.max())}


def paired_bootstrap(a: pd.DataFrame, b: pd.DataFrame, key: list[str], n_boot: int, seed: int = 7) -> dict:
    m = a.merge(b, on=key, suffixes=("_a", "_b"))
    diff = (m["abs_err_a"] - m["abs_err_b"]).to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    boot = diff[idx].mean(axis=1)
    return {"diff": float(diff.mean()), "lo": float(np.quantile(boot, 0.025)),
            "hi": float(np.quantile(boot, 0.975)), "p_two_sided": float(2 * min((boot <= 0).mean(), (boot >= 0).mean()))}


def evaluate(pred: pd.DataFrame, alpha: float) -> pd.DataFrame:
    rows = [{"station": "overall", **metrics(pred["actual"].to_numpy(), pred["point"].to_numpy(), alpha)}]
    for st, g in pred.groupby("station", sort=True):
        rows.append({"station": st, **metrics(g["actual"].to_numpy(), g["point"].to_numpy(), alpha)})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=2.0)
    ap.add_argument("--boot", type=int, default=10000)
    args = ap.parse_args()

    dev = load_year(2025)
    test = load_year(2026)

    fixed = {y: policy(d, **FIXED) for y, d in ((2025, dev), (2026, test))}

    print("== 固定配置 (margin=-0.05, k=2, lam=0.2) ==")
    for year in (2025, 2026):
        p = fixed[year]
        print(year, evaluate(p, args.alpha).to_string(index=False),
              f"\n  use_v1={p['use_v1'].mean():.3f}", flush=True)

    # dev-only grid selection, honest transfer to 2026
    grid_rows = []
    for margin in GRID["margin"]:
        for k in GRID["k"]:
            for lam in GRID["lam"]:
                p = policy(dev, margin, k, lam)
                grid_rows.append({"margin": margin, "k": k, "lam": lam,
                                  "dev_mae": float(p["abs_err_new"].mean())})
    grid = pd.DataFrame(grid_rows).sort_values("dev_mae").reset_index(drop=True)
    best = grid.iloc[0]
    dev_best = policy(dev, best["margin"], int(best["k"]), best["lam"])
    test_from_dev = policy(test, best["margin"], int(best["k"]), best["lam"])
    print("\n== 仅用 2025-09 选参 ==")
    print("best dev config:", best.to_dict())
    print("2025 dev MAE:", round(dev_best["abs_err_new"].mean(), 4),
          "| 2026 test MAE:", round(test_from_dev["abs_err_new"].mean(), 4))
    print("2025 dev plain v1 MAE:", round(abs(dev["point"] - dev["actual"]).mean(), 4),
          "| 2026 plain v1 MAE:", round(abs(test["point"] - test["actual"]).mean(), 4))
    grid.to_csv(OUT / "2026_rsi_v1_1_gate_bias_devgrid.csv", index=False)

    devsel = test_from_dev.copy()
    devsel["abs_err"] = devsel["abs_err_new"]
    devsel.to_csv(OUT / "2026_rsi_v1_1_gate_bias_devsel_details.csv", index=False)
    evaluate(devsel, args.alpha).to_csv(OUT / "2026_rsi_v1_1_gate_bias_devsel_summary.csv", index=False)

    # candidate artifact (separate from v1)
    cand = fixed[2026].copy()
    cand["abs_err"] = cand["abs_err_new"]
    cand.to_csv(OUT / "2026_rsi_v1_1_gate_bias_details.csv", index=False)
    summary = evaluate(cand, args.alpha)
    summary.to_csv(OUT / "2026_rsi_v1_1_gate_bias_summary.csv", index=False)

    ci = paired_bootstrap(fixed[2026][["station", "origin", "date", "abs_err_new"]].rename(columns={"abs_err_new": "abs_err_a"}),
                          fixed[2026][["station", "origin", "date", "abs_err_v1"]].rename(columns={"abs_err_v1": "abs_err_b"}),
                          ["station", "origin", "date"], args.boot)

    report = {
        "v1_2026_mae": float(abs(test["point"] - test["actual"]).mean()),
        "candidate_2026_mae": float(fixed[2026]["abs_err_new"].mean()),
        "v1_2025_mae": float(abs(dev["point"] - dev["actual"]).mean()),
        "candidate_2025_mae": float(fixed[2025]["abs_err_new"].mean()),
        "paired_bootstrap_2026": ci,
        "use_v1_rate_2026": float(fixed[2026]["use_v1"].mean()),
        "use_v1_rate_2025": float(fixed[2025]["use_v1"].mean()),
        "fixed": FIXED,
        "dev_selected": {k: (float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer, int)) else v)
                         for k, v in best.to_dict().items()},
        "dev_selected_2026_mae": float(test_from_dev["abs_err_new"].mean()),
        "dev_selected_use_v1_rate_2026": float(test_from_dev["use_v1"].mean()),
    }
    (OUT / "2026_rsi_v1_1_gate_bias_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n== 2026 配对 bootstrap ==")
    print(json.dumps(ci, ensure_ascii=False))
    print("\n已写出:")
    for name in ("2026_rsi_v1_1_gate_bias_details.csv", "2026_rsi_v1_1_gate_bias_summary.csv",
                 "2026_rsi_v1_1_gate_bias_devgrid.csv", "2026_rsi_v1_1_gate_bias_report.json"):
        print(" ", OUT / name)


if __name__ == "__main__":
    main()
