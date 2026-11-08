"""评估 H3 "未来天气只给 RSI" 实验 (2026-09)。

对比:
  rsi_v1         : 2026_rsi_v1_max_h3_details.csv
  rsi_v2         : 2026_rsi_max_h3_details.csv      (canonical)
  rsi_v2_fwx     : 2026_rsi_max_h3_fwx_details.csv  (融合时注入 t+1..t+3 未来天气)
  tsfm_chronos2  : 2026_chronos2_max_ctx28_h3_details.csv (无未来天气)

注意: 未来天气取自逐日实测/再分析, 不是预报存档, 因此 fwx 是"完美未来天气"上界实验。
只给 RSI 的融合环节, TSFM 与所有基线仍只用 <=origin 的信息。

Usage:
    python scripts/eval_future_weather.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "replay"

FILES = {
    "rsi_v1": "2026_rsi_v1_max_h3_details.csv",
    "rsi_v2": "2026_rsi_max_h3_details.csv",
    "rsi_v2_fwx": "2026_rsi_max_h3_fwx_details.csv",
    "tsfm_chronos2": "2026_chronos2_max_ctx28_h3_details.csv",
}
OPTIONAL_FILES = {
    "rsi_v2_fwx_cap10": "2026_rsi_max_h3_fwx_cap0.1_details.csv",
    "rsi_v2_fwx_relax_cap10": "2026_rsi_max_h3_fwx_relax_cap0.1_details.csv",
}
KEY = ["station", "origin", "date", "h"]


def metrics(g: pd.DataFrame, alpha: float = 2.0) -> dict:
    y = g["actual"].to_numpy(dtype=float)
    yhat = g["point"].to_numpy(dtype=float)
    e = yhat - y
    w = 1.0 + alpha * y / y.max()
    return {"n": len(g),
            "mae": round(float(np.abs(e).mean()), 3),
            "rmse": round(float(np.sqrt((e ** 2).mean())), 3),
            "wmae": round(float((w * np.abs(e)).sum() / w.sum()), 3),
            "wrmse": round(float(np.sqrt((w * e ** 2).sum() / w.sum())), 3),
            "pre": round(float(abs(y.max() - yhat.max()) / y.max()), 3)}


def paired_bootstrap(a: pd.DataFrame, b: pd.DataFrame, n_boot: int = 10000, seed: int = 7) -> dict:
    m = a[KEY + ["actual", "point"]].merge(b[KEY + ["actual", "point"]], on=KEY, suffixes=("_a", "_b"))
    ea = (m["point_a"] - m["actual_a"]).abs().to_numpy(dtype=float)
    eb = (m["point_b"] - m["actual_b"]).abs().to_numpy(dtype=float)
    diff = ea - eb
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    boot = diff[idx].mean(axis=1)
    return {"n": len(diff), "diff": round(float(diff.mean()), 4),
            "lo": round(float(np.quantile(boot, 0.025)), 4),
            "hi": round(float(np.quantile(boot, 0.975)), 4),
            "p_two_sided": round(float(2 * min((boot <= 0).mean(), (boot >= 0).mean())), 4),
            "win_rate": round(float((diff > 0).mean()), 3)}


def load() -> dict[str, pd.DataFrame]:
    files = dict(FILES)
    for name, fname in OPTIONAL_FILES.items():
        if (OUT / fname).exists():
            files[name] = fname
    frames = {}
    for name, fname in files.items():
        p = OUT / fname
        if not p.exists():
            raise SystemExit(f"缺少 {p}")
        d = pd.read_csv(p)
        if "scored" in d.columns:
            d = d[d["scored"] == True].copy()
        frames[name] = d
    base = set(map(tuple, frames["rsi_v2"][KEY].to_numpy()))
    for name, d in frames.items():
        got = set(map(tuple, d[KEY].to_numpy()))
        if got != base:
            raise SystemExit(f"{name} 评分样本集合不一致: {len(got)} vs {len(base)}")
    return frames


def main() -> None:
    frames = load()
    rows = []
    for name, d in frames.items():
        rows.append({"system": name, "segment": "overall", **metrics(d)})
        for st, g in d.groupby("station", sort=True):
            rows.append({"system": name, "segment": st, **metrics(g)})
        for h, g in d.groupby("h", sort=True):
            rows.append({"system": name, "segment": f"t+{h}", **metrics(g)})
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "2026_h3_future_weather_eval.csv", index=False)

    print("== H3 总体 ==")
    print(table[table["segment"] == "overall"].to_string(index=False))
    print("\n== 逐站 ==")
    print(table[table["segment"].isin(["jiangjunyan", "meichengmatou", "yanlingwu", "yanziling"])]
          .pivot(index="segment", columns="system", values="mae").to_string())
    print("\n== 逐 horizon (MAE) ==")
    print(table[table["segment"].isin(["t+1", "t+2", "t+3"])]
          .pivot(index="segment", columns="system", values="mae").to_string())

    comparisons = {f"fwx_vs_{name}": paired_bootstrap(frames["rsi_v2_fwx"], frames[name])
                   for name in frames if name != "rsi_v2_fwx"}
    if "rsi_v2_fwx_cap10" in frames:
        comparisons["cap10_vs_v2"] = paired_bootstrap(frames["rsi_v2_fwx_cap10"], frames["rsi_v2"])
        comparisons["cap10_vs_fwx"] = paired_bootstrap(frames["rsi_v2_fwx_cap10"], frames["rsi_v2_fwx"])
    if "rsi_v2_fwx_relax_cap10" in frames:
        comparisons["relax_vs_v2"] = paired_bootstrap(frames["rsi_v2_fwx_relax_cap10"], frames["rsi_v2"])
        comparisons["relax_vs_fwx"] = paired_bootstrap(frames["rsi_v2_fwx_relax_cap10"], frames["rsi_v2_fwx"])
        comparisons["relax_vs_cap10"] = paired_bootstrap(frames["rsi_v2_fwx_relax_cap10"], frames["rsi_v2_fwx_cap10"])
    rep = {
        "note": "未来天气为逐日实测/再分析值(非预报存档), 因此 rsi_v2_fwx 是完美未来天气上界实验; "
                "未来天气只进入 RSI 融合 prompt, TSFM/基线不变。",
        "files": FILES,
        "overall": {name: metrics(d) for name, d in frames.items()},
        **comparisons,
    }
    (OUT / "2026_h3_future_weather_report.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n== 配对 bootstrap (diff = 前者误差 - 后者误差; 负值表示前者更好) ==")
    print(json.dumps(comparisons, ensure_ascii=False, indent=2))
    print("\n已写出:", OUT / "2026_h3_future_weather_eval.csv", OUT / "2026_h3_future_weather_report.json")


if __name__ == "__main__":
    main()
