"""传统深度学习基线逐日回放: LSTM / GRU / CNN-LSTM, 每站独立训练。

训练=2024全年+2025/2026的5~8月(标签仅实测日), 验证=2025-09(early stopping), 测试=2026-09。
输入窗口28天: [蓝藻ctx, Tmax, 降水, 辐射, 缺测mask], 天气只用≤origin, 窗口与标签不跨年。
--horizon 1: 预测t+1; --horizon 3: 预测t+1~t+3逐日值(标签/评分均要求3天全实测)。
--target mean: 日均值; max: 日最高值(cyano_ugl_max, 缺测日ctx用均值估测近似)。
输出q10/q50/q90(pinball loss, q50±softplus参数化保序), 蓝藻log1p+z-score, 3种子分位数平均。

Usage:
    python scripts/replay_baselines.py --target max --horizon 1
    python scripts/replay_baselines.py --target max --horizon 3
    python scripts/replay_baselines.py --archs transformer_lite --target max --horizon 1  # 负对照
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.replay_daily import STATIONS, load_station, summarize

OUT = ROOT / "data" / "replay"
CKPT = ROOT / "data" / "baselines"
WEATHER_COLS = ["temperature_2m_max", "precipitation_sum", "shortwave_radiation_sum"]
QUANTILES = (0.1, 0.5, 0.9)
ARCHS = ("lstm", "cnn_lstm", "gru", "transformer")
ABLATION_ARCHS = ("transformer_lastpool", "transformer_nopos", "transformer_meanpool",
                  "transformer_small", "transformer_lite")
ARCH_CHOICES = ARCHS + ABLATION_ARCHS
TRANSFORMER_VARIANTS = {
    # 标准基线: 可学习位置编码 + 均值池化 + hidden=32
    "transformer": {"pos": True, "pool": "mean", "hidden": None, "nhead": 4, "ffn_mult": 2},
    # 旧实现: 末步池化
    "transformer_lastpool": {"pos": True, "pool": "last", "hidden": None, "nhead": 4, "ffn_mult": 2},
    # 去掉可学习位置编码, 其余不变
    "transformer_nopos": {"pos": False, "pool": "mean", "hidden": None, "nhead": 4, "ffn_mult": 2},
    # 保留位置编码, 改为均值池化 (与标准基线相同, 保留兼容)
    "transformer_meanpool": {"pos": True, "pool": "mean", "hidden": None, "nhead": 4, "ffn_mult": 2},
    # 保留位置编码与末步池化, 缩小容量
    "transformer_small": {"pos": True, "pool": "last", "hidden": 8, "nhead": 2, "ffn": 16},
    # 去位置编码 + 均值池化 + 缩小容量: 负对照
    "transformer_lite": {"pos": False, "pool": "mean", "hidden": 8, "nhead": 2, "ffn": 16},
}


def build_frame(station: str, target: str = "mean") -> pd.DataFrame:
    w = pd.read_csv(ROOT / "data" / "weather" / f"{station}_daily.csv", parse_dates=["date"]).set_index("date")
    parts = []
    for year in (2024, 2025, 2026):
        d = load_station(station, year, target).join(w[WEATHER_COLS])
        d["year"] = year
        d["missing"] = (d["n_obs"] == 0).astype(float)
        d["ctx"] = d["ctx"].interpolate(limit_direction="both").ffill().bfill()
        parts.append(d)
    df = pd.concat(parts)
    assert df["ctx"].notna().all() and df[WEATHER_COLS].notna().all().all()
    return df


def make_windows(df: pd.DataFrame, L: int, H: int, observed_only: bool):
    """窗口=i-L..i-1, 标签=i..i+H-1; 同年约束; observed_only要求H个标签全实测。"""
    ctx = df["ctx"].to_numpy()
    wx = df[WEATHER_COLS].to_numpy()
    miss = df["missing"].to_numpy()
    actual = df["actual"].to_numpy()
    nobs = df["n_obs"].to_numpy()
    years = df["year"].to_numpy()
    X, y, idx = [], [], []
    for i in range(L, len(df) - H + 1):
        if years[i - L] != years[i + H - 1]:  # 年际9/30→5/1断档, 窗口与标签必须同年
            continue
        lab_ok = (nobs[i:i + H] > 0) & np.isfinite(actual[i:i + H])
        if observed_only and not lab_ok.all():
            continue
        X.append(np.column_stack([ctx[i - L:i], wx[i - L:i], miss[i - L:i]]))
        y.append(actual[i:i + H])
        idx.append(i)
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32), np.asarray(idx)


def split_mask(df: pd.DataFrame, idx: np.ndarray):
    dates = df.index[idx]
    train = (dates.year == 2024) | ((dates.year >= 2025) & (dates.month <= 8))
    val = (dates.year == 2025) & (dates.month == 9)
    return train, val


class Scaler:
    """蓝藻log1p后z-score, 天气z-score; 只用训练段统计。"""

    def fit(self, X: np.ndarray, y: np.ndarray):
        cy = np.log1p(np.concatenate([X[..., 0].ravel(), y.ravel()]))
        self.cy_mu, self.cy_sd = float(cy.mean()), float(cy.std() + 1e-8)
        wx = X[..., 1:4].reshape(-1, 3)
        self.wx_mu, self.wx_sd = wx.mean(0), wx.std(0) + 1e-8
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        out = X.copy()
        out[..., 0] = (np.log1p(X[..., 0]) - self.cy_mu) / self.cy_sd
        out[..., 1:4] = (X[..., 1:4] - self.wx_mu) / self.wx_sd
        return out

    def transform_y(self, y: np.ndarray) -> np.ndarray:
        return (np.log1p(y) - self.cy_mu) / self.cy_sd

    def inverse_y(self, y: np.ndarray) -> np.ndarray:
        return np.expm1(y * self.cy_sd + self.cy_mu)

    def state(self) -> dict:
        return {"cy_mu": self.cy_mu, "cy_sd": self.cy_sd,
                "wx_mu": self.wx_mu.tolist(), "wx_sd": self.wx_sd.tolist()}

    @classmethod
    def from_state(cls, s: dict) -> "Scaler":
        obj = cls()
        obj.cy_mu, obj.cy_sd = s["cy_mu"], s["cy_sd"]
        obj.wx_mu, obj.wx_sd = np.asarray(s["wx_mu"]), np.asarray(s["wx_sd"])
        return obj


class BaselineNet(nn.Module):
    def __init__(self, arch: str, horizon: int = 1, n_feat: int = 5, hidden: int = 32, dropout: float = 0.2):
        super().__init__()
        self.arch = arch
        self.horizon = horizon
        self.pool = "last"
        if arch == "cnn_lstm":
            self.front = nn.Sequential(
                nn.Conv1d(n_feat, hidden, 3, padding=1), nn.ReLU(),
                nn.Conv1d(hidden, hidden, 3, padding=1), nn.ReLU())
            self.backbone = nn.LSTM(hidden, hidden, batch_first=True)
        elif arch in ("lstm", "gru"):
            self.front = None
            self.backbone = (nn.GRU if arch == "gru" else nn.LSTM)(n_feat, hidden, batch_first=True)
        elif arch.startswith("transformer"):
            cfg = TRANSFORMER_VARIANTS[arch]
            hidden = cfg["hidden"] or hidden
            self.front = nn.Linear(n_feat, hidden)
            self.pos = nn.Parameter(torch.zeros(1, 512, hidden)) if cfg["pos"] else None
            layer = nn.TransformerEncoderLayer(
                hidden, nhead=cfg["nhead"],
                dim_feedforward=cfg["ffn"] if "ffn" in cfg else hidden * cfg["ffn_mult"],
                dropout=dropout, activation="gelu", batch_first=True)
            self.backbone = nn.TransformerEncoder(layer, num_layers=1)
            self.pool = cfg["pool"]
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, horizon * 3)

    def forward(self, x):
        if self.arch == "cnn_lstm":
            h = self.backbone(self.front(x.transpose(1, 2)).transpose(1, 2))[0]
            last = h[:, -1]
        elif self.arch.startswith("transformer"):
            z = self.front(x)
            if self.pos is not None:
                z = z + self.pos[:, :x.shape[1]]
            h = self.backbone(z)
            last = h.mean(dim=1) if self.pool == "mean" else h[:, -1]
        else:
            h = self.backbone(x)[0]
            last = h[:, -1]
        raw = self.head(self.drop(last)).view(-1, self.horizon, 3)
        q50 = raw[..., 1:2]
        return torch.cat([q50 - F.softplus(raw[..., 0:1]), q50, q50 + F.softplus(raw[..., 2:3])], dim=-1)


def load_checkpoint(path, device: str = "cpu"):
    """兼容重构前 rnn./conv. 前缀的旧 checkpoint。"""
    ck = torch.load(path, weights_only=False)
    model = BaselineNet(ck["arch"], horizon=ck.get("horizon", 1)).to(device)
    sd = ck["state_dict"]
    if any(k.startswith(("rnn.", "conv.")) for k in sd):
        sd = {("backbone." + k[4:] if k.startswith("rnn.") else
               "front." + k[5:] if k.startswith("conv.") else k): v for k, v in sd.items()}
    model.load_state_dict(sd)
    return model, ck


def pinball_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    q = torch.tensor(QUANTILES, device=pred.device)
    e = target.unsqueeze(-1) - pred
    return torch.maximum(q * e, (q - 1) * e).mean()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one(station: str, arch: str, seed: int, df: pd.DataFrame, L: int, H: int,
              epochs: int, device: str, target: str):
    set_seed(seed)
    X, y, idx = make_windows(df, L, H, observed_only=True)
    train, val = split_mask(df, idx)
    scaler = Scaler().fit(X[train], y[train])
    Xt = torch.tensor(scaler.transform(X[train]), device=device)
    yt = torch.tensor(scaler.transform_y(y[train]), device=device)
    Xv = torch.tensor(scaler.transform(X[val]), device=device)
    yv = torch.tensor(scaler.transform_y(y[val]), device=device)

    model = BaselineNet(arch, horizon=H).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_val, best_epoch, best_state, curve = float("inf"), 0, None, []
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(Xt), device=device)
        for b in range(0, len(Xt), 32):
            sel = perm[b:b + 32]
            opt.zero_grad()
            loss = pinball_loss(model(Xt[sel]), yt[sel])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = pinball_loss(model(Xv), yv).item()
        curve.append(v)
        if v < best_val - 1e-6:
            best_val, best_epoch = v, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if epoch - best_epoch >= 20:
            break
    model.load_state_dict(best_state)
    CKPT.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "scaler": scaler.state(), "arch": arch, "station": station,
                "seed": seed, "target": target, "horizon": H, "context_days": L,
                "best_epoch": best_epoch, "val_curve": curve},
               CKPT / f"{station}_{arch}_{target}_h{H}_s{seed}.pt")
    return model, scaler, best_epoch


def predict(model: BaselineNet, scaler: Scaler, X: np.ndarray, device: str) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        q = model(torch.tensor(scaler.transform(X), device=device)).cpu().numpy()
    return np.clip(scaler.inverse_y(q), 0, None)


def self_test(df: pd.DataFrame, L: int, H: int) -> None:
    X, y, idx = make_windows(df, L, H, observed_only=True)
    years = df["year"].to_numpy()
    assert (years[idx - L] == years[idx + H - 1]).all(), "窗口或标签跨年"
    assert np.isfinite(X).all() and np.isfinite(y).all()
    train, _ = split_mask(df, idx)
    scaler = Scaler().fit(X[train], y[train])
    assert np.max(np.abs(scaler.inverse_y(scaler.transform_y(y[train])) - y[train])) < 1e-3
    print("self-test PASS: 窗口/标签同年/无NaN/scaler往返")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", nargs="+", choices=ARCH_CHOICES, default=list(ARCHS))
    ap.add_argument("--stations", nargs="+", default=list(STATIONS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--target", choices=["mean", "max"], default="mean")
    ap.add_argument("--horizon", type=int, choices=[1, 3], default=1)
    ap.add_argument("--context-days", type=int, default=28)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    L, H, target = args.context_days, args.horizon, args.target
    tag = f"{target}_h{H}_ctx{L}"
    OUT.mkdir(parents=True, exist_ok=True)

    rows_by_arch = {a: [] for a in args.archs}
    seed_rows = []
    for station in args.stations:
        df = build_frame(station, target)
        self_test(df, L, H)
        Xtest, _, tidx = make_windows(df, L, H, observed_only=False)
        tdates = df.index[tidx]
        test_sel = (tdates.year == 2026) & (tdates.month == 9)
        Xtest, tidx = Xtest[test_sel], tidx[test_sel]
        test_df = df.iloc[tidx]
        lab_obs = np.stack([(df["n_obs"].to_numpy()[i:i + H] > 0) &
                            np.isfinite(df["actual"].to_numpy()[i:i + H]) for i in tidx])
        window_obs = lab_obs.all(axis=1)  # H3评分口径: 3天全实测
        actual = np.stack([df["actual"].to_numpy()[i:i + H] for i in tidx])
        score = window_obs if H > 1 else lab_obs[:, 0]
        persistence_mae = np.abs(df["ctx"].to_numpy()[tidx - 1][score] - actual[score, 0]).mean()

        for arch in args.archs:
            seed_preds = []
            for seed in args.seeds:
                model, scaler, best_epoch = train_one(station, arch, seed, df, L, H, args.epochs, args.device, target)
                q = predict(model, scaler, Xtest, args.device)
                assert (q[..., 0] <= q[..., 1]).all() and (q[..., 1] <= q[..., 2]).all()
                seed_preds.append(q)
                cov = ((q[..., 0] <= actual) & (actual <= q[..., 2]))[score].mean()
                mae = np.abs(q[..., 1] - actual)[score].mean()
                seed_rows.append({"arch": arch, "station": station, "seed": seed, "best_epoch": best_epoch,
                                  "days": int(score.sum() * H), "coverage": round(cov, 3), "mae": round(mae, 3)})
                print(f"{station}/{arch}/s{seed}: best_epoch={best_epoch} 覆盖{cov:.1%} MAE{mae:.3f}", flush=True)
            q = np.mean(seed_preds, axis=0)
            assert (q[..., 0] <= q[..., 1]).all() and (q[..., 1] <= q[..., 2]).all()
            mae = np.abs(q[..., 1] - actual)[score].mean()
            print(f"{station}/{arch}: 种子平均MAE{mae:.3f} vs persistence{persistence_mae:.3f}", flush=True)
            for j, (date, _) in enumerate(test_df.iterrows()):
                origin = date - pd.Timedelta(days=1)
                for h in range(H):
                    tdate = date + pd.Timedelta(days=h)
                    row = df.loc[tdate]
                    o = bool(lab_obs[j, h])
                    scored = bool(window_obs[j]) if H > 1 else o
                    rows_by_arch[arch].append({
                        "station": station, "date": str(tdate.date()), "origin": str(origin.date()), "h": h + 1,
                        "model": arch, "target": target, "context_len": L,
                        "actual": round(float(row["actual"]), 3) if o else None,
                        "point": round(float(q[j, h, 1]), 3),
                        "q10": round(float(q[j, h, 0]), 3), "q90": round(float(q[j, h, 2]), 3),
                        "observed": o, "window_observed": bool(window_obs[j]),
                        "covered": bool(o and q[j, h, 0] <= row["actual"] <= q[j, h, 2]) if o else None,
                        "abs_err": round(float(abs(q[j, h, 1] - row["actual"])), 3) if o else None,
                        "scored": scored,
                    })

    for arch, rows in rows_by_arch.items():
        det = pd.DataFrame(rows)
        det.to_csv(OUT / f"2026_{arch}_{tag}_details.csv", index=False)
        summary = summarize(det[det["scored"]])
        summary.to_csv(OUT / f"2026_{arch}_{tag}_summary.csv", index=False)
        print(f"\n== {arch} {tag} ==")
        print(summary.to_string(index=False))
        o = det[det["scored"]]
        print(f"总体: 覆盖率 {o['covered'].mean():.1%}, MAE {o['abs_err'].mean():.3f}, 平均区间宽 {(o['q90'] - o['q10']).mean():.2f}")
    sm = pd.DataFrame(seed_rows)
    for arch in args.archs:
        sm[sm["arch"] == arch].to_csv(CKPT / f"seed_metrics_{arch}_{tag}.csv", index=False)


if __name__ == "__main__":
    main()