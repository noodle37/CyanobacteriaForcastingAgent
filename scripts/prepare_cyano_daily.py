"""Parse data/*.xls hourly藻类观测 into daily series and fetch同期天气.

Usage:
    python scripts/prepare_cyano_daily.py            # parse xls + fetch weather (cached)
    python scripts/prepare_cyano_daily.py --no-weather
    python scripts/prepare_cyano_daily.py --refresh-weather

Outputs:
    data/daily/{station}_{year}_daily.csv   日历日表(5/1-9/30), 缺测日历日为NaN, 内部缺口线性插值
    data/weather/{station}_daily.csv        2024-05-01..2026-09-30 逐日天气 (Open-Meteo ERA5)
    samples/{station}_{year}_daily.json     {item_id, freq:"D", timestamps, values}, 截取首尾有效观测段
"""

import argparse
import json
import urllib.request
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# 坐标为近似位置(ERA5网格~9km, 同江段天气差异很小), 可按实测坐标修正
STATIONS = {
    "jiangjunyan": {"cn": "将军岩小微站(建德大洋)", "lat": 29.45, "lon": 119.50},
    "meichengmatou": {"cn": "梅城码头", "lat": 29.54, "lon": 119.55},
    "yanlingwu": {"cn": "严陵坞BBE", "lat": 29.776, "lon": 119.617},
    "yanziling": {"cn": "严子陵藻类剖面浮标站", "lat": 29.782, "lon": 119.633},
}
YEARS = (2024, 2025, 2026)

COLS = [
    "site", "time", "water_temp_c", "ph", "do_mgl", "bbe_cyano_ugl",
    "total_chla_ugl", "total_density", "cyano_chla_ugl", "cyano_density",
    "green_chla_ugl", "green_density", "diatom_chla_ugl", "diatom_density",
    "crypto_chla_ugl", "crypto_density", "cyano_ratio_pct", "cdom", "transparency_pct",
]
ENV_COLS = ["water_temp_c", "ph", "do_mgl", "cyano_ratio_pct", "transparency_pct"]

WEATHER_DAILY = [
    "temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
    "precipitation_sum", "shortwave_radiation_sum",
    "relative_humidity_2m_mean", "wind_speed_10m_max",
]


def find_xls(station: str, year: int) -> Path:
    names = [f"{station}_{year}_data_summer.xls"]
    if station == "meichengmatou" and year == 2024:
        names.append("machengmatou_2024_data_summer.xls")  # 原始文件名拼写
    for name in names:
        p = ROOT / "data" / name
        if p.exists():
            return p
    raise FileNotFoundError(f"{station} {year}: tried {names}")


def parse_hourly(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, header=None, skiprows=2)
    df.columns = COLS[: df.shape[1]]
    df["time"] = pd.to_datetime(df["time"], format="%Y-%m-%d %H", errors="coerce")
    df = df.dropna(subset=["time"])
    for c in COLS[2:]:
        df[c] = pd.to_numeric(df[c].replace("/", None), errors="coerce")
    df["date"] = df["time"].dt.normalize()
    # 蓝绿藻(BBE探头)优先, 无探头数据的站用蓝藻叶绿素a(实验室)
    df["cyano_ugl"] = df["bbe_cyano_ugl"].fillna(df["cyano_chla_ugl"])
    df["target_source"] = np.where(df["bbe_cyano_ugl"].notna(), "bbe", "lab_chla")
    return df.sort_values("time")


def to_daily(hourly: pd.DataFrame, year: int) -> pd.DataFrame:
    g = hourly.groupby("date")
    daily = g["cyano_ugl"].mean().rename("cyano_ugl").to_frame()
    daily["cyano_ugl_max"] = g["cyano_ugl"].max()
    daily["n_obs"] = g["cyano_ugl"].count()
    for c in ENV_COLS:
        daily[c] = g[c].mean()
    cal = pd.date_range(f"{year}-05-01", f"{year}-09-30", freq="D")
    daily = daily.reindex(cal)
    daily.index.name = "date"
    daily["n_obs"] = daily["n_obs"].fillna(0).astype(int)
    daily["interpolated"] = daily["cyano_ugl"].isna() & (daily["n_obs"] == 0)
    # 只插值观测段内部的缺口, 首尾缺测保持NaN(不 extrapolate)
    daily["cyano_ugl"] = daily["cyano_ugl"].interpolate(limit_area="inside")
    daily["interpolated"] &= daily["cyano_ugl"].notna()
    return daily.round(3)


def write_sample(daily: pd.DataFrame, station: str, year: int) -> dict:
    s = daily["cyano_ugl"].dropna()
    sample = {
        "item_id": f"{station}_{year}_daily",
        "freq": "D",
        "timestamps": [d.strftime("%Y-%m-%d") for d in s.index],
        "values": [round(float(v), 3) for v in s],
    }
    out = ROOT / "samples" / f"{station}_{year}_daily.json"
    out.write_text(json.dumps(sample, ensure_ascii=False, indent=1))
    return sample


def fetch_weather(station: str, refresh: bool) -> Path:
    out = ROOT / "data" / "weather" / f"{station}_daily.csv"
    if out.exists() and not refresh:
        return out
    cfg = STATIONS[station]
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={cfg['lat']}&longitude={cfg['lon']}"
        "&start_date=2024-05-01&end_date=2026-09-30"
        f"&daily={','.join(WEATHER_DAILY)}&timezone=Asia%2FShanghai"
    )
    with urllib.request.urlopen(url, timeout=60) as r:
        payload = json.load(r)
    w = pd.DataFrame(payload["daily"])
    w = w.rename(columns={"time": "date"})
    w.attrs.update(grid_lat=payload["latitude"], grid_lon=payload["longitude"])
    out.parent.mkdir(parents=True, exist_ok=True)
    w.to_csv(out, index=False)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-weather", action="store_true")
    ap.add_argument("--refresh-weather", action="store_true")
    args = ap.parse_args()
    warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

    daily_dir = ROOT / "data" / "daily"
    daily_dir.mkdir(parents=True, exist_ok=True)
    (ROOT / "samples").mkdir(exist_ok=True)

    rows = []
    for station in STATIONS:
        for year in YEARS:
            hourly = parse_hourly(find_xls(station, year))
            daily = to_daily(hourly, year)
            daily.to_csv(daily_dir / f"{station}_{year}_daily.csv")
            sample = write_sample(daily, station, year)
            src = hourly.loc[hourly["cyano_ugl"].notna(), "target_source"]
            rows.append({
                "station": station, "year": year,
                "span": f"{sample['timestamps'][0]}..{sample['timestamps'][-1]}",
                "days": len(sample["values"]),
                "observed": int((daily["n_obs"] > 0).sum()),
                "interp": int(daily["interpolated"].sum()),
                "sep_days": int(daily.loc[f"{year}-09-01":f"{year}-09-30", "cyano_ugl"].notna().sum()),
                "range": f"{daily['cyano_ugl'].min():.2f}..{daily['cyano_ugl'].max():.2f}",
                "source": src.mode().iat[0] if len(src) else "-",
            })
    print(pd.DataFrame(rows).to_string(index=False))

    if not args.no_weather:
        for station in STATIONS:
            p = fetch_weather(station, args.refresh_weather)
            w = pd.read_csv(p)
            print(f"weather {station}: {len(w)} days {w['date'].min()}..{w['date'].max()} -> {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()