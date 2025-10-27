#!/usr/bin/env python3
"""
sl_by_power.py

Her bir 'power' değeri için SL olan işlemleri sayar ve özet istatistikleri çıkarır.
Çıktı:
  - terminale ilk 50 satırı basar
  - outputs/SL_by_power.csv dosyasına kaydeder

Çalıştırma:
  python sl_by_power.py
"""
import os
import json
from pathlib import Path

import pandas as pd

DATA_DIR = os.getenv("DATA_DIR", "./data")
FILE_PATH = os.path.join(DATA_DIR, "sim_closed.json")
OUT_DIR = "outputs"

def load_data(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dosya bulunamadı: {path}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def compute_duration_if_missing(df):
    if "duration_sec" not in df.columns:
        df["duration_sec"] = pd.NA
    if "open_time" in df.columns and "close_time" in df.columns:
        ot = pd.to_datetime(df["open_time"], utc=True, errors="coerce")
        ct = pd.to_datetime(df["close_time"], utc=True, errors="coerce")
        mask = df["duration_sec"].isna() & ot.notna() & ct.notna()
        df.loc[mask, "duration_sec"] = (ct[mask] - ot[mask]).dt.total_seconds()
    if "open_ts" in df.columns and "close_ts" in df.columns:
        mask = df["duration_sec"].isna() & df["open_ts"].notna() & df["close_ts"].notna()
        df.loc[mask, "duration_sec"] = df.loc[mask, "close_ts"] - df.loc[mask, "open_ts"]
    df["duration_sec"] = pd.to_numeric(df["duration_sec"], errors="coerce")
    df.loc[df["duration_sec"] < 0, "duration_sec"] = pd.NA
    return df

def main():
    data = load_data(FILE_PATH)
    df = pd.DataFrame(data)

    # Güvenlik: power numeric
    if "power" not in df.columns:
        print("Veride 'power' sütunu yok.")
        return
    df["power"] = pd.to_numeric(df["power"], errors="coerce")

    df = compute_duration_if_missing(df)

    # SL filtrelemesi (case-insensitive)
    if "exit_reason" not in df.columns:
        print("Veride 'exit_reason' sütunu yok; SL filtreleme yapılamaz.")
        return
    sl_df = df[df["exit_reason"].astype(str).str.upper() == "SL"].copy()
    if sl_df.empty:
        print("Veride 'SL' olarak işaretlenmiş hiç işlem yok.")
        return

    # group by power: count, avg gain, avg duration, median duration, min/max
    summary = (
        sl_df.groupby("power")
        .agg(
            sl_count=("exit_reason", "count"),
            avg_gain_pct=("gain_pct", "mean"),
            median_gain_pct=("gain_pct", "median"),
            avg_duration_sec=("duration_sec", "mean"),
            median_duration_sec=("duration_sec", "median"),
            min_duration_sec=("duration_sec", "min"),
            max_duration_sec=("duration_sec", "max"),
        )
        .reset_index()
        .sort_values("sl_count", ascending=False)
    )

    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    out_csv = Path(OUT_DIR) / "SL_by_power.csv"
    summary.to_csv(out_csv, index=False)

    # Print top results
    pd.set_option("display.float_format", "{:.3f}".format)
    print(f"SL by power summary saved to: {out_csv}\n")
    print("İlk 50 power (sl_count desc):")
    print(summary.head(50).to_string(index=False))

if __name__ == "__main__":
    main()
