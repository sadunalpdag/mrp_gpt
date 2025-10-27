#!/usr/bin/env python3
"""
per_power_summary.py

Her bir 'power' değeri için:
 - toplam açılan işlem sayısı (total_trades)
 - kapatılmış işlem sayısı (closed_trades; status == 'CLOSED' veya exit_reason mevcut)
 - açık kalan işlem sayısı (open_trades)
 - kapatılmış işlemler için ortalama/medyan/min/max kapanma süresi (saniye)

Çalıştırma:
  python per_power_summary.py

Çıktı:
  - prints: kısa özet ve en hızlı kapanan ilk 10 power değeri
  - saves: outputs/per_power_summary.csv
"""
import os
import json
from pathlib import Path

import pandas as pd

DATA_DIR = os.getenv("DATA_DIR", "./data")
FILE_PATH = os.path.join(DATA_DIR, "sim_closed.json")
OUT_DIR = "outputs"

def load_json(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dosya bulunamadı: {path}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def compute_duration(df):
    # Ensure duration_sec exists; if not, try to compute from close_time/open_time or close_ts/open_ts
    if "duration_sec" not in df.columns:
        df["duration_sec"] = pd.NA

    # Use ISO timestamps if available
    if "open_time" in df.columns and "close_time" in df.columns:
        ot = pd.to_datetime(df["open_time"], utc=True, errors="coerce")
        ct = pd.to_datetime(df["close_time"], utc=True, errors="coerce")
        mask = df["duration_sec"].isna() & ot.notna() & ct.notna()
        df.loc[mask, "duration_sec"] = (ct[mask] - ot[mask]).dt.total_seconds()

    # Use unix timestamps if available
    if "open_ts" in df.columns and "close_ts" in df.columns:
        mask = df["duration_sec"].isna() & df["open_ts"].notna() & df["close_ts"].notna()
        df.loc[mask, "duration_sec"] = df.loc[mask, "close_ts"] - df.loc[mask, "open_ts"]

    # Ensure numeric and drop negative values
    df["duration_sec"] = pd.to_numeric(df["duration_sec"], errors="coerce")
    df.loc[df["duration_sec"] < 0, "duration_sec"] = pd.NA

    return df

def prepare_power(df):
    # Ensure power numeric
    if "power" not in df.columns:
        df["power"] = pd.NA
    df["power"] = pd.to_numeric(df["power"], errors="coerce")
    return df

def summarize_by_power(df):
    # Basic counts per power
    total = df.groupby("power").size().rename("total_trades")
    # closed definition: status == 'CLOSED' OR exit_reason notna
    is_closed = pd.Series(False, index=df.index)
    if "status" in df.columns:
        is_closed = is_closed | (df["status"].astype(str).str.upper() == "CLOSED")
    if "exit_reason" in df.columns:
        is_closed = is_closed | df["exit_reason"].notna()

    closed_series = df[is_closed].groupby("power").size().rename("closed_trades")
    open_series = (total - closed_series).rename("open_trades").fillna(total)  # if no closed entries, open == total

    # Duration stats computed only on closed trades (dropna durations)
    dur_df = df[is_closed].dropna(subset=["duration_sec"]).copy()
    dur_stats = dur_df.groupby("power")["duration_sec"].agg(
        avg_duration_sec="mean",
        median_duration_sec="median",
        min_duration_sec="min",
        max_duration_sec="max"
    )

    # Combine all into one DataFrame
    summary = pd.concat([total, closed_series, open_series, dur_stats], axis=1).fillna(0)
    # Convert counts to int
    for c in ["total_trades", "closed_trades", "open_trades"]:
        if c in summary.columns:
            summary[c] = summary[c].astype(int)
    # Sort by power ascending
    summary = summary.reset_index().sort_values("power").reset_index(drop=True)
    return summary

def human(sec):
    try:
        s = float(sec)
    except Exception:
        return ""
    m = int(s // 60)
    ss = int(s % 60)
    return f"{s:.2f}s ({m}m{ss}s)"

def main():
    try:
        data = load_json(FILE_PATH)
    except FileNotFoundError as e:
        print(e)
        return

    df = pd.DataFrame(data)
    df = compute_duration(df)
    df = prepare_power(df)

    summary = summarize_by_power(df)

    Path(OUT_DIR).mkdir(exist_ok=True, parents=True)
    out_csv = Path(OUT_DIR) / "per_power_summary.csv"
    summary.to_csv(out_csv, index=False)
    
    # print brief summary and top 10 fastest avg durations (only where closed_trades > 0)
    print(f"Per-power summary saved to: {out_csv}\n")
    print("İlk 20 satır (power, total, closed, open, avg_duration):")
    display_cols = ["power", "total_trades", "closed_trades", "open_trades", "avg_duration_sec"]
    # show up to 20 rows
    print(summary[display_cols].head(20).to_string(index=False, float_format="%.3f"))

    fastest = summary[summary["closed_trades"] > 0].sort_values("avg_duration_sec").head(10)
    if not fastest.empty:
        print("\nEn hızlı kapanan ilk 10 power (ortalama süreye göre):")
        for _, r in fastest.iterrows():
            print(f"  power={r['power']:.6f}  total={r['total_trades']}  closed={r['closed_trades']}  avg={human(r['avg_duration_sec'])}")
    else:
        print("\nKapatılmış işlem içeren hiç power bulunamadı.")

if __name__ == "__main__":
    main()
