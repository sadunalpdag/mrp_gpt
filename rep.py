#!/usr/bin/env python3
"""
per_power_integer.py

Her bir tam sayı power değeri veya tam sayı aralıkları (bin_size) için özet çıkarır:
 - total_trades, closed_trades, open_trades
 - kapatılmış işlemler için avg/median/min/max duration (saniye)

MODE:
 - "per_integer": power'ı round ile tam sayıya çevirir ve her tam sayı için özet çıkarır.
 - "integer_bins": bin_size ile tam sayı aralıkları oluşturur (ör. bin_size=5 -> 0-4,5-9,...).
"""
import os
import json
from pathlib import Path

import pandas as pd

DATA_DIR = os.getenv("DATA_DIR", "./data")
FILE_PATH = os.path.join(DATA_DIR, "sim_closed.json")
OUT_DIR = "outputs"

# Ayarlar: per_integer veya integer_bins
MODE = "per_integer"   # "per_integer" veya "integer_bins"
BIN_SIZE = 1           # integer_bins için; 1 olursa her tam sayı ayrı aralık olur
MIN_POWER = 0          # binler için alt sınır (inclusive)
MAX_POWER = 100        # binler için üst sınır (exclusive üst sınır olarak kullanılabilir)

def load_json(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dosya bulunamadı: {path}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def compute_duration(df):
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

def prepare_power(df):
    if "power" not in df.columns:
        df["power"] = pd.NA
    df["power"] = pd.to_numeric(df["power"], errors="coerce")
    return df

def add_integer_groups(df, mode="per_integer", bin_size=1, min_p=0, max_p=100):
    if mode == "per_integer":
        # En yakın tam sayıya yuvarla; NaN'lar korunur
        df["power_int"] = df["power"].round().astype("Int64")
        group_col = "power_int"
    else:
        # integer_bins: 0..bin_size-1, bin_size..2*bin_size-1, ...
        if bin_size < 1:
            raise ValueError("bin_size must be >= 1")
        bins = list(range(int(min_p), int(max_p) + bin_size, int(bin_size)))
        # labels: "0-0" (bin_size=1) or "0-4" (bin_size=5)
        labels = [f"{b}-{b+bin_size-1}" for b in bins[:-1]]
        df["power_bin"] = pd.cut(df["power"], bins=bins, labels=labels, include_lowest=True, right=False)
        group_col = "power_bin"
    return df, group_col

def summarize(df, group_col):
    # total trades per group
    total = df.groupby(group_col).size().rename("total_trades")
    # closed definition
    is_closed = pd.Series(False, index=df.index)
    if "status" in df.columns:
        is_closed = is_closed | (df["status"].astype(str).str.upper() == "CLOSED")
    if "exit_reason" in df.columns:
        is_closed = is_closed | df["exit_reason"].notna()
    closed_series = df[is_closed].groupby(group_col).size().rename("closed_trades")
    open_series = (total - closed_series).rename("open_trades").fillna(total)

    # duration stats on closed trades with duration
    dur_df = df[is_closed].dropna(subset=["duration_sec"]).copy()
    dur_stats = dur_df.groupby(group_col)["duration_sec"].agg(
        avg_duration_sec="mean",
        median_duration_sec="median",
        min_duration_sec="min",
        max_duration_sec="max",
    )

    summary = pd.concat([total, closed_series, open_series, dur_stats], axis=1).fillna(0)
    # cast counts to int where appropriate
    for c in ["total_trades", "closed_trades", "open_trades"]:
        if c in summary.columns:
            summary[c] = summary[c].astype(int)
    summary = summary.reset_index()
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

    df, group_col = add_integer_groups(df, mode=MODE, bin_size=BIN_SIZE, min_p=MIN_POWER, max_p=MAX_POWER)

    summary = summarize(df, group_col)

    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    out_csv = Path(OUT_DIR) / f"per_power_integer_summary_{MODE}.csv"
    summary.to_csv(out_csv, index=False)

    print(f"Summary saved to: {out_csv}")
    # show top 30 (by closed_trades desc) and fastest by avg_duration
    print("\nTop 30 by closed_trades:")
    if "closed_trades" in summary.columns:
        print(summary.sort_values("closed_trades", ascending=False).head(30).to_string(index=False))
    else:
        print(summary.head(30).to_string(index=False))

    if "avg_duration_sec" in summary.columns and summary["closed_trades"].sum() > 0:
        fastest = summary[summary["closed_trades"] > 0].sort_values("avg_duration_sec").head(20)
        print("\nEn hızlı kapanan ilk 20 (ortalama süreye göre):")
        for _, r in fastest.iterrows():
            label = r[group_col]
            print(f"  {label}: total={r['total_trades']} closed={r['closed_trades']} avg={human(r['avg_duration_sec'])}")
    else:
        print("\nKapatılmış işlem içeren grup bulunamadı veya avg_duration yok.")

if __name__ == "__main__":
    main()
