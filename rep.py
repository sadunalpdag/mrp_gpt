#!/usr/bin/env python3
"""
per_power_integer_interval.py

Her tam sayı aralığı için (ör. 60-61 => 60 <= power < 61):
 - toplam işlem sayısı
 - closed işlemler (status == 'CLOSED' veya exit_reason dolu)
 - TP sayısı, SL sayısı
 - kapatılmış işlemler için avg/median/min/max duration (saniye)
 - avg gain_pct (kapatılmış işlemler için)

Usage:
  python per_power_integer_interval.py            # bütün integer aralıkları hesaplar ve outputs/power_integer_summary.csv kaydeder
  python per_power_integer_interval.py --power 60 # sadece 60-61 aralığını yazdırır ve kayıtta gösterir
  python per_power_integer_interval.py --min 60 --max 65  # 60..65 aralıklarını yazdırır (her bir tam sayı için ayrı satır)
"""
from pathlib import Path
import os
import json
import argparse
import math

import pandas as pd
import numpy as np

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
    # ensure duration_sec column exists
    if "duration_sec" not in df.columns:
        df["duration_sec"] = pd.NA

    # Prefer ISO datetimes if present
    if "open_time" in df.columns and "close_time" in df.columns:
        ot = pd.to_datetime(df["open_time"], utc=True, errors="coerce")
        ct = pd.to_datetime(df["close_time"], utc=True, errors="coerce")
        mask = df["duration_sec"].isna() & ot.notna() & ct.notna()
        df.loc[mask, "duration_sec"] = (ct[mask] - ot[mask]).dt.total_seconds()

    # Use unix timestamps if available
    if "open_ts" in df.columns and "close_ts" in df.columns:
        mask = df["duration_sec"].isna() & df["open_ts"].notna() & df["close_ts"].notna()
        df.loc[mask, "duration_sec"] = df.loc[mask, "close_ts"] - df.loc[mask, "open_ts"]

    # Fallback: if there is open_after_ts maybe approximate
    if "open_ts" in df.columns and "open_after_ts" in df.columns:
        mask = df["duration_sec"].isna() & df["open_ts"].notna() & df["open_after_ts"].notna()
        df.loc[mask, "duration_sec"] = df.loc[mask, "open_after_ts"] - df.loc[mask, "open_ts"]

    df["duration_sec"] = pd.to_numeric(df["duration_sec"], errors="coerce")
    # negatif değerleri NaN yap
    df.loc[df["duration_sec"] < 0, "duration_sec"] = pd.NA
    return df

def prepare_power(df):
    if "power" not in df.columns:
        df["power"] = pd.NA
    df["power"] = pd.to_numeric(df["power"], errors="coerce")
    # create integer floor group: for power in [n, n+1) => power_int == n
    df["power_int"] = np.floor(df["power"]).astype("Int64")
    return df

def is_closed_series(df):
    closed = pd.Series(False, index=df.index)
    if "status" in df.columns:
        closed = closed | (df["status"].astype(str).str.upper() == "CLOSED")
    if "exit_reason" in df.columns:
        closed = closed | df["exit_reason"].notna()
    return closed

def summarize_by_integer(df, min_int=None, max_int=None):
    closed_mask = is_closed_series(df)
    # consider only rows with non-null power_int for grouping
    df_valid = df[df["power_int"].notna()].copy()
    if df_valid.empty:
        return pd.DataFrame()

    # if min/max not provided, infer from data
    ints = df_valid["power_int"].dropna().unique().astype(int)
    min_data = ints.min()
    max_data = ints.max()
    if min_int is None:
        min_int = min_data
    if max_int is None:
        max_int = max_data

    # limit to provided range
    int_range = list(range(min_int, max_int + 1))
    rows = []
    for n in int_range:
        mask = df_valid["power_int"] == n
        group = df_valid[mask]
        total = int(group.shape[0])
        closed = int(group[is_closed_series(group)].shape[0])
        tp = 0
        sl = 0
        avg_gain = None
        dur_stats = {"avg": None, "median": None, "min": None, "max": None}
        if closed > 0:
            closed_df = group[is_closed_series(group)].copy()
            # TP/SL counts if exit_reason present
            if "exit_reason" in closed_df.columns:
                er = closed_df["exit_reason"].astype(str).str.upper()
                tp = int((er == "TP").sum())
                sl = int((er == "SL").sum())
            # avg gain_pct on closed (if exists)
            if "gain_pct" in closed_df.columns:
                avg_gain = float(pd.to_numeric(closed_df["gain_pct"], errors="coerce").dropna().mean()) if not closed_df["gain_pct"].dropna().empty else None
            # duration stats
            if "duration_sec" in closed_df.columns:
                ds = pd.to_numeric(closed_df["duration_sec"], errors="coerce").dropna()
                if not ds.empty:
                    dur_stats["avg"] = float(ds.mean())
                    dur_stats["median"] = float(ds.median())
                    dur_stats["min"] = float(ds.min())
                    dur_stats["max"] = float(ds.max())
        rows.append({
            "power_int": n,
            "range": f"{n}-{n+1}",
            "total_trades": total,
            "closed_trades": closed,
            "tp_count": tp,
            "sl_count": sl,
            "avg_gain_pct": round(avg_gain, 6) if avg_gain is not None else None,
            "avg_duration_sec": round(dur_stats["avg"], 3) if dur_stats["avg"] is not None else None,
            "median_duration_sec": round(dur_stats["median"], 3) if dur_stats["median"] is not None else None,
            "min_duration_sec": round(dur_stats["min"], 3) if dur_stats["min"] is not None else None,
            "max_duration_sec": round(dur_stats["max"], 3) if dur_stats["max"] is not None else None
        })
    out_df = pd.DataFrame(rows)
    return out_df

def human(sec):
    if sec is None:
        return ""
    s = float(sec)
    m = int(s // 60)
    ss = int(s % 60)
    return f"{s:.2f}s ({m}m{ss}s)"

def main():
    parser = argparse.ArgumentParser(description="Per-integer power interval summary")
    parser.add_argument("--power", type=int, help="Show only this integer interval (e.g. --power 60 => 60-61)")
    parser.add_argument("--min", type=int, dest="min_i", help="Minimum integer to include (inclusive)")
    parser.add_argument("--max", type=int, dest="max_i", help="Maximum integer to include (inclusive)")
    parser.add_argument("--out", type=str, default=OUT_DIR, help="Output directory")
    args = parser.parse_args()

    try:
        data = load_json(FILE_PATH)
    except FileNotFoundError as e:
        print(e)
        return

    df = pd.DataFrame(data)
    df = compute_duration(df)
    df = prepare_power(df)

    min_i = args.min_i
    max_i = args.max_i
    summary = summarize_by_integer(df, min_int=min_i, max_int=max_i)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "power_integer_summary.csv"
    summary.to_csv(out_csv, index=False)
    print(f"Summary saved to: {out_csv}")

    # If user requested single power integer, print a friendly summary
    if args.power is not None:
        p = args.power
        row = summary[summary["power_int"] == p]
        if row.empty:
            print(f"No data for interval {p}-{p+1}. (power may be missing or out of range)")
        else:
            r = row.iloc[0]
            print(f"\nStats for interval {r['range']}:")
            print(f"  Total trades: {r['total_trades']}")
            print(f"  Closed trades: {r['closed_trades']}")
            print(f"  TP count: {r['tp_count']}")
            print(f"  SL count: {r['sl_count']}")
            print(f"  Avg gain_pct (closed): {r['avg_gain_pct']}")
            print(f"  Avg duration (closed): {human(r['avg_duration_sec'])}")
            print(f"  Median duration: {human(r['median_duration_sec'])}")
            print(f"  Min duration: {human(r['min_duration_sec'])}")
            print(f"  Max duration: {human(r['max_duration_sec'])}")
    else:
        # Print top 20 intervals with closed trades by avg duration ascending
        if not summary.empty:
            fastest = summary[summary["closed_trades"] > 0].sort_values("avg_duration_sec").head(20)
            if not fastest.empty:
                print("\nEn hızlı kapanan ilk 20 integer interval (ortalama süreye göre):")
                for _, r in fastest.iterrows():
                    print(f"  {r['range']}: total={r['total_trades']} closed={r['closed_trades']} tp={r['tp_count']} sl={r['sl_count']} avg={human(r['avg_duration_sec'])}")
            else:
                print("\nKapatılmış işlem içeren interval yok.")
        else:
            print("Özet boş.")

if __name__ == "__main__":
    main()
