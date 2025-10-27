# power_band_summary.py
# Gereksinimler: pandas, matplotlib, seaborn
# Çalıştırma: python power_band_summary.py
import os
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = os.getenv("DATA_DIR", "./data")
FILE_PATH = os.path.join(DATA_DIR, "sim_closed.json")

def load_json_safe(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dosya bulunamadı: {path}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def prepare_dataframe(data):
    df = pd.DataFrame(data)

    cols = ["symbol", "dir", "power", "exit_reason", "gain_pct", "duration_sec"]
    present = [c for c in cols if c in df.columns]
    df = df[present].copy()

    if "power" in df.columns:
        df["power"] = pd.to_numeric(df["power"], errors="coerce")
    else:
        df["power"] = pd.NA

    if "gain_pct" in df.columns:
        df["gain_pct"] = pd.to_numeric(df["gain_pct"], errors="coerce")
    if "duration_sec" in df.columns:
        df["duration_sec"] = pd.to_numeric(df["duration_sec"], errors="coerce")

    bins = [0, 60, 70, 80, 90, 100]
    labels = ["<60", "60-70", "70-80", "80-90", ">90"]
    df["power_band"] = pd.cut(df["power"], bins=bins, labels=labels, include_lowest=True)

    return df

def compute_summary(df):
    # Güvenli bir şekilde trade_count, avg_gain_pct, avg_duration_min oluşturur
    group_cols = ["power_band", "exit_reason"]

    # Her zaman hesaplanabilen trade_count (groupby.size)
    counts = df.groupby(group_cols).size().reset_index(name="trade_count")

    # avg_gain_pct yalnızca gain_pct sütunu varsa
    if "gain_pct" in df.columns:
        avg_gain = (
            df.groupby(group_cols)["gain_pct"]
            .mean()
            .reset_index(name="avg_gain_pct")
        )
    else:
        avg_gain = pd.DataFrame(columns=group_cols + ["avg_gain_pct"])

    # avg_duration_min yalnızca duration_sec sütunu varsa
    if "duration_sec" in df.columns:
        avg_dur = (
            df.groupby(group_cols)["duration_sec"]
            .mean()
            .reset_index(name="avg_duration_min")
        )
        # saniyeyi dakikaya çevir
        if not avg_dur.empty:
            avg_dur["avg_duration_min"] = avg_dur["avg_duration_min"] / 60.0
    else:
        avg_dur = pd.DataFrame(columns=group_cols + ["avg_duration_min"])

    # Merge sonuçları; eksik değerler NaN kalır
    summary = counts.merge(avg_gain, on=group_cols, how="left").merge(avg_dur, on=group_cols, how="left")

    # Pivot: power_band x exit_reason için trade sayıları
    pivot_counts = counts.pivot(index="power_band", columns="exit_reason", values="trade_count").fillna(0)

    # TP ve SL sütunları garanti değil, get ile alıyoruz
    TP = pivot_counts.get("TP", pd.Series(0, index=pivot_counts.index))
    SL = pivot_counts.get("SL", pd.Series(0, index=pivot_counts.index))
    denom = TP + SL
    # Eğer denom == 0 ise NaN bırak
    tp_rate = (TP / denom.replace({0: pd.NA})) *
