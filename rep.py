#!/usr/bin/env python3
"""
fastest_power.py
Veride duration_sec yoksa hesaplar, sonra power band bazında ortalama kapanış süresini
hesaplayıp en kısa ortalama süreye sahip power band'ı gösterir.
Çalıştırma: python fastest_power.py
"""
import os
import json
from pathlib import Path
from datetime import datetime

import pandas as pd

DATA_DIR = os.getenv("DATA_DIR", "./data")
FILE_PATH = os.path.join(DATA_DIR, "sim_closed.json")

BINS = [0, 60, 70, 80, 90, 100]
LABELS = ["<60", "60-70", "70-80", "80-90", ">90"]

def load_data(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dosya bulunamadı: {path}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def compute_duration_if_missing(df):
    # Eğer duration_sec yoksa hesapla:
    if "duration_sec" not in df.columns:
        df["duration_sec"] = pd.NA

    # Öncelikle ISO zamanları kullanarak hesapla (close_time - open_time)
    if "close_time" in df.columns and "open_time" in df.columns:
        # to_datetime güvenli dönüşüm
        try:
            ot = pd.to_datetime(df["open_time"], utc=True, errors="coerce")
            ct = pd.to_datetime(df["close_time"], utc=True, errors="coerce")
            mask = df["duration_sec"].isna() & ot.notna() & ct.notna()
            df.loc[mask, "duration_sec"] = (ct[mask] - ot[mask]).dt.total_seconds()
        except Exception:
            pass

    # Eğer hala eksik ve unix ts alanları varsa (open_ts, close_ts)
    if "open_ts" in df.columns and "close_ts" in df.columns:
        mask = df["duration_sec"].isna() & df["open_ts"].notna() & df["close_ts"].notna()
        df.loc[mask, "duration_sec"] = df.loc[mask, "close_ts"] - df.loc[mask, "open_ts"]

    # Eğer sadece open_ts ve open_after_ts varsa (örnek bazen close_ts yok) -> tahmini süre:
    # (open_after_ts - open_ts) veya open_after_ts - open_ts olabilir; bu satır isteğe bağlıdır.
    if "open_ts" in df.columns and "open_after_ts" in df.columns:
        mask = df["duration_sec"].isna() & df["open_ts"].notna() & df["open_after_ts"].notna()
        # Bu iki ts'nin anlamı projeye göre farklı olabilir; burada 'open_after_ts - open_ts' kısa süre hesaplaması için ekleniyor
        df.loc[mask, "duration_sec"] = df.loc[mask, "open_after_ts"] - df.loc[mask, "open_ts"]

    # Son olarak numeric çevir ve negatif/uyumsuz değerleri NaN yap
    df["duration_sec"] = pd.to_numeric(df["duration_sec"], errors="coerce")
    df.loc[df["duration_sec"] < 0, "duration_sec"] = pd.NA

    return df

def prepare_power_band(df):
    # power sütununu numeric yap
    if "power" not in df.columns:
        df["power"] = pd.NA
    else:
        df["power"] = pd.to_numeric(df["power"], errors="coerce")
    df["power_band"] = pd.cut(df["power"], bins=BINS, labels=LABELS, include_lowest=True)
    return df

def find_fastest_power_band(df):
    # duration_sec gereklidir
    if "duration_sec" not in df.columns:
        print("duration_sec sütunu yok ve hesaplanamadı.")
        return None, None

    # Grup ortalama
    grouped = (
        df.dropna(subset=["power_band", "duration_sec"])
        .groupby("power_band")["duration_sec"]
        .agg(["count", "mean"])
        .reset_index()
    )

    # Eğer hiçbir geçerli satır yoksa
    if grouped.empty:
        return None, grouped

    # en kısa ortalama süre
    idx = grouped["mean"].idxmin()
    fastest = grouped.loc[idx]
    return fastest, grouped

def human_readable(sec):
    sec = float(sec)
    m = int(sec // 60)
    s = int(sec % 60)
    return f"{sec:.2f}s ({m}m {s}s)"

def main():
    try:
        data = load_data(FILE_PATH)
    except FileNotFoundError as e:
        print(e)
        return

    df = pd.DataFrame(data)

    df = compute_duration_if_missing(df)
    df = prepare_power_band(df)

    fastest, grouped = find_fastest_power_band(df)

    if fastest is None:
        print("Geçerli veri bulunamadı veya duration hesaplanamadı.")
        if grouped is not None and grouped.empty:
            print("Gruplama sonucu boş.")
        return

    print("En hızlı kapanan power band:")
    print(f"  Power band: {fastest['power_band']}")
    print(f"  Ortalama süre: {human_readable(fastest['mean'])}")
    print(f"  İşlem sayısı (bu band): {int(fastest['count'])}")
    print("\nTüm bandlar (adet, ort. saniye):")
    for _, row in grouped.sort_values("mean").iterrows():
        print(f"  {row['power_band']}: count={int(row['count'])}, mean={human_readable(row['mean'])}")

if __name__ == "__main__":
    main()
