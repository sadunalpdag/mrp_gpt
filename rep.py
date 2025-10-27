#!/usr/bin/env python3
"""
fastest_power.py
En hızlı (ortalama süre olarak en kısa) kapanan power band'ı gösterir.
Çalıştırma: python fastest_power.py
"""
import os
import json
from pathlib import Path

import pandas as pd

DATA_DIR = os.getenv("DATA_DIR", "./data")
FILE_PATH = os.path.join(DATA_DIR, "sim_closed.json")

def load_data(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dosya bulunamadı: {path}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def main():
    try:
        data = load_data(FILE_PATH)
    except FileNotFoundError as e:
        print(e)
        return

    df = pd.DataFrame(data)

    # Gerekli sütunları güvenli al
    if "power" not in df.columns:
        print("Veride 'power' sütunu bulunmuyor; power band analizi yapılamaz.")
        return
    if "duration_sec" not in df.columns:
        print("Veride 'duration_sec' sütunu yok; süre analizi yapılamaz.")
        return
    if "exit_reason" not in df.columns:
        # exit_reason olmadan da ortalama süre hesaplanabilir; uyarı veriyoruz ancak devam edebiliriz
        print("Uyarı: 'exit_reason' sütunu yok; sadece power band bazlı ortalama süre hesaplanacak.")

    # numeric dönüşümleri
    df["power"] = pd.to_numeric(df["power"], errors="coerce")
    df["duration_sec"] = pd.to_numeric(df["duration_sec"], errors="coerce")

    # power band'leri oluşturalım (orijinal script ile uyumlu)
    bins = [0, 60, 70, 80, 90, 100]
    labels = ["<60", "60-70", "70-80", "80-90", ">90"]
    df["power_band"] = pd.cut(df["power"], bins=bins, labels=labels, include_lowest=True)

    # groupby ortalama süresi (saniye)
    grouped = df.groupby("power_band")["duration_sec"].agg(["count", "mean"]).reset_index()
    # Boş veya NaN olan band'leri çıkar
    grouped = grouped[grouped["count"] > 0].copy()
    if grouped.empty:
        print("Geçerli (duration_sec içeren) hiç işlem yok.")
        return

    # en küçük ortalama süreyi bul
    idx = grouped["mean"].idxmin()
    fastest = grouped.loc[idx]

    # Gösterim: dakika:saniye ve adet
    mean_sec = float(fastest["mean"])
    minutes = int(mean_sec // 60)
    seconds = int(mean_sec % 60)
    pb = fastest["power_band"]

    print("En hızlı kapanan power band:")
    print(f"  Power band: {pb}")
    print(f"  Ortalama süre: {mean_sec:.2f} saniye ({minutes}m {seconds}s)")
    print(f"  İşlem sayısı (bu band): {int(fastest['count'])}")
    print("\nTüm bandlar (adet, ortalama saniye):")
    for _, row in grouped.sort_values("mean").iterrows():
        m = float(row["mean"])
        mm = int(m // 60)
        ss = int(m % 60)
        print(f"  {row['power_band']}: count={int(row['count'])}, mean={m:.2f}s ({mm}m{ss}s)")

if __name__ == "__main__":
    main()
