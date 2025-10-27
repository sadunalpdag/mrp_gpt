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
    tp_rate = (TP / denom.replace({0: pd.NA})) * 100

    pivot = pivot_counts.reset_index()
    pivot["TP_Rate(%)"] = tp_rate.values
    # Toplam tüm exit_reason sütunlarını toplayarak Total_Trades hesapla
    # (power_band kolonu hariç)
    pivot["Total_Trades"] = pivot.drop(columns=["power_band"]).sum(axis=1)

    return summary, pivot

def pretty_print(summary, pivot):
    print("==== POWER BANDS PERFORMANCE ====")
    if summary.empty:
        print("(Özet boş - veri yok veya uygun sütunlar eksik.)")
    else:
        print(summary.to_string(index=False, float_format="%.3f"))
    print("\n==== TP RATE BY POWER ====")
    if pivot.empty:
        print("(Pivot boş - veri yok veya uygun sütunlar eksik.)")
    else:
        print(pivot.to_string(index=False, float_format="%.3f"))

def plot_results(df, pivot, out_dir="outputs"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    sns.set(style="whitegrid")

    if "TP_Rate(%)" in pivot.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(data=pivot, x="power_band", y="TP_Rate(%)", palette="viridis", ax=ax)
        ax.set_title("TP Rate (%) by Power Band")
        ax.set_ylabel("TP Rate (%)")
        plt.tight_layout()
        plt.savefig(Path(out_dir) / "tp_rate_by_power.png", dpi=150)
        plt.close(fig)

    if "gain_pct" in df.columns and not df["gain_pct"].dropna().empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.boxplot(data=df, x="power_band", y="gain_pct", palette="pastel", ax=ax)
        ax.set_title("Gain % distribution by Power Band")
        ax.set_ylabel("Gain %")
        plt.tight_layout()
        plt.savefig(Path(out_dir) / "gainpct_box_by_power.png", dpi=150)
        plt.close(fig)

    counts = df.groupby(["power_band", "exit_reason"]).size().unstack(fill_value=0)
    if not counts.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.heatmap(counts, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title("Trade counts (power band x exit reason)")
        plt.tight_layout()
        plt.savefig(Path(out_dir) / "counts_heatmap.png", dpi=150)
        plt.close(fig)

def main():
    try:
        raw = load_json_safe(FILE_PATH)
    except FileNotFoundError as e:
        print(e)
        return

    df = prepare_dataframe(raw)
    summary, pivot = compute_summary(df)
    pretty_print(summary, pivot)

    out_dir = "outputs"
    Path(out_dir).mkdir(exist_ok=True)
    summary.to_csv(Path(out_dir) / "summary_by_power_and_exit.csv", index=False)
    pivot.to_csv(Path(out_dir) / "pivot_tp_rate_by_power.csv", index=False)

    plot_results(df, pivot, out_dir=out_dir)
    print(f"\nÖzet ve grafikler '{out_dir}' dizinine kaydedildi.")

if __name__ == "__main__":
    main()
