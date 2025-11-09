# -*- coding: utf-8 -*-
"""
Binance USD-M Futures Backtest (requests sürümü)
Veri kaynağı: https://fapi.binance.com/fapi/v1
Zaman aralığı: 45 gün, 30 dakikalık mumlar
Strateji: EMA(5>9>13>21) + Doji sinyali, SL yok
"""

import requests, time, pandas as pd
from datetime import datetime, timedelta, timezone

BASE_URL = "https://fapi.binance.com"
TIMEFRAME = "30m"
DAYS = 45
START_BALANCE = 1000
EMA_PERIODS = [5, 9, 13, 21]
DOJI_BODY_RATIO = 0.25
CSV_PATH = "futures_backtest_requests.csv"

# ========================== Yardımcı Fonksiyonlar ==========================

def get_futures_symbols() -> list:
    """Tüm USDT perpetual futures sembollerini getir"""
    url = f"{BASE_URL}/fapi/v1/exchangeInfo"
    r = requests.get(url, timeout=10).json()
    symbols = [s["symbol"] for s in r["symbols"]
               if s["quoteAsset"] == "USDT" and s["contractType"] == "PERPETUAL" and s["status"] == "TRADING"]
    return symbols

def get_klines(symbol: str, interval: str, days: int = 45) -> pd.DataFrame:
    """Binance Futures'tan 45 günlük 30m veri çeker (maks. 1500 mum per call)"""
    limit = 1500
    end = int(time.time() * 1000)
    start = end - days * 24 * 60 * 60 * 1000
    all_data = []
    while True:
        url = f"{BASE_URL}/fapi/v1/klines?symbol={symbol}&interval={interval}&limit={limit}&startTime={start}&endTime={end}"
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            print(f"❌ {symbol} hata: {r.text}")
            break
        data = r.json()
        if not data:
            break
        all_data += data
        first_time = int(data[0][0])
        last_time = int(data[-1][0])
        if len(data) < limit or last_time >= end:
            break
        start = last_time + 1
        time.sleep(0.2)
    if not all_data:
        return pd.DataFrame()
    df = pd.DataFrame(all_data, columns=[
        "time","open","high","low","close","volume",
        "close_time","qv","trades","tb_base","tb_quote","ignore"
    ])
    df = df[["time","open","high","low","close","volume"]].astype(float)
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    return df

def ema(series, period): return series.ewm(span=period, adjust=False).mean()

def detect_doji(df, ratio=DOJI_BODY_RATIO):
    body = (df["close"] - df["open"]).abs()
    rng = (df["high"] - df["low"]).replace(0, 1e-9)
    df["doji"] = body < rng * ratio
    return df

def ema_aligned(row):
    return all(row[f"ema{EMA_PERIODS[i]}"] > row[f"ema{EMA_PERIODS[i+1]}"]
               for i in range(len(EMA_PERIODS)-1))

# ========================== Strateji Fonksiyonu ==========================

def backtest_symbol(df: pd.DataFrame, symbol: str) -> dict | None:
    if df.empty or len(df) < 50:
        return None
    for p in EMA_PERIODS:
        df[f"ema{p}"] = ema(df["close"], p)
    detect_doji(df)

    entry_price = None
    for i in range(max(EMA_PERIODS)+3, len(df)):
        row = df.iloc[i]
        prev3 = df.iloc[i-3:i]
        if ema_aligned(row) and all(ema_aligned(prev3.iloc[k]) for k in range(3)) and prev3["doji"].any():
            entry_price = float(row["close"])
            entry_time = pd.Timestamp(row["time"])
            break
    if entry_price is None:
        return None
    final_price = float(df.iloc[-1]["close"])
    pnl_pct = (final_price - entry_price) / entry_price * 100
    final_balance = START_BALANCE * (1 + pnl_pct/100)
    return {
        "symbol": symbol,
        "entry_time": entry_time,
        "entry_price": round(entry_price, 6),
        "exit_time": pd.Timestamp(df.iloc[-1]["time"]),
        "exit_price": round(final_price, 6),
        "pnl_%": round(pnl_pct, 2),
        "final_balance_$": round(final_balance, 2)
    }

# ========================== Ana Çalışma ==========================

def main():
    print("🔍 Binance Futures USDT sembolleri alınıyor...")
    symbols = get_futures_symbols()
    print(f"📈 {len(symbols)} adet sembol bulundu.")
    results = []

    for i, sym in enumerate(symbols, 1):
        try:
            print(f"[{i}/{len(symbols)}] {sym} verisi çekiliyor...")
            df = get_klines(sym, TIMEFRAME, DAYS)
            if df.empty:
                print("   ⚠️ Veri yok.")
                continue
            res = backtest_symbol(df, sym)
            if not res:
                print("   ⚠️ Hiç sinyal oluşmadı.")
                continue
            results.append(res)
            print(f"   ✅ {sym} | PnL: {res['pnl_%']}% | Bakiye: ${res['final_balance_$']}")
        except Exception as e:
            print(f"   ❌ Hata {sym}: {e}")
            continue

    if not results:
        print("❌ Hiç işlem sonucu yok.")
        return
    df_res = pd.DataFrame(results).sort_values("final_balance_$", ascending=False)
    print("\n📊 En kârlı 15 sembol:")
    print(df_res.head(15)[["symbol","pnl_%","final_balance_$"]].to_string(index=False))
    df_res.to_csv(CSV_PATH, index=False)
    print(f"\n💾 Sonuçlar kaydedildi: {CSV_PATH}")

if __name__ == "__main__":
    main()
