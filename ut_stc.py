# ============================================================
# 📘 UT BOT + STC Backtest (TÜM Binance Futures USDT Pariteleri)
# ============================================================

import ccxt
import pandas as pd
import numpy as np
import ta
from datetime import datetime

# ------------------------------------------------------------
# Binance Futures (USDT-M) verisi çekme
# ------------------------------------------------------------
def fetch_binance(symbol="BTC/USDT", timeframe="5m", days=15):
    exchange = ccxt.binance({
        "options": {"defaultType": "future"}  # ✅ Futures verisi
    })
    limit = 1500
    all_data = []
    since = exchange.parse8601((datetime.now(datetime.UTC) - pd.Timedelta(days=days)).isoformat())

    while True:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        if not ohlcv:
            break
        df_chunk = pd.DataFrame(ohlcv, columns=["Timestamp","Open","High","Low","Close","Volume"])
        all_data.append(df_chunk)
        since = ohlcv[-1][0] + 1
        if len(ohlcv) < limit:
            break

    if not all_data:
        return pd.DataFrame()
    df = pd.concat(all_data)
    df["Date"] = pd.to_datetime(df["Timestamp"], unit="ms")
    df.set_index("Date", inplace=True)
    return df

# ------------------------------------------------------------
# UT Bot
# ------------------------------------------------------------
def ut_bot(df, key_value=2, atr_period=3):
    df = df.copy()
    atr = ta.volatility.AverageTrueRange(df["High"], df["Low"], df["Close"], atr_period).average_true_range()
    df["upperband"] = df["Close"] - (atr * key_value)
    df["lowerband"] = df["Close"] + (atr * key_value)
    buy, sell, trend = [], [], 1
    for i in range(len(df)):
        if df["Close"].iloc[i] > df["lowerband"].iloc[i]:
            signal = 1
        elif df["Close"].iloc[i] < df["upperband"].iloc[i]:
            signal = -1
        else:
            signal = trend
        buy.append(1 if signal == 1 and trend != 1 else 0)
        sell.append(1 if signal == -1 and trend != -1 else 0)
        trend = signal
    df["UT_Buy"] = buy
    df["UT_Sell"] = sell
    return df

# ------------------------------------------------------------
# STC Oscillator
# ------------------------------------------------------------
def stc(df, length=40, fast=120):
    macd = ta.trend.MACD(df["Close"], window_slow=length, window_fast=int(length/2))
    stc_line = macd.macd_diff()
    df["STC"] = ta.trend.ema_indicator(stc_line, window=int(fast/50))
    return df

# ------------------------------------------------------------
# Tek coin backtest
# ------------------------------------------------------------
def run_backtest(symbol, timeframe="5m", days=15):
    df = fetch_binance(symbol, timeframe, days)
    if df.empty or len(df) < 100:
        return None

    df = ut_bot(df, key_value=2, atr_period=3)
    df = ut_bot(df, key_value=2, atr_period=100)
    df = stc(df)

    df["buy_signal"] = df["UT_Buy"] & (df["STC"] > 0)
    df["sell_signal"] = df["UT_Sell"] & (df["STC"] < 0)

    position, trades = None, []

    for i in range(1, len(df)):
        price = df["Close"].iloc[i]
        if position is None:
            if df["buy_signal"].iloc[i]:
                entry, sl = price, df["Low"].iloc[i-1]
                tp = entry + 2 * (entry - sl)
                position = {"side": "long", "entry": entry, "sl": sl, "tp": tp}
            elif df["sell_signal"].iloc[i]:
                entry, sl = price, df["High"].iloc[i-1]
                tp = entry - 2 * (sl - entry)
                position = {"side": "short", "entry": entry, "sl": sl, "tp": tp}
        else:
            if position["side"] == "long":
                if price <= position["sl"] or price >= position["tp"]:
                    pnl = (price - position["entry"]) / position["entry"] * 100
                    trades.append(pnl)
                    position = None
            elif position["side"] == "short":
                if price >= position["sl"] or price <= position["tp"]:
                    pnl = (position["entry"] - price) / position["entry"] * 100
                    trades.append(pnl)
                    position = None

    if not trades:
        return None

    pnl_series = pd.Series(trades)
    return {
        "symbol": symbol,
        "trades": len(trades),
        "win_rate": (pnl_series > 0).mean() * 100,
        "avg_pnl": pnl_series.mean(),
        "total_pnl": pnl_series.sum()
    }

# ------------------------------------------------------------
# Ana döngü: Tüm Futures USDT coinleri
# ------------------------------------------------------------
def main():
    exchange = ccxt.binance({
        "options": {"defaultType": "future"}
    })
    markets = exchange.load_markets()
    usdt_pairs = [m for m in markets if m.endswith("/USDT") and "PERP" not in m]

    results = []
    print(f"🚀 {len(usdt_pairs)} adet USDT paritesi bulundu, test başlıyor...\n")

    for sym in usdt_pairs:
        try:
            r = run_backtest(sym)
            if r:
                results.append(r)
                print(f"✅ {r['symbol']}: {r['trades']} işlem | {r['win_rate']:.1f}% | {r['total_pnl']:.2f}%")
            else:
                print(f"⚪ {sym}: veri az veya sinyal yok")
        except Exception as e:
            print(f"❌ {sym}: hata -> {e}")

    if results:
        df = pd.DataFrame(results)
        df = df.sort_values("total_pnl", ascending=False)
        df.to_csv("ut_stc_futures_report.csv", index=False)
        print("\n📁 Sonuçlar kaydedildi: ut_stc_futures_report.csv")
        print(df.head(10))
    else:
        print("Hiç sonuç bulunamadı.")

# ------------------------------------------------------------
if __name__ == "__main__":
    main()
