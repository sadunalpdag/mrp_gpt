#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TG Capital - London Kill Zone Strategy Backtest
-----------------------------------------------
Binance Futures USDT pairs
30-min candles, last 30 days
Strategy:
 - 3:00–6:30 NY time (08:00–11:30 UTC)
 - EMA alignment (5>9>13>21), price > EMA200
 - Fair Value Gap (3-bar)
 - Doji candle in gap
 - Next candle closes below Doji high
 - Entry on Doji close, SL=Doji low, TP=1:20
"""

import ccxt, pandas as pd, numpy as np
from datetime import datetime, timedelta, timezone
import pytz

# ---------------- CONFIG ---------------- #
exchange = ccxt.binance({
    "options": {"defaultType": "future"},
    "enableRateLimit": True
})
symbols = [
    "BTC/USDT","ETH/USDT","BNB/USDT","SOL/USDT","AVAX/USDT","ADA/USDT",
    "XRP/USDT","DOGE/USDT","LTC/USDT","DOT/USDT","LINK/USDT",
    "MATIC/USDT","APT/USDT","ARB/USDT","OP/USDT","NEAR/USDT",
    "SUI/USDT","INJ/USDT","RNDR/USDT","FIL/USDT"
]
timeframe = "30m"
days = 30
rr_ratio = 20.0
ny_tz = pytz.timezone("America/New_York")

# ---------------- HELPERS ---------------- #
def ema(series, n):
    return series.ewm(span=n, adjust=False).mean()

def is_doji(row):
    body = abs(row["close"] - row["open"])
    range_ = row["high"] - row["low"]
    return body <= range_ * 0.25

def fetch_data(sym):
    try:
        since = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)
        ohlc = exchange.fetch_ohlcv(sym, timeframe, since=since, limit=1500)
        df = pd.DataFrame(ohlc, columns=["ts","open","high","low","close","vol"])
        df["time"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        df.set_index("time", inplace=True)
        df = df.astype(float)
        return df
    except Exception as e:
        print(f"⚠️ Fetch error {sym}: {e}")
        return None

# ---------------- STRATEGY ---------------- #
def backtest(df):
    df["ema5"] = ema(df["close"], 5)
    df["ema9"] = ema(df["close"], 9)
    df["ema13"] = ema(df["close"], 13)
    df["ema21"] = ema(df["close"], 21)
    df["ema200"] = ema(df["close"], 200)

    df["ny_time"] = df.index.tz_convert(ny_tz)
    mask = (df["ny_time"].dt.hour >= 3) & (df["ny_time"].dt.hour <= 6)
    df = df.loc[mask]

    trades = []

    for i in range(3, len(df)-1):
        c = df.iloc[i]
        prev2 = df.iloc[i-2]
        prev1 = df.iloc[i-1]
        nxt = df.iloc[i+1]

        # Trend alignment
        if not (c["ema5"] > c["ema9"] > c["ema13"] > c["ema21"] and c["close"] > c["ema200"]):
            continue

        # Fair Value Gap (bullish)
        if c["low"] > prev2["high"]:
            # Doji check (Trident)
            if is_doji(prev1):
                # Validation: next candle closes below doji high
                if nxt["close"] < prev1["high"]:
                    entry = prev1["close"]
                    stop = prev1["low"]
                    target = entry + (entry - stop) * rr_ratio
                    trades.append((df.index[i], entry, stop, target))

    results = []
    for t in trades:
        ts, entry, stop, target = t
        after = df.loc[df.index > ts]
        hit_tp = hit_sl = None
        for j, r in after.iterrows():
            if r["low"] <= stop:
                hit_sl = j
                break
            if r["high"] >= target:
                hit_tp = j
                break
        if hit_tp:
            pnl = target - entry
        elif hit_sl:
            pnl = stop - entry
        else:
            pnl = 0
        results.append(pnl / entry * 100)
    return results

# ---------------- MAIN LOOP ---------------- #
summary = []
for sym in symbols:
    print(f"\n📊 Testing {sym} ...")
    df = fetch_data(sym)
    if df is None or df.empty:
        continue
    pnl_list = backtest(df)
    if not pnl_list:
        summary.append([sym,0,0,0])
        continue
    pnl_arr = np.array(pnl_list)
    total = len(pnl_arr)
    winrate = np.sum(pnl_arr > 0) / total * 100
    avgpnl = pnl_arr.mean()
    totalpnl = pnl_arr.sum()
    summary.append([sym,total,round(winrate,2),round(avgpnl,3),round(totalpnl,3)])
    print(f"✅ {sym}: {total} trades | Win% {winrate:.2f} | Avg {avgpnl:.3f}% | Total {totalpnl:.3f}%")

# ---------------- SAVE ---------------- #
df_res = pd.DataFrame(summary, columns=["Symbol","Trades","WinRate%","AvgPnL%","TotalPnL%"])
df_res.sort_values("TotalPnL%", ascending=False, inplace=True)
df_res.to_csv("tgcapital_london_results.csv", index=False)
print("\n===== SUMMARY =====")
print(df_res)
print("\nSaved to tgcapital_london_results.csv ✅")
