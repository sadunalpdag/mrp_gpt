#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1-Minute Fibonacci Gold Zone Scalping Backtest (All Binance Futures USDT pairs)
--------------------------------------------------------------------------
- Strategy:
    * Detect break of structure (BoS)
    * Wait for retracement into Fibonacci 0.5–0.618 zone
    * Entry in trend direction
    * Stop = 1.0 level, TP = 1:1.5 R/R
- Backtests last 30 days of 1m OHLCV data from Binance Futures
- Outputs per-symbol performance summary (CSV)

DATA RETRIEVAL: 30 days of 1-minute candles (~43,200 bars)
"""

import ccxt, pandas as pd, numpy as np
from datetime import datetime, timedelta, timezone

# ---------------- CONFIG ---------------- #
exchange = ccxt.binance({
    "options": {"defaultType": "future"},
    "enableRateLimit": True
})

symbols = [
    "BTC/USDT","ETH/USDT","BNB/USDT","SOL/USDT","AVAX/USDT","ADA/USDT",
    "LINK/USDT","DOT/USDT","XRP/USDT","DOGE/USDT","LTC/USDT","MATIC/USDT",
    "APT/USDT","INJ/USDT","RNDR/USDT","ARB/USDT","NEAR/USDT","OP/USDT",
    "SUI/USDT","FIL/USDT"
]
timeframe = "1m"
# Data retrieval period: 30 days of 1-minute candles (~43,200 bars)
# This is sufficient for Fibonacci scalping backtest
days = 30
rr_ratio = 1.5
since = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp()*1000)

# ---------------- FUNCTIONS ---------------- #
def fetch_data(sym):
    try:
        ohlc = exchange.fetch_ohlcv(sym, timeframe, since=since, limit=1500)
        df = pd.DataFrame(ohlc, columns=["ts","open","high","low","close","vol"])
        df["time"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        df.set_index("time", inplace=True)
        df.sort_index(inplace=True)
        return df
    except Exception as e:
        print(f"⚠️ {sym} fetch error: {e}")
        return None

def fib_levels(high, low):
    diff = high - low
    return {"0.5": low + diff*0.5, "0.618": low + diff*0.618, "1.0": high}

def detect_break(df):
    highs = df["high"].rolling(10).max().shift(1)
    lows = df["low"].rolling(10).min().shift(1)
    bos_up = df["close"] > highs
    bos_down = df["close"] < lows
    return bos_up, bos_down

def backtest(df):
    trades = []
    bos_up, bos_down = detect_break(df)
    state = "neutral"
    anchor_high = anchor_low = None
    waiting = True

    for i in range(10, len(df)):
        row = df.iloc[i]
        c, h, l = row["close"], row["high"], row["low"]

        if state == "neutral":
            if bos_up.iloc[i]:
                state = "uptrend"
                anchor_low = df["low"].iloc[i-5:i].min()
                anchor_high = df["high"].iloc[i]
                fib = fib_levels(anchor_high, anchor_low)
                waiting = True
            elif bos_down.iloc[i]:
                state = "downtrend"
                anchor_high = df["high"].iloc[i-5:i].max()
                anchor_low = df["low"].iloc[i]
                fib = fib_levels(anchor_high, anchor_low)
                waiting = True
            continue

        if state == "uptrend":
            if waiting and fib["0.5"] <= l <= fib["0.618"]:
                entry = c
                stop = fib["1.0"]
                target = entry + (entry - stop)*rr_ratio
                waiting = False
                pos = "long"
            elif not waiting:
                if l <= stop:
                    trades.append(("LONG", entry, stop, stop-entry))
                    waiting = True; state = "neutral"
                elif h >= target:
                    trades.append(("LONG", entry, target, target-entry))
                    waiting = True; state = "neutral"

        if state == "downtrend":
            if waiting and fib["0.5"] >= h >= fib["0.618"]:
                entry = c
                stop = fib["1.0"]
                target = entry - (stop - entry)*rr_ratio
                waiting = False
                pos = "short"
            elif not waiting:
                if h >= stop:
                    trades.append(("SHORT", entry, stop, entry-stop))
                    waiting = True; state = "neutral"
                elif l <= target:
                    trades.append(("SHORT", entry, target, entry-target))
                    waiting = True; state = "neutral"
    return trades

# ---------------- MAIN LOOP ---------------- #
results = []
for sym in symbols:
    print(f"\n📊 Backtesting {sym}...")
    df = fetch_data(sym)
    if df is None or df.empty:
        continue
    trades = backtest(df)
    if not trades:
        results.append([sym, 0, 0, 0, 0])
        continue
    df_t = pd.DataFrame(trades, columns=["Side","Entry","Exit","PnL"])
    df_t["PnL_%"] = df_t["PnL"]/df_t["Entry"]*100
    total = len(df_t)
    win = (df_t["PnL_%"]>0).sum()
    winrate = round(win/total*100,2)
    avgpnl = round(df_t["PnL_%"].mean(),3)
    totalpnl = round(df_t["PnL_%"].sum(),3)
    results.append([sym,total,winrate,avgpnl,totalpnl])
    print(f"✅ {sym}: {total} trades | Win% {winrate} | Avg {avgpnl}% | Total {totalpnl}%")

# ---------------- RESULTS ---------------- #
df_res = pd.DataFrame(results, columns=["Symbol","Trades","WinRate%","AvgPnL%","TotalPnL%"])
df_res.sort_values("TotalPnL%", ascending=False, inplace=True)
df_res.to_csv("fib_scalp_results.csv", index=False)
print("\n===== SUMMARY =====")
print(df_res)
print("\nSaved to fib_scalp_results.csv ✅")
