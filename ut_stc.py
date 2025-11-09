#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Opening Range Break & Retest Strategy (New York 09:30 EST)
- Runs across top Binance Futures USDT pairs
- Uses CCXT to fetch 1m OHLCV for last 30 days
- Calculates first 5m range from 09:30–09:35 NY time
- Looks for breakout + retest entries
- Target: 1:2 Risk/Reward
- Exports CSV summary per symbol
"""

import ccxt, pandas as pd, pytz
from datetime import datetime, timedelta, timezone
import numpy as np

# ---------------- SETTINGS ---------------- #
exchange = ccxt.binance({
    "options": {"defaultType": "future"},
    "enableRateLimit": True
})
symbols = [
    "BTC/USDT","ETH/USDT","BNB/USDT","SOL/USDT","AVAX/USDT","ADA/USDT","LINK/USDT",
    "DOT/USDT","XRP/USDT","DOGE/USDT","LTC/USDT","MATIC/USDT","APT/USDT","INJ/USDT",
    "RNDR/USDT","ARB/USDT","NEAR/USDT","OP/USDT","SUI/USDT","FIL/USDT"
]
timeframe = "1m"
days = 30
rr_ratio = 2.0  # risk:reward
session_hour = 9
session_minute = 30
ny_tz = pytz.timezone("America/New_York")
utc = pytz.UTC

# ---------------- STRATEGY ---------------- #
def get_opening_range(df):
    """Return 5-minute opening range (09:30–09:35 NYT)"""
    df = df.copy()
    df["ny_time"] = df.index.tz_convert(ny_tz)
    range_mask = (df["ny_time"].dt.hour == session_hour) & (df["ny_time"].dt.minute.between(session_minute, session_minute+4))
    rng = df.loc[range_mask]
    if rng.empty:
        return None, None
    return rng["high"].max(), rng["low"].min()

def run_strategy(df):
    """Apply breakout + retest logic per day"""
    trades = []
    grouped = df.groupby(df.index.date)
    for date, data in grouped:
        H0, L0 = get_opening_range(data)
        if not H0 or not L0:
            continue
        post_open = data.loc[data.index > data.index[0] + timedelta(minutes=10)]
        broke_high = broke_low = False
        entry = stop = target = None
        pos = None
        for t, r in post_open.iterrows():
            c = r["close"]
            if not broke_high and c > H0:
                broke_high = True
            if not broke_low and c < L0:
                broke_low = True

            if broke_high and (r["low"] <= H0) and (r["close"] > H0) and entry is None:
                entry, stop = r["close"], H0 - (H0 - L0)*0.2
                target = entry + (entry - stop)*rr_ratio
                pos = "long"
            elif broke_low and (r["high"] >= L0) and (r["close"] < L0) and entry is None:
                entry, stop = r["close"], L0 + (H0 - L0)*0.2
                target = entry - (stop - entry)*rr_ratio
                pos = "short"

            if pos == "long" and entry:
                if r["low"] <= stop:
                    trades.append((date,"LONG",entry,stop,stop-entry))
                    break
                elif r["high"] >= target:
                    trades.append((date,"LONG",entry,target,target-entry))
                    break
            elif pos == "short" and entry:
                if r["high"] >= stop:
                    trades.append((date,"SHORT",entry,stop,entry-stop))
                    break
                elif r["low"] <= target:
                    trades.append((date,"SHORT",entry,target,entry-target))
                    break
    return trades

# ---------------- MAIN LOOP ---------------- #
results = []
since = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp()*1000)

for sym in symbols:
    try:
        print(f"\nFetching {sym}...")
        ohlc = exchange.fetch_ohlcv(sym, timeframe, since=since, limit=1500)
        if not ohlc:
            print("No data.")
            continue
        df = pd.DataFrame(ohlc, columns=["ts","open","high","low","close","vol"])
        df["time"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        df.set_index("time", inplace=True)
        trades = run_strategy(df)
        if not trades:
            results.append([sym,0,0,0,0])
            continue
        df_t = pd.DataFrame(trades, columns=["Date","Side","Entry","Exit","PnL"])
        df_t["PnL_%"] = df_t["PnL"]/df_t["Entry"]*100
        total = len(df_t)
        win = (df_t["PnL_%"]>0).sum()
        winrate = round(win/total*100,2)
        avgpnl = round(df_t["PnL_%"].mean(),3)
        totalpnl = round(df_t["PnL_%"].sum(),3)
        results.append([sym,total,winrate,avgpnl,totalpnl])
        print(f"✅ {sym}: {total} trades | Win% {winrate} | Avg {avgpnl}% | Total {totalpnl}%")
    except Exception as e:
        print(f"⚠️ {sym} error: {e}")
        continue

# ---------------- SAVE SUMMARY ---------------- #
df_res = pd.DataFrame(results, columns=["Symbol","Trades","WinRate%","AvgPnL%","TotalPnL%"])
df_res.sort_values("TotalPnL%", ascending=False, inplace=True)
df_res.to_csv("results_futures.csv", index=False)
print("\n===== SUMMARY =====")
print(df_res)
print("\nSaved to results_futures.csv ✅")
