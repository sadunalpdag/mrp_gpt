#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
VWAP + EMA(8) Backtest (30-day default) using CCXT OHLCV
- Direction filter: trade only long if price > VWAP; only short if price < VWAP
- Entry: VWAP retest + candle closes back in trend direction and aligns with EMA(8)
- Exit: opposite-side close of EMA(8) (dynamic trailing)
- Optional ATR filter to avoid weak retests
- Handles per-day anchored VWAP (resets each UTC day)
"""

import argparse
import math
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

import ccxt
import numpy as np
import pandas as pd


# ----------------------------- Utilities ---------------------------------- #

def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()

def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df['close'].shift(1)
    tr = pd.concat([
        (df['high'] - df['low']),
        (df['high'] - prev_close).abs(),
        (df['low'] - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr

def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    return true_range(df).rolling(length).mean()

def per_day_vwap(df: pd.DataFrame) -> pd.Series:
    """
    Anchored VWAP resets each UTC day:
    vwap = cumsum(typical_price * volume) / cumsum(volume)  (per day)
    """
    tp = (df['high'] + df['low'] + df['close']) / 3.0
    day = df.index.tz_convert("UTC").date
    grouped = pd.Series(tp.values, index=df.index).groupby(day)
    vol_grouped = df['volume'].groupby(day)

    tpv_cum = grouped.cumsum() * 0  # placeholder to align index
    vol_cum = vol_grouped.cumsum() * 0

    # We can't directly use the above; build with transform:
    tpv_cum = (tp * df['volume']).groupby(day).cumsum()
    vol_cum = df['volume'].groupby(day).cumsum()

    vwap = tpv_cum / vol_cum
    return vwap

def fetch_ohlcv_all(exchange, symbol: str, timeframe: str, since_ms: int, limit_per_fetch: int = 1500) -> pd.DataFrame:
    """
    Robust OHLCV fetcher for long ranges (looping since).
    """
    all_rows = []
    next_since = since_ms
    # Safety: stop after ~50 fetches (enough for 30 days on 5m)
    for _ in range(50):
        chunk = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=next_since, limit=limit_per_fetch)
        if not chunk:
            break
        all_rows += chunk
        if len(chunk) < limit_per_fetch:
            break
        next_since = chunk[-1][0] + 1  # advance since

    if not all_rows:
        raise RuntimeError("No OHLCV data returned. Check symbol/timeframe/exchange.")

    df = pd.DataFrame(all_rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    df = df.sort_index()
    return df

# ----------------------------- Strategy ----------------------------------- #

def generate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ema8"] = ema(out["close"], 8)
    out["atr14"] = atr(out, 14)
    out["vwap"] = per_day_vwap(out)
    return out

def vwap_retest_entry_signals(df: pd.DataFrame,
                              atr_mult: float = 0.0,
                              retest_tol: float = 0.0005) -> pd.DataFrame:
    """
    Build long/short entry booleans.
    - Long entry when:
        price above VWAP, candle tests at/near VWAP (low <= vwap*(1+tol)), and closes above VWAP,
        AND close > EMA8
        AND optional |close - vwap| >= atr_mult * ATR
    - Short entry when symmetric below.
    Only one active position at a time in backtest loop, but we mark signals here.
    """
    df = df.copy()
    vwap = df["vwap"]
    ema8 = df["ema8"]
    c = df["close"]
    hi = df["high"]
    lo = df["low"]
    atr14 = df["atr14"].fillna(0.0)

    # Long conditions
    cond_above_vwap = c > vwap
    cond_retest_long = (lo <= vwap * (1 + retest_tol)) & (c > vwap)
    cond_ema_long = c > ema8
    cond_atr_ok_long = ((c - vwap).abs() >= atr_mult * atr14)

    # Short conditions
    cond_below_vwap = c < vwap
    cond_retest_short = (hi >= vwap * (1 - retest_tol)) & (c < vwap)
    cond_ema_short = c < ema8
    cond_atr_ok_short = ((c - vwap).abs() >= atr_mult * atr14)

    df["long_signal"] = cond_above_vwap & cond_retest_long & cond_ema_long & cond_atr_ok_long
    df["short_signal"] = cond_below_vwap & cond_retest_short & cond_ema_short & cond_atr_ok_short
    return df

def run_backtest(df: pd.DataFrame,
                 fee_per_side: float = 0.0008,
                 slippage_bps: float = 1.0,
                 allow_shorts: bool = True,
                 risk_per_trade: float = 1.0) -> Tuple[pd.DataFrame, dict]:
    """
    Simple 1x notional backtest:
      - One position at a time (either flat, long, short)
      - Size = 1 notional unit (PnL reported in %). risk_per_trade left as placeholder for extensions.
      - Commission: fee_per_side per entry and per exit. Slippage: slippage_bps (basis points) each side.

    Exit rule:
      - Long: exit when close < EMA8 (candle close)
      - Short: exit when close > EMA8
    """
    df = df.copy()

    position = None  # None | "long" | "short"
    entry_px = None
    entry_time = None

    records = []

    for ts, row in df.iterrows():
        c = row["close"]
        ema8 = row["ema8"]

        # exit logic
        if position == "long" and c < ema8:
            # apply exit costs
            exit_px = c * (1 - slippage_bps / 1e4)
            gross = (exit_px / entry_px) - 1.0
            net = gross - (2 * fee_per_side)  # entry+exit fees
            records.append({"entry_time": entry_time, "exit_time": ts, "side": "LONG",
                            "entry": entry_px, "exit": exit_px, "ret_pct": net * 100})
            position, entry_px, entry_time = None, None, None

        elif position == "short" and c > ema8:
            exit_px = c * (1 + slippage_bps / 1e4)
            gross = (entry_px / exit_px) - 1.0
            net = gross - (2 * fee_per_side)
            records.append({"entry_time": entry_time, "exit_time": ts, "side": "SHORT",
                            "entry": entry_px, "exit": exit_px, "ret_pct": net * 100})
            position, entry_px, entry_time = None, None, None

        # entry logic (only if flat)
        if position is None:
            if row.get("long_signal", False):
                position = "long"
                entry_px = c * (1 + slippage_bps / 1e4)
                entry_time = ts
            elif allow_shorts and row.get("short_signal", False):
                position = "short"
                entry_px = c * (1 - slippage_bps / 1e4)
                entry_time = ts

    # If still in a trade at the end, close at last price
    if position is not None and entry_px is not None:
        last_ts = df.index[-1]
        c = df["close"].iloc[-1]
        if position == "long":
            exit_px = c * (1 - slippage_bps / 1e4)
            gross = (exit_px / entry_px) - 1.0
        else:
            exit_px = c * (1 + slippage_bps / 1e4)
            gross = (entry_px / exit_px) - 1.0
        net = gross - (2 * fee_per_side)
        records.append({"entry_time": entry_time, "exit_time": last_ts, "side": position.upper(),
                        "entry": entry_px, "exit": exit_px, "ret_pct": net * 100})

    trades = pd.DataFrame(records)
    stats = {}
    if not trades.empty:
        stats["trades"] = len(trades)
        stats["wins"] = int((trades["ret_pct"] > 0).sum())
        stats["win_rate_%"] = round(100 * stats["wins"] / stats["trades"], 2)
        stats["avg_ret_%"] = round(trades["ret_pct"].mean(), 3)
        stats["median_ret_%"] = round(trades["ret_pct"].median(), 3)
        stats["total_ret_%"] = round(trades["ret_pct"].sum(), 3)
        # simple equity curve
        trades["equity"] = (1 + trades["ret_pct"] / 100.0).cumprod()
        stats["max_equity"] = round(trades["equity"].max(), 4)
        stats["final_equity"] = round(trades["equity"].iloc[-1], 4)
        # drawdown from trade-to-trade equity
        roll_max = trades["equity"].cummax()
        dd = trades["equity"] / roll_max - 1.0
        stats["max_drawdown_%"] = round(dd.min() * 100, 2)
    else:
        stats = {"trades": 0, "wins": 0, "win_rate_%": 0.0, "avg_ret_%": 0.0,
                 "median_ret_%": 0.0, "total_ret_%": 0.0, "final_equity": 1.0, "max_drawdown_%": 0.0}

    return trades, stats

# ----------------------------- Main --------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="VWAP + EMA(8) backtest using CCXT (30-day default).")
    parser.add_argument("--exchange", type=str, default="binance", help="Exchange id in ccxt, e.g., binance, bybit")
    parser.add_argument("--symbol", type=str, default="BTC/USDT", help="Symbol, e.g., BTC/USDT")
    parser.add_argument("--timeframe", type=str, default="5m", help="OHLCV timeframe, e.g., 1m, 5m, 15m")
    parser.add_argument("--days", type=int, default=30, help="How many days back to fetch")
    parser.add_argument("--fee", type=float, default=0.0008, help="Fee per side (e.g., 0.0008 = 8 bps per side)")
    parser.add_argument("--slip_bps", type=float, default=1.0, help="Slippage in basis points (1 bps = 0.01%) per side")
    parser.add_argument("--atr_mult", type=float, default=0.0, help="Min |close - vwap| >= atr_mult * ATR to validate retest")
    parser.add_argument("--retest_tol", type=float, default=0.0005, help="VWAP proximity tolerance (fraction)")
    parser.add_argument("--no_shorts", action="store_true", help="Disable short trades")
    parser.add_argument("--csv", type=str, default="", help="Optional path to write trades CSV")
    args = parser.parse_args()

    # Build exchange
    if not hasattr(ccxt, args.exchange):
        raise ValueError(f"Exchange '{args.exchange}' not supported in ccxt.")
    ex_class = getattr(ccxt, args.exchange)
    exchange = ex_class({"enableRateLimit": True})

    # Compute since
    now = datetime.now(timezone.utc)
    since_dt = now - timedelta(days=args.days + 2)  # +2 days warmup for EMA/ATR
    since_ms = int(since_dt.timestamp() * 1000)

    # Fetch data
    df = fetch_ohlcv_all(exchange, args.symbol, args.timeframe, since_ms)
    # Restrict to last N days exactly (post indicators)
    df = df[df.index >= (now - timedelta(days=args.days + 0)).replace(tzinfo=timezone.utc)]

    # Indicators & signals
    df_i = generate_indicators(df)
    df_s = vwap_retest_entry_signals(df_i, atr_mult=args.atr_mult, retest_tol=args.retest_tol)

    # Backtest
    trades, stats = run_backtest(
        df_s,
        fee_per_side=args.fee,
        slippage_bps=args.slip_bps,
        allow_shorts=not args.no_shorts
    )

    # Print summary
    print("\n=== VWAP + EMA(8) Backtest Summary ===")
    print(f"Exchange: {args.exchange} | Symbol: {args.symbol} | TF: {args.timeframe} | Days: {args.days}")
    print(f"Trades: {stats['trades']} | Wins: {stats['wins']} | Win rate: {stats['win_rate_%']}%")
    print(f"Avg ret: {stats['avg_ret_%']}% | Median: {stats['median_ret_%']}%")
    print(f"Total ret: {stats['total_ret_%']}% | Final equity: x{stats['final_equity']}")
    print(f"Max DD: {stats['max_drawdown_%']}%")

    if not trades.empty:
        print("\nHead of trades:")
        print(trades.head(10).to_string(index=False))
        if args.csv:
            trades.to_csv(args.csv, index=False)
            print(f"\nSaved trades to: {args.csv}")

if __name__ == "__main__":
    main()
