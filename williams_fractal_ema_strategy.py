#!/usr/bin/env python3
"""
Williams Fractal + 20/50/100 EMA Scalping Strategy
====================================================

STRATEGY RULES (Based on Turkish requirements):
------------------------------------------------

📌 TIMEFRAME: 
- Primary: 1 minute (1m)
- Also works on: 3m, 5m, 15m (but fewer signals)

📌 INDICATORS:
1. Williams Fractals (Period 2)
   - Green arrow → Long signal
   - Red arrow → Short signal

2. Moving Averages (EMA):
   - MA1 = 20 (green)
   - MA2 = 50 (yellow)  
   - MA3 = 100 (red)

📌 TREND FILTER (CRITICAL):
🔵 LONG: 20 MA > 50 MA > 100 MA
   - MAs must not cross, clean uptrend required
   
🔴 SHORT: 100 MA > 50 MA > 20 MA
   - Exact opposite order required

📌 LONG ENTRY RULES:
1. Trend check: 20 > 50 > 100 (no MA crossings)
2. Price pullback:
   - Scenario A: Price dips below 20 MA
   - Scenario B: Price dips below 50 MA
   ⛔ If price goes below 100 MA → SIGNAL INVALID (trend broken)
3. Williams Fractal green arrow appears
4. Stop Loss:
   - Scenario A: 1-2 ticks below 50 MA
   - Scenario B: Below 100 MA
5. Take Profit: R:R = 1:1.5

⚠️ CRITICAL RULE: If price goes below 100 MA, long signal is INVALID
   even if green arrow appears (trend is broken)

📌 SHORT ENTRY RULES:
1. Trend check: 100 > 50 > 20 (perfect downtrend)
2. Price pullback: Price rises above 20 MA
3. Williams Fractal red arrow appears
4. Stop Loss: Slightly above 50 MA
5. Take Profit: R:R = 1:1.5

⚠️ CRITICAL: If price goes above 100 MA, trend is broken → signal invalid

📌 WHY THIS WORKS:
✔ Trend locked by MA ordering
✔ Fractal = micro reversal signal
✔ Pullback = better entry price
✔ SL based on dynamic MA levels (safe)
✔ 1.5R = aggressive but reasonable reward
✔ 1m gives many signals → but only trade with trend

Most scalpers enter on random arrow signals.
This system combines: trend direction + retracement + micro reversal.
That's why it's simple but surprisingly effective.
"""

import time
import math
import requests
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np

BINANCE_FAPI = "https://fapi.binance.com"

# ==========================
# STRATEGY SETTINGS
# ==========================
DAYS_BACK = 90                    # How many days to backtest
INITIAL_CAPITAL = 5000.0          # Starting capital (USD)
PROFIT_TARGET_USD = 1.5           # Profit target per trade (USD)
POSITION_SIZE_USD = 100.0         # Position size per trade (USD)
MAX_SYMBOLS = 20                  # None = all USDT perpetual; number = top N by volume
REQUEST_SLEEP = 0.15              # API rate limit delay
RISK_REWARD_RATIO = 1.5           # R:R ratio for take profit

# Williams Fractal settings
FRACTAL_PERIOD = 2                # Period for fractal detection

# EMA settings
EMA_FAST = 20                     # Fast EMA (green)
EMA_MEDIUM = 50                   # Medium EMA (yellow)
EMA_SLOW = 100                    # Slow EMA (red)

# Entry settings
MA_CROSS_BUFFER = 0.0001          # Buffer to detect MA crossings (0.01%)
TICK_SIZE = 0.01                  # Tick size for SL offset

# ==========================
# UTILS
# ==========================

def ms_since_epoch(dt: datetime) -> int:
    """Datetime -> ms timestamp (UTC)."""
    return int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000)


def get_futures_symbols():
    """
    Get USDT margined, PERPETUAL (perpetual) contract symbols from Binance Futures.
    Sort by volume.
    """
    url = f"{BINANCE_FAPI}/fapi/v1/exchangeInfo"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    symbols = []
    for s in data["symbols"]:
        if (
            s.get("quoteAsset") == "USDT"
            and s.get("contractType") == "PERPETUAL"
            and s.get("status") == "TRADING"
        ):
            symbols.append(s["symbol"])

    # Get 24h volume and sort by volume
    tickers_url = f"{BINANCE_FAPI}/fapi/v1/ticker/24hr"
    r2 = requests.get(tickers_url, timeout=10)
    r2.raise_for_status()
    tickers = {t["symbol"]: float(t["volume"]) for t in r2.json()}

    symbols = [s for s in symbols if s in tickers]
    symbols.sort(key=lambda s: tickers[s], reverse=True)

    if MAX_SYMBOLS is not None:
        symbols = symbols[:MAX_SYMBOLS]

    return symbols


def fetch_klines(symbol, interval, start_ms, end_ms=None, limit=1500):
    """
    Fetch klines from Binance for the specified interval and time range.
    """
    url = f"{BINANCE_FAPI}/fapi/v1/klines"
    all_rows = []

    cur_start = start_ms
    while True:
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
            "startTime": cur_start,
        }
        if end_ms is not None:
            params["endTime"] = end_ms

        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code != 200:
            print(f"[{symbol}][{interval}] HTTP {resp.status_code}: {resp.text}")
            break

        rows = resp.json()
        if not rows:
            break

        all_rows.extend(rows)

        last_open_time = rows[-1][0]
        next_start = last_open_time + 1
        if end_ms is not None and next_start > end_ms:
            break

        if len(rows) < limit:
            break

        cur_start = next_start
        time.sleep(REQUEST_SLEEP)

    if not all_rows:
        return pd.DataFrame()

    # Convert to DataFrame
    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "num_trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ]
    df = pd.DataFrame(all_rows, columns=cols)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)

    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    df.set_index("open_time", inplace=True)
    return df


def ema(series, length):
    """Calculate Exponential Moving Average"""
    return series.ewm(span=length, adjust=False).mean()


# ==========================
# WILLIAMS FRACTAL DETECTION
# ==========================

def detect_williams_fractals(df, period=2):
    """
    Detect Williams Fractals on the DataFrame.
    
    A fractal high (red arrow - short signal) is formed when:
    - The current high is the highest among (period) bars before and after
    
    A fractal low (green arrow - long signal) is formed when:
    - The current low is the lowest among (period) bars before and after
    
    Args:
        df: DataFrame with 'high' and 'low' columns
        period: Number of bars to check on each side (default 2)
    
    Returns:
        df with 'fractal_high' and 'fractal_low' columns (True/False)
    """
    highs = df['high'].values
    lows = df['low'].values
    n = len(df)
    
    fractal_high = np.zeros(n, dtype=bool)
    fractal_low = np.zeros(n, dtype=bool)
    
    for i in range(period, n - period):
        # Check for fractal high (resistance - short signal)
        is_high = True
        for j in range(i - period, i + period + 1):
            if j != i and highs[j] >= highs[i]:
                is_high = False
                break
        fractal_high[i] = is_high
        
        # Check for fractal low (support - long signal)
        is_low = True
        for j in range(i - period, i + period + 1):
            if j != i and lows[j] <= lows[i]:
                is_low = False
                break
        fractal_low[i] = is_low
    
    df['fractal_high'] = fractal_high
    df['fractal_low'] = fractal_low
    
    return df


# ==========================
# TREND ANALYSIS
# ==========================

def check_trend_valid(ma20, ma50, ma100, trend_type='long'):
    """
    Check if MAs are properly aligned for the given trend type.
    
    For LONG: 20 > 50 > 100 (with small buffer)
    For SHORT: 100 > 50 > 20
    
    Args:
        ma20, ma50, ma100: MA values at current bar
        trend_type: 'long' or 'short'
    
    Returns:
        bool: True if trend is valid
    """
    if pd.isna(ma20) or pd.isna(ma50) or pd.isna(ma100):
        return False
    
    buffer = MA_CROSS_BUFFER
    
    if trend_type == 'long':
        # Long trend: 20 > 50 > 100
        return (ma20 > ma50 * (1 + buffer)) and (ma50 > ma100 * (1 + buffer))
    else:  # short
        # Short trend: 100 > 50 > 20
        return (ma100 > ma50 * (1 + buffer)) and (ma50 > ma20 * (1 + buffer))


def check_pullback_scenario(close, low, high, ma20, ma50, ma100, trend_type='long'):
    """
    Check pullback scenario for entry.
    
    For LONG:
        - Scenario A: Price dips below 20 MA
        - Scenario B: Price dips below 50 MA
        - INVALID: Price goes below 100 MA (trend broken)
    
    For SHORT:
        - Price rises above 20 MA
        - INVALID: Price goes above 100 MA (trend broken)
    
    Args:
        close, low, high: Current bar prices
        ma20, ma50, ma100: MA values
        trend_type: 'long' or 'short'
    
    Returns:
        str: 'scenario_a', 'scenario_b', 'invalid', or None
    """
    if trend_type == 'long':
        # Check if price went below 100 MA (invalidates signal)
        if low < ma100:
            return 'invalid'
        
        # Check scenarios
        if low < ma50:
            return 'scenario_b'  # Below 50 MA
        elif low < ma20:
            return 'scenario_a'  # Below 20 MA
        
    else:  # short
        # Check if price went above 100 MA (invalidates signal)
        if high > ma100:
            return 'invalid'
        
        # Check if price rose above 20 MA
        if high > ma20:
            return 'scenario_a'
    
    return None


def calculate_stop_loss(entry_price, ma20, ma50, ma100, scenario, trend_type='long'):
    """
    Calculate stop loss based on scenario and trend type.
    
    For LONG:
        - Scenario A: 1-2 ticks below 50 MA
        - Scenario B: Below 100 MA
    
    For SHORT:
        - Slightly above 50 MA
    
    Args:
        entry_price: Entry price
        ma20, ma50, ma100: MA values
        scenario: 'scenario_a' or 'scenario_b'
        trend_type: 'long' or 'short'
    
    Returns:
        float: Stop loss price
    """
    if trend_type == 'long':
        if scenario == 'scenario_a':
            # 1-2 ticks below 50 MA
            return ma50 - (2 * TICK_SIZE)
        else:  # scenario_b
            # Below 100 MA
            return ma100 - TICK_SIZE
    else:  # short
        # Slightly above 50 MA
        return ma50 + (2 * TICK_SIZE)


# ==========================
# BACKTEST ENGINE
# ==========================

def backtest_symbol(symbol, start_dt, end_dt, interval='1m'):
    """
    Backtest Williams Fractal + EMA strategy for a single symbol.
    
    Args:
        symbol: Trading symbol
        start_dt: Start datetime
        end_dt: End datetime
        interval: Timeframe (1m, 3m, 5m, 15m)
    
    Returns:
        DataFrame with trade results or None
    """
    print(f"\n=== {symbol} - Fetching data... ===")
    start_ms = ms_since_epoch(start_dt)
    end_ms = ms_since_epoch(end_dt)

    # Fetch data
    df = fetch_klines(symbol, interval, start_ms, end_ms)
    if df.empty:
        print(f"[{symbol}] No data available, skipping.")
        return None

    if len(df) < EMA_SLOW + FRACTAL_PERIOD + 10:
        print(f"[{symbol}] Insufficient data, skipping.")
        return None

    print(f"[{symbol}] Data loaded: {len(df)} bars")

    # Calculate EMAs
    df['ema20'] = ema(df['close'], EMA_FAST)
    df['ema50'] = ema(df['close'], EMA_MEDIUM)
    df['ema100'] = ema(df['close'], EMA_SLOW)

    # Detect Williams Fractals
    df = detect_williams_fractals(df, period=FRACTAL_PERIOD)

    # Drop rows with NaN EMAs
    df = df.dropna(subset=['ema20', 'ema50', 'ema100'])
    
    if len(df) < 100:
        print(f"[{symbol}] Insufficient data after EMA calculation, skipping.")
        return None

    # Trade tracking
    trades = []
    in_position = False
    pos_side = None
    entry_price = None
    stop_price = None
    tp_price = None
    entry_time = None
    entry_scenario = None

    for i in range(FRACTAL_PERIOD + 5, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i - 1]
        
        # If in position, check for TP/SL
        if in_position:
            hit_tp = False
            hit_sl = False
            exit_price = None
            exit_reason = None
            
            if pos_side == 'long':
                if row['high'] >= tp_price:
                    hit_tp = True
                    exit_price = tp_price
                    exit_reason = 'TP'
                elif row['low'] <= stop_price:
                    hit_sl = True
                    exit_price = stop_price
                    exit_reason = 'SL'
            else:  # short
                if row['low'] <= tp_price:
                    hit_tp = True
                    exit_price = tp_price
                    exit_reason = 'TP'
                elif row['high'] >= stop_price:
                    hit_sl = True
                    exit_price = stop_price
                    exit_reason = 'SL'
            
            if hit_tp or hit_sl:
                # Calculate PnL
                if pos_side == 'long':
                    pnl_pct = ((exit_price / entry_price) - 1.0) * 100
                    pnl_usd = (exit_price - entry_price) * (POSITION_SIZE_USD / entry_price)
                else:
                    pnl_pct = ((entry_price / exit_price) - 1.0) * 100
                    pnl_usd = (entry_price - exit_price) * (POSITION_SIZE_USD / entry_price)
                
                trades.append({
                    'symbol': symbol,
                    'interval': interval,
                    'side': pos_side,
                    'entry_time': entry_time,
                    'exit_time': row.name,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'stop_loss': stop_price,
                    'take_profit': tp_price,
                    'pnl_pct': pnl_pct,
                    'pnl_usd': pnl_usd,
                    'exit_reason': exit_reason,
                    'scenario': entry_scenario,
                    'bars_held': i - df.index.get_loc(entry_time)
                })
                
                # Print trade
                profit_sign = '+' if pnl_usd >= 0 else ''
                print(f"  ├─ Trade #{len(trades)} [{pos_side.upper()}] "
                      f"Entry: {entry_time.strftime('%Y-%m-%d %H:%M')} @ ${entry_price:.4f} → "
                      f"Exit: {row.name.strftime('%Y-%m-%d %H:%M')} @ ${exit_price:.4f} | "
                      f"PnL: {profit_sign}${pnl_usd:.2f} ({profit_sign}{pnl_pct:.2f}%) | "
                      f"Reason: {exit_reason}")
                
                # Reset position
                in_position = False
                pos_side = None
                entry_price = stop_price = tp_price = None
                entry_time = None
                entry_scenario = None
        
        # If not in position, look for entry signals
        if not in_position:
            # Get current values
            close = row['close']
            high = row['high']
            low = row['low']
            ma20 = row['ema20']
            ma50 = row['ema50']
            ma100 = row['ema100']
            
            # Check for LONG signal
            if row['fractal_low']:
                # 1. Check trend validity
                if check_trend_valid(ma20, ma50, ma100, 'long'):
                    # 2. Check pullback scenario
                    scenario = check_pullback_scenario(close, low, high, ma20, ma50, ma100, 'long')
                    
                    if scenario in ['scenario_a', 'scenario_b']:
                        # Valid long signal!
                        entry = close
                        stop = calculate_stop_loss(entry, ma20, ma50, ma100, scenario, 'long')
                        risk = entry - stop
                        
                        if risk > 0:
                            tp = entry + (risk * RISK_REWARD_RATIO)
                            
                            # Enter position
                            in_position = True
                            pos_side = 'long'
                            entry_price = entry
                            stop_price = stop
                            tp_price = tp
                            entry_time = row.name
                            entry_scenario = scenario
            
            # Check for SHORT signal
            elif row['fractal_high']:
                # 1. Check trend validity
                if check_trend_valid(ma20, ma50, ma100, 'short'):
                    # 2. Check pullback scenario
                    scenario = check_pullback_scenario(close, low, high, ma20, ma50, ma100, 'short')
                    
                    if scenario == 'scenario_a':
                        # Valid short signal!
                        entry = close
                        stop = calculate_stop_loss(entry, ma20, ma50, ma100, scenario, 'short')
                        risk = stop - entry
                        
                        if risk > 0:
                            tp = entry - (risk * RISK_REWARD_RATIO)
                            
                            # Enter position
                            in_position = True
                            pos_side = 'short'
                            entry_price = entry
                            stop_price = stop
                            tp_price = tp
                            entry_time = row.name
                            entry_scenario = scenario

    if not trades:
        print(f"[{symbol}] No trades found.\n")
        return None

    # Create results DataFrame
    df_trades = pd.DataFrame(trades)
    
    # Calculate statistics
    total_pnl = df_trades['pnl_usd'].sum()
    win_rate = (df_trades['pnl_usd'] > 0).mean() * 100
    avg_pnl = df_trades['pnl_usd'].mean()
    num_trades = len(df_trades)
    
    print(f"\n{'='*70}")
    print(f"[{symbol}] SUMMARY: {num_trades} trades | Win Rate: {win_rate:.1f}% | "
          f"Total: ${total_pnl:.2f} | Avg: ${avg_pnl:.2f}")
    print(f"{'='*70}\n")

    return df_trades


def main():
    """Main backtest execution"""
    print("=" * 80)
    print("WILLIAMS FRACTAL + 20/50/100 EMA SCALPING STRATEGY BACKTEST")
    print("=" * 80)
    print(f"\nSettings:")
    print(f"  Period: {DAYS_BACK} days")
    print(f"  Timeframe: 1m (primary)")
    print(f"  Initial Capital: ${INITIAL_CAPITAL:.2f}")
    print(f"  Position Size: ${POSITION_SIZE_USD:.2f}")
    print(f"  Risk:Reward: 1:{RISK_REWARD_RATIO}")
    print(f"  Fractals: Period {FRACTAL_PERIOD}")
    print(f"  EMAs: {EMA_FAST}/{EMA_MEDIUM}/{EMA_SLOW}")
    print(f"  Max Symbols: {MAX_SYMBOLS if MAX_SYMBOLS else 'All'}")
    print()

    # Calculate date range
    end_dt = datetime.now(timezone.utc).replace(microsecond=0, second=0)
    start_dt = end_dt - timedelta(days=DAYS_BACK + 2)

    # Get symbols
    symbols = get_futures_symbols()
    print(f"Symbols to test: {len(symbols)}")
    print(f"Symbols: {symbols[:10]}{'...' if len(symbols) > 10 else ''}\n")

    all_trades = []
    
    # Test on 1m timeframe
    for sym in symbols:
        try:
            res = backtest_symbol(sym, start_dt, end_dt, interval='1m')
            if res is not None:
                all_trades.append(res)
        except Exception as e:
            print(f"[{sym}] ERROR: {e}")
            continue

    if not all_trades:
        print("No trades found across all symbols.")
        return

    # Combine all trades
    df_all = pd.concat(all_trades, ignore_index=True)
    
    # Sort by entry time
    df_all = df_all.sort_values('entry_time').reset_index(drop=True)

    # Overall statistics
    total_trades = len(df_all)
    winners = df_all[df_all['pnl_usd'] > 0]
    losers = df_all[df_all['pnl_usd'] <= 0]
    
    total_pnl = df_all['pnl_usd'].sum()
    win_rate = (len(winners) / total_trades * 100) if total_trades > 0 else 0
    avg_win = winners['pnl_usd'].mean() if len(winners) > 0 else 0
    avg_loss = losers['pnl_usd'].mean() if len(losers) > 0 else 0
    avg_pnl = df_all['pnl_usd'].mean()
    
    # Calculate final capital
    df_all['cumulative_pnl'] = df_all['pnl_usd'].cumsum()
    final_capital = INITIAL_CAPITAL + df_all['cumulative_pnl'].iloc[-1]
    total_return_pct = ((final_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100

    print("\n" + "=" * 80)
    print("OVERALL BACKTEST RESULTS")
    print("=" * 80)
    print(f"\nCapital:")
    print(f"  Starting Capital: ${INITIAL_CAPITAL:.2f}")
    print(f"  Final Capital: ${final_capital:.2f}")
    print(f"  Total Return: {total_return_pct:+.2f}%")
    
    print(f"\nTrade Statistics:")
    print(f"  Total Trades: {total_trades}")
    print(f"  Winners: {len(winners)} ({win_rate:.1f}%)")
    print(f"  Losers: {len(losers)} ({100-win_rate:.1f}%)")
    print(f"  Total PnL: ${total_pnl:.2f}")
    print(f"  Average PnL: ${avg_pnl:.2f}")
    print(f"  Average Win: ${avg_win:.2f}")
    print(f"  Average Loss: ${avg_loss:.2f}")
    
    if avg_loss != 0:
        profit_factor = abs(winners['pnl_usd'].sum() / losers['pnl_usd'].sum()) if len(losers) > 0 else float('inf')
        print(f"  Profit Factor: {profit_factor:.2f}")
    
    # By direction
    print(f"\nBy Direction:")
    for direction in ['long', 'short']:
        dir_trades = df_all[df_all['side'] == direction]
        if len(dir_trades) > 0:
            dir_wins = dir_trades[dir_trades['pnl_usd'] > 0]
            dir_win_rate = (len(dir_wins) / len(dir_trades) * 100)
            dir_pnl = dir_trades['pnl_usd'].sum()
            print(f"  {direction.upper()}: {len(dir_trades)} trades | "
                  f"Win Rate: {dir_win_rate:.1f}% | PnL: ${dir_pnl:.2f}")
    
    # By scenario
    print(f"\nBy Scenario:")
    for scenario in df_all['scenario'].unique():
        scen_trades = df_all[df_all['scenario'] == scenario]
        scen_wins = scen_trades[scen_trades['pnl_usd'] > 0]
        scen_win_rate = (len(scen_wins) / len(scen_trades) * 100)
        scen_pnl = scen_trades['pnl_usd'].sum()
        print(f"  {scenario}: {len(scen_trades)} trades | "
              f"Win Rate: {scen_win_rate:.1f}% | PnL: ${scen_pnl:.2f}")
    
    # Top performers
    print(f"\nTop 10 Performing Symbols:")
    symbol_perf = df_all.groupby('symbol').agg({
        'pnl_usd': ['count', 'sum', 'mean']
    }).round(2)
    symbol_perf.columns = ['Trades', 'Total_PnL', 'Avg_PnL']
    symbol_perf = symbol_perf.sort_values('Total_PnL', ascending=False)
    print(symbol_perf.head(10))
    
    print("\n" + "=" * 80)
    print("BACKTEST COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
