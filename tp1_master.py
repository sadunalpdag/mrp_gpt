#!/usr/bin/env python3
"""
TP1 SCALP MASTER STRATEGY
Ultra Precision Scalp System combining 3 strategies:
1. MRPZ (Mean Reversion Price Zone)
2. Stochastic Hybrid (Momentum)
3. Price Action (Double Bottom/Top in Structure Zones)

Entry Rules:
- Trend Filter: Stoch RSI white line (>50 long, <50 short)
- Mean Reversion: MRPZ spike/zone + Stoch K oversold/overbought
- Momentum: Stoch K/D cross
- Price Action: 1H double bottom/top in 4H structure zone

Exit: TP1 only at 1.2R-1.4R (optimized for 68-78% win rate)
"""

import time
import math
import requests
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np

BINANCE_FAPI = "https://fapi.binance.com"

# ==========================
# CONFIGURATION
# ==========================
DAYS_BACK = 90                      # Historical data period
INITIAL_CAPITAL = 5000.0            # Starting capital (USD)
PROFIT_TARGET = 1.5                 # Target profit per trade (USD)
SESSION_PROFIT_TARGET = 20.0        # Session profit target (USD)
USE_STOP_LOSS = True                # Enable stop loss
MAX_SYMBOLS = None                  # None = all USDT perpetual
REQUEST_SLEEP = 0.15                # API rate limit delay

# Strategy Parameters
TP1_R_MIN = 1.2                     # Minimum R for TP1
TP1_R_MAX = 1.4                     # Maximum R for TP1
BODY_THRESH = 0.6                   # Strong candle body ratio

# Stochastic RSI Trend Filter (3,3,14,134)
STOCH_RSI_K = 3
STOCH_RSI_D = 3
STOCH_RSI_RSI_LEN = 14
STOCH_RSI_STOCH_LEN = 134

# Stochastic Momentum (7,3,3)
STOCH_K_LEN = 7
STOCH_K_SMOOTH = 3
STOCH_D_SMOOTH = 3
STOCH_OVERSOLD = 20
STOCH_OVERBOUGHT = 80

# MRPZ Parameters
MRPZ_LENGTH = 34
MRPZ_MULT = 2.0

# Price Action Parameters
DOUBLE_PATTERN_TOLERANCE = 0.002   # 0.2% tolerance for double bottom/top
STRUCTURE_LOOKBACK = 20            # Bars to look back for structure zones

# ==========================
# UTILITY FUNCTIONS
# ==========================

def ms_since_epoch(dt: datetime) -> int:
    """Convert datetime to milliseconds timestamp (UTC)."""
    return int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000)


def get_futures_symbols():
    """
    Fetch USDT margined PERPETUAL contracts from Binance Futures.
    Sort by 24h volume for better liquidity.
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

    # Sort by 24h volume
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
    Fetch klines from Binance for specified interval and time range.
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


# ==========================
# INDICATOR FUNCTIONS
# ==========================

def ema(series, length):
    """Exponential Moving Average"""
    return series.ewm(span=length, adjust=False).mean()


def calculate_rsi(series, period=14):
    """Calculate RSI indicator"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_stochastic(high, low, close, k_period=14, k_smooth=3, d_smooth=3):
    """Calculate Stochastic Oscillator"""
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    
    k_raw = 100 * (close - lowest_low) / (highest_high - lowest_low)
    k = k_raw.rolling(window=k_smooth).mean()
    d = k.rolling(window=d_smooth).mean()
    
    return k, d


def calculate_stoch_rsi(close, rsi_len=14, stoch_len=14, k_smooth=3, d_smooth=3):
    """
    Calculate Stochastic RSI
    Returns K (blue line) and D (white line for trend filter)
    """
    rsi = calculate_rsi(close, period=rsi_len)
    
    lowest_rsi = rsi.rolling(window=stoch_len).min()
    highest_rsi = rsi.rolling(window=stoch_len).max()
    
    stoch_rsi_raw = 100 * (rsi - lowest_rsi) / (highest_rsi - lowest_rsi)
    stoch_rsi_k = stoch_rsi_raw.rolling(window=k_smooth).mean()
    stoch_rsi_d = stoch_rsi_k.rolling(window=d_smooth).mean()
    
    return stoch_rsi_k, stoch_rsi_d


def calculate_mrpz(df, length=34, mult=2.0):
    """
    Calculate Mean Reversion Price Zone (MRPZ)
    Returns upper zone, lower zone, and histogram spike signals
    """
    # Calculate basis (SMA or EMA)
    basis = df['close'].rolling(window=length).mean()
    
    # Calculate standard deviation for zones
    std_dev = df['close'].rolling(window=length).std()
    
    upper_zone = basis + (mult * std_dev)
    lower_zone = basis - (mult * std_dev)
    
    # Calculate histogram (price distance from basis)
    histogram = df['close'] - basis
    
    # Detect spikes (when price moves significantly from mean)
    histogram_ma = histogram.rolling(window=3).mean()
    upper_spike = histogram > histogram_ma.shift(1) * 1.5
    lower_spike = histogram < histogram_ma.shift(1) * 1.5
    
    # Price in zones
    in_upper_zone = df['close'] > upper_zone
    in_lower_zone = df['close'] < lower_zone
    
    return {
        'upper_zone': upper_zone,
        'lower_zone': lower_zone,
        'histogram': histogram,
        'upper_spike': upper_spike,
        'lower_spike': lower_spike,
        'in_upper_zone': in_upper_zone,
        'in_lower_zone': in_lower_zone
    }


def detect_double_bottom(df, lookback=20, tolerance=0.002):
    """
    Detect double bottom pattern (for LONG)
    Returns True if double bottom found in recent bars
    """
    if len(df) < lookback:
        return False, None
    
    recent_df = df.iloc[-lookback:]
    lows = recent_df['low'].values
    
    # Find local minima
    local_mins = []
    for i in range(1, len(lows) - 1):
        if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
            local_mins.append((i, lows[i]))
    
    if len(local_mins) < 2:
        return False, None
    
    # Check last two minima for double bottom
    for i in range(len(local_mins) - 1):
        low1_idx, low1_val = local_mins[i]
        low2_idx, low2_val = local_mins[i + 1]
        
        # Check if lows are approximately equal (within tolerance)
        if abs(low1_val - low2_val) / low1_val <= tolerance:
            # Found double bottom
            support_level = (low1_val + low2_val) / 2
            return True, support_level
    
    return False, None


def detect_double_top(df, lookback=20, tolerance=0.002):
    """
    Detect double top pattern (for SHORT)
    Returns True if double top found in recent bars
    """
    if len(df) < lookback:
        return False, None
    
    recent_df = df.iloc[-lookback:]
    highs = recent_df['high'].values
    
    # Find local maxima
    local_maxs = []
    for i in range(1, len(highs) - 1):
        if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
            local_maxs.append((i, highs[i]))
    
    if len(local_maxs) < 2:
        return False, None
    
    # Check last two maxima for double top
    for i in range(len(local_maxs) - 1):
        high1_idx, high1_val = local_maxs[i]
        high2_idx, high2_val = local_maxs[i + 1]
        
        # Check if highs are approximately equal (within tolerance)
        if abs(high1_val - high2_val) / high1_val <= tolerance:
            # Found double top
            resistance_level = (high1_val + high2_val) / 2
            return True, resistance_level
    
    return False, None


def is_in_structure_zone(price, df_4h, lookback=20):
    """
    Check if price is in a 4H structure support/resistance zone
    (OTZ - Optimal Trading Zone)
    """
    if len(df_4h) < lookback:
        return False, None
    
    recent_4h = df_4h.iloc[-lookback:]
    
    # Find support/resistance levels (levels tested multiple times)
    levels = []
    
    # Check highs and lows
    for i in range(len(recent_4h)):
        high = recent_4h.iloc[i]['high']
        low = recent_4h.iloc[i]['low']
        
        # Count how many times this level was tested
        high_tests = sum(abs(recent_4h['high'] - high) / high < 0.005)
        low_tests = sum(abs(recent_4h['low'] - low) / low < 0.005)
        
        if high_tests >= 2:
            levels.append(('resistance', high, high_tests))
        if low_tests >= 2:
            levels.append(('support', low, low_tests))
    
    # Check if current price is near any significant level
    for level_type, level_price, tests in levels:
        if abs(price - level_price) / level_price < 0.01:  # Within 1%
            return True, level_type
    
    return False, None


# ==========================
# BACKTEST STRATEGY
# ==========================

def backtest_symbol(symbol, start_dt, end_dt):
    """
    Backtest TP1 SCALP MASTER STRATEGY for a single symbol.
    Combines MRPZ, Stochastic, and Price Action filters.
    """
    print(f"\n=== {symbol} - TP1 SCALP MASTER STRATEGY ===")
    start_ms = ms_since_epoch(start_dt)
    end_ms = ms_since_epoch(end_dt)

    # Fetch data for different timeframes
    print(f"[{symbol}] Fetching 5m data...")
    df_5m = fetch_klines(symbol, "5m", start_ms, end_ms)
    if df_5m.empty:
        print(f"[{symbol}] No 5m data, skipping.")
        return None

    print(f"[{symbol}] Fetching 1h data...")
    df_1h = fetch_klines(symbol, "1h", start_ms, end_ms)
    if df_1h.empty:
        print(f"[{symbol}] No 1h data, skipping.")
        return None

    print(f"[{symbol}] Fetching 4h data...")
    df_4h = fetch_klines(symbol, "4h", start_ms, end_ms)
    if df_4h.empty:
        print(f"[{symbol}] No 4h data, skipping.")
        return None

    # Ensure UTC timezone
    df_5m.index = df_5m.index.tz_convert("UTC")
    df_1h.index = df_1h.index.tz_convert("UTC")
    df_4h.index = df_4h.index.tz_convert("UTC")

    print(f"[{symbol}] Calculating indicators...")
    
    # Calculate indicators on 1H timeframe (for execution)
    # Stoch RSI Trend Filter (3,3,14,134)
    df_1h['stoch_rsi_k'], df_1h['stoch_rsi_d'] = calculate_stoch_rsi(
        df_1h['close'], 
        rsi_len=STOCH_RSI_RSI_LEN,
        stoch_len=STOCH_RSI_STOCH_LEN,
        k_smooth=STOCH_RSI_K,
        d_smooth=STOCH_RSI_D
    )
    
    # Stochastic Momentum (7,3,3)
    df_1h['stoch_k'], df_1h['stoch_d'] = calculate_stochastic(
        df_1h['high'],
        df_1h['low'],
        df_1h['close'],
        k_period=STOCH_K_LEN,
        k_smooth=STOCH_K_SMOOTH,
        d_smooth=STOCH_D_SMOOTH
    )
    
    # MRPZ
    mrpz = calculate_mrpz(df_1h, length=MRPZ_LENGTH, mult=MRPZ_MULT)
    df_1h['mrpz_upper'] = mrpz['upper_zone']
    df_1h['mrpz_lower'] = mrpz['lower_zone']
    df_1h['mrpz_histogram'] = mrpz['histogram']
    df_1h['mrpz_upper_spike'] = mrpz['upper_spike']
    df_1h['mrpz_lower_spike'] = mrpz['lower_spike']
    df_1h['mrpz_in_upper'] = mrpz['in_upper_zone']
    df_1h['mrpz_in_lower'] = mrpz['in_lower_zone']
    
    # Candle properties
    df_1h['body'] = (df_1h['close'] - df_1h['open']).abs()
    df_1h['range'] = df_1h['high'] - df_1h['low']
    df_1h['strong'] = (df_1h['range'] > 0) & (df_1h['body'] / df_1h['range'] >= BODY_THRESH)

    # Trade tracking
    trades = []
    in_position = False
    pos_side = None
    entry_price = None
    stop_price = None
    tp_price = None
    position_size = None
    entry_time = None
    
    session_profit = 0.0
    session_count = 0

    print(f"[{symbol}] Running strategy simulation...")
    
    # Iterate through 1H bars for signals
    for i in range(max(STOCH_RSI_STOCH_LEN, MRPZ_LENGTH, STRUCTURE_LOOKBACK) + 10, len(df_1h)):
        current_time = df_1h.index[i]
        
        # Check if in position - manage exits
        if in_position:
            high_i = df_1h.iloc[i]['high']
            low_i = df_1h.iloc[i]['low']
            
            exit_reason = None
            exit_price = None
            
            if pos_side == "long":
                if high_i >= tp_price:
                    exit_price = tp_price
                    exit_reason = "TP1"
                elif USE_STOP_LOSS and low_i <= stop_price:
                    exit_price = stop_price
                    exit_reason = "SL"
            else:  # short
                if low_i <= tp_price:
                    exit_price = tp_price
                    exit_reason = "TP1"
                elif USE_STOP_LOSS and high_i >= stop_price:
                    exit_price = stop_price
                    exit_reason = "SL"
            
            if exit_reason is not None:
                # Calculate PnL
                if pos_side == "long":
                    pnl_usd = (exit_price - entry_price) * position_size
                else:
                    pnl_usd = (entry_price - exit_price) * position_size
                
                session_profit += pnl_usd
                
                trades.append({
                    "symbol": symbol,
                    "side": pos_side,
                    "entry_time": entry_time,
                    "exit_time": current_time,
                    "entry": entry_price,
                    "exit": exit_price,
                    "stop": stop_price,
                    "tp": tp_price,
                    "pnl_usd": pnl_usd,
                    "position_size": position_size,
                    "reason": exit_reason,
                    "session": session_count,
                    "session_profit": session_profit,
                })
                
                trade_num = len(trades)
                profit_sign = "+" if pnl_usd >= 0 else ""
                print(f"  ├─ Trade #{trade_num} [{pos_side.upper()}] "
                      f"{entry_time.strftime('%Y-%m-%d %H:%M')} @ ${entry_price:.4f} → "
                      f"{current_time.strftime('%Y-%m-%d %H:%M')} @ ${exit_price:.4f} | "
                      f"PnL: {profit_sign}${pnl_usd:.2f} [{exit_reason}]")
                
                in_position = False
                pos_side = None
                entry_price = stop_price = tp_price = None
                position_size = None
                entry_time = None
                
                # Check session target
                if session_profit >= SESSION_PROFIT_TARGET:
                    print(f"  └─ ✓ SESSION #{session_count} COMPLETE! Profit: ${session_profit:.2f}\n")
                    session_count += 1
                    session_profit = 0.0
            
            continue
        
        # Look for new setup
        row = df_1h.iloc[i]
        
        # Skip if indicators not ready
        if pd.isna(row['stoch_rsi_d']) or pd.isna(row['stoch_k']) or pd.isna(row['mrpz_histogram']):
            continue
        
        # Get current 4H context
        current_4h_idx = None
        for j in range(len(df_4h)):
            if df_4h.index[j] <= current_time:
                current_4h_idx = j
        
        if current_4h_idx is None or current_4h_idx < STRUCTURE_LOOKBACK:
            continue
        
        # ======================
        # FILTER 1: TREND (Stoch RSI White Line)
        # ======================
        stoch_rsi_white = row['stoch_rsi_d']  # D = White line
        trend_long = stoch_rsi_white > 50
        trend_short = stoch_rsi_white < 50
        
        # ======================
        # FILTER 2: MEAN REVERSION & MOMENTUM
        # ======================
        # LONG: MRPZ lower spike/zone + Stoch K oversold + K cross D up
        mrpz_long_signal = (
            (row['mrpz_lower_spike'] or row['mrpz_in_lower']) and
            row['stoch_k'] < STOCH_OVERSOLD
        )
        
        # Check K/D cross
        stoch_k_cross_up = False
        if i > 0:
            prev_row = df_1h.iloc[i-1]
            if not pd.isna(prev_row['stoch_k']) and not pd.isna(prev_row['stoch_d']):
                stoch_k_cross_up = (
                    prev_row['stoch_k'] <= prev_row['stoch_d'] and
                    row['stoch_k'] > row['stoch_d']
                )
        
        momentum_long = stoch_k_cross_up
        
        # SHORT: MRPZ upper spike/zone + Stoch K overbought + K cross D down
        mrpz_short_signal = (
            (row['mrpz_upper_spike'] or row['mrpz_in_upper']) and
            row['stoch_k'] > STOCH_OVERBOUGHT
        )
        
        stoch_k_cross_down = False
        if i > 0:
            prev_row = df_1h.iloc[i-1]
            if not pd.isna(prev_row['stoch_k']) and not pd.isna(prev_row['stoch_d']):
                stoch_k_cross_down = (
                    prev_row['stoch_k'] >= prev_row['stoch_d'] and
                    row['stoch_k'] < row['stoch_d']
                )
        
        momentum_short = stoch_k_cross_down
        
        # ======================
        # FILTER 3: PRICE ACTION (1H Double Bottom/Top in 4H Structure)
        # ======================
        # Check for double bottom/top in recent 1H data
        lookback_start = max(0, i - STRUCTURE_LOOKBACK)
        df_1h_subset = df_1h.iloc[lookback_start:i+1]
        
        has_double_bottom, support_level = detect_double_bottom(
            df_1h_subset,
            lookback=min(STRUCTURE_LOOKBACK, len(df_1h_subset)),
            tolerance=DOUBLE_PATTERN_TOLERANCE
        )
        
        has_double_top, resistance_level = detect_double_top(
            df_1h_subset,
            lookback=min(STRUCTURE_LOOKBACK, len(df_1h_subset)),
            tolerance=DOUBLE_PATTERN_TOLERANCE
        )
        
        # Check 4H structure zone
        current_price = row['close']
        df_4h_subset = df_4h.iloc[:current_4h_idx+1]
        in_structure, structure_type = is_in_structure_zone(
            current_price,
            df_4h_subset,
            lookback=min(STRUCTURE_LOOKBACK, len(df_4h_subset))
        )
        
        # Price action filter for LONG
        price_action_long = (
            has_double_bottom and
            in_structure and
            structure_type == 'support' and
            row['strong']  # Strong trigger candle
        )
        
        # Price action filter for SHORT
        price_action_short = (
            has_double_top and
            in_structure and
            structure_type == 'resistance' and
            row['strong']  # Strong trigger candle
        )
        
        # ======================
        # COMBINED SIGNAL
        # ======================
        long_signal = (
            trend_long and
            mrpz_long_signal and
            momentum_long and
            price_action_long
        )
        
        short_signal = (
            trend_short and
            mrpz_short_signal and
            momentum_short and
            price_action_short
        )
        
        if not long_signal and not short_signal:
            continue
        
        # Entry setup
        entry = row['close']
        
        if long_signal:
            # Stop: Below double bottom support
            stop = support_level if support_level else row['low'] * 0.995
            risk = entry - stop
            
            # TP1: 1.2R to 1.4R
            tp_r = (TP1_R_MIN + TP1_R_MAX) / 2  # Use average
            tp = entry + (risk * tp_r)
            
            # Position size for target profit
            reward = tp - entry
            pos_size = PROFIT_TARGET / reward if reward > 0 else 0
            
            if pos_size <= 0:
                continue
            
            pos_side = "long"
            
        elif short_signal:
            # Stop: Above double top resistance
            stop = resistance_level if resistance_level else row['high'] * 1.005
            risk = stop - entry
            
            # TP1: 1.2R to 1.4R
            tp_r = (TP1_R_MIN + TP1_R_MAX) / 2
            tp = entry - (risk * tp_r)
            
            # Position size for target profit
            reward = entry - tp
            pos_size = PROFIT_TARGET / reward if reward > 0 else 0
            
            if pos_size <= 0:
                continue
            
            pos_side = "short"
        
        # Open position
        in_position = True
        entry_price = entry
        stop_price = stop
        tp_price = tp
        position_size = pos_size
        entry_time = current_time
        
        print(f"  ⚡ {pos_side.upper()} signal @ {current_time.strftime('%Y-%m-%d %H:%M')} "
              f"Entry: ${entry:.4f} | SL: ${stop:.4f} | TP1: ${tp:.4f}")

    if not trades:
        print(f"[{symbol}] No trades generated.\n")
        return None

    # Results summary
    df_trades = pd.DataFrame(trades)
    total_pnl = df_trades["pnl_usd"].sum()
    win_rate = (df_trades["pnl_usd"] > 0).mean() * 100
    avg_pnl = df_trades["pnl_usd"].mean()
    num_trades = len(df_trades)

    print(f"\n{'='*70}")
    print(f"[{symbol}] SUMMARY: {num_trades} trades | Win: {win_rate:.1f}% | "
          f"Total: ${total_pnl:.2f} | Avg: ${avg_pnl:.2f}")
    print(f"{'='*70}\n")

    return df_trades


def main():
    """Main execution function"""
    end_dt = datetime.now(timezone.utc).replace(microsecond=0, second=0)
    start_dt = end_dt - timedelta(days=DAYS_BACK + 2)

    print("="*80)
    print("TP1 SCALP MASTER STRATEGY - BACKTEST")
    print("="*80)
    print(f"Period: {DAYS_BACK} days")
    print(f"Initial Capital: ${INITIAL_CAPITAL:.2f}")
    print(f"Profit Target per Trade: ${PROFIT_TARGET:.2f}")
    print(f"Session Target: ${SESSION_PROFIT_TARGET:.2f}")
    print(f"TP1 Range: {TP1_R_MIN}R - {TP1_R_MAX}R")
    print("="*80)

    symbols = get_futures_symbols()
    print(f"\nTotal USDT PERPETUAL symbols: {len(symbols)}")
    if MAX_SYMBOLS:
        print(f"Testing top {MAX_SYMBOLS} by volume")
    
    all_trades = []

    for sym in symbols:
        try:
            res = backtest_symbol(sym, start_dt, end_dt)
            if res is not None:
                all_trades.append(res)
        except Exception as e:
            print(f"[{sym}] ERROR: {e}")
            continue

    if not all_trades:
        print("No trade data generated.")
        return

    df_all = pd.concat(all_trades, ignore_index=True)
    df_all = df_all.sort_values('entry_time').reset_index(drop=True)

    # Overall statistics
    total_pnl = df_all["pnl_usd"].sum()
    win_rate = (df_all["pnl_usd"] > 0).mean() * 100
    avg_pnl = df_all["pnl_usd"].mean()
    num_trades = len(df_all)
    
    num_sessions = df_all["session"].max() + 1 if "session" in df_all.columns else 0
    completed_sessions = int(total_pnl / SESSION_PROFIT_TARGET)
    
    df_all['cumulative_pnl'] = df_all['pnl_usd'].cumsum()
    df_all['capital'] = INITIAL_CAPITAL + df_all['cumulative_pnl']
    
    final_capital = df_all['capital'].iloc[-1]
    total_return_pct = ((final_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100

    print("\n" + "="*80)
    print("OVERALL RESULTS - TP1 SCALP MASTER STRATEGY")
    print("="*80)
    print(f"Initial Capital: ${INITIAL_CAPITAL:.2f}")
    print(f"Session Target: ${SESSION_PROFIT_TARGET:.2f} (Per trade: ${PROFIT_TARGET:.2f})")
    print(f"Completed Sessions: {completed_sessions}")
    print(f"Total Trades: {num_trades}")
    print(f"Win Rate: {win_rate:.1f}%")
    print(f"Total PnL: ${total_pnl:.2f}")
    print(f"Average PnL per Trade: ${avg_pnl:.2f}")
    print(f"\n{'='*50}")
    print(f"FINAL CAPITAL: ${final_capital:.2f}")
    print(f"Total Return: {total_return_pct:+.2f}%")
    print(f"{'='*50}")

    # Symbol breakdown
    sym_group = df_all.groupby("symbol")["pnl_usd"].agg(["count", "sum", "mean"])
    sym_group = sym_group.sort_values("sum", ascending=False)
    print("\nTop 30 symbols by performance:")
    print(sym_group.head(30))
    
    print("\n" + "="*80)
    print("BACKTEST COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
