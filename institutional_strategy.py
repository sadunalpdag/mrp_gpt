#!/usr/bin/env python3
"""
Institutional Supply/Demand Zone Trading Strategy
90-day backtest with complete trade logging

Strategy Flow:
1. Get bias from 30m timeframe (mark supply/demand zones)
2. Find institutional zones on 5m
3. Wait for price return and candlestick confirmation
4. Execute trades with proper entry/stop/target
5. Apply risk management rules
"""

import time
import math
import requests
from datetime import datetime, timedelta, timezone
import pandas as pd
import json

from supply_demand_zones import detect_all_zones, is_price_in_zone
from candlestick_patterns import detect_bullish_confirmation, detect_bearish_confirmation
from zone_filters import apply_all_filters, find_logical_target, calculate_atr

# Configuration
BINANCE_FAPI = "https://fapi.binance.com"
DAYS_BACK = 90
INITIAL_CAPITAL = 5000.0
RISK_PER_TRADE_PCT = 0.01  # 1% risk per trade
MAX_TRADES_PER_SESSION = 4
MAX_LOSSES_PER_SESSION = 2
SESSION_TARGET_R = 4.0  # Target R-multiple per session
REQUEST_SLEEP = 0.15

# Trade log file
TRADE_LOG_FILE = "institutional_trades.json"


def ms_since_epoch(dt: datetime) -> int:
    """Convert datetime to milliseconds timestamp (UTC)."""
    return int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000)


def get_futures_symbols():
    """Get USDT perpetual futures symbols from Binance."""
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
    
    # Get 24h volume and sort by liquidity
    tickers_url = f"{BINANCE_FAPI}/fapi/v1/ticker/24hr"
    r2 = requests.get(tickers_url, timeout=10)
    r2.raise_for_status()
    tickers = {t["symbol"]: float(t["volume"]) for t in r2.json()}
    
    symbols = [s for s in symbols if s in tickers]
    symbols.sort(key=lambda s: tickers[s], reverse=True)
    
    return symbols


def fetch_klines(symbol, interval, start_ms, end_ms=None, limit=1500):
    """Fetch klines from Binance."""
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
            print(f"[{symbol}][{interval}] HTTP {resp.status_code}")
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


def get_30m_bias(df_30m, current_time):
    """
    Get trading bias from 30m timeframe.
    
    Returns:
        str: 'bullish', 'bearish', or 'neutral'
    """
    # Get zones on 30m
    zones_30m = detect_all_zones(df_30m, lookback=30)
    
    if not zones_30m:
        return 'neutral'
    
    # Check which zones are active at current time
    current_price = df_30m[df_30m['close_time'] <= current_time]['close'].iloc[-1]
    
    # Find nearest zones
    demand_zones = [z for z in zones_30m if z['direction'] == 'demand']
    supply_zones = [z for z in zones_30m if z['direction'] == 'supply']
    
    # Check if price is near a demand zone (bullish bias)
    for zone in demand_zones:
        if abs(current_price - zone['high']) / current_price < 0.02:  # Within 2%
            return 'bullish'
    
    # Check if price is near a supply zone (bearish bias)
    for zone in supply_zones:
        if abs(current_price - zone['low']) / current_price < 0.02:  # Within 2%
            return 'bearish'
    
    return 'neutral'


def backtest_symbol(symbol, start_dt, end_dt):
    """
    Backtest institutional strategy on a single symbol.
    
    Returns:
        list: List of trade dictionaries
    """
    print(f"\n=== Processing {symbol} ===")
    
    start_ms = ms_since_epoch(start_dt)
    end_ms = ms_since_epoch(end_dt)
    
    # Fetch data
    df_5m = fetch_klines(symbol, "5m", start_ms, end_ms)
    if df_5m.empty or len(df_5m) < 100:
        print(f"[{symbol}] Insufficient 5m data")
        return []
    
    df_30m = fetch_klines(symbol, "30m", start_ms, end_ms)
    if df_30m.empty or len(df_30m) < 50:
        print(f"[{symbol}] Insufficient 30m data")
        return []
    
    # Ensure timezone
    df_5m.index = df_5m.index.tz_convert("UTC")
    df_5m["close_time"] = df_5m["close_time"].dt.tz_convert("UTC")
    df_30m.index = df_30m.index.tz_convert("UTC")
    df_30m["close_time"] = df_30m["close_time"].dt.tz_convert("UTC")
    
    print(f"[{symbol}] Loaded {len(df_5m)} 5m bars, {len(df_30m)} 30m bars")
    
    # Detect zones on 5m
    print(f"[{symbol}] Detecting supply/demand zones...")
    zones_5m = detect_all_zones(df_5m, lookback=50)
    print(f"[{symbol}] Found {len(zones_5m)} zones")
    
    if not zones_5m:
        print(f"[{symbol}] No zones found")
        return []
    
    # Trading simulation
    trades = []
    in_position = False
    position = None
    
    # Session tracking
    session_trades = 0
    session_losses = 0
    session_r_total = 0.0
    
    capital = INITIAL_CAPITAL
    
    # Main loop through 5m bars
    for i in range(100, len(df_5m)):
        current_bar = df_5m.iloc[i]
        current_time = current_bar.name
        current_price = current_bar['close']
        
        # Check position exit first
        if in_position:
            high_price = current_bar['high']
            low_price = current_bar['low']
            
            exit_triggered = False
            exit_price = None
            exit_reason = None
            
            if position['side'] == 'long':
                if high_price >= position['tp']:
                    exit_price = position['tp']
                    exit_reason = 'TP'
                    exit_triggered = True
                elif low_price <= position['sl']:
                    exit_price = position['sl']
                    exit_reason = 'SL'
                    exit_triggered = True
            else:  # short
                if low_price <= position['tp']:
                    exit_price = position['tp']
                    exit_reason = 'TP'
                    exit_triggered = True
                elif high_price >= position['sl']:
                    exit_price = position['sl']
                    exit_reason = 'SL'
                    exit_triggered = True
            
            if exit_triggered:
                # Calculate PnL
                if position['side'] == 'long':
                    pnl_usd = (exit_price - position['entry']) * position['size']
                else:
                    pnl_usd = (position['entry'] - exit_price) * position['size']
                
                # Calculate R-multiple
                risk_usd = abs(position['entry'] - position['sl']) * position['size']
                r_multiple = pnl_usd / risk_usd if risk_usd > 0 else 0
                
                # Update session tracking
                session_r_total += r_multiple
                if pnl_usd < 0:
                    session_losses += 1
                
                # Update capital
                capital += pnl_usd
                
                # Log trade
                trade_record = {
                    'symbol': symbol,
                    'side': position['side'],
                    'entry_time': position['entry_time'].strftime('%Y-%m-%d %H:%M:%S'),
                    'exit_time': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'entry_price': position['entry'],
                    'exit_price': exit_price,
                    'sl_price': position['sl'],
                    'tp_price': position['tp'],
                    'size': position['size'],
                    'pnl_usd': pnl_usd,
                    'r_multiple': r_multiple,
                    'exit_reason': exit_reason,
                    'zone_type': position['zone_type'],
                    'confirmation_pattern': position['confirmation'],
                    'capital': capital
                }
                
                trades.append(trade_record)
                
                # Print trade
                sign = '+' if pnl_usd >= 0 else ''
                print(f"  └─ Trade #{len(trades)}: {position['side'].upper()} | "
                      f"Entry: {position['entry_time'].strftime('%m-%d %H:%M')} @ ${position['entry']:.2f} → "
                      f"Exit: {current_time.strftime('%m-%d %H:%M')} @ ${exit_price:.2f} | "
                      f"PnL: {sign}${pnl_usd:.2f} ({sign}{r_multiple:.2f}R) | "
                      f"Reason: {exit_reason} | Pattern: {position['confirmation']}")
                
                # Reset position
                in_position = False
                position = None
                
                # Check session limits
                if session_losses >= MAX_LOSSES_PER_SESSION:
                    print(f"  ⚠️  Session limit: {session_losses} losses reached. Resetting session.")
                    session_trades = 0
                    session_losses = 0
                    session_r_total = 0.0
                elif session_r_total >= SESSION_TARGET_R:
                    print(f"  ✓ Session target reached: {session_r_total:.2f}R. Resetting session.")
                    session_trades = 0
                    session_losses = 0
                    session_r_total = 0.0
                
                continue
        
        # Look for new entry if not in position
        if not in_position and session_trades < MAX_TRADES_PER_SESSION:
            # Get 30m bias
            bias = get_30m_bias(df_30m, current_time)
            
            # Check each zone for entry opportunity
            for zone in zones_5m:
                # Skip zones that haven't been created yet
                if zone['created_at'] > current_time:
                    continue
                
                # Skip zones that have been tested too many times
                if zone['tested'] >= 1:
                    continue
                
                # Check if price is in zone
                if not is_price_in_zone(current_price, zone):
                    continue
                
                # Check bias alignment
                if zone['direction'] == 'demand' and bias == 'bearish':
                    continue
                if zone['direction'] == 'supply' and bias == 'bullish':
                    continue
                
                # Look for candlestick confirmation
                confirmation = None
                if zone['direction'] == 'demand':
                    confirmation = detect_bullish_confirmation(df_5m, i)
                else:
                    confirmation = detect_bearish_confirmation(df_5m, i)
                
                if not confirmation:
                    continue
                
                # We have a confirmation! Now apply filters
                # Calculate entry, stop, target
                entry_price = current_price
                
                if zone['direction'] == 'demand':
                    # Long setup
                    stop_price = zone['low'] - (zone['high'] - zone['low']) * 0.1
                    target_price = find_logical_target(df_5m, i, 'demand', entry_price)
                    side = 'long'
                else:
                    # Short setup
                    stop_price = zone['high'] + (zone['high'] - zone['low']) * 0.1
                    target_price = find_logical_target(df_5m, i, 'supply', entry_price)
                    side = 'short'
                
                # Apply all filters
                filter_results = apply_all_filters(
                    df_5m, zone, i, entry_price, stop_price, target_price
                )
                
                if not filter_results['passed']:
                    # Failed filters, skip this zone
                    continue
                
                # All filters passed! Calculate position size
                risk_per_trade = capital * RISK_PER_TRADE_PCT
                risk_per_unit = abs(entry_price - stop_price)
                position_size = risk_per_trade / risk_per_unit if risk_per_unit > 0 else 0
                
                if position_size <= 0:
                    continue
                
                # Mark zone as tested
                zone['tested'] += 1
                
                # Enter position
                position = {
                    'side': side,
                    'entry': entry_price,
                    'sl': stop_price,
                    'tp': target_price,
                    'size': position_size,
                    'entry_time': current_time,
                    'zone_type': zone['type'],
                    'confirmation': confirmation
                }
                
                in_position = True
                session_trades += 1
                
                print(f"  ├─ ENTRY: {side.upper()} @ ${entry_price:.2f} | "
                      f"SL: ${stop_price:.2f} | TP: ${target_price:.2f} | "
                      f"Zone: {zone['type']} | Pattern: {confirmation}")
                
                # Only one entry per bar
                break
    
    print(f"[{symbol}] Completed: {len(trades)} trades")
    return trades


def main():
    """Main execution function."""
    print("=" * 80)
    print("INSTITUTIONAL SUPPLY/DEMAND ZONE STRATEGY - 90 DAY BACKTEST")
    print("=" * 80)
    
    end_dt = datetime.now(timezone.utc).replace(microsecond=0, second=0)
    start_dt = end_dt - timedelta(days=DAYS_BACK + 2)
    
    print(f"\nBacktest Period: {start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')}")
    print(f"Initial Capital: ${INITIAL_CAPITAL:.2f}")
    print(f"Risk per Trade: {RISK_PER_TRADE_PCT*100:.1f}%")
    
    # Get symbols
    print("\nFetching symbols...")
    symbols = get_futures_symbols()
    print(f"Total symbols: {len(symbols)}")
    
    # For testing, limit to top 20 symbols
    symbols = symbols[:20]
    print(f"Testing with top {len(symbols)} symbols by volume")
    
    # Run backtest on all symbols
    all_trades = []
    
    for sym in symbols:
        try:
            trades = backtest_symbol(sym, start_dt, end_dt)
            all_trades.extend(trades)
        except Exception as e:
            print(f"[{sym}] ERROR: {e}")
            continue
    
    if not all_trades:
        print("\n❌ No trades generated")
        return
    
    # Save trades to JSON
    with open(TRADE_LOG_FILE, 'w') as f:
        json.dump(all_trades, f, indent=2)
    
    print(f"\n✓ Trades saved to {TRADE_LOG_FILE}")
    
    # Calculate statistics
    df_trades = pd.DataFrame(all_trades)
    
    total_trades = len(df_trades)
    winning_trades = (df_trades['pnl_usd'] > 0).sum()
    losing_trades = (df_trades['pnl_usd'] < 0).sum()
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    
    total_pnl = df_trades['pnl_usd'].sum()
    avg_pnl = df_trades['pnl_usd'].mean()
    
    total_r = df_trades['r_multiple'].sum()
    avg_r = df_trades['r_multiple'].mean()
    
    final_capital = INITIAL_CAPITAL + total_pnl
    return_pct = (total_pnl / INITIAL_CAPITAL) * 100
    
    # Print summary
    print("\n" + "=" * 80)
    print("BACKTEST SUMMARY")
    print("=" * 80)
    print(f"Total Trades: {total_trades}")
    print(f"Winning Trades: {winning_trades}")
    print(f"Losing Trades: {losing_trades}")
    print(f"Win Rate: {win_rate:.1f}%")
    print(f"\nTotal PnL: ${total_pnl:.2f}")
    print(f"Average PnL per Trade: ${avg_pnl:.2f}")
    print(f"Total R-Multiple: {total_r:.2f}R")
    print(f"Average R per Trade: {avg_r:.2f}R")
    print(f"\nInitial Capital: ${INITIAL_CAPITAL:.2f}")
    print(f"Final Capital: ${final_capital:.2f}")
    print(f"Total Return: {return_pct:+.2f}%")
    print("=" * 80)
    
    # Pattern performance
    print("\nPattern Performance:")
    pattern_stats = df_trades.groupby('confirmation_pattern').agg({
        'pnl_usd': ['count', 'sum', 'mean'],
        'r_multiple': 'mean'
    }).round(2)
    print(pattern_stats)
    
    # Zone type performance
    print("\nZone Type Performance:")
    zone_stats = df_trades.groupby('zone_type').agg({
        'pnl_usd': ['count', 'sum', 'mean'],
        'r_multiple': 'mean'
    }).round(2)
    print(zone_stats)
    
    # Symbol performance
    print("\nTop 10 Symbols by Total PnL:")
    symbol_pnl = df_trades.groupby('symbol')['pnl_usd'].sum().sort_values(ascending=False).head(10)
    for sym, pnl in symbol_pnl.items():
        print(f"  {sym}: ${pnl:.2f}")


if __name__ == "__main__":
    main()
