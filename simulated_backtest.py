#!/usr/bin/env python3
"""
Simulated backtest using synthetic price data to demonstrate strategy
This version doesn't require external API access
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import json

from supply_demand_zones import detect_all_zones, is_price_in_zone
from candlestick_patterns import detect_bullish_confirmation, detect_bearish_confirmation
from zone_filters import apply_all_filters, find_logical_target

# Configuration
DAYS_BACK = 90
INITIAL_CAPITAL = 5000.0
RISK_PER_TRADE_PCT = 0.01
MAX_TRADES_PER_SESSION = 4
MAX_LOSSES_PER_SESSION = 2
SESSION_TARGET_R = 4.0
TRADE_LOG_FILE = "institutional_trades_simulated.json"


def generate_realistic_price_data(start_dt, end_dt, interval_minutes=5, base_price=100.0):
    """
    Generate realistic OHLC price data with trends and zones.
    
    Args:
        start_dt: Start datetime
        end_dt: End datetime
        interval_minutes: Candle interval in minutes
        base_price: Starting price
    
    Returns:
        DataFrame with OHLC data
    """
    # Calculate number of bars
    total_minutes = int((end_dt - start_dt).total_seconds() / 60)
    num_bars = total_minutes // interval_minutes
    
    # Generate timestamps
    timestamps = pd.date_range(start=start_dt, periods=num_bars, freq=f'{interval_minutes}min', tz='UTC')
    
    # Generate price with trend and volatility
    np.random.seed(42)  # For reproducibility
    
    # Create trend components
    trend = np.cumsum(np.random.randn(num_bars) * 0.002)  # Long-term trend
    
    # Add zones (areas of consolidation followed by breakouts)
    zones_pattern = np.zeros(num_bars)
    for i in range(0, num_bars, 200):
        # Create consolidation zone
        zone_size = min(50, num_bars - i)
        zones_pattern[i:i+zone_size] = np.random.randn() * 0.001
        
        # Add breakout after zone
        if i + zone_size < num_bars:
            breakout_size = min(20, num_bars - i - zone_size)
            direction = 1 if np.random.rand() > 0.5 else -1
            zones_pattern[i+zone_size:i+zone_size+breakout_size] = direction * 0.01
    
    # Combine components
    returns = trend * 0.5 + zones_pattern + np.random.randn(num_bars) * 0.005
    prices = base_price * (1 + returns).cumprod()
    
    # Generate OHLC with realistic intrabar movement
    volatility = 0.003
    data = []
    
    for i, price in enumerate(prices):
        # Generate realistic OHLC
        open_price = price * (1 + np.random.randn() * volatility * 0.5)
        close_price = price * (1 + np.random.randn() * volatility * 0.5)
        
        # High and low based on volatility
        high_offset = abs(np.random.randn() * volatility)
        low_offset = abs(np.random.randn() * volatility)
        
        high_price = max(open_price, close_price) * (1 + high_offset)
        low_price = min(open_price, close_price) * (1 - low_offset)
        
        # Volume
        volume = np.random.randint(1000, 10000)
        
        data.append({
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        })
    
    df = pd.DataFrame(data, index=timestamps)
    df['close_time'] = df.index + timedelta(minutes=interval_minutes)
    
    return df


def simulate_backtest(symbol, df_5m, df_30m):
    """
    Run backtest simulation on provided data.
    
    Args:
        symbol: Symbol name
        df_5m: 5-minute OHLC data
        df_30m: 30-minute OHLC data
    
    Returns:
        list: List of trade dictionaries
    """
    print(f"\n=== Simulating {symbol} ===")
    print(f"[{symbol}] Data: {len(df_5m)} 5m bars, {len(df_30m)} 30m bars")
    
    # Detect zones on 5m
    print(f"[{symbol}] Detecting zones...")
    zones_5m = detect_all_zones(df_5m, lookback=50)
    print(f"[{symbol}] Found {len(zones_5m)} zones")
    
    if not zones_5m:
        return []
    
    # Detect zones on 30m for bias
    zones_30m = detect_all_zones(df_30m, lookback=30)
    print(f"[{symbol}] Found {len(zones_30m)} 30m zones for bias")
    
    # Trading simulation
    trades = []
    in_position = False
    position = None
    
    session_trades = 0
    session_losses = 0
    session_r_total = 0.0
    
    capital = INITIAL_CAPITAL
    
    # Main loop
    for i in range(100, len(df_5m)):
        current_bar = df_5m.iloc[i]
        current_time = current_bar.name
        current_price = current_bar['close']
        
        # Exit logic
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
            else:
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
                
                # Update tracking
                session_r_total += r_multiple
                if pnl_usd < 0:
                    session_losses += 1
                
                capital += pnl_usd
                
                # Log trade
                trade_record = {
                    'symbol': symbol,
                    'side': position['side'],
                    'entry_time': position['entry_time'].strftime('%Y-%m-%d %H:%M:%S'),
                    'exit_time': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'entry_price': float(position['entry']),
                    'exit_price': float(exit_price),
                    'sl_price': float(position['sl']),
                    'tp_price': float(position['tp']),
                    'size': float(position['size']),
                    'pnl_usd': float(pnl_usd),
                    'r_multiple': float(r_multiple),
                    'exit_reason': exit_reason,
                    'zone_type': position['zone_type'],
                    'confirmation_pattern': position['confirmation'],
                    'capital': float(capital)
                }
                
                trades.append(trade_record)
                
                # Print
                sign = '+' if pnl_usd >= 0 else ''
                print(f"  └─ Trade #{len(trades)}: {position['side'].upper()} | "
                      f"PnL: {sign}${pnl_usd:.2f} ({sign}{r_multiple:.2f}R) | "
                      f"{exit_reason} | {position['confirmation']}")
                
                in_position = False
                position = None
                
                # Check session limits
                if session_losses >= MAX_LOSSES_PER_SESSION:
                    print(f"  ⚠️  Session limit: {session_losses} losses. Resetting.")
                    session_trades = 0
                    session_losses = 0
                    session_r_total = 0.0
                elif session_r_total >= SESSION_TARGET_R:
                    print(f"  ✓ Session target: {session_r_total:.2f}R. Resetting.")
                    session_trades = 0
                    session_losses = 0
                    session_r_total = 0.0
                
                continue
        
        # Entry logic
        if not in_position and session_trades < MAX_TRADES_PER_SESSION:
            # Simple bias: check if 30m zones support direction
            bias = 'neutral'
            
            for zone in zones_5m:
                if zone['created_at'] > current_time:
                    continue
                
                if zone['tested'] >= 1:
                    continue
                
                if not is_price_in_zone(current_price, zone):
                    continue
                
                # Look for confirmation
                confirmation = None
                if zone['direction'] == 'demand':
                    confirmation = detect_bullish_confirmation(df_5m, i)
                else:
                    confirmation = detect_bearish_confirmation(df_5m, i)
                
                if not confirmation:
                    continue
                
                # Calculate levels
                entry_price = current_price
                
                if zone['direction'] == 'demand':
                    stop_price = zone['low'] - (zone['high'] - zone['low']) * 0.1
                    target_price = find_logical_target(df_5m, i, 'demand', entry_price)
                    side = 'long'
                else:
                    stop_price = zone['high'] + (zone['high'] - zone['low']) * 0.1
                    target_price = find_logical_target(df_5m, i, 'supply', entry_price)
                    side = 'short'
                
                # Apply filters
                filter_results = apply_all_filters(
                    df_5m, zone, i, entry_price, stop_price, target_price
                )
                
                if not filter_results['passed']:
                    continue
                
                # Calculate position size
                risk_per_trade = capital * RISK_PER_TRADE_PCT
                risk_per_unit = abs(entry_price - stop_price)
                position_size = risk_per_trade / risk_per_unit if risk_per_unit > 0 else 0
                
                if position_size <= 0:
                    continue
                
                zone['tested'] += 1
                
                # Enter
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
                      f"{zone['type']} | {confirmation}")
                
                break
    
    print(f"[{symbol}] Completed: {len(trades)} trades")
    return trades


def main():
    """Main execution."""
    print("=" * 80)
    print("INSTITUTIONAL SUPPLY/DEMAND STRATEGY - SIMULATED 90-DAY BACKTEST")
    print("=" * 80)
    
    end_dt = datetime(2024, 11, 17, tzinfo=timezone.utc)
    start_dt = end_dt - timedelta(days=DAYS_BACK)
    
    print(f"\nPeriod: {start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')}")
    print(f"Initial Capital: ${INITIAL_CAPITAL:.2f}")
    print(f"Risk per Trade: {RISK_PER_TRADE_PCT*100:.1f}%")
    
    # Generate synthetic data for multiple symbols
    symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "SOLUSDT"]
    print(f"\nSimulating {len(symbols)} symbols")
    
    all_trades = []
    
    for i, symbol in enumerate(symbols):
        # Generate unique data for each symbol
        np.random.seed(42 + i)
        
        # Generate 5m data
        df_5m = generate_realistic_price_data(
            start_dt, end_dt, interval_minutes=5,
            base_price=100.0 * (i + 1)
        )
        
        # Generate 30m data
        df_30m = generate_realistic_price_data(
            start_dt, end_dt, interval_minutes=30,
            base_price=100.0 * (i + 1)
        )
        
        # Run backtest
        trades = simulate_backtest(symbol, df_5m, df_30m)
        all_trades.extend(trades)
    
    if not all_trades:
        print("\n❌ No trades generated")
        return
    
    # Save trades
    with open(TRADE_LOG_FILE, 'w') as f:
        json.dump(all_trades, f, indent=2)
    
    print(f"\n✓ Trades saved to {TRADE_LOG_FILE}")
    
    # Statistics
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
    if 'confirmation_pattern' in df_trades.columns:
        print("\nPattern Performance:")
        pattern_stats = df_trades.groupby('confirmation_pattern').agg({
            'pnl_usd': ['count', 'sum', 'mean'],
            'r_multiple': 'mean'
        }).round(2)
        print(pattern_stats)
    
    # Zone type performance
    if 'zone_type' in df_trades.columns:
        print("\nZone Type Performance:")
        zone_stats = df_trades.groupby('zone_type').agg({
            'pnl_usd': ['count', 'sum', 'mean'],
            'r_multiple': 'mean'
        }).round(2)
        print(zone_stats)
    
    print("\n✓ Simulation completed successfully!")


if __name__ == "__main__":
    main()
