#!/usr/bin/env python3
"""
Fractal Support & Resistance Strategy
A fractally validated zone trading system built around clustered swing highs/lows.

Core Concept:
- Clusters multiple fractal points within ATR-scaled window
- Zone validated if enough fractals (≥3 for default, 2 for crypto) form near same level
- Green zones: Support (clustered low fractals)
- Orange zones: Resistance (clustered high fractals)

Trade Logic:
- Long: Validated support zone → price moves away → retest with wick → entry
- Short: Validated resistance zone → price moves away → retest with wick → entry
- Stop: Just below/above zone
- Take Profit: 1.5 × risk

Market-Specific Tuning:
- Crypto: fractals_qty=2, type='Extreme'
- Forex: fractals_qty=3, type='Average'
- Stocks: fractals_qty=4, type='Average'
"""

import os
import sys
import json
import time
import argparse
import requests
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

BINANCE_FAPI = "https://fapi.binance.com"
RESULTS_FILE = "fractal_sr_results.json"
TRADES_FILE = "fractal_sr_trades.json"
RATE_LIMIT_DELAY = 0.1

# ===================== FRACTAL DETECTION =====================

def detect_fractals(highs, lows, closes, fractal_period=5):
    """
    Detect fractal highs and lows using N-bar pattern.
    A fractal high is formed when high[i] is highest among surrounding bars.
    A fractal low is formed when low[i] is lowest among surrounding bars.
    
    Returns:
        fractal_highs: list of (index, price) tuples for resistance fractals
        fractal_lows: list of (index, price) tuples for support fractals
    """
    fractal_highs = []
    fractal_lows = []
    
    # Need at least fractal_period bars on each side
    lookback = fractal_period // 2
    
    for i in range(lookback, len(highs) - lookback):
        # Check for fractal high (resistance)
        is_fractal_high = True
        for j in range(i - lookback, i + lookback + 1):
            if j != i and highs[j] >= highs[i]:
                is_fractal_high = False
                break
        
        if is_fractal_high:
            fractal_highs.append((i, highs[i]))
        
        # Check for fractal low (support)
        is_fractal_low = True
        for j in range(i - lookback, i + lookback + 1):
            if j != i and lows[j] <= lows[i]:
                is_fractal_low = False
                break
        
        if is_fractal_low:
            fractal_lows.append((i, lows[i]))
    
    return fractal_highs, fractal_lows


def calculate_atr(highs, lows, closes, period=14):
    """Calculate Average True Range"""
    if len(highs) < period + 1:
        return [0] * len(highs)
    
    tr = []
    for i in range(len(highs)):
        if i == 0:
            tr.append(highs[i] - lows[i])
        else:
            tr.append(max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            ))
    
    atr_values = []
    # First ATR is simple average
    atr_values.append(sum(tr[:period]) / period)
    
    # Subsequent ATRs use smoothing
    for i in range(period, len(tr)):
        atr_values.append((atr_values[-1] * (period - 1) + tr[i]) / period)
    
    # Pad beginning with first ATR value
    return [atr_values[0]] * (period - 1) + atr_values


def cluster_fractals(fractals, atr_values, closes, cluster_tolerance=1.5, zone_type='Average'):
    """
    Cluster fractals that are close to each other (within ATR-scaled window).
    
    Args:
        fractals: list of (index, price) tuples
        atr_values: ATR values for each bar
        closes: close prices for each bar
        cluster_tolerance: ATR multiplier for clustering window
        zone_type: 'Average' (center) or 'Extreme' (edge)
    
    Returns:
        zones: list of dicts with zone info
    """
    if not fractals:
        return []
    
    zones = []
    used = set()
    
    for i, (idx1, price1) in enumerate(fractals):
        if i in used:
            continue
        
        # Get ATR at this point
        atr = atr_values[idx1] if idx1 < len(atr_values) else atr_values[-1]
        if atr == 0:
            atr = closes[idx1] * 0.01  # Fallback to 1% of price
        
        # Define clustering window
        cluster_window = atr * cluster_tolerance
        
        # Find all fractals within this window
        cluster = [(idx1, price1)]
        cluster_indices = [i]
        
        for j, (idx2, price2) in enumerate(fractals[i+1:], start=i+1):
            if j in used:
                continue
            
            if abs(price2 - price1) <= cluster_window:
                cluster.append((idx2, price2))
                cluster_indices.append(j)
        
        # Mark as used
        for idx in cluster_indices:
            used.add(idx)
        
        # Calculate zone level
        if zone_type == 'Average':
            zone_price = sum(p for _, p in cluster) / len(cluster)
        else:  # Extreme
            zone_price = cluster[0][1]  # Use first (most extreme) fractal
        
        # Calculate zone boundaries (half ATR above/below)
        zone_upper = zone_price + (atr * 0.5)
        zone_lower = zone_price - (atr * 0.5)
        
        zones.append({
            'price': zone_price,
            'upper': zone_upper,
            'lower': zone_lower,
            'count': len(cluster),
            'fractals': cluster,
            'first_bar': min(idx for idx, _ in cluster),
            'last_bar': max(idx for idx, _ in cluster),
            'atr': atr
        })
    
    return zones


# ===================== ZONE VALIDATION =====================

def validate_zones(zones, min_fractals=3):
    """
    Filter zones that have minimum required fractals.
    
    Args:
        zones: list of zone dicts
        min_fractals: minimum number of fractals required
    
    Returns:
        validated_zones: list of validated zone dicts
    """
    return [z for z in zones if z['count'] >= min_fractals]


def is_price_in_zone(price, zone):
    """Check if price is within zone boundaries"""
    return zone['lower'] <= price <= zone['upper']


def has_price_moved_away(closes, current_idx, zone, min_bars_away=8):
    """
    Check if price moved away from zone for at least min_bars_away bars.
    
    Args:
        closes: close prices
        current_idx: current bar index
        zone: zone dict
        min_bars_away: minimum bars to be away from zone
    
    Returns:
        bool: True if price moved away
    """
    if current_idx < zone['last_bar'] + min_bars_away:
        return False
    
    # Check if price was away from zone for min_bars_away bars
    away_count = 0
    for i in range(zone['last_bar'] + 1, current_idx):
        if not is_price_in_zone(closes[i], zone):
            away_count += 1
        else:
            away_count = 0  # Reset if touched zone
    
    return away_count >= min_bars_away


def is_retest_with_wick(highs, lows, closes, current_idx, zone, zone_type='support'):
    """
    Check if current bar retests zone with wick only.
    
    Args:
        highs, lows, closes: price arrays
        current_idx: current bar index
        zone: zone dict
        zone_type: 'support' or 'resistance'
    
    Returns:
        bool: True if wick-only retest
    """
    current_high = highs[current_idx]
    current_low = lows[current_idx]
    current_close = closes[current_idx]
    
    if zone_type == 'support':
        # For support: low should touch zone, but close should be above
        wick_touches = is_price_in_zone(current_low, zone)
        close_above = current_close > zone['upper']
        return wick_touches and close_above
    else:  # resistance
        # For resistance: high should touch zone, but close should be below
        wick_touches = is_price_in_zone(current_high, zone)
        close_below = current_close < zone['lower']
        return wick_touches and close_below


# ===================== TRADE LOGIC =====================

def generate_trade_signal(symbol, klines, support_zones, resistance_zones, 
                         bar_index, min_bars_away=8, wait_bars_confirm=5):
    """
    Generate trade signal based on fractal zones.
    
    Returns:
        signal dict or None
    """
    if bar_index < wait_bars_confirm:
        return None
    
    closes = [float(k[4]) for k in klines]
    highs = [float(k[2]) for k in klines]
    lows = [float(k[3]) for k in klines]
    
    current_close = closes[bar_index]
    
    # Check for LONG setup (support zone retest)
    for zone in support_zones:
        if zone['last_bar'] >= bar_index:
            continue  # Zone not formed yet
        
        # Check if price moved away
        if not has_price_moved_away(closes, bar_index, zone, min_bars_away):
            continue
        
        # Check for wick retest
        if is_retest_with_wick(highs, lows, closes, bar_index, zone, 'support'):
            # Wait for fractal confirmation (5 bars)
            if bar_index + wait_bars_confirm >= len(klines):
                continue
            
            # Entry on next bar after confirmation
            entry_price = closes[bar_index]
            stop_loss = zone['lower']  # Just below zone
            risk = entry_price - stop_loss
            take_profit = entry_price + (risk * 1.5)  # 1.5R
            
            return {
                'symbol': symbol,
                'direction': 'LONG',
                'entry': entry_price,
                'stop': stop_loss,
                'tp': take_profit,
                'zone_price': zone['price'],
                'zone_count': zone['count'],
                'entry_bar': bar_index,
                'zone_type': 'support'
            }
    
    # Check for SHORT setup (resistance zone retest)
    for zone in resistance_zones:
        if zone['last_bar'] >= bar_index:
            continue  # Zone not formed yet
        
        # Check if price moved away
        if not has_price_moved_away(closes, bar_index, zone, min_bars_away):
            continue
        
        # Check for wick retest
        if is_retest_with_wick(highs, lows, closes, bar_index, zone, 'resistance'):
            # Wait for fractal confirmation (5 bars)
            if bar_index + wait_bars_confirm >= len(klines):
                continue
            
            # Entry on next bar after confirmation
            entry_price = closes[bar_index]
            stop_loss = zone['upper']  # Just above zone
            risk = stop_loss - entry_price
            take_profit = entry_price - (risk * 1.5)  # 1.5R
            
            return {
                'symbol': symbol,
                'direction': 'SHORT',
                'entry': entry_price,
                'stop': stop_loss,
                'tp': take_profit,
                'zone_price': zone['price'],
                'zone_count': zone['count'],
                'entry_bar': bar_index,
                'zone_type': 'resistance'
            }
    
    return None


def simulate_trade(signal, klines, entry_bar_index):
    """
    Simulate trade execution from entry to TP/SL.
    
    Returns:
        trade result dict or None
    """
    entry = signal['entry']
    tp = signal['tp']
    sl = signal['stop']
    direction = signal['direction']
    
    entry_time = int(klines[entry_bar_index][0])
    
    # Look forward from entry to find TP or SL hit
    for i in range(entry_bar_index + 1, len(klines)):
        kline = klines[i]
        high = float(kline[2])
        low = float(kline[3])
        close = float(kline[4])
        current_time = int(kline[0])
        
        hit_tp = False
        hit_sl = False
        exit_price = close
        
        if direction == 'LONG':
            if high >= tp:
                hit_tp = True
                exit_price = tp
            elif low <= sl:
                hit_sl = True
                exit_price = sl
        else:  # SHORT
            if low <= tp:
                hit_tp = True
                exit_price = tp
            elif high >= sl:
                hit_sl = True
                exit_price = sl
        
        if hit_tp or hit_sl:
            # Calculate P&L
            if direction == 'LONG':
                pnl_pct = ((exit_price / entry) - 1.0) * 100
            else:
                pnl_pct = ((entry / exit_price) - 1.0) * 100
            
            duration_hours = (current_time - entry_time) / (1000 * 3600)
            
            return {
                'symbol': signal['symbol'],
                'direction': direction,
                'entry_time': datetime.fromtimestamp(entry_time / 1000, tz=timezone.utc).isoformat(),
                'exit_time': datetime.fromtimestamp(current_time / 1000, tz=timezone.utc).isoformat(),
                'entry_price': entry,
                'exit_price': exit_price,
                'tp': tp,
                'sl': sl,
                'exit_reason': 'TP' if hit_tp else 'SL',
                'pnl_pct': round(pnl_pct, 3),
                'duration_hours': round(duration_hours, 2),
                'zone_price': signal['zone_price'],
                'zone_count': signal['zone_count'],
                'zone_type': signal['zone_type']
            }
    
    # Trade not completed within data range
    return None


# ===================== BACKTEST ENGINE =====================

def log(msg):
    """Print and log messages"""
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}", flush=True)


def get_all_futures_symbols():
    """Get all USDT futures symbols from Binance"""
    try:
        info = requests.get(f"{BINANCE_FAPI}/fapi/v1/exchangeInfo", timeout=10).json()
        symbols = [
            s["symbol"] for s in info["symbols"]
            if s.get("quoteAsset") == "USDT" and s.get("status") == "TRADING"
        ]
        log(f"Found {len(symbols)} futures symbols")
        return sorted(symbols)
    except Exception as e:
        log(f"Error getting symbols: {e}")
        return [
            "BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "DOGEUSDT",
            "XRPUSDT", "DOTUSDT", "UNIUSDT", "LINKUSDT", "MATICUSDT",
            "SOLUSDT", "AVAXUSDT", "ATOMUSDT", "LTCUSDT", "ETCUSDT"
        ]


def get_historical_klines(symbol, interval="1h", days=90):
    """Get historical klines for a symbol"""
    try:
        end_time = int(datetime.now(timezone.utc).timestamp() * 1000)
        start_time = end_time - (days * 24 * 60 * 60 * 1000)
        
        all_klines = []
        current_start = start_time
        
        while current_start < end_time:
            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": current_start,
                "endTime": end_time,
                "limit": 1500
            }
            
            response = requests.get(
                f"{BINANCE_FAPI}/fapi/v1/klines",
                params=params,
                timeout=15
            )
            
            if response.status_code != 200:
                log(f"  {symbol}: Error {response.status_code}")
                break
            
            klines = response.json()
            
            if not klines:
                break
            
            all_klines.extend(klines)
            
            # Move to next batch
            current_start = int(klines[-1][0]) + 1
            
            # Rate limiting
            time.sleep(RATE_LIMIT_DELAY)
            
            if len(klines) < 1500:
                break
        
        log(f"  {symbol}: Retrieved {len(all_klines)} klines")
        return all_klines
        
    except Exception as e:
        log(f"  {symbol}: Error - {e}")
        return []


def backtest_symbol(symbol, fractals_qty=2, zone_type='Extreme', days=90):
    """
    Backtest Fractal S&R strategy for a single symbol.
    
    Args:
        symbol: trading symbol
        fractals_qty: minimum fractals required per zone (2 for crypto)
        zone_type: 'Average' (center) or 'Extreme' (edge)
        days: backtest period in days
    
    Returns:
        list of completed trades
    """
    klines = get_historical_klines(symbol, interval="1h", days=days)
    
    if len(klines) < 100:
        return []
    
    # Extract price data
    closes = [float(k[4]) for k in klines]
    highs = [float(k[2]) for k in klines]
    lows = [float(k[3]) for k in klines]
    
    # Calculate ATR
    atr_values = calculate_atr(highs, lows, closes, period=14)
    
    # Detect fractals
    fractal_highs, fractal_lows = detect_fractals(highs, lows, closes, fractal_period=5)
    
    log(f"  {symbol}: Found {len(fractal_highs)} resistance fractals, {len(fractal_lows)} support fractals")
    
    # Cluster fractals into zones
    resistance_zones = cluster_fractals(fractal_highs, atr_values, closes, 
                                       cluster_tolerance=1.5, zone_type=zone_type)
    support_zones = cluster_fractals(fractal_lows, atr_values, closes, 
                                    cluster_tolerance=1.5, zone_type=zone_type)
    
    # Validate zones
    resistance_zones = validate_zones(resistance_zones, min_fractals=fractals_qty)
    support_zones = validate_zones(support_zones, min_fractals=fractals_qty)
    
    log(f"  {symbol}: {len(support_zones)} validated support zones, "
        f"{len(resistance_zones)} validated resistance zones")
    
    if not support_zones and not resistance_zones:
        return []
    
    # Scan for trade signals
    trades = []
    signals_found = 0
    
    for i in range(60, len(klines) - 10):  # Leave some bars at end for trade completion
        signal = generate_trade_signal(
            symbol, klines, support_zones, resistance_zones, i,
            min_bars_away=8, wait_bars_confirm=5
        )
        
        if signal:
            signals_found += 1
            
            # Simulate the trade
            trade_result = simulate_trade(signal, klines, i)
            
            if trade_result:
                trades.append(trade_result)
    
    if signals_found > 0:
        log(f"  {symbol}: {signals_found} signals, {len(trades)} completed trades")
    
    return trades


def analyze_results(all_trades):
    """Analyze backtest results and generate statistics"""
    if not all_trades:
        return {
            "error": "No trades found",
            "total_trades": 0
        }
    
    stats = {
        "total_trades": len(all_trades),
        "by_direction": {"LONG": [], "SHORT": []},
        "by_zone_count": {},
        "winners": [t for t in all_trades if t["exit_reason"] == "TP"],
        "losers": [t for t in all_trades if t["exit_reason"] == "SL"],
        "overall": {}
    }
    
    # Overall statistics
    win_count = len(stats["winners"])
    loss_count = len(stats["losers"])
    total = len(all_trades)
    
    stats["overall"]["win_rate"] = round((win_count / total * 100) if total > 0 else 0, 2)
    stats["overall"]["total_pnl"] = round(sum(t["pnl_pct"] for t in all_trades), 2)
    stats["overall"]["avg_pnl"] = round(sum(t["pnl_pct"] for t in all_trades) / total if total > 0 else 0, 3)
    stats["overall"]["avg_winner"] = round(sum(t["pnl_pct"] for t in stats["winners"]) / win_count if win_count > 0 else 0, 3)
    stats["overall"]["avg_loser"] = round(sum(t["pnl_pct"] for t in stats["losers"]) / loss_count if loss_count > 0 else 0, 3)
    stats["overall"]["avg_duration_hours"] = round(sum(t["duration_hours"] for t in all_trades) / total if total > 0 else 0, 2)
    
    # By direction
    for direction in ["LONG", "SHORT"]:
        dir_trades = [t for t in all_trades if t["direction"] == direction]
        dir_winners = [t for t in dir_trades if t["exit_reason"] == "TP"]
        
        if dir_trades:
            stats["by_direction"][direction] = {
                "total_trades": len(dir_trades),
                "wins": len(dir_winners),
                "win_rate": round((len(dir_winners) / len(dir_trades) * 100), 2),
                "total_pnl": round(sum(t["pnl_pct"] for t in dir_trades), 2),
                "avg_pnl": round(sum(t["pnl_pct"] for t in dir_trades) / len(dir_trades), 3)
            }
    
    # By zone count
    zone_counts = set(t["zone_count"] for t in all_trades)
    for count in sorted(zone_counts):
        count_trades = [t for t in all_trades if t["zone_count"] == count]
        count_winners = [t for t in count_trades if t["exit_reason"] == "TP"]
        
        stats["by_zone_count"][f"{count}_fractals"] = {
            "total_trades": len(count_trades),
            "wins": len(count_winners),
            "win_rate": round((len(count_winners) / len(count_trades) * 100), 2),
            "avg_pnl": round(sum(t["pnl_pct"] for t in count_trades) / len(count_trades), 3)
        }
    
    return stats


def print_statistics(stats):
    """Print formatted statistics"""
    print("\n" + "=" * 80)
    print("FRACTAL SUPPORT & RESISTANCE BACKTEST RESULTS")
    print("=" * 80)
    
    print(f"\nOverall Statistics:")
    print(f"  Total Trades: {stats.get('total_trades', 0)}")
    
    if stats.get("error"):
        print(f"  {stats['error']}")
        return
    
    print(f"  Win Rate: {stats['overall']['win_rate']}%")
    print(f"  Total PnL: {stats['overall']['total_pnl']}%")
    print(f"  Average PnL per Trade: {stats['overall']['avg_pnl']}%")
    print(f"  Average Winner: {stats['overall']['avg_winner']}%")
    print(f"  Average Loser: {stats['overall']['avg_loser']}%")
    print(f"  Average Duration: {stats['overall']['avg_duration_hours']} hours")
    
    print(f"\nBy Direction:")
    for direction, dir_stats in stats["by_direction"].items():
        if dir_stats:
            print(f"\n  {direction}:")
            print(f"    Trades: {dir_stats['total_trades']} (Wins: {dir_stats['wins']})")
            print(f"    Win Rate: {dir_stats['win_rate']}%")
            print(f"    Total PnL: {dir_stats['total_pnl']}%")
            print(f"    Avg PnL: {dir_stats['avg_pnl']}%")
    
    print(f"\nBy Zone Validation (Fractal Count):")
    for zone_label, zone_stats in stats["by_zone_count"].items():
        print(f"\n  {zone_label}:")
        print(f"    Trades: {zone_stats['total_trades']} (Wins: {zone_stats['wins']})")
        print(f"    Win Rate: {zone_stats['win_rate']}%")
        print(f"    Avg PnL: {zone_stats['avg_pnl']}%")
    
    print("\n" + "=" * 80)


def main():
    """Main backtest execution"""
    parser = argparse.ArgumentParser(description='Backtest Fractal Support & Resistance strategy')
    parser.add_argument('--quick', action='store_true', help='Quick test (20 symbols, 30 days)')
    parser.add_argument('--symbols', nargs='+', help='Specific symbols to test')
    parser.add_argument('--days', type=int, default=90, help='Number of days to backtest (default: 90)')
    parser.add_argument('--fractals-qty', type=int, default=2, help='Min fractals per zone (default: 2 for crypto)')
    parser.add_argument('--zone-type', choices=['Average', 'Extreme'], default='Extreme',
                       help='Zone placement type (default: Extreme for crypto)')
    
    args = parser.parse_args()
    
    log("=" * 80)
    log("Fractal Support & Resistance Strategy Backtest")
    log(f"Period: {args.days} days | Interval: 1 hour")
    log(f"Fractals Required: {args.fractals_qty} | Zone Type: {args.zone_type}")
    log("=" * 80)
    
    # Get symbols
    if args.symbols:
        symbols = [s.upper() if not s.endswith('USDT') else s.upper() for s in args.symbols]
        symbols = [s if s.endswith('USDT') else f"{s}USDT" for s in symbols]
    else:
        symbols = get_all_futures_symbols()
    
    if not symbols:
        log("No symbols found. Exiting.")
        return
    
    # Quick mode: limit symbols and days
    if args.quick:
        symbols = symbols[:20]
        args.days = 30
        log("Quick mode: Testing first 20 symbols with 30 days of data")
    
    log(f"\nBacktesting {len(symbols)} symbols...")
    log("This may take a while...\n")
    
    all_trades = []
    processed = 0
    
    # Process symbols in parallel
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(backtest_symbol, symbol, args.fractals_qty, args.zone_type, args.days): symbol
            for symbol in symbols
        }
        
        for future in as_completed(futures):
            symbol = futures[future]
            processed += 1
            
            try:
                trades = future.result()
                all_trades.extend(trades)
                
                if processed % 10 == 0:
                    log(f"Progress: {processed}/{len(symbols)} symbols processed, {len(all_trades)} trades so far")
                    
            except Exception as e:
                log(f"Error processing {symbol}: {e}")
    
    log(f"\nBacktest complete!")
    log(f"Processed {processed} symbols")
    log(f"Total trades found: {len(all_trades)}")
    
    # Save all trades
    with open(TRADES_FILE, "w", encoding="utf-8") as f:
        json.dump(all_trades, f, indent=2, ensure_ascii=False)
    log(f"\nTrades saved to: {TRADES_FILE}")
    
    # Analyze results
    stats = analyze_results(all_trades)
    stats["backtest_info"] = {
        "start_time": (datetime.now(timezone.utc) - timedelta(days=args.days)).isoformat(),
        "end_time": datetime.now(timezone.utc).isoformat(),
        "symbols_tested": len(symbols),
        "fractals_qty": args.fractals_qty,
        "zone_type": args.zone_type,
        "days": args.days
    }
    
    # Save statistics
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    log(f"Results saved to: {RESULTS_FILE}")
    
    # Print statistics
    print_statistics(stats)
    
    log("\nBacktest complete!")


if __name__ == "__main__":
    main()
