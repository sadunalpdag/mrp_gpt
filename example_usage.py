#!/usr/bin/env python3
"""
Example Usage of Fractal Support & Resistance Strategy

This script demonstrates the basic workflow of the strategy:
1. Load/generate price data
2. Detect fractals
3. Cluster into zones
4. Validate zones
5. Generate trade signals
6. Simulate trades
"""

import numpy as np
from datetime import datetime, timezone
from fractal_sr import (
    detect_fractals,
    calculate_atr,
    cluster_fractals,
    validate_zones,
    generate_trade_signal,
    simulate_trade
)


def create_example_data():
    """Create simple example price data"""
    # Price oscillating between 95-105 with support/resistance
    np.random.seed(123)
    
    prices = []
    base = 100.0
    
    # Create 200 bars with mean reversion
    for i in range(200):
        noise = np.random.randn() * 0.3
        
        # Trend up first 50 bars
        if i < 50:
            base += 0.05
        # Range 50-150
        elif i < 150:
            if base > 105:
                base -= 0.2
            elif base < 95:
                base += 0.2
        # Trend down last 50 bars
        else:
            base -= 0.05
        
        price = base + noise
        prices.append(price)
    
    # Convert to OHLC format
    klines = []
    timestamp_start = int(datetime(2024, 1, 1).timestamp() * 1000)
    
    for i, close in enumerate(prices):
        timestamp = timestamp_start + (i * 3600000)  # Hourly
        
        open_price = close + np.random.randn() * 0.1
        high = max(open_price, close) + abs(np.random.randn() * 0.2)
        low = min(open_price, close) - abs(np.random.randn() * 0.2)
        
        klines.append([
            timestamp, f"{open_price:.2f}", f"{high:.2f}", 
            f"{low:.2f}", f"{close:.2f}", "1000",
            timestamp + 3599999, "100000", 100, "500", "50000", "0"
        ])
    
    return klines


def main():
    """Run example demonstration"""
    print("=" * 80)
    print("FRACTAL SUPPORT & RESISTANCE STRATEGY - EXAMPLE")
    print("=" * 80)
    
    # 1. Generate example data
    print("\n[Step 1] Generating example price data...")
    klines = create_example_data()
    print(f"✓ Created {len(klines)} hourly bars")
    
    # Extract prices
    closes = [float(k[4]) for k in klines]
    highs = [float(k[2]) for k in klines]
    lows = [float(k[3]) for k in klines]
    
    print(f"  Price range: {min(closes):.2f} - {max(closes):.2f}")
    
    # 2. Calculate ATR
    print("\n[Step 2] Calculating ATR (Average True Range)...")
    atr_values = calculate_atr(highs, lows, closes, period=14)
    print(f"✓ ATR calculated for {len(atr_values)} bars")
    print(f"  Current ATR: {atr_values[-1]:.4f}")
    print(f"  Average ATR: {np.mean(atr_values):.4f}")
    
    # 3. Detect fractals
    print("\n[Step 3] Detecting fractals (swing highs and lows)...")
    fractal_highs, fractal_lows = detect_fractals(highs, lows, closes, fractal_period=5)
    print(f"✓ Found {len(fractal_highs)} resistance fractals (swing highs)")
    print(f"✓ Found {len(fractal_lows)} support fractals (swing lows)")
    
    # Show some examples
    if fractal_highs:
        print(f"  Example resistance fractal: bar {fractal_highs[0][0]}, price {fractal_highs[0][1]:.2f}")
    if fractal_lows:
        print(f"  Example support fractal: bar {fractal_lows[0][0]}, price {fractal_lows[0][1]:.2f}")
    
    # 4. Cluster fractals into zones
    print("\n[Step 4] Clustering fractals into zones...")
    resistance_zones = cluster_fractals(
        fractal_highs, atr_values, closes, 
        cluster_tolerance=1.5, zone_type='Extreme'
    )
    support_zones = cluster_fractals(
        fractal_lows, atr_values, closes, 
        cluster_tolerance=1.5, zone_type='Extreme'
    )
    
    print(f"✓ Created {len(resistance_zones)} resistance zones")
    print(f"✓ Created {len(support_zones)} support zones")
    
    # Show zone details
    if resistance_zones:
        zone = resistance_zones[0]
        print(f"\n  Example Resistance Zone:")
        print(f"    Price: {zone['price']:.2f}")
        print(f"    Range: {zone['lower']:.2f} - {zone['upper']:.2f}")
        print(f"    Fractals: {zone['count']}")
        print(f"    Bars: {zone['first_bar']} to {zone['last_bar']}")
    
    if support_zones:
        zone = support_zones[0]
        print(f"\n  Example Support Zone:")
        print(f"    Price: {zone['price']:.2f}")
        print(f"    Range: {zone['lower']:.2f} - {zone['upper']:.2f}")
        print(f"    Fractals: {zone['count']}")
        print(f"    Bars: {zone['first_bar']} to {zone['last_bar']}")
    
    # 5. Validate zones (minimum 2 fractals for crypto)
    print("\n[Step 5] Validating zones (min 2 fractals for crypto)...")
    resistance_zones = validate_zones(resistance_zones, min_fractals=2)
    support_zones = validate_zones(support_zones, min_fractals=2)
    
    print(f"✓ {len(resistance_zones)} validated resistance zones")
    print(f"✓ {len(support_zones)} validated support zones")
    
    # 6. Generate trade signals
    print("\n[Step 6] Scanning for trade signals...")
    signals_found = 0
    trades_completed = 0
    
    for i in range(100, len(klines) - 10):
        signal = generate_trade_signal(
            "EXAMPLEUSDT", klines, support_zones, resistance_zones, i,
            min_bars_away=8, wait_bars_confirm=5
        )
        
        if signal:
            signals_found += 1
            
            # Simulate the trade
            trade_result = simulate_trade(signal, klines, i)
            
            if trade_result:
                trades_completed += 1
                
                # Print first few trades
                if trades_completed <= 3:
                    print(f"\n  Trade #{trades_completed}:")
                    print(f"    Direction: {trade_result['direction']}")
                    print(f"    Entry: {trade_result['entry_price']:.2f}")
                    print(f"    Exit: {trade_result['exit_price']:.2f}")
                    print(f"    Reason: {trade_result['exit_reason']}")
                    print(f"    P&L: {trade_result['pnl_pct']:.2f}%")
                    print(f"    Duration: {trade_result['duration_hours']:.1f} hours")
                    print(f"    Zone: {trade_result['zone_type']} with {trade_result['zone_count']} fractals")
    
    print(f"\n✓ Generated {signals_found} signals")
    print(f"✓ Completed {trades_completed} trades")
    
    # 7. Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total Bars Analyzed: {len(klines)}")
    print(f"Fractals Detected: {len(fractal_highs)} resistance + {len(fractal_lows)} support")
    print(f"Zones Created: {len(resistance_zones)} resistance + {len(support_zones)} support")
    print(f"Trade Signals: {signals_found}")
    print(f"Completed Trades: {trades_completed}")
    
    print("\n" + "=" * 80)
    print("This example demonstrates the complete workflow:")
    print("1. Price data (real or synthetic)")
    print("2. ATR calculation for volatility measurement")
    print("3. Fractal detection (swing highs/lows)")
    print("4. Zone clustering (grouping nearby fractals)")
    print("5. Zone validation (minimum fractal count)")
    print("6. Trade signal generation (entry rules)")
    print("7. Trade simulation (TP/SL execution)")
    print("\nFor real backtesting, use: python3 fractal_sr.py --quick")
    print("=" * 80)


if __name__ == "__main__":
    main()
