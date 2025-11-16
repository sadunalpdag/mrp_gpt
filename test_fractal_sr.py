#!/usr/bin/env python3
"""
Test Fractal Support & Resistance Strategy with synthetic data
"""

import sys
import numpy as np
from datetime import datetime, timezone

# Import functions from fractal_sr
from fractal_sr import (
    detect_fractals,
    calculate_atr,
    cluster_fractals,
    validate_zones,
    is_price_in_zone,
    has_price_moved_away,
    is_retest_with_wick,
    generate_trade_signal,
    simulate_trade
)


def create_synthetic_data(num_bars=500):
    """
    Create synthetic price data with support/resistance levels
    """
    np.random.seed(42)
    
    # Start price
    base_price = 100.0
    
    # Generate price movements with support at 95 and resistance at 105
    prices = [base_price]
    for i in range(num_bars - 1):
        # Random walk with mean reversion to support/resistance
        change = np.random.randn() * 0.5
        new_price = prices[-1] + change
        
        # Add support level at 95
        if new_price < 95:
            new_price = 95 + abs(np.random.randn() * 0.3)
        
        # Add resistance level at 105
        if new_price > 105:
            new_price = 105 - abs(np.random.randn() * 0.3)
        
        prices.append(new_price)
    
    # Create OHLC data
    klines = []
    for i, price in enumerate(prices):
        timestamp = int(datetime(2024, 1, 1).timestamp() * 1000) + (i * 3600000)  # Hourly
        
        # Create realistic OHLC with some variation
        open_price = price + np.random.randn() * 0.1
        close_price = price + np.random.randn() * 0.1
        high_price = max(open_price, close_price) + abs(np.random.randn() * 0.2)
        low_price = min(open_price, close_price) - abs(np.random.randn() * 0.2)
        
        klines.append([
            timestamp,  # Open time
            f"{open_price:.2f}",  # Open
            f"{high_price:.2f}",  # High
            f"{low_price:.2f}",  # Low
            f"{close_price:.2f}",  # Close
            "1000",  # Volume
            timestamp + 3599999,  # Close time
            "100000",  # Quote asset volume
            100,  # Number of trades
            "500",  # Taker buy base volume
            "50000",  # Taker buy quote volume
            "0"  # Ignore
        ])
    
    return klines


def test_fractal_detection():
    """Test fractal detection"""
    print("\n=== Testing Fractal Detection ===")
    
    klines = create_synthetic_data(200)
    closes = [float(k[4]) for k in klines]
    highs = [float(k[2]) for k in klines]
    lows = [float(k[3]) for k in klines]
    
    fractal_highs, fractal_lows = detect_fractals(highs, lows, closes, fractal_period=5)
    
    print(f"✓ Detected {len(fractal_highs)} resistance fractals")
    print(f"✓ Detected {len(fractal_lows)} support fractals")
    
    # Show first few fractals
    if fractal_highs:
        print(f"  First 3 resistance fractals: {fractal_highs[:3]}")
    if fractal_lows:
        print(f"  First 3 support fractals: {fractal_lows[:3]}")
    
    assert len(fractal_highs) > 0, "Should detect some resistance fractals"
    assert len(fractal_lows) > 0, "Should detect some support fractals"
    
    return True


def test_atr_calculation():
    """Test ATR calculation"""
    print("\n=== Testing ATR Calculation ===")
    
    klines = create_synthetic_data(100)
    closes = [float(k[4]) for k in klines]
    highs = [float(k[2]) for k in klines]
    lows = [float(k[3]) for k in klines]
    
    atr_values = calculate_atr(highs, lows, closes, period=14)
    
    print(f"✓ Calculated {len(atr_values)} ATR values")
    print(f"  First ATR: {atr_values[0]:.4f}")
    print(f"  Last ATR: {atr_values[-1]:.4f}")
    print(f"  Avg ATR: {np.mean(atr_values):.4f}")
    
    assert len(atr_values) == len(closes), "ATR values should match closes length"
    assert all(v >= 0 for v in atr_values), "ATR values should be non-negative"
    
    return True


def test_zone_clustering():
    """Test fractal clustering into zones"""
    print("\n=== Testing Zone Clustering ===")
    
    klines = create_synthetic_data(300)
    closes = [float(k[4]) for k in klines]
    highs = [float(k[2]) for k in klines]
    lows = [float(k[3]) for k in klines]
    
    atr_values = calculate_atr(highs, lows, closes, period=14)
    fractal_highs, fractal_lows = detect_fractals(highs, lows, closes, fractal_period=5)
    
    # Test with Extreme type (crypto)
    resistance_zones = cluster_fractals(fractal_highs, atr_values, closes, 
                                       cluster_tolerance=1.5, zone_type='Extreme')
    support_zones = cluster_fractals(fractal_lows, atr_values, closes, 
                                    cluster_tolerance=1.5, zone_type='Extreme')
    
    print(f"✓ Created {len(resistance_zones)} resistance zones")
    print(f"✓ Created {len(support_zones)} support zones")
    
    if resistance_zones:
        print(f"  Sample resistance zone: price={resistance_zones[0]['price']:.2f}, "
              f"count={resistance_zones[0]['count']}")
    if support_zones:
        print(f"  Sample support zone: price={support_zones[0]['price']:.2f}, "
              f"count={support_zones[0]['count']}")
    
    assert len(resistance_zones) > 0, "Should create some resistance zones"
    assert len(support_zones) > 0, "Should create some support zones"
    
    return True


def test_zone_validation():
    """Test zone validation"""
    print("\n=== Testing Zone Validation ===")
    
    klines = create_synthetic_data(300)
    closes = [float(k[4]) for k in klines]
    highs = [float(k[2]) for k in klines]
    lows = [float(k[3]) for k in klines]
    
    atr_values = calculate_atr(highs, lows, closes, period=14)
    fractal_highs, fractal_lows = detect_fractals(highs, lows, closes, fractal_period=5)
    
    resistance_zones = cluster_fractals(fractal_highs, atr_values, closes, 
                                       cluster_tolerance=1.5, zone_type='Extreme')
    support_zones = cluster_fractals(fractal_lows, atr_values, closes, 
                                    cluster_tolerance=1.5, zone_type='Extreme')
    
    # Test with min 2 fractals (crypto)
    validated_resistance = validate_zones(resistance_zones, min_fractals=2)
    validated_support = validate_zones(support_zones, min_fractals=2)
    
    print(f"✓ Validated {len(validated_resistance)}/{len(resistance_zones)} resistance zones (min 2 fractals)")
    print(f"✓ Validated {len(validated_support)}/{len(support_zones)} support zones (min 2 fractals)")
    
    # Test with min 3 fractals (forex/stocks)
    validated_resistance_3 = validate_zones(resistance_zones, min_fractals=3)
    validated_support_3 = validate_zones(support_zones, min_fractals=3)
    
    print(f"✓ Validated {len(validated_resistance_3)}/{len(resistance_zones)} resistance zones (min 3 fractals)")
    print(f"✓ Validated {len(validated_support_3)}/{len(support_zones)} support zones (min 3 fractals)")
    
    # Stricter validation should give fewer zones
    assert len(validated_resistance_3) <= len(validated_resistance), "Stricter validation should give fewer zones"
    assert len(validated_support_3) <= len(validated_support), "Stricter validation should give fewer zones"
    
    return True


def test_trade_signal_generation():
    """Test trade signal generation"""
    print("\n=== Testing Trade Signal Generation ===")
    
    klines = create_synthetic_data(300)
    closes = [float(k[4]) for k in klines]
    highs = [float(k[2]) for k in klines]
    lows = [float(k[3]) for k in klines]
    
    atr_values = calculate_atr(highs, lows, closes, period=14)
    fractal_highs, fractal_lows = detect_fractals(highs, lows, closes, fractal_period=5)
    
    resistance_zones = cluster_fractals(fractal_highs, atr_values, closes, 
                                       cluster_tolerance=1.5, zone_type='Extreme')
    support_zones = cluster_fractals(fractal_lows, atr_values, closes, 
                                    cluster_tolerance=1.5, zone_type='Extreme')
    
    resistance_zones = validate_zones(resistance_zones, min_fractals=2)
    support_zones = validate_zones(support_zones, min_fractals=2)
    
    # Scan for signals
    signals_found = 0
    for i in range(100, len(klines) - 10):
        signal = generate_trade_signal(
            "TESTUSDT", klines, support_zones, resistance_zones, i,
            min_bars_away=8, wait_bars_confirm=5
        )
        if signal:
            signals_found += 1
            if signals_found == 1:  # Show first signal
                print(f"✓ Found signal: {signal['direction']} at bar {i}")
                print(f"  Entry: {signal['entry']:.2f}, Stop: {signal['stop']:.2f}, TP: {signal['tp']:.2f}")
                print(f"  Zone: {signal['zone_type']} with {signal['zone_count']} fractals")
    
    print(f"✓ Generated {signals_found} trade signals")
    
    return True


def test_full_backtest():
    """Test full backtest with synthetic data"""
    print("\n=== Testing Full Backtest ===")
    
    klines = create_synthetic_data(500)
    closes = [float(k[4]) for k in klines]
    highs = [float(k[2]) for k in klines]
    lows = [float(k[3]) for k in klines]
    
    # Run full backtest logic
    atr_values = calculate_atr(highs, lows, closes, period=14)
    fractal_highs, fractal_lows = detect_fractals(highs, lows, closes, fractal_period=5)
    
    resistance_zones = cluster_fractals(fractal_highs, atr_values, closes, 
                                       cluster_tolerance=1.5, zone_type='Extreme')
    support_zones = cluster_fractals(fractal_lows, atr_values, closes, 
                                    cluster_tolerance=1.5, zone_type='Extreme')
    
    resistance_zones = validate_zones(resistance_zones, min_fractals=2)
    support_zones = validate_zones(support_zones, min_fractals=2)
    
    trades = []
    for i in range(100, len(klines) - 10):
        signal = generate_trade_signal(
            "TESTUSDT", klines, support_zones, resistance_zones, i,
            min_bars_away=8, wait_bars_confirm=5
        )
        
        if signal:
            trade_result = simulate_trade(signal, klines, i)
            if trade_result:
                trades.append(trade_result)
    
    print(f"✓ Completed backtest with {len(trades)} trades")
    
    if trades:
        winners = [t for t in trades if t['exit_reason'] == 'TP']
        win_rate = len(winners) / len(trades) * 100
        avg_pnl = sum(t['pnl_pct'] for t in trades) / len(trades)
        
        print(f"  Win Rate: {win_rate:.2f}%")
        print(f"  Avg PnL: {avg_pnl:.3f}%")
        print(f"  Total PnL: {sum(t['pnl_pct'] for t in trades):.2f}%")
    
    return True


def run_all_tests():
    """Run all tests"""
    print("=" * 80)
    print("FRACTAL SUPPORT & RESISTANCE STRATEGY TESTS")
    print("=" * 80)
    
    tests = [
        ("Fractal Detection", test_fractal_detection),
        ("ATR Calculation", test_atr_calculation),
        ("Zone Clustering", test_zone_clustering),
        ("Zone Validation", test_zone_validation),
        ("Trade Signal Generation", test_trade_signal_generation),
        ("Full Backtest", test_full_backtest),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✓ {test_name} PASSED")
        except Exception as e:
            failed += 1
            print(f"✗ {test_name} FAILED: {e}")
    
    print("\n" + "=" * 80)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 80)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
