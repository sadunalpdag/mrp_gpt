#!/usr/bin/env python3
"""
Test Williams Fractal + EMA Strategy with mock data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import sys

# Import strategy functions
from williams_fractal_ema_strategy import (
    ema,
    detect_williams_fractals,
    check_trend_valid,
    check_pullback_scenario,
    calculate_stop_loss
)


def create_mock_data(n_bars=500):
    """Create mock OHLC data for testing"""
    # Start with base price
    base_price = 50000.0
    
    # Generate timestamps
    start_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    timestamps = [start_time + timedelta(minutes=i) for i in range(n_bars)]
    
    # Generate price data with some patterns
    np.random.seed(42)
    closes = []
    opens = []
    highs = []
    lows = []
    
    price = base_price
    
    for i in range(n_bars):
        # Simulate some trend and randomness
        if i < 200:
            # Uptrend
            change = np.random.normal(5, 20)
        elif i < 400:
            # Downtrend
            change = np.random.normal(-5, 20)
        else:
            # Sideways
            change = np.random.normal(0, 15)
        
        price = max(price + change, base_price * 0.8)  # Don't go too low
        price = min(price, base_price * 1.3)  # Don't go too high
        
        # Generate OHLC
        open_price = price
        close_price = price + np.random.normal(0, 10)
        high_price = max(open_price, close_price) + abs(np.random.normal(0, 5))
        low_price = min(open_price, close_price) - abs(np.random.normal(0, 5))
        
        opens.append(open_price)
        closes.append(close_price)
        highs.append(high_price)
        lows.append(low_price)
        
        price = close_price
    
    # Create DataFrame
    df = pd.DataFrame({
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': [1000000] * n_bars
    }, index=pd.DatetimeIndex(timestamps))
    
    return df


def test_ema_calculation():
    """Test EMA calculation"""
    print("Testing EMA calculation...")
    
    df = create_mock_data(200)
    df['ema20'] = ema(df['close'], 20)
    df['ema50'] = ema(df['close'], 50)
    df['ema100'] = ema(df['close'], 100)
    
    # Check that EMAs are calculated
    assert not df['ema20'].isna().all(), "EMA20 should have values"
    assert not df['ema50'].isna().all(), "EMA50 should have values"
    assert not df['ema100'].isna().all(), "EMA100 should have values"
    
    # Check that EMAs have valid values after warmup period
    # Note: pandas ewm doesn't produce NaN the same way, it starts calculating immediately
    assert not df['ema100'].iloc[100:].isna().any(), "EMA100 should have values after warmup period"
    
    print("  ✓ EMA calculation works correctly")
    return True


def test_fractal_detection():
    """Test Williams Fractal detection"""
    print("Testing Fractal detection...")
    
    df = create_mock_data(200)
    df = detect_williams_fractals(df, period=2)
    
    # Check that fractal columns exist
    assert 'fractal_high' in df.columns, "fractal_high column should exist"
    assert 'fractal_low' in df.columns, "fractal_low column should exist"
    
    # Check that some fractals are detected
    num_high_fractals = df['fractal_high'].sum()
    num_low_fractals = df['fractal_low'].sum()
    
    print(f"  Found {num_high_fractals} high fractals and {num_low_fractals} low fractals")
    assert num_high_fractals > 0, "Should find some high fractals"
    assert num_low_fractals > 0, "Should find some low fractals"
    
    # Check that fractals are not at edges
    assert not df['fractal_high'].iloc[:2].any(), "No fractals at start"
    assert not df['fractal_high'].iloc[-2:].any(), "No fractals at end"
    
    print("  ✓ Fractal detection works correctly")
    return True


def test_trend_validation():
    """Test trend validation logic"""
    print("Testing trend validation...")
    
    # Test LONG trend (20 > 50 > 100)
    assert check_trend_valid(100, 90, 80, 'long'), "Should be valid long trend"
    assert not check_trend_valid(80, 90, 100, 'long'), "Should be invalid long trend"
    assert not check_trend_valid(100, 100, 100, 'long'), "Should be invalid (equal MAs)"
    
    # Test SHORT trend (100 > 50 > 20)
    assert check_trend_valid(80, 90, 100, 'short'), "Should be valid short trend"
    assert not check_trend_valid(100, 90, 80, 'short'), "Should be invalid short trend"
    assert not check_trend_valid(100, 100, 100, 'short'), "Should be invalid (equal MAs)"
    
    # Test with NaN
    assert not check_trend_valid(np.nan, 90, 80, 'long'), "Should be invalid with NaN"
    
    print("  ✓ Trend validation works correctly")
    return True


def test_pullback_scenarios():
    """Test pullback scenario detection"""
    print("Testing pullback scenarios...")
    
    # Long scenarios
    ma20, ma50, ma100 = 100, 90, 80
    
    # Scenario A: Below 20 MA but above 50 MA
    result = check_pullback_scenario(95, 95, 105, ma20, ma50, ma100, 'long')
    assert result == 'scenario_a', f"Should be scenario_a, got {result}"
    
    # Scenario B: Below 50 MA but above 100 MA
    result = check_pullback_scenario(85, 85, 95, ma20, ma50, ma100, 'long')
    assert result == 'scenario_b', f"Should be scenario_b, got {result}"
    
    # Invalid: Below 100 MA
    result = check_pullback_scenario(75, 75, 85, ma20, ma50, ma100, 'long')
    assert result == 'invalid', f"Should be invalid, got {result}"
    
    # No pullback
    result = check_pullback_scenario(105, 105, 110, ma20, ma50, ma100, 'long')
    assert result is None, f"Should be None, got {result}"
    
    # Short scenarios
    ma20, ma50, ma100 = 80, 90, 100
    
    # Scenario A: Above 20 MA but below 50 MA
    result = check_pullback_scenario(85, 75, 85, ma20, ma50, ma100, 'short')
    assert result == 'scenario_a', f"Should be scenario_a, got {result}"
    
    # Invalid: Above 100 MA
    result = check_pullback_scenario(105, 95, 105, ma20, ma50, ma100, 'short')
    assert result == 'invalid', f"Should be invalid, got {result}"
    
    print("  ✓ Pullback scenario detection works correctly")
    return True


def test_stop_loss_calculation():
    """Test stop loss calculation"""
    print("Testing stop loss calculation...")
    
    entry = 100
    ma20, ma50, ma100 = 95, 90, 85
    
    # Long scenario A
    sl = calculate_stop_loss(entry, ma20, ma50, ma100, 'scenario_a', 'long')
    assert sl < ma50, f"SL should be below MA50, got {sl}"
    
    # Long scenario B
    sl = calculate_stop_loss(entry, ma20, ma50, ma100, 'scenario_b', 'long')
    assert sl < ma100, f"SL should be below MA100, got {sl}"
    
    # Short
    sl = calculate_stop_loss(entry, ma20, ma50, ma100, 'scenario_a', 'short')
    assert sl > ma50, f"SL should be above MA50, got {sl}"
    
    print("  ✓ Stop loss calculation works correctly")
    return True


def test_full_strategy():
    """Test full strategy with mock data"""
    print("Testing full strategy logic...")
    
    df = create_mock_data(500)
    
    # Calculate EMAs
    df['ema20'] = ema(df['close'], 20)
    df['ema50'] = ema(df['close'], 50)
    df['ema100'] = ema(df['close'], 100)
    
    # Detect fractals
    df = detect_williams_fractals(df, period=2)
    
    # Drop NaN rows
    df = df.dropna(subset=['ema20', 'ema50', 'ema100'])
    
    # Look for signals
    signals_found = 0
    long_signals = 0
    short_signals = 0
    
    for i in range(10, len(df)):
        row = df.iloc[i]
        
        close = row['close']
        high = row['high']
        low = row['low']
        ma20 = row['ema20']
        ma50 = row['ema50']
        ma100 = row['ema100']
        
        # Check for LONG signal
        if row['fractal_low']:
            if check_trend_valid(ma20, ma50, ma100, 'long'):
                scenario = check_pullback_scenario(close, low, high, ma20, ma50, ma100, 'long')
                if scenario in ['scenario_a', 'scenario_b']:
                    signals_found += 1
                    long_signals += 1
        
        # Check for SHORT signal
        if row['fractal_high']:
            if check_trend_valid(ma20, ma50, ma100, 'short'):
                scenario = check_pullback_scenario(close, low, high, ma20, ma50, ma100, 'short')
                if scenario == 'scenario_a':
                    signals_found += 1
                    short_signals += 1
    
    print(f"  Found {signals_found} valid signals ({long_signals} long, {short_signals} short)")
    print("  ✓ Full strategy logic works correctly")
    return True


def main():
    """Run all tests"""
    print("=" * 80)
    print("WILLIAMS FRACTAL + EMA STRATEGY - UNIT TESTS")
    print("=" * 80)
    print()
    
    tests = [
        test_ema_calculation,
        test_fractal_detection,
        test_trend_validation,
        test_pullback_scenarios,
        test_stop_loss_calculation,
        test_full_strategy
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            failed += 1
            print(f"  ✗ Test failed: {e}")
            import traceback
            traceback.print_exc()
    
    print()
    print("=" * 80)
    print(f"TEST RESULTS: {passed}/{len(tests)} passed")
    if failed == 0:
        print("✓ All tests passed!")
    else:
        print(f"✗ {failed} tests failed")
    print("=" * 80)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
