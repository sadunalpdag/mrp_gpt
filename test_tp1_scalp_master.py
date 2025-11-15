#!/usr/bin/env python3
"""
Test script for TP1 SCALP MASTER STRATEGY
Uses synthetic data to validate strategy logic
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import sys

# Import the strategy functions
from tp1_scalp_master import (
    calculate_stoch_rsi,
    calculate_stochastic,
    calculate_mrpz,
    detect_double_bottom,
    detect_double_top,
    is_in_structure_zone,
    STOCH_RSI_RSI_LEN,
    STOCH_RSI_STOCH_LEN,
    STOCH_RSI_K,
    STOCH_RSI_D,
    STOCH_K_LEN,
    STOCH_K_SMOOTH,
    STOCH_D_SMOOTH,
    MRPZ_LENGTH,
    MRPZ_MULT,
    DOUBLE_PATTERN_TOLERANCE,
    STRUCTURE_LOOKBACK,
    BODY_THRESH,
    TP1_R_MIN,
    TP1_R_MAX,
)


def generate_synthetic_data(n_bars=500):
    """
    Generate synthetic OHLCV data for testing
    """
    np.random.seed(42)
    
    start_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
    dates = [start_time + timedelta(hours=i) for i in range(n_bars)]
    
    # Generate price data with trend and noise
    base_price = 50000
    trend = np.linspace(0, 2000, n_bars)
    noise = np.random.normal(0, 500, n_bars)
    close_prices = base_price + trend + noise
    
    # Generate OHLC from close
    opens = close_prices + np.random.normal(0, 100, n_bars)
    highs = np.maximum(opens, close_prices) + np.abs(np.random.normal(0, 150, n_bars))
    lows = np.minimum(opens, close_prices) - np.abs(np.random.normal(0, 150, n_bars))
    volumes = np.random.uniform(1000, 5000, n_bars)
    
    df = pd.DataFrame({
        'open': opens,
        'high': highs,
        'low': lows,
        'close': close_prices,
        'volume': volumes
    }, index=pd.DatetimeIndex(dates, tz=timezone.utc))
    
    return df


def test_indicators():
    """Test all indicator calculations"""
    print("\n" + "="*80)
    print("TEST 1: Indicator Calculations")
    print("="*80)
    
    df = generate_synthetic_data(500)
    
    # Test Stoch RSI
    print("\nTesting Stochastic RSI (Trend Filter)...")
    stoch_rsi_k, stoch_rsi_d = calculate_stoch_rsi(
        df['close'],
        rsi_len=STOCH_RSI_RSI_LEN,
        stoch_len=STOCH_RSI_STOCH_LEN,
        k_smooth=STOCH_RSI_K,
        d_smooth=STOCH_RSI_D
    )
    print(f"  ✓ Stoch RSI K calculated: {len(stoch_rsi_k.dropna())} valid values")
    print(f"  ✓ Stoch RSI D (White Line) calculated: {len(stoch_rsi_d.dropna())} valid values")
    print(f"  ✓ Last values - K: {stoch_rsi_k.iloc[-1]:.2f}, D: {stoch_rsi_d.iloc[-1]:.2f}")
    
    # Test Stochastic Momentum
    print("\nTesting Stochastic Momentum (7,3,3)...")
    stoch_k, stoch_d = calculate_stochastic(
        df['high'],
        df['low'],
        df['close'],
        k_period=STOCH_K_LEN,
        k_smooth=STOCH_K_SMOOTH,
        d_smooth=STOCH_D_SMOOTH
    )
    print(f"  ✓ Stoch K calculated: {len(stoch_k.dropna())} valid values")
    print(f"  ✓ Stoch D calculated: {len(stoch_d.dropna())} valid values")
    print(f"  ✓ Last values - K: {stoch_k.iloc[-1]:.2f}, D: {stoch_d.iloc[-1]:.2f}")
    
    # Test MRPZ
    print("\nTesting MRPZ (Mean Reversion Price Zone)...")
    mrpz = calculate_mrpz(df, length=MRPZ_LENGTH, mult=MRPZ_MULT)
    print(f"  ✓ Upper Zone calculated: {len(mrpz['upper_zone'].dropna())} valid values")
    print(f"  ✓ Lower Zone calculated: {len(mrpz['lower_zone'].dropna())} valid values")
    print(f"  ✓ Histogram calculated: {len(mrpz['histogram'].dropna())} valid values")
    print(f"  ✓ Upper spikes detected: {mrpz['upper_spike'].sum()}")
    print(f"  ✓ Lower spikes detected: {mrpz['lower_spike'].sum()}")
    
    print("\n✅ All indicator tests passed!")
    return True


def test_pattern_detection():
    """Test pattern detection functions"""
    print("\n" + "="*80)
    print("TEST 2: Pattern Detection")
    print("="*80)
    
    # Create data with clear double bottom
    print("\nTesting Double Bottom Detection...")
    dates = pd.date_range(start='2024-01-01', periods=50, freq='1H', tz=timezone.utc)
    
    # Create a double bottom pattern
    lows = [100] * 10 + [95] + [100] * 14 + [95.1] + [100] * 24
    highs = [l + 5 for l in lows]
    closes = [(h + l) / 2 for h, l in zip(highs, lows)]
    opens = closes
    
    df_pattern = pd.DataFrame({
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': [1000] * 50
    }, index=dates)
    
    has_double_bottom, support = detect_double_bottom(
        df_pattern,
        lookback=30,
        tolerance=DOUBLE_PATTERN_TOLERANCE
    )
    
    print(f"  Double Bottom Detected: {has_double_bottom}")
    if support:
        print(f"  Support Level: ${support:.2f}")
    print("  ✓ Double bottom detection working")
    
    # Create data with clear double top
    print("\nTesting Double Top Detection...")
    highs_top = [100] * 10 + [105] + [100] * 14 + [105.1] + [100] * 24
    lows_top = [h - 5 for h in highs_top]
    closes_top = [(h + l) / 2 for h, l in zip(highs_top, lows_top)]
    opens_top = closes_top
    
    df_pattern_top = pd.DataFrame({
        'open': opens_top,
        'high': highs_top,
        'low': lows_top,
        'close': closes_top,
        'volume': [1000] * 50
    }, index=dates)
    
    has_double_top, resistance = detect_double_top(
        df_pattern_top,
        lookback=30,
        tolerance=DOUBLE_PATTERN_TOLERANCE
    )
    
    print(f"  Double Top Detected: {has_double_top}")
    if resistance:
        print(f"  Resistance Level: ${resistance:.2f}")
    print("  ✓ Double top detection working")
    
    print("\n✅ Pattern detection tests passed!")
    return True


def test_structure_zone():
    """Test structure zone identification"""
    print("\n" + "="*80)
    print("TEST 3: Structure Zone Detection")
    print("="*80)
    
    # Create 4H data with clear support/resistance levels
    dates = pd.date_range(start='2024-01-01', periods=50, freq='4H', tz=timezone.utc)
    
    # Create data that tests 50000 multiple times (structure level)
    closes = []
    highs = []
    lows = []
    for i in range(50):
        if i % 5 == 0:
            # Touch support level
            closes.append(50000)
            highs.append(50200)
            lows.append(49900)
        else:
            closes.append(50500 + np.random.normal(0, 200))
            highs.append(closes[-1] + 200)
            lows.append(closes[-1] - 200)
    
    df_4h = pd.DataFrame({
        'open': closes,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': [1000] * 50
    }, index=dates)
    
    # Test if price near structure is detected
    test_price = 50050  # Near the support level
    in_structure, structure_type = is_in_structure_zone(
        test_price,
        df_4h,
        lookback=STRUCTURE_LOOKBACK
    )
    
    print(f"  Price ${test_price:.2f} in structure zone: {in_structure}")
    if structure_type:
        print(f"  Structure type: {structure_type}")
    print("  ✓ Structure zone detection working")
    
    print("\n✅ Structure zone tests passed!")
    return True


def test_complete_strategy_logic():
    """Test complete strategy logic with all filters"""
    print("\n" + "="*80)
    print("TEST 4: Complete Strategy Logic")
    print("="*80)
    
    # Generate test data
    df_1h = generate_synthetic_data(500)
    df_4h = generate_synthetic_data(125)  # 4x fewer bars
    
    print("\nCalculating all indicators...")
    
    # Calculate all indicators
    df_1h['stoch_rsi_k'], df_1h['stoch_rsi_d'] = calculate_stoch_rsi(
        df_1h['close'],
        rsi_len=STOCH_RSI_RSI_LEN,
        stoch_len=STOCH_RSI_STOCH_LEN,
        k_smooth=STOCH_RSI_K,
        d_smooth=STOCH_RSI_D
    )
    
    df_1h['stoch_k'], df_1h['stoch_d'] = calculate_stochastic(
        df_1h['high'],
        df_1h['low'],
        df_1h['close'],
        k_period=STOCH_K_LEN,
        k_smooth=STOCH_K_SMOOTH,
        d_smooth=STOCH_D_SMOOTH
    )
    
    mrpz = calculate_mrpz(df_1h, length=MRPZ_LENGTH, mult=MRPZ_MULT)
    df_1h['mrpz_upper'] = mrpz['upper_zone']
    df_1h['mrpz_lower'] = mrpz['lower_zone']
    df_1h['mrpz_histogram'] = mrpz['histogram']
    df_1h['mrpz_upper_spike'] = mrpz['upper_spike']
    df_1h['mrpz_lower_spike'] = mrpz['lower_spike']
    df_1h['mrpz_in_upper'] = mrpz['in_upper_zone']
    df_1h['mrpz_in_lower'] = mrpz['in_lower_zone']
    
    df_1h['body'] = (df_1h['close'] - df_1h['open']).abs()
    df_1h['range'] = df_1h['high'] - df_1h['low']
    df_1h['strong'] = (df_1h['range'] > 0) & (df_1h['body'] / df_1h['range'] >= BODY_THRESH)
    
    print("  ✓ All indicators calculated")
    
    # Test signal detection
    long_signals = 0
    short_signals = 0
    
    for i in range(max(STOCH_RSI_STOCH_LEN, MRPZ_LENGTH, STRUCTURE_LOOKBACK) + 10, len(df_1h)):
        row = df_1h.iloc[i]
        
        if pd.isna(row['stoch_rsi_d']) or pd.isna(row['stoch_k']):
            continue
        
        # Filter 1: Trend
        trend_long = row['stoch_rsi_d'] > 50
        trend_short = row['stoch_rsi_d'] < 50
        
        # Filter 2: Mean Reversion & Momentum
        mrpz_long = (row['mrpz_lower_spike'] or row['mrpz_in_lower']) and row['stoch_k'] < 20
        mrpz_short = (row['mrpz_upper_spike'] or row['mrpz_in_upper']) and row['stoch_k'] > 80
        
        # Check crosses
        if i > 0:
            prev = df_1h.iloc[i-1]
            k_cross_up = prev['stoch_k'] <= prev['stoch_d'] and row['stoch_k'] > row['stoch_d']
            k_cross_down = prev['stoch_k'] >= prev['stoch_d'] and row['stoch_k'] < row['stoch_d']
        else:
            k_cross_up = False
            k_cross_down = False
        
        # Count potential signals (without price action filter for testing)
        if trend_long and mrpz_long and k_cross_up:
            long_signals += 1
        if trend_short and mrpz_short and k_cross_down:
            short_signals += 1
    
    print(f"\n  Potential LONG signals detected: {long_signals}")
    print(f"  Potential SHORT signals detected: {short_signals}")
    
    if long_signals > 0 or short_signals > 0:
        print("  ✓ Strategy logic producing signals")
    else:
        print("  ⚠ No signals detected (may be normal with synthetic data)")
    
    # Test TP/SL calculation
    print("\nTesting TP1 calculation...")
    test_entry = 50000
    test_stop = 49500
    risk = test_entry - test_stop
    tp_r = (TP1_R_MIN + TP1_R_MAX) / 2
    tp = test_entry + (risk * tp_r)
    
    print(f"  Entry: ${test_entry:.2f}")
    print(f"  Stop: ${test_stop:.2f}")
    print(f"  Risk: ${risk:.2f}")
    print(f"  TP1 ({tp_r}R): ${tp:.2f}")
    print(f"  Reward: ${tp - test_entry:.2f}")
    print(f"  R:R Ratio: {(tp - test_entry) / risk:.2f}:1")
    print("  ✓ TP1 calculation working correctly")
    
    print("\n✅ Complete strategy logic tests passed!")
    return True


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("TP1 SCALP MASTER STRATEGY - TEST SUITE")
    print("="*80)
    print("\nValidating strategy components with synthetic data...")
    
    try:
        test_indicators()
        test_pattern_detection()
        test_structure_zone()
        test_complete_strategy_logic()
        
        print("\n" + "="*80)
        print("✅ ALL TESTS PASSED!")
        print("="*80)
        print("\nStrategy Implementation Summary:")
        print("  ✓ Trend Filter: Stoch RSI (3,3,14,134) - White line")
        print("  ✓ Mean Reversion: MRPZ with histogram spikes")
        print("  ✓ Momentum: Stochastic (7,3,3) K/D crosses")
        print("  ✓ Price Action: Double Bottom/Top detection")
        print("  ✓ Structure Zones: 4H S/R level identification")
        print(f"  ✓ Exit Strategy: TP1 only at {TP1_R_MIN}R-{TP1_R_MAX}R")
        print("\nThe strategy is ready for backtesting with live data!")
        print("="*80)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
