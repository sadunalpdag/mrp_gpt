#!/usr/bin/env python3
"""
Simple tests for institutional strategy components
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone

from supply_demand_zones import (
    identify_base_candles, is_strong_impulse, 
    detect_dbr_zones, detect_rbd_zones,
    is_price_in_zone
)
from candlestick_patterns import (
    is_hammer, is_shooting_star, is_doji,
    is_bullish_engulfing, is_bearish_engulfing,
    detect_bullish_confirmation, detect_bearish_confirmation
)
from zone_filters import (
    filter_candle_strength, filter_freshness,
    calculate_reward_risk_ratio, filter_reward_risk
)


def create_sample_data(num_bars=100):
    """Create sample OHLC data for testing."""
    dates = pd.date_range(start='2024-01-01', periods=num_bars, freq='5min', tz='UTC')
    
    # Create a simple price pattern
    prices = 100 + np.cumsum(np.random.randn(num_bars) * 0.5)
    
    data = {
        'open': prices + np.random.randn(num_bars) * 0.1,
        'high': prices + np.abs(np.random.randn(num_bars)) * 0.5,
        'low': prices - np.abs(np.random.randn(num_bars)) * 0.5,
        'close': prices + np.random.randn(num_bars) * 0.1,
        'volume': np.random.randint(1000, 10000, num_bars),
    }
    
    df = pd.DataFrame(data, index=dates)
    
    # Ensure high is highest and low is lowest
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)
    
    # Add close_time
    df['close_time'] = df.index + timedelta(minutes=5)
    
    return df


def test_zone_detection():
    """Test supply/demand zone detection."""
    print("Testing zone detection...")
    
    df = create_sample_data(200)
    
    # Test DBR detection
    dbr_zones = detect_dbr_zones(df, lookback=30)
    print(f"  ✓ DBR zones detected: {len(dbr_zones)}")
    
    # Test RBD detection
    rbd_zones = detect_rbd_zones(df, lookback=30)
    print(f"  ✓ RBD zones detected: {len(rbd_zones)}")
    
    # Test price in zone
    if dbr_zones:
        zone = dbr_zones[0]
        test_price = (zone['low'] + zone['high']) / 2
        assert is_price_in_zone(test_price, zone), "Price should be in zone"
        print(f"  ✓ is_price_in_zone works correctly")


def test_candlestick_patterns():
    """Test candlestick pattern recognition."""
    print("\nTesting candlestick patterns...")
    
    # Test hammer
    hammer = is_hammer(open_price=100, high=100.2, low=98, close=99.8)
    print(f"  ✓ Hammer detection: {hammer}")
    
    # Test shooting star
    shooting_star = is_shooting_star(open_price=100, high=102, low=99.8, close=100.2)
    print(f"  ✓ Shooting star detection: {shooting_star}")
    
    # Test doji
    doji = is_doji(open_price=100, high=100.5, low=99.5, close=100.05)
    print(f"  ✓ Doji detection: {doji}")
    
    # Test bullish engulfing
    bullish_engulf = is_bullish_engulfing(
        prev_open=100, prev_close=99,
        curr_open=98.5, curr_close=100.5
    )
    print(f"  ✓ Bullish engulfing detection: {bullish_engulf}")
    
    # Test bearish engulfing
    bearish_engulf = is_bearish_engulfing(
        prev_open=99, prev_close=100,
        curr_open=100.5, curr_close=98.5
    )
    print(f"  ✓ Bearish engulfing detection: {bearish_engulf}")


def test_zone_filters():
    """Test zone filtering system."""
    print("\nTesting zone filters...")
    
    # Test freshness filter
    zone_fresh = {'tested': 0}
    zone_used = {'tested': 2}
    
    assert filter_freshness(zone_fresh), "Fresh zone should pass"
    assert not filter_freshness(zone_used), "Used zone should fail"
    print(f"  ✓ Freshness filter works")
    
    # Test R:R calculation
    rr = calculate_reward_risk_ratio(entry_price=100, stop_price=99, target_price=102)
    expected_rr = 2.0
    assert abs(rr - expected_rr) < 0.1, f"R:R should be {expected_rr}, got {rr}"
    print(f"  ✓ R:R calculation: {rr:.2f}")
    
    # Test R:R filter
    assert filter_reward_risk(100, 99, 102, min_rr=2.0), "Should pass with 2:1 R:R"
    assert not filter_reward_risk(100, 99, 101, min_rr=2.0), "Should fail with 1:1 R:R"
    print(f"  ✓ R:R filter works")


def test_integration():
    """Test integrated workflow."""
    print("\nTesting integrated workflow...")
    
    df = create_sample_data(200)
    
    # Detect zones
    dbr_zones = detect_dbr_zones(df, lookback=30)
    
    if dbr_zones:
        zone = dbr_zones[0]
        print(f"  ✓ Zone detected: {zone['type']} at [{zone['low']:.2f}, {zone['high']:.2f}]")
        
        # Test confirmation detection on sample data
        for i in range(len(df) - 10, len(df)):
            confirmation = detect_bullish_confirmation(df, i)
            if confirmation:
                print(f"  ✓ Confirmation pattern found: {confirmation} at bar {i}")
                break
    
    print("  ✓ Integration test completed")


def main():
    """Run all tests."""
    print("=" * 60)
    print("INSTITUTIONAL STRATEGY - COMPONENT TESTS")
    print("=" * 60)
    
    try:
        test_zone_detection()
        test_candlestick_patterns()
        test_zone_filters()
        test_integration()
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
