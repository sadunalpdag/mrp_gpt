#!/usr/bin/env python3
"""
Zone Filtering System
Applies 4 filters to validate supply/demand zones:
1. Candle Strength
2. Freshness (first retest only)
3. Break of Structure (BOS)
4. Reward:Risk >= 2:1
"""

import pandas as pd
import numpy as np


def filter_candle_strength(df, zone, strength_threshold=0.6):
    """
    Filter 1: Check if candles leaving the zone are strong.
    
    Strong candles have:
    - Large same-color candles
    - Small wicks
    - Not choppy or mixed colors
    
    Args:
        df: DataFrame with OHLC data
        zone: Zone dict
        strength_threshold: Minimum body/range ratio for strength
    
    Returns:
        bool: True if zone passes strength filter
    """
    # Get candles after the zone (impulse move)
    base_end_idx = zone['base_end_idx']
    
    # Check next 3-5 candles for strength
    impulse_start = base_end_idx
    impulse_end = min(base_end_idx + 5, len(df))
    
    if impulse_end - impulse_start < 3:
        return False
    
    impulse_candles = df.iloc[impulse_start:impulse_end]
    
    # Calculate body and range
    body = (impulse_candles['close'] - impulse_candles['open']).abs()
    candle_range = impulse_candles['high'] - impulse_candles['low']
    
    # Avoid division by zero
    body_ratio = body / candle_range.replace(0, 1)
    
    # Check if most candles are strong
    strong_count = (body_ratio >= strength_threshold).sum()
    
    # At least 60% of candles should be strong
    if strong_count < len(impulse_candles) * 0.6:
        return False
    
    # Check for consistent direction (not choppy)
    if zone['direction'] == 'demand':
        # For demand zones, expect bullish candles
        bullish_count = (impulse_candles['close'] > impulse_candles['open']).sum()
        return bullish_count >= len(impulse_candles) * 0.6
    else:
        # For supply zones, expect bearish candles
        bearish_count = (impulse_candles['close'] < impulse_candles['open']).sum()
        return bearish_count >= len(impulse_candles) * 0.6


def filter_freshness(zone, max_tests=1):
    """
    Filter 2: Check if zone is fresh (first retest only).
    
    First retest is the strongest. If price already tapped the zone,
    institutional orders are likely filled and edge is gone.
    
    Args:
        zone: Zone dict
        max_tests: Maximum number of retests allowed
    
    Returns:
        bool: True if zone is fresh enough
    """
    return zone.get('tested', 0) < max_tests


def filter_break_of_structure(df, zone, lookback=20):
    """
    Filter 3: Check if move out of zone broke a prior high/low.
    
    The impulse move should break structure (sweep liquidity).
    
    Args:
        df: DataFrame with OHLC data
        zone: Zone dict
        lookback: How many bars to look back for structure
    
    Returns:
        bool: True if structure was broken
    """
    base_end_idx = zone['base_end_idx']
    
    if base_end_idx < lookback or base_end_idx >= len(df) - 5:
        return False
    
    # Get the impulse move (candles after zone)
    impulse_end_idx = min(base_end_idx + 5, len(df))
    impulse_high = df['high'].iloc[base_end_idx:impulse_end_idx].max()
    impulse_low = df['low'].iloc[base_end_idx:impulse_end_idx].min()
    
    # Get prior structure (before the zone)
    structure_start = max(0, zone['base_start_idx'] - lookback)
    structure_end = zone['base_start_idx']
    
    if structure_end <= structure_start:
        return False
    
    prior_high = df['high'].iloc[structure_start:structure_end].max()
    prior_low = df['low'].iloc[structure_start:structure_end].min()
    
    # Check if structure was broken
    if zone['direction'] == 'demand':
        # For demand zones, expect break of prior high
        return impulse_high > prior_high
    else:
        # For supply zones, expect break of prior low
        return impulse_low < prior_low


def calculate_reward_risk_ratio(entry_price, stop_price, target_price):
    """
    Calculate Reward:Risk ratio.
    
    Args:
        entry_price: Entry price
        stop_price: Stop loss price
        target_price: Take profit price
    
    Returns:
        float: R:R ratio
    """
    risk = abs(entry_price - stop_price)
    reward = abs(target_price - entry_price)
    
    if risk == 0:
        return 0
    
    return reward / risk


def filter_reward_risk(entry_price, stop_price, target_price, min_rr=2.0):
    """
    Filter 4: Check if Reward:Risk ratio is >= 2:1.
    
    Args:
        entry_price: Entry price
        stop_price: Stop loss price
        target_price: Take profit price
        min_rr: Minimum R:R ratio required
    
    Returns:
        bool: True if R:R is acceptable
    """
    rr_ratio = calculate_reward_risk_ratio(entry_price, stop_price, target_price)
    return rr_ratio >= min_rr


def find_logical_target(df, current_idx, zone_direction, entry_price):
    """
    Find logical target based on structure levels.
    
    For longs: nearest resistance / prior swing high / opposing supply
    For shorts: nearest support / prior swing low / opposing demand
    
    Args:
        df: DataFrame with OHLC data
        current_idx: Current bar index
        zone_direction: 'demand' or 'supply'
        entry_price: Entry price
    
    Returns:
        float: Target price
    """
    lookback = 50
    lookforward = 20
    
    start_idx = max(0, current_idx - lookback)
    end_idx = min(len(df), current_idx + lookforward)
    
    if zone_direction == 'demand':
        # For longs, find resistance above
        future_highs = df['high'].iloc[current_idx:end_idx]
        if len(future_highs) > 0:
            resistance = future_highs.max()
            if resistance > entry_price:
                return resistance
        
        # If no clear resistance, use ATR-based target
        atr = calculate_atr(df, current_idx)
        return entry_price + (atr * 3)
    else:
        # For shorts, find support below
        future_lows = df['low'].iloc[current_idx:end_idx]
        if len(future_lows) > 0:
            support = future_lows.min()
            if support < entry_price:
                return support
        
        # If no clear support, use ATR-based target
        atr = calculate_atr(df, current_idx)
        return entry_price - (atr * 3)


def calculate_atr(df, idx, period=14):
    """
    Calculate Average True Range at given index.
    
    Args:
        df: DataFrame with OHLC data
        idx: Current index
        period: ATR period
    
    Returns:
        float: ATR value
    """
    if idx < period:
        period = max(1, idx)
    
    start_idx = max(0, idx - period)
    candles = df.iloc[start_idx:idx+1]
    
    if len(candles) < 2:
        return (df.iloc[idx]['high'] - df.iloc[idx]['low'])
    
    # True Range = max(high-low, abs(high-prev_close), abs(low-prev_close))
    high_low = candles['high'] - candles['low']
    high_close = (candles['high'] - candles['close'].shift(1)).abs()
    low_close = (candles['low'] - candles['close'].shift(1)).abs()
    
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.mean()
    
    return atr if not np.isnan(atr) else (df.iloc[idx]['high'] - df.iloc[idx]['low'])


def apply_all_filters(df, zone, current_idx, entry_price, stop_price, target_price):
    """
    Apply all 4 filters to a zone.
    
    Args:
        df: DataFrame with OHLC data
        zone: Zone dict
        current_idx: Current bar index
        entry_price: Proposed entry price
        stop_price: Proposed stop loss
        target_price: Proposed target price
    
    Returns:
        dict: Filter results with pass/fail for each filter
    """
    results = {
        'strength': False,
        'freshness': False,
        'bos': False,
        'reward_risk': False,
        'passed': False
    }
    
    # Filter 1: Candle Strength
    results['strength'] = filter_candle_strength(df, zone)
    
    # Filter 2: Freshness
    results['freshness'] = filter_freshness(zone)
    
    # Filter 3: Break of Structure
    results['bos'] = filter_break_of_structure(df, zone)
    
    # Filter 4: Reward:Risk
    results['reward_risk'] = filter_reward_risk(entry_price, stop_price, target_price)
    
    # Overall pass
    results['passed'] = all([
        results['strength'],
        results['freshness'],
        results['bos'],
        results['reward_risk']
    ])
    
    return results
