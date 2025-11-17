#!/usr/bin/env python3
"""
Supply/Demand Zone Detection
Identifies institutional footprints: DBR, RBD, RBR, DBD patterns
"""

import pandas as pd
import numpy as np


def identify_base_candles(df, start_idx, end_idx, threshold=0.3):
    """
    Identify base candles - candles with small bodies where institutions accumulate/distribute.
    
    Args:
        df: DataFrame with OHLC data
        start_idx: Start index of potential base
        end_idx: End index of potential base
        threshold: Maximum body/range ratio for base candles
    
    Returns:
        bool: True if candles form a valid base
    """
    if start_idx >= end_idx:
        return False
    
    base_candles = df.iloc[start_idx:end_idx]
    
    # Calculate body and range
    body = (base_candles['close'] - base_candles['open']).abs()
    range_candle = base_candles['high'] - base_candles['low']
    
    # Base candles should have small bodies relative to range
    body_ratio = body / range_candle.replace(0, 1)
    
    # Most candles in base should have small bodies
    small_body_count = (body_ratio <= threshold).sum()
    
    return small_body_count >= len(base_candles) * 0.6


def is_strong_impulse(df, start_idx, end_idx, direction='up'):
    """
    Check if candles form a strong impulsive move.
    
    Args:
        df: DataFrame with OHLC data
        start_idx: Start index of impulse
        end_idx: End index of impulse
        direction: 'up' or 'down'
    
    Returns:
        bool: True if strong impulse
    """
    if start_idx >= end_idx:
        return False
    
    impulse_candles = df.iloc[start_idx:end_idx]
    
    # Calculate body and range
    body = impulse_candles['close'] - impulse_candles['open']
    range_candle = impulse_candles['high'] - impulse_candles['low']
    
    if direction == 'up':
        # For upward impulse, bodies should be positive and large
        positive_bodies = (body > 0).sum()
        body_ratio = body / range_candle.replace(0, 1)
        strong_candles = (body_ratio >= 0.6).sum()
        
        return positive_bodies >= len(impulse_candles) * 0.7 and strong_candles >= len(impulse_candles) * 0.5
    else:
        # For downward impulse, bodies should be negative and large
        negative_bodies = (body < 0).sum()
        body_ratio = body.abs() / range_candle.replace(0, 1)
        strong_candles = (body_ratio >= 0.6).sum()
        
        return negative_bodies >= len(impulse_candles) * 0.7 and strong_candles >= len(impulse_candles) * 0.5


def detect_dbr_zones(df, lookback=50, min_base_size=2, max_base_size=10):
    """
    Detect Drop-Base-Rally (DBR) zones - Bullish reversal demand zones.
    
    Args:
        df: DataFrame with OHLC data
        lookback: How many bars to look back
        min_base_size: Minimum number of candles in base
        max_base_size: Maximum number of candles in base
    
    Returns:
        list: List of detected zones with [low, high, base_start, base_end, type]
    """
    zones = []
    
    for i in range(lookback, len(df)):
        # Look for pattern: Drop -> Base -> Rally
        for base_size in range(min_base_size, max_base_size + 1):
            if i < lookback + base_size + 5:
                continue
            
            # Define indices
            drop_start = i - lookback
            drop_end = i - base_size - 5
            base_start = i - base_size - 5
            base_end = i - 5
            rally_start = i - 5
            rally_end = i
            
            # Check drop phase
            drop_move = df['close'].iloc[drop_start] - df['close'].iloc[drop_end]
            if drop_move <= 0:
                continue
            
            # Check base phase
            if not identify_base_candles(df, base_start, base_end):
                continue
            
            # Check rally phase
            if not is_strong_impulse(df, rally_start, rally_end, direction='up'):
                continue
            
            rally_move = df['close'].iloc[rally_end] - df['close'].iloc[rally_start]
            if rally_move <= 0:
                continue
            
            # Valid DBR pattern found
            base_low = df['low'].iloc[base_start:base_end].min()
            base_high = df['high'].iloc[base_start:base_end].max()
            
            zones.append({
                'low': base_low,
                'high': base_high,
                'base_start_idx': base_start,
                'base_end_idx': base_end,
                'type': 'DBR',
                'direction': 'demand',
                'created_at': df.index[i],
                'tested': 0
            })
            break
    
    return zones


def detect_rbd_zones(df, lookback=50, min_base_size=2, max_base_size=10):
    """
    Detect Rally-Base-Drop (RBD) zones - Bearish reversal supply zones.
    
    Args:
        df: DataFrame with OHLC data
        lookback: How many bars to look back
        min_base_size: Minimum number of candles in base
        max_base_size: Maximum number of candles in base
    
    Returns:
        list: List of detected zones
    """
    zones = []
    
    for i in range(lookback, len(df)):
        # Look for pattern: Rally -> Base -> Drop
        for base_size in range(min_base_size, max_base_size + 1):
            if i < lookback + base_size + 5:
                continue
            
            # Define indices
            rally_start = i - lookback
            rally_end = i - base_size - 5
            base_start = i - base_size - 5
            base_end = i - 5
            drop_start = i - 5
            drop_end = i
            
            # Check rally phase
            rally_move = df['close'].iloc[rally_end] - df['close'].iloc[rally_start]
            if rally_move <= 0:
                continue
            
            # Check base phase
            if not identify_base_candles(df, base_start, base_end):
                continue
            
            # Check drop phase
            if not is_strong_impulse(df, drop_start, drop_end, direction='down'):
                continue
            
            drop_move = df['close'].iloc[drop_start] - df['close'].iloc[drop_end]
            if drop_move <= 0:
                continue
            
            # Valid RBD pattern found
            base_low = df['low'].iloc[base_start:base_end].min()
            base_high = df['high'].iloc[base_start:base_end].max()
            
            zones.append({
                'low': base_low,
                'high': base_high,
                'base_start_idx': base_start,
                'base_end_idx': base_end,
                'type': 'RBD',
                'direction': 'supply',
                'created_at': df.index[i],
                'tested': 0
            })
            break
    
    return zones


def detect_rbr_zones(df, lookback=50, min_base_size=2, max_base_size=10):
    """
    Detect Rally-Base-Rally (RBR) zones - Continuation demand zones.
    
    Args:
        df: DataFrame with OHLC data
        lookback: How many bars to look back
        min_base_size: Minimum number of candles in base
        max_base_size: Maximum number of candles in base
    
    Returns:
        list: List of detected zones
    """
    zones = []
    
    for i in range(lookback, len(df)):
        # Look for pattern: Rally -> Base -> Rally
        for base_size in range(min_base_size, max_base_size + 1):
            if i < lookback + base_size + 5:
                continue
            
            # Define indices
            rally1_start = i - lookback
            rally1_end = i - base_size - 5
            base_start = i - base_size - 5
            base_end = i - 5
            rally2_start = i - 5
            rally2_end = i
            
            # Check first rally phase
            rally1_move = df['close'].iloc[rally1_end] - df['close'].iloc[rally1_start]
            if rally1_move <= 0:
                continue
            
            # Check base phase
            if not identify_base_candles(df, base_start, base_end):
                continue
            
            # Check second rally phase
            if not is_strong_impulse(df, rally2_start, rally2_end, direction='up'):
                continue
            
            rally2_move = df['close'].iloc[rally2_end] - df['close'].iloc[rally2_start]
            if rally2_move <= 0:
                continue
            
            # Valid RBR pattern found
            base_low = df['low'].iloc[base_start:base_end].min()
            base_high = df['high'].iloc[base_start:base_end].max()
            
            zones.append({
                'low': base_low,
                'high': base_high,
                'base_start_idx': base_start,
                'base_end_idx': base_end,
                'type': 'RBR',
                'direction': 'demand',
                'created_at': df.index[i],
                'tested': 0
            })
            break
    
    return zones


def detect_dbd_zones(df, lookback=50, min_base_size=2, max_base_size=10):
    """
    Detect Drop-Base-Drop (DBD) zones - Continuation supply zones.
    
    Args:
        df: DataFrame with OHLC data
        lookback: How many bars to look back
        min_base_size: Minimum number of candles in base
        max_base_size: Maximum number of candles in base
    
    Returns:
        list: List of detected zones
    """
    zones = []
    
    for i in range(lookback, len(df)):
        # Look for pattern: Drop -> Base -> Drop
        for base_size in range(min_base_size, max_base_size + 1):
            if i < lookback + base_size + 5:
                continue
            
            # Define indices
            drop1_start = i - lookback
            drop1_end = i - base_size - 5
            base_start = i - base_size - 5
            base_end = i - 5
            drop2_start = i - 5
            drop2_end = i
            
            # Check first drop phase
            drop1_move = df['close'].iloc[drop1_start] - df['close'].iloc[drop1_end]
            if drop1_move <= 0:
                continue
            
            # Check base phase
            if not identify_base_candles(df, base_start, base_end):
                continue
            
            # Check second drop phase
            if not is_strong_impulse(df, drop2_start, drop2_end, direction='down'):
                continue
            
            drop2_move = df['close'].iloc[drop2_start] - df['close'].iloc[drop2_end]
            if drop2_move <= 0:
                continue
            
            # Valid DBD pattern found
            base_low = df['low'].iloc[base_start:base_end].min()
            base_high = df['high'].iloc[base_start:base_end].max()
            
            zones.append({
                'low': base_low,
                'high': base_high,
                'base_start_idx': base_start,
                'base_end_idx': base_end,
                'type': 'DBD',
                'direction': 'supply',
                'created_at': df.index[i],
                'tested': 0
            })
            break
    
    return zones


def detect_all_zones(df, lookback=50):
    """
    Detect all supply/demand zones (DBR, RBD, RBR, DBD).
    
    Args:
        df: DataFrame with OHLC data
        lookback: How many bars to look back
    
    Returns:
        list: List of all detected zones
    """
    all_zones = []
    
    # Detect all pattern types
    all_zones.extend(detect_dbr_zones(df, lookback))
    all_zones.extend(detect_rbd_zones(df, lookback))
    all_zones.extend(detect_rbr_zones(df, lookback))
    all_zones.extend(detect_dbd_zones(df, lookback))
    
    # Sort by creation time
    all_zones.sort(key=lambda x: x['created_at'])
    
    return all_zones


def is_price_in_zone(price, zone):
    """
    Check if price is inside a zone.
    
    Args:
        price: Current price
        zone: Zone dict with 'low' and 'high'
    
    Returns:
        bool: True if price is in zone
    """
    return zone['low'] <= price <= zone['high']
