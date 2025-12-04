#!/usr/bin/env python3
"""
Japanese Candlestick Pattern Recognition
Implements: Tonkachi (Hammer), Nagare Boshi (Shooting Star), Doji, Tsutsumi (Engulfing), Harami
"""

import pandas as pd
import numpy as np


def is_hammer(open_price, high, low, close, body_ratio_threshold=0.3, wick_ratio_threshold=2.0):
    """
    Tonkachi (Hammer) - Bullish reversal pattern.
    
    Characteristics:
    - Small body at the top
    - Long lower wick (at least 2x body size)
    - Little to no upper wick
    
    Args:
        open_price: Opening price
        high: High price
        low: Low price
        close: Closing price
        body_ratio_threshold: Maximum body/range ratio
        wick_ratio_threshold: Minimum lower_wick/body ratio
    
    Returns:
        bool: True if hammer pattern
    """
    body = abs(close - open_price)
    candle_range = high - low
    
    if candle_range == 0:
        return False
    
    # Small body relative to range
    body_ratio = body / candle_range
    if body_ratio > body_ratio_threshold:
        return False
    
    # Long lower wick
    lower_wick = min(open_price, close) - low
    upper_wick = high - max(open_price, close)
    
    if body == 0:
        body = candle_range * 0.01  # Avoid division by zero
    
    # Lower wick should be at least 2x the body
    if lower_wick < body * wick_ratio_threshold:
        return False
    
    # Upper wick should be small
    if upper_wick > body:
        return False
    
    return True


def is_shooting_star(open_price, high, low, close, body_ratio_threshold=0.3, wick_ratio_threshold=2.0):
    """
    Nagare Boshi (Shooting Star) - Bearish reversal pattern.
    
    Characteristics:
    - Small body at the bottom
    - Long upper wick (at least 2x body size)
    - Little to no lower wick
    
    Args:
        open_price: Opening price
        high: High price
        low: Low price
        close: Closing price
        body_ratio_threshold: Maximum body/range ratio
        wick_ratio_threshold: Minimum upper_wick/body ratio
    
    Returns:
        bool: True if shooting star pattern
    """
    body = abs(close - open_price)
    candle_range = high - low
    
    if candle_range == 0:
        return False
    
    # Small body relative to range
    body_ratio = body / candle_range
    if body_ratio > body_ratio_threshold:
        return False
    
    # Long upper wick
    lower_wick = min(open_price, close) - low
    upper_wick = high - max(open_price, close)
    
    if body == 0:
        body = candle_range * 0.01  # Avoid division by zero
    
    # Upper wick should be at least 2x the body
    if upper_wick < body * wick_ratio_threshold:
        return False
    
    # Lower wick should be small
    if lower_wick > body:
        return False
    
    return True


def is_doji(open_price, high, low, close, body_ratio_threshold=0.1):
    """
    Doji - Indecision pattern.
    
    Characteristics:
    - Open ≈ Close
    - Can have long wicks on both sides
    
    Args:
        open_price: Opening price
        high: High price
        low: Low price
        close: Closing price
        body_ratio_threshold: Maximum body/range ratio for doji
    
    Returns:
        bool: True if doji pattern
    """
    body = abs(close - open_price)
    candle_range = high - low
    
    if candle_range == 0:
        return True  # Perfect doji if no range
    
    # Body should be very small relative to range
    body_ratio = body / candle_range
    
    return body_ratio <= body_ratio_threshold


def is_bullish_engulfing(prev_open, prev_close, curr_open, curr_close):
    """
    Tsutsumi Age (Bullish Engulfing) - Strong bullish confirmation.
    
    Characteristics:
    - Previous candle is bearish (close < open)
    - Current candle is bullish (close > open)
    - Current candle's body fully engulfs previous candle's body
    
    Args:
        prev_open: Previous candle open
        prev_close: Previous candle close
        curr_open: Current candle open
        curr_close: Current candle close
    
    Returns:
        bool: True if bullish engulfing
    """
    # Previous candle should be bearish
    if prev_close >= prev_open:
        return False
    
    # Current candle should be bullish
    if curr_close <= curr_open:
        return False
    
    # Current candle's body should engulf previous candle's body
    prev_body_top = max(prev_open, prev_close)
    prev_body_bottom = min(prev_open, prev_close)
    curr_body_top = max(curr_open, curr_close)
    curr_body_bottom = min(curr_open, curr_close)
    
    # Full engulfing
    return curr_body_bottom < prev_body_bottom and curr_body_top > prev_body_top


def is_bearish_engulfing(prev_open, prev_close, curr_open, curr_close):
    """
    Tsutsumi Sagari (Bearish Engulfing) - Strong bearish confirmation.
    
    Characteristics:
    - Previous candle is bullish (close > open)
    - Current candle is bearish (close < open)
    - Current candle's body fully engulfs previous candle's body
    
    Args:
        prev_open: Previous candle open
        prev_close: Previous candle close
        curr_open: Current candle open
        curr_close: Current candle close
    
    Returns:
        bool: True if bearish engulfing
    """
    # Previous candle should be bullish
    if prev_close <= prev_open:
        return False
    
    # Current candle should be bearish
    if curr_close >= curr_open:
        return False
    
    # Current candle's body should engulf previous candle's body
    prev_body_top = max(prev_open, prev_close)
    prev_body_bottom = min(prev_open, prev_close)
    curr_body_top = max(curr_open, curr_close)
    curr_body_bottom = min(curr_open, curr_close)
    
    # Full engulfing
    return curr_body_bottom < prev_body_bottom and curr_body_top > prev_body_top


def is_bullish_harami(prev_open, prev_high, prev_low, prev_close, 
                       curr_open, curr_high, curr_low, curr_close):
    """
    Bullish Harami - Small candle inside previous larger candle.
    
    Characteristics:
    - Previous candle is large bearish
    - Current candle is small and inside previous candle's range
    - Signals potential reversal or consolidation
    
    Args:
        prev_open, prev_high, prev_low, prev_close: Previous candle OHLC
        curr_open, curr_high, curr_low, curr_close: Current candle OHLC
    
    Returns:
        bool: True if bullish harami
    """
    # Previous candle should be bearish and large
    if prev_close >= prev_open:
        return False
    
    prev_body = abs(prev_close - prev_open)
    curr_body = abs(curr_close - curr_open)
    
    # Current candle should be smaller
    if curr_body >= prev_body * 0.5:
        return False
    
    # Current candle should be inside previous candle's range
    prev_body_top = max(prev_open, prev_close)
    prev_body_bottom = min(prev_open, prev_close)
    curr_body_top = max(curr_open, curr_close)
    curr_body_bottom = min(curr_open, curr_close)
    
    return (curr_body_bottom > prev_body_bottom and 
            curr_body_top < prev_body_top)


def is_bearish_harami(prev_open, prev_high, prev_low, prev_close, 
                       curr_open, curr_high, curr_low, curr_close):
    """
    Bearish Harami - Small candle inside previous larger candle.
    
    Characteristics:
    - Previous candle is large bullish
    - Current candle is small and inside previous candle's range
    - Signals potential reversal or consolidation
    
    Args:
        prev_open, prev_high, prev_low, prev_close: Previous candle OHLC
        curr_open, curr_high, curr_low, curr_close: Current candle OHLC
    
    Returns:
        bool: True if bearish harami
    """
    # Previous candle should be bullish and large
    if prev_close <= prev_open:
        return False
    
    prev_body = abs(prev_close - prev_open)
    curr_body = abs(curr_close - curr_open)
    
    # Current candle should be smaller
    if curr_body >= prev_body * 0.5:
        return False
    
    # Current candle should be inside previous candle's range
    prev_body_top = max(prev_open, prev_close)
    prev_body_bottom = min(prev_open, prev_close)
    curr_body_top = max(curr_open, curr_close)
    curr_body_bottom = min(curr_open, curr_close)
    
    return (curr_body_bottom > prev_body_bottom and 
            curr_body_top < prev_body_top)


def detect_bullish_confirmation(df, idx, prev_idx=None):
    """
    Detect bullish confirmation patterns at given index.
    
    Args:
        df: DataFrame with OHLC data
        idx: Current candle index
        prev_idx: Previous candle index (if None, uses idx-1)
    
    Returns:
        str or None: Pattern name if found, else None
    """
    if idx < 1:
        return None
    
    if prev_idx is None:
        prev_idx = idx - 1
    
    curr = df.iloc[idx]
    prev = df.iloc[prev_idx]
    
    # Check for Hammer
    if is_hammer(curr['open'], curr['high'], curr['low'], curr['close']):
        return 'hammer'
    
    # Check for Bullish Engulfing
    if is_bullish_engulfing(prev['open'], prev['close'], 
                            curr['open'], curr['close']):
        return 'bullish_engulfing'
    
    # Check for Bullish Harami
    if is_bullish_harami(prev['open'], prev['high'], prev['low'], prev['close'],
                         curr['open'], curr['high'], curr['low'], curr['close']):
        return 'bullish_harami'
    
    # Check for Doji (can signal reversal after strong move)
    if is_doji(curr['open'], curr['high'], curr['low'], curr['close']):
        # Only consider doji as bullish if previous candle was bearish
        if prev['close'] < prev['open']:
            return 'doji_bullish'
    
    return None


def detect_bearish_confirmation(df, idx, prev_idx=None):
    """
    Detect bearish confirmation patterns at given index.
    
    Args:
        df: DataFrame with OHLC data
        idx: Current candle index
        prev_idx: Previous candle index (if None, uses idx-1)
    
    Returns:
        str or None: Pattern name if found, else None
    """
    if idx < 1:
        return None
    
    if prev_idx is None:
        prev_idx = idx - 1
    
    curr = df.iloc[idx]
    prev = df.iloc[prev_idx]
    
    # Check for Shooting Star
    if is_shooting_star(curr['open'], curr['high'], curr['low'], curr['close']):
        return 'shooting_star'
    
    # Check for Bearish Engulfing
    if is_bearish_engulfing(prev['open'], prev['close'], 
                            curr['open'], curr['close']):
        return 'bearish_engulfing'
    
    # Check for Bearish Harami
    if is_bearish_harami(prev['open'], prev['high'], prev['low'], prev['close'],
                         curr['open'], curr['high'], curr['low'], curr['close']):
        return 'bearish_harami'
    
    # Check for Doji (can signal reversal after strong move)
    if is_doji(curr['open'], curr['high'], curr['low'], curr['close']):
        # Only consider doji as bearish if previous candle was bullish
        if prev['close'] > prev['open']:
            return 'doji_bearish'
    
    return None
