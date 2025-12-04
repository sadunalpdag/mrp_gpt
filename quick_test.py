#!/usr/bin/env python3
"""
Quick test run with limited scope to verify functionality
"""

import time
import requests
from datetime import datetime, timedelta, timezone
import pandas as pd
import json

from institutional_strategy import backtest_symbol, ms_since_epoch

def quick_test():
    """Run a quick test with BTC only for 7 days."""
    print("=" * 60)
    print("QUICK TEST: 7 days, BTCUSDT only")
    print("=" * 60)
    
    # Test with just BTC for 7 days
    symbol = "BTCUSDT"
    end_dt = datetime.now(timezone.utc).replace(microsecond=0, second=0)
    start_dt = end_dt - timedelta(days=7)
    
    print(f"\nSymbol: {symbol}")
    print(f"Period: {start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')}")
    
    try:
        trades = backtest_symbol(symbol, start_dt, end_dt)
        
        if trades:
            print(f"\n✓ Generated {len(trades)} trades")
            
            # Calculate quick stats
            df = pd.DataFrame(trades)
            total_pnl = df['pnl_usd'].sum()
            win_rate = (df['pnl_usd'] > 0).mean() * 100
            
            print(f"Total PnL: ${total_pnl:.2f}")
            print(f"Win Rate: {win_rate:.1f}%")
            
            # Show first few trades
            print("\nFirst 3 trades:")
            for i, trade in enumerate(trades[:3]):
                print(f"  {i+1}. {trade['side'].upper()} | "
                      f"Entry: {trade['entry_price']:.2f} | "
                      f"Exit: {trade['exit_price']:.2f} | "
                      f"PnL: ${trade['pnl_usd']:.2f} | "
                      f"Pattern: {trade['confirmation_pattern']}")
            
            print("\n✓ Quick test completed successfully!")
            return 0
        else:
            print("\n⚠️  No trades generated (this may be normal for short test period)")
            return 0
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(quick_test())
