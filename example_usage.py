#!/usr/bin/env python3
"""
Example usage of Williams Fractal + EMA Strategy

This script demonstrates how to use the strategy with custom settings.
"""

from williams_fractal_ema_strategy import *
from datetime import datetime, timedelta, timezone

# ============================================
# Example 1: Quick Test (Single Symbol)
# ============================================
def example_quick_test():
    """Run a quick test on a single symbol with 7 days of data"""
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Quick Test - Single Symbol (7 days)")
    print("=" * 80)
    
    # Override settings
    global DAYS_BACK
    DAYS_BACK = 7
    
    # Set date range
    end_dt = datetime.now(timezone.utc).replace(microsecond=0, second=0)
    start_dt = end_dt - timedelta(days=DAYS_BACK + 2)
    
    # Test single symbol
    symbol = 'BTCUSDT'
    print(f"\nTesting {symbol}...")
    
    try:
        result = backtest_symbol(symbol, start_dt, end_dt, interval='1m')
        if result is not None:
            print(f"\n✓ Success! Found {len(result)} trades")
        else:
            print("\n⚠ No trades found (normal for short period)")
    except Exception as e:
        print(f"\n✗ Error: {e}")


# ============================================
# Example 2: Multiple Timeframes
# ============================================
def example_multiple_timeframes():
    """Test the same symbol on different timeframes"""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Multiple Timeframes Comparison")
    print("=" * 80)
    
    global DAYS_BACK
    DAYS_BACK = 30
    
    end_dt = datetime.now(timezone.utc).replace(microsecond=0, second=0)
    start_dt = end_dt - timedelta(days=DAYS_BACK + 2)
    
    symbol = 'ETHUSDT'
    intervals = ['1m', '5m', '15m']
    
    results = {}
    
    for interval in intervals:
        print(f"\n--- Testing {symbol} on {interval} timeframe ---")
        try:
            result = backtest_symbol(symbol, start_dt, end_dt, interval=interval)
            if result is not None:
                results[interval] = {
                    'trades': len(result),
                    'win_rate': (result['pnl_usd'] > 0).mean() * 100,
                    'total_pnl': result['pnl_usd'].sum()
                }
        except Exception as e:
            print(f"Error on {interval}: {e}")
    
    # Compare results
    print("\n" + "-" * 60)
    print("COMPARISON:")
    print("-" * 60)
    for interval, stats in results.items():
        print(f"{interval:>5} | Trades: {stats['trades']:>3} | "
              f"Win Rate: {stats['win_rate']:>5.1f}% | "
              f"PnL: ${stats['total_pnl']:>8.2f}")


# ============================================
# Example 3: Custom Settings
# ============================================
def example_custom_settings():
    """Run backtest with custom risk/reward settings"""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Custom Settings")
    print("=" * 80)
    
    # Customize settings
    global RISK_REWARD_RATIO, POSITION_SIZE_USD, DAYS_BACK
    RISK_REWARD_RATIO = 2.0        # More aggressive R:R
    POSITION_SIZE_USD = 200.0      # Larger position
    DAYS_BACK = 14
    
    print(f"\nCustom Settings:")
    print(f"  Risk:Reward = 1:{RISK_REWARD_RATIO}")
    print(f"  Position Size = ${POSITION_SIZE_USD}")
    print(f"  Period = {DAYS_BACK} days")
    
    end_dt = datetime.now(timezone.utc).replace(microsecond=0, second=0)
    start_dt = end_dt - timedelta(days=DAYS_BACK + 2)
    
    symbol = 'BNBUSDT'
    print(f"\nTesting {symbol}...")
    
    try:
        result = backtest_symbol(symbol, start_dt, end_dt, interval='5m')
        if result is not None:
            print(f"\n✓ Found {len(result)} trades")
            print(f"  Total PnL: ${result['pnl_usd'].sum():.2f}")
            print(f"  Win Rate: {(result['pnl_usd'] > 0).mean() * 100:.1f}%")
    except Exception as e:
        print(f"\n✗ Error: {e}")


# ============================================
# Example 4: Analyzing Trade Distribution
# ============================================
def example_trade_analysis():
    """Analyze the distribution of trades by scenario"""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Trade Analysis by Scenario")
    print("=" * 80)
    
    global DAYS_BACK
    DAYS_BACK = 30
    
    end_dt = datetime.now(timezone.utc).replace(microsecond=0, second=0)
    start_dt = end_dt - timedelta(days=DAYS_BACK + 2)
    
    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
    all_trades = []
    
    for symbol in symbols:
        print(f"\nProcessing {symbol}...")
        try:
            result = backtest_symbol(symbol, start_dt, end_dt, interval='5m')
            if result is not None:
                all_trades.append(result)
        except Exception as e:
            print(f"  Error: {e}")
    
    if not all_trades:
        print("\n⚠ No trades found")
        return
    
    # Combine all trades
    import pandas as pd
    df_all = pd.concat(all_trades, ignore_index=True)
    
    print("\n" + "-" * 60)
    print("TRADE DISTRIBUTION:")
    print("-" * 60)
    
    # By scenario
    print("\nBy Entry Scenario:")
    for scenario in df_all['scenario'].unique():
        scen_trades = df_all[df_all['scenario'] == scenario]
        wins = scen_trades[scen_trades['pnl_usd'] > 0]
        print(f"  {scenario:>12}: {len(scen_trades):>3} trades | "
              f"Win Rate: {len(wins)/len(scen_trades)*100:>5.1f}% | "
              f"Avg PnL: ${scen_trades['pnl_usd'].mean():>6.2f}")
    
    # By direction
    print("\nBy Direction:")
    for direction in df_all['side'].unique():
        dir_trades = df_all[df_all['side'] == direction]
        wins = dir_trades[dir_trades['pnl_usd'] > 0]
        print(f"  {direction.upper():>5}: {len(dir_trades):>3} trades | "
              f"Win Rate: {len(wins)/len(dir_trades)*100:>5.1f}% | "
              f"Total PnL: ${dir_trades['pnl_usd'].sum():>8.2f}")


# ============================================
# Main Menu
# ============================================
def main():
    """Main menu for example selection"""
    print("\n" + "=" * 80)
    print("WILLIAMS FRACTAL + EMA STRATEGY - EXAMPLE USAGE")
    print("=" * 80)
    print("\nAvailable Examples:")
    print("  1. Quick Test (single symbol, 7 days)")
    print("  2. Multiple Timeframes Comparison")
    print("  3. Custom Settings (R:R = 2.0)")
    print("  4. Trade Analysis by Scenario")
    print("  5. Run all examples")
    print("  0. Exit")
    
    try:
        choice = input("\nSelect example (0-5): ").strip()
        
        if choice == '1':
            example_quick_test()
        elif choice == '2':
            example_multiple_timeframes()
        elif choice == '3':
            example_custom_settings()
        elif choice == '4':
            example_trade_analysis()
        elif choice == '5':
            example_quick_test()
            example_multiple_timeframes()
            example_custom_settings()
            example_trade_analysis()
        elif choice == '0':
            print("\nExiting...")
            return
        else:
            print("\n⚠ Invalid choice")
            
    except KeyboardInterrupt:
        print("\n\nExiting...")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Note: These examples require network access to Binance API
    print("\n⚠️  Note: These examples require internet access to fetch data from Binance.")
    print("    If you're in a sandboxed environment, they may not work.")
    print("    Use the test script (test_williams_fractal_ema.py) instead.")
    
    response = input("\nContinue anyway? (y/n): ").strip().lower()
    if response == 'y':
        main()
    else:
        print("\nUse: python3 test_williams_fractal_ema.py")
        print("to run tests with mock data instead.")
