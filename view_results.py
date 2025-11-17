#!/usr/bin/env python3
"""
Quick script to view backtest results
"""

import json

def main():
    print("=" * 80)
    print("INSTITUTIONAL SUPPLY/DEMAND STRATEGY - 90 DAY RESULTS")
    print("=" * 80)
    
    # Load trades
    with open('institutional_trades_90day.json', 'r') as f:
        trades = json.load(f)
    
    # Basic stats
    total_trades = len(trades)
    winners = sum(1 for t in trades if t['pnl_usd'] > 0)
    losers = sum(1 for t in trades if t['pnl_usd'] < 0)
    win_rate = (winners / total_trades * 100) if total_trades > 0 else 0
    
    total_pnl = sum(t['pnl_usd'] for t in trades)
    initial_capital = 5000.0
    final_capital = trades[-1]['capital']
    
    print(f"\n📊 PERFORMANCE METRICS")
    print(f"{'='*80}")
    print(f"Period: {trades[0]['entry_time'][:10]} to {trades[-1]['entry_time'][:10]}")
    print(f"Total Trades: {total_trades}")
    print(f"Winners: {winners} | Losers: {losers}")
    print(f"Win Rate: {win_rate:.1f}%")
    print(f"\nInitial Capital: ${initial_capital:,.2f}")
    print(f"Final Capital: ${final_capital:,.2f}")
    print(f"Total Profit: ${total_pnl:,.2f}")
    print(f"Return: {((final_capital-initial_capital)/initial_capital*100):+.1f}%")
    
    # Pattern breakdown
    print(f"\n🎯 CONFIRMATION PATTERNS")
    print(f"{'='*80}")
    patterns = {}
    for t in trades:
        p = t['confirmation_pattern']
        if p not in patterns:
            patterns[p] = {'count': 0, 'wins': 0}
        patterns[p]['count'] += 1
        if t['pnl_usd'] > 0:
            patterns[p]['wins'] += 1
    
    for pattern in sorted(patterns.keys(), key=lambda x: patterns[x]['count'], reverse=True):
        stats = patterns[pattern]
        wr = (stats['wins'] / stats['count'] * 100) if stats['count'] > 0 else 0
        print(f"{pattern:25s}: {stats['count']:3d} trades | Win Rate: {wr:5.1f}%")
    
    # Zone breakdown
    print(f"\n📍 ZONE TYPES")
    print(f"{'='*80}")
    zones = {}
    for t in trades:
        z = t['zone_type']
        if z not in zones:
            zones[z] = {'count': 0, 'wins': 0}
        zones[z]['count'] += 1
        if t['pnl_usd'] > 0:
            zones[z]['wins'] += 1
    
    for zone in ['DBR', 'RBD', 'RBR', 'DBD']:
        if zone in zones:
            stats = zones[zone]
            wr = (stats['wins'] / stats['count'] * 100) if stats['count'] > 0 else 0
            print(f"{zone:10s}: {stats['count']:3d} trades | Win Rate: {wr:5.1f}%")
    
    # Sample trades
    print(f"\n📝 SAMPLE TRADES (First 5)")
    print(f"{'='*80}")
    for i, t in enumerate(trades[:5], 1):
        sign = '+' if t['pnl_usd'] >= 0 else ''
        print(f"{i}. {t['symbol']:10s} | {t['side']:5s} | "
              f"{t['entry_time'][:16]} | {t['exit_reason']:2s} | "
              f"PnL: {sign}${t['pnl_usd']:7.2f} | {t['confirmation_pattern']}")
    
    print(f"\n{'='*80}")
    print(f"✅ Complete trade log available in: institutional_trades_90day.json")
    print(f"✅ Total of {total_trades} trades with full details")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
