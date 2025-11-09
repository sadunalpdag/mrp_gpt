#!/usr/bin/env python3
"""
Trading Strategy Analysis Tool
Analyzes ai_rl_log.json (opened trades) and real_closed.json (closed trades)
to provide insights about strategy effectiveness and timing.
"""
import json
from datetime import datetime, timedelta
from collections import defaultdict, Counter

def load_json_data(filepath):
    """Load JSON data from file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: {filepath} not found!")
        return []
    except json.JSONDecodeError:
        print(f"Error: {filepath} is not a valid JSON file!")
        return []

def analyze_strategy_effectiveness(opened_trades, closed_trades):
    """
    Option 1: Analyze which strategy works best
    Based on the number of successful trades per strategy
    """
    print("\n" + "="*70)
    print("OPTION 1: STRATEGY EFFECTIVENESS ANALYSIS")
    print("="*70)
    
    # Count opened trades by strategy
    opened_by_strategy = Counter()
    for trade in opened_trades:
        strategy = trade.get('kind', 'UNKNOWN')
        opened_by_strategy[strategy] += 1
    
    # Count closed trades by strategy
    closed_by_strategy = Counter()
    for trade in closed_trades:
        strategy = trade.get('strategy', 'UNKNOWN')
        closed_by_strategy[strategy] += 1
    
    # Calculate success rate
    print("\nStrategy Performance:")
    print(f"{'Strategy':<30} {'Opened':>10} {'Closed':>10} {'Success Rate':>15}")
    print("-" * 70)
    
    all_strategies = set(opened_by_strategy.keys()) | set(closed_by_strategy.keys())
    strategy_stats = []
    
    for strategy in sorted(all_strategies):
        opened = opened_by_strategy[strategy]
        closed = closed_by_strategy[strategy]
        success_rate = (closed / opened * 100) if opened > 0 else 0
        strategy_stats.append((strategy, opened, closed, success_rate))
        print(f"{strategy:<30} {opened:>10} {closed:>10} {success_rate:>14.2f}%")
    
    # Find best strategy
    if strategy_stats:
        best_strategy = max(strategy_stats, key=lambda x: x[3])
        print("\n" + "="*70)
        print(f"BEST STRATEGY: {best_strategy[0]}")
        print(f"  - Opened: {best_strategy[1]} trades")
        print(f"  - Closed: {best_strategy[2]} trades")
        print(f"  - Success Rate: {best_strategy[3]:.2f}%")
        print("="*70)

def parse_time(time_str):
    """Parse ISO 8601 timestamp"""
    try:
        return datetime.fromisoformat(time_str.replace('Z', '+00:00'))
    except:
        return None

def get_half_hour_slot(dt):
    """Get the half-hour time slot (e.g., '14:00-14:30')"""
    if dt is None:
        return None
    hour = dt.hour
    minute = 0 if dt.minute < 30 else 30
    end_minute = 30 if minute == 0 else 0
    end_hour = hour if minute == 0 else (hour + 1) % 24
    return f"{hour:02d}:{minute:02d}-{end_hour:02d}:{end_minute:02d}"

def analyze_time_performance(closed_trades):
    """
    Option 2: Analyze which time periods have faster closing trades
    Shows average closure time by half-hour intervals
    """
    print("\n" + "="*70)
    print("OPTION 2: TIME PERIOD PERFORMANCE ANALYSIS")
    print("="*70)
    
    time_durations = defaultdict(list)
    
    for trade in closed_trades:
        open_time = parse_time(trade.get('open_time'))
        close_time = parse_time(trade.get('close_time'))
        
        if open_time and close_time:
            duration = (close_time - open_time).total_seconds() / 60  # in minutes
            time_slot = get_half_hour_slot(open_time)
            if time_slot:
                time_durations[time_slot].append(duration)
    
    # Calculate averages
    print("\nAverage Trade Duration by Time Period:")
    print(f"{'Time Slot':<20} {'Trades':>10} {'Avg Duration (min)':>20} {'Avg Duration (hrs)':>20}")
    print("-" * 70)
    
    time_stats = []
    for time_slot in sorted(time_durations.keys()):
        durations = time_durations[time_slot]
        avg_duration = sum(durations) / len(durations)
        time_stats.append((time_slot, len(durations), avg_duration))
        print(f"{time_slot:<20} {len(durations):>10} {avg_duration:>20.2f} {avg_duration/60:>20.2f}")
    
    # Find fastest time slot
    if time_stats:
        fastest_slot = min(time_stats, key=lambda x: x[2])
        print("\n" + "="*70)
        print(f"FASTEST TIME SLOT: {fastest_slot[0]}")
        print(f"  - Number of trades: {fastest_slot[1]}")
        print(f"  - Average duration: {fastest_slot[2]:.2f} minutes ({fastest_slot[2]/60:.2f} hours)")
        print("="*70)

def analyze_strategy_time_performance(closed_trades):
    """
    Option 3: Analyze which strategy closes trades fastest by time period
    Shows average closure time by strategy and half-hour intervals
    """
    print("\n" + "="*70)
    print("OPTION 3: STRATEGY-TIME PERFORMANCE ANALYSIS")
    print("="*70)
    
    strategy_time_durations = defaultdict(lambda: defaultdict(list))
    
    for trade in closed_trades:
        open_time = parse_time(trade.get('open_time'))
        close_time = parse_time(trade.get('close_time'))
        strategy = trade.get('strategy', 'UNKNOWN')
        
        if open_time and close_time:
            duration = (close_time - open_time).total_seconds() / 60  # in minutes
            time_slot = get_half_hour_slot(open_time)
            if time_slot:
                strategy_time_durations[strategy][time_slot].append(duration)
    
    # Display results by strategy
    all_stats = []
    for strategy in sorted(strategy_time_durations.keys()):
        print(f"\nStrategy: {strategy}")
        print(f"{'Time Slot':<20} {'Trades':>10} {'Avg Duration (min)':>20} {'Avg Duration (hrs)':>20}")
        print("-" * 70)
        
        for time_slot in sorted(strategy_time_durations[strategy].keys()):
            durations = strategy_time_durations[strategy][time_slot]
            avg_duration = sum(durations) / len(durations)
            all_stats.append((strategy, time_slot, len(durations), avg_duration))
            print(f"{time_slot:<20} {len(durations):>10} {avg_duration:>20.2f} {avg_duration/60:>20.2f}")
    
    # Find fastest strategy-time combination
    if all_stats:
        fastest = min(all_stats, key=lambda x: x[3])
        print("\n" + "="*70)
        print(f"FASTEST COMBINATION:")
        print(f"  - Strategy: {fastest[0]}")
        print(f"  - Time Slot: {fastest[1]}")
        print(f"  - Number of trades: {fastest[2]}")
        print(f"  - Average duration: {fastest[3]:.2f} minutes ({fastest[3]/60:.2f} hours)")
        print("="*70)

def analyze_power_based_performance(closed_trades):
    """
    Option 5: Analyze trades by power ranges with step increments
    Shows closing speed average and strategy for trades in power ranges (e.g., 65-66)
    """
    print("\n" + "="*70)
    print("OPTION 5: POWER-BASED PERFORMANCE ANALYSIS")
    print("="*70)
    
    # Collect data for power ranges
    power_strategy_data = defaultdict(lambda: defaultdict(list))
    
    for trade in closed_trades:
        open_time = parse_time(trade.get('open_time'))
        close_time = parse_time(trade.get('close_time'))
        strategy = trade.get('strategy', 'UNKNOWN')
        power = trade.get('power')
        
        if open_time and close_time and power is not None:
            duration = (close_time - open_time).total_seconds() / 60  # in minutes
            # Round power to nearest integer for grouping
            power_range = int(power)
            power_strategy_data[power_range][strategy].append(duration)
    
    # Display results by power range in steps
    print("\nAverage Closing Speed by Power Range and Strategy:")
    print(f"{'Power Range':<15} {'Strategy':<20} {'Trades':>10} {'Avg Speed (min)':>18} {'Avg Speed (hrs)':>18}")
    print("-" * 85)
    
    all_stats = []
    # Sort power ranges and display in order
    for power_range in sorted(power_strategy_data.keys()):
        # Show power range as "X-X+1" (e.g., "65-66")
        range_label = f"{power_range}-{power_range+1}"
        
        for strategy in sorted(power_strategy_data[power_range].keys()):
            durations = power_strategy_data[power_range][strategy]
            avg_duration = sum(durations) / len(durations)
            all_stats.append((power_range, strategy, len(durations), avg_duration))
            print(f"{range_label:<15} {strategy:<20} {len(durations):>10} {avg_duration:>18.2f} {avg_duration/60:>18.2f}")
    
    # Find fastest power-strategy combination
    if all_stats:
        fastest = min(all_stats, key=lambda x: x[3])
        print("\n" + "="*70)
        print(f"FASTEST COMBINATION:")
        print(f"  - Power Range: {fastest[0]}-{fastest[0]+1}")
        print(f"  - Strategy: {fastest[1]}")
        print(f"  - Number of trades: {fastest[2]}")
        print(f"  - Average closing speed: {fastest[3]:.2f} minutes ({fastest[3]/60:.2f} hours)")
        print("="*70)
    
    # Summary statistics by power range (all strategies combined)
    print("\n" + "="*70)
    print("SUMMARY BY POWER RANGE (All Strategies):")
    print("-" * 85)
    print(f"{'Power Range':<15} {'Total Trades':>15} {'Avg Speed (min)':>18} {'Avg Speed (hrs)':>18}")
    print("-" * 85)
    
    for power_range in sorted(power_strategy_data.keys()):
        range_label = f"{power_range}-{power_range+1}"
        all_durations = []
        for strategy_durations in power_strategy_data[power_range].values():
            all_durations.extend(strategy_durations)
        
        if all_durations:
            avg_duration = sum(all_durations) / len(all_durations)
            print(f"{range_label:<15} {len(all_durations):>15} {avg_duration:>18.2f} {avg_duration/60:>18.2f}")
    
    print("="*70)

def main():
    """Main function with menu system"""
    print("="*70)
    print("TRADING STRATEGY ANALYSIS TOOL")
    print("="*70)
    
    # Load data
    print("\nLoading data...")
    opened_trades = load_json_data('ai_rl_log.json')
    closed_trades = load_json_data('real_closed.json')
    
    print(f"Loaded {len(opened_trades)} opened trades")
    print(f"Loaded {len(closed_trades)} closed trades")
    
    if not opened_trades and not closed_trades:
        print("No data loaded. Exiting...")
        return
    
    # Menu system
    while True:
        print("\n" + "="*70)
        print("SELECT ANALYSIS OPTION:")
        print("="*70)
        print("1. Which strategy works best? (Most effective)")
        print("2. Which time periods have faster closing trades?")
        print("3. Which strategy closes trades fastest by time period?")
        print("4. Run all analyses")
        print("5. Power-based analysis (closing speed by power ranges)")
        print("0. Exit")
        print("="*70)
        
        choice = input("\nEnter your choice (0-5): ").strip()
        
        if choice == '1':
            analyze_strategy_effectiveness(opened_trades, closed_trades)
        elif choice == '2':
            analyze_time_performance(closed_trades)
        elif choice == '3':
            analyze_strategy_time_performance(closed_trades)
        elif choice == '4':
            analyze_strategy_effectiveness(opened_trades, closed_trades)
            analyze_time_performance(closed_trades)
            analyze_strategy_time_performance(closed_trades)
            analyze_power_based_performance(closed_trades)
        elif choice == '5':
            analyze_power_based_performance(closed_trades)
        elif choice == '0':
            print("\nExiting... Goodbye!")
            break
        else:
            print("\nInvalid choice. Please select 0-5.")

if __name__ == '__main__':
    main()
