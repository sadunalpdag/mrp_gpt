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
    (Excludes trades closed by profit target)
    """
    print("\n" + "="*70)
    print("OPTION 1: STRATEGY EFFECTIVENESS ANALYSIS")
    print("="*70)
    
    # Count opened trades by strategy
    opened_by_strategy = Counter()
    for trade in opened_trades:
        strategy = trade.get('kind', 'UNKNOWN')
        opened_by_strategy[strategy] += 1
    
    # Count closed trades by strategy (excluding profit target closed)
    closed_by_strategy = Counter()
    for trade in closed_trades:
        # Skip trades closed by profit target
        if trade.get('closed_by_profit_target') == True:
            continue
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

def analyze_time_performance(opened_trades, closed_trades):
    """
    Option 2: Analyze which time periods have faster closing trades
    Shows average closure time and open/close ratio by half-hour intervals
    (Excludes trades closed by profit target)
    """
    print("\n" + "="*70)
    print("OPTION 2: TIME PERIOD PERFORMANCE ANALYSIS")
    print("="*70)
    
    time_durations = defaultdict(list)
    time_opened = defaultdict(int)
    
    # Track opened trades by time slot
    for trade in opened_trades:
        open_time = parse_time(trade.get('time'))
        
        if open_time:
            time_slot = get_half_hour_slot(open_time)
            if time_slot:
                time_opened[time_slot] += 1
    
    # Track closed trades by time slot (excluding profit target closed)
    for trade in closed_trades:
        # Skip trades closed by profit target
        if trade.get('closed_by_profit_target') == True:
            continue
        open_time = parse_time(trade.get('open_time'))
        close_time = parse_time(trade.get('close_time'))
        
        if open_time and close_time:
            duration = (close_time - open_time).total_seconds() / 60  # in minutes
            time_slot = get_half_hour_slot(open_time)
            if time_slot:
                time_durations[time_slot].append(duration)
    
    # Calculate averages
    print("\nAverage Trade Duration and Open/Close Ratio by Time Period:")
    print(f"{'Time Slot':<20} {'Closed/Opened':>15} {'Ratio':>10} {'Avg Duration (min)':>20} {'Avg Duration (hrs)':>20}")
    print("-" * 90)
    
    time_stats = []
    for time_slot in sorted(time_durations.keys()):
        durations = time_durations[time_slot]
        closed_count = len(durations)
        opened_count = time_opened.get(time_slot, 0)
        
        # Calculate close ratio
        if opened_count > 0:
            close_ratio = (closed_count / opened_count) * 100
            ratio_str = f"{closed_count}/{opened_count}"
        else:
            close_ratio = 0.0
            ratio_str = f"{closed_count}/0"
        
        avg_duration = sum(durations) / len(durations)
        time_stats.append((time_slot, closed_count, opened_count, close_ratio, avg_duration))
        print(f"{time_slot:<20} {ratio_str:>15} {close_ratio:>9.1f}% {avg_duration:>20.2f} {avg_duration/60:>20.2f}")
    
    # Find fastest time slot
    if time_stats:
        fastest_slot = min(time_stats, key=lambda x: x[4])
        print("\n" + "="*70)
        print(f"FASTEST TIME SLOT: {fastest_slot[0]}")
        print(f"  - Closed/Opened: {fastest_slot[1]}/{fastest_slot[2]} ({fastest_slot[3]:.1f}%)")
        print(f"  - Average duration: {fastest_slot[4]:.2f} minutes ({fastest_slot[4]/60:.2f} hours)")
        print("="*70)
        
        # Find best close ratio
        best_ratio = max(time_stats, key=lambda x: x[3])
        print(f"\nBEST CLOSE RATIO TIME SLOT: {best_ratio[0]}")
        print(f"  - Closed/Opened: {best_ratio[1]}/{best_ratio[2]} ({best_ratio[3]:.1f}%)")
        print(f"  - Average duration: {best_ratio[4]:.2f} minutes ({best_ratio[4]/60:.2f} hours)")
        print("="*70)

def analyze_strategy_time_performance(opened_trades, closed_trades):
    """
    Option 3: Analyze which strategy closes trades fastest by time period
    Shows average closure time, open/closed ratio by strategy and half-hour intervals
    (Excludes trades closed by profit target)
    """
    print("\n" + "="*70)
    print("OPTION 3: STRATEGY-TIME PERFORMANCE ANALYSIS")
    print("="*70)
    
    strategy_time_durations = defaultdict(lambda: defaultdict(list))
    strategy_time_opened = defaultdict(lambda: defaultdict(int))
    
    # Track opened trades by strategy and time slot
    for trade in opened_trades:
        open_time = parse_time(trade.get('time'))
        strategy = trade.get('kind', 'UNKNOWN')
        
        if open_time:
            time_slot = get_half_hour_slot(open_time)
            if time_slot:
                strategy_time_opened[strategy][time_slot] += 1
    
    # Track closed trades by strategy and time slot (excluding profit target closed)
    for trade in closed_trades:
        # Skip trades closed by profit target
        if trade.get('closed_by_profit_target') == True:
            continue
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
        print(f"{'Time Slot':<20} {'Closed/Opened':>15} {'Ratio':>10} {'Avg Duration (min)':>20} {'Avg Duration (hrs)':>20}")
        print("-" * 90)
        
        for time_slot in sorted(strategy_time_durations[strategy].keys()):
            durations = strategy_time_durations[strategy][time_slot]
            closed_count = len(durations)
            opened_count = strategy_time_opened[strategy].get(time_slot, 0)
            
            # Calculate close ratio
            if opened_count > 0:
                close_ratio = (closed_count / opened_count) * 100
                ratio_str = f"{closed_count}/{opened_count}"
            else:
                close_ratio = 0.0
                ratio_str = f"{closed_count}/0"
            
            avg_duration = sum(durations) / len(durations)
            all_stats.append((strategy, time_slot, closed_count, opened_count, close_ratio, avg_duration))
            print(f"{time_slot:<20} {ratio_str:>15} {close_ratio:>9.1f}% {avg_duration:>20.2f} {avg_duration/60:>20.2f}")
    
    # Find fastest strategy-time combination
    if all_stats:
        fastest = min(all_stats, key=lambda x: x[5])
        print("\n" + "="*70)
        print(f"FASTEST COMBINATION:")
        print(f"  - Strategy: {fastest[0]}")
        print(f"  - Time Slot: {fastest[1]}")
        print(f"  - Closed/Opened: {fastest[2]}/{fastest[3]} ({fastest[4]:.1f}%)")
        print(f"  - Average duration: {fastest[5]:.2f} minutes ({fastest[5]/60:.2f} hours)")
        print("="*70)
        
        # Find best close ratio
        best_ratio = max(all_stats, key=lambda x: x[4])
        print(f"\nBEST CLOSE RATIO:")
        print(f"  - Strategy: {best_ratio[0]}")
        print(f"  - Time Slot: {best_ratio[1]}")
        print(f"  - Closed/Opened: {best_ratio[2]}/{best_ratio[3]} ({best_ratio[4]:.1f}%)")
        print(f"  - Average duration: {best_ratio[5]:.2f} minutes ({best_ratio[5]/60:.2f} hours)")
        print("="*70)

def analyze_open_trades(opened_trades, closed_trades):
    """
    Option 6: Analyze currently open trades
    Shows how long trades have been open, their power values, and strategy
    """
    print("\n" + "="*70)
    print("OPTION 6: CURRENTLY OPEN TRADES ANALYSIS")
    print("="*70)
    
    # Get current time with UTC timezone
    from datetime import timezone
    current_time = datetime.now(timezone.utc)
    
    # Find trades that are still open (in opened_trades but not in closed_trades)
    closed_symbols_times = set()
    for trade in closed_trades:
        symbol = trade.get('symbol')
        open_time = trade.get('open_time')
        if symbol and open_time:
            closed_symbols_times.add((symbol, open_time))
    
    open_trades_info = []
    for trade in opened_trades:
        symbol = trade.get('symbol')
        time = trade.get('time')
        if symbol and time:
            # Check if this trade is not in closed trades
            if (symbol, time) not in closed_symbols_times:
                open_time = parse_time(time)
                if open_time:
                    duration = (current_time - open_time).total_seconds() / 3600  # in hours
                    strategy = trade.get('kind', 'UNKNOWN')
                    power = trade.get('power', 0)
                    direction = trade.get('dir', 'UNKNOWN')
                    open_trades_info.append({
                        'symbol': symbol,
                        'strategy': strategy,
                        'direction': direction,
                        'power': power,
                        'open_time': time,
                        'hours_open': duration,
                        'entry': trade.get('entry')
                    })
    
    # Sort by hours open (descending)
    open_trades_info.sort(key=lambda x: x['hours_open'], reverse=True)
    
    print(f"\nTotal Currently Open Trades: {len(open_trades_info)}")
    print(f"\n{'Symbol':<15} {'Strategy':<15} {'Dir':<6} {'Power':<10} {'Hours Open':<12} {'Entry Price':<12} {'Open Time':<25}")
    print("-" * 110)
    
    for trade in open_trades_info:
        print(f"{trade['symbol']:<15} {trade['strategy']:<15} {trade['direction']:<6} "
              f"{trade['power']:<10.2f} {trade['hours_open']:<12.2f} "
              f"{trade['entry']:<12.6f} {trade['open_time']:<25}")
    
    # Summary statistics
    if open_trades_info:
        print("\n" + "="*70)
        print("SUMMARY STATISTICS:")
        print("-" * 70)
        
        total_hours = sum(t['hours_open'] for t in open_trades_info)
        avg_hours = total_hours / len(open_trades_info)
        max_hours = max(t['hours_open'] for t in open_trades_info)
        min_hours = min(t['hours_open'] for t in open_trades_info)
        
        avg_power = sum(t['power'] for t in open_trades_info) / len(open_trades_info)
        max_power = max(t['power'] for t in open_trades_info)
        min_power = min(t['power'] for t in open_trades_info)
        
        print(f"Average time open: {avg_hours:.2f} hours")
        print(f"Maximum time open: {max_hours:.2f} hours")
        print(f"Minimum time open: {min_hours:.2f} hours")
        print(f"\nAverage power: {avg_power:.2f}")
        print(f"Maximum power: {max_power:.2f}")
        print(f"Minimum power: {min_power:.2f}")
        
        # Group by strategy
        strategy_counts = Counter(t['strategy'] for t in open_trades_info)
        print("\nOpen trades by strategy:")
        for strategy, count in strategy_counts.most_common():
            print(f"  - {strategy}: {count} trades")
        
        print("="*70)

def analyze_open_transaction_ratio(opened_trades, closed_trades):
    """
    Option 7: Show total open transaction ratio
    Displays the ratio of closed to opened trades
    """
    print("\n" + "="*70)
    print("OPTION 7: OPEN TRANSACTION RATIO ANALYSIS")
    print("="*70)
    
    total_opened = len(opened_trades)
    total_closed = len(closed_trades)
    total_still_open = total_opened - total_closed
    
    close_ratio = (total_closed / total_opened * 100) if total_opened > 0 else 0
    open_ratio = (total_still_open / total_opened * 100) if total_opened > 0 else 0
    
    print(f"\nTotal Opened Trades: {total_opened}")
    print(f"Total Closed Trades: {total_closed}")
    print(f"Currently Open Trades: {total_still_open}")
    print(f"\nClosed Ratio: {close_ratio:.2f}%")
    print(f"Still Open Ratio: {open_ratio:.2f}%")
    
    # Breakdown by strategy
    print("\n" + "="*70)
    print("BREAKDOWN BY STRATEGY:")
    print("-" * 70)
    print(f"{'Strategy':<30} {'Opened':>10} {'Closed':>10} {'Still Open':>12} {'Close %':>10} {'Open %':>10}")
    print("-" * 90)
    
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
    
    all_strategies = set(opened_by_strategy.keys()) | set(closed_by_strategy.keys())
    
    for strategy in sorted(all_strategies):
        opened = opened_by_strategy[strategy]
        closed = closed_by_strategy[strategy]
        still_open = opened - closed
        close_pct = (closed / opened * 100) if opened > 0 else 0
        open_pct = (still_open / opened * 100) if opened > 0 else 0
        
        print(f"{strategy:<30} {opened:>10} {closed:>10} {still_open:>12} {close_pct:>9.1f}% {open_pct:>9.1f}%")
    
    print("="*70)

def analyze_open_trades_by_power(opened_trades, closed_trades):
    """
    Option 8: Analyze currently open trades by power ranges
    Shows how long open trades have been open, grouped by power ranges
    """
    print("\n" + "="*70)
    print("OPTION 8: OPEN TRADES BY POWER RANGE ANALYSIS")
    print("="*70)
    
    # Get current time with UTC timezone
    from datetime import timezone
    current_time = datetime.now(timezone.utc)
    
    # Find trades that are still open
    closed_symbols_times = set()
    for trade in closed_trades:
        symbol = trade.get('symbol')
        open_time = trade.get('open_time')
        if symbol and open_time:
            closed_symbols_times.add((symbol, open_time))
    
    # Collect open trades by power range
    power_strategy_open = defaultdict(lambda: defaultdict(list))
    
    for trade in opened_trades:
        symbol = trade.get('symbol')
        time = trade.get('time')
        power = trade.get('power')
        
        if symbol and time and power is not None:
            # Check if this trade is not in closed trades
            if (symbol, time) not in closed_symbols_times:
                open_time = parse_time(time)
                if open_time:
                    duration = (current_time - open_time).total_seconds() / 3600  # in hours
                    strategy = trade.get('kind', 'UNKNOWN')
                    power_range = int(power)
                    power_strategy_open[power_range][strategy].append(duration)
    
    # Display results
    print("\nCurrently Open Trades by Power Range and Strategy:")
    print(f"{'Power Range':<15} {'Strategy':<20} {'Count':>10} {'Avg Hours Open':>18} {'Max Hours Open':>18}")
    print("-" * 90)
    
    all_stats = []
    for power_range in sorted(power_strategy_open.keys()):
        range_label = f"{power_range}-{power_range+1}"
        
        for strategy in sorted(power_strategy_open[power_range].keys()):
            durations = power_strategy_open[power_range][strategy]
            count = len(durations)
            avg_duration = sum(durations) / len(durations)
            max_duration = max(durations)
            
            all_stats.append((power_range, strategy, count, avg_duration, max_duration))
            print(f"{range_label:<15} {strategy:<20} {count:>10} {avg_duration:>18.2f} {max_duration:>18.2f}")
    
    # Summary by power range
    print("\n" + "="*70)
    print("SUMMARY BY POWER RANGE (All Strategies):")
    print("-" * 90)
    print(f"{'Power Range':<15} {'Total Open':>12} {'Avg Hours Open':>18} {'Max Hours Open':>18}")
    print("-" * 90)
    
    for power_range in sorted(power_strategy_open.keys()):
        range_label = f"{power_range}-{power_range+1}"
        all_durations = []
        
        for strategy_durations in power_strategy_open[power_range].values():
            all_durations.extend(strategy_durations)
        
        if all_durations:
            count = len(all_durations)
            avg_duration = sum(all_durations) / len(all_durations)
            max_duration = max(all_durations)
            print(f"{range_label:<15} {count:>12} {avg_duration:>18.2f} {max_duration:>18.2f}")
    
    # Find longest open trade
    if all_stats:
        longest = max(all_stats, key=lambda x: x[4])
        print("\n" + "="*70)
        print(f"LONGEST OPEN TRADE:")
        print(f"  - Power Range: {longest[0]}-{longest[0]+1}")
        print(f"  - Strategy: {longest[1]}")
        print(f"  - Count in this range: {longest[2]}")
        print(f"  - Average hours open: {longest[3]:.2f} hours")
        print(f"  - Maximum hours open: {longest[4]:.2f} hours")
        print("="*70)

def analyze_power_based_performance(opened_trades, closed_trades):
    """
    Option 5: Analyze trades by power ranges with step increments
    Shows closing speed average, open/close ratio and strategy for trades in power ranges (e.g., 65-66)
    (Excludes trades closed by profit target)
    """
    print("\n" + "="*70)
    print("OPTION 5: POWER-BASED PERFORMANCE ANALYSIS")
    print("="*70)
    
    # Collect data for power ranges
    power_strategy_data = defaultdict(lambda: defaultdict(list))
    power_strategy_opened = defaultdict(lambda: defaultdict(int))
    
    # Track opened trades by power range and strategy
    for trade in opened_trades:
        strategy = trade.get('kind', 'UNKNOWN')
        power = trade.get('power')
        
        if power is not None:
            # Round power to nearest integer for grouping
            power_range = int(power)
            power_strategy_opened[power_range][strategy] += 1
    
    # Track closed trades by power range and strategy (excluding profit target closed)
    for trade in closed_trades:
        # Skip trades closed by profit target
        if trade.get('closed_by_profit_target') == True:
            continue
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
    print("\nAverage Closing Speed and Open/Close Ratio by Power Range and Strategy:")
    print(f"{'Power Range':<15} {'Strategy':<20} {'Closed/Opened':>15} {'Ratio':>10} {'Avg Speed (min)':>18} {'Avg Speed (hrs)':>18}")
    print("-" * 110)
    
    all_stats = []
    # Sort power ranges and display in order
    for power_range in sorted(power_strategy_data.keys()):
        # Show power range as "X-X+1" (e.g., "65-66")
        range_label = f"{power_range}-{power_range+1}"
        
        for strategy in sorted(power_strategy_data[power_range].keys()):
            durations = power_strategy_data[power_range][strategy]
            closed_count = len(durations)
            opened_count = power_strategy_opened[power_range].get(strategy, 0)
            
            # Calculate close ratio
            if opened_count > 0:
                close_ratio = (closed_count / opened_count) * 100
                ratio_str = f"{closed_count}/{opened_count}"
            else:
                close_ratio = 0.0
                ratio_str = f"{closed_count}/0"
            
            avg_duration = sum(durations) / len(durations)
            all_stats.append((power_range, strategy, closed_count, opened_count, close_ratio, avg_duration))
            print(f"{range_label:<15} {strategy:<20} {ratio_str:>15} {close_ratio:>9.1f}% {avg_duration:>18.2f} {avg_duration/60:>18.2f}")
    
    # Find fastest power-strategy combination
    if all_stats:
        fastest = min(all_stats, key=lambda x: x[5])
        print("\n" + "="*70)
        print(f"FASTEST COMBINATION:")
        print(f"  - Power Range: {fastest[0]}-{fastest[0]+1}")
        print(f"  - Strategy: {fastest[1]}")
        print(f"  - Closed/Opened: {fastest[2]}/{fastest[3]} ({fastest[4]:.1f}%)")
        print(f"  - Average closing speed: {fastest[5]:.2f} minutes ({fastest[5]/60:.2f} hours)")
        print("="*70)
        
        # Find best close ratio
        best_ratio = max(all_stats, key=lambda x: x[4])
        print(f"\nBEST CLOSE RATIO:")
        print(f"  - Power Range: {best_ratio[0]}-{best_ratio[0]+1}")
        print(f"  - Strategy: {best_ratio[1]}")
        print(f"  - Closed/Opened: {best_ratio[2]}/{best_ratio[3]} ({best_ratio[4]:.1f}%)")
        print(f"  - Average closing speed: {best_ratio[5]:.2f} minutes ({best_ratio[5]/60:.2f} hours)")
        print("="*70)
    
    # Summary statistics by power range (all strategies combined)
    print("\n" + "="*70)
    print("SUMMARY BY POWER RANGE (All Strategies):")
    print("-" * 110)
    print(f"{'Power Range':<15} {'Closed/Opened':>15} {'Ratio':>10} {'Avg Speed (min)':>18} {'Avg Speed (hrs)':>18}")
    print("-" * 110)
    
    for power_range in sorted(power_strategy_data.keys()):
        range_label = f"{power_range}-{power_range+1}"
        all_durations = []
        total_closed = 0
        total_opened = 0
        
        for strategy_durations in power_strategy_data[power_range].values():
            all_durations.extend(strategy_durations)
            total_closed += len(strategy_durations)
        
        for strategy_opened in power_strategy_opened[power_range].values():
            total_opened += strategy_opened
        
        if all_durations:
            avg_duration = sum(all_durations) / len(all_durations)
            if total_opened > 0:
                close_ratio = (total_closed / total_opened) * 100
                ratio_str = f"{total_closed}/{total_opened}"
            else:
                close_ratio = 0.0
                ratio_str = f"{total_closed}/0"
            print(f"{range_label:<15} {ratio_str:>15} {close_ratio:>9.1f}% {avg_duration:>18.2f} {avg_duration/60:>18.2f}")
    
    print("="*70)

def analyze_profit_target_by_strategy(opened_trades, closed_trades):
    """
    Option 10: Analyze trades closed by profit target by strategy
    Shows total opened vs profit target closed per strategy
    """
    print("\n" + "="*70)
    print("OPTION 10: PROFIT TARGET CLOSED TRADES BY STRATEGY")
    print("="*70)
    
    # Count opened trades by strategy
    opened_by_strategy = Counter()
    for trade in opened_trades:
        strategy = trade.get('kind', 'UNKNOWN')
        opened_by_strategy[strategy] += 1
    
    # Count profit target closed trades by strategy
    profit_target_closed_by_strategy = Counter()
    for trade in closed_trades:
        if trade.get('closed_by_profit_target') == True:
            strategy = trade.get('strategy', 'UNKNOWN')
            profit_target_closed_by_strategy[strategy] += 1
    
    # Calculate statistics
    print("\nProfit Target Closed Trades by Strategy:")
    print(f"{'Strategy':<30} {'Opened':>10} {'PT Closed':>12} {'PT Closed %':>15}")
    print("-" * 70)
    
    all_strategies = set(opened_by_strategy.keys()) | set(profit_target_closed_by_strategy.keys())
    strategy_stats = []
    
    for strategy in sorted(all_strategies):
        opened = opened_by_strategy[strategy]
        pt_closed = profit_target_closed_by_strategy[strategy]
        pt_closed_pct = (pt_closed / opened * 100) if opened > 0 else 0
        strategy_stats.append((strategy, opened, pt_closed, pt_closed_pct))
        print(f"{strategy:<30} {opened:>10} {pt_closed:>12} {pt_closed_pct:>14.2f}%")
    
    # Total summary
    total_opened = sum(opened_by_strategy.values())
    total_pt_closed = sum(profit_target_closed_by_strategy.values())
    total_pt_pct = (total_pt_closed / total_opened * 100) if total_opened > 0 else 0
    
    print("\n" + "="*70)
    print("TOTAL SUMMARY:")
    print(f"  - Total Opened: {total_opened}")
    print(f"  - Total Closed by Profit Target: {total_pt_closed}")
    print(f"  - Percentage: {total_pt_pct:.2f}%")
    
    # Find strategy with highest PT close rate
    if strategy_stats:
        highest_pt = max(strategy_stats, key=lambda x: x[3])
        print(f"\nHIGHEST PROFIT TARGET CLOSE RATE:")
        print(f"  - Strategy: {highest_pt[0]}")
        print(f"  - Opened: {highest_pt[1]}")
        print(f"  - PT Closed: {highest_pt[2]} ({highest_pt[3]:.2f}%)")
    print("="*70)

def analyze_profit_target_by_time(opened_trades, closed_trades):
    """
    Option 11: Analyze trades closed by profit target by hour
    Shows total opened vs profit target closed per half-hour time slot
    """
    print("\n" + "="*70)
    print("OPTION 11: PROFIT TARGET CLOSED TRADES BY TIME PERIOD")
    print("="*70)
    
    time_opened = defaultdict(int)
    time_pt_closed = defaultdict(int)
    
    # Track opened trades by time slot
    for trade in opened_trades:
        open_time = parse_time(trade.get('time'))
        if open_time:
            time_slot = get_half_hour_slot(open_time)
            if time_slot:
                time_opened[time_slot] += 1
    
    # Track profit target closed trades by time slot
    for trade in closed_trades:
        if trade.get('closed_by_profit_target') == True:
            open_time = parse_time(trade.get('open_time'))
            if open_time:
                time_slot = get_half_hour_slot(open_time)
                if time_slot:
                    time_pt_closed[time_slot] += 1
    
    # Calculate statistics
    print("\nProfit Target Closed Trades by Time Period:")
    print(f"{'Time Slot':<20} {'Opened':>10} {'PT Closed':>12} {'PT Closed %':>15}")
    print("-" * 60)
    
    time_stats = []
    all_time_slots = set(time_opened.keys()) | set(time_pt_closed.keys())
    
    for time_slot in sorted(all_time_slots):
        opened = time_opened.get(time_slot, 0)
        pt_closed = time_pt_closed.get(time_slot, 0)
        pt_closed_pct = (pt_closed / opened * 100) if opened > 0 else 0
        time_stats.append((time_slot, opened, pt_closed, pt_closed_pct))
        print(f"{time_slot:<20} {opened:>10} {pt_closed:>12} {pt_closed_pct:>14.2f}%")
    
    # Total summary
    total_opened = sum(time_opened.values())
    total_pt_closed = sum(time_pt_closed.values())
    total_pt_pct = (total_pt_closed / total_opened * 100) if total_opened > 0 else 0
    
    print("\n" + "="*70)
    print("TOTAL SUMMARY:")
    print(f"  - Total Opened: {total_opened}")
    print(f"  - Total Closed by Profit Target: {total_pt_closed}")
    print(f"  - Percentage: {total_pt_pct:.2f}%")
    
    # Find time slot with highest PT close rate
    if time_stats:
        highest_pt = max(time_stats, key=lambda x: x[3])
        print(f"\nHIGHEST PROFIT TARGET CLOSE RATE:")
        print(f"  - Time Slot: {highest_pt[0]}")
        print(f"  - Opened: {highest_pt[1]}")
        print(f"  - PT Closed: {highest_pt[2]} ({highest_pt[3]:.2f}%)")
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
        print("1. Which strategy works best? (Most effective) [Excludes PT closed]")
        print("2. Which time periods have faster closing trades? [Excludes PT closed]")
        print("3. Which strategy closes fastest by time period? [Excludes PT closed]")
        print("4. Run all analyses (1-5) [Excludes PT closed]")
        print("5. Power-based analysis (closing speed by power ranges) [Excludes PT closed]")
        print("6. Analyze currently open trades (hours open, power values)")
        print("7. Show total open transaction ratio")
        print("8. Analyze open trades by power ranges")
        print("9. Run all open trade analyses (6+7+8)")
        print("10. Analyze profit target closed trades by strategy")
        print("11. Analyze profit target closed trades by time period")
        print("12. Run all profit target analyses (10+11)")
        print("0. Exit")
        print("="*70)
        
        choice = input("\nEnter your choice (0-12): ").strip()
        
        if choice == '1':
            analyze_strategy_effectiveness(opened_trades, closed_trades)
        elif choice == '2':
            analyze_time_performance(opened_trades, closed_trades)
        elif choice == '3':
            analyze_strategy_time_performance(opened_trades, closed_trades)
        elif choice == '4':
            analyze_strategy_effectiveness(opened_trades, closed_trades)
            analyze_time_performance(opened_trades, closed_trades)
            analyze_strategy_time_performance(opened_trades, closed_trades)
            analyze_power_based_performance(opened_trades, closed_trades)
        elif choice == '5':
            analyze_power_based_performance(opened_trades, closed_trades)
        elif choice == '6':
            analyze_open_trades(opened_trades, closed_trades)
        elif choice == '7':
            analyze_open_transaction_ratio(opened_trades, closed_trades)
        elif choice == '8':
            analyze_open_trades_by_power(opened_trades, closed_trades)
        elif choice == '9':
            analyze_open_trades(opened_trades, closed_trades)
            analyze_open_transaction_ratio(opened_trades, closed_trades)
            analyze_open_trades_by_power(opened_trades, closed_trades)
        elif choice == '10':
            analyze_profit_target_by_strategy(opened_trades, closed_trades)
        elif choice == '11':
            analyze_profit_target_by_time(opened_trades, closed_trades)
        elif choice == '12':
            analyze_profit_target_by_strategy(opened_trades, closed_trades)
            analyze_profit_target_by_time(opened_trades, closed_trades)
        elif choice == '0':
            print("\nExiting... Goodbye!")
            break
        else:
            print("\nInvalid choice. Please select 0-12.")

if __name__ == '__main__':
    main()
