# Fractal Support & Resistance Strategy

A fractally validated zone trading system built around clustered swing highs/lows instead of single-point pivots.

## 🧩 Core Concept

Instead of drawing dozens of random support/resistance lines, this method clusters multiple fractal points within an ATR-scaled window. A zone only becomes "validated" if enough fractals (e.g., ≥ 3) form near the same level.

- **Green zones**: Support (clustered low fractals)
- **Orange zones**: Resistance (clustered high fractals)
- **Validation count**: Number near each zone (3 = 3 lows clustered together)

This filters noise and gives only high-conviction areas where price has repeatedly reacted.

## ⚙️ Indicator Settings

Based on TradingView "Fractal Support & Resistance" by Big Beluga:

| Setting | Meaning | Recommended |
|---------|---------|-------------|
| Fractals QTY | Min. same-side fractals to form a zone | 3 default |
| Support/Resistance Fractals QTY | Separate thresholds for longs/shorts | Optional |
| Type | Zone placement: Average (center) or Extreme (edge) | Average = balanced, Extreme = precise |
| Markers / Boxes / Lines | Visualization options | All ON |

## 🧠 Trade Logic

### 🟩 Long Setup

1. Validated green support zone with count ≥ 3
2. Price moves away (≈ 8–12 candles without touching)
3. On retest, a wick only dips into the zone
4. Wait for the fractal marker to confirm after 5 bars
5. **Entry**: Next bar after confirmation
6. **Stop-Loss**: Just below the zone (or bottom of box)
7. **Take-Profit**: 1.5 × risk

### 🟧 Short Setup

Mirror the above:

1. Validated orange resistance zone, count ≥ 3
2. Gap → wick retest → confirmed high fractal
3. **Entry**: Next bar
4. **SL**: Above zone
5. **TP**: 1.5 × risk

## ⚡ Market-Specific Tuning

| Market | Fractals QTY | Type | Notes |
|--------|--------------|------|-------|
| Forex | 3 | Average | Focus on London/NY sessions |
| **Crypto** | **2** | **Extreme** | Faster zones, volatile swings |
| Stocks | 4 | Average | Stricter validation |
| Gold (XAU) | 3 | Extreme | Catches sharp reversals |

## 🚀 Usage

### Quick Test (Crypto - 2 symbols, 30 days)
```bash
python3 fractal_sr.py --quick --symbols BTC ETH
```

### Full Backtest (All symbols, 90 days, crypto settings)
```bash
python3 fractal_sr.py --fractals-qty 2 --zone-type Extreme --days 90
```

### Custom Parameters
```bash
# Forex settings (3 fractals, Average)
python3 fractal_sr.py --fractals-qty 3 --zone-type Average --days 90

# Stocks settings (4 fractals, Average)
python3 fractal_sr.py --fractals-qty 4 --zone-type Average --days 90

# Specific symbols
python3 fractal_sr.py --symbols BTC ETH SOL AVAX --days 60
```

## 📊 Command Line Options

```
--quick              Quick test (20 symbols, 30 days)
--symbols SYMBOLS    Specific symbols to test (e.g., BTC ETH SOL)
--days DAYS          Number of days to backtest (default: 90)
--fractals-qty N     Min fractals per zone (default: 2 for crypto)
--zone-type TYPE     'Average' or 'Extreme' (default: Extreme for crypto)
```

## 📈 Output Files

- **fractal_sr_trades.json**: Detailed trade log with entry/exit prices, P&L
- **fractal_sr_results.json**: Statistics including win rate, avg P&L, breakdown by direction and zone count

## 🧪 Testing

Run the test suite to verify strategy logic:

```bash
python3 test_fractal_sr.py
```

Tests include:
- Fractal detection (swing highs/lows)
- ATR calculation
- Zone clustering algorithm
- Zone validation
- Trade signal generation
- Full backtest simulation

## 📐 Strategy Implementation Details

### Fractal Detection
Uses 5-bar pattern to identify swing highs and lows:
- **Fractal High**: Middle bar's high is highest among surrounding 2 bars on each side
- **Fractal Low**: Middle bar's low is lowest among surrounding 2 bars on each side

### Zone Clustering
Groups fractals within ATR-scaled window:
- **Cluster tolerance**: 1.5 × ATR
- **Zone price**: Average of all fractals (Average type) or first fractal (Extreme type)
- **Zone boundaries**: ±0.5 ATR from zone price

### Entry Conditions
1. Zone validated (min fractals requirement met)
2. Price moved away for 8+ bars
3. Wick-only retest (low touches support zone but close above, or high touches resistance but close below)
4. Wait 5 bars for fractal confirmation
5. Enter on next bar

### Risk Management
- **Stop Loss**: Just outside zone boundary
- **Take Profit**: 1.5R (1.5 times risk)
- Position sizing can be adjusted based on account size

## 🎯 Strategy Advantages

1. **Noise Filtering**: Only trades validated zones with multiple confirmations
2. **High Conviction**: Multiple fractals = stronger support/resistance
3. **Objective Rules**: No subjective line drawing
4. **Adaptable**: Different parameters for different markets
5. **Mean Reversion**: Captures price reactions at key levels

## 📝 Notes

- Strategy works best in ranging or choppy markets where price respects levels
- May underperform in strong trending markets
- Backtesting with crypto-optimized parameters (2 fractals, Extreme type)
- Real-time trading would require live data feed and order execution system

## 🔗 References

- Based on TradingView "Fractal Support & Resistance" indicator by Big Beluga
- Fractal concept originated from Bill Williams' trading methodology
- Enhanced with ATR-based clustering for better zone identification
