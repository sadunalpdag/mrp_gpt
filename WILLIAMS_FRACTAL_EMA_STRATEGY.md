# Williams Fractal + 20/50/100 EMA Scalping Strategy

A professional implementation of the Williams Fractal scalping strategy combined with triple EMA trend filtering for cryptocurrency futures trading.

## 📋 Table of Contents

- [Overview](#overview)
- [Strategy Rules](#strategy-rules)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Strategy Logic](#strategy-logic)
- [Testing](#testing)
- [Performance Notes](#performance-notes)

## 🎯 Overview

This strategy combines **Williams Fractals** (period 2) with a **triple EMA system** (20/50/100) to identify high-probability scalping entries. The strategy is designed primarily for **1-minute timeframes** but works on 3m, 5m, and 15m with fewer signals.

### Why This Strategy Works

Most scalpers enter on random fractal signals. This system is different because it combines:

1. ✅ **Trend locked by MA ordering** - No false signals in choppy markets
2. ✅ **Fractal = micro reversal signal** - Precise entry timing
3. ✅ **Pullback = better entry price** - Trade from value zones
4. ✅ **Dynamic SL based on MA levels** - Safe stop placement
5. ✅ **1.5R reward** - Aggressive but reasonable profit target
6. ✅ **1m gives many signals** - But only trades WITH the trend

## 📌 Strategy Rules

### Timeframe
- **Primary**: 1 minute (1m) - Most signals
- **Secondary**: 3m, 5m, 15m - Fewer but potentially more reliable signals

### Indicators

1. **Williams Fractals (Period 2)**
   - Green arrow (fractal low) → Long signal
   - Red arrow (fractal high) → Short signal

2. **Moving Averages (EMA)**
   - MA1 = 20 (fast - green)
   - MA2 = 50 (medium - yellow)
   - MA3 = 100 (slow - red)

## 🔵 LONG Entry Rules

### 1. Trend Filter (CRITICAL)
- **20 MA > 50 MA > 100 MA**
- MAs must NOT cross each other
- Clear uptrend required

### 2. Price Pullback Required
Two valid scenarios:
- **Scenario A**: Price dips below 20 MA (but stays above 50 MA)
- **Scenario B**: Price dips below 50 MA (but stays above 100 MA)

⚠️ **CRITICAL INVALIDATION RULE**: 
If price goes below 100 MA → Signal is **INVALID** even if green fractal appears!  
Reason: Trend is broken, continuation unlikely.

### 3. Williams Fractal Signal
- Wait for green arrow (fractal low) to appear
- This confirms the micro reversal

### 4. Stop Loss Placement
- **Scenario A**: 1-2 ticks below 50 MA
- **Scenario B**: 1 tick below 100 MA

### 5. Take Profit
- **Risk:Reward = 1:1.5**
- If risk is $10, profit target is $15

## 🔴 SHORT Entry Rules

### 1. Trend Filter (CRITICAL)
- **100 MA > 50 MA > 20 MA**
- Exact opposite of long trend
- Perfect downtrend required

### 2. Price Pullback Required
- Price rises above 20 MA
- Creates a retracement opportunity

⚠️ **CRITICAL INVALIDATION RULE**: 
If price goes above 100 MA → Signal is **INVALID**!  
Reason: Downtrend is broken.

### 3. Williams Fractal Signal
- Wait for red arrow (fractal high) to appear
- Confirms micro reversal to downside

### 4. Stop Loss Placement
- 2 ticks above 50 MA

### 5. Take Profit
- **Risk:Reward = 1:1.5**

## 🚀 Installation

### Prerequisites
```bash
Python 3.7+
pip install pandas numpy requests
```

### Quick Start
```bash
# Clone the repository
cd /path/to/mrp_gpt

# Install dependencies
pip install -r requirements.txt

# Run the strategy backtest
python3 williams_fractal_ema_strategy.py
```

## 📊 Usage

### Basic Usage
```bash
python3 williams_fractal_ema_strategy.py
```

This will:
1. Fetch the top 20 liquid USDT perpetual futures from Binance
2. Download 90 days of 1-minute data
3. Run the backtest on all symbols
4. Display comprehensive statistics

### Running Tests
```bash
python3 test_williams_fractal_ema.py
```

All 6 unit tests should pass:
- ✅ EMA calculation
- ✅ Fractal detection
- ✅ Trend validation
- ✅ Pullback scenario detection
- ✅ Stop loss calculation
- ✅ Full strategy logic

## ⚙️ Configuration

Edit the settings in `williams_fractal_ema_strategy.py`:

```python
# Backtest period
DAYS_BACK = 90                    # How many days to backtest

# Capital management
INITIAL_CAPITAL = 5000.0          # Starting capital (USD)
POSITION_SIZE_USD = 100.0         # Position size per trade

# Strategy parameters
RISK_REWARD_RATIO = 1.5           # R:R ratio (1:1.5 recommended)
FRACTAL_PERIOD = 2                # Williams Fractal period
EMA_FAST = 20                     # Fast EMA
EMA_MEDIUM = 50                   # Medium EMA
EMA_SLOW = 100                    # Slow EMA

# Symbol selection
MAX_SYMBOLS = 20                  # Test top N symbols by volume
                                  # Set to None for all USDT perpetuals
```

## 🔍 Strategy Logic Deep Dive

### Williams Fractal Detection

A fractal is formed when a bar's high/low is the highest/lowest among surrounding bars:

```
Fractal Low (Green Arrow):
         /\
        /  \
       /    \
      /      \___  ← Current bar low is lowest among 2 bars before and 2 bars after
     /           \
```

```
Fractal High (Red Arrow):
     ___
    /   \
   /     \        ← Current bar high is highest among 2 bars before and 2 bars after
  /       \
 /         \
```

### Trend Filtering Logic

The strategy only takes trades when MAs are properly aligned:

**Long Trend (Uptrend)**:
```
Price ▲
20 MA  ━━━━━━━  (green)
50 MA  ━━━━━━━  (yellow)
100 MA ━━━━━━━  (red)
```

**Short Trend (Downtrend)**:
```
100 MA ━━━━━━━  (red)
50 MA  ━━━━━━━  (yellow)
20 MA  ━━━━━━━  (green)
Price ▼
```

### Entry Timing

The strategy waits for:
1. **Established trend** (MA alignment)
2. **Pullback** (price retraces to MA)
3. **Fractal signal** (micro reversal confirmation)

This 3-step process filters out most false signals.

### Position Sizing

Currently uses fixed position size:
```python
POSITION_SIZE_USD = 100.0  # $100 per trade
```

For dynamic position sizing based on risk:
```python
risk_per_trade = entry_price - stop_loss
position_size = RISK_AMOUNT / risk_per_trade
```

## 🧪 Testing

The strategy includes comprehensive unit tests covering:

1. **EMA Calculation**: Verifies exponential moving average computation
2. **Fractal Detection**: Tests Williams Fractal identification logic
3. **Trend Validation**: Confirms MA ordering checks work correctly
4. **Pullback Scenarios**: Validates scenario A/B detection and invalidation
5. **Stop Loss Calculation**: Tests dynamic SL placement
6. **Full Strategy**: End-to-end signal generation test

### Test Results
```
================================================================================
WILLIAMS FRACTAL + EMA STRATEGY - UNIT TESTS
================================================================================

Testing EMA calculation...
  ✓ EMA calculation works correctly
Testing Fractal detection...
  Found 28 high fractals and 22 low fractals
  ✓ Fractal detection works correctly
Testing trend validation...
  ✓ Trend validation works correctly
Testing pullback scenarios...
  ✓ Pullback scenario detection works correctly
Testing stop loss calculation...
  ✓ Stop loss calculation works correctly
Testing full strategy logic...
  Found 18 valid signals (10 long, 8 short)
  ✓ Full strategy logic works correctly

================================================================================
TEST RESULTS: 6/6 passed
✓ All tests passed!
================================================================================
```

## 📈 Performance Notes

### What to Expect

**1-Minute Timeframe:**
- ✅ High frequency signals (many opportunities)
- ⚠️ Requires tight spreads and low slippage
- ⚠️ Higher commission impact
- 💡 Best for liquid pairs (BTC, ETH, BNB)

**5-Minute Timeframe:**
- ✅ Fewer but more reliable signals
- ✅ Lower commission impact
- ✅ More forgiving of spreads
- 💡 Good balance for most traders

**15-Minute Timeframe:**
- ✅ Most reliable signals
- ✅ Minimal commission impact
- ⚠️ Fewer trading opportunities
- 💡 Best for conservative traders

### Optimization Tips

1. **Filter by volume**: Only trade highly liquid pairs
2. **Avoid news events**: Strategy works best in normal market conditions
3. **Consider time zones**: Asian session often has lower volatility
4. **Monitor spreads**: Wide spreads can eat into the 1.5R profit target
5. **Use limit orders**: Reduce slippage on entries

### Risk Management

The strategy includes several safety features:

- ✅ **Trend filter** prevents counter-trend trades
- ✅ **100 MA invalidation** stops trading in broken trends
- ✅ **Dynamic stop loss** adapts to market structure
- ✅ **Fixed R:R ratio** ensures consistent risk management

### Recommended Settings

For live trading:
```python
RISK_REWARD_RATIO = 1.5          # Don't change (tested value)
FRACTAL_PERIOD = 2               # Don't change (standard)
POSITION_SIZE_USD = 100.0        # Adjust to your capital
MAX_SYMBOLS = 10                 # Focus on most liquid
```

## 🎓 Strategy Psychology

### Why Traders Fail at Scalping

1. ❌ Enter on any fractal without trend filter
2. ❌ Don't wait for pullbacks (FOMO entries)
3. ❌ Ignore when 100 MA is breached
4. ❌ Move stop loss when trade goes against them
5. ❌ Don't stick to fixed R:R ratio

### How This Strategy Fixes It

1. ✅ **Strict trend filter** - Only trades WITH momentum
2. ✅ **Requires pullback** - Better entry prices
3. ✅ **100 MA rule** - Automatic invalidation
4. ✅ **Fixed stop loss** - No emotional decisions
5. ✅ **Fixed take profit** - No greed, no fear

## 📝 Example Trade Walkthrough

### Long Trade Example

1. **Pre-conditions**:
   - 20 EMA = $50,100
   - 50 EMA = $50,000
   - 100 EMA = $49,900
   - ✅ Trend valid: 20 > 50 > 100

2. **Pullback occurs**:
   - Price dips to $50,020
   - Below 20 MA but above 50 MA
   - ✅ Scenario A detected

3. **Fractal signal**:
   - Green arrow appears at $50,020
   - ✅ Entry triggered

4. **Position sizing**:
   - Entry: $50,020
   - Stop: $49,998 (2 ticks below 50 MA)
   - Risk: $22
   - Take profit: $50,020 + ($22 × 1.5) = $50,053
   - Position size: $100 / $50,020 ≈ 0.002 BTC

5. **Outcome**:
   - Price rallies to $50,053
   - ✅ Take profit hit
   - Profit: $33 (1.5R on $22 risk)

### What Would Invalidate This Trade

❌ If price dropped below $49,900 (100 MA) before the fractal appeared, the signal would be invalid even if green arrow showed up later.

## 🔒 Security & Best Practices

1. **API Keys**: Never commit API keys to version control
2. **Rate Limits**: The strategy respects Binance rate limits
3. **Error Handling**: Includes comprehensive error handling for network issues
4. **Data Validation**: Validates all data before processing

## 🤝 Contributing

This strategy is part of the `mrp_gpt` repository. To contribute:

1. Test thoroughly with paper trading first
2. Document any changes to strategy logic
3. Maintain backward compatibility
4. Add unit tests for new features

## 📜 License

This code is provided for educational purposes. Use at your own risk.

## ⚠️ Disclaimer

**IMPORTANT**: 
- This is a backtest implementation
- Past performance does not guarantee future results
- Always paper trade before using real money
- Consider transaction costs, slippage, and spreads
- Never risk more than you can afford to lose

## 📚 References

- Williams Fractals: Bill Williams' "Trading Chaos" methodology
- EMA Strategy: Advent Trading 3 MA system
- Original strategy description: Turkish scalping community

---

**Happy Trading! 🚀**

*Remember: The best strategy is the one you can follow consistently with proper risk management.*
