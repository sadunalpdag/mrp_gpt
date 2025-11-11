# UT/STC Strategy Backtest

## 📖 Overview

This script implements a backtesting system for the **UT/STC (Ultimate Schaff Trend Cycle)** trading strategy on Binance USD-M Futures market. It tests all USDT perpetual contracts using historical data.

## 🎯 Strategy Details

### Indicators
- **EMA13**: 13-period Exponential Moving Average
- **EMA50**: 50-period Exponential Moving Average  
- **STC**: Schaff Trend Cycle (fast=23, slow=50, cycle=10)
- **Additional**: RSI, ATR for reference

### Entry Signals

#### 🟢 BUY (LONG) Signal
```
Conditions:
1. EMA13 > EMA50 (uptrend confirmation)
2. STC crosses above 60 (from <= 60 to > 60)
```

#### 🔴 SELL (SHORT) Signal
```
Conditions:
1. EMA13 < EMA50 (downtrend confirmation)
2. STC crosses below 40 (from >= 40 to < 40)
```

### Risk Management
- **Take Profit (TP)**: 0.6% (conservative target)
- **Stop Loss (SL)**: 20% (wide SL for trend following)
- **Timeframe**: 1 hour candles
- **Backtest Period**: 90 days
- **Initial Balance**: $1000 per trade

## 🚀 Usage

### Basic Usage
```bash
python ut_stc.py
```

The script will automatically:
1. Fetch all USDT perpetual contract symbols from Binance
2. Download 90 days of 1-hour historical data for each
3. Apply the UT/STC strategy
4. Track TP/SL execution
5. Generate statistics and save results to CSV

### Output Files
- **Console**: Detailed statistics and top/bottom trades
- **CSV**: `ut_stc_backtest.csv` with all trade details

## 📊 Results Include

For each trade:
- `symbol`: Coin/contract name
- `direction`: UP (long) or DOWN (short)
- `entry_price`, `exit_price`: Entry and exit prices
- `entry_time`, `exit_time`: Entry and exit timestamps
- `exit_reason`: TP (take profit), SL (stop loss), or END (end of data)
- `tp_price`, `sl_price`: TP and SL levels
- `pnl_%`: Profit/loss percentage
- `final_balance_$`: Balance after trade
- `power`: Signal strength (>65 = strong)
- `rsi`, `atr`, `ema13`, `ema50`, `stc`: Indicator values

### Statistics Provided
- Total trades count
- Win rate and win/loss ratio
- TP/SL/END hit counts
- Average and total PnL
- UP/DOWN trade distribution
- Top 15 most profitable trades
- Bottom 10 worst trades

## ⚙️ Customization

Edit constants at the top of `ut_stc.py`:

```python
TIMEFRAME = "1h"        # Time interval ("1h", "4h", "1d", etc.)
DAYS = 90               # Backtest period in days
START_BALANCE = 1000    # Starting balance per trade
EMA13_PERIOD = 13       # EMA13 period
EMA50_PERIOD = 50       # EMA50 period
STC_FAST = 23          # STC fast period
STC_SLOW = 50          # STC slow period
STC_CYCLE = 10         # STC cycle period
```

## 📈 Strategy Logic

### What is STC (Schaff Trend Cycle)?
The Schaff Trend Cycle is an oscillator that identifies trend changes earlier than traditional indicators. It combines:
- MACD concept
- Stochastic oscillator smoothing
- Cycle analysis

**Key Levels:**
- Above 60: Bullish momentum (overbought)
- Below 40: Bearish momentum (oversold)
- Crossovers at these levels generate signals

### Why EMA + STC?
Combining EMA trend confirmation with STC timing creates a robust system:
- **EMA**: Confirms the overall trend direction
- **STC**: Times the entry at optimal moments
- Together: Reduces false signals and improves win rate

## 💡 Important Notes

1. **Rate Limiting**: Script automatically handles API rate limits
2. **Error Handling**: Failed symbols are skipped, backtest continues
3. **Simulation Only**: This is backtesting, not live trading
4. **Past Performance**: Historical results don't guarantee future returns
5. **Consistency**: Strategy matches implementation in `ema_margin.py`

## 🔧 Requirements

```bash
pip install pandas numpy requests
```

## 📝 Example Output

```
================================================================================
🚀 UT/STC STRATEGY BACKTEST
================================================================================
Strategy: EMA13 vs EMA50 + Schaff Trend Cycle
Timeframe: 1h | Period: 90 days
Buy: EMA13 > EMA50 AND STC crosses above 60
Sell: EMA13 < EMA50 AND STC crosses below 40
TP: 0.6% | SL: 20%
================================================================================

🔍 Binance Futures USDT sembolleri alınıyor...
📈 285 adet sembol bulundu.
⏳ Backtest başlıyor... (Bu işlem birkaç dakika sürebilir)

[1/285] BTCUSDT verisi çekiliyor... ✅ 🟢 UP   | Exit: TP  | PnL:    0.60% | Balance: $1006.00 | Power: 67.3
[2/285] ETHUSDT verisi çekiliyor... ⚠️ Hiç sinyal oluşmadı.
[3/285] BNBUSDT verisi çekiliyor... ✅ 🔴 DOWN | Exit: SL  | PnL:  -20.00% | Balance: $ 800.00 | Power: 72.1
...

================================================================================
📊 BACKTEST SONUÇLARI
================================================================================

📈 Genel İstatistikler:
  • Toplam İşlem: 127
  • Kazanan: 89 (70.1%)
  • Kaybeden: 38
  • TP Hit: 85 | SL Hit: 32 | End: 10
  • Ortalama PnL: 1.23%
  • Toplam PnL: 156.21%
  • Ortalama Kazanç: 0.58%
  • Ortalama Kayıp: -18.76%
```

## 🤝 Contributing

Issues and pull requests are welcome! Please check the project's GitHub page.

## 📄 License

See project root for license information.

## 🔗 Related Files

- `ema_margin.py` - Main trading bot with live UT/STC strategy
- `back.py` - Backtest for other strategies (LO_ORB, NYR, ICT_P3)
- `rep.py` - Analysis and reporting tools

---
**Created**: 2025-11-11  
**Strategy**: UT/STC (EMA13/EMA50 + Schaff Trend Cycle)  
**Purpose**: Backtest all coins for strategy validation
