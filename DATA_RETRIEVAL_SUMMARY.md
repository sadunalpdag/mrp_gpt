# Veri Çekme Süresi Özeti / Data Retrieval Summary

## 🇹🇷 Türkçe

### Soru: "Kaç günlük data çekiyor bu sistem?"

Bu sistem **farklı bileşenlerinde farklı sürelerde** veri çeker:

#### 📊 Hızlı Cevap:

| Sistem | Gün Sayısı | Bar Sayısı | Mum Aralığı |
|--------|-----------|------------|-------------|
| 🔬 **Backtest (back.py)** | **90 gün** | ~2,160 | 1 saat |
| 📈 **Fibonacci Scalping (ut_stc.py)** | **30 gün** | ~43,200 | 1 dakika |
| 🔍 **UT/STC Analizi** | **15 gün** | ~4,320 | 5 dakika |
| 🤖 **Canlı Trading (ema_margin.py)** | **~8.3 gün** | 200 | 1 saat |

#### 📝 Detaylar:

1. **Backtest Sistemi** (`back.py`)
   - 🗓️ **90 gün** (3 ay) geriye gider
   - ⏱️ Her 1 saatte bir mum
   - 📊 Yaklaşık 2,160 bar veri
   - 🎯 Amaç: Yeni stratejileri test etmek
   - ⚙️ Özelleştirilebilir: `--days` parametresi ile

2. **Fibonacci Scalping** (`ut_stc.py`)
   - 🗓️ **30 gün** geriye gider
   - ⏱️ Her 1 dakikada bir mum
   - 📊 Yaklaşık 43,200 bar veri
   - 🎯 Amaç: 1 dakikalık scalping stratejisi test

3. **UT/STC Analizi** (`python ut_stc_15day.py`)
   - 🗓️ **15 gün** geriye gider
   - ⏱️ Her 5 dakikada bir mum
   - 📊 Yaklaşık 4,320 bar veri
   - 🎯 Amaç: Ultimate Oscillator ve Schaff Trend Cycle analizi

4. **Canlı Trading Sistemi** (`ema_margin.py`)
   - 🗓️ **~8.3 gün** geriye gider (200 saatlik bar)
   - ⏱️ Her 1 saatte bir mum
   - 📊 Tam 200 bar veri
   - 🎯 Amaç: Gerçek zamanlı trading sinyalleri
   - 🔄 Her 30 saniyede güncellenir

---

## 🇬🇧 English

### Question: "How many days of data does this system pull?"

This system pulls **different periods of data** in different components:

#### 📊 Quick Answer:

| System | Days | Bars | Candle Interval |
|--------|------|------|-----------------|
| 🔬 **Backtest (back.py)** | **90 days** | ~2,160 | 1 hour |
| 📈 **Fibonacci Scalping (ut_stc.py)** | **30 days** | ~43,200 | 1 minute |
| 🔍 **UT/STC Analysis** | **15 days** | ~4,320 | 5 minutes |
| 🤖 **Live Trading (ema_margin.py)** | **~8.3 days** | 200 | 1 hour |

#### 📝 Details:

1. **Backtest System** (`back.py`)
   - 🗓️ Looks back **90 days** (3 months)
   - ⏱️ 1-hour candles
   - 📊 Approximately 2,160 bars
   - 🎯 Purpose: Test new trading strategies
   - ⚙️ Customizable: Use `--days` parameter

2. **Fibonacci Scalping** (`ut_stc.py`)
   - 🗓️ Looks back **30 days**
   - ⏱️ 1-minute candles
   - 📊 Approximately 43,200 bars
   - 🎯 Purpose: 1-minute scalping strategy backtest

3. **UT/STC Analysis** (`python ut_stc_15day.py`)
   - 🗓️ Looks back **15 days**
   - ⏱️ 5-minute candles
   - 📊 Approximately 4,320 bars
   - 🎯 Purpose: Ultimate Oscillator and Schaff Trend Cycle analysis

4. **Live Trading System** (`ema_margin.py`)
   - 🗓️ Looks back **~8.3 days** (200 hourly bars)
   - ⏱️ 1-hour candles
   - 📊 Exactly 200 bars
   - 🎯 Purpose: Real-time trading signals
   - 🔄 Updates every 30 seconds

---

## 🔗 Additional Resources

- 📖 Full documentation: [README.md](README.md)
- 💻 Source code with inline comments in each file
- 🌐 Data source: Binance Futures API (`/fapi/v1/klines`)

## ⚠️ Important Notes

1. The **live trading system** (`ema_margin.py`) uses the least amount of historical data (200 bars) because it needs to run efficiently in real-time
2. The **backtest system** uses the most historical data (90 days) to ensure thorough strategy testing
3. All systems can handle API rate limits with built-in delays (0.1s between calls)
4. Data is fetched from Binance Futures for USDT pairs only

---

**Last Updated**: 2025-11-09
**Version**: 1.0
