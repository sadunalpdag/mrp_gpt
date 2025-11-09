# MRP GPT - Automated Trading System

## Veri Çekme Süreleri (Data Retrieval Periods)

Bu sistem, farklı amaçlar için farklı sürelerde geçmiş veri çeker:

### 1. Backtest Sistemi (`back.py`)
- **Süre**: **90 gün (3 ay)**
- **Aralık**: 1 saatlik (1h) mumlar
- **Amaç**: Yeni trading stratejilerini (LO_ORB, NYR, ICT_P3) test etmek
- **Toplam Bar Sayısı**: ~2,160 bar (90 gün × 24 saat)

```python
# Örnek kullanım:
python3 back.py                    # Tam backtest (tüm semboller, 90 gün)
python3 back.py --quick            # Hızlı test (20 sembol, 7 gün)
python3 back.py --days 60          # Özel süre (60 gün)
```

### 2. Fibonacci Scalping Stratejisi (`ut_stc.py`)
- **Süre**: **30 gün**
- **Aralık**: 1 dakikalık (1m) mumlar
- **Amaç**: 1 dakikalık Fibonacci Golden Zone Scalping backtesti
- **Toplam Bar Sayısı**: ~43,200 bar (30 gün × 24 saat × 60 dakika)

### 3. 15 Günlük UT/STC Analizi (`python ut_stc_15day.py`)
- **Süre**: **15 gün**
- **Aralık**: 5 dakikalık (5m) mumlar
- **Amaç**: Ultimate Oscillator ve Schaff Trend Cycle stratejisi analizi
- **Toplam Bar Sayısı**: ~4,320 bar (15 gün × 24 saat × 12)

### 4. Ana Trading Sistemi (`ema_margin.py`)
- **Süre**: **200 saatlik bar** (~8.3 gün)
- **Aralık**: 1 saatlik (1h) mumlar
- **Amaç**: Gerçek zamanlı trading sinyalleri üretmek
- **Stratejiler**: 
  - MACD (EMA20/200 + MACD crossover)
  - FVG (Fair Value Gap Break)
  - EMA PULLBACK (EMA200 + EMA9/30)
  - KIVANC CONFIRM (SuperTrend + EMA9/30)
  - C.E.S.T. (50 MA Double Top/Bottom)
  - ORB + FVG CONFIRM (Opening Range Breakout)
  - LONDON BREAKOUT (LO Session ORB)
  - NY REVERSAL (Liquidity Sweep)
  - ICT POWER OF 3
  - ASIAN RANGE BREAKOUT
  - FVG + BREAKER BLOCK

## Özet Tablo

| Dosya | Süre | Aralık | Bar Sayısı | Amaç |
|-------|------|--------|------------|------|
| `back.py` | 90 gün | 1h | ~2,160 | Backtest |
| `ut_stc.py` | 30 gün | 1m | ~43,200 | Fibonacci Scalping |
| `python ut_stc_15day.py` | 15 gün | 5m | ~4,320 | UT/STC Analizi |
| `ema_margin.py` | ~8.3 gün (200 bar) | 1h | 200 | Canlı Trading |

## Notlar

1. **Backtest sisteminde** (`back.py`), `--days` parametresi ile özel süre belirlenebilir.
2. **Ana trading sistemi** (`ema_margin.py`) her 30 saniyede bir çalışır ve sadece son 200 saatlik veriyi kullanır.
3. Tüm sistemler Binance Futures API'sini kullanarak USDT paritelerinden veri çeker.
4. API rate limiting nedeniyle sistemler arasında 0.1 saniye bekleme süresi vardır.

## API Kullanımı

Sistemler şu API endpoint'lerini kullanır:
- **Kline Data**: `GET /fapi/v1/klines` - Mum (candlestick) verisi
- **Exchange Info**: `GET /fapi/v1/exchangeInfo` - Sembol bilgileri
- **Position Risk**: `GET /fapi/v2/positionRisk` - Pozisyon durumu
- **Orders**: `POST /fapi/v1/order` - Emir gönderme

## English Summary

### Data Retrieval Periods

This system fetches different periods of historical data for different purposes:

1. **Backtest System** (`back.py`): **90 days** (3 months) of 1h candles
2. **Fibonacci Scalping** (`ut_stc.py`): **30 days** of 1m candles
3. **UT/STC Analysis** (`python ut_stc_15day.py`): **15 days** of 5m candles
4. **Live Trading System** (`ema_margin.py`): **~8.3 days** (200 bars of 1h candles)

See table above for complete breakdown.
