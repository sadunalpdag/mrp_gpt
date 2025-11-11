
# -*- coding: utf-8 -*-
"""
Binance USD-M Futures Backtest - UT/STC Strategy
Veri kaynağı: https://fapi.binance.com/fapi/v1
Zaman aralığı: 90 gün, 1 saatlik mumlar (hourly candles)
Strateji: EMA13 vs EMA50 + Schaff Trend Cycle (STC)
  - Buy: EMA13 > EMA50 AND STC crosses above 60
  - Sell: EMA13 < EMA50 AND STC crosses below 40
  - TP: 0.6%, SL: 20% (matching ema_margin.py)
"""

import requests, time, pandas as pd, numpy as np
from datetime import datetime, timedelta, timezone

BASE_URL = "https://fapi.binance.com"
TIMEFRAME = "1h"  # Changed from 30m to 1h to match ema_margin.py
DAYS = 90  # Increased from 45 to 90 days for more data
START_BALANCE = 1000
EMA13_PERIOD = 13
EMA50_PERIOD = 50
STC_FAST = 23
STC_SLOW = 50
STC_CYCLE = 10
CSV_PATH = "ut_stc_backtest.csv"

# ========================== Yardımcı Fonksiyonlar ==========================

def get_futures_symbols() -> list:
    """Tüm USDT perpetual futures sembollerini getir"""
    url = f"{BASE_URL}/fapi/v1/exchangeInfo"
    r = requests.get(url, timeout=10).json()
    symbols = [s["symbol"] for s in r["symbols"]
               if s["quoteAsset"] == "USDT" and s["contractType"] == "PERPETUAL" and s["status"] == "TRADING"]
    return symbols

def get_klines(symbol: str, interval: str, days: int = 45) -> pd.DataFrame:
    """Binance Futures'tan 45 günlük 30m veri çeker (maks. 1500 mum per call)"""
    limit = 1500
    end = int(time.time() * 1000)
    start = end - days * 24 * 60 * 60 * 1000
    all_data = []
    while True:
        url = f"{BASE_URL}/fapi/v1/klines?symbol={symbol}&interval={interval}&limit={limit}&startTime={start}&endTime={end}"
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            print(f"❌ {symbol} hata: {r.text}")
            break
        data = r.json()
        if not data:
            break
        all_data += data
        first_time = int(data[0][0])
        last_time = int(data[-1][0])
        if len(data) < limit or last_time >= end:
            break
        start = last_time + 1
        time.sleep(0.2)
    if not all_data:
        return pd.DataFrame()
    df = pd.DataFrame(all_data, columns=[
        "time","open","high","low","close","volume",
        "close_time","qv","trades","tb_base","tb_quote","ignore"
    ])
    df = df[["time","open","high","low","close","volume"]].astype(float)
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    return df

def ema(vals, period):
    """Calculate EMA using list-based approach (matching ema_margin.py)"""
    k = 2 / (period + 1)
    e = [vals[0]]
    for v in vals[1:]:
        e.append(v * k + e[-1] * (1 - k))
    return e

def rsi(vals, period=14):
    """Calculate RSI indicator"""
    if len(vals) < period + 2:
        return [50] * len(vals)
    d = np.diff(vals)
    g = np.maximum(d, 0)
    l = -np.minimum(d, 0)
    ag = np.mean(g[:period])
    al = np.mean(l[:period])
    out = [50] * period
    for i in range(period, len(d)):
        ag = (ag * (period - 1) + g[i]) / period
        al = (al * (period - 1) + l[i]) / period
        rs = ag / al if al > 0 else 0
        out.append(100 - 100 / (1 + rs))
    return [50] * (len(vals) - len(out)) + out

def macd(vals, fast=12, slow=26, signal=9):
    """Calculate MACD indicator"""
    ema_fast = ema(vals, fast)
    ema_slow = ema(vals, slow)
    macd_line = np.array(ema_fast) - np.array(ema_slow)
    sig_line = ema(macd_line.tolist(), signal)
    hist = macd_line - np.array(sig_line)
    return macd_line.tolist(), sig_line, hist.tolist()

def schaff_tc(vals, fast=23, slow=50, cycle=10):
    """Calculate Schaff Trend Cycle (STC) indicator"""
    macd_line, _, _ = macd(vals, fast, slow, cycle)
    return rsi(macd_line, cycle)

def atr_like(h, l, c, period=14):
    """Calculate ATR-like indicator"""
    tr = []
    for i in range(len(h)):
        if i == 0:
            tr.append(h[i] - l[i])
        else:
            tr.append(max(h[i] - l[i], abs(h[i] - c[i-1]), abs(l[i] - c[i-1])))
    a = [sum(tr[:period]) / period]
    for i in range(period, len(tr)):
        a.append((a[-1] * (period - 1) + tr[i]) / period)
    return [0] * (len(h) - len(a)) + a

# ========================== Strateji Fonksiyonu ==========================

def backtest_symbol(df: pd.DataFrame, symbol: str) -> dict | None:
    """
    Backtest UT/STC strategy for a single symbol
    Strategy: EMA13 vs EMA50 + Schaff Trend Cycle
    - Buy: EMA13 > EMA50 AND STC crosses above 60
    - Sell: EMA13 < EMA50 AND STC crosses below 40
    """
    if df.empty or len(df) < 60:
        return None
    
    # Convert to lists for indicator calculation
    closes = df["close"].tolist()
    highs = df["high"].tolist()
    lows = df["low"].tolist()
    
    # Calculate indicators
    e13 = ema(closes, EMA13_PERIOD)
    e50 = ema(closes, EMA50_PERIOD)
    stc_vals = schaff_tc(closes, fast=STC_FAST, slow=STC_SLOW, cycle=STC_CYCLE)
    atr_vals = atr_like(highs, lows, closes)
    rsi_vals = rsi(closes)
    
    # Find entry signal
    entry_price = None
    entry_time = None
    direction = None
    entry_idx = None
    
    for i in range(60, len(df)):
        # Check for BUY signal: EMA13 > EMA50 AND STC crosses above 60
        if (e13[i] > e50[i] and 
            stc_vals[i] > 60 and stc_vals[i-1] <= 60):
            direction = "UP"
            entry_price = closes[i]
            entry_time = pd.Timestamp(df.iloc[i]["time"])
            entry_idx = i
            break
        
        # Check for SELL signal: EMA13 < EMA50 AND STC crosses below 40
        elif (e13[i] < e50[i] and 
              stc_vals[i] < 40 and stc_vals[i-1] >= 40):
            direction = "DOWN"
            entry_price = closes[i]
            entry_time = pd.Timestamp(df.iloc[i]["time"])
            entry_idx = i
            break
    
    if entry_price is None:
        return None
    
    # Calculate TP and SL based on ema_margin.py
    if direction == "UP":
        tp_price = entry_price * 1.006  # 0.6% TP
        sl_price = entry_price * 0.8    # 20% SL
    else:  # DOWN
        tp_price = entry_price * 0.994  # 0.6% TP
        sl_price = entry_price * 1.2    # 20% SL
    
    # Simulate trade execution: check for TP/SL hit
    exit_price = None
    exit_time = None
    exit_reason = None
    
    for i in range(entry_idx + 1, len(df)):
        current_high = highs[i]
        current_low = lows[i]
        current_close = closes[i]
        
        if direction == "UP":
            # Check TP hit
            if current_high >= tp_price:
                exit_price = tp_price
                exit_time = pd.Timestamp(df.iloc[i]["time"])
                exit_reason = "TP"
                break
            # Check SL hit
            elif current_low <= sl_price:
                exit_price = sl_price
                exit_time = pd.Timestamp(df.iloc[i]["time"])
                exit_reason = "SL"
                break
        else:  # DOWN
            # Check TP hit
            if current_low <= tp_price:
                exit_price = tp_price
                exit_time = pd.Timestamp(df.iloc[i]["time"])
                exit_reason = "TP"
                break
            # Check SL hit
            elif current_high >= sl_price:
                exit_price = sl_price
                exit_time = pd.Timestamp(df.iloc[i]["time"])
                exit_reason = "SL"
                break
    
    # If no TP/SL hit, exit at last candle
    if exit_price is None:
        exit_price = closes[-1]
        exit_time = pd.Timestamp(df.iloc[-1]["time"])
        exit_reason = "END"
    
    # Calculate PnL
    if direction == "UP":
        pnl_pct = (exit_price - entry_price) / entry_price * 100
    else:
        pnl_pct = (entry_price - exit_price) / entry_price * 100
    
    final_balance = START_BALANCE * (1 + pnl_pct / 100)
    
    # Calculate additional metrics
    power = 55 + abs(e13[entry_idx] - e50[entry_idx]) * 200 + (rsi_vals[entry_idx] - 50) / 2
    
    return {
        "symbol": symbol,
        "direction": direction,
        "entry_time": entry_time,
        "entry_price": round(entry_price, 6),
        "exit_time": exit_time,
        "exit_price": round(exit_price, 6),
        "exit_reason": exit_reason,
        "tp_price": round(tp_price, 6),
        "sl_price": round(sl_price, 6),
        "pnl_%": round(pnl_pct, 2),
        "final_balance_$": round(final_balance, 2),
        "power": round(power, 2),
        "rsi": round(rsi_vals[entry_idx], 2),
        "atr": round(atr_vals[entry_idx], 6),
        "ema13": round(e13[entry_idx], 6),
        "ema50": round(e50[entry_idx], 6),
        "stc": round(stc_vals[entry_idx], 2)
    }

# ========================== Ana Çalışma ==========================

def main():
    print("=" * 80)
    print("🚀 UT/STC STRATEGY BACKTEST")
    print("=" * 80)
    print("Strategy: EMA13 vs EMA50 + Schaff Trend Cycle")
    print(f"Timeframe: {TIMEFRAME} | Period: {DAYS} days")
    print(f"Buy: EMA13 > EMA50 AND STC crosses above 60")
    print(f"Sell: EMA13 < EMA50 AND STC crosses below 40")
    print(f"TP: 0.6% | SL: 20%")
    print("=" * 80)
    
    print("\n🔍 Binance Futures USDT sembolleri alınıyor...")
    symbols = get_futures_symbols()
    print(f"📈 {len(symbols)} adet sembol bulundu.")
    print(f"⏳ Backtest başlıyor... (Bu işlem birkaç dakika sürebilir)\n")
    
    results = []
    signals_found = 0
    tp_count = 0
    sl_count = 0
    end_count = 0

    for i, sym in enumerate(symbols, 1):
        try:
            print(f"[{i}/{len(symbols)}] {sym} verisi çekiliyor...", end=" ")
            df = get_klines(sym, TIMEFRAME, DAYS)
            if df.empty:
                print("⚠️ Veri yok.")
                continue
            
            res = backtest_symbol(df, sym)
            if not res:
                print("⚠️ Hiç sinyal oluşmadı.")
                continue
            
            signals_found += 1
            results.append(res)
            
            # Count exit reasons
            if res["exit_reason"] == "TP":
                tp_count += 1
            elif res["exit_reason"] == "SL":
                sl_count += 1
            else:
                end_count += 1
            
            # Display result with emoji
            emoji = "✅" if res["pnl_%"] > 0 else "❌"
            direction_emoji = "🟢" if res["direction"] == "UP" else "🔴"
            print(f"{emoji} {direction_emoji} {res['direction']:4s} | "
                  f"Exit: {res['exit_reason']:3s} | "
                  f"PnL: {res['pnl_%']:>7.2f}% | "
                  f"Balance: ${res['final_balance_$']:>8.2f} | "
                  f"Power: {res['power']:.1f}")
        except Exception as e:
            print(f"❌ Hata: {e}")
            continue

    print("\n" + "=" * 80)
    print("📊 BACKTEST SONUÇLARI")
    print("=" * 80)
    
    if not results:
        print("❌ Hiç işlem sonucu yok.")
        return
    
    df_res = pd.DataFrame(results)
    
    # Calculate statistics
    total_trades = len(results)
    winning_trades = len([r for r in results if r["pnl_%"] > 0])
    losing_trades = total_trades - winning_trades
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    
    avg_pnl = df_res["pnl_%"].mean()
    total_pnl = df_res["pnl_%"].sum()
    avg_win = df_res[df_res["pnl_%"] > 0]["pnl_%"].mean() if winning_trades > 0 else 0
    avg_loss = df_res[df_res["pnl_%"] < 0]["pnl_%"].mean() if losing_trades > 0 else 0
    
    print(f"\n📈 Genel İstatistikler:")
    print(f"  • Toplam İşlem: {total_trades}")
    print(f"  • Kazanan: {winning_trades} ({win_rate:.1f}%)")
    print(f"  • Kaybeden: {losing_trades}")
    print(f"  • TP Hit: {tp_count} | SL Hit: {sl_count} | End: {end_count}")
    print(f"  • Ortalama PnL: {avg_pnl:.2f}%")
    print(f"  • Toplam PnL: {total_pnl:.2f}%")
    print(f"  • Ortalama Kazanç: {avg_win:.2f}%")
    print(f"  • Ortalama Kayıp: {avg_loss:.2f}%")
    
    # Direction statistics
    up_trades = len([r for r in results if r["direction"] == "UP"])
    down_trades = len([r for r in results if r["direction"] == "DOWN"])
    print(f"\n📊 Yön Dağılımı:")
    print(f"  • UP (BUY): {up_trades}")
    print(f"  • DOWN (SELL): {down_trades}")
    
    # Sort by profitability
    df_res_sorted = df_res.sort_values("final_balance_$", ascending=False)
    
    print(f"\n🏆 En Kârlı 15 İşlem:")
    print(df_res_sorted.head(15)[["symbol", "direction", "exit_reason", "pnl_%", "final_balance_$", "power"]].to_string(index=False))
    
    print(f"\n💸 En Zararlı 10 İşlem:")
    print(df_res_sorted.tail(10)[["symbol", "direction", "exit_reason", "pnl_%", "final_balance_$", "power"]].to_string(index=False))
    
    # Save results
    df_res_sorted.to_csv(CSV_PATH, index=False)
    print(f"\n💾 Tüm sonuçlar kaydedildi: {CSV_PATH}")
    print("=" * 80)

if __name__ == "__main__":
    main()

# ========================== USAGE INSTRUCTIONS ==========================
"""
UT/STC STRATEGY BACKTEST - KULLANIM KILAVUZU

📖 GENEL BİLGİ:
Bu script, EMA13 vs EMA50 + Schaff Trend Cycle (STC) stratejisini kullanarak
Binance USD-M Futures market'inde tüm USDT perpetual kontratları için 
backtest yapar.

🎯 STRATEJİ DETAYLARI:
- Timeframe: 1 saat (hourly candles)
- Backtest Süresi: 90 gün
- Göstergeler:
  * EMA13 (13 periyotluk Exponential Moving Average)
  * EMA50 (50 periyotluk Exponential Moving Average)
  * STC (Schaff Trend Cycle: fast=23, slow=50, cycle=10)
  * RSI, ATR (ek bilgi için)

📊 SİNYAL KURALLARI:
✅ BUY (LONG) Sinyali:
  - EMA13 > EMA50 (uptrend)
  - STC bir önceki barda <= 60
  - STC şu anki barda > 60 (yukarı kesişme)

✅ SELL (SHORT) Sinyali:
  - EMA13 < EMA50 (downtrend)
  - STC bir önceki barda >= 40
  - STC şu anki barda < 40 (aşağı kesişme)

💰 RİSK YÖNETİMİ:
- Take Profit (TP): %0.6 (conservative)
- Stop Loss (SL): %20 (geniş SL, trend takibi için)
- Başlangıç Bakiyesi: $1000 per trade

🚀 NASIL KULLANILIR:
1. Temel kullanım:
   $ python ut_stc.py

2. Script otomatik olarak:
   - Tüm USDT perpetual kontratları listeler
   - Her biri için 90 günlük 1h veri çeker
   - UT/STC stratejisini uygular
   - TP/SL takibi yapar
   - Sonuçları CSV'ye kaydeder

📁 ÇIKTILAR:
- Konsol çıktısı: Detaylı istatistikler ve en iyi/kötü işlemler
- CSV dosyası: "ut_stc_backtest.csv" (tüm işlem detayları)

📈 ÇIKTI İÇERİĞİ:
Her işlem için:
- symbol: Coin/kontrat adı
- direction: UP (long) veya DOWN (short)
- entry_price, exit_price: Giriş ve çıkış fiyatları
- entry_time, exit_time: Giriş ve çıkış zamanları
- exit_reason: TP (kar al), SL (zarar kes), END (veri sonu)
- tp_price, sl_price: TP ve SL seviyeleri
- pnl_%: Kar/zarar yüzdesi
- final_balance_$: İşlem sonrası bakiye
- power: Sinyal gücü (>65 = güçlü)
- rsi, atr, ema13, ema50, stc: Gösterge değerleri

📊 İSTATİSTİKLER:
- Toplam işlem sayısı
- Kazanan/kaybeden işlem sayısı ve oranı
- TP/SL/END hit sayıları
- Ortalama ve toplam PnL
- UP/DOWN işlem dağılımı
- En karlı ve zararlı işlemler

⚙️ ÖZELLEŞTIRME:
Üstteki sabitleri değiştirerek:
- TIMEFRAME: Zaman dilimi ("1h", "4h", "1d", vb.)
- DAYS: Backtest süresi
- START_BALANCE: Başlangıç bakiyesi
- EMA13_PERIOD, EMA50_PERIOD: EMA periyotları
- STC_FAST, STC_SLOW, STC_CYCLE: STC parametreleri

💡 NOTLAR:
1. Script API rate limiting için otomatik olarak bekler
2. Hata durumunda işlem atlanır ve devam edilir
3. Gerçek trade değil, sadece backtest (simülasyon)
4. Sonuçlar geçmiş performans gösterir, gelecek garantisi değildir
5. ema_margin.py ile tutarlı strateji implementasyonu

🔍 EK BİLGİ:
- STC (Schaff Trend Cycle): Trend dönüşlerini erken yakalayan oscillator
- 60 seviyesi üstü = bullish momentum
- 40 seviyesi altı = bearish momentum
- EMA crossover ile birleşince güçlü sinyaller üretir

📞 DESTEK:
Sorular için projenin GitHub sayfasına bakın veya issue açın.
"""
