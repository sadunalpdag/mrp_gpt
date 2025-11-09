# -*- coding: utf-8 -*-
"""
Binance USD-M Futures — 45 Günlük Backtest (Tüm USDT Perp Semboller)
Strateji: EMA(5>9>13>21) hizası + son 3 mumda doji (esnek), SL yok
- FIRST_SIGNAL_HOLD_TO_END=True: İlk sinyalde gir, veri sonuna kadar tut
- False: Ters sinyalde pozisyonu kapat (reversal exit)
Başlangıç sermayesi: 1000 USDT (her sembol bağımsız)
Zaman filtresi: New York 03:00–06:30 (London Kill Zone)
Zaman dilimi: 30m
Çıktı: Konsol + backtest_results.csv
"""

import ccxt
import pandas as pd
import time
from datetime import datetime, timedelta, timezone

# ===================== Ayarlar =====================
DAYS = 45
TIMEFRAME = "30m"
START_BALANCE = 1000.0
EMA_PERIODS = [5, 9, 13, 21]
DOJI_BODY_RATIO = 0.25      # body < range * ratio  => doji
USE_KILLZONE = True         # NY 03:00–06:30
FIRST_SIGNAL_HOLD_TO_END = True  # True: buy & hold; False: reversal exit
MAX_SYMBOLS = None          # None => tüm semboller; örn 60 yazıp sınırlayabilirsin
SAVE_CSV = True
CSV_PATH = "backtest_results.csv"

# ccxt borsası: USD-M (USDT bazlı perpetual)
exchange = ccxt.binanceusdm({
    "enableRateLimit": True,
})

NY_TZ = timezone(timedelta(hours=-5))
LIMIT = int((24 * 60 / 30) * DAYS)  # 30 dakikalık mum sayısı
SINCE_MS = exchange.parse8601((datetime.now(timezone.utc) - timedelta(days=DAYS)).isoformat())

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def detect_doji(df: pd.DataFrame, ratio: float = DOJI_BODY_RATIO) -> pd.DataFrame:
    body = (df["close"] - df["open"]).abs()
    rng = (df["high"] - df["low"]).replace(0, 1e-12)
    df["doji"] = body < (rng * ratio)
    return df

def in_killzone(ts_utc: pd.Timestamp) -> bool:
    if not USE_KILLZONE:
        return True
    ny = ts_utc.tz_localize("UTC").tz_convert("America/New_York")
    h = ny.hour
    m = ny.minute
    # 03:00 <= time <= 06:30
    if 3 <= h <= 5:
        return True
    if h == 6 and m <= 30:
        return True
    return False

def fetch_30m(symbol: str) -> pd.DataFrame:
    data = exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME, since=SINCE_MS, limit=LIMIT)
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data, columns=["time","open","high","low","close","volume"])
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    if USE_KILLZONE:
        df = df[df["time"].apply(in_killzone)]
    df.reset_index(drop=True, inplace=True)
    return df

def compute_emas(df: pd.DataFrame) -> None:
    for p in EMA_PERIODS:
        df[f"ema{p}"] = ema(df["close"], p)

def ema_aligned(row) -> bool:
    # ema5 > ema9 > ema13 > ema21
    return all(row[f"ema{EMA_PERIODS[i]}"] > row[f"ema{EMA_PERIODS[i+1]}"] for i in range(len(EMA_PERIODS)-1))

def backtest_symbol(df: pd.DataFrame) -> dict | None:
    if df.empty or len(df) < 50:
        return None

    df = detect_doji(df)
    compute_emas(df)

    entry_price = None
    entry_time = None
    final_price = None
    pnl_pct = 0.0
    trades = []

    for i in range(max(EMA_PERIODS)+5, len(df)):
        row = df.iloc[i]
        prev3 = df.iloc[i-3:i]

        ema_ok_now = ema_aligned(row)
        ema_ok_prev3 = all(ema_aligned(prev3.iloc[k]) for k in range(len(prev3)))
        doji_signal = prev3["doji"].any()

        # İlk sinyalde giriş
        if entry_price is None and ema_ok_now and ema_ok_prev3 and doji_signal:
            entry_price = float(row["close"])
            entry_time = pd.Timestamp(row["time"])
            if FIRST_SIGNAL_HOLD_TO_END:
                break  # buy & hold modunda ilk sinyal yeter
            else:
                # reversal modunda giriş yap, gezmeye devam
                trades.append({"side":"LONG", "entry_time":entry_time, "entry":entry_price})
                continue

        # reversal exit modu
        if not FIRST_SIGNAL_HOLD_TO_END and trades and trades[-1].get("exit") is None:
            # Hizalama bozulursa çık
            if not ema_ok_now:
                exit_price = float(row["close"])
                trades[-1]["exit_time"] = pd.Timestamp(row["time"])
                trades[-1]["exit"] = exit_price

    if FIRST_SIGNAL_HOLD_TO_END:
        if entry_price is None:
            return None
        final_price = float(df.iloc[-1]["close"])
        pnl_pct = (final_price - entry_price) / entry_price * 100.0
        final_balance = START_BALANCE * (1 + pnl_pct/100.0)
        return {
            "mode": "BUY_AND_HOLD_TO_END",
            "entry_time": entry_time,
            "entry_price": round(entry_price, 6),
            "exit_time": pd.Timestamp(df.iloc[-1]["time"]),
            "exit_price": round(final_price, 6),
            "pnl_%": round(pnl_pct, 3),
            "final_balance_$": round(final_balance, 2),
        }
    else:
        # reversal: son kapanışta açık trade varsa onu da kapat
        if trades and trades[-1].get("exit") is None:
            trades[-1]["exit_time"] = pd.Timestamp(df.iloc[-1]["time"])
            trades[-1]["exit"] = float(df.iloc[-1]["close"])

        if not trades:
            return None

        balance = START_BALANCE
        completed = [t for t in trades if t.get("exit") is not None]
        if not completed:
            return None

        for t in completed:
            pnl = (t["exit"] - t["entry"]) / t["entry"] * 100.0
            balance *= (1 + pnl/100.0)

        avg_pnl = sum(((t["exit"] - t["entry"]) / t["entry"] * 100.0) for t in completed) / len(completed)
        return {
            "mode": "REVERSAL_EXIT",
            "trades": len(completed),
            "avg_trade_pnl_%": round(avg_pnl, 3),
            "final_balance_$": round(balance, 2),
            "first_entry_time": completed[0]["entry_time"],
            "last_exit_time": completed[-1]["exit_time"],
        }

def main():
    markets = exchange.load_markets()
    symbols = [s for s, info in markets.items() if s.endswith("/USDT")]
    # İsteğe bağlı sınırlama
    if MAX_SYMBOLS:
        symbols = symbols[:MAX_SYMBOLS]

    print(f"📈 Toplam {len(symbols)} USDT futures sembol bulundu (USD-M).")
    results = []

    for idx, sym in enumerate(symbols, 1):
        try:
            print(f"[{idx}/{len(symbols)}] 🔹 {sym} verisi alınıyor...")
            df = fetch_30m(sym)
            if df.empty:
                print("   ⚠️ Veri yok / killzone filtresinde mum yok.")
                continue
            res = backtest_symbol(df)
            if res is None:
                print("   ⚠️ Sinyal/işlem oluşmadı.")
                continue
            row = {"symbol": sym} | res
            results.append(row)
            if "final_balance_$" in row:
                print(f"   ✅ Son Bakiye: ${row['final_balance_$']} | PnL: {row.get('pnl_%', row.get('avg_trade_pnl_%'))}%")
            time.sleep(0.25)  # nazik hız limiti
        except Exception as e:
            print(f"   ❌ Hata: {e}")
            continue

    if not results:
        print("\n❌ Hiç sonuç yok (veri/sinyal bulunamadı).")
        return

    df_res = pd.DataFrame(results)
    if FIRST_SIGNAL_HOLD_TO_END:
        df_res = df_res.sort_values("final_balance_$", ascending=False)
        cols = ["symbol","final_balance_$","pnl_%","entry_time","exit_time","entry_price","exit_price"]
    else:
        df_res = df_res.sort_values("final_balance_$", ascending=False)
        cols = ["symbol","final_balance_$","avg_trade_pnl_%","trades","first_entry_time","last_exit_time"]

    print("\n📊 EN KÂRLI 20 SEMBOL:")
    print(df_res[cols].head(20).to_string(index=False))

    if SAVE_CSV:
        df_res.to_csv(CSV_PATH, index=False)
        print(f"\n💾 Sonuçlar kaydedildi: {CSV_PATH}")

if __name__ == "__main__":
    main()
