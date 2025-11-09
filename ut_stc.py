import ccxt
import pandas as pd
from datetime import datetime, timedelta, timezone
import time

# ==========================================================
#  TG CAPITAL BACKTEST — BINANCE FUTURES (USD-M)
# ==========================================================

# 🔧 1️⃣ Futures API’yı açıkça seçiyoruz
exchange = ccxt.binanceusdm()   # ✅ sadece USD-M (USDT bazlı) futures

NY_TZ = timezone(timedelta(hours=-5))
DAYS = 30
TIMEFRAME = '30m'
LIMIT = int((24 * 60 / 30) * DAYS)
EMA_PERIODS = [5, 9, 13, 21]

# ----------------------------------------------------------
def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def fetch_data(symbol):
    since = exchange.parse8601((datetime.now(timezone.utc) - timedelta(days=DAYS)).isoformat())
    data = exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME, since=since, limit=LIMIT)
    df = pd.DataFrame(data, columns=['time','open','high','low','close','volume'])
    df['time'] = pd.to_datetime(df['time'], unit='ms')
    df['ny_hour'] = df['time'].dt.tz_localize('UTC').dt.tz_convert('America/New_York').dt.hour
    df = df[(df['ny_hour']>=3) & (df['ny_hour']<=6)]   # London Kill Zone (NY saati)
    return df.reset_index(drop=True)

def detect_doji(df):
    body = abs(df['close'] - df['open'])
    rng = df['high'] - df['low']
    df['doji'] = body < rng * 0.25
    return df

def backtest(df):
    if df.empty or len(df) < 30:
        return pd.DataFrame(), 1000

    df = detect_doji(df)
    for p in EMA_PERIODS:
        df[f'ema{p}'] = ema(df['close'], p)

    balance = 1000
    entry = None
    trades = []

    for i in range(25, len(df)):
        row = df.iloc[i]
        prev3 = df.iloc[i-3:i]

        ema_ok = all(row[f'ema{EMA_PERIODS[j]}'] > row[f'ema{EMA_PERIODS[j+1]}'] for j in range(len(EMA_PERIODS)-1))
        ema_recent = all(
            all(prev3.iloc[k][f'ema{EMA_PERIODS[j]}'] > prev3.iloc[k][f'ema{EMA_PERIODS[j+1]}']
                for j in range(len(EMA_PERIODS)-1))
            for k in range(len(prev3))
        )
        doji_signal = prev3['doji'].any()

        if entry is None and ema_ok and ema_recent and doji_signal:
            entry = row['close']
            entry_time = row['time']
            continue

        if entry is not None and not ema_ok:
            exit_price = row['close']
            pnl = (exit_price - entry) / entry * 100
            balance *= (1 + pnl/100)
            trades.append({
                'entry_time': entry_time,
                'exit_time': row['time'],
                'entry': round(entry, 4),
                'exit': round(exit_price, 4),
                'pnl_%': round(pnl, 3)
            })
            entry = None

    return pd.DataFrame(trades), balance

# ----------------------------------------------------------
# 2️⃣ Futures marketlerini yükle (USD-M)
# ----------------------------------------------------------
markets = exchange.load_markets()
symbols = [s for s in markets if s.endswith("USDT")]

print(f"📈 Toplam {len(symbols)} USDT futures sembol bulundu.\n")

results_summary = []

for sym in symbols:
    try:
        print(f"🔹 {sym} verisi alınıyor...")
        df = fetch_data(sym)
        if df.empty:
            print("⚠️ Yetersiz veri (London aralığında mum yok).")
            continue
        trades, balance = backtest(df)
        if not trades.empty:
            avg_pnl = trades['pnl_%'].mean()
            print(f"✅ {len(trades)} trade | Ortalama PnL: {avg_pnl:.2f}% | Son bakiye: ${balance:.2f}")
            results_summary.append({
                'symbol': sym,
                'trades': len(trades),
                'avg_pnl': round(avg_pnl, 2),
                'final_balance': round(balance, 2)
            })
        else:
            print("⚠️ Sinyal bulunamadı.")
        time.sleep(0.5)
    except Exception as e:
        print(f"❌ {sym} hata: {e}")
        continue

# ----------------------------------------------------------
if results_summary:
    df_sum = pd.DataFrame(results_summary).sort_values('final_balance', ascending=False)
    print("\n📊 EN KÂRLI COINLER (30 gün, SL yok):\n")
    print(df_sum.head(10))
else:
    print("\n❌ Hiç trade oluşmadı.")
