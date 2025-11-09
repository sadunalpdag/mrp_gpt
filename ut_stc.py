import ccxt
import pandas as pd
from datetime import datetime, timedelta, timezone
import time

# ==========================================================
#  TG CAPITAL BACKTEST — NO STOPLOSS / NO EXIT
# ==========================================================
exchange = ccxt.binanceusdm()  # ✅ USD-M Futures (USDT bazlı)
NY_TZ = timezone(timedelta(hours=-5))

DAYS = 30
TIMEFRAME = '30m'
LIMIT = int((24 * 60 / 30) * DAYS)
EMA_PERIODS = [5, 9, 13, 21]

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def fetch_data(symbol):
    since = exchange.parse8601((datetime.now(timezone.utc) - timedelta(days=DAYS)).isoformat())
    data = exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME, since=since, limit=LIMIT)
    df = pd.DataFrame(data, columns=['time','open','high','low','close','volume'])
    df['time'] = pd.to_datetime(df['time'], unit='ms')
    df['ny_hour'] = df['time'].dt.tz_localize('UTC').dt.tz_convert('America/New_York').dt.hour
    df = df[(df['ny_hour']>=3) & (df['ny_hour']<=6)]
    return df.reset_index(drop=True)

def detect_doji(df):
    body = abs(df['close'] - df['open'])
    rng = df['high'] - df['low']
    df['doji'] = body < rng * 0.25
    return df

def backtest(df):
    if df.empty or len(df) < 30:
        return None

    df = detect_doji(df)
    for p in EMA_PERIODS:
        df[f'ema{p}'] = ema(df['close'], p)

    entry = None
    entry_time = None
    entry_price = None

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
            entry_price = row['close']
            entry_time = row['time']
            break  # sadece ilk sinyal alınır

    if entry_price is not None:
        final_price = df.iloc[-1]['close']
        pnl = (final_price - entry_price) / entry_price * 100
        return {
            'entry_time': entry_time,
            'entry_price': round(entry_price, 4),
            'exit_time': df.iloc[-1]['time'],
            'exit_price': round(final_price, 4),
            'pnl_%': round(pnl, 3)
        }
    else:
        return None

# ----------------------------------------------------------
markets = exchange.load_markets()
symbols = [s for s in markets if s.endswith("USDT")]

print(f"📈 Toplam {len(symbols)} USDT futures sembol bulundu.\n")

results = []

for sym in symbols:
    try:
        print(f"🔹 {sym} verisi alınıyor...")
        df = fetch_data(sym)
        result = backtest(df)
        if result:
            print(f"✅ {sym} | PnL: {result['pnl_%']}%")
            result['symbol'] = sym
            results.append(result)
        else:
            print("⚠️ Sinyal bulunamadı.")
        time.sleep(0.5)
    except Exception as e:
        print(f"❌ {sym} hata: {e}")
        continue

if results:
    df_results = pd.DataFrame(results).sort_values('pnl_%', ascending=False)
    print("\n📊 EN KÂRLI 10 COIN (30 gün, NO STOPLOSS):\n")
    print(df_results[['symbol','pnl_%','entry_time','exit_time']].head(10))
else:
    print("\n❌ Hiç sinyal veya trade oluşmadı.")
