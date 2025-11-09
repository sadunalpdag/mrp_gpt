import ccxt
import pandas as pd
from datetime import datetime, timedelta, timezone
import time

exchange = ccxt.binance({'options': {'defaultType': 'future'}})
NY_TZ = timezone(timedelta(hours=-5))

days = 30
timeframe = '30m'
limit = int((24 * 60 / 30) * days)

ema_periods = [5, 9, 13, 21]

def ema(series, period): return series.ewm(span=period, adjust=False).mean()

def fetch_data(symbol):
    since = exchange.parse8601((datetime.now(timezone.utc) - timedelta(days=days)).isoformat())
    data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
    df = pd.DataFrame(data, columns=['time','open','high','low','close','volume'])
    df['time'] = pd.to_datetime(df['time'], unit='ms')
    df['ny_hour'] = df['time'].dt.tz_localize('UTC').dt.tz_convert('America/New_York').dt.hour
    # London Kill Zone 03:00–06:30 NY
    df = df[(df['ny_hour']>=3) & (df['ny_hour']<=6)]
    return df.reset_index(drop=True)

def detect_doji(df):
    body = abs(df['close'] - df['open'])
    rng = df['high'] - df['low']
    df['doji'] = body < rng * 0.25
    return df

def backtest(df):
    df = detect_doji(df)
    for p in ema_periods:
        df[f'ema{p}'] = ema(df['close'], p)

    balance = 1000
    entry = None
    trades = []

    for i in range(25, len(df)):
        row = df.iloc[i]
        prev3 = df.iloc[i-3:i]

        ema_ok = all(row[f'ema{ema_periods[j]}'] > row[f'ema{ema_periods[j+1]}'] for j in range(len(ema_periods)-1))
        ema_trend_recent = all(
            all(prev3.iloc[k][f'ema{ema_periods[j]}'] > prev3.iloc[k][f'ema{ema_periods[j+1]}']
                for j in range(len(ema_periods)-1))
            for k in range(len(prev3))
        )
        doji_signal = prev3['doji'].any()

        if entry is None and ema_ok and ema_trend_recent and doji_signal:
            entry = row['close']
            entry_time = row['time']
            continue

        if entry is not None:
            if not ema_ok:
                exit_price = row['close']
                pnl = (exit_price - entry) / entry * 100
                balance *= (1 + pnl/100)
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': row['time'],
                    'entry': entry,
                    'exit': exit_price,
                    'pnl_%': round(pnl,3)
                })
                entry = None

    return pd.DataFrame(trades), balance

# ---------- MAIN ----------
symbols = ["BTC/USDT","ETH/USDT","BNB/USDT","SOL/USDT","AVAX/USDT","XRP/USDT"]

for sym in symbols:
    print(f"\n🟩 {sym} backtest başlıyor...")
    try:
        df = fetch_data(sym)
        if df.empty:
            print("⚠️ Yetersiz veri (London saat aralığına denk gelmedi).")
            continue
        trades, balance = backtest(df)
        print(f"{len(trades)} trade bulundu | Final Balance: ${balance:.2f}")
        if not trades.empty:
            print(trades.tail(3))
        time.sleep(0.7)
    except Exception as e:
        print(f"❌ Hata: {e}")
