import ccxt
import pandas as pd
from datetime import datetime, timedelta, timezone
import time

# ===============================
# TG CAPITAL STYLE BACKTEST
# ===============================

exchange = ccxt.binance({
    'options': {'defaultType': 'future'}
})

# New York time (UTC-5)
NY_TZ = timezone(timedelta(hours=-5))

# ===============================
# SETTINGS
# ===============================
days = 30
timeframe = '30m'
limit = int((24 * 60 / 30) * days)  # 48 mum/gün * 30 gün = 1440 mum
ema_periods = [5, 9, 13, 21]
rr_ratio = 20

# ===============================
# FUNCTIONS
# ===============================

def fetch_data(symbol):
    """Binance'ten 30 günlük 30m veri çek"""
    since = exchange.parse8601((datetime.now(timezone.utc) - timedelta(days=days)).isoformat())
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['time','open','high','low','close','volume'])
    df['time'] = pd.to_datetime(df['time'], unit='ms')
    return df

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def detect_trident(df):
    """Doji + FVG yapısı kontrolü"""
    df['body'] = abs(df['close'] - df['open'])
    df['range'] = df['high'] - df['low']
    df['doji'] = (df['body'] / df['range']) < 0.25
    return df

def backtest(df):
    df = detect_trident(df)
    for p in ema_periods:
        df[f'ema{p}'] = ema(df['close'], p)

    balance = 1000
    position = None
    entry = 0
    results = []

    for i in range(30, len(df)):
        now = df.iloc[i]
        prev = df.iloc[i-1]

        # EMA sıralaması → long bias
        bullish = all(df.iloc[i][f'ema{ema_periods[j]}'] > df.iloc[i][f'ema{ema_periods[j+1]}'] for j in range(len(ema_periods)-1))
        bearish = all(df.iloc[i][f'ema{ema_periods[j]}'] < df.iloc[i][f'ema{ema_periods[j+1]}'] for j in range(len(ema_periods)-1))

        # Giriş koşulu
        if position is None and bullish and prev['doji']:
            entry = now['close']
            position = 'long'
            entry_time = now['time']
            continue

        # Çıkış koşulu → EMA dizilimi bozulunca (SL yok)
        if position == 'long' and not bullish:
            pnl = (now['close'] - entry) / entry * 100
            balance *= (1 + pnl/100)
            results.append({
                'entry_time': entry_time,
                'exit_time': now['time'],
                'entry_price': entry,
                'exit_price': now['close'],
                'pnl_%': pnl
            })
            position = None

    return pd.DataFrame(results), balance

# ===============================
# RUN
# ===============================
markets = exchange.load_markets()
symbols = [s for s in markets if s.endswith("USDT") and ":USDT" not in s and markets[s].get('future')]
symbols = symbols[:20]  # örnek olarak ilk 20 futures

for sym in symbols:
    print(f"\n🟩 {sym} için 30 günlük veri çekiliyor...")
    try:
        df = fetch_data(sym)
        results, balance = backtest(df)
        print(f"{sym} sonuç: {len(results)} trade, final balance: ${balance:.2f}")
        if not results.empty:
            print(results.tail(3))
        time.sleep(0.8)
    except Exception as e:
        print(f"❌ {sym} hata: {e}")
        continue
