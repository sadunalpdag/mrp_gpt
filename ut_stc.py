import ccxt, pandas as pd
from datetime import datetime, timedelta, timezone

# ---------------- SETTINGS ---------------- #
exchange = ccxt.binance()
symbol = "BTC/USDT"
timeframe = "1m"
days = 30
session_open_hour = 14  # 09:30 EST ≈ 14:30 UTC
session_open_minute = 30
rr_ratio = 2.0

# -------------- FETCH DATA ---------------- #
since = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)
df = pd.DataFrame(exchange.fetch_ohlcv(symbol, timeframe, since))
df.columns = ["ts","open","high","low","close","vol"]
df["time"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
df.set_index("time", inplace=True)

# -------------- STRATEGY ------------------ #
trades = []
pos = None

for date, group in df.groupby(df.index.date):
    # 09:30 - 09:35 UTC (adjust for local) -> first 5 min candle
    start = datetime(date.year, date.month, date.day, session_open_hour, session_open_minute, tzinfo=timezone.utc)
    end = start + timedelta(minutes=5)
    open_candles = group.loc[start:end]
    if len(open_candles)==0: 
        continue
    H0, L0 = open_candles["high"].max(), open_candles["low"].min()
    later = group.loc[end:]
    broke_high = broke_low = False
    entry, stop, target = None, None, None

    for t, r in later.iterrows():
        if not broke_high and r["close"]>H0:
            broke_high=True
        if not broke_low and r["close"]<L0:
            broke_low=True

        # long setup
        if broke_high and r["low"]<=H0 and r["close"]>H0 and entry is None:
            entry = r["close"]; stop = H0 - (H0-L0)*0.2; target = entry + (entry-stop)*rr_ratio
            pos="long"
        # short setup
        elif broke_low and r["high"]>=L0 and r["close"]<L0 and entry is None:
            entry = r["close"]; stop = L0 + (H0-L0)*0.2; target = entry - (stop-entry)*rr_ratio
            pos="short"

        # exits
        if pos=="long" and entry:
            if r["low"]<=stop: 
                trades.append(("LONG",entry,stop,stop-entry))
                break
            elif r["high"]>=target:
                trades.append(("LONG",entry,target,target-entry))
                break
        if pos=="short" and entry:
            if r["high"]>=stop:
                trades.append(("SHORT",entry,stop,entry-stop))
                break
            elif r["low"]<=target:
                trades.append(("SHORT",entry,target,entry-target))
                break

# -------------- RESULTS ------------------ #
if trades:
    df_trades = pd.DataFrame(trades, columns=["Side","Entry","Exit","PnL"])
    df_trades["PnL_%"] = df_trades["PnL"]/df_trades["Entry"]*100
    winrate = (df_trades["PnL_%"]>0).mean()*100
    print(df_trades)
    print(f"\nTrades: {len(df_trades)} | Win-rate: {winrate:.1f}% | "
          f"Avg PnL %: {df_trades['PnL_%'].mean():.2f}")
else:
    print("No trades detected in sample.")
