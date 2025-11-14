import time
import math
import requests
from datetime import datetime, timedelta, timezone

import pandas as pd

BINANCE_FAPI = "https://fapi.binance.com"

# ==========================
# AYARLAR
# ==========================
DAYS_BACK = 90              # Kaç gün geriye gideceğiz
RR_TARGET = 2.0             # Risk/Ödül oranı (TP = 2R)
BODY_THRESH = 0.6           # Güçlü mum gövde oranı (0-1)
USE_TREND_FILTER = True     # 4H trend filtresi (EMA'ya göre)
TREND_EMA_LEN = 50          # 4H EMA periyodu
MAX_SYMBOLS = None          # None = tüm USDT perpetual; sayı verirsen o kadarını alır (hacme göre sıralayıp)
REQUEST_SLEEP = 0.15        # Her API çağrısı sonrası bekleme (rate limit için)

# ==========================
# UTILS
# ==========================

def ms_since_epoch(dt: datetime) -> int:
    """Datetime -> ms timestamp (UTC)."""
    return int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000)


def get_futures_symbols():
    """
    Binance Futures exchangeInfo'dan
    USDT margined, PERPETUAL (süresiz) kontrat sembollerini çeker.
    Hacme göre sıralar.
    """
    url = f"{BINANCE_FAPI}/fapi/v1/exchangeInfo"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    symbols = []
    for s in data["symbols"]:
        if (
            s.get("quoteAsset") == "USDT"
            and s.get("contractType") == "PERPETUAL"
            and s.get("status") == "TRADING"
        ):
            symbols.append(s["symbol"])

    # 24h hacmi çekip hacme göre sıralayalım ki en likitlerden başlayalım
    tickers_url = f"{BINANCE_FAPI}/fapi/v1/ticker/24hr"
    r2 = requests.get(tickers_url, timeout=10)
    r2.raise_for_status()
    tickers = {t["symbol"]: float(t["volume"]) for t in r2.json()}

    symbols = [s for s in symbols if s in tickers]
    symbols.sort(key=lambda s: tickers[s], reverse=True)

    if MAX_SYMBOLS is not None:
        symbols = symbols[:MAX_SYMBOLS]

    return symbols


def fetch_klines(symbol, interval, start_ms, end_ms=None, limit=1500):
    """
    Binance kline endpointinden belirtilen interval için
    [start_ms, end_ms] aralığındaki tüm veriyi çeker.
    """
    url = f"{BINANCE_FAPI}/fapi/v1/klines"
    all_rows = []

    cur_start = start_ms
    while True:
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
            "startTime": cur_start,
        }
        if end_ms is not None:
            params["endTime"] = end_ms

        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code != 200:
            print(f"[{symbol}][{interval}] HTTP {resp.status_code}: {resp.text}")
            break

        rows = resp.json()
        if not rows:
            break

        all_rows.extend(rows)

        last_open_time = rows[-1][0]
        # Bir sonraki batch için startTime ilerlet
        next_start = last_open_time + 1
        if end_ms is not None and next_start > end_ms:
            break

        # Eğer gelen satır sayısı limitin altındaysa, devam edecek veri yoktur
        if len(rows) < limit:
            break

        cur_start = next_start
        time.sleep(REQUEST_SLEEP)  # Rate limit tedbiri

    if not all_rows:
        return pd.DataFrame()

    # DataFrame'e dönüştür
    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "num_trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ]
    df = pd.DataFrame(all_rows, columns=cols)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)

    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    df.set_index("open_time", inplace=True)
    return df


def ema(series, length):
    return series.ewm(span=length, adjust=False).mean()


# ==========================
# STRATEJİ BACKTEST
# ==========================

def backtest_symbol(symbol, start_dt, end_dt):
    """
    Tek bir sembol için 4H Range – 5m Re-entry strateji backtest'i.
    R bazlı PnL döndürür.
    """
    print(f"\n=== {symbol} için veri çekiliyor... ===")
    start_ms = ms_since_epoch(start_dt)
    end_ms = ms_since_epoch(end_dt)

    # 5m verisi
    df_5m = fetch_klines(symbol, "5m", start_ms, end_ms)
    if df_5m.empty:
        print(f"[{symbol}] 5m veri yok, atlanıyor.")
        return None

    # 4H verisi
    df_4h = fetch_klines(symbol, "4h", start_ms, end_ms)
    if df_4h.empty:
        print(f"[{symbol}] 4H veri yok, atlanıyor.")
        return None

    # 4H EMA & trend
    df_4h["close_ema"] = ema(df_4h["close"], TREND_EMA_LEN)
    df_4h["trend_up"] = df_4h["close"] > df_4h["close_ema"]
    df_4h["trend_down"] = df_4h["close"] < df_4h["close_ema"]

    # 4H mumların close_time'ını tut
    h4_index = df_4h.index
    h4_close_times = df_4h["close_time"].values

    # 5m DF'ye kolay erişim için kolonlar
    df_5m["body"] = (df_5m["close"] - df_5m["open"]).abs()
    df_5m["range"] = df_5m["high"] - df_5m["low"]
    df_5m["strong"] = (df_5m["range"] > 0) & (df_5m["body"] / df_5m["range"] >= BODY_THRESH)

    # 5m için hangi 4H mumunu kullanacağımızı bul (ref = son kapanmış 4H)
    ref_h4_idx_for_5m = []
    h4_i = 0
    num_h4 = len(df_4h)

    for t_close in df_5m["close_time"].values:
        # t_close zamanına kadar kapanmış son 4H mumu bul
        # (close_time <= t_close)
        while h4_i + 1 < num_h4 and df_4h["close_time"].iloc[h4_i + 1] <= t_close:
            h4_i += 1
        # Eğer hiç kapanmış 4H yoksa, -1
        if df_4h["close_time"].iloc[h4_i] <= t_close:
            ref_h4_idx_for_5m.append(h4_i)
        else:
            ref_h4_idx_for_5m.append(-1)

    df_5m["h4_idx"] = ref_h4_idx_for_5m

    # Trade simülasyonu
    in_position = False
    pos_side = None   # "long" veya "short"
    entry_price = None
    stop_price = None
    tp_price = None
    entry_time = None

    trades = []

    closes = df_5m["close"].values
    opens = df_5m["open"].values
    highs = df_5m["high"].values
    lows = df_5m["low"].values
    strongs = df_5m["strong"].values
    h4_idx_arr = df_5m["h4_idx"].values
    times = df_5m.index.to_list()

    for i in range(1, len(df_5m)):
        # Pozisyon açık ise önce SL/TP kontrolü
        if in_position:
            high_i = highs[i]
            low_i = lows[i]
            exit_reason = None
            exit_price = None

            if pos_side == "long":
                # Pessimistic: önce stop'a bak, sonra TP
                if low_i <= stop_price:
                    exit_price = stop_price
                    exit_reason = "SL"
                elif high_i >= tp_price:
                    exit_price = tp_price
                    exit_reason = "TP"
            else:  # short
                if high_i >= stop_price:
                    exit_price = stop_price
                    exit_reason = "SL"
                elif low_i <= tp_price:
                    exit_price = tp_price
                    exit_reason = "TP"

            if exit_reason is not None:
                # R hesabı
                if pos_side == "long":
                    risk_per_unit = entry_price - stop_price
                    pnl_per_unit = exit_price - entry_price
                else:
                    risk_per_unit = stop_price - entry_price
                    pnl_per_unit = entry_price - exit_price

                R = pnl_per_unit / risk_per_unit if risk_per_unit != 0 else 0.0

                trades.append({
                    "symbol": symbol,
                    "side": pos_side,
                    "entry_time": entry_time,
                    "exit_time": times[i],
                    "entry": entry_price,
                    "exit": exit_price,
                    "stop": stop_price,
                    "tp": tp_price,
                    "R": R,
                    "reason": exit_reason,
                })

                in_position = False
                pos_side = None
                entry_price = stop_price = tp_price = None
                entry_time = None

            # SL/TP olduktan sonra aynı bar’da yeni trade aramayalım
            # (istersen burada continue koyabilirsin)
            # continue

        # Pozisyon yoksa yeni setup arayalım
        if in_position:
            continue

        h4_idx_curr = h4_idx_arr[i]
        h4_idx_prev = h4_idx_arr[i - 1]

        # 4H referansı yoksa (henüz oluşmamış) atla
        if h4_idx_curr <= 0:
            continue

        # Bölge: son kapanmış 4H mumu (video: bir önceki mum)
        ref_idx = h4_idx_curr
        h4_row = df_4h.iloc[ref_idx]
        h4_high = h4_row["high"]
        h4_low = h4_row["low"]

        # Trend filtresi
        trend_ok_long = True
        trend_ok_short = True
        if USE_TREND_FILTER:
            trend_ok_long = bool(h4_row["trend_up"])
            trend_ok_short = bool(h4_row["trend_down"])

        # Aynı 4H bölgesinde miyiz? (breakout & re-entry için)
        if h4_idx_curr != h4_idx_prev:
            continue

        prev_close = closes[i - 1]
        curr_close = closes[i]

        prev_strong = strongs[i - 1]
        curr_strong = strongs[i]

        # Pozisyon ilişkisine göre konum
        def pos_rel(price):
            if price > h4_high:
                return "above"
            elif price < h4_low:
                return "below"
            else:
                return "inside"

        prev_rel = pos_rel(prev_close)
        curr_rel = pos_rel(curr_close)

        # LONG setup: önce aşağıda kapanış, sonra içeri güçlü mum
        long_signal = (
            prev_rel == "below" and
            curr_rel == "inside" and
            prev_strong and curr_strong and
            trend_ok_long
        )

        # SHORT setup: önce yukarıda kapanış, sonra içeri güçlü mum
        short_signal = (
            prev_rel == "above" and
            curr_rel == "inside" and
            prev_strong and curr_strong and
            trend_ok_short
        )

        if not long_signal and not short_signal:
            continue

        # Giriş fiyatı: current close
        entry = curr_close

        if long_signal:
            stop = h4_low
            risk = entry - stop
            if risk <= 0:
                continue
            tp = entry + RR_TARGET * risk
            pos_side = "long"

        elif short_signal:
            stop = h4_high
            risk = stop - entry
            if risk <= 0:
                continue
            tp = entry - RR_TARGET * risk
            pos_side = "short"

        else:
            continue

        in_position = True
        entry_price = entry
        stop_price = stop
        tp_price = tp
        entry_time = times[i]

    if not trades:
        print(f"[{symbol}] Hiç trade çıkmadı.")
        return None

    # Sonuçları özetle
    df_trades = pd.DataFrame(trades)
    total_R = df_trades["R"].sum()
    win_rate = (df_trades["R"] > 0).mean() * 100
    avg_R = df_trades["R"].mean()
    num_trades = len(df_trades)

    print(f"[{symbol}] İşlem sayısı: {num_trades}, Win rate: {win_rate:.1f}%, Toplam R: {total_R:.2f}, Ortalama R: {avg_R:.2f}")

    return df_trades


def main():
    end_dt = datetime.now(timezone.utc).replace(microsecond=0, second=0)
    start_dt = end_dt - timedelta(days=DAYS_BACK + 2)  # biraz pay

    symbols = get_futures_symbols()
    print(f"Toplam sembol (USDT PERPETUAL): {len(symbols)}\nSemboller: {symbols}")

    all_trades = []

    for sym in symbols:
        try:
            res = backtest_symbol(sym, start_dt, end_dt)
            if res is not None:
                all_trades.append(res)
        except Exception as e:
            print(f"[{sym}] HATA: {e}")
            continue

    if not all_trades:
        print("Hiç trade datası oluşmadı.")
        return

    df_all = pd.concat(all_trades, ignore_index=True)

    # Genel özet
    total_R = df_all["R"].sum()
    win_rate = (df_all["R"] > 0).mean() * 100
    avg_R = df_all["R"].mean()
    num_trades = len(df_all)

    print("\n==================== GENEL ÖZET ====================")
    print(f"Toplam işlem: {num_trades}")
    print(f"Genel Win rate: {win_rate:.1f}%")
    print(f"Toplam R: {total_R:.2f}")
    print(f"Ortalama R: {avg_R:.2f}")

    # Sembol bazlı özet
    sym_group = df_all.groupby("symbol")["R"].agg(["count", "sum", "mean"])
    sym_group = sym_group.sort_values("sum", ascending=False)
    print("\nSembol bazlı performans (ilk 30):")
    print(sym_group.head(30))


if __name__ == "__main__":
    main()
