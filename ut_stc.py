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
USE_KILLZONE_FILTER = True  # Kill Zone filtresi (London/NY seansları)
MAX_ZONE_USAGE = 2          # Aynı 4H bölgesi max kaç kere kullanılabilir
MAX_ZONE_AGE_BARS = 15      # 5m barında, 4H bölgesi max kaç bar eski olabilir

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


def is_killzone(dt):
    """
    Kill Zone kontrolü: London Open (08:00-10:00 UTC), NY Open (13:00-15:00 UTC), London Close (16:00-17:00 UTC)
    UTC-5 referansı var, ama kodda UTC kullanıyoruz.
    UTC bazlı kontrol yapıyoruz.
    """
    hour = dt.hour
    # London Open: 08:00-10:00 UTC
    # NY Open: 13:00-15:00 UTC
    # London Close: 16:00-17:00 UTC
    if (8 <= hour < 10) or (13 <= hour < 15) or (16 <= hour < 17):
        return True
    return False


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
    df_5m.index = df_5m.index.tz_convert("UTC")
    df_5m["close_time"] = df_5m["close_time"].dt.tz_convert("UTC")

    df_4h.index = df_4h.index.tz_convert("UTC")
    df_4h["close_time"] = df_4h["close_time"].dt.tz_convert("UTC")

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

    # Zone tracking: Her 4H bölgesi için kullanım sayısı ve ilk kullanım barı
    zone_usage = {}  # {h4_idx: {"count": int, "first_bar": int}}
    
    # Breakout tracking: Her 4H bölgesi için son breakout durumu
    # {"h4_idx": last_h4_idx, "side": "above"/"below"/"inside", "bar_idx": i, "breakout_low": float, "breakout_high": float}
    last_breakout = {"h4_idx": -1, "side": "inside", "bar_idx": -1, "breakout_low": None, "breakout_high": None}

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

        # Pozisyon ilişkisine göre konum
        def pos_rel(price):
            if price > h4_high:
                return "above"
            elif price < h4_low:
                return "below"
            else:
                return "inside"

        prev_close = closes[i - 1]
        curr_close = closes[i]
        prev_high = highs[i - 1]
        prev_low = lows[i - 1]

        prev_strong = strongs[i - 1]
        curr_strong = strongs[i]

        prev_rel = pos_rel(prev_close)
        curr_rel = pos_rel(curr_close)

        # 4H bölgesi değişti mi? (yeni bölge)
        if h4_idx_curr != h4_idx_prev:
            # Yeni bölge başladı, breakout durumunu resetle
            last_breakout = {"h4_idx": h4_idx_curr, "side": "inside", "bar_idx": i, "breakout_low": None, "breakout_high": None}

        # Zone freshness check: Bölge çok eski veya çok kullanılmış mı?
        if ref_idx not in zone_usage:
            zone_usage[ref_idx] = {"count": 0, "first_bar": i}
        
        zone_info = zone_usage[ref_idx]
        zone_age = i - zone_info["first_bar"]
        
        # Bölge çok eski (MAX_ZONE_AGE_BARS'dan fazla bar geçmiş) -> atla
        if zone_age > MAX_ZONE_AGE_BARS:
            continue
        
        # Bölge çok kullanılmış (MAX_ZONE_USAGE'dan fazla trade alınmış) -> atla
        if zone_info["count"] >= MAX_ZONE_USAGE:
            continue

        # Kill Zone filtresi
        if USE_KILLZONE_FILTER:
            current_time = times[i]
            if not is_killzone(current_time):
                # Kill Zone dışında, işlem alma
                continue

        # Breakout tracking: Fiyat bölgeden dışarı çıktı mı güçlü bir mumla?
        # LONG için: below'a çıkmalı (h4_low altına)
        # SHORT için: above'a çıkmalı (h4_high üstüne)
        
        # Breakout durumunu güncelle (sadece güçlü mumlarla)
        if last_breakout["h4_idx"] == h4_idx_curr:
            # Aynı bölgedeyiz, breakout durumu güncel mi kontrol et
            if prev_rel == "below" and prev_strong:
                # Güçlü bir mumla alt bölgeden çıktık
                last_breakout["side"] = "below"
                last_breakout["bar_idx"] = i - 1
                last_breakout["breakout_low"] = prev_low
                last_breakout["breakout_high"] = prev_high
            elif prev_rel == "above" and prev_strong:
                # Güçlü bir mumla üst bölgeden çıktık
                last_breakout["side"] = "above"
                last_breakout["bar_idx"] = i - 1
                last_breakout["breakout_low"] = prev_low
                last_breakout["breakout_high"] = prev_high

        # Re-entry kontrolü: Breakout olduktan sonra içeri dönüş
        # LONG setup: Önce below'da breakout, sonra inside'a güçlü dönüş
        long_signal = (
            last_breakout["h4_idx"] == h4_idx_curr and
            last_breakout["side"] == "below" and
            curr_rel == "inside" and
            curr_strong and
            trend_ok_long
        )

        # SHORT setup: Önce above'da breakout, sonra inside'a güçlü dönüş
        short_signal = (
            last_breakout["h4_idx"] == h4_idx_curr and
            last_breakout["side"] == "above" and
            curr_rel == "inside" and
            curr_strong and
            trend_ok_short
        )

        if not long_signal and not short_signal:
            continue

        # Giriş fiyatı: current close
        entry = curr_close

        if long_signal:
            # Stop: Breakout mumunun low'unun biraz altı
            stop = last_breakout["breakout_low"] if last_breakout["breakout_low"] is not None else h4_low
            risk = entry - stop
            if risk <= 0:
                continue
            tp = entry + RR_TARGET * risk
            pos_side = "long"

        elif short_signal:
            # Stop: Breakout mumunun high'ının biraz üstü
            stop = last_breakout["breakout_high"] if last_breakout["breakout_high"] is not None else h4_high
            risk = stop - entry
            if risk <= 0:
                continue
            tp = entry - RR_TARGET * risk
            pos_side = "short"

        else:
            continue

        # Zone usage'ı artır
        zone_usage[ref_idx]["count"] += 1

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
