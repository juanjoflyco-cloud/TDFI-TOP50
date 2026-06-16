"""
TDFI Screener - Top 50 por Market Cap (1h)
-------------------------------------------
Obtiene el top 50 de criptomonedas por market cap desde CoinGecko,
calcula el TDFI (Trend Direction & Force Index) sobre velas de 1h
y envía una alerta a Telegram SOLO cuando alguna moneda pasa de zona
NEUTRAL a zona VERDE (>+0.05) o zona ROJA (<-0.05) en vela cerrada.

Si en una hora ninguna moneda da señal, no se envía ningún mensaje.
"""

import os
import json
import time
import requests
import numpy as np
import pandas as pd

# ── Configuración ─────────────────────────────────────────────────────────────
INTERVAL       = "1h"
LOOKBACK       = 13
MMA_LEN        = 13
SMMA_LEN       = 13
N_POWER        = 3
ZONE_THRESHOLD = 0.05
KLINES_LIMIT   = 200
DELAY_SECONDS  = 3      # delay entre peticiones para no superar rate limit de CoinGecko
TOP_N          = 50     # número de monedas a escanear

STATE_FILE = "state.json"

TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")


# ── CoinGecko: obtener top N por market cap ───────────────────────────────────
def fetch_top_coins(n=TOP_N):
    """Devuelve lista de dicts con {id, symbol, name} para las top N monedas."""
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        "vs_currency": "usd",
        "order": "market_cap_desc",
        "per_page": n,
        "page": 1,
        "sparkline": False,
    }
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    return r.json()  # lista de monedas con id, symbol, name, current_price...


# ── CoinGecko: obtener precios horarios de una moneda ────────────────────────
def fetch_hourly_prices(coin_id, days=9, limit=KLINES_LIMIT):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {"vs_currency": "usd", "days": str(days), "interval": "hourly"}
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
    prices = data["prices"]  # [[timestamp_ms, price], ...]
    df = pd.DataFrame(prices, columns=["open_time", "close"])
    df["close"] = df["close"].astype(float)
    df["open_time"] = df["open_time"].astype(np.int64)
    df["close_time"] = (
        df["open_time"].shift(-1)
        .fillna(df["open_time"].iloc[-1] + 3_600_000)
        .astype(np.int64) - 1
    )
    return df.tail(limit).reset_index(drop=True)


# ── Indicador TDFI ────────────────────────────────────────────────────────────
def calc_tdfi(closes, lookback=LOOKBACK, mma_len=MMA_LEN, smma_len=SMMA_LEN, n=N_POWER):
    s = pd.Series(closes)
    mma = s.ewm(span=mma_len, adjust=False).mean()
    smma = mma.ewm(span=smma_len, adjust=False).mean()
    impetmma  = mma.diff()
    impetsmma = smma.diff()
    divma     = (mma - smma).abs()
    averimpet = (impetmma + impetsmma) / 2
    tdf       = divma * (averimpet ** n)
    roll_max  = tdf.abs().rolling(lookback * n).max()
    tdfi      = tdf / roll_max.replace(0, np.nan)
    return tdfi


def get_zone(value):
    if value > ZONE_THRESHOLD:
        return "green"
    if value < -ZONE_THRESHOLD:
        return "red"
    return "neutral"


# ── Estado persistente ────────────────────────────────────────────────────────
def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {}   # dict vacío: {coin_id: {last_close_time, last_zone}}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


# ── Telegram ──────────────────────────────────────────────────────────────────
def send_telegram(message):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️  Telegram no configurado.")
        print(message)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    resp = requests.post(
        url,
        data={"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"},
        timeout=15,
    )
    if not resp.ok:
        print("Error enviando a Telegram:", resp.status_code, resp.text)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    now_ms = int(time.time() * 1000)
    state  = load_state()

    # 1. Obtener top 50
    print(f"Obteniendo top {TOP_N} monedas por market cap...")
    coins = fetch_top_coins(TOP_N)
    time.sleep(DELAY_SECONDS)

    signals_green = []
    signals_red   = []
    min_candles   = MMA_LEN + SMMA_LEN + LOOKBACK * N_POWER + 5

    # 2. Escanear cada moneda
    for i, coin in enumerate(coins):
        coin_id = coin["id"]
        symbol  = coin["symbol"].upper()
        name    = coin["name"]
        price   = coin.get("current_price", 0)

        print(f"[{i+1}/{TOP_N}] {symbol} ({coin_id})...", end=" ")

        try:
            df = fetch_hourly_prices(coin_id)
            closed = df[df["close_time"] < now_ms].reset_index(drop=True)

            if len(closed) < min_candles:
                print("pocas velas, omitida")
                time.sleep(DELAY_SECONDS)
                continue

            tdfi        = calc_tdfi(closed["close"].values)
            last_value  = tdfi.iloc[-1]
            last_close_time = int(closed["close_time"].iloc[-1])

            if pd.isna(last_value):
                print("warm-up incompleto")
                time.sleep(DELAY_SECONDS)
                continue

            current_zone  = get_zone(last_value)
            coin_state    = state.get(coin_id, {})
            previous_zone = coin_state.get("last_zone")
            prev_close    = coin_state.get("last_close_time")

            print(f"TDFI={last_value:.4f} zona={current_zone} (antes={previous_zone})")

            # Solo actuar si es una vela nueva
            if prev_close != last_close_time:
                if previous_zone == "neutral" and current_zone == "green":
                    signals_green.append({
                        "symbol": symbol, "name": name,
                        "tdfi": last_value, "price": price,
                    })
                elif previous_zone == "neutral" and current_zone == "red":
                    signals_red.append({
                        "symbol": symbol, "name": name,
                        "tdfi": last_value, "price": price,
                    })

                # Actualizar estado
                state[coin_id] = {
                    "last_close_time": last_close_time,
                    "last_zone": current_zone,
                }

        except Exception as e:
            print(f"Error: {e}")

        time.sleep(DELAY_SECONDS)

    # 3. Guardar estado
    save_state(state)

    # 4. Enviar alertas solo si hay señales
    total_signals = len(signals_green) + len(signals_red)

    if total_signals == 0:
        print("Sin señales esta hora. No se envía mensaje.")
        return

    lines = [f"🔔 <b>TDFI Screener</b> · Top {TOP_N} · {INTERVAL}\n"]

    if signals_green:
        lines.append("🟢 <b>ALCISTAS</b> (neutral → verde)")
        for s in signals_green:
            lines.append(
                f"  • <b>{s['symbol']}</b> ({s['name']})\n"
                f"    TDFI: {s['tdfi']:.4f} | Precio: ${s['price']:,.4f}"
            )

    if signals_red:
        if signals_green:
            lines.append("")
        lines.append("🔴 <b>BAJISTAS</b> (neutral → rojo)")
        for s in signals_red:
            lines.append(
                f"  • <b>{s['symbol']}</b> ({s['name']})\n"
                f"    TDFI: {s['tdfi']:.4f} | Precio: ${s['price']:,.4f}"
            )

    message = "\n".join(lines)
    print("\n" + message)
    send_telegram(message)


if __name__ == "__main__":
    main()
