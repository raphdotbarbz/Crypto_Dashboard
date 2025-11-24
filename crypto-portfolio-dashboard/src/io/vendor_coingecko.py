from __future__ import annotations
import requests, pandas as pd
from typing import Optional

BASE = "https://pro-api.coingecko.com/api/v3"

class CoinGeckoError(RuntimeError): pass

def fetch_coingecko_daily_pro(
    coin_id: str,
    vs: str = "usd",
    *,
    api_key: str,
    start: Optional[str] = None,   # e.g. "2018-01-01"
    end: Optional[str] = None,     # e.g. "2025-11-17"
    days: str | int = "max"        # used if start/end not provided
) -> pd.DataFrame:
    headers = {"x-cg-pro-api-key": api_key}
    if start and end:
        url = f"{BASE}/coins/{coin_id}/market_chart/range"
        ts_from = int(pd.Timestamp(start, tz="UTC").timestamp())
        ts_to   = int(pd.Timestamp(end, tz="UTC").timestamp())
        params = {"vs_currency": vs, "from": ts_from, "to": ts_to}
    else:
        url = f"{BASE}/coins/{coin_id}/market_chart"
        params = {"vs_currency": vs, "days": days}

    r = requests.get(url, params=params, headers=headers, timeout=30)
    if r.status_code != 200:
        try:
            msg = r.json()
        except Exception:
            msg = r.text
        raise CoinGeckoError(f"HTTP {r.status_code}: {msg}")

    data = r.json()
    if "prices" not in data:
        raise CoinGeckoError("Malformed response: missing 'prices'.")

    df = pd.DataFrame(data["prices"], columns=["ts_ms", "price"])
    df["date"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True).dt.floor("D")
    df = df.drop(columns=["ts_ms"]).groupby("date", as_index=False).last().sort_values("date")
    return df[["date", "price"]]
