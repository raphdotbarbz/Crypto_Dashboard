# src/io/vendor_coingecko.py
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime, timezone, timedelta
import os
import time
import logging

import requests
from requests.adapters import HTTPAdapter, Retry
import pandas as pd

# ---------------------------------------------------------------------
# Config (server.py exports these from settings.yaml, or set in your env)
# ---------------------------------------------------------------------
PRO_BASE = "https://pro-api.coingecko.com/api/v3"
FREE_BASE = "https://api.coingecko.com/api/v3"

API_KEY = os.getenv("COINGECKO_API_KEY")
BASE_OVERRIDE = os.getenv("COINGECKO_BASE_URL")  # optional override

log = logging.getLogger("coingecko")

def _cg_base() -> str:
    return BASE_OVERRIDE or (PRO_BASE if API_KEY else FREE_BASE)

def _cg_headers() -> dict:
    # Connection: close reduces mid-stream resets from some WAFs.
    h = {
        "accept": "application/json",
        "user-agent": "crypto-portfolio-dashboard/0.1 (+https://localhost)",
        "connection": "close",
    }
    if API_KEY:
        h["x-cg-pro-api-key"] = API_KEY
    return h

def _cg_params(extra: dict | None = None) -> dict:
    p = dict(extra or {})
    # Some deployments accept/require the key as a query param too
    if API_KEY:
        p["x_cg_pro_api_key"] = API_KEY
    return p

# One shared session with automatic retries/backoff
_SESSION = requests.Session()
_RETRIES = Retry(
    total=6, connect=6, read=6,
    backoff_factor=1.5,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"],
    respect_retry_after_header=True,
)
_SESSION.mount("https://", HTTPAdapter(max_retries=_RETRIES))
_SESSION.mount("http://", HTTPAdapter(max_retries=_RETRIES))

def _cg_get(path: str, *, params: dict | None = None, timeout: int = 30) -> requests.Response:
    """
    Robust GET to CoinGecko. Retries for rate limits, 5xx, and connection resets.
    Forces Pro base when a key is present.
    """
    base_url = _cg_base()
    url = f"{base_url}{path}"
    last_err: Exception | None = None

    for attempt in range(6):
        try:
            log.info("CG GET %s params=%s attempt=%d", url, params, attempt + 1)
            r = _SESSION.get(url, params=_cg_params(params), headers=_cg_headers(), timeout=timeout)
            if r.status_code in (429, 401):
                # 401 with a key? force Pro base next try
                if r.status_code == 401 and API_KEY and base_url != PRO_BASE:
                    base_url = PRO_BASE
                    url = f"{base_url}{path}"
                time.sleep(1.5 * (attempt + 1))
                continue
            r.raise_for_status()
            return r
        except (requests.exceptions.ConnectionError,
                requests.exceptions.ChunkedEncodingError,
                requests.exceptions.ReadTimeout) as e:
            last_err = e
            time.sleep(1.5 * (attempt + 1))
            continue

    raise RuntimeError(f"CoinGecko request failed after retries: {url} :: {last_err}")

# ---------------------------------------------------------------------
# Local cache paths
# ---------------------------------------------------------------------
COIN_MAP_PATH = Path("data/coin_map.csv")
PRICE_DIR = Path("data/prices/coingecko")
PRICE_DIR.mkdir(parents=True, exist_ok=True)

def cache_path_for(symbol: str) -> Path:
    """
    Parquet path for a symbol's daily price cache.
    Example: data/prices/coingecko/BTC.parquet
    """
    return PRICE_DIR / f"{str(symbol).upper().strip()}.parquet"

# ---------------------------------------------------------------------
# Coin mapping helpers
# ---------------------------------------------------------------------
def _read_coin_map() -> pd.DataFrame:
    if COIN_MAP_PATH.exists():
        df = pd.read_csv(COIN_MAP_PATH)
        if "symbol" in df and "coingecko_id" in df:
            df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
            df["coingecko_id"] = df["coingecko_id"].astype(str).str.strip()
            return df
    return pd.DataFrame(columns=["symbol", "coingecko_id", "name"])

def _write_coin_map(df: pd.DataFrame) -> None:
    df = df.drop_duplicates(subset=["symbol"], keep="first")
    df.to_csv(COIN_MAP_PATH, index=False)

def search_coingecko_symbol(symbol: str) -> List[dict]:
    q = symbol.strip().lower()
    r = _cg_get("/search", params={"query": q}, timeout=20)
    coins = r.json().get("coins", [])
    # Prefer exact symbol matches, then by market cap rank
    def score(c):
        exact = (str(c.get("symbol", "")).lower() == q)
        rank = c.get("market_cap_rank") or 10**9
        return (0 if exact else 1, rank)
    coins.sort(key=score)
    return coins

def resolve_coin_id(symbol: str) -> Tuple[str, str]:
    sym = symbol.upper().strip()
    cm = _read_coin_map()
    row = cm.loc[cm["symbol"] == sym]
    if not row.empty:
        return sym, str(row.iloc[0]["coingecko_id"])
    cands = search_coingecko_symbol(sym)
    if not cands:
        raise ValueError(f"No CoinGecko match for symbol '{sym}'")
    pick = cands[0]
    cg_id, name = pick["id"], pick.get("name", "")
    cm = pd.concat([cm, pd.DataFrame([{"symbol": sym, "coingecko_id": cg_id, "name": name}])], ignore_index=True)
    _write_coin_map(cm)
    return sym, cg_id

# ---------------------------------------------------------------------
# Price fetchers (CoinGecko)
# ---------------------------------------------------------------------
def fetch_daily_prices_series(coin_id: str, vs: str = "usd", days: str = "max") -> pd.Series:
    """
    Fetch daily close from /market_chart (prices). Returns UTC-normalized daily series.
    Tries 'max' then degrades to 365/180 if plan forbids max.
    """
    try_days = ["max", "365", "180"] if days == "max" else [days, "365", "180"]
    last_exc: Exception | None = None
    for d in try_days:
        try:
            r = _cg_get(f"/coins/{coin_id}/market_chart", params={"vs_currency": vs, "days": d}, timeout=45)
            prices = r.json().get("prices", [])
            if not prices:
                continue
            df = pd.DataFrame(prices, columns=["ms", "price"])
            idx = pd.to_datetime(df["ms"], unit="ms", utc=True).dt.normalize()
            s = pd.Series(pd.to_numeric(df["price"], errors="coerce").astype(float).values, index=idx, name="close")
            s = s.groupby(level=0).last().sort_index()
            if not s.empty:
                return s
        except Exception as e:
            last_exc = e
            continue
    raise RuntimeError(f"CoinGecko returned no prices for {coin_id}; last_error={last_exc}")

def fetch_range_from_cg(symbol: str,
                        start: datetime,
                        end: datetime,
                        vs: str = "usd",
                        chunk_days: int = 180) -> pd.DataFrame:
    """
    Fetch [start, end] in chunks via /coins/{id}/market_chart/range.
    Returns tidy df[['date','price']] at daily frequency. No forward-fill is persisted.
    """
    if start.tzinfo is None: start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:   end   = end.replace(tzinfo=timezone.utc)
    sym, coin_id = resolve_coin_id(symbol)

    frames: list[pd.DataFrame] = []
    cur = start
    while cur <= end:
        chunk_end = min(cur + timedelta(days=chunk_days), end)
        r = _cg_get(
            f"/coins/{coin_id}/market_chart/range",
            params={"vs_currency": vs, "from": int(cur.timestamp()), "to": int(chunk_end.timestamp())},
            timeout=60,
        )
        prices = r.json().get("prices", [])
        if prices:
            df = pd.DataFrame(prices, columns=["ms", "price"])
            df["date"] = pd.to_datetime(df["ms"], unit="ms", utc=True).dt.normalize()
            df = df.drop(columns=["ms"])
            frames.append(df)

        # Gentle pacing helps stay under rate-limits
        time.sleep(0.3)
        cur = chunk_end + timedelta(days=1)

    if not frames:
        return pd.DataFrame(columns=["date", "price"])

    out = pd.concat(frames, ignore_index=True)
    out["price"] = pd.to_numeric(out["price"], errors="coerce")
    out = (out.dropna(subset=["price"])
             .sort_values("date")
             .groupby("date", as_index=False)
             .last())
    return out[["date", "price"]]

# ---------------------------------------------------------------------
# Public: ensure_price_cache (append not overwrite; no ffill persisted)
# ---------------------------------------------------------------------
def ensure_price_cache(symbol: str,
                       days: str | int = "max",
                       refresh_window_days: int = 120,
                       force: bool = False) -> Path:
    """
    Ensure a parquet cache exists and is up to date for `symbol`.

    Behavior:
      - Appends new vendor rows and de-dups on date
      - NEVER persists forward-filled data
      - If cache looks current but file mtime is stale, fetch a recent window
      - `force=True` refreshes a recent window regardless of cache contents
    """
    path = cache_path_for(symbol)
    old = pd.DataFrame()

    if path.exists():
        try:
            old = pd.read_parquet(path)
            if "date" in old.columns:
                old["date"] = pd.to_datetime(old["date"], utc=True)
            else:
                old = pd.DataFrame()
        except Exception:
            old = pd.DataFrame()

    now = datetime.now(timezone.utc)
    end = now

    # Default incremental start
    if not old.empty:
        start = (old["date"].max() + pd.Timedelta(days=1))
    else:
        if isinstance(days, int):
            start = now - timedelta(days=days)
        else:
            start = datetime(2017, 1, 1, tzinfo=timezone.utc)

    # Heuristic: if file mtime is stale (>1 day), fetch a recent window anyway
    mtime_stale = False
    if path.exists():
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        mtime_stale = (now - mtime).days > 1

    if force or mtime_stale:
        start = max(now - timedelta(days=refresh_window_days), datetime(2017, 1, 1, tzinfo=timezone.utc))

    # If start > end (e.g., due to earlier FFill-polluted caches), still fetch a recent window
    if start.date() > end.date():
        start = max(now - timedelta(days=refresh_window_days), datetime(2017, 1, 1, tzinfo=timezone.utc))

    # Try chunked range first; if it fails, fall back to recent /market_chart
    try:
        new = fetch_range_from_cg(symbol, start, end)
    except Exception as e:
        log.warning("range fetch failed for %s (%s); falling back to /market_chart recent window", symbol, e)
        sym, coin_id = resolve_coin_id(symbol)
        s = fetch_daily_prices_series(coin_id, vs="usd", days=str(refresh_window_days))
        df = s.reset_index()
        df.columns = ["date", "price"]
        df["date"] = pd.to_datetime(df["date"], utc=True)
        new = df[(df["date"] >= start) & (df["date"] <= end)].copy()

    both = pd.concat([old, new], ignore_index=True)
    if "date" not in both.columns:
        both["date"] = pd.to_datetime(both.index, utc=True)

    both = (both.drop_duplicates(["date"])
               .sort_values("date")
               .reset_index(drop=True))

    path.parent.mkdir(parents=True, exist_ok=True)
    both.to_parquet(path, index=False)
    return path
