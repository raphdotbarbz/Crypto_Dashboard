# src/io/vendor_coingecko.py
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple
from datetime import datetime, timezone, timedelta
import os
import time
import logging

import requests
from requests.adapters import HTTPAdapter, Retry
import pandas as pd
import yaml

# ---------------------------------------------------------------------
# Config (reads committed key from config/settings.yaml or env)
# ---------------------------------------------------------------------
PRO_BASE = "https://pro-api.coingecko.com/api/v3"
FREE_BASE = "https://api.coingecko.com/api/v3"

BASE_OVERRIDE = os.getenv("COINGECKO_BASE_URL")  # optional override

def _settings():
    try:
        with open("config/settings.yaml", "r") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}

def _cfg(path: str, default=None):
    d = _settings()
    cur = d
    for k in path.split("."):
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
    return default if cur is None else cur

def _api_key() -> str:
    # prefer committed key; fallback to env
    return (
        _cfg("vendor_options.coingecko.api_key")
        or _cfg("coingecko.api_key")
        or os.getenv("COINGECKO_API_KEY", "")
    )

# Incremental refresh policy (env-tunable)
INCR_OVERLAP_DAYS = int(os.getenv("CG_INCR_OVERLAP_DAYS", "2"))        # refetch this many days back
REFRESH_WINDOW_DAYS = int(os.getenv("CG_REFRESH_WINDOW_DAYS", "120"))  # upper bound on recent window
ALWAYS_INCREMENTAL = os.getenv("CG_ALWAYS_INCREMENTAL", "1") != "0"    # always touch network if True

log = logging.getLogger("coingecko")

def _cg_base() -> str:
    key = _api_key()
    return BASE_OVERRIDE or (PRO_BASE if key else FREE_BASE)

def _cg_headers() -> dict:
    # Connection: close reduces mid-stream resets from some WAFs.
    h = {
        "accept": "application/json",
        "user-agent": "crypto-portfolio-dashboard/0.1 (+https://localhost)",
        "connection": "close",
    }
    key = _api_key()
    if key:
        h["x-cg-pro-api-key"] = key
    return h

def _cg_params(extra: dict | None = None) -> dict:
    p = dict(extra or {})
    # Some deployments accept/require the key as a query param too
    key = _api_key()
    if key:
        p["x_cg_pro_api_key"] = key
    return p

# One shared session with automatic retries/backoff
_SESSION = requests.Session()
_RETRIES = Retry(
    total=2, connect=2, read=2,
    backoff_factor=0.3,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"],
    respect_retry_after_header=True,
)
_SESSION.mount("https://", HTTPAdapter(max_retries=_RETRIES))
_SESSION.mount("http://", HTTPAdapter(max_retries=_RETRIES))

def _cg_get(path: str, *, params: dict | None = None, timeout: int | float = 30) -> requests.Response:
    """
    Robust GET to CoinGecko using the adapter-level retries only.
    """
    base_url = _cg_base()
    url = f"{base_url}{path}"
    log.info("CG GET %s params=%s", url, params)
    r = _SESSION.get(url, params=_cg_params(params), headers=_cg_headers(), timeout=timeout)
    # If Pro key exists but we somehow hit the free base, force Pro and retry once
    if r.status_code == 401 and _api_key() and base_url != PRO_BASE:
        url = f"{PRO_BASE}{path}"
        r = _SESSION.get(url, params=_cg_params(params), headers=_cg_headers(), timeout=timeout)
    r.raise_for_status()
    return r

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
def fetch_daily_prices_series(coin_id: str, vs: str = "usd", days: str = "max", *, timeout: int | float = 45) -> pd.Series:
    """
    Fetch daily close from /market_chart (prices). Returns UTC-normalized daily series.
    Tries 'max' then degrades to 365/180 if plan forbids max.
    """
    try_days = ["max", "365", "180"] if days == "max" else [str(days), "365", "180"]
    last_exc: Exception | None = None
    for d in try_days:
        try:
            r = _cg_get(f"/coins/{coin_id}/market_chart", params={"vs_currency": vs, "days": d}, timeout=timeout)
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
                        chunk_days: int = 180,
                        *,
                        timeout: int | float = 60) -> pd.DataFrame:
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
            timeout=timeout,
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
# Public: ensure_price_cache (read-through incremental merge)
# ---------------------------------------------------------------------
def ensure_price_cache(symbol: str,
                       days: str | int = "max",
                       refresh_window_days: int = REFRESH_WINDOW_DAYS,
                       force: bool = False,
                       overlap_days: int = INCR_OVERLAP_DAYS,
                       always_incremental: bool = ALWAYS_INCREMENTAL,
                       *,
                       http_timeout: int | float = 8.0) -> Path:
    """
    Ensure a parquet cache exists and is up to date for `symbol`.

    Behavior:
      - ALWAYS performs a small incremental fetch (overlap window) unless disabled
      - Appends new vendor rows and de-dups on date
      - NEVER persists forward-filled data
      - `force=True` widens the refresh window regardless of cache state
      - `http_timeout` controls each vendor HTTP call (fixes kwarg TypeError)
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

    earliest = datetime(2017, 1, 1, tzinfo=timezone.utc)

    if old.empty:
        # bootstrap lookback
        if isinstance(days, int):
            start = max(now - timedelta(days=days), earliest)
        else:
            start = earliest
    else:
        last_date = pd.to_datetime(old["date"].max(), utc=True)
        # Always fetch with a small overlap to capture corrections/backfills
        overlap_start = (last_date - pd.Timedelta(days=int(max(0, overlap_days)))).normalize()
        # Do not fetch absurd windows; cap to recent window as an upper bound
        window_cap = now - timedelta(days=int(refresh_window_days))
        start = max(overlap_start, window_cap, earliest)

    # Optional widening if force=True
    if force:
        start = max(now - timedelta(days=int(refresh_window_days)), earliest)

    # If start is somehow beyond end (clock skew), clamp it
    if start > end:
        start = max(end - timedelta(days=int(max(1, overlap_days))), earliest)

    # Decide whether to hit network: default yes (always_incremental)
    should_fetch = always_incremental or force or old.empty
    new = pd.DataFrame(columns=["date", "price"])
    vs = _cfg("pricing.vs_currency", "usd")

    if should_fetch:
        try:
            new = fetch_range_from_cg(symbol, start, end, vs=vs, timeout=http_timeout)
        except Exception as e:
            log.warning("range fetch failed for %s (%s); falling back to /market_chart recent window", symbol, e)
            _, coin_id = resolve_coin_id(symbol)
            lookback = int(max(refresh_window_days, overlap_days))
            s = fetch_daily_prices_series(coin_id, vs=vs, days=str(lookback), timeout=http_timeout)
            df = s.reset_index()
            df.columns = ["date", "price"]
            df["date"] = pd.to_datetime(df["date"], utc=True)
            new = df[(df["date"] >= start) & (df["date"] <= end)].copy()

    both = pd.concat([old, new], ignore_index=True) if not old.empty else new
    if "date" not in both.columns:
        both["date"] = pd.to_datetime(both.index, utc=True)

    both = (both.dropna(subset=["price"])
               .drop_duplicates(["date"])
               .sort_values("date")
               .reset_index(drop=True))

    path.parent.mkdir(parents=True, exist_ok=True)
    both.to_parquet(path, index=False)
    return path


def fetch_last_prices(symbols: list[str], vs: str = "usd", *, timeout: int | float = 8) -> dict[str, float | None]:
    """
    Live last prices via /simple/price. Uses your Pro key & batches up to 200 ids.
    Returns: {SYMBOL: price or None}
    """
    # 1) resolve symbols -> coingecko ids
    pairs: list[tuple[str, str | None]] = []
    for s in symbols:
        sym = str(s).upper().strip()
        try:
            _, cid = resolve_coin_id(sym)
            pairs.append((sym, cid))
        except Exception:
            pairs.append((sym, None))

    ids = [cid for (_, cid) in pairs if cid]
    out: dict[str, float | None] = {sym: None for (sym, _) in pairs}
    if not ids:
        return out

    # 2) batch call /simple/price
    for i in range(0, len(ids), 200):
        chunk = ids[i:i+200]
        r = _cg_get("/simple/price",
                    params={"ids": ",".join(chunk), "vs_currencies": vs},
                    timeout=timeout)
        data = r.json()
        # 3) map back to symbols
        for sym, cid in pairs:
            if cid in chunk and cid in data:
                val = data[cid].get(vs)
                out[sym] = float(val) if val is not None else None
    return out

