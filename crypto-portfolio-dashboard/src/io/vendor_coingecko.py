# src/io/vendor_coingecko.py
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import os, time, logging, requests, pandas as pd

# ---------------------------------------------------------------------
# Config (server.py exports these from settings.yaml, or set in your env)
# ---------------------------------------------------------------------
PRO_BASE = "https://pro-api.coingecko.com/api/v3"
FREE_BASE = "https://api.coingecko.com/api/v3"

API_KEY = os.getenv("COINGECKO_API_KEY")
BASE_OVERRIDE = os.getenv("COINGECKO_BASE_URL")  # optional override
log = logging.getLogger(__name__)

def _cg_base() -> str:
    return BASE_OVERRIDE or (PRO_BASE if API_KEY else FREE_BASE)

def _cg_headers() -> dict:
    h = {"accept": "application/json"}
    if API_KEY:
        # CoinGecko Pro header
        h["x-cg-pro-api-key"] = API_KEY
    return h

def _cg_params(extra: dict | None = None) -> dict:
    p = dict(extra or {})
    if API_KEY:
        # Some deployments accept/require the key as a query param too
        p["x_cg_pro_api_key"] = API_KEY
    return p

def _cg_get(path: str, *, params: dict | None = None, timeout: int = 30) -> requests.Response:
    """
    Robust GET to CoinGecko only.
    Retries for 401/429, and forces the Pro base when a key is present.
    """
    base_url = _cg_base()
    url = f"{base_url}{path}"
    r = None
    for attempt in range(3):
        r = requests.get(url, params=_cg_params(params), headers=_cg_headers(), timeout=timeout)
        # Rate-limit / unauthorized handling
        if r.status_code in (429, 401):
            # If unauthorized with a key and not on pro base, force pro on next try
            if r.status_code == 401 and API_KEY and base_url != PRO_BASE:
                base_url = PRO_BASE
                url = f"{base_url}{path}"
            time.sleep(1.5 * (attempt + 1))
            continue
        r.raise_for_status()
        return r
    # If we got here, last response is still error
    assert r is not None
    r.raise_for_status()
    return r  # pragma: no cover

# ---------------------------------------------------------------------
# Local cache paths
# ---------------------------------------------------------------------
COIN_MAP_PATH = Path("data/coin_map.csv")
PRICE_DIR = Path("data/prices/coingecko")
PRICE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# Coin mapping helpers
# ---------------------------------------------------------------------
def _read_coin_map() -> pd.DataFrame:
    if COIN_MAP_PATH.exists():
        df = pd.read_csv(COIN_MAP_PATH)
        if "symbol" in df and "coingecko_id" in df:
            df["symbol"] = df["symbol"].str.upper()
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
        exact = (c.get("symbol", "").lower() == q)
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
# Price fetchers (CoinGecko ONLY)
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
            r = _cg_get(f"/coins/{coin_id}/market_chart", params={"vs_currency": vs, "days": d}, timeout=30)
            prices = r.json().get("prices", [])
            if not prices:
                continue
            df = pd.DataFrame(prices, columns=["ms", "price"])
            idx = pd.to_datetime(df["ms"], unit="ms", utc=True).dt.normalize()  # <- critical: .dt.normalize
            s = pd.Series(pd.to_numeric(df["price"], errors="coerce").astype(float).values,
                          index=idx, name="close")
            s = s.groupby(level=0).last().sort_index()
            if not s.empty:
                return s
        except Exception as e:
            last_exc = e
            continue
    raise RuntimeError(f"CoinGecko returned no prices for {coin_id}; last_error={last_exc}")

def ensure_price_cache(symbol: str, *, vs: str = "usd", days: str = "max") -> Path:
    """Guarantee data/prices/coingecko/SYMBOL.parquet exists; fetch & write if missing (CoinGecko-only)."""
    sym, cg_id = resolve_coin_id(symbol)
    out = PRICE_DIR / f"{sym}.parquet"
    if out.exists():
        return out
    s = fetch_daily_prices_series(cg_id, vs=vs, days=days)
    out.parent.mkdir(parents=True, exist_ok=True)
    s.to_frame().to_parquet(out)
    return out
