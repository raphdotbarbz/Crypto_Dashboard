from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Optional
from pathlib import Path
import os
import time
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED

import pandas as pd

from .vendor_coingecko import ensure_price_cache, fetch_last_prices

from src.utils.logging import get_logger

log = get_logger(__name__)

# Where vendor parquet files live by convention
PRICE_DIR = Path("data/prices/coingecko")

# ---------------------------------------------------------------------
# Incremental refresh policy (env-tunable; sane defaults)
# ---------------------------------------------------------------------
CG_ALWAYS_INCREMENTAL = os.getenv("CG_ALWAYS_INCREMENTAL", "1") != "0"     # touch network by default
CG_INCR_OVERLAP_DAYS = int(os.getenv("CG_INCR_OVERLAP_DAYS", "2"))         # refetch this many days back
CG_REFRESH_WINDOW_DAYS = int(os.getenv("CG_REFRESH_WINDOW_DAYS", "120"))   # cap recent window
CG_HTTP_TIMEOUT = int(os.getenv("CG_HTTP_TIMEOUT", "8"))                   # per HTTP call timeout (s)
CG_REFRESH_BUDGET_SECONDS = float(os.getenv("CG_REFRESH_BUDGET_SECONDS", "5"))  # total time budget per bulk refresh
CG_REFRESH_MAX_WORKERS = int(os.getenv("CG_REFRESH_MAX_WORKERS", "4"))     # parallelism for multi-symbol refreshes


def load_coin_map(path: str | Path) -> pd.DataFrame:
    """Load symbol→id coin mapping (CSV with columns: symbol, coingecko_id, name)."""
    df = pd.read_csv(path)
    df["symbol"] = df["symbol"].str.upper()
    return df


def _df_to_series(df: pd.DataFrame) -> pd.Series:
    """
    Normalize any of our known parquet layouts into a daily Series named 'close',
    indexed by UTC-normalized date.

    Supported inputs:
      1) Vendor style: index = dates, column 'close'
      2) Store style: columns 'date' and 'price'
      3) Generic: single numeric column with a DatetimeIndex
    """
    # Case 1: vendor (index=datetime, column 'close')
    if "close" in df.columns and "date" not in df.columns:
        idx = pd.to_datetime(df.index, utc=True).normalize()
        s = pd.Series(df["close"].astype(float).values, index=idx, name="close")
        return s.sort_index()

    # Case 2: store (explicit 'date' + 'price')
    if "date" in df.columns and ("price" in df.columns or "close" in df.columns):
        col = "price" if "price" in df.columns else "close"
        dt = pd.to_datetime(df["date"], utc=True).dt.normalize()
        s = pd.Series(pd.to_numeric(df[col], errors="coerce").astype(float).values, index=dt, name="close")
        return s.sort_index()

    # Case 3: fallback — single column with datetime index
    idx = pd.to_datetime(df.index, utc=True).normalize()
    vals = pd.to_numeric(df.squeeze(), errors="coerce").astype(float)
    s = pd.Series(vals.values, index=idx, name="close")
    return s.sort_index()


@dataclass
class PriceStore:
    """
    Parquet-backed price store with *budgeted read-through incremental refresh*.

    On each read (get/get_many), we try a tiny overlap refresh from CoinGecko,
    but we cap total time spent so endpoints don't time out. If the budget
    is exceeded or vendor is slow, we serve cached data.
    """
    data_path: Path
    vendor: str = "coingecko"
    auto_backfill: bool = True  # if file missing, fetch from CoinGecko
    always_incremental: bool = CG_ALWAYS_INCREMENTAL
    overlap_days: int = CG_INCR_OVERLAP_DAYS
    refresh_window_days: int = CG_REFRESH_WINDOW_DAYS
    http_timeout: int = CG_HTTP_TIMEOUT
    bulk_refresh_budget_s: float = CG_REFRESH_BUDGET_SECONDS
    bulk_refresh_workers: int = CG_REFRESH_MAX_WORKERS

    def __post_init__(self):
        self.base = Path(self.data_path) / "prices" / self.vendor
        self.base.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.base / f"{key.upper().strip()}.parquet"

    def ensure(self, symbol: str, *, fresh: bool = True) -> Path:
        """
        Ensure a parquet exists for the symbol and (optionally) refresh it incrementally.

        - If missing and auto_backfill=True, fetch a bootstrap window.
        - If fresh=True, perform a small overlap fetch and merge before returning.
        """
        p = self._path(symbol)
        sym = symbol.upper().strip()

        # If file is missing, try to build it first
        if not p.exists() and self.auto_backfill:
            try:
                ensure_price_cache(
                    sym,
                    refresh_window_days=self.refresh_window_days,
                    force=False,
                    overlap_days=self.overlap_days,
                    always_incremental=True,     # force network at bootstrap
                    http_timeout=self.http_timeout,
                )
            except Exception as e:
                log.warning("Auto-backfill failed for %s: %s", sym, e)

        # Even if it exists, do a tiny incremental merge to keep fresh
        if fresh:
            try:
                ensure_price_cache(
                    sym,
                    refresh_window_days=self.refresh_window_days,
                    force=False,
                    overlap_days=self.overlap_days,
                    always_incremental=self.always_incremental,
                    http_timeout=self.http_timeout,
                )
            except Exception as e:
                # Non-fatal: serve last cached if vendor call fails
                log.warning("Incremental refresh failed for %s: %s", sym, e)

        return self._path(symbol)

    def put(self, symbol: str, df: pd.DataFrame) -> None:
        """
        Persist a DataFrame with either:
          - columns ['date','price'], or
          - index datetime and a column 'close'
        """
        out = df.copy()

        # normalize to ['date','price'] on disk for our own puts
        if "date" in out.columns and ("price" in out.columns or "close" in out.columns):
            if "close" in out.columns and "price" not in out.columns:
                out = out.rename(columns={"close": "price"})
            out["date"] = pd.to_datetime(out["date"], utc=True)
            out = out.sort_values("date")[["date", "price"]]
        else:
            # assume vendor style: index datetime + 'close'
            if "close" not in out.columns:
                raise ValueError("Expected a 'close' column or ['date','price'] columns")
            out = out.copy()
            out = out.rename(columns={"close": "price"})
            out = out.reset_index().rename(columns={"index": "date"})
            out["date"] = pd.to_datetime(out["date"], utc=True)
            out = out.sort_values("date")[["date", "price"]]

        out.to_parquet(self._path(symbol))

    def _refresh_many_budgeted(self, symbols: list[str], fresh: bool) -> None:
        """Refresh multiple symbols sequentially within a strict time budget."""
        if not fresh or not symbols:
            return
        start = time.time()
        budget = max(0.0, self.bulk_refresh_budget_s)

        for s in symbols:
            if (time.time() - start) >= budget:
                log.info("Budget hit: served cached for remaining %d symbols", len(symbols) - symbols.index(s))
                break
            try:
                ensure_price_cache(
                    s.upper().strip(),
                    refresh_window_days=self.refresh_window_days,
                    force=False,
                    overlap_days=self.overlap_days,
                    always_incremental=self.always_incremental,
                    http_timeout=self.http_timeout,
                )
            except Exception as e:
                log.warning("Refresh failed for %s: %s", s, e)

    def get(self, symbol: str, start: Optional[str] = None, end: Optional[str] = None, *, fresh: bool = True) -> pd.DataFrame:
        """
        Return a DataFrame with columns ['date','price'] for the requested symbol.

        By default, performs a budgeted read-through incremental refresh (`fresh=True`).
        Set `fresh=False` to read cached data only.
        """
        sym = symbol.upper().strip()
        p = self.ensure(sym, fresh=fresh)

        if not p.exists():
            log.warning("No cached prices for %s at %s", sym, p)
            return pd.DataFrame(columns=["date", "price"])

        raw = pd.read_parquet(p)
        s = _df_to_series(raw)
        df = s.to_frame(name="price").reset_index().rename(columns={"index": "date"})

        if start:
            df = df[df["date"] >= pd.to_datetime(start, utc=True)]
        if end:
            df = df[df["date"] <= pd.to_datetime(end, utc=True)]
        return df

    def get_many(self, symbols: Iterable[str], start: Optional[str] = None, end: Optional[str] = None, *, fresh: bool = True) -> pd.DataFrame:
        """
        Concatenate multiple symbols into one tidy frame: ['date','symbol','price'].

        If `fresh=True`, attempt a budgeted, concurrent incremental refresh of all symbols,
        then serve whatever is in cache (new or old).
        """
        symbols = [s.upper().strip() for s in symbols]
        self._refresh_many_budgeted(symbols, fresh=fresh)

        frames: list[pd.DataFrame] = []
        for s in symbols:
            p = self._path(s)
            if not p.exists() and self.auto_backfill:
                try:
                    self.ensure(s, fresh=False)
                except Exception as e:
                    log.warning("Auto-backfill failed for %s: %s", s, e)
            if p.exists():
                raw = pd.read_parquet(p)
                ser = _df_to_series(raw)
                if not ser.empty:
                    frames.append(ser.to_frame("price").reset_index().rename(columns={"index": "date"}).assign(symbol=s))

        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["date", "symbol", "price"])


def load_price_series(symbol: str, *, auto_backfill: bool = True, fresh: bool = True) -> pd.Series:
    """
    Load a **daily close** Series for a single symbol.

    If `fresh=True` (default), perform a tiny incremental vendor fetch first,
    but honor the HTTP timeout and budget limits set via env vars.
    """
    sym = symbol.upper().strip()
    path = PRICE_DIR / f"{sym}.parquet"

    try:
        ensure_price_cache(
            sym,
            refresh_window_days=CG_REFRESH_WINDOW_DAYS,
            force=False,
            overlap_days=CG_INCR_OVERLAP_DAYS,
            always_incremental=(fresh or CG_ALWAYS_INCREMENTAL),
            http_timeout=CG_HTTP_TIMEOUT,
        )
    except Exception as e:
        if not path.exists() and auto_backfill:
            log.warning("Auto-backfill failed for %s: %s", sym, e)
        elif fresh:
            log.warning("Fresh fetch failed for %s, serving last cached data: %s", sym, e)

    if not path.exists():
        if auto_backfill:
            try:
                ensure_price_cache(sym, refresh_window_days=CG_REFRESH_WINDOW_DAYS, force=True,
                                   overlap_days=CG_INCR_OVERLAP_DAYS, always_incremental=True,
                                   http_timeout=CG_HTTP_TIMEOUT)
            except Exception as e:
                log.warning("Auto-backfill (force) failed for %s: %s", sym, e)
        if not path.exists():
            raise FileNotFoundError(f"Missing price cache for {sym}: {path}")

    df = pd.read_parquet(path)
    s = _df_to_series(df)
    return s

def load_last_prices(symbols: Iterable[str], vs: str = "usd", http_timeout: int = CG_HTTP_TIMEOUT) -> pd.Series:
    syms = [str(s).upper().strip() for s in symbols]
    mp = fetch_last_prices(syms, vs=vs, timeout=http_timeout)
    return pd.Series(mp, index=syms, dtype="float64")

