from __future__ import annotations

import os, json
from typing import Dict
from fastapi import APIRouter, Query, HTTPException
import pandas as pd
import numpy as np
import requests

from src.risk.var import (
    parametric_decomposition,
    historical_es_decomposition,
    percent_of_total,
)

router = APIRouter(prefix="/risk", tags=["risk"])

API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")
HTTP_TIMEOUT = int(os.getenv("RISK_HTTP_TIMEOUT", "8"))

@router.get("/ping")
def ping():
    return {"ok": True}

# -------- helpers (weights & returns; no CSVs, no current_weights) --------
def _normalize_weights_dict(d: Dict[str, float]) -> pd.Series:
    s = pd.Series({str(k).upper().strip(): float(v) for k, v in d.items()})
    s = s[s > 0]
    tot = float(s.sum())
    if not np.isfinite(tot) or tot <= 0:
        raise ValueError("No positive weights.")
    return (s / tot).rename("weight")

def _weights_from_app(owner_id: str) -> pd.Series:
    """Ask your existing app for the portfolio; parse common shapes."""
    url = f"{API_BASE}/portfolios"
    r = requests.get(url, params={"owner_id": owner_id}, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    j = r.json()

    # Shape A: {"weights": {"BTC":0.5,...}}
    if isinstance(j, dict) and isinstance(j.get("weights"), dict):
        return _normalize_weights_dict(j["weights"])

    # Shape B: {"holdings":[{"symbol":"BTC","weight":0.5}, ...]} or quantities
    if isinstance(j, dict) and isinstance(j.get("holdings"), list):
        rows = j["holdings"]
        wrows = [row for row in rows if "weight" in row]
        if wrows:
            return _normalize_weights_dict({row["symbol"]: row["weight"] for row in wrows})
        qrows = [row for row in rows if "quantity" in row]
        if qrows:
            # value quantities by last price from cache
            from pathlib import Path
            from src.io.loaders import PriceStore
            store = PriceStore(data_path=Path("data"), auto_backfill=False)
            notionals = {}
            for row in qrows:
                sym = str(row["symbol"]).upper().strip()
                df = store.get(sym, fresh=False)
                if not df.empty:
                    px = float(df["price"].iloc[-1])
                    qty = float(row["quantity"])
                    if qty > 0 and np.isfinite(px):
                        notionals[sym] = qty * px
            if notionals:
                s = pd.Series(notionals, dtype=float)
                return (s / float(s.sum())).rename("weight")

    # Shape C: list of portfolios → try first element
    if isinstance(j, list) and j and isinstance(j[0], dict):
        first = j[0]
        if isinstance(first.get("weights"), dict):
            return _normalize_weights_dict(first["weights"])
        if isinstance(first.get("holdings"), list):
            rows = first["holdings"]
            wrows = [row for row in rows if "weight" in row]
            if wrows:
                return _normalize_weights_dict({row["symbol"]: row["weight"] for row in wrows})

    raise ValueError("Unrecognized /portfolios payload; please expose weights or holdings.")

def _load_returns_for_symbols(symbols: list[str], window_days: int, fresh: bool) -> pd.DataFrame:
    """Build wide returns (T×N) for given symbols from cached prices."""
    from pathlib import Path
    from src.io.loaders import PriceStore

    store = PriceStore(data_path=Path("data"), auto_backfill=True)
    now_utc = pd.Timestamp.now(tz="UTC")
    start = (now_utc - pd.Timedelta(days=window_days + 5)).isoformat()
    prices = store.get_many(symbols, start=start, end=None, fresh=fresh)  # tidy ['date','symbol','price']
    if prices.empty:
        return pd.DataFrame()

    wide = (
        prices.pivot(index="date", columns="symbol", values="price")
              .sort_index()
              .ffill()
    )
    rets = wide.pct_change().dropna(how="all")
    # keep only requested symbols in order
    rets = rets.reindex(columns=[s.upper().strip() for s in symbols]).dropna(how="all", axis=1)
    if len(rets) > window_days:
        rets = rets.iloc[-window_days:]
    return rets

# ------------------------------- route --------------------------------------
@router.get("/vares")
def vares(
    method: str = Query("parametric"),  # "parametric" | "historical_es"
    alpha: float = 0.99,
    window_days: int = 180,
    shrinkage: bool = True,
    lam: float = 0.1,
    abs_share: bool = False,
    fresh: bool = False,
    owner_id: str = Query("2", description="Use your app’s owner_id to fetch current portfolio"),
    weights_json: str | None = Query(None, description='Optional JSON dict: {"BTC":0.5,"ETH":0.3,...}'),
):
    m = method.lower().strip()
    if m not in {"parametric", "historical_es"}:
        raise HTTPException(status_code=422, detail="method must be 'parametric' or 'historical_es'")

    # 1) Resolve weights from existing app (or explicit weights_json override)
    try:
        if weights_json:
            w = _normalize_weights_dict(json.loads(weights_json))
        else:
            w = _weights_from_app(owner_id)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to resolve weights: {e}")

    # 2) Load returns for exactly these symbols
    rets = _load_returns_for_symbols(list(w.index), window_days, fresh)
    if rets.empty:
        raise HTTPException(status_code=422, detail="No returns available for requested symbols.")

    # 3) Compute VaR/ES + contributions
    if m == "parametric":
        out = parametric_decomposition(rets, w, alpha=alpha, shrinkage=shrinkage, lam=lam)
        pct = percent_of_total(out.comp_var, out.var_total, use_abs=abs_share)
        return {
            "method": "parametric",
            "alpha": alpha,
            "sigma_total": out.sigma_total,
            "var_total": out.var_total,
            "es_total": out.es_total,
            "components": {
                "sigma": out.comp_sigma.to_dict(),
                "var": out.comp_var.to_dict(),
                "es": out.comp_es.to_dict(),
                "percent_of_total_var": pct.to_dict(),
            },
        }
    else:
        out = historical_es_decomposition(rets, w, alpha=alpha)
        pct = percent_of_total(out.comp_es, out.es_total, use_abs=abs_share)
        return {
            "method": "historical_es",
            "alpha": alpha,
            "es_total": out.es_total,
            "tail_count": out.tail_count,
            "var_level_return": out.var_level,
            "components": {
                "es": out.comp_es.to_dict(),
                "percent_of_total_es": pct.to_dict(),
            },
        }
