# src/api/server.py
from __future__ import annotations

import math
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, List

import numpy as np
import pandas as pd
import yaml
from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.responses import RedirectResponse

# Schemas
from src.api.schemas import (
    BackfillRequest,
    HoldingWeight,
    PortfolioMetrics,
    PortfolioOverview,
    RiskTiles,
    TransactionIn,
)

# Portfolio/valuation/risk
from src.portfolio.positions import daily_positions, transactions_to_positions
from src.portfolio.returns import horizon_return
from src.portfolio.valuation import merge_with_prices_daily, total_value_series, weights_on_date
from src.risk.drawdown import max_drawdown
from src.risk.ratios import calmar, sharpe, sortino, trailing_slice

# DB (SQLModel)
from sqlmodel import delete, select
from src.api.db import get_session, init_db
from src.api.models import Portfolio, Tx, User

# Prices / vendors
from src.io.loaders import load_last_prices, load_price_series
from src.io.vendor_coingecko import ensure_price_cache  # used by backfill endpoints

# Optional routers
from .prices_router import router as prices_router  # <- make sure this file exists
from .risk_router import router as risk_router


# -----------------------------------------------------------------------------
# App & settings
# -----------------------------------------------------------------------------

app = FastAPI(title="Crypto Portfolio API", version="0.1.0")
router = APIRouter()


@app.on_event("startup")
def _startup():
    init_db()


def load_settings() -> dict:
    path = Path("config/settings.yaml")
    if not path.exists():
        return {"data_path": "data", "timezone": "UTC", "vendors": {"prices": "coingecko"}}
    return yaml.safe_load(path.read_text()) or {}


def _expand_env(obj):
    """Recursively expand ${VAR} in settings."""
    if isinstance(obj, dict):
        return {k: _expand_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env(x) for x in obj]
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    return obj


SETTINGS = _expand_env(load_settings())
DATA_PATH = Path(SETTINGS.get("data_path", "data"))
VENDOR = (SETTINGS.get("vendors", {}) or {}).get("prices", "coingecko")

# Read CG key/base from either vendor_options.coingecko or vendors.coingecko
_co_opts = (SETTINGS.get("vendor_options", {}) or {}).get("coingecko", {}) or {}
_co_legacy = (SETTINGS.get("vendors", {}) or {}).get("coingecko", {}) or {}
cfg_key = _co_opts.get("api_key") or _co_legacy.get("api_key")
base_url = _co_opts.get("base_url") or _co_legacy.get("base_url")

# Export to env so vendor module uses them
if cfg_key and not os.getenv("COINGECKO_API_KEY"):
    os.environ["COINGECKO_API_KEY"] = str(cfg_key).strip()
if base_url and not os.getenv("COINGECKO_BASE_URL"):
    os.environ["COINGECKO_BASE_URL"] = str(base_url).strip()


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def json_safe(obj: Any) -> Any:
    """Recursively convert NaN/Inf -> None and Pandas/Numpy types -> builtins."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        return val if math.isfinite(val) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, Iterable)) and not isinstance(obj, (str, bytes)):
        return [json_safe(x) for x in obj]
    if isinstance(obj, pd.Series):
        return json_safe(obj.to_dict())
    if isinstance(obj, pd.DataFrame):
        df = obj.replace([np.inf, -np.inf], np.nan)
        df = df.where(pd.notna(df), None)
        return json_safe(df.to_dict(orient="records"))
    return obj


def _tx_df_from_db(session, portfolio_id: int) -> pd.DataFrame:
    rows = session.exec(select(Tx).where(Tx.portfolio_id == portfolio_id).order_by(Tx.date)).all()
    if not rows:
        return pd.DataFrame(columns=["date", "symbol", "quantity", "price", "fees", "notes"])
    df = pd.DataFrame(
        [
            {
                "date": r.date,
                "symbol": r.symbol.upper().strip(),
                "quantity": r.quantity,
                "price": r.price,
                "fees": r.fees,
                "notes": r.notes,
            }
            for r in rows
        ]
    )
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df


def _build_prices_tidy(symbols: list[str]) -> pd.DataFrame:
    """
    Build tidy prices frame: ['date','symbol','price'] by auto-loading (and backfilling) each symbol.
    Robust to different index/column names after reset_index.
    """
    frames: list[pd.DataFrame] = []
    for s in symbols:
        sym = s.upper().strip()
        try:
            ser = load_price_series(sym, auto_backfill=True)
        except Exception:
            continue
        if ser is None or len(ser) == 0:
            continue

        # Force Series and reset index
        ser = pd.Series(ser).astype(float)
        df = ser.to_frame(name="price").reset_index()

        # Determine date & price columns
        if df.shape[1] < 2:
            continue
        date_col = df.columns[0]
        if "price" not in df.columns:
            price_col = df.columns[1]
            df = df.rename(columns={price_col: "price"})

        df = df.rename(columns={date_col: "date"})
        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
        df = df.dropna(subset=["date"])
        df["symbol"] = sym

        frames.append(df[["date", "symbol", "price"]])

    if not frames:
        return pd.DataFrame(columns=["date", "symbol", "price"])
    out = pd.concat(frames, ignore_index=True)
    # De-dup in case multiple loaders write the same day
    out = out.sort_values(["symbol", "date"]).drop_duplicates(["symbol", "date"], keep="last")
    return out


def _safe_float(x):
    """Return float(x) or None if x is NaN/inf/None."""
    if x is None:
        return None
    try:
        xf = float(x)
        if np.isnan(xf) or np.isinf(xf):
            return None
        return xf
    except Exception:
        return None


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------

@app.get("/")
def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "time": datetime.now(timezone.utc).isoformat(),
        "data_path": str(DATA_PATH),
        "vendor": VENDOR,
        "coingecko_key_detected": bool(os.getenv("COINGECKO_API_KEY")),
        "coingecko_base_url": os.getenv("COINGECKO_BASE_URL") or "",
    }


@app.get("/config")
def config():
    return SETTINGS


# --- users & portfolios -------------------------------------------------------

@app.post("/users")
def create_user(email: str):
    with get_session() as s:
        got = s.exec(select(User).where(User.email == email)).first()
        if got:
            return {"ok": True, "id": got.id}
        u = User(email=email)
        s.add(u)
        s.commit()
        s.refresh(u)
        return {"ok": True, "id": u.id}


@app.post("/portfolios")
def create_portfolio(owner_id: int, name: str):
    with get_session() as s:
        owner = s.get(User, owner_id)
        if not owner:
            raise HTTPException(400, "owner_id not found")
        p = Portfolio(name=name, owner_id=owner_id)
        s.add(p)
        s.commit()
        s.refresh(p)
        return {"ok": True, "id": p.id}


@app.get("/portfolios")
def list_portfolios(owner_id: int):
    with get_session() as s:
        rows = s.exec(select(Portfolio).where(Portfolio.owner_id == owner_id)).all()
        return [{"id": p.id, "name": p.name, "created_at": p.created_at.isoformat()} for p in rows]


# --- transactions -------------------------------------------------------------

@app.post("/transactions/db/append")
def transactions_append_db(portfolio_id: int, items: List[TransactionIn]):
    with get_session() as s:
        if not s.get(Portfolio, portfolio_id):
            raise HTTPException(400, "portfolio_id not found")
        for i in items:
            s.add(
                Tx(
                    portfolio_id=portfolio_id,
                    date=pd.to_datetime(i.date, utc=True).to_pydatetime(),
                    symbol=i.symbol.upper().strip(),
                    quantity=float(i.quantity),
                    price=(float(i.price) if i.price is not None else None),
                    fees=(float(i.fees) if i.fees is not None else 0.0),
                    notes=i.notes or "",
                )
            )
        s.commit()
    return {"ok": True}


@app.get("/transactions/db")
def list_transactions_db(portfolio_id: int, limit: int = 200):
    with get_session() as s:
        rows = s.exec(select(Tx).where(Tx.portfolio_id == portfolio_id).order_by(Tx.date.desc())).all()
        return [
            {
                "id": r.id,
                "date": pd.to_datetime(r.date).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "symbol": r.symbol,
                "quantity": r.quantity,
                "price": (None if r.price is None else float(r.price)),
                "fees": float(r.fees),
                "notes": r.notes or "",
            }
            for r in rows[:limit]
        ]


# --- portfolio analytics ------------------------------------------------------

@app.get("/portfolio/overview", response_model=PortfolioOverview)
def portfolio_overview(portfolio_id: int | None = None):
    if portfolio_id is None:
        raise HTTPException(status_code=400, detail="portfolio_id is required")
    with get_session() as s:
        tx = _tx_df_from_db(s, portfolio_id)
        if tx.empty:
            raise HTTPException(status_code=400, detail="No transactions in this portfolio")

    # Positions → daily positions
    pos = transactions_to_positions(tx)
    dpos = daily_positions(pos)

    # Prices (auto-backfill missing)
    syms = sorted(dpos["symbol"].unique())
    prices = _build_prices_tidy(syms)
    if prices.empty:
        raise HTTPException(status_code=400, detail="No prices available for symbols in this portfolio.")

    # Valuation
    mv = merge_with_prices_daily(dpos, prices)  # [date,symbol,quantity,price,value]
    totals = total_value_series(mv)             # [date,total_value,ret_1d,ret_7d,...]
    valid = totals.dropna(subset=["total_value"])
    if valid.empty:
        raise HTTPException(status_code=400, detail="No valued days produced. Check symbol mapping or ingestion window.")

    last = valid.iloc[-1]

    # Build clean top holdings (avoid NaNs/inf for JSON)
    w = weights_on_date(mv, last["date"])
    if "weight" in w.columns:
        w["weight"] = pd.to_numeric(w["weight"], errors="coerce")
    if "value" in w.columns:
        w["value"] = pd.to_numeric(w["value"], errors="coerce")
    w = (
        w.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["weight"])
        .loc[lambda x: x["weight"] >= 0]
        .sort_values("weight", ascending=False)
        .head(10)
    )
    holdings = []
    for r in w.to_dict("records"):
        rec = {
            "symbol": r.get("symbol"),
            "weight": (None if pd.isna(r.get("weight")) else float(r.get("weight"))),
        }
        if "value" in r:
            rec["value"] = (None if pd.isna(r.get("value")) else float(r.get("value")))
        holdings.append(HoldingWeight(**rec))

    # Last prices for the holdings
    symbols_for_last = sorted({h.symbol for h in holdings if getattr(h, "symbol", None)})
    last_prices_ser = load_last_prices(symbols_for_last, vs="usd", http_timeout=6)
    last_prices = {s: (None if pd.isna(v) else float(v)) for s, v in last_prices_ser.to_dict().items()}

    # Horizon returns (NaN-safe)
    r30 = horizon_return(valid, 30) if len(valid) > 30 else None
    r180 = horizon_return(valid, 180) if len(valid) > 180 else None
    r365 = horizon_return(valid, 365) if len(valid) > 365 else None

    return PortfolioOverview(
        as_of=last["date"].to_pydatetime(),
        total_value=float(last["total_value"]),
        ret_1d=(None if pd.isna(last.get("ret_1d", np.nan)) else float(last["ret_1d"])) if "ret_1d" in valid else None,
        ret_7d=(None if pd.isna(last.get("ret_7d", np.nan)) else float(last["ret_7d"])) if "ret_7d" in valid else None,
        ret_30d=_safe_float(r30),
        ret_180d=_safe_float(r180),
        ret_365d=_safe_float(r365),
        top_holdings=holdings,
        last_prices=last_prices,
    )


@app.get("/portfolio/metrics", response_model=PortfolioMetrics)
def portfolio_metrics(lookbacks: str = "90,180,365", rf_annual: float = 0.0, portfolio_id: int | None = None):
    if portfolio_id is None:
        raise HTTPException(status_code=400, detail="portfolio_id is required")
    with get_session() as s:
        tx = _tx_df_from_db(s, portfolio_id)
        if tx.empty:
            raise HTTPException(status_code=400, detail="No transactions in this portfolio")

    # Positions → daily positions
    pos = transactions_to_positions(tx)
    dpos = daily_positions(pos)

    # Prices (auto-backfill missing)
    syms = sorted(dpos["symbol"].unique())
    prices = _build_prices_tidy(syms)
    if prices.empty:
        raise HTTPException(status_code=400, detail="No prices available for symbols in this portfolio.")

    # Valuation
    mv = merge_with_prices_daily(dpos, prices)
    totals = total_value_series(mv).dropna(subset=["total_value"])
    if totals.empty:
        raise HTTPException(status_code=400, detail="No valued days produced.")

    as_of = totals["date"].iloc[-1].to_pydatetime()

    # Build tiles (NaN-safe)
    out_tiles = []
    for tok in [t.strip() for t in lookbacks.split(",") if t.strip()]:
        try:
            lb = int(tok)
        except ValueError:
            out_tiles.append(RiskTiles(lookback_days=None))
            continue

        t = trailing_slice(totals, lb)
        if t.empty:
            out_tiles.append(RiskTiles(lookback_days=lb))
            continue

        md = _safe_float(max_drawdown(t))
        sh = _safe_float(sharpe(t, rf_annual)) if len(t) > 2 else None
        so = _safe_float(sortino(t, rf_annual)) if len(t) > 2 else None
        ca = _safe_float(calmar(t)) if len(t) > 2 else None

        out_tiles.append(RiskTiles(lookback_days=lb, max_drawdown=md, sharpe=sh, sortino=so, calmar=ca))

    payload = {"as_of": as_of, "tiles": out_tiles}
    return PortfolioMetrics(**payload)


@app.get("/portfolio/totals")
def portfolio_totals(portfolio_id: int | None = None):
    if portfolio_id is None:
        raise HTTPException(status_code=400, detail="portfolio_id is required")
    with get_session() as s:
        tx = _tx_df_from_db(s, portfolio_id)
        if tx.empty:
            raise HTTPException(status_code=400, detail="No transactions in this portfolio")

    # Positions → daily positions
    pos = transactions_to_positions(tx)
    dpos = daily_positions(pos)

    # Prices (auto-backfill missing)
    syms = sorted(dpos["symbol"].unique())
    prices = _build_prices_tidy(syms)
    if prices.empty:
        raise HTTPException(400, "No prices available for symbols in this portfolio.")

    # Valuation
    mv = merge_with_prices_daily(dpos, prices)
    totals = total_value_series(mv).dropna(subset=["total_value"]).sort_values("date")

    # JSON-safe rows
    out = []
    for _, r in totals.iterrows():
        out.append(
            {
                "date": pd.to_datetime(r["date"]).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "total_value": float(r["total_value"]),
                "ret_1d": (None if pd.isna(r.get("ret_1d", np.nan)) else float(r["ret_1d"])) if "ret_1d" in totals else None,
                "ret_7d": (None if pd.isna(r.get("ret_7d", np.nan)) else float(r["ret_7d"])) if "ret_7d" in totals else None,
            }
        )
    return out


# --- price backfill & freshness ----------------------------------------------

@router.post("/prices/backfill")
def prices_backfill(req: BackfillRequest):
    if not req.symbols:
        raise HTTPException(status_code=400, detail="No symbols provided.")
    results = []
    for sym in req.symbols:
        try:
            path = ensure_price_cache(sym, days=req.days or "max")
            results.append({"symbol": str(sym).upper(), "path": str(path)})
        except Exception as e:
            results.append({"symbol": str(sym).upper(), "error": str(e)})
    return {"results": results}


@router.post("/prices/backfill/portfolio")
def prices_backfill_portfolio(portfolio_id: int, days: str = "max"):
    with get_session() as s:
        rows = s.exec(select(Tx).where(Tx.portfolio_id == portfolio_id)).all()
        syms = sorted({r.symbol.upper().strip() for r in rows})
    if not syms:
        raise HTTPException(status_code=400, detail="No symbols found for portfolio.")
    results = []
    for sym in syms:
        try:
            path = ensure_price_cache(sym, days=days)
            results.append({"symbol": sym, "path": str(path)})
        except Exception as e:
            results.append({"symbol": sym, "error": str(e)})
    return {"results": results}


@app.get("/data/staleness")
def data_staleness(portfolio_id: int | None = None):
    if portfolio_id is None:
        raise HTTPException(status_code=400, detail="portfolio_id is required")
    with get_session() as s:
        tx = _tx_df_from_db(s, portfolio_id)
        if tx.empty:
            raise HTTPException(status_code=400, detail="No transactions in this portfolio")

    syms = sorted(tx["symbol"].unique().tolist()) if not tx.empty else []

    prices = _build_prices_tidy(syms)
    if prices.empty:
        return {"overall_last": None, "overall_stale_days": None, "symbols": []}

    prices["date"] = pd.to_datetime(prices["date"], utc=True)
    per = prices.groupby("symbol")["date"].max().sort_values(ascending=False)
    today = datetime.now(timezone.utc).date()
    rows = []
    for sym, dt in per.items():
        d = dt.date()
        rows.append({"symbol": sym, "last": str(d), "stale_days": (today - d).days})
    overall_last = max(per).date()
    return {
        "overall_last": str(overall_last),
        "overall_stale_days": (today - overall_last).days,
        "symbols": rows,
    }


@router.post("/prices/refresh_if_stale")
def prices_refresh_if_stale(portfolio_id: int, max_stale_days: int = 1, days: str = "max"):
    info = data_staleness(portfolio_id)
    if info["overall_stale_days"] is None or info["overall_stale_days"] <= max_stale_days:
        return {"refreshed": False, "reason": "fresh"}
    # pull symbols and backfill
    with get_session() as s:
        rows = s.exec(select(Tx).where(Tx.portfolio_id == portfolio_id)).all()
        syms = sorted({r.symbol.upper().strip() for r in rows})
    results = []
    for sym in syms:
        try:
            path = ensure_price_cache(sym, days=days)
            results.append({"symbol": sym, "path": str(path)})
        except Exception as e:
            results.append({"symbol": sym, "error": str(e)})
    return {"refreshed": True, "results": results}


@app.delete("/portfolios/{portfolio_id}")
def delete_portfolio(portfolio_id: int, owner_id: int | None = None, confirm: bool = False):
    """
    Permanently delete a portfolio and ALL its transactions.
    Safety: require confirm=true; if owner_id is supplied, verify ownership.
    """
    if not confirm:
        raise HTTPException(status_code=400, detail="Set confirm=true to delete this portfolio.")
    with get_session() as s:
        p = s.get(Portfolio, portfolio_id)
        if not p:
            raise HTTPException(status_code=404, detail="portfolio_id not found")
        if owner_id is not None and p.owner_id != owner_id:
            raise HTTPException(status_code=403, detail="owner_id does not own this portfolio")
        # delete transactions then portfolio
        s.exec(delete(Tx).where(Tx.portfolio_id == portfolio_id))
        s.delete(p)
        s.commit()
    return {"ok": True, "deleted": portfolio_id}


# -----------------------------------------------------------------------------
# Register routers
# -----------------------------------------------------------------------------

# local APIRouter (backfill + refresh) endpoints
app.include_router(router)

# <-- This line unblocks your 404s: mount the prices router so /prices/series & /prices/matrix exist
app.include_router(prices_router)

# risk endpoints (if you’ve defined any there)
app.include_router(risk_router)


