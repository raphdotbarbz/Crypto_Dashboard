# src/api/server.py
from __future__ import annotations

from datetime import datetime, timezone
from typing import List
from pathlib import Path
import os

import yaml
import pandas as pd
import numpy as np

from fastapi import FastAPI, HTTPException, APIRouter
from fastapi.responses import RedirectResponse

from src.api.schemas import (
    PortfolioOverview,
    HoldingWeight,
    PortfolioMetrics,
    RiskTiles,
    TransactionIn,
    BackfillRequest,
)
from src.portfolio.positions import transactions_to_positions, daily_positions
from src.portfolio.valuation import (
    merge_with_prices_daily,
    total_value_series,
    weights_on_date,
)
from src.portfolio.returns import horizon_return
from src.risk.drawdown import max_drawdown
from src.risk.ratios import sharpe, sortino, calmar, trailing_slice

# DB
from sqlmodel import select
from src.api.db import init_db, get_session
from src.api.models import User, Portfolio, Tx

# NEW: auto-backfilling price loader (replaces direct parquet reads / PriceStore)
from src.io.loaders import load_price_series
from src.io.vendor_coingecko import ensure_price_cache  # used by backfill endpoints

import math
from typing import Any, Iterable
import numpy as np
import pandas as pd


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
    df = pd.DataFrame([{
        "date": r.date,
        "symbol": r.symbol.upper().strip(),
        "quantity": r.quantity,
        "price": r.price,
        "fees": r.fees,
        "notes": r.notes
    } for r in rows])
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df


def _parse_dates_iso(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
    """Robustly parse date strings into UTC datetimes (drops blanks, coerces mixed ISO)."""
    raw = df[col].astype(str).str.strip()
    droppable = raw.str.lower().isin(["", "nan", "nat", "none", "null"])
    if droppable.any():
        df = df[~droppable].copy()
        raw = raw[~droppable]
    parsed = pd.to_datetime(raw, format="mixed", utc=True, errors="coerce")
    bad = parsed.isna()
    if bad.any():
        parsed.loc[bad] = pd.to_datetime(raw[bad], utc=True, errors="coerce")
    still_bad = parsed.isna()
    if still_bad.any():
        examples = raw[still_bad].head(3).tolist()
        raise HTTPException(status_code=400, detail={"msg": "Unparseable dates in transactions.csv", "examples": examples})
    df[col] = parsed
    return df


def _load_transactions_csv() -> pd.DataFrame:
    tx_path = Path("data/transactions.csv")
    if not tx_path.exists():
        raise HTTPException(status_code=400, detail="Missing data/transactions.csv")
    tx = pd.read_csv(tx_path)
    if tx.empty:
        raise HTTPException(status_code=400, detail="transactions.csv is empty")
    tx = _parse_dates_iso(tx, "date")
    tx["symbol"] = tx["symbol"].astype(str).str.upper().str.strip()
    tx["quantity"] = pd.to_numeric(tx["quantity"], errors="coerce")
    if tx["quantity"].isna().any():
        raise HTTPException(status_code=400, detail="Non-numeric quantities detected in transactions.csv")
    if "price" in tx:
        tx["price"] = pd.to_numeric(tx["price"], errors="coerce")
    if "fees" in tx:
        tx["fees"] = pd.to_numeric(tx["fees"], errors="coerce").fillna(0.0)
    return tx


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

        # Determine which column is date (first col after reset_index)
        # and which is price (the non-date column).
        if df.shape[1] < 2:
            continue  # nothing to do

        date_col = df.columns[0]
        # If 'price' isn't there (some legacy parquet), take the second column as price
        if "price" in df.columns:
            price_col = "price"
        else:
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


# --- create user ---
@app.post("/users")
def create_user(email: str):
    with get_session() as s:
        got = s.exec(select(User).where(User.email == email)).first()
        if got:
            return {"ok": True, "id": got.id}
        u = User(email=email)
        s.add(u); s.commit(); s.refresh(u)
        return {"ok": True, "id": u.id}


# --- create portfolio ---
@app.post("/portfolios")
def create_portfolio(owner_id: int, name: str):
    with get_session() as s:
        owner = s.get(User, owner_id)
        if not owner:
            raise HTTPException(400, "owner_id not found")
        p = Portfolio(name=name, owner_id=owner_id)
        s.add(p); s.commit(); s.refresh(p)
        return {"ok": True, "id": p.id}


# --- list portfolios for a user ---
@app.get("/portfolios")
def list_portfolios(owner_id: int):
    with get_session() as s:
        rows = s.exec(select(Portfolio).where(Portfolio.owner_id == owner_id)).all()
        return [{"id": p.id, "name": p.name, "created_at": p.created_at.isoformat()} for p in rows]


# --- append tx to DB ---
@app.post("/transactions/db/append")
def transactions_append_db(portfolio_id: int, items: List[TransactionIn]):
    with get_session() as s:
        if not s.get(Portfolio, portfolio_id):
            raise HTTPException(400, "portfolio_id not found")
        for i in items:
            s.add(Tx(
                portfolio_id=portfolio_id,
                date=pd.to_datetime(i.date, utc=True).to_pydatetime(),
                symbol=i.symbol.upper().strip(),
                quantity=float(i.quantity),
                price=(float(i.price) if i.price is not None else None),
                fees=(float(i.fees) if i.fees is not None else 0.0),
                notes=i.notes or ""
            ))
        s.commit()
    return {"ok": True}


@app.get("/portfolio/overview", response_model=PortfolioOverview)
def portfolio_overview(portfolio_id: int | None = None):
    # Load transactions (CSV or DB)
    if portfolio_id is not None:
        with get_session() as s:
            tx = _tx_df_from_db(s, portfolio_id)
            if tx.empty:
                raise HTTPException(400, "No transactions in this portfolio")
    else:
        tx = _load_transactions_csv()

    # Positions → daily positions
    pos = transactions_to_positions(tx)
    dpos = daily_positions(pos)

    # Prices (auto-backfill missing)
    syms = sorted(dpos["symbol"].unique())
    prices = _build_prices_tidy(syms)
    if prices.empty:
        raise HTTPException(status_code=400, detail="No prices available for symbols in this portfolio.")

    # Valuation
    mv = merge_with_prices_daily(dpos, prices)   # [date,symbol,quantity,price,value]
    totals = total_value_series(mv)              # [date,total_value,ret_1d,ret_7d,...]
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
    w = (w.replace([np.inf, -np.inf], np.nan)
           .dropna(subset=["weight"])
           .loc[lambda x: x["weight"] >= 0]
           .sort_values("weight", ascending=False)
           .head(10))
    holdings = []
    for r in w.to_dict("records"):
        rec = {
            "symbol": r.get("symbol"),
            "weight": (None if pd.isna(r.get("weight")) else float(r.get("weight"))),
        }
        if "value" in r:
            rec["value"] = (None if pd.isna(r.get("value")) else float(r.get("value")))
        holdings.append(HoldingWeight(**rec))

    # Guard horizon returns against NaN
    r30  = horizon_return(valid, 30)  if len(valid) > 30  else None
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
    )


@app.get("/portfolio/metrics", response_model=PortfolioMetrics)
def portfolio_metrics(lookbacks: str = "90,180,365", rf_annual: float = 0.0, portfolio_id: int | None = None):
    # Load transactions (CSV or DB)
    if portfolio_id is not None:
        with get_session() as s:
            tx = _tx_df_from_db(s, portfolio_id)
            if tx.empty:
                raise HTTPException(status_code=400, detail="No transactions in this portfolio")
    else:
        tx = _load_transactions_csv()

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

    # Build tiles
    out_tiles = []
    for tok in [t.strip() for t in lookbacks.split(",") if t.strip()]:
        try:
            lb = int(tok)
        except ValueError:
            continue
        t = trailing_slice(totals, lb)
        if t.empty:
            out_tiles.append(RiskTiles(lookback_days=lb))
            continue
        out_tiles.append(RiskTiles(
            lookback_days=lb,
            max_drawdown=float(max_drawdown(t)),
            sharpe=float(sharpe(t, rf_annual)) if len(t) > 2 else None,
            sortino=float(sortino(t, rf_annual)) if len(t) > 2 else None,
            calmar=float(calmar(t)) if len(t) > 2 else None,
        ))

    return {
        "as_of": as_of,
        "tiles": [t.model_dump() if hasattr(t, "model_dump") else dict(t) for t in out_tiles],
    }


@app.get("/portfolio/totals")
def portfolio_totals(portfolio_id: int | None = None):
    # Load transactions (CSV or DB)
    if portfolio_id is not None:
        with get_session() as s:
            tx = _tx_df_from_db(s, portfolio_id)
            if tx.empty:
                raise HTTPException(400, "No transactions in this portfolio")
    else:
        tx = _load_transactions_csv()

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
        out.append({
            "date": pd.to_datetime(r["date"]).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "total_value": float(r["total_value"]),
            "ret_1d": (None if pd.isna(r.get("ret_1d", np.nan)) else float(r["ret_1d"])) if "ret_1d" in totals else None,
            "ret_7d": (None if pd.isna(r.get("ret_7d", np.nan)) else float(r["ret_7d"])) if "ret_7d" in totals else None,
        })
    return out


# --- CSV append ---
@app.post("/transactions/append")
def transactions_append(items: List[TransactionIn]):
    if not items:
        raise HTTPException(status_code=400, detail="No items provided.")

    tx_path = Path("data/transactions.csv")

    # Read existing or create fresh with correct dtypes
    if tx_path.exists():
        existing = pd.read_csv(tx_path)
    else:
        existing = pd.DataFrame(columns=["date", "symbol", "quantity", "price", "fees", "notes"])

    # Normalize existing
    if not existing.empty:
        existing = _parse_dates_iso(existing, "date")
        existing["symbol"] = existing["symbol"].astype(str).str.upper().str.strip()
        existing["quantity"] = pd.to_numeric(existing["quantity"], errors="coerce")
        existing["price"] = pd.to_numeric(existing.get("price", np.nan), errors="coerce")
        existing["fees"] = pd.to_numeric(existing.get("fees", 0.0), errors="coerce").fillna(0.0)
        if "notes" not in existing:
            existing["notes"] = ""
    else:
        existing = pd.DataFrame({
            "date": pd.Series([], dtype="datetime64[ns, UTC]"),
            "symbol": pd.Series([], dtype="string"),
            "quantity": pd.Series([], dtype="float"),
            "price": pd.Series([], dtype="float"),
            "fees": pd.Series([], dtype="float"),
            "notes": pd.Series([], dtype="string"),
        })

    # Normalize incoming
    new = pd.DataFrame([{
        "date": pd.to_datetime(i.date, utc=True),
        "symbol": i.symbol.upper().strip(),
        "quantity": float(i.quantity),
        "price": float(i.price) if i.price is not None else np.nan,
        "fees": float(i.fees) if i.fees is not None else 0.0,
        "notes": i.notes or "",
    } for i in items])

    # De-dup key: ISO timestamp + normalized symbol/qty/price
    def _key(df: pd.DataFrame) -> pd.Series:
        iso = pd.to_datetime(df["date"], utc=True, errors="coerce").dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        sym = df["symbol"].astype(str).str.upper().str.strip()
        qty = pd.to_numeric(df["quantity"], errors="coerce").round(10).astype(str)
        prc = pd.to_numeric(df["price"], errors="coerce").fillna(-0.0).round(10).astype(str)
        return iso + "|" + sym + "|" + qty + "|" + prc

    existing["_k"] = _key(existing) if not existing.empty else pd.Series([], dtype="string")
    new["_k"] = _key(new)

    out = (pd.concat([existing, new], ignore_index=True)
           .drop_duplicates("_k")
           .drop(columns="_k")
           .sort_values("date"))

    out.to_csv(tx_path, index=False)
    return {"ok": True, "rows": int(len(out))}


@app.get("/transactions")
def list_transactions(limit: int = 100):
    p = Path("data/transactions.csv")
    if not p.exists():
        return []
    df = pd.read_csv(p)
    df = _parse_dates_iso(df, "date")
    df = df.sort_values("date", ascending=False).head(limit)
    df["date"] = df["date"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    df = df.where(pd.notnull(df), None)
    return df.to_dict(orient="records")


@app.get("/transactions/db")
def list_transactions_db(portfolio_id: int, limit: int = 200):
    with get_session() as s:
        rows = s.exec(select(Tx).where(Tx.portfolio_id == portfolio_id).order_by(Tx.date.desc())).all()
        return [{
            "id": r.id,
            "date": pd.to_datetime(r.date).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "symbol": r.symbol,
            "quantity": r.quantity,
            "price": (None if r.price is None else float(r.price)),
            "fees": float(r.fees),
            "notes": r.notes or ""
        } for r in rows[:limit]]


# -----------------------------------------------------------------------------
# Price backfill endpoints
# -----------------------------------------------------------------------------

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


# register the router
app.include_router(router)
