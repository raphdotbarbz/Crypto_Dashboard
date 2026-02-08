# dashboards/python/app.py — DB-only (CSV removed)
from __future__ import annotations

import os
import sys
from datetime import datetime, date as _date, time as _time, timezone
from typing import Optional, Any, Dict, List

import numpy as np
import pandas as pd
import requests
import streamlit as st
import io


# Optional plotting libs
try:
    import plotly.express as px
    HAS_PX = True
except Exception:
    px = None
    HAS_PX = False

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

try:
    import plotly.io as pio
    import plotly.graph_objects as go
    pio.templates["bigger_text"] = go.layout.Template(
        layout=go.Layout(font=dict(size=22))  # tweak size if you want even bigger/smaller
    )
    pio.templates.default = "bigger_text"
except Exception:
    pass

# --- local src path for VaR imports ---
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from risk.var import (
    VaRConfig,
    historical_var_series,
    parametric_var_series,
    monte_carlo_var_point,
    historical_cvar_point,
    monte_carlo_cvar_point,
    parametric_student_t_var_es,
    fhs_var_es,
    rolling_historical_var,      # NEW
    backtest_exceptions,
    parametric_decomposition,
    historical_es_decomposition,
    # NEW
)


# Consistent colors for VaR methods
VAR_COLORS = {
    "Parametric (Normal)": "#E11900",      # Red
    "Historical":          "#DAA520",      # Gold
    "Monte Carlo":         "#2E8B57",      # Green
    "Parametric (Student-t)": "#7F3FBF",   # Purple  (NEW)
    "Filtered Historical (EWMA)": "#1F9DA3",  # Teal  (NEW)
}


# -------------------------------------------------------------
# Config
# -------------------------------------------------------------
API_BASE_DEFAULT = "http://127.0.0.1:8000"
API_BASE = os.getenv("PORTFOLIO_API_BASE", API_BASE_DEFAULT)

# Simple styling
# --- BIGGER GLOBAL FONTS (≈2×) ---

st.set_page_config(page_title="VEGA by reckless_baguette", layout="wide")

st.markdown("""
<style>
.brand-vega { display:flex; align-items:baseline; gap:8px; margin: 0 0 6px 0; }
.brand-vega .logo { font-family: 'Times New Roman', Times, serif !important;
                    font-size: 144px !important; letter-spacing: 0.06em; line-height: 1; }
.brand-vega .by { font-size: 50px !important; vertical-align: super; opacity: 0.8; }
</style>
<div class="brand-vega">
  <span class="logo">VEGA</span>
  <span class="by">by reckless_baguette</span>
</div>
""", unsafe_allow_html=True)

st.markdown(
    """
    <style>
    /* Make nearly all app text bigger */
    html, body, [data-testid="stAppViewContainer"] * { font-size: 30px !important; }

    /* Sidebar text */
    section[data-testid="stSidebar"] * { font-size: 30px !important; }

    /* Headings */
    h1, h2, h3, h4, h5, h6 { font-size: 4em !important; }

    /* Tabs */
    .stTabs [data-baseweb="tab"] { font-size: 2em !important; }

    /* Metrics */
    [data-testid="stMetricLabel"] { font-size: 2em !important; }
    [data-testid="stMetricValue"] { font-size: 3em !important; }
    [data-testid="stMetricDelta"] { font-size: 2em !important; }

    /* Tables / dataframes */
    [data-testid="stDataFrame"] div { font-size: 30px !important; }

    /* Form labels & inputs */
    label, .stTextInput label, .stNumberInput label, .stSelectbox label,
    .stDateInput label, .stTimeInput label, .stSlider label { font-size: 1.5em !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


DATA_DIR = ROOT / "data"
ARTIFACTS = DATA_DIR / "artifacts"

# -------------------------------------------------------------
# Small HTTP helpers
# -------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=15)
def api_get(path: str, params: Optional[dict] = None):
    url = f"{API_BASE}{path}"
    try:
        r = requests.get(url, params=params or {}, timeout=30)
        if r.status_code >= 400:
            return {"_error": f"GET {path} → {r.status_code}", "detail": _safe_detail(r)}
        return r.json()
    except Exception as e:
        return {"_error": f"GET {path} failed: {e}"}

@st.cache_data(show_spinner=False, ttl=0)
def api_delete(path: str, params: Optional[dict] = None):
    url = f"{API_BASE}{path}"
    try:
        r = requests.delete(url, params=params or {}, timeout=45)
        if r.status_code >= 400:
            return {"_error": f"DELETE {path} → {r.status_code}", "detail": _safe_detail(r)}
        return r.json()
    except Exception as e:
        return {"_error": f"DELETE {path} failed: {e}"}


@st.cache_data(show_spinner=False, ttl=0)
def api_post(path: str, params: Optional[dict] = None, json_body: Optional[dict | list] = None):
    url = f"{API_BASE}{path}"
    try:
        r = requests.post(url, params=params or {}, json=json_body, timeout=45)
        if r.status_code >= 400:
            return {"_error": f"POST {path} → {r.status_code}", "detail": _safe_detail(r)}
        return r.json()
    except Exception as e:
        return {"_error": f"POST {path} failed: {e}"}


def _safe_detail(resp: requests.Response):
    try:
        return resp.json()
    except Exception:
        return resp.text[:300]


def fmt_pct(x: Optional[float]) -> str:
    if x is None:
        return "—"
    try:
        return f"{x*100:.2f}%"
    except Exception:
        return "—"


@st.cache_data(show_spinner=False, ttl=60)
def _load_asset_returns(symbols: List[str], days: int = 365) -> pd.DataFrame:
    """Fetch daily closes for each symbol via API (handles /series, /history, /prices) and
    return a wide daily-returns DataFrame."""
    frames: list[pd.Series] = []

    for sym in symbols:
        for endpoint in ("/prices/series", "/prices/history", "/prices"):
            data = api_get(endpoint, params={"symbol": sym, "days": int(days)})

            ser = None
            try:
                # /prices/series -> {"ts":[ms...], "price":[...]}
                if isinstance(data, dict) and "ts" in data and "price" in data:
                    ts = pd.to_datetime(np.array(data["ts"], dtype="int64"), unit="ms", utc=True)
                    px = pd.to_numeric(np.array(data["price"], dtype=float), errors="coerce")
                    ser = pd.Series(px, index=ts, name=sym)

                # /prices/history -> {"items":[{"date":..., "price":...}, ...]}
                elif isinstance(data, dict) and "items" in data and isinstance(data["items"], list):
                    tmp = pd.DataFrame(data["items"])
                    if not tmp.empty:
                        date_col = next((c for c in tmp.columns if c.lower() in ("date", "timestamp", "time")), tmp.columns[0])
                        price_col = next((c for c in tmp.columns if "price" in c.lower() or "close" in c.lower()), tmp.columns[-1])
                        ts = pd.to_datetime(tmp[date_col], utc=True, errors="coerce")
                        px = pd.to_numeric(tmp[price_col], errors="coerce")
                        ser = pd.Series(px.values, index=ts, name=sym)

                # Legacy list-of-rows shape
                elif isinstance(data, list) and data:
                    tmp = pd.DataFrame(data)
                    date_col = next((c for c in tmp.columns if c.lower() in ("date", "timestamp", "time")), tmp.columns[0])
                    price_col = next((c for c in tmp.columns if "price" in c.lower() or "close" in c.lower()), tmp.columns[-1])
                    ts = pd.to_datetime(tmp[date_col], utc=True, errors="coerce")
                    px = pd.to_numeric(tmp[price_col], errors="coerce")
                    ser = pd.Series(px.values, index=ts, name=sym)
            except Exception:
                ser = None

            if ser is not None and not ser.dropna().empty:
                ser = ser.dropna()
                ser = ser[~ser.index.duplicated(keep="last")].sort_index()
                frames.append(ser.rename(sym))
                break  # move to next symbol

    if not frames:
        return pd.DataFrame()

    # Prices → pad to daily → returns
    prices = pd.concat(frames, axis=1).sort_index()
    prices = prices.asfreq("D").ffill()
    rets = prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna(how="all")
    return rets

@st.cache_data(show_spinner=False, ttl=60)
def _returns_for_symbol(sym: str, days: int = 365) -> pd.Series:
    """Daily returns for a single symbol via the same prices API used elsewhere."""
    df = _load_asset_returns([sym], days=days)
    if isinstance(df, pd.DataFrame) and sym in df.columns:
        s = df[sym].dropna()
        s.index = pd.to_datetime(s.index, utc=True)
        return s.sort_index()
    return pd.Series(dtype=float)

# -------------------------------------------------------------
# Sidebar controls (DB-only)
# -------------------------------------------------------------
st.sidebar.header("Settings")
API_BASE = st.sidebar.text_input("API base", value=API_BASE, help="FastAPI root, e.g. http://127.0.0.1:8000")
rf_annual = st.sidebar.number_input("Risk-free (annual)", min_value=0.0, max_value=1.0, value=0.05, step=0.005)
lookbacks = st.sidebar.text_input("Risk lookbacks (days)", value="30,180,365")

# User & Portfolio (email-based)
st.sidebar.subheader("User & Portfolio")
email = st.sidebar.text_input("Owner email", value="me@example.com")

col_u1, col_u2 = st.sidebar.columns(2)
if col_u1.button("Ensure user"):
    st.cache_data.clear()
    api_post("/users", params={"email": email})
if col_u2.button("Refresh portfolios"):
    st.session_state.pop("plist", None)

# Resolve owner_id from email via POST /users
_u = api_post("/users", params={"email": email})
owner_id = _u.get("id") if isinstance(_u, dict) else None

# Load or refresh portfolio list
plist = st.session_state.get("plist")
if plist is None and owner_id:
    got = api_get("/portfolios", params={"owner_id": owner_id})
    plist = got if isinstance(got, list) else []
    st.session_state["plist"] = plist

# Create portfolio
name = st.sidebar.text_input("New portfolio name", value="Demo")
if st.sidebar.button("Create portfolio") and owner_id:
    api_post("/portfolios", params={"owner_id": owner_id, "name": name})
    st.session_state.pop("plist", None)
    st.rerun()

# Select portfolio
portfolio_id: Optional[int] = None
if isinstance(plist, list) and plist:
    options = {f"#{p['id']}  {p['name']}": p["id"] for p in plist}
    label = st.sidebar.selectbox("Select portfolio", list(options.keys()))
    portfolio_id = options[label]
else:
    st.sidebar.info("No portfolios yet — create one above.")

# Danger zone (AFTER owner_id & portfolio_id exist)
with st.sidebar.expander("Danger zone", expanded=False):
    st.caption("Deleting a portfolio removes all its transactions. This cannot be undone.")
    confirm_text = st.text_input("Type DELETE to confirm", value="", key="confirm_delete")
    can_delete = bool(portfolio_id and owner_id and confirm_text.strip().upper() == "DELETE")
    if st.button(
        "Delete portfolio",
        type="primary",
        disabled=not can_delete,
        help="This permanently deletes the selected portfolio.",
    ):
        res = api_delete(f"/portfolios/{int(portfolio_id)}", params={"owner_id": int(owner_id), "confirm": True})
        if isinstance(res, dict) and res.get("_error"):
            st.error(res["_error"])
        else:
            st.success(f"Portfolio #{portfolio_id} deleted.")
            st.session_state.pop("plist", None)
            st.rerun()

# Params for downstream API calls
params = {"portfolio_id": int(portfolio_id)} if portfolio_id else {}


# -------------------------------------------------------------
# Helpers to build returns for VaR (DB first → artifacts)
# -------------------------------------------------------------

def _returns_from_api_totals() -> pd.Series:
    tot = api_get("/portfolio/totals", params=params)
    if isinstance(tot, list) and tot:
        df = pd.DataFrame(tot)
        cols = {c.lower(): c for c in df.columns}
        date_col = cols.get("date") or next((c for c in df.columns if "date" in c.lower() or "time" in c.lower()), None)
        val_col  = cols.get("total_value") or cols.get("portfolio_value") or next((c for c in df.columns if "value" in c.lower()), None)
        if date_col and val_col:
            s_val = pd.Series(df[val_col].values, index=pd.to_datetime(df[date_col], utc=True)).sort_index()
            return s_val.pct_change().dropna()
    return pd.Series([], dtype=float)


@st.cache_data(show_spinner=False)
def _load_value_curve() -> pd.Series:
    """Load portfolio total value curve from artifacts and return daily returns (fallback)."""
    candidates = [
        ARTIFACTS / "totals.parquet",
        ARTIFACTS / "market_values.parquet",
        ARTIFACTS / "portfolio_values.parquet",
    ]
    df = None
    for path in candidates:
        if path.exists():
            df = pd.read_parquet(path)
            break
    if df is None:
        return pd.Series([], dtype=float)

    if isinstance(df, pd.DataFrame):
        date_col = next((c for c in df.columns if c.lower() in ("date", "timestamp")), df.columns[0])
        val_col = next((c for c in df.columns if "value" in c.lower()), df.columns[-1])
        s = pd.Series(df[val_col].values, index=pd.to_datetime(df[date_col], utc=True)).sort_index()
        s = s.asfreq("D").interpolate(limit=3)
    else:
        s = pd.Series(dtype=float)

    returns = s.pct_change().dropna()
    returns.name = "portfolio_return"
    return returns


def _returns_for_var() -> pd.Series:
    # DB totals first
    if portfolio_id:
        r = _returns_from_api_totals()
        if len(r) > 0:
            return r
    # fallback: artifacts
    return _load_value_curve()


# -------------------------------------------------------------
# Overview Tab
# -------------------------------------------------------------

def render_overview():
    if not portfolio_id:
        st.info("Select or create a portfolio in the sidebar to see data.")
        return

    err_box = st.empty()

    # Fetch overview + risk tiles
    ov = api_get("/portfolio/overview", params=params)
    mt = api_get("/portfolio/metrics", params={**params, "lookbacks": lookbacks, "rf_annual": rf_annual})

    # Top tiles
    colA, colB, colC, colD = st.columns(4)
    if isinstance(ov, dict) and "total_value" in ov:
        # show as-of date
        as_of = ov.get("as_of")
        try:
            as_of_dt = pd.to_datetime(as_of, utc=True)
            asof_str = as_of_dt.strftime("%Y-%m-%d")
        except Exception:
            asof_str = "—"
        colA.metric("Total value", f"${ov['total_value']:,.2f}", help=f"As of {asof_str} UTC")
        colB.metric("1d", fmt_pct(ov.get("ret_1d")))
        colC.metric("7d", fmt_pct(ov.get("ret_7d")))
        colD.metric("30d", fmt_pct(ov.get("ret_30d")))
    else:
        if isinstance(ov, dict) and ov.get("_error"):
            err_box.error(ov.get("_error"))
        else:
            err_box.error("Failed to load /portfolio/overview")

    # Risk tile strip
    st.subheader("Risk tiles")
    if isinstance(mt, dict) and "tiles" in mt:
        tiles = mt.get("tiles", [])
        if tiles:
            cols = st.columns(min(len(tiles), 4))
            for i, t in enumerate(tiles):
                with cols[i % len(cols)]:
                    st.caption(f"Lookback: {t.get('lookback_days')}d")
                    st.metric("Sharpe", f"{t['sharpe']:.2f}" if t.get("sharpe") is not None else "—")
                    st.metric("Sortino", f"{t['sortino']:.2f}" if t.get("sortino") is not None else "—")
                    st.metric("Calmar", f"{t['calmar']:.2f}" if t.get("calmar") is not None else "—")
                    dd = t.get("max_drawdown")
                    st.metric("Max DD", fmt_pct(dd) if dd is not None else "—")
        else:
            st.info("No risk tiles for the chosen window(s).")
    else:
        if isinstance(mt, dict) and mt.get("_error"):
            st.error(mt["_error"])  # show status

    # --- Portfolio vol (annualized) tiles ---
    port_rets = _returns_for_var()
    if port_rets is not None and len(port_rets) >= 20:
        def _ann_vol(s: pd.Series) -> float | None:
            s = pd.Series(s).dropna()
            if s.empty:
                return None
            v = float(s.std(ddof=1)) * np.sqrt(365)
            return v if np.isfinite(v) else None

        v30 = _ann_vol(port_rets.tail(30)) if len(port_rets) >= 30 else None
        v180 = _ann_vol(port_rets.tail(180)) if len(port_rets) >= 180 else None
        v365 = _ann_vol(port_rets.tail(365)) if len(port_rets) >= 365 else None

        vcols = st.columns(3)
        vcols[0].metric("Annualized Volatility", "—" if v30 is None else f"{v30:.1%}")
        vcols[1].metric("Annualized Volatility", "—" if v180 is None else f"{v180:.1%}")
        vcols[2].metric("Annualized Volatility", "—" if v365 is None else f"{v365:.1%}")

    # Charts: totals curve & drawdown
    st.subheader("Value curve & drawdown")
    tot = api_get("/portfolio/totals", params=params)
    if isinstance(tot, list) and tot:
        df = pd.DataFrame(tot)
        try:
            df["date"] = pd.to_datetime(df["date"], utc=True)
            df = df.sort_values("date")

            # Staleness warning (avoid phantom flat tails)
            last_day = df["date"].max().date()
            stale_days = (datetime.now(timezone.utc).date() - last_day).days
            if stale_days > 1:
                st.warning(f"Price data is {stale_days} days stale (last: {last_day}).")

            st.line_chart(df.set_index("date")["total_value"], height=240)
            peak = df["total_value"].cummax()
            dd = df["total_value"] / peak - 1.0
            dd_df = pd.DataFrame({"date": df["date"], "drawdown": dd})
            st.line_chart(dd_df.set_index("date")["drawdown"], height=160)
        except Exception as e:
            st.warning(f"Could not parse totals for charting: {e}")
    else:
        st.info("/portfolio/totals not available yet or no data.")

    # --- Top holdings (now with qty / $ / 1d-7d-30d) ---
    st.subheader("Top holdings")
    if isinstance(ov, dict) and ov.get("top_holdings"):
        th = pd.DataFrame(ov["top_holdings"]).sort_values("weight", ascending=False)

        max_slices = 6  # keep pie legible
        top_assets = th.head(max_slices).copy()

        # --- latest prices per symbol (from overview payload first) ---
        price_map = pd.Series(ov.get("last_prices", {}), dtype=float)
        if price_map.empty:
            try:
                from src.io.loaders import load_last_prices
                price_map = load_last_prices(symbols=top_assets["symbol"].unique().tolist()).astype(float)
            except Exception:
                price_map = pd.Series(index=top_assets["symbol"].unique(), dtype=float)

        # attach price for calculations + display
        top_assets["price"] = top_assets["symbol"].map(price_map)

        # helpers
        def _fmt_price(x):
            if pd.isna(x): return None
            return f"${x:,.2f}" if x >= 1 else f"${x:.6f}"

        def _fmt_qty(q):
            if pd.isna(q): return "—"
            q = float(q)
            if q >= 100:   return f"{q:,.2f}"
            if q >= 1:     return f"{q:,.4f}"
            return f"{q:,.8f}"

        # --- position $ and quantity ---
        portfolio_value = float(ov.get("total_value", np.nan)) if isinstance(ov, dict) else np.nan
        # prefer per-holding "value" from API, fallback to weight * portfolio_value
        top_assets["pos_value"] = np.where(
            top_assets.get("value").notna() if "value" in top_assets else False,
            top_assets["value"].astype(float),
            (top_assets["weight"].astype(float) * portfolio_value) if np.isfinite(portfolio_value) else np.nan
        )
        top_assets["quantity"] = np.where(
            (top_assets["price"] > 0) & top_assets["pos_value"].notna(),
            top_assets["pos_value"] / top_assets["price"],
            np.nan
        )

        # --- per-asset returns (1d / 7d / 30d) ---
        sym_list = top_assets["symbol"].tolist()
        rets = _load_asset_returns(sym_list, days=40)  # daily returns; ~1m buffer

        def _hret(s: pd.Series, n: int) -> float | None:
            if s is None or s.dropna().empty or s.tail(n).dropna().shape[0] < max(1, n // 2):
                return None
            try:
                return float(np.prod(1 + s.tail(n).astype(float).dropna()) - 1)
            except Exception:
                return None

        r1 = {};
        r7 = {};
        r30 = {}
        for s in sym_list:
            col = (rets[s] if isinstance(rets, pd.DataFrame) and s in rets.columns else None)
            r1[s] = (None if col is None or col.dropna().empty else float(col.tail(1).iloc[-1]))
            r7[s] = _hret(col, 7)
            r30[s] = _hret(col, 30)

        top_assets["ret_1d"] = top_assets["symbol"].map(r1)
        top_assets["ret_7d"] = top_assets["symbol"].map(r7)
        top_assets["ret_30d"] = top_assets["symbol"].map(r30)

        # ---- Pie on the left; richer table on the right ----
        plot_df = top_assets[["symbol", "weight"]].copy()
        col1, col2 = st.columns([1.1, 1])

        # one shared height for both visuals
        PLOT_HEIGHT = 420  # tweak to taste (e.g., 440/460)

        if HAS_PLOTLY and not plot_df.empty:
            with col1:
                fig = go.Figure([
                    go.Pie(
                        labels=plot_df["symbol"],
                        values=plot_df["weight"],
                        hole=0.45,
                        sort=False,
                        # 👇 turn on labels & percents right on the chart
                        textinfo="label+percent",
                        textposition="inside",
                        insidetextorientation="radial",
                        textfont=dict(size=16),
                        # keep hover clean
                        hovertemplate="%{label}: %{percent:.1%}<extra></extra>",
                        marker=dict(line=dict(color="#0e1117", width=1)),
                    )
                ])
                fig.update_layout(
                    template=None,
                    showlegend=False,
                    paper_bgcolor="#0e1117",
                    plot_bgcolor="#0e1117",
                    font=dict(color="white"),
                    hoverlabel=dict(bgcolor="#222", font_color="white", bordercolor="#555", font_size=13),
                    margin=dict(l=10, r=10, t=10, b=10),
                    height=PLOT_HEIGHT,  # 👈 match the table height
                )
                st.plotly_chart(fig, use_container_width=True)
        elif HAS_MPL and not plot_df.empty:
            with col1:
                fig, ax = plt.subplots(figsize=(4, 4))
                ax.set_facecolor("black");
                fig.patch.set_facecolor("black")
                ax.pie(plot_df["weight"], startangle=90, counterclock=False,
                       wedgeprops={"width": 0.45}, labels=plot_df["symbol"])
                ax.axis("equal")
                st.pyplot(fig, clear_figure=True, use_container_width=True)
        else:
            with col1:
                st.bar_chart(plot_df.set_index("symbol")["weight"], height=PLOT_HEIGHT)

        # Right-side table with all the goodies
        with col2:
            table = top_assets[
                ["symbol", "weight", "price", "quantity", "pos_value", "ret_1d", "ret_7d", "ret_30d"]].copy()
            table["weight %"] = (table["weight"] * 100).round(1)
            table["price"] = table["price"].apply(_fmt_price)
            table["quantity"] = table["quantity"].apply(_fmt_qty)
            table["Position $"] = table["pos_value"].map(lambda x: ("—" if pd.isna(x) else f"${x:,.2f}"))
            for c in ["ret_1d", "ret_7d", "ret_30d"]:
                table[c] = table[c].map(lambda v: ("—" if v is None or pd.isna(v) else f"{v:.2%}"))

            st.dataframe(
                table.rename(columns={
                    "symbol": "symbol",
                    "weight %": "weight %",
                    "price": "price",
                    "quantity": "qty",
                    "ret_1d": "1d",
                    "ret_7d": "7d",
                    "ret_30d": "30d",
                })[["symbol", "weight %", "price", "qty", "Position $", "1d", "7d", "30d"]],
                use_container_width=True,
                hide_index=True,
                height=PLOT_HEIGHT,  # 👈 same as the pie
            )

    # --- Benchmark comparison: beta / IR / excess ---
    st.subheader("Benchmark comparison")
    bcol1, bcol2, bcol3 = st.columns([1.2, 1, 2])

        # Two-sentence help blurbs for all the benchmark stats
    BENCH_HELP = {
        "beta": (
            "Beta measures how sensitive your portfolio is to moves in the benchmark. "
            "A beta of 1.2 implies a 1% benchmark move is associated with ~1.2% portfolio move on average."
        ),
        "ir": (
            "The Information Ratio is excess return divided by tracking error (active risk). "
            "Higher is better; ~0.5 is decent and ~1.0+ suggests consistent active outperformance."
        ),
        "excess": (
            "Annualized excess return is your portfolio’s return minus the benchmark’s, scaled to a yearly rate. "
            "Positive values indicate outperformance after accounting for market direction."
         ),
        "alpha": (
            "Jensen’s alpha estimates risk-adjusted out/underperformance after removing the portion explained by beta and the risk-free rate. "
            "Positive alpha suggests value beyond market exposure; negative implies a drag."
        ),
        "te": (
            "Tracking error is the standard deviation of the active return (portfolio − benchmark). "
            "Lower TE means you hug the benchmark closely; higher TE means bigger active bets."
        ),
        "corr": (
            "Correlation shows how tightly your returns move with the benchmark on a −1 to 1 scale. "
            "Near 1 = move together, near 0 = unrelated, negative = move in opposite directions."
        ),
        "bench_sym": (
            "Choose the asset used as your market proxy (e.g., BTC). "
            "If your API supports it, try a total-market series such as TOTAL for crypto market cap."
        ),
        "lookback": (
            "Window over which statistics are computed (requires enough overlapping daily data). "
            "Short windows react faster; long windows are more stable."
        ),
    }

        # Let user choose a benchmark symbol. Default to BTC.
    bench_sym = bcol1.text_input(
        "Benchmark symbol",
        value=st.session_state.get("bench_sym", "BTC"),
        help=BENCH_HELP["bench_sym"],
    ).upper().strip()
    lookback = bcol2.select_slider(
        "Lookback (days)",
        options=[30, 90, 180, 365],
        value=180,
        help=BENCH_HELP["lookback"],
    )
    bcol3.caption("Tip: if your API supports it, try 'TOTAL' (or your alias) for total crypto market cap.")

    b_rets = _returns_for_symbol(bench_sym, days=max(lookback + 10, 400))
    if b_rets.empty and bench_sym != "BTC":
        st.info(f"Could not load {bench_sym}; falling back to BTC.")
        bench_sym = "BTC"
        b_rets = _returns_for_symbol(bench_sym, days=max(lookback + 10, 400))

    if port_rets is None or port_rets.empty or b_rets.empty:
        st.info("Need overlapping returns for portfolio and benchmark.")
    else:
        pr = port_rets.tail(lookback).dropna()
        br = b_rets.reindex(pr.index).dropna()
        pr = pr.reindex(br.index).dropna()
        if len(pr) < 20:
            st.info("Not enough overlap to compute metrics.")
        else:
            # Daily excess (portfolio - benchmark)
            ex = pr - br

            # Tracking error (ann.) and Information Ratio
            te_daily = float(ex.std(ddof=1)) if ex.std(ddof=1) is not None else np.nan
            te_ann = te_daily * np.sqrt(365) if np.isfinite(te_daily) else np.nan
            ir = (float(ex.mean()) / te_daily * np.sqrt(365)) if (
                        np.isfinite(te_daily) and te_daily > 0) else np.nan

            # Beta and correlation
            var_b = float(np.var(br, ddof=1))
            cov_pb = float(np.cov(pr, br)[0, 1]) if len(pr) == len(br) else np.nan
            beta = (cov_pb / var_b) if (np.isfinite(cov_pb) and var_b > 0) else np.nan
            corr = float(np.corrcoef(pr, br)[0, 1]) if len(pr) == len(br) else np.nan

            # Annualized excess return (geometric)
            try:
                ex_ann = float((np.prod(1 + ex) ** (365 / len(ex)) - 1))
            except Exception:
                ex_ann = np.nan

                # Jensen's alpha (annualized) using rf_annual from the sidebar
            rf_d = float(rf_annual) / 365.0
            if np.isfinite(beta):
                alpha_daily = float((pr - rf_d - beta * (br - rf_d)).mean())
                alpha_ann = alpha_daily * 365
            else:
                alpha_ann = np.nan

            m1, m2, m3, m4, m5, m6 = st.columns(6)

            m1.metric(
                f"β vs {bench_sym}",
                "—" if not np.isfinite(beta) else f"{beta:.2f}",
                help=BENCH_HELP["beta"],
            )
            m2.metric(
                "Info ratio",
                "—" if not np.isfinite(ir) else f"{ir:.2f}",
                help=BENCH_HELP["ir"],
            )
            m3.metric(
                "Excess (ann.)",
                "—" if not np.isfinite(ex_ann) else f"{ex_ann:.1%}",
                help=BENCH_HELP["excess"],
            )
            m4.metric(
                "α (ann.)",
                "—" if not np.isfinite(alpha_ann) else f"{alpha_ann:.1%}",
                help=BENCH_HELP["alpha"],
            )
            m5.metric(
                "Tracking error (ann.)",
                "—" if not np.isfinite(te_ann) else f"{te_ann:.1%}",
                help=BENCH_HELP["te"],
            )
            m6.metric(
                "Correlation",
                "—" if not np.isfinite(corr) else f"{corr:.2f}",
                help=BENCH_HELP["corr"],
            )
    # -------------------------------------------------------------
    # Append transactions (DB only)
    # -------------------------------------------------------------
    def _to_iso_utc(d: _date, t: _time) -> str:
        dt = datetime.combine(d, t).replace(tzinfo=timezone.utc)
        return dt.isoformat().replace("+00:00", "Z")

    st.subheader("Append transaction")
    with st.form("append_tx"):
        today = datetime.now(timezone.utc).date()
        c1, c2, c3, c4, c5 = st.columns([1.2, 1, 1, 1, 1.2])

        tx_date = c1.date_input("Trade date (UTC)", value=today, max_value=today, key="tx_date")
        tx_time = c2.time_input("Time (UTC)", value=_time(0, 0), step=60, key="tx_time")
        sym_raw = c3.text_input("Symbol", value="ONDO", key="tx_symbol")
        qty = c4.number_input("Quantity (+/-)", value=1.0, key="tx_qty")
        price_s = c5.text_input("Price (optional)", value="", key="tx_price")

        submitted = st.form_submit_button("Append")

        if submitted:
            if not portfolio_id:
                st.error("Select a portfolio first.")
                st.stop()

            sym = sym_raw.strip().upper()
            if not sym:
                st.error("Symbol cannot be empty.")
                st.stop()

            ts_iso = _to_iso_utc(tx_date, tx_time)
            item: Dict[str, Any] = {"date": ts_iso, "symbol": sym, "quantity": float(qty)}
            if price_s.strip():
                try:
                    item["price"] = float(price_s)
                except ValueError:
                    st.error("Price must be numeric if provided.")
                    st.stop()

            res = api_post("/transactions/db/append", params={"portfolio_id": int(portfolio_id)}, json_body=[item])
            if isinstance(res, dict) and res.get("_error"):
                st.error(res["_error"])
            else:
                st.success(f"Appended {sym} on {ts_iso}. Refreshing prices…")
                # Auto-backfill prices for this symbol so overview/metrics/totals won't 500
                try:
                    _backfill = api_post("/prices/backfill", json_body={"symbols": [sym], "days": "max"})
                    if isinstance(_backfill, dict) and "results" in _backfill:
                        errs = [r for r in _backfill["results"] if r.get("error")]
                        if errs:
                            st.warning(f"Price backfill had issues: {errs}")
                except Exception as e:
                    st.warning(f"Backfill request failed: {e}")

                st.cache_data.clear()
                _ = api_get("/portfolio/totals", params={"portfolio_id": int(portfolio_id)})
                _ = api_get("/portfolio/overview", params={"portfolio_id": int(portfolio_id)})
                st.rerun()

    # -------------------------------------------------------------
    # Recent transactions (DB only)
    # -------------------------------------------------------------
    st.subheader("Recent transactions")
    tx = api_get("/transactions/db", params={"portfolio_id": int(portfolio_id), "limit": 200})

    if isinstance(tx, list) and tx:
        df_tx = pd.DataFrame(tx)
        if "date" in df_tx.columns:
            df_tx["date"] = pd.to_datetime(df_tx["date"], errors="coerce")
            df_tx = df_tx.sort_values("date", ascending=False)
        st.dataframe(df_tx, use_container_width=True)
    elif isinstance(tx, dict) and tx.get("_error"):
        st.error(tx["_error"])  # status code insight
    else:
        st.caption("No transactions found.")


# -------------------------------------------------------------
# VaR Tab (multi-method VaR + CVaR)
# -------------------------------------------------------------

def render_var_tab():
    st.subheader("Value at Risk (VaR)")

    # --- data (DB totals first → artifacts fallback) ---
    base_returns = _returns_for_var()
    if base_returns is None or len(base_returns) < 20:
        st.warning("Need at least ~20 daily points to compute VaR.")
        return

    # Latest portfolio value (for $ amounts)
    ov = api_get("/portfolio/overview", params=params) if params else {}
    portfolio_value = float(ov.get("total_value", np.nan)) if isinstance(ov, dict) else np.nan

    # --- controls ---
    left, right = st.columns([2, 1])
    use_student_t = st.checkbox(
        "Use Student-t for Parametric VaR",
        value=True,
        help="Leptokurtic/heavy-tail friendly. Falls back to Normal if SciPy fit fails."
    )

    with left:
        horizon = st.select_slider("Horizon (days)", options=[1, 10, 30], value=10)
        confidence = st.slider("Confidence level", 0.90, 0.999, 0.95, step=0.01)
        METHOD_CHOICES = [
            "Historical",
            "Parametric (Normal)",
            "Monte Carlo",
            "Parametric (Student-t)",
            "Filtered Historical (EWMA)",
        ]
        methods = st.multiselect(
            "Methods",
            METHOD_CHOICES,
            default=["Historical", "Parametric (Normal)", "Monte Carlo"],
        )
        if use_student_t and "Parametric (Student-t)" not in methods:
            methods = ["Parametric (Student-t)"] + methods

    with right:
        sims = st.number_input("Monte Carlo simulations", value=10000, min_value=1000, max_value=200000, step=1000)
        seed = st.number_input("Random seed", value=42, step=1)

    # 1-sentence tooltips
    explain = {
        "Historical": "Historical VaR uses the past return distribution directly; the cut marks the worst 1−confidence tail.",
        "Parametric (Normal)": "Parametric VaR assumes returns are ~normal; it uses μ and σ to estimate the tail.",
        "Monte Carlo": "Monte Carlo VaR simulates many return paths from an estimated distribution and takes the tail quantile.",
        "Parametric (Student-t)": "Student-t VaR allows fat tails via ν degrees of freedom; more realistic for crypto drawdowns.",
        "Filtered Historical (EWMA)": "FHS de-vols returns with EWMA, resamples residuals, then re-vols—capturing volatility clustering.",
        "ES": "Expected Shortfall is the average loss once you’re already beyond the VaR threshold.",
    }

    # --- helper to compute latest VaR for a given window & method (returns positive loss %) ---
    def _var_last(window: int, method: str) -> float | None:
        sub = base_returns.tail(window)
        if len(sub) < 20:
            return None

        cfg = VaRConfig(
            confidence=confidence,
            window=len(sub),
            horizon_days=horizon,
            n_sims=int(sims),
            seed=int(seed),
        )
        try:
            if method == "Historical":
                s = historical_var_series(sub, cfg=cfg)
                v = float(s.dropna().iloc[-1]) if s.notna().any() else None
            elif method == "Parametric (Normal)":
                s = parametric_var_series(sub, cfg=cfg)
                v = float(s.dropna().iloc[-1]) if s.notna().any() else None
            elif method == "Monte Carlo":
                v = float(monte_carlo_var_point(sub, cfg=cfg))
            elif method == "Parametric (Student-t)":
                v, _ = parametric_student_t_var_es(sub, confidence=confidence, horizon_days=horizon)
                v = float(v) if np.isfinite(v) else None
            elif method == "Filtered Historical (EWMA)":
                v, _ = fhs_var_es(sub, confidence=confidence, horizon_days=horizon, n_sims=int(sims), seed=int(seed))
                v = float(v) if np.isfinite(v) else None
            else:
                v = None
        except Exception:
            v = None
        return (abs(v) if v is not None else None)

    # --- latest VaR for 30/180/365-day lookbacks ---
    windows = [30, 180, 365]
    rows = []
    for m in methods:
        for w in windows:
            v = _var_last(w, m)
            if v is not None:
                rows.append({"method": m, "lookback": w, "var_pct": v})
    df_var = pd.DataFrame(rows)

    if df_var.empty:
        st.info("No VaR values for the selected settings.")
        return

    # $ amounts
    if np.isfinite(portfolio_value):
        df_var["var_$"] = -(df_var["var_pct"] * portfolio_value)

    # --- rolling Student-t VaR helper (series) ---
    def student_t_var_series(returns: pd.Series, cfg: VaRConfig) -> pd.Series:
        r = pd.Series(returns).astype(float).replace([np.inf, -np.inf], np.nan).dropna()
        if r.empty:
            return pd.Series(dtype=float, index=returns.index)
        vals, idx = [], []
        for end in range(cfg.window - 1, len(r)):
            window = r.iloc[end - cfg.window + 1 : end + 1]
            v, _ = parametric_student_t_var_es(window.values, confidence=cfg.confidence, horizon_days=cfg.horizon_days)
            vals.append(float(v) if v is not None else np.nan)
            idx.append(r.index[end])
        s = pd.Series(vals, index=idx)
        return s.reindex(returns.index)

    # --- chart: latest VaR by lookback ---
    st.subheader("Latest VaR by lookback (30 / 180 / 365 days)")
    if HAS_PX:
        fig = px.bar(
            df_var,
            x="lookback",
            y="var_pct",
            color="method",
            barmode="group",
            labels={"lookback": "Lookback (days)", "var_pct": "VaR (loss %)"},
            color_discrete_map=VAR_COLORS,
            category_orders={"method": [
                "Parametric (Normal)",
                "Parametric (Student-t)",
                "Historical",
                "Filtered Historical (EWMA)",
                "Monte Carlo",
            ]},
        )
        hover_text = [
            f"<b>{r['method']}</b><br>Lookback: {int(r['lookback'])}d<br>VaR: {r['var_pct']*100:.2f}%<br>{explain.get(r['method'],'')}"
            for _, r in df_var.iterrows()
        ]
        fig.update_traces(hovertemplate="%{customdata}", customdata=np.array(hover_text).reshape(-1, 1))
        fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), legend_title_text="Method")
        st.plotly_chart(fig, use_container_width=True)
    else:
        pivot = df_var.pivot(index="lookback", columns="method", values="var_pct").sort_index()
        st.dataframe(pivot.applymap(lambda v: f"{v:.2%}"), use_container_width=True)
        st.bar_chart(pivot)

    # --- tiles: % and $ loss (with hover help) ---
    st.subheader("VaR tiles (percent and amount)")
    cols = st.columns(min(3, len(methods)))
    for i, (m, g) in enumerate(df_var.groupby("method")):
        with cols[i % len(cols)]:
            v = float(g["var_pct"].max())  # most conservative across 30/180/365
            amt = float(-(v * portfolio_value)) if np.isfinite(portfolio_value) else None
            st.metric(
                label=f"{m} VaR @ {int(confidence*100)}% (h={horizon}d)",
                value=(f"{v*100:.2f}%" if pd.notna(v) else "—"),
                help=explain.get(m, ""),
            )
            if amt is not None:
                st.caption(f"≈ ${amt:,.0f} loss on a ${portfolio_value:,.0f} portfolio")

    # --- recent returns + rolling VaR overlays ---
    st.subheader("Recent returns + VaR lines (+/−1σ)")
    recent = base_returns.tail(365).astype(float)
    if len(recent) == 0:
        st.warning("No recent returns to plot.")

    sigma30 = recent.rolling(30, min_periods=20).std(ddof=1)
    roll_window = max(20, min(60, len(recent)))
    cfg_roll = VaRConfig(confidence=confidence, window=roll_window, horizon_days=horizon, n_sims=int(sims), seed=int(seed))

    pvar = hvar = tvar = None
    p_err = h_err = t_err = None
    if "Parametric (Normal)" in methods:
        try:
            pvar = parametric_var_series(recent, cfg=cfg_roll).dropna()
        except Exception as e:
            p_err = str(e)
    if "Historical" in methods:
        try:
            hvar = historical_var_series(recent, cfg=cfg_roll).dropna()
        except Exception as e:
            h_err = str(e)
    if "Parametric (Student-t)" in methods:
        try:
            tvar = student_t_var_series(recent, cfg=cfg_roll).dropna()
        except Exception as e:
            t_err = str(e)

    def _naive_index(ix):
        try:
            return ix.tz_convert(None)
        except Exception:
            try:
                return ix.tz_localize(None)
            except Exception:
                return ix

    idx = _naive_index(recent.index)
    df_plot = pd.DataFrame(index=idx)
    df_plot["Return"] = recent.values
    df_plot["+1σ (30d)"] = sigma30.reindex(recent.index).values
    df_plot["-1σ (30d)"] = -sigma30.reindex(recent.index).values
    if pvar is not None and len(pvar) > 0:
        df_plot["Parametric VaR (rolling)"] = -pvar.reindex(recent.index).values
    if hvar is not None and len(hvar) > 0:
        df_plot["Historical VaR (rolling)"] = -hvar.reindex(recent.index).values
    if tvar is not None and len(tvar) > 0:
        df_plot["Student-t VaR (rolling)"] = -tvar.reindex(recent.index).values

    if HAS_PX:
        melt = (
            df_plot.reset_index()
            .rename(columns={"index": "date"})
            .melt(id_vars="date", var_name="series", value_name="value")
            .dropna(subset=["value"])
        )
        if melt.empty:
            st.line_chart(df_plot["Return"])
        else:
            fig = px.line(
                melt, x="date", y="value", color="series",
                color_discrete_map={
                    "Return": "#4B5563",
                    "Parametric VaR (rolling)": VAR_COLORS.get("Parametric (Normal)", "#E11900"),
                    "Historical VaR (rolling)": VAR_COLORS.get("Historical", "#DAA520"),
                    "Student-t VaR (rolling)": VAR_COLORS.get("Parametric (Student-t)", "#7F3FBF"),
                    "+1σ (30d)": "#6E7B8B",
                    "-1σ (30d)": "#6E7B8B",
                },
                labels={"value": "Daily return", "series": "", "date": ""},
            )
            for tr in fig.data:
                if tr.name == "Return":
                    tr.line.update(width=2)
                elif tr.name in ("+1σ (30d)", "-1σ (30d)"):
                    tr.line.update(width=1.5, dash="dash" if tr.name == "-1σ (30d)" else None)
                elif "VaR" in tr.name:
                    tr.line.update(width=2)
            fig.update_yaxes(tickformat=".1%")
            fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), showlegend=True,
                              legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.line_chart(df_plot)

    # Surface calc errors (if any)
    if p_err: st.caption(f"Parametric VaR error: {p_err}")
    if h_err: st.caption(f"Historical VaR error: {h_err}")
    if t_err: st.caption(f"Student-t VaR error: {t_err}")

    # --- VaR backtest (rolling Historical VaR on recent data) ---
    st.subheader(f"Backtest: exceptions vs Historical VaR @ {int(confidence * 100)}%")

    # Keep 0.95 here: rolling VaR uses the quantile level
    rv = rolling_historical_var(recent, window=roll_window, alpha=confidence)

    # IMPORTANT: Kupiec uses exception probability p = 1 - confidence
    tail_p = 1.0 - float(confidence)
    bt = backtest_exceptions(recent, rv, alpha=tail_p)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Exceptions", f"{bt.exceptions}/{bt.n}")
    with c2:
        st.metric("POF p-value", "—" if bt.pof_pvalue is None else f"{bt.pof_pvalue:.3f}",
                  help="POF p-value – Kupiec “Proportion of Failures” test. It checks whether your breach rate ≈ expected rate (here, 5%). High p-value (e.g., > 0.05) → coverage looks fine. Low p-value (e.g., < 0.05) → VaR is mis-calibrated (too tight or too loose).")
    with c3:
        st.metric("IND p-value", "—" if bt.ind_pvalue is None else f"{bt.ind_pvalue:.3f}",
                  help="IND p-value – Christoffersen independence test. It checks. Whether breaches are independent (not clustered). High p-value → no evidence of clustering.Low p-value → breaches tend to come in clusters (volatility regimes).")

    # Show magnitudes only: VaR ≥ 0, and realized loss ≥ 0 (zero on up days)
    loss_mag = (-recent).clip(lower=0)  # negative returns → positive loss; gains → 0
    var_mag = pd.Series(rv, index=recent.index).abs()  # make sure VaR is positive magnitude

    bt_frame = pd.DataFrame({"loss": loss_mag, "VaR": var_mag}).dropna().tail(180)
    st.area_chart(bt_frame)

    # --- $ comparison vs portfolio size ---
    st.subheader("Risk in $ vs Portfolio size")
    sigma_recent = float(np.nan_to_num(recent.std(ddof=1), nan=0.0))
    summary_rows = []
    if np.isfinite(portfolio_value):
        summary_rows.append({"metric": "Portfolio size ($)", "amount": float(portfolio_value)})

    for m in methods:
        g = df_var.loc[df_var["method"] == m]
        if not g.empty and np.isfinite(portfolio_value):
            v = float(g["var_pct"].max())
            summary_rows.append({"metric": f"{m} VaR $ (max 30/180/365)", "amount": float(-(v * portfolio_value))})

    if np.isfinite(portfolio_value) and sigma_recent > 0:
        summary_rows.append({"metric": "Std dev $ (1σ/day)", "amount": float(-(sigma_recent * portfolio_value))})

    df_amt = pd.DataFrame(summary_rows)
    if not df_amt.empty and HAS_PX:
        fig_amt = px.bar(df_amt, x="amount", y="metric", orientation="h", labels={"amount": "Amount ($)", "metric": ""})
        fig_amt.update_layout(margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_amt, use_container_width=True)
    elif not df_amt.empty:
        st.dataframe(df_amt, use_container_width=True)
    else:
        st.caption("Add a portfolio or enable methods to see $ comparisons.")

    # --- forward risk cone (parametric, next 30 days) ---
    st.subheader("Forward risk cone (next 30 days)")
    try:
        cone_days = 30
        est = base_returns.tail(max(windows))
        mu_est = float(est.mean())
        sigma_est = float(est.std(ddof=1))
        alpha = 1 - confidence
        try:
            from scipy.stats import norm
            z = float(norm.ppf(alpha))  # negative
        except Exception:
            z_table = {0.10: -1.2816, 0.05: -1.6449, 0.025: -1.9600, 0.01: -2.3263, 0.005: -2.5758}
            z = z_table.get(round(alpha, 3), -2.3263)
        dates = pd.date_range(datetime.now(timezone.utc).date(), periods=cone_days + 1, freq="D")
        t = np.arange(cone_days + 1)
        med = mu_est * t
        low = (mu_est * t) + z * sigma_est * np.sqrt(t + 1e-9)
        df_cone = pd.DataFrame({"date": dates, "median": med, "lower": low})
        if HAS_PX:
            fcone = go.Figure()
            fcone.add_trace(go.Scatter(x=df_cone["date"], y=df_cone["lower"], name=f"Lower ({int(confidence*100)}%)", mode="lines"))
            fcone.add_trace(go.Scatter(x=df_cone["date"], y=df_cone["median"], name="Median", mode="lines", fill="tonexty"))
            fcone.update_layout(yaxis_title="Cumulative return", margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fcone, use_container_width=True)
        else:
            st.line_chart(df_cone.set_index("date")[["lower", "median"]])
        if np.isfinite(portfolio_value):
            st.caption("Approximate $ cone: multiply the y-axis return by your current portfolio value.")
    except Exception as e:
        st.info(f"Could not draw cone: {e}")

    # --- CVaR tiles (Expected Shortfall) ---
    st.subheader("Conditional VaR (Expected Shortfall)")
    es_rows = []
    es_window = 30
    tail = base_returns.tail(es_window)

    if "Historical" in methods:
        try:
            cfg_es = VaRConfig(confidence=confidence, window=len(tail), horizon_days=horizon, n_sims=int(sims), seed=int(seed))
            es_hist = historical_cvar_point(tail, cfg=cfg_es)
            es_rows.append(("Historical ES", es_hist))
        except Exception:
            pass

    if "Monte Carlo" in methods:
        try:
            cfg_es = VaRConfig(confidence=confidence, window=len(tail), horizon_days=horizon, n_sims=int(sims), seed=int(seed))
            es_mc = monte_carlo_cvar_point(tail, cfg=cfg_es)
            es_rows.append(("Monte Carlo ES", es_mc))
        except Exception:
            pass

    if "Parametric (Student-t)" in methods:
        try:
            _v, es_t = parametric_student_t_var_es(tail, confidence=confidence, horizon_days=horizon)
            es_rows.append(("Student-t ES", es_t))
        except Exception:
            pass

    if "Filtered Historical (EWMA)" in methods:
        try:
            _v, es_fhs = fhs_var_es(tail, confidence=confidence, horizon_days=horizon, n_sims=int(sims), seed=int(seed))
            es_rows.append(("FHS ES", es_fhs))
        except Exception:
            pass

    cols = st.columns(max(1, min(3, len(es_rows))))
    for i, (label, val) in enumerate(es_rows):
        with cols[i % len(cols)]:
            st.metric(
                label=f"{label} ({int(confidence * 100)}% | {horizon}d)",
                value=("—" if val is None or not np.isfinite(val) else f"{abs(val):.2%}"),
                help=explain["ES"],
            )

    # --- VaR decomposition (Parametric Normal) ---
    st.subheader("VaR decomposition by asset")

    # Only run if Parametric Normal is enabled
    if "Parametric (Normal)" not in methods:
        st.info("Enable “Parametric (Normal)” in Methods to see decomposition.")
    else:
        # 1) Choose lookback
        decomp_days = st.select_slider(
            "Decomposition lookback (days)",
            options=[30, 60, 90, 180, 365],
            value=90
        )

        # Pull latest holdings from the same overview payload you already fetched
        holds = (ov.get("top_holdings") or []) if isinstance(ov, dict) else []
        if not holds:
            st.info("No holdings to decompose.")
        else:
            # Use the top K assets automatically (no user control)
            TOP_CAP = 8
            hh = sorted(holds, key=lambda x: x.get("weight", 0.0), reverse=True)[:TOP_CAP]
            if len(holds) > TOP_CAP:
                st.caption(f"Using top {TOP_CAP} assets by weight.")

            symbols = [h["symbol"] for h in hh if h.get("symbol")]
            w_raw = np.array([float(h.get("weight", 0.0)) for h in hh], dtype=float)
            if w_raw.sum() <= 0:
                st.info("Holdings weights sum to zero.")
            else:
                w = w_raw / w_raw.sum()

                # 2) Build aligned close matrix from /prices/series
                def _prices_for(sym: str, days: int) -> pd.Series:
                    # returns a UTC-indexed Series of price (float)
                    resp = api_get("/prices/series", params={"symbol": sym, "days": int(days)})
                    if not isinstance(resp, dict) or "ts" not in resp or "price" not in resp:
                        return pd.Series(dtype=float)
                    ts = pd.to_datetime(np.array(resp["ts"], dtype="int64"), unit="ms", utc=True)
                    px = pd.to_numeric(np.array(resp["price"], dtype=float), errors="coerce")
                    s = pd.Series(px, index=ts).dropna()
                    s = s[~s.index.duplicated(keep="last")].sort_index()
                    return s

                frames = []
                for s in symbols:
                    ser = _prices_for(s, decomp_days + 5)  # a few extra days to absorb weekend gaps
                    if len(ser) > 0:
                        frames.append(ser.rename(s))

                if not frames:
                    st.info("No price series available for the selected assets.")
                else:
                    closes = pd.concat(frames, axis=1, join="inner").dropna(how="any")
                    closes = closes.tail(decomp_days)  # enforce requested window
                    if closes.shape[1] < 2 or closes.shape[0] < 20:
                        st.warning("Need at least 2 assets and ~20 common daily points.")
                    else:
                        rets = closes.pct_change().dropna(how="any")

                        # 3) Parametric (Normal) contributions
                        Σ = rets.cov()
                        symbols_aligned = list(Σ.columns)

                        # map weights to aligned order
                        w_map = {sym: w[i] for i, sym in enumerate(symbols)}
                        w_vec = np.array([w_map.get(sym, 0.0) for sym in symbols_aligned], dtype=float)
                        if np.allclose(w_vec.sum(), 0.0):
                            st.info("Weights collapsed to zero after alignment.")
                        else:
                            # z > 0 at high confidence (e.g., 1.645 @ 95%)
                            try:
                                from scipy.stats import norm
                                z = float(norm.ppf(confidence))
                            except Exception:
                                z_table = {0.90: 1.2816, 0.95: 1.6449, 0.975: 1.9600, 0.99: 2.3263, 0.995: 2.5758}
                                z = z_table.get(round(float(confidence), 3), 1.6449)

                            # portfolio stdev over 1 day
                            sigma_p = float(np.sqrt(w_vec @ Σ.values @ w_vec))
                            if not np.isfinite(sigma_p) or sigma_p <= 0:
                                st.info("Could not compute portfolio volatility from returns.")
                            else:
                                scale = z * np.sqrt(int(horizon))  # VaR ≈ z * σ_p * sqrt(h)
                                # marginal vols: (Σ w)_i / σ_p
                                grad_sigma = (Σ.values @ w_vec) / sigma_p
                                # component VaR_i = w_i * scale * grad_sigma_i
                                cvar = w_vec * scale * grad_sigma
                                # keep positive magnitudes for shares
                                cvar = np.maximum(cvar, 0.0)

                                total_var_mag = float(abs(scale * sigma_p))
                                if not np.isfinite(total_var_mag) or total_var_mag <= 0:
                                    st.info("Total VaR magnitude is zero.")
                                else:
                                    df_c = pd.DataFrame({
                                        "symbol": symbols_aligned,
                                        "weight": w_vec,
                                        "contrib_pct": (cvar / total_var_mag)
                                    }).sort_values("contrib_pct", ascending=False)

                                    if np.isfinite(portfolio_value):
                                        df_c["contrib_$"] = df_c["contrib_pct"] * (total_var_mag * portfolio_value)

                                    # 4) Chart — pie only (bar toggle removed)
                                    if HAS_PX:
                                        import plotly.express as px
                                        fig = px.pie(df_c, names="symbol", values="contrib_pct", hole=0.35)
                                        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
                                        st.plotly_chart(fig, use_container_width=True)
                                    else:
                                        st.dataframe(
                                            df_c.set_index("symbol")[["contrib_pct", "weight"]]
                                            .assign(contrib_pct=lambda x: x["contrib_pct"].map(lambda v: f"{v:.2%}"),
                                                    weight=lambda x: x["weight"].map(lambda v: f"{v:.2%}")),
                                            use_container_width=True
                                        )

                                    # 5) Helpful footers
                                    st.caption(
                                        f"Parametric VaR @ {int(confidence * 100)}% over {horizon}d; "
                                        f"contributions sum to the portfolio VaR magnitude (for the selected assets)."
                                    )
                                    if np.isfinite(portfolio_value):
                                        st.caption(
                                            f"≈ Total VaR: ${(total_var_mag * portfolio_value):,.0f} "
                                            f"on a ${portfolio_value:,.0f} portfolio."
                                        )



# --- NEW: Correlation / Cointegration / Beta Tab ---
def render_corr_tab():
    st.subheader("Correlation, beta & cointegration")

    if not portfolio_id:
        st.info("Select or create a portfolio in the sidebar to see data.")
        return

    # Pull holdings to decide which assets to include
    ov = api_get("/portfolio/overview", params={"portfolio_id": int(portfolio_id)})
    holds = pd.DataFrame(ov.get("top_holdings", [])) if isinstance(ov, dict) else pd.DataFrame()

    if holds.empty:
        st.info("No holdings found.")
        return

    # Controls
    left, right = st.columns([1.2, 1])
    with left:
        lookback = st.select_slider("Lookback (days)", options=[30, 60, 90, 180, 365], value=180)
    with right:
        top_n = st.slider("Assets (top by weight)", min_value=2, max_value=min(15, len(holds)), value=min(8, len(holds)))

    # Select symbols by weight
    syms = (
        holds.sort_values("weight", ascending=False)
        .head(int(top_n))
        .symbol.dropna().astype(str).tolist()
    )
    if len(syms) < 2:
        st.info("Need at least two assets.")
        return

    # Returns matrix (daily)
    R = _load_asset_returns(syms, days=lookback + 10).tail(lookback).dropna(how="all")
    if R.shape[1] < 2 or R.shape[0] < 20:
        st.info("Not enough overlapping returns to compute statistics.")
        return

    # -------------------------
    # 1) Correlation heatmap
    # -------------------------
    st.markdown("**Correlation (daily returns)**")
    corr = R.corr()

    if HAS_PX:
        # custom scale: -1 → red, 0 → red, +1 → green
        colorscale_corr = [
            [0.00, "#b30000"],  # deep red at -1
            [0.50, "#b30000"],  # keep 0 as red (your request)
            [0.75, "#fdae61"],  # amber as it gets closer to +1
            [1.00, "#1a9850"],  # green at +1
        ]
        zmin, zmax = -1.0, 1.0
        fig = go.Figure(go.Heatmap(
            z=corr.values,
            x=corr.columns, y=corr.index,
            zmin=zmin, zmax=zmax,
            colorscale=colorscale_corr,
            colorbar=dict(title="ρ"),
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            hovertemplate="x=%{x}<br>y=%{y}<br>ρ=%{z:.2f}<extra></extra>",
        ))
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=480)
        st.plotly_chart(fig, use_container_width=True)
    else:
        # Fallback: styled dataframe (diverging around 0)
        st.dataframe(
            corr.style.format("{:.2f}").background_gradient(cmap="RdYlGn", vmin=-1, vmax=1),
            use_container_width=True
        )

    # --- Beta heatmap (asset i vs asset j) ---
    st.markdown("**Beta heatmap (asset i vs asset j)**")
    colsR = list(R.columns)
    B = pd.DataFrame(np.nan, index=colsR, columns=colsR)

    for j, bj in enumerate(colsR):
        y = R[bj].dropna()
        var_j = float(y.var(ddof=1))
        if not np.isfinite(var_j) or var_j <= 0:
            continue
        for i, ai in enumerate(colsR):
            x = R[ai].dropna()
            idx = x.index.intersection(y.index)
            if len(idx) < 20:
                continue
            xi, yj = x.loc[idx], y.loc[idx]
            cov_ij = float(np.cov(xi, yj, ddof=1)[0, 1])
            B.iat[i, j] = (cov_ij / var_j) if np.isfinite(cov_ij) else np.nan

    np.fill_diagonal(B.values, 1.0)
    B = B.astype(float)

    delta = np.nanmax(np.abs(B.values - 1.0))
    if not np.isfinite(delta) or delta == 0:
        delta = 0.25
    zmin, zmax = 1.0 - delta, 1.0 + delta

    if HAS_PX:
        figb = px.imshow(
            B,
            text_auto=".2f",
            color_continuous_scale="RdYlGn",
            zmin=zmin, zmax=zmax,
            color_continuous_midpoint=1.0,
            aspect="auto",
            labels=dict(color="β (i vs j)"),
        )
        figb.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=480)
        st.plotly_chart(figb, use_container_width=True)

    elif HAS_PLOTLY:
        figb = go.Figure(go.Heatmap(
            z=B.values, x=B.columns, y=B.index,
            zmin=zmin, zmax=zmax, zmid=1.0,
            colorscale="RdYlGn",
            colorbar=dict(title="β"),
            text=np.round(B.values, 2),
            texttemplate="%{text}",
            hovertemplate="i=%{y}<br>j=%{x}<br>β=%{z:.2f}<extra></extra>",
        ))
        figb.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=480)
        st.plotly_chart(figb, use_container_width=True)

    else:
        # No Plotly available → just a styled table. DO NOT touch `figb` here.
        st.dataframe(
            B.style.format("{:.2f}").background_gradient(cmap="RdYlGn", vmin=zmin, vmax=zmax),
            use_container_width=True
        )

    # 3) Cointegration matrix (Engle–Granger p-values)
    #    Run test on log-price indices reconstructed from returns.
    # -------------------------
    st.markdown("**Cointegration (Engle–Granger p-values)**")
    try:
        from statsmodels.tsa.stattools import coint

        cols = list(R.columns)
        pvals = pd.DataFrame(np.nan, index=cols, columns=cols)
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                a = R[cols[i]].dropna()
                b = R[cols[j]].dropna()
                idx = a.index.intersection(b.index)
                if len(idx) < 60:
                    continue
                x = np.log1p(a.loc[idx]).cumsum()
                y = np.log1p(b.loc[idx]).cumsum()
                try:
                    _, p, _ = coint(x.values, y.values, trend="c")
                    pvals.iat[i, j] = p
                    pvals.iat[j, i] = p
                except Exception:
                    continue
        np.fill_diagonal(pvals.values, 0.0)

        if HAS_PX:
            # reverse RdYlGn so low p (good evidence) is green
            figp = px.imshow(
                pvals,
                text_auto=".3f",
                zmin=0.0, zmax=0.20,
                color_continuous_scale="RdYlGn_r",
                aspect="auto",
                labels=dict(color="p")
            )
            figp.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=480)
            st.plotly_chart(figp, use_container_width=True)
        else:
            st.dataframe(
                pvals.style.format("{:.3f}").background_gradient(cmap="RdYlGn_r", vmin=0.0, vmax=0.20),
                use_container_width=True
            )

        pairs = []
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                p = pvals.iat[i, j]
                if pd.notna(p):
                    pairs.append({"pair": f"{cols[i]}–{cols[j]}", "p": float(p)})
        if pairs:
            df_pairs = pd.DataFrame(pairs).sort_values("p")
            st.caption("Potentially cointegrated (lower p is stronger evidence; common cutoffs: 0.10/0.05)")
            st.dataframe(df_pairs.head(10), use_container_width=True, hide_index=True)
    except Exception:
        st.info("`statsmodels` not available — `pip install statsmodels` to enable cointegration results.")
    # -------------------------
    # 4) Rolling correlation explorer
    # -------------------------
    st.markdown("**Rolling correlation explorer**")
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        a1 = st.selectbox("Asset A", syms, index=0, key="roll_a1")
    with c2:
        a2 = st.selectbox("Asset B", [s for s in syms if s != a1], index=0, key="roll_a2")
    with c3:
        win = st.slider("Window (days)", min_value=20, max_value=min(120, len(R)), value=60)

    rc = R[a1].rolling(win).corr(R[a2]).dropna()
    if rc.empty:
        st.caption("Not enough data for the selected window.")
    else:
        st.metric("Current ρ", f"{rc.iloc[-1]:.2f}")
        st.line_chart(rc)

    # -------------------------
    # 5) Concentration tiles
    # -------------------------
    st.markdown("**Concentration & diversification**")
    w_map = {row["symbol"]: float(row.get("weight", 0.0)) for _, row in holds.iterrows()}
    w = np.array([w_map.get(s, 0.0) for s in R.columns], dtype=float)
    if w.sum() > 0:
        w = w / w.sum()
        hhi = float((w ** 2).sum())
        enb = float(1.0 / hhi) if hhi > 0 else np.nan
        top3 = float(np.sort(w)[-3:].sum()) if len(w) >= 3 else float(w.max())
    else:
        hhi = enb = top3 = np.nan

    if corr.shape[1] >= 2:
        upper = np.triu_indices_from(corr.values, k=1)
        avg_abs_rho = float(np.nanmean(np.abs(corr.values[upper])))
    else:
        avg_abs_rho = np.nan

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("HHI (∑w²)", "—" if not np.isfinite(hhi) else f"{hhi:.2f}",
              help="The Herfindahl–Hirschman Index sums the squared portfolio weights; it ranges from 1/N (perfectly equal-weighted across N names) to 1 (one-name portfolio), so lower = more diversified. ")
    m2.metric("Effective # of bets", "—" if not np.isfinite(enb) else f"{enb:.1f}",
              help="Effective # of bets — Defined as 1 / ∑w², it’s the number of equal-weight positions that would give the same concentration as your current weights.")
    m3.metric("Top-3 weight", "—" if not np.isfinite(top3) else f"{top3:.1%}",
              help="The sum of your three largest position weights; a quick, intuitive concentration gauge.")
    m4.metric("Avg |ρ|", "—" if not np.isfinite(avg_abs_rho) else f"{avg_abs_rho:.2f}",
              help="The average absolute pairwise correlation of asset returns (ignores sign), capturing how tightly your holdings move together.")

    # Downloads
    cdl, bdl, pdl = st.columns(3)
    cdl.download_button("Download correlation CSV", corr.to_csv().encode("utf-8"), "correlation.csv", "text/csv")
    bdl.download_button("Download beta CSV", B.to_csv().encode("utf-8"), "beta_matrix.csv", "text/csv")
    try:
        pdl.download_button("Download cointegration p-values CSV", pvals.to_csv().encode("utf-8"), "cointegration_pvalues.csv", "text/csv")
    except Exception:
        pass


# Tabs
# -------------------------------------------------------------

tab_overview, tab_var, tab_corr = st.tabs(["Overview", "VaR", "Corr & Coint"])
with tab_overview: render_overview()
with tab_var:      render_var_tab()
with tab_corr:     render_corr_tab()
