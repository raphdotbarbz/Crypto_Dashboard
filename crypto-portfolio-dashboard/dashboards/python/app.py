import os
import json
from datetime import datetime, date as _date, time as _time, timezone  # at top if not already
from typing import Optional

import pandas as pd
import streamlit as st
import requests
import dataclasses

from pathlib import Path
import numpy as np
# plotly.express is optional; fall back to Matplotlib/Streamlit charts if missing
try:
    import plotly.express as px
    HAS_PX = True
except Exception:
    px = None
    HAS_PX = False

# Optional plotting libs
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

# VaR methods
from pathlib import Path
import sys
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
)

# -------------------------------------------------------------
# Config
# -------------------------------------------------------------
API_BASE_DEFAULT = "http://127.0.0.1:8000"
API_BASE = os.getenv("PORTFOLIO_API_BASE", API_BASE_DEFAULT)

st.set_page_config(page_title="Crypto Portfolio Dashboard", layout="wide")
st.title("📊 Crypto Portfolio Dashboard (MVP)")

# Chrome-like tabs styling (once, at top)
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] { min-width: 360px; max-width: 360px; }
    [data-testid="stSidebarContent"] { height: 100vh; overflow-y: auto; }

    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: white;
        border: 1px solid #e6e6e6;
        border-bottom: 3px solid transparent;
        border-radius: 12px 12px 0 0;
        padding: 8px 16px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }
    .stTabs [aria-selected="true"] {
        border-bottom: 3px solid #0e76fd !important;
        font-weight: 600 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
ARTIFACTS = DATA_DIR / "artifacts"

# -------------------------------------------------------------
# Small HTTP helpers
# -------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=15)
def api_get(path: str, params: Optional[dict] = None):
    url = f"{API_BASE}{path}"
    try:
        r = requests.get(url, params=params or {}, timeout=20)
        if r.status_code >= 400:
            return {"_error": f"GET {path} → {r.status_code}", "detail": _safe_detail(r)}
        return r.json()
    except Exception as e:
        return {"_error": f"GET {path} failed: {e}"}

@st.cache_data(show_spinner=False, ttl=0)
def api_post(path: str, params: Optional[dict] = None, json_body: Optional[dict | list] = None):
    url = f"{API_BASE}{path}"
    try:
        r = requests.post(url, params=params or {}, json=json_body, timeout=20)
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


def fmt_pct(x):
    if x is None:
        return "—"
    try:
        return f"{x*100:.2f}%"
    except Exception:
        return "—"

# -------------------------------------------------------------
# Sidebar controls (available to all tabs)
# -------------------------------------------------------------
st.sidebar.header("Settings")
API_BASE = st.sidebar.text_input("API base", value=API_BASE, help="FastAPI root, e.g. http://127.0.0.1:8000")
rf_annual = st.sidebar.number_input("Risk-free (annual)", min_value=0.0, max_value=1.0, value=0.05, step=0.005)
lookbacks = st.sidebar.text_input("Risk lookbacks (days)", value="30,180,365")

mode = st.sidebar.radio("Data source", ["CSV (legacy)", "DB portfolio"], horizontal=False)
portfolio_id: Optional[int] = None

if mode == "DB portfolio":
    st.sidebar.subheader("User & Portfolio")
    email = st.sidebar.text_input("Owner email", value="me@example.com")
    col_u1, col_u2 = st.sidebar.columns(2)
    if col_u1.button("Ensure user"):
        st.cache_data.clear()
        api_post("/users", params={"email": email})
    if col_u2.button("List users"):
        st.session_state["users_list"] = api_get("/users")
    if "users_list" in st.session_state:
        st.sidebar.json(st.session_state["users_list"])  # quick debug

    name = st.sidebar.text_input("New portfolio name", value="Demo")
    col_p1, col_p2 = st.sidebar.columns(2)
    if col_p1.button("Create portfolio"):
        # simple: use first user with this email
        u = api_post("/users", params={"email": email})
        if isinstance(u, dict) and "id" in u:
            api_post("/portfolios", params={"owner_id": u["id"], "name": name})
            st.cache_data.clear()
    if col_p2.button("List portfolios"):
        u = api_post("/users", params={"email": email})
        if isinstance(u, dict) and "id" in u:
            st.session_state["plist"] = api_get("/portfolios", params={"owner_id": u["id"]})
    plist = st.session_state.get("plist", [])
    if isinstance(plist, list) and plist:
        options = {f"#{p['id']}  {p['name']}": p["id"] for p in plist}
        label = st.sidebar.selectbox("Select portfolio", list(options.keys()))
        portfolio_id = options[label]

# Build query params for endpoints used in Overview
params = {"portfolio_id": portfolio_id} if (mode == "DB portfolio" and portfolio_id) else {}

def _returns_from_api_totals() -> pd.Series:
    """Build daily returns from /portfolio/totals when using DB portfolios."""
    tot = api_get("/portfolio/totals", params=params)
    if isinstance(tot, list) and tot:
        df = pd.DataFrame(tot)
        cols = {c.lower(): c for c in df.columns}
        date_col = cols.get("date") or next((c for c in df.columns if "date" in c.lower() or "time" in c.lower()), None)
        val_col  = cols.get("total_value") or cols.get("portfolio_value") or next((c for c in df.columns if "value" in c.lower()), None)
        if date_col and val_col:
            s_val = pd.Series(df[val_col].values, index=pd.to_datetime(df[date_col])).sort_index()
            return s_val.pct_change().dropna()
    return pd.Series([], dtype=float)


def _returns_for_var() -> pd.Series:
    # DB portfolio first → else artifacts
    if mode == "DB portfolio" and portfolio_id:
        r = _returns_from_api_totals()
        if len(r) > 0:
            return r
    return _load_value_curve()


# -------------------------------------------------------------
# Data helpers for VaR tab
# -------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _load_value_curve() -> pd.Series:
    """Load portfolio total value curve from artifacts and return daily returns."""
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
        s = (
            pd.Series(df[val_col].values, index=pd.to_datetime(df[date_col]))
            .sort_index()
            .asfreq("D")
            .interpolate(limit=3)
        )
    else:
        s = pd.Series(dtype=float)

    returns = s.pct_change().dropna()
    returns.name = "portfolio_return"
    return returns


from typing import Optional

def _maybe_upload_returns() -> Optional[pd.Series]:
    """
    Upload a CSV and return a returns series. Robust to common schemas.

    Accepted (case-insensitive):
      - date-like:  date | timestamp | time | datetime
      - value-like: value | total_value | portfolio_value | nav | price | close
      - return-like: return | ret | daily_return
    """
    up = st.file_uploader("Upload portfolio value or returns CSV (date,value|return)", type=["csv"])
    if not up:
        return None

    try:
        df = pd.read_csv(up)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        return None

    cols_map = {c.lower().strip(): c for c in df.columns}
    date_key = next((k for k in ("date","timestamp","time","datetime") if k in cols_map), None)
    ret_key  = next((k for k in ("return","ret","daily_return") if k in cols_map), None)
    val_key  = next((k for k in ("value","total_value","portfolio_value","nav","price","close") if k in cols_map), None)

    if not date_key:
        st.warning(f"Could not detect a date column. Found: {list(df.columns)}")
        return None

    date_col = cols_map[date_key]
    if ret_key:
        s = pd.Series(df[cols_map[ret_key]].values, index=pd.to_datetime(df[date_col])).sort_index()
    elif val_key:
        s_val = pd.Series(df[cols_map[val_key]].values, index=pd.to_datetime(df[date_col])).sort_index()
        s = s_val.pct_change().dropna()
    else:
        st.warning("No return/value column detected. Include one of: "
                   "return|ret|daily_return or value|total_value|portfolio_value|nav|price|close")
        return None

    s.name = "portfolio_return"
    return s


# -------------------------------------------------------------
# Overview Tab (existing main page)
# -------------------------------------------------------------

def render_overview():
    err_box = st.empty()

    # Fetch overview + risk tiles
    ov = api_get("/portfolio/overview", params=params)
    mt = api_get("/portfolio/metrics", params={**params, "lookbacks": lookbacks, "rf_annual": rf_annual})

    # Top tiles
    colA, colB, colC, colD = st.columns(4)
    if isinstance(ov, dict) and "total_value" in ov:
        colA.metric("Total value", f"${ov['total_value']:,.2f}")
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

    # Charts: totals curve & drawdown (requires /portfolio/totals)
    st.subheader("Value curve & drawdown")
    tot = api_get("/portfolio/totals", params=params)
    if isinstance(tot, list) and tot:
        df = pd.DataFrame(tot)
        try:
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date")
            st.line_chart(df.set_index("date")["total_value"], height=240)
            # drawdown
            peak = df["total_value"].cummax()
            dd = df["total_value"] / peak - 1.0
            dd_df = pd.DataFrame({"date": df["date"], "drawdown": dd})
            st.line_chart(dd_df.set_index("date")["drawdown"], height=160)
        except Exception as e:
            st.warning(f"Could not parse totals for charting: {e}")
    else:
        st.info("/portfolio/totals not available yet or no data. You can still use the tiles above.")

    # Top holdings
    st.subheader("Top holdings")
    if isinstance(ov, dict) and ov.get("top_holdings"):
        th = pd.DataFrame(ov["top_holdings"]).sort_values("weight", ascending=False)  # symbol, value, weight
        # Aggregate tail into OTHERS for a clean small chart
        max_slices = 6
        plot_df = th[["symbol", "weight"]].copy()
        plot_df["weight"] = plot_df["weight"].clip(lower=0)

        others_detail = pd.DataFrame(columns=["symbol", "weight"])  # full list that becomes OTHERS
        if len(plot_df) > max_slices:
            top = plot_df.head(max_slices - 1)
            others_detail = plot_df.iloc[max_slices - 1 :][["symbol", "weight"]].copy().sort_values("weight", ascending=False)
            others_w = others_detail["weight"].sum()
            plot_df = pd.concat([top, pd.DataFrame([{"symbol": "OTHERS", "weight": others_w}])], ignore_index=True)

        # Sidebar breakdown of OTHERS (if any)
        if not others_detail.empty:
            with st.sidebar.expander("OTHERS breakdown", expanded=False):
                tmp = others_detail.copy()
                tmp["weight %"] = (tmp["weight"] * 100).round(1)
                st.dataframe(tmp[["symbol", "weight %"]], hide_index=True, use_container_width=True)

        col1, col2 = st.columns([1.1, 1])

        if HAS_PLOTLY and not plot_df.empty:
            with col1:
                fig = go.Figure(
                    data=[
                        go.Pie(
                            labels=plot_df["symbol"],
                            values=plot_df["weight"],
                            hole=0.45,
                            sort=False,
                            textinfo="none",
                            hovertemplate="%{label}: %{percent:.1%}<extra></extra>",
                            marker=dict(line=dict(color="#0e1117", width=1)),
                        )
                    ]
                )
                fig.update_layout(
                    template=None,
                    showlegend=False,
                    paper_bgcolor="#0e1117",
                    plot_bgcolor="#0e1117",
                    font=dict(color="white"),
                    hoverlabel=dict(bgcolor="#222", font_color="white", bordercolor="#555", font_size=13),
                    margin=dict(l=10, r=10, t=10, b=10),
                    height=320,
                )
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        elif HAS_MPL and not plot_df.empty:
            with col1:
                fig, ax = plt.subplots(figsize=(4, 4))
                ax.set_facecolor("black")
                fig.patch.set_facecolor("black")
                ax.pie(
                    plot_df["weight"],
                    startangle=90,
                    counterclock=False,
                    wedgeprops={"width": 0.45},
                    labels=None,
                )
                ax.axis("equal")
                st.pyplot(fig, clear_figure=True, use_container_width=True)
        else:
            with col1:
                st.bar_chart(plot_df.set_index("symbol")["weight"], height=240)

        with col2:
            table = plot_df.copy()
            table["weight %"] = (table["weight"] * 100).round(1)
            st.dataframe(
                table[["symbol", "weight %"]],
                use_container_width=True,
                hide_index=True,
                height=320,
            )
    else:
        st.caption("No holdings to show.")

    def _to_iso_utc(d: _date, t: _time) -> str:
        dt = datetime.combine(d, t).replace(tzinfo=timezone.utc)
        return dt.isoformat().replace("+00:00", "Z")

    # -------------------------------------------------------------
    # Append transactions (DB or legacy)
    # -------------------------------------------------------------
    st.subheader("Append transaction")
    with st.form("append_tx"):
        today = datetime.utcnow().date()
        c1, c2, c3, c4, c5 = st.columns([1.2, 1, 1, 1, 1.2])

        tx_date = c1.date_input("Trade date (UTC)", value=today, max_value=today, key="tx_date")
        tx_time = c2.time_input("Time (UTC)", value=_time(0, 0), step=60, key="tx_time")
        sym_raw = c3.text_input("Symbol", value="ONDO", key="tx_symbol")
        qty = c4.number_input("Quantity (+/-)", value=1.0, key="tx_qty")
        price_s = c5.text_input("Price (optional)", value="", key="tx_price")

        submitted = st.form_submit_button("Append")

        if submitted:
            sym = sym_raw.strip().upper()
            if not sym:
                st.error("Symbol cannot be empty.")
                st.stop()

            ts_iso = _to_iso_utc(tx_date, tx_time)
            item = {"date": ts_iso, "symbol": sym, "quantity": qty}
            if price_s.strip():
                try:
                    item["price"] = float(price_s)
                except ValueError:
                    st.error("Price must be numeric if provided.")
                    st.stop()

            # Append via API (DB preferred)
            if mode == "DB portfolio" and portfolio_id:
                res = api_post("/transactions/db/append", params={"portfolio_id": int(portfolio_id)}, json_body=[item])
            else:
                res = api_post("/transactions/append", json_body=[item])

            if isinstance(res, dict) and res.get("_error"):
                st.error(res["_error"])
            else:
                st.success(f"Appended {sym} on {ts_iso}. Fetching/refreshing prices…")

                # Auto-backfill prices for this symbol so overview/metrics/totals won't 500
                try:
                    _backfill = api_post("/prices/backfill", json_body={"symbols": [sym], "days": "max"})
                    # optional: show brief result
                    if isinstance(_backfill, dict) and "results" in _backfill:
                        errs = [r for r in _backfill["results"] if r.get("error")]
                        if errs:
                            st.warning(f"Price backfill had issues: {errs}")
                except Exception as e:
                    st.warning(f"Backfill request failed: {e}")

                # Nudge caches to refresh
                st.cache_data.clear()
                _ = api_get("/portfolio/totals", params={"portfolio_id": int(portfolio_id)} if (
                            mode == "DB portfolio" and portfolio_id) else {})
                _ = api_get("/portfolio/overview", params={"portfolio_id": int(portfolio_id)} if (
                            mode == "DB portfolio" and portfolio_id) else {})

    # -------------------------------------------------------------
    # Recent transactions
    # -------------------------------------------------------------
    st.subheader("Recent transactions")
    if mode == "DB portfolio" and portfolio_id:
        tx = api_get("/transactions/db", params={"portfolio_id": int(portfolio_id), "limit": 100})
    else:
        tx = api_get("/transactions", params={"limit": 100})

    if isinstance(tx, list) and tx:
        df_tx = pd.DataFrame(tx)
        # make sure it's sorted by date desc if the API doesn't already
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

    # Source priority: Uploaded CSV → DB totals → artifacts
    src_label: str | None = None
    base_returns: pd.Series | None = None

    with st.expander("Data source", expanded=False):
        colA, colB = st.columns([2, 1])

        with colB:
            st.caption("Optional override via CSV upload (not needed for DB mode):")
            up = None
            try:
                up = _maybe_upload_returns()  # may be None or a Series
            except Exception as e:
                st.warning(f"Upload parse failed: {e}")
            if isinstance(up, pd.Series) and len(up) > 0:
                base_returns = up
                src_label = f"uploaded CSV ({len(up):,} pts)"
                st.success("Using uploaded returns.")

        with colA:
            if base_returns is None:
                r = _returns_for_var()
                if len(r) > 0:
                    base_returns = r
                    src_label = (
                        f"/portfolio/totals API ({len(r):,} pts)"
                        if (mode == "DB portfolio" and portfolio_id)
                        else f"artifacts ({len(r):,} pts)"
                    )

        st.write(
            f"Loaded {0 if base_returns is None else len(base_returns):,} daily returns from "
            f"{src_label or 'artifacts'}."
        )

    if base_returns is None or len(base_returns) < 20:
        st.warning("Need at least ~20 daily points to compute VaR. Back-date a few buys in the portfolio builder below.")
        return

    # Controls
    left, right = st.columns([2, 1])
    with left:
        horizon = st.select_slider("Horizon (days)", options=[1, 10, 30], value=1)
        window = st.select_slider("Lookback window (days)", options=[20, 30, 60, 90, 180, 252], value=30)
        confidence = st.slider("Confidence level", 0.90, 0.999, 0.99, step=0.001)
        methods = st.multiselect(
            "Methods",
            ["Historical", "Parametric (Normal)", "Monte Carlo"],
            default=["Historical", "Parametric (Normal)"],
        )
    with right:
        sims = st.number_input("Monte Carlo simulations", value=10000, min_value=1000, max_value=200000, step=1000)
        seed = st.number_input("Random seed", value=42, step=1)

    # Auto-shrink window to available history (min 20)
    eff_window = max(20, min(window, len(base_returns)))
    if eff_window < window:
        st.info(f"Using effective lookback {eff_window}d (limited by available data).")
    cfg = VaRConfig(confidence=confidence, window=eff_window, horizon_days=horizon, n_sims=int(sims), seed=int(seed))

    # Compute
    cols = st.columns(3)
    latest_vals: dict[str, float] = {}

    hist_series = None
    para_series = None

    if "Historical" in methods:
        hist_series = historical_var_series(base_returns, cfg=cfg)
        latest_vals["Historical"] = float(hist_series.dropna().iloc[-1]) if hist_series.notna().any() else np.nan

    if "Parametric (Normal)" in methods:
        para_series = parametric_var_series(base_returns, cfg=cfg)
        latest_vals["Parametric (Normal)"] = float(para_series.dropna().iloc[-1]) if para_series.notna().any() else np.nan

    if "Monte Carlo" in methods:
        latest_vals["Monte Carlo"] = monte_carlo_var_point(base_returns, cfg=cfg)

    # Tiles
    for i, (name, val) in enumerate(latest_vals.items()):
        with cols[i]:
            st.metric(
                label=f"{name} VaR ({int(confidence*100)}% | {horizon}d)",
                value=f"{val:.2%}" if pd.notna(val) else "—",
            )

    # Rolling chart (skip if nothing to plot)
    lines = []
    if hist_series is not None and hist_series.notna().any():
        lines.append(hist_series.rename("Historical"))
    if para_series is not None and para_series.notna().any():
        lines.append(para_series.rename("Parametric (Normal)"))

    if lines:
        df_plot = pd.concat(lines, axis=1).dropna(how="all")
        if df_plot.empty:
            st.info("No non-NaN points for the selected lookback; try a shorter window or add a bit more history.")
        else:
            if HAS_PX:
                fig = px.line(
                    df_plot,
                    title=f"Rolling VaR ({eff_window}d lookback, horizon={horizon}d, conf={confidence:.3f})",
                )
                fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), legend_title_text="Method")
                st.plotly_chart(fig, use_container_width=True)
            elif HAS_MPL:
                import matplotlib.pyplot as _plt
                _fig, _ax = _plt.subplots()
                df_plot.plot(ax=_ax)
                _ax.set_title(f"Rolling VaR ({eff_window}d lookback, horizon={horizon}d, conf={confidence:.3f})")
                _ax.set_xlabel("Date")
                _ax.set_ylabel("VaR")
                st.pyplot(_fig, clear_figure=True, use_container_width=True)
            else:
                st.line_chart(df_plot)

    with st.expander("Show Expected Shortfall (CVaR)", expanded=False):
        c1, c2 = st.columns(2)
        if "Historical" in methods:
            es_hist = historical_cvar_point(base_returns, cfg=cfg)
            c1.metric(label=f"Historical CVaR ({int(confidence*100)}% | {horizon}d)", value=f"{es_hist:.2%}")
        if "Monte Carlo" in methods:
            es_mc = monte_carlo_cvar_point(base_returns, cfg=cfg)
            c2.metric(label=f"Monte Carlo CVaR ({int(confidence*100)}% | {horizon}d)", value=f"{es_mc:.2%}")
        # --- Dev: which parametric_var_series is actually running? ---
    with st.sidebar.expander("Dev • Loaded VaR function", expanded=False):
        import importlib, inspect
        from risk import var as _v
        _v = importlib.reload(_v)  # force-reload edited code
        st.sidebar.caption("parametric_var_series loaded from:")
        st.sidebar.code(_v.parametric_var_series.__code__.co_filename, language="text")
        st.sidebar.caption("Source:")
        st.sidebar.code(inspect.getsource(_v.parametric_var_series), language="python")
    with st.sidebar.expander("Dev • Loaded VaR function", expanded=False):
        import inspect, importlib
        from risk import var as _v
        _v = importlib.reload(_v)  # ensure latest code after edits
        st.caption("parametric_var_series loaded from:")
        st.code(_v.parametric_var_series.__code__.co_filename, language="text")
        st.caption("Source:")
        st.code(inspect.getsource(_v.parametric_var_series), language="python")


# -------------------------------------------------------------
# Top-level tab bar (Chrome-like)
# -------------------------------------------------------------

tab_overview, tab_var = st.tabs(["Overview", "VaR"])

with tab_overview:
    render_overview()

with tab_var:
    render_var_tab()


