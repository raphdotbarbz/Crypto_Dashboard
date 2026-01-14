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

st.set_page_config(page_title="Crypto Portfolio Dashboard", layout="wide")
st.title("📊 Crypto Portfolio Dashboard (DB-only)")

# Simple styling
# --- BIGGER GLOBAL FONTS (≈2×) ---
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
    st.experimental_rerun()

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
            st.experimental_rerun()

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

    # Top holdings
    st.subheader("Top holdings")
    if isinstance(ov, dict) and ov.get("top_holdings"):
        th = pd.DataFrame(ov["top_holdings"]).sort_values("weight", ascending=False)
        max_slices = 6
        plot_df = th[["symbol", "weight"]].copy()
        plot_df["weight"] = plot_df["weight"].clip(lower=0)

        others_detail = pd.DataFrame(columns=["symbol", "weight"])  # tail that becomes OTHERS
        if len(plot_df) > max_slices:
            top = plot_df.head(max_slices - 1)
            others_detail = plot_df.iloc[max_slices - 1 :][["symbol", "weight"]].copy().sort_values("weight", ascending=False)
            others_w = others_detail["weight"].sum()
            plot_df = pd.concat([top, pd.DataFrame([{"symbol": "OTHERS", "weight": others_w}])], ignore_index=True)

        if not others_detail.empty:
            with st.sidebar.expander("OTHERS breakdown", expanded=False):
                tmp = others_detail.copy(); tmp["weight %"] = (tmp["weight"] * 100).round(1)
                st.dataframe(tmp[["symbol", "weight %"]], hide_index=True, use_container_width=True)

        col1, col2 = st.columns([1.1, 1])
        if HAS_PLOTLY and not plot_df.empty:
            with col1:
                fig = go.Figure([
                    go.Pie(
                        labels=plot_df["symbol"], values=plot_df["weight"], hole=0.45, sort=False,
                        textinfo="none", hovertemplate="%{label}: %{percent:.1%}<extra></extra>",
                        marker=dict(line=dict(color="#0e1117", width=1)),
                    )
                ])
                fig.update_layout(template=None, showlegend=False, paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                                  font=dict(color="white"), hoverlabel=dict(bgcolor="#222", font_color="white", bordercolor="#555", font_size=13),
                                  margin=dict(l=10, r=10, t=10, b=10), height=320)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        elif HAS_MPL and not plot_df.empty:
            with col1:
                fig, ax = plt.subplots(figsize=(4, 4))
                ax.set_facecolor("black"); fig.patch.set_facecolor("black")
                ax.pie(plot_df["weight"], startangle=90, counterclock=False, wedgeprops={"width": 0.45}, labels=None)
                ax.axis("equal")
                st.pyplot(fig, clear_figure=True, use_container_width=True)
        else:
            with col1:
                st.bar_chart(plot_df.set_index("symbol")["weight"], height=240)

        with col2:
            table = plot_df.copy(); table["weight %"] = (table["weight"] * 100).round(1)
            st.dataframe(table[["symbol", "weight %"]], use_container_width=True, hide_index=True, height=320)
    else:
        st.caption("No holdings to show.")

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
                st.experimental_rerun()

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
    use_student_t = st.checkbox("Use Student-t for Parametric VaR", value=True,
                                help="Leptokurtic/heavy-tail friendly. Falls back to Normal if SciPy fit fails.")

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
            # keep your original three as default; you can add the new ones when you want
            default=["Historical", "Parametric (Normal)", "Monte Carlo"],
        )

    with right:
        sims = st.number_input("Monte Carlo simulations", value=10000, min_value=1000, max_value=200000, step=1000)
        seed = st.number_input("Random seed", value=42, step=1)

    # 1-sentence hover/tooltips
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
        """Return the latest VaR (positive loss fraction) for a window & method."""
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
                v, _es = parametric_student_t_var_es(sub, confidence=confidence, horizon_days=horizon)
                v = float(v) if np.isfinite(v) else None

            elif method == "Filtered Historical (EWMA)":
                v, _es = fhs_var_es(sub, confidence=confidence, horizon_days=horizon, n_sims=int(sims), seed=int(seed))
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

    def student_t_var_series(returns: pd.Series, cfg: VaRConfig) -> pd.Series:
        """Rolling Student-t VaR series (positive loss fraction)."""
        r = pd.Series(returns).astype(float).replace([np.inf, -np.inf], np.nan).dropna()
        if r.empty:
            return pd.Series(dtype=float, index=returns.index)

        vals, idx = [], []
        roll = r.rolling(cfg.window, min_periods=min(20, cfg.window))
        for end in range(cfg.window - 1, len(r)):
            window = r.iloc[end - cfg.window + 1: end + 1]
            v, _ = parametric_student_t_var_es(window.values,
                                               confidence=cfg.confidence,
                                               horizon_days=cfg.horizon_days)
            vals.append(float(v) if v is not None else np.nan)
            idx.append(r.index[end])
        s = pd.Series(vals, index=idx)
        return s.reindex(returns.index)

    # --- chart: only last VaR @ lookbacks 30/180/365 ---
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

        # nicer hover with method explainer
        hover_text = [
            f"<b>{r['method']}</b><br>"
            f"Lookback: {int(r['lookback'])}d<br>"
            f"VaR: {r['var_pct']*100:.2f}%<br>"
            f"{explain.get(r['method'],'')}"
            for _, r in df_var.iterrows()
        ]
        fig.update_traces(hovertemplate="%{customdata}", customdata=np.array(hover_text).reshape(-1, 1))
        fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), legend_title_text="Method")
        st.plotly_chart(fig, use_container_width=True)
    else:
        pivot = (
            df_var.pivot(index="lookback", columns="method", values="var_pct")
            .sort_index()
        )
        st.dataframe(pivot.applymap(lambda v: f"{v:.2%}"), use_container_width=True)

        # chart still needs numeric values (decimals), so use the unformatted `pivot`
        st.bar_chart(pivot)

    # --- tiles: % and $ loss (with hover help) ---
    st.subheader("VaR tiles (percent and amount)")
    cols = st.columns(min(3, len(methods)))
    by_method = df_var.groupby("method")
    for i, (m, g) in enumerate(by_method):
        with cols[i % len(cols)]:
            # show the most conservative (max) across 30/180/365
            v = float(g["var_pct"].max())
            amt = float(-(v * portfolio_value)) if np.isfinite(portfolio_value) else None
            st.metric(
                label=f"{m} VaR @ {int(confidence*100)}% (h={horizon}d)",
                value=(f"{v*100:.2f}%" if pd.notna(v) else "—"),
                help=explain.get(m, ""),
            )
            if amt is not None:
                st.caption(f"≈ ${amt:,.0f} loss on a ${portfolio_value:,.0f} portfolio")

    # --- recent returns: time-series with μ/±σ band + max Parametric VaR shading ---


    # --- recent returns: time-series with μ/±σ band + rolling/static VaR overlays ---
    # --- returns + rolling VaR lines + 30d ±σ (no shading) ---
    st.subheader("Recent returns + VaR lines (+/−1σ)")

    # 1) Data
    recent = base_returns.tail(365).astype(float)
    if len(recent) == 0:
        st.warning("No recent returns to plot.")
        # keep going so you see why it’s empty
    mu = recent.mean()
    sigma30 = recent.rolling(30, min_periods=20).std(ddof=1)

    # rolling window for VaR series (robust defaults)
    roll_window = max(20, min(60, len(recent)))
    cfg_roll = VaRConfig(
        confidence=confidence,
        window=roll_window,
        horizon_days=horizon,
        n_sims=int(sims),
        seed=int(seed),
    )

    # 2) Rolling VaR series (don’t swallow errors)
    pvar = None;
    hvar = None;
    p_err = None;
    h_err = None
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

    # 3) Build a single frame to plot; ensure tz-naive index for Plotly
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

    # 4) Plot
    if HAS_PX:
        import plotly.express as px

        melt = (
            df_plot.reset_index()
            .rename(columns={"index": "date"})
            .melt(id_vars="date", var_name="series", value_name="value")
            .dropna(subset=["value"])
        )
        if melt.empty:
            st.line_chart(df_plot["Return"])  # at least show something
        else:
            fig = px.line(
                melt, x="date", y="value", color="series",
                color_discrete_map={
                    "Return": "#4B5563",
                    "Parametric VaR (rolling)": VAR_COLORS.get("Parametric (Normal)", "#E11900"),
                    "Historical VaR (rolling)": VAR_COLORS.get("Historical", "#DAA520"),
                    "+1σ (30d)": "#6E7B8B",
                    "-1σ (30d)": "#6E7B8B",
                },
                labels={"value": "Daily return", "series": "", "date": ""},
            )
            # line styles
            for tr in fig.data:
                if tr.name == "Return":
                    tr.line.update(width=2)
                elif tr.name in ("+1σ (30d)",):
                    tr.line.update(width=1.5)
                elif tr.name in ("-1σ (30d)",):
                    tr.line.update(width=1.5, dash="dash")
                elif "VaR" in tr.name:
                    tr.line.update(width=2)

            fig.update_yaxes(tickformat=".1%")
            fig.update_layout(
                margin=dict(l=10, r=10, t=10, b=10),
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    else:
        st.line_chart(df_plot)

    # 5) If VaR calc failed, surface it (so you see why a line is missing)
    if p_err: st.caption(f"Parametric VaR error: {p_err}")
    if h_err: st.caption(f"Historical VaR error: {h_err}")

    # --- $ comparison vs portfolio size ---
    # --- $ comparison vs Portfolio size ---
    st.subheader("Risk in $ vs Portfolio size")

    # Make sure σ is defined here (independent of any plotting branch)
    # recent is already defined above as: recent = base_returns.tail(max(windows)).astype(float)
    sigma_recent = float(np.nan_to_num(recent.std(ddof=1), nan=0.0))  # daily std dev of recent returns

    summary_rows = []
    if np.isfinite(portfolio_value):
        summary_rows.append({"metric": "Portfolio size ($)", "amount": float(portfolio_value)})

    # Most conservative VaR across 30/180/365 per method
    for m in methods:
        if "method" in df_var.columns:
            g = df_var.loc[df_var["method"] == m]
            if not g.empty and np.isfinite(portfolio_value):
                v = float(g["var_pct"].max())  # max loss % among 30/180/365
                summary_rows.append({
                    "metric": f"{m} VaR $ (max 30/180/365)",
                    "amount": float(-(v * portfolio_value))
                })

    # Daily 1σ in $
    if np.isfinite(portfolio_value) and sigma_recent > 0:
        summary_rows.append({
            "metric": "Std dev $ (1σ/day)",
            "amount": float(-(sigma_recent * portfolio_value))
        })

    df_amt = pd.DataFrame(summary_rows)
    if not df_amt.empty and HAS_PX:
        fig_amt = px.bar(
            df_amt, x="amount", y="metric", orientation="h",
            labels={"amount": "Amount ($)", "metric": ""},
            title=None
        )
        fig_amt.update_layout(margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_amt, use_container_width=True)
    elif not df_amt.empty:
        st.dataframe(df_amt, use_container_width=True)
    else:
        st.caption("Add a portfolio or enable methods to see $ comparisons.")

    # --- forecast cone (parametric) for next 30 days ---
    st.subheader("Forward risk cone (next 30 days)")
    try:
        cone_days = 30
        # estimate μ, σ from the longest window we used
        est = base_returns.tail(max(windows))
        mu_est = float(est.mean())
        sigma_est = float(est.std(ddof=1))
        alpha = 1 - confidence
        # z from scipy if available, otherwise common-level fallback
        try:
            from scipy.stats import norm
            z = float(norm.ppf(alpha))  # negative
        except Exception:
            z_table = {0.10: -1.2816, 0.05: -1.6449, 0.025: -1.9600, 0.01: -2.3263, 0.005: -2.5758}
            z = z_table.get(round(alpha, 3), -2.3263)
        dates = pd.date_range(datetime.now(timezone.utc).date(), periods=cone_days + 1, freq="D")
        t = np.arange(cone_days + 1)
        # cumulative returns (parametric)
        med = mu_est * t
        low = (mu_est * t) + z * sigma_est * np.sqrt(t + 1e-9)
        df_cone = pd.DataFrame({"date": dates, "median": med, "lower": low})
        if HAS_PLOTLY:
            import plotly.graph_objects as go
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

    # compute ES for selected methods using the last 30 days (or any window you prefer)
    es_window = 30
    tail = base_returns.tail(es_window)

    if "Historical" in methods:
        try:
            cfg_es = VaRConfig(confidence=confidence, window=len(tail), horizon_days=horizon, n_sims=int(sims),
                               seed=int(seed))
            es_hist = historical_cvar_point(tail, cfg=cfg_es)
            es_rows.append(("Historical ES", es_hist))
        except Exception:
            pass

    if "Monte Carlo" in methods:
        try:
            cfg_es = VaRConfig(confidence=confidence, window=len(tail), horizon_days=horizon, n_sims=int(sims),
                               seed=int(seed))
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
            st.metric(label=f"{label} ({int(confidence * 100)}% | {horizon}d)",
                      value=("—" if val is None or not np.isfinite(val) else f"{abs(val):.2%}"), help=explain["ES"])


# -------------------------------------------------------------
# Tabs
# -------------------------------------------------------------

tab_overview, tab_var = st.tabs(["Overview", "VaR"])

with tab_overview:
    render_overview()

with tab_var:
    render_var_tab()
