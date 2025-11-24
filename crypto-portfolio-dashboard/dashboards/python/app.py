import os
import json
from datetime import datetime
from typing import Optional

import pandas as pd
import streamlit as st
import requests

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

# -------------------------------------------------------------
# Config
# -------------------------------------------------------------
API_BASE_DEFAULT = "http://127.0.0.1:8000"
API_BASE = os.getenv("PORTFOLIO_API_BASE", API_BASE_DEFAULT)

st.set_page_config(page_title="Crypto Portfolio Dashboard", layout="wide")
st.title("📊 Crypto Portfolio Dashboard (MVP)")

# Make sidebar fill viewport height and widen slightly
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] { min-width: 360px; max-width: 360px; }
    [data-testid="stSidebarContent"] { height: 100vh; overflow-y: auto; }
    </style>
    """,
    unsafe_allow_html=True,
)

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

# -------------------------------------------------------------
# Sidebar controls
# -------------------------------------------------------------
st.sidebar.header("Settings")
API_BASE = st.sidebar.text_input("API base", value=API_BASE, help="FastAPI root, e.g. http://127.0.0.1:8000")
rf_annual = st.sidebar.number_input("Risk-free (annual)", min_value=0.0, max_value=1.0, value=0.05, step=0.005)
lookbacks = st.sidebar.text_input("Risk lookbacks (days)", value="30,180,365")

mode = st.sidebar.radio("Data source", ["CSV (legacy)", "DB portfolio"], horizontal=False)
portfolio_id = None

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

# Build query params
params = {"portfolio_id": portfolio_id} if (mode == "DB portfolio" and portfolio_id) else {}

# -------------------------------------------------------------
# Fetch overview + risk tiles
# -------------------------------------------------------------
ov = api_get("/portfolio/overview", params=params)
mt = api_get("/portfolio/metrics", params={**params, "lookbacks": lookbacks, "rf_annual": rf_annual})

err_box = st.empty()

def fmt_pct(x):
    if x is None:
        return "—"
    try:
        return f"{x*100:.2f}%"
    except Exception:
        return "—"

# -------------------------------------------------------------
# Top tiles
# -------------------------------------------------------------
colA, colB, colC, colD = st.columns(4)
if isinstance(ov, dict) and "total_value" in ov:
    colA.metric("Total value", f"${ov['total_value']:,.2f}")
    colB.metric("1d", fmt_pct(ov.get("ret_1d")))
    colC.metric("7d", fmt_pct(ov.get("ret_7d")))
    colD.metric("30d", fmt_pct(ov.get("ret_30d")))
else:
    err_box.error(ov.get("_error", "Failed to load /portfolio/overview"))

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

# -------------------------------------------------------------
# Charts: totals curve & drawdown (requires /portfolio/totals)
# -------------------------------------------------------------
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

# -------------------------------------------------------------
# Top holdings
# -------------------------------------------------------------
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

# -------------------------------------------------------------
# Append transactions (CSV or DB)
# -------------------------------------------------------------
st.subheader("Append transaction")
with st.form("append_tx"):
    dflt_dt = datetime.utcnow().isoformat() + "Z"
    c1, c2, c3, c4 = st.columns([2,1,1,1])
    ts = c1.text_input("Date (ISO)", value=dflt_dt)
    sym = c2.text_input("Symbol", value="ONDO").upper()
    qty = c3.number_input("Quantity (+/-)", value=1.0)
    price_txt = c4.text_input("Price (optional)", value="")
    submitted = st.form_submit_button("Append")

    if submitted:
        item = {"date": ts, "symbol": sym, "quantity": qty}
        if price_txt.strip():
            try:
                item["price"] = float(price_txt)
            except ValueError:
                st.error("Price must be numeric if provided.")
        if mode == "DB portfolio" and portfolio_id:
            res = api_post("/transactions/db/append", params={"portfolio_id": int(portfolio_id)}, json_body=[item])
        else:
            res = api_post("/transactions/append", json_body=[item])
        st.cache_data.clear()
        if isinstance(res, dict) and res.get("_error"):
            st.error(res["_error"])  # show status
        else:
            st.success("Appended. Refreshing tiles…")

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
    st.dataframe(df_tx)
elif isinstance(tx, dict) and tx.get("_error"):
    st.error(tx["_error"])  # status code insight
else:
    st.caption("No transactions found.")
