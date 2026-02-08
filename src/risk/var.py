# src/risk/var.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal
import numpy as np
import pandas as pd
from math import sqrt

# --- Normal quantile with SciPy fallback (Acklam) ---
try:
    from scipy.stats import norm  # type: ignore
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False
    class _Norm:
        @staticmethod
        def ppf(p: float) -> float:
            a1=-3.969683028665376e+01; a2=2.209460984245205e+02
            a3=-2.759285104469687e+02; a4=1.383577518672690e+02
            a5=-3.066479806614716e+01; a6=2.506628277459239e+00
            b1=-5.447609879822406e+01; b2=1.615858368580409e+02
            b3=-1.556989798598866e+02; b4=6.680131188771972e+01
            b5=-1.328068155288572e+01
            c1=-7.784894002430293e-03; c2=-3.223964580411365e-01
            c3=-2.400758277161838e+00; c4=-2.549732539343734e+00
            c5= 4.374664141464968e+00; c6= 2.938163982698783e+00
            d1= 7.784695709041462e-03; d2= 3.224671290700398e-01
            d3= 2.445134137142996e+00; d4= 3.754408661907416e+00
            plow=0.02425; phigh=1-plow
            import numpy as _np
            if p < plow:
                q = _np.sqrt(-2*_np.log(p))
                return (((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q+c6)/((((d1*q+d2)*q+d3)*q+d4)*q+1)
            if phigh < p:
                q = _np.sqrt(-2*_np.log(1-p))
                return -(((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q+c6)/((((d1*q+d2)*q+d3)*q+d4)*q+1)
            q = p-0.5; r=q*q
            num=((((((a1*r+a2)*r+a3)*r+a4)*r+a5)*r+a6)*q)
            den=((((((b1*r+b2)*r+b3)*r+b4)*r+b5)*r+1))
            return num/den
    norm = _Norm()  # type: ignore

# Optional (for t and chi2 in backtests)
try:
    from scipy.stats import t as student_t, chi2
    _HAS_ST_CHI2 = True
except Exception:
    student_t = None  # type: ignore
    chi2 = None       # type: ignore
    _HAS_ST_CHI2 = False

# ---------------------------------------------------------------------
# Config & helpers
# ---------------------------------------------------------------------
@dataclass
class VaRConfig:
    confidence: float = 0.99
    window: int = 252
    horizon_days: int = 1
    gaussian: bool = True
    n_sims: int = 10000
    seed: Optional[int] = 42
    use_mean: bool = False  # parametric mean shift

def _clean_returns(returns: pd.Series) -> pd.Series:
    r = pd.Series(returns).astype(float).dropna()
    r.index = pd.to_datetime(r.index)
    return r

def _sqrt_time_scale(x: float | pd.Series, h: int) -> float | pd.Series:
    return x * np.sqrt(max(int(h), 1))

# ---------------------------------------------------------------------
# Historical VaR/ES
# ---------------------------------------------------------------------
def historical_var_series(returns: pd.Series, *, cfg: VaRConfig) -> pd.Series:
    r = _clean_returns(returns)
    # lower-tail quantile of returns; typically ≤ 0
    q = r.rolling(cfg.window, min_periods=cfg.window).quantile(1 - cfg.confidence)
    var_series = -_sqrt_time_scale(q, cfg.horizon_days)
    var_series.name = f"Hist VaR {int(cfg.confidence*100)}% ({cfg.window}d)"
    return var_series

def historical_cvar_point(returns: pd.Series, *, cfg: VaRConfig) -> float:
    r = _clean_returns(returns).iloc[-cfg.window:]
    if len(r) < cfg.window:
        return float('nan')
    cutoff = r.quantile(1 - cfg.confidence)
    es = -r[r <= cutoff].mean() * np.sqrt(cfg.horizon_days)
    return float(es)

# ---------------------------------------------------------------------
# Parametric (Normal) VaR
# ---------------------------------------------------------------------
def parametric_var_series(returns: pd.Series, *, cfg: VaRConfig) -> pd.Series:
    r = _clean_returns(returns)
    mu = r.rolling(cfg.window, min_periods=cfg.window).mean()
    sigma = r.rolling(cfg.window, min_periods=cfg.window).std(ddof=1)
    z_left = float(norm.ppf(1 - cfg.confidence))          # negative
    h = float(max(cfg.horizon_days, 1))
    mu_h = mu * h if cfg.use_mean else 0.0
    q = mu_h + z_left * sigma * np.sqrt(h)                # lower-tail
    var = (-q).clip(lower=0).astype(float)
    var.name = f"Param VaR {int(cfg.confidence*100)}% ({cfg.window}d)"
    return var

def monte_carlo_var_point(returns: pd.Series, *, cfg: VaRConfig) -> float:
    r = _clean_returns(returns).iloc[-cfg.window:]
    if len(r) < cfg.window:
        return float('nan')
    mu = float(r.mean()); sigma = float(r.std(ddof=1))
    sims = np.random.default_rng(cfg.seed).normal(mu, sigma, size=(cfg.n_sims, cfg.horizon_days))
    horizon_ret = (1 + sims).prod(axis=1) - 1
    q = np.quantile(horizon_ret, 1 - cfg.confidence)
    return float(-q)

def monte_carlo_cvar_point(returns: pd.Series, *, cfg: VaRConfig) -> float:
    r = _clean_returns(returns).iloc[-cfg.window:]
    if len(r) < cfg.window:
        return float('nan')
    mu = float(r.mean()); sigma = float(r.std(ddof=1))
    sims = np.random.default_rng(cfg.seed).normal(mu, sigma, size=(cfg.n_sims, cfg.horizon_days))
    horizon_ret = (1 + sims).prod(axis=1) - 1
    cutoff = np.quantile(horizon_ret, 1 - cfg.confidence)
    es = -horizon_ret[horizon_ret <= cutoff].mean()
    return float(es)

# ---------------------------------------------------------------------
# Student-t VaR/ES (with CF fallback if SciPy missing)
# ---------------------------------------------------------------------
def _cornish_fisher_quantile(z, skew, kurt):
    z2, z3 = z*z, z*z*z
    return (z
            + (1/6)*(z2 - 1)*skew
            + (1/24)*(z3 - 3*z)*kurt
            - (1/36)*(2*z3 - 5*z)*(skew**2))

def parametric_student_t_var_es(returns, confidence=0.99, horizon_days=1):
    r = np.asarray(returns, float)
    r = r[np.isfinite(r)]
    if len(r) < 20:
        return np.nan, np.nan
    mu = float(np.mean(r))
    sigma = float(np.std(r, ddof=1))
    if sigma <= 0:
        return 0.0, 0.0
    tail = 1.0 - confidence
    if _HAS_ST_CHI2 and student_t is not None:
        try:
            df, loc, scale = student_t.fit(r)
        except Exception:
            df, loc, scale = 6.0, mu, sigma
        q1 = student_t.ppf(tail, df=df)  # negative
        pdf_q1 = student_t.pdf(q1, df=df)
        var_1d_r = loc + scale * q1
        es_1d_r = loc - scale * ((df + q1**2) / (max(df - 1, 1e-9))) * (pdf_q1 / tail) if df > 1 else var_1d_r
        var_h_r = mu*horizon_days + (var_1d_r - mu) * sqrt(horizon_days)
        es_h_r  = mu*horizon_days + (es_1d_r  - mu) * sqrt(horizon_days)
    else:
        z = float(norm.ppf(tail))
        skew = float(np.mean(((r-mu)/sigma)**3))
        kurt = float(np.mean(((r-mu)/sigma)**4) - 3.0)
        z_cf = _cornish_fisher_quantile(z, skew, kurt)
        var_h_r = mu*horizon_days + sigma*sqrt(horizon_days)*z_cf
        phi = np.exp(-0.5*z_cf*z_cf)/np.sqrt(2*np.pi)
        es_h_r  = mu*horizon_days - sigma*sqrt(horizon_days) * (phi / tail)
    return max(0.0, -var_h_r), max(0.0, -es_h_r)

# ---------------------------------------------------------------------
# Filtered Historical Simulation (FHS)
# ---------------------------------------------------------------------
def ewma_vol(r, lam=0.94):
    r = np.asarray(r, float)
    s2 = np.zeros_like(r)
    s2[0] = np.var(r, ddof=1)
    for t in range(1, len(r)):
        s2[t] = lam * s2[t-1] + (1 - lam) * r[t-1]**2
    return np.sqrt(s2)

def fhs_var_es(returns, confidence=0.99, horizon_days=1, n_sims=20000, lam=0.94, seed=42):
    rng = np.random.default_rng(seed)
    r = np.asarray(returns, float)
    r = r[np.isfinite(r)]
    if len(r) < 50:
        return np.nan, np.nan
    mu = float(np.mean(r))
    sig = ewma_vol(r, lam=lam)
    sig_t = float(sig[-1])
    z = np.divide(r, sig, out=np.zeros_like(r), where=sig>0)
    draws = rng.choice(z, size=(n_sims, horizon_days), replace=True)
    sim = mu*horizon_days + sig_t*np.sqrt(horizon_days) * draws.sum(axis=1)
    tail = 1.0 - confidence
    var_r = np.quantile(sim, tail)
    es_r  = sim[sim <= var_r].mean() if np.any(sim <= var_r) else var_r
    return max(0.0, -var_r), max(0.0, -es_r)

# ---------------------------------------------------------------------
# Decompositions
# ---------------------------------------------------------------------
RiskMethod = Literal["parametric", "historical_es"]

def _to_vec(w: pd.Series, index: pd.Index) -> np.ndarray:
    return w.reindex(index).fillna(0.0).astype(float).values

def _to_cov(df: pd.DataFrame, shrinkage: bool = False, lam: float = 0.1) -> pd.DataFrame:
    S = df.cov()
    if not shrinkage:
        return S
    T = pd.DataFrame(np.diag(np.diag(S.values)), index=S.index, columns=S.columns)
    return (1 - lam) * S + lam * T

@dataclass
class ParametricRisk:
    sigma_total: float
    var_total: float
    es_total: float
    comp_sigma: pd.Series
    comp_var: pd.Series
    comp_es: pd.Series

def parametric_decomposition(returns: pd.DataFrame, weights: pd.Series, alpha: float = 0.99,
                             shrinkage: bool = True, lam: float = 0.1) -> ParametricRisk:
    Sigma = _to_cov(returns, shrinkage=shrinkage, lam=lam)
    cols = Sigma.index
    w = _to_vec(weights, cols)
    S = Sigma.values
    sig_p = sqrt(float(w @ S @ w))
    if sig_p <= 0 or not np.isfinite(sig_p):
        raise ValueError("Non-positive portfolio sigma.")
    grad = (S @ w) / sig_p
    comp_sigma = pd.Series(w * grad, index=cols, name="component_sigma")
    z = float(norm.ppf(alpha))
    c = float(np.exp(-0.5*z*z)/np.sqrt(2*np.pi) / (1 - alpha))  # φ(z)/(1-α)
    comp_var = z * comp_sigma
    comp_es  = c * comp_sigma
    return ParametricRisk(
        sigma_total=sig_p,
        var_total=z * sig_p,
        es_total=c * sig_p,
        comp_sigma=comp_sigma,
        comp_var=comp_var.rename("component_var"),
        comp_es=comp_es.rename("component_es"),
    )

@dataclass
class HistoricalESRisk:
    es_total: float
    comp_es: pd.Series
    tail_count: int
    var_level: float  # return quantile at (1-α)

def historical_es_decomposition(returns: pd.DataFrame, weights: pd.Series, alpha: float = 0.99) -> HistoricalESRisk:
    cols = returns.columns
    w = _to_vec(weights, cols)
    R = returns.astype(float).values
    r_p = R @ w
    q = np.quantile(r_p, 1 - alpha)
    tail_mask = r_p <= q
    if not np.any(tail_mask):
        raise ValueError("No tail observations at this alpha; increase window or lower alpha.")
    L_tail = -r_p[tail_mask]
    ES_total = float(L_tail.mean())
    comp = (-R[tail_mask] * w).mean(axis=0)
    comp_es = pd.Series(comp, index=cols, name="component_es")
    return HistoricalESRisk(es_total=ES_total, comp_es=comp_es, tail_count=int(tail_mask.sum()), var_level=float(q))

def percent_of_total(components: pd.Series, total: float, use_abs: bool = False) -> pd.Series:
    base = components.abs().sum() if use_abs else total
    return (components / base * 100).rename("% of total ({}base)".format("abs " if use_abs else "")) if base != 0 else components*0

def build_returns_from_prices(price_df: pd.DataFrame) -> pd.DataFrame:
    wide = price_df.pivot(index="date", columns="symbol", values="price").sort_index()
    return wide.pct_change().dropna(how="all")

# ---------------------------------------------------------------------
# Backtests (Kupiec POF & Christoffersen independence)
# ---------------------------------------------------------------------
@dataclass
class BacktestResult:
    alpha: float
    n: int
    exceptions: int
    pof_pvalue: float | None
    ind_pvalue: float | None
    exceptions_series: pd.Series

def rolling_historical_var(returns: pd.Series, *, window: int, alpha: float) -> pd.Series:
    r = _clean_returns(returns)
    return r.rolling(window).apply(lambda x: np.quantile(-x, alpha)).rename(f"Hist VaR {int(alpha*100)}%")

def backtest_exceptions(returns: pd.Series, var_series: pd.Series, *, alpha: float) -> BacktestResult:
    r, v = _clean_returns(returns).align(var_series, join="inner")
    exc = ((-r) > v).astype(int)  # loss > VaR
    n = int(exc.count()); x = int(exc.sum())
    p_pof = None; p_ind = None
    if _HAS_ST_CHI2 and chi2 is not None and n > 0:
        pi = x / n if n else 0.0
        # Kupiec POF LR
        num = ((1 - alpha) ** (n - x)) * (alpha ** x)
        den = ((1 - max(pi, 1e-12)) ** (n - x)) * (max(pi, 1e-12) ** x)
        lr_pof = -2.0 * np.log(num / den)
        p_pof = float(1 - chi2.cdf(lr_pof, df=1))
        # Christoffersen independence (1st-order Markov)
        n00 = int(((exc.shift(1) == 0) & (exc == 0)).sum())
        n01 = int(((exc.shift(1) == 0) & (exc == 1)).sum())
        n10 = int(((exc.shift(1) == 1) & (exc == 0)).sum())
        n11 = int(((exc.shift(1) == 1) & (exc == 1)).sum())
        def _sr(a,b): return a / b if b > 0 else 0.0
        pi01 = _sr(n01, n00 + n01); pi11 = _sr(n11, n10 + n11); pi1 = _sr(n01 + n11, n00 + n01 + n10 + n11)
        ll_indep = (n00 + n10) * np.log(max(1 - pi1, 1e-12)) + (n01 + n11) * np.log(max(pi1, 1e-12))
        ll_dep   = n00 * np.log(max(1 - pi01, 1e-12)) + n01 * np.log(max(pi01, 1e-12)) + \
                   n10 * np.log(max(1 - pi11, 1e-12)) + n11 * np.log(max(pi11, 1e-12))
        lr_ind = -2.0 * (ll_indep - ll_dep)
        p_ind = float(1 - chi2.cdf(lr_ind, df=1))
    return BacktestResult(alpha=float(alpha), n=n, exceptions=x, pof_pvalue=p_pof, ind_pvalue=p_ind, exceptions_series=exc)