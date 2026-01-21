from __future__ import annotations
from typing import Optional
import pandas as pd
from dataclasses import dataclass
from typing import Literal, Tuple, Dict
from math import sqrt
from scipy.stats import norm  # if you prefer no dependency, hardcode z & pdf

# Optional SciPy for Normal quantile; fallback to Acklam approximation
try:
    from scipy.stats import norm  # type: ignore
except Exception:  # pragma: no cover - lightweight fallback when SciPy isn't installed
    class _Norm:
        @staticmethod
        def ppf(p: float) -> float:
            # Peter J. Acklam's inverse normal CDF approximation
            a1 = -3.969683028665376e+01
            a2 = 2.209460984245205e+02
            a3 = -2.759285104469687e+02
            a4 = 1.383577518672690e+02
            a5 = -3.066479806614716e+01
            a6 = 2.506628277459239e+00
            b1 = -5.447609879822406e+01
            b2 = 1.615858368580409e+02
            b3 = -1.556989798598866e+02
            b4 = 6.680131188771972e+01
            b5 = -1.328068155288572e+01
            c1 = -7.784894002430293e-03
            c2 = -3.223964580411365e-01
            c3 = -2.400758277161838e+00
            c4 = -2.549732539343734e+00
            c5 = 4.374664141464968e+00
            c6 = 2.938163982698783e+00
            d1 = 7.784695709041462e-03
            d2 = 3.224671290700398e-01
            d3 = 2.445134137142996e+00
            d4 = 3.754408661907416e+00
            plow = 0.02425
            phigh = 1 - plow
            if p < plow:
                q = np.sqrt(-2*np.log(p))
                return (((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q+c6) / ((((d1*q+d2)*q+d3)*q+d4)*q+1)
            if phigh < p:
                q = np.sqrt(-2*np.log(1-p))
                return -(((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q+c6) / ((((d1*q+d2)*q+d3)*q+d4)*q+1)
            q = p - 0.5
            r = q*q
            num = (((((a1*r+a2)*r+a3)*r+a4)*r+a5)*r+a6)*q
            den = (((((b1*r+b2)*r+b3)*r+b4)*r+b5)*r+1)
            return num/den
    norm = _Norm()  # type: ignore


@dataclass
class VaRConfig:
    """Configuration for VaR calculations.

    Attributes
    ----------
    confidence : float
        Confidence level (e.g., 0.95, 0.99, 0.995).
    window : int
        Rolling lookback window in days.
    horizon_days : int
        VaR horizon in days. We use sqrt-time scaling.
    gaussian : bool
        Placeholder for future non-Gaussian options.
    n_sims : int
        Number of Monte Carlo simulations.
    seed : Optional[int]
        RNG seed for reproducibility.
    """

    confidence: float = 0.99
    window: int = 252
    horizon_days: int = 1
    gaussian: bool = True
    n_sims: int = 10000
    seed: Optional[int] = 42


# --------------------------
# Helpers
# --------------------------

def _clean_returns(returns: pd.Series) -> pd.Series:
    r = pd.Series(returns).astype(float).dropna()
    r.index = pd.to_datetime(r.index)
    r = r.asfreq('D')  # align to daily; gaps -> NaN
    return r


def _sqrt_time_scale(x: float | pd.Series, h: int) -> float | pd.Series:
    return x * np.sqrt(max(h, 1))


# --------------------------
# Historical VaR
# --------------------------

def historical_var_series(returns: pd.Series, *, cfg: VaRConfig) -> pd.Series:
    """Rolling Historical VaR (positive number, i.e., loss).

    VaR_t = - Quantile_{(1 - confidence)}(returns_{t-window+1:t}) * sqrt(horizon_days)
    """
    r = _clean_returns(returns)
    q = r.rolling(cfg.window, min_periods=cfg.window).quantile(1 - cfg.confidence)
    var_series = -_sqrt_time_scale(q, cfg.horizon_days)
    var_series.name = f"Hist VaR {int(cfg.confidence*100)}% ({cfg.window}d)"
    return var_series


def historical_cvar_point(returns: pd.Series, *, cfg: VaRConfig) -> float:
    """Latest Conditional VaR (Expected Shortfall) from the last ``window`` days."""
    r = _clean_returns(returns)
    tail = r.dropna().iloc[-cfg.window:]
    if len(tail) < cfg.window:
        return float('nan')
    cutoff = tail.quantile(1 - cfg.confidence)
    es = -tail[tail <= cutoff].mean() * np.sqrt(cfg.horizon_days)
    return float(es)


# --------------------------
# Parametric (Normal) VaR
# --------------------------

def parametric_var_series(returns: pd.Series, *, cfg: VaRConfig) -> pd.Series:
    r = _clean_returns(returns)
    mu = r.rolling(cfg.window, min_periods=cfg.window).mean()
    sigma = r.rolling(cfg.window, min_periods=cfg.window).std(ddof=1)

    z_left = float(norm.ppf(1 - cfg.confidence))     # negative
    h = float(max(cfg.horizon_days, 1))
    use_mean = getattr(cfg, "use_mean", False)       # many desks ignore μ
    mu_h = mu * h if use_mean else 0.0

    q = mu_h + z_left * sigma * np.sqrt(h)           # lower-tail quantile (≤ 0 if there’s risk)
    var = (-q).clip(lower=0).astype(float)           # positive loss; 0 if even the tail is a gain
    var.name = f"Param VaR {int(cfg.confidence*100)}% ({cfg.window}d)"
    return var

def monte_carlo_var_point(returns: pd.Series, *, cfg: VaRConfig) -> float:
    """Point estimate of VaR via Monte Carlo using i.i.d. Gaussian sims."""
    r = _clean_returns(returns).dropna()
    tail = r.iloc[-cfg.window:]
    if len(tail) < cfg.window:
        return float('nan')
    mu = float(tail.mean())
    sigma = float(tail.std(ddof=1))
    rng = np.random.default_rng(cfg.seed)
    sims = rng.normal(mu, sigma, size=(cfg.n_sims, cfg.horizon_days))
    horizon_ret = (1 + sims).prod(axis=1) - 1
    q = np.quantile(horizon_ret, 1 - cfg.confidence)
    return float(-q)


def monte_carlo_cvar_point(returns: pd.Series, *, cfg: VaRConfig) -> float:
    r = _clean_returns(returns).dropna()
    tail = r.iloc[-cfg.window:]
    if len(tail) < cfg.window:
        return float('nan')
    mu = float(tail.mean())
    sigma = float(tail.std(ddof=1))
    rng = np.random.default_rng(cfg.seed)
    sims = rng.normal(mu, sigma, size=(cfg.n_sims, cfg.horizon_days))
    horizon_ret = (1 + sims).prod(axis=1) - 1
    cutoff = np.quantile(horizon_ret, 1 - cfg.confidence)
    es = -horizon_ret[horizon_ret <= cutoff].mean()
    return float(es)


def component_es(returns_df: pd.DataFrame, weights, confidence=0.99):
    """
    returns_df: T x N dataframe of daily returns per asset (columns = symbols)
    weights   : length-N weights (either portfolio weights sum=1, or $ notionals)
    Returns:
      es_total (scalar) in same units as weights
      df with columns: asset, component_es (same units), share (percentage of ES)
    """
    R = returns_df.to_numpy(dtype=float)          # (T, N)
    w = np.asarray(weights, dtype=float).reshape(-1)  # (N,)
    L = -(R @ w)                                  # portfolio loss scenarios

    alpha = confidence
    q = np.quantile(L, alpha)                     # VaR threshold (loss)
    tail = L >= q
    if tail.sum() == 0:
        raise ValueError("No tail scenarios; increase sample or lower confidence.")

    # Per-asset loss in tail scenarios: -w_i * r_{t,i}
    contrib = -(R[tail] * w.reshape(1, -1))       # (T_tail, N)
    ces = contrib.mean(axis=0)                    # component ES per asset
    es_total = float(ces.sum())

    out = pd.DataFrame({
        "asset": returns_df.columns,
        "component_es": ces,
        "share": (ces / es_total) if es_total != 0 else np.nan
    }).sort_values("component_es", ascending=False).reset_index(drop=True)

    return es_total, out


from math import sqrt
try:
    from scipy.stats import t as student_t
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

def _cornish_fisher_quantile(z, skew, kurt):
    # Simple CF expansion (excess kurtosis = kurt)
    z2, z3 = z*z, z*z*z
    return (z
            + (1/6)*(z2 - 1)*skew
            + (1/24)*(z3 - 3*z)*kurt
            - (1/36)*(2*z3 - 5*z)*(skew**2))

def parametric_student_t_var_es(returns, confidence=0.99, horizon_days=1):
    """
    Return positive VaR and ES (loss fractions) for the left tail.
    If SciPy isn't available, falls back to Cornish–Fisher adjusted Normal.
    """
    r = np.asarray(returns, float)
    r = r[np.isfinite(r)]
    if len(r) < 20:
        return np.nan, np.nan

    mu = float(np.mean(r))
    sigma = float(np.std(r, ddof=1))
    if sigma <= 0:
        return 0.0, 0.0

    tail = 1.0 - confidence

    if _HAS_SCIPY:
        # Fit Student-t by MLE: returns df, loc, scale
        # For stability on small samples you can fix loc=mu, scale=sigma and search df if you prefer.
        try:
            df, loc, scale = student_t.fit(r)  # MLE
        except Exception:
            df, loc, scale = 6.0, mu, sigma    # fallback df

        # 1-day quantiles in return space
        q1 = student_t.ppf(tail, df=df)                 # negative number
        pdf_q1 = student_t.pdf(q1, df=df)

        # VaR (return) and ES (return) for 1 day
        var_1d_r = loc + scale * q1
        # ES formula for Student-t left tail:
        # ES = E[R | R <= q] = loc - scale * ((df + q^2)/(df - 1)) * pdf(q) / tail
        # (valid for df > 1; df>2 advisable)
        if df > 1:
            es_1d_r = loc - scale * ((df + q1**2) / (df - 1)) * (pdf_q1 / tail)
        else:
            es_1d_r = loc + scale * q1  # degrade to VaR if df too small

        # Horizon scaling (square-root-of-time on sigma; mu linear)
        # For t, a rough-but-common approach:
        mu_h = mu * horizon_days
        sig_h = sigma * sqrt(horizon_days)
        # Re-express by scaling only the centered part:
        var_h_r = mu_h + (var_1d_r - mu) * sqrt(horizon_days)
        es_h_r  = mu_h + (es_1d_r  - mu) * sqrt(horizon_days)
    else:
        # Cornish–Fisher adjusted Normal (fallback)
        z = -abs(np.quantile(np.random.standard_normal(500000), tail))  # crude; or use scipy if present
        skew = float((np.mean(((r-mu)/sigma)**3)))
        kurt = float((np.mean(((r-mu)/sigma)**4) - 3.0))
        z_cf = _cornish_fisher_quantile(z, skew, kurt)
        var_h_r = mu*horizon_days + sigma*sqrt(horizon_days)*z_cf
        # ES fallback (Normal) ~ mu*h + sigma*sqrt(h) * φ(z)/tail
        phi = np.exp(-0.5*z_cf*z_cf)/np.sqrt(2*np.pi)
        es_h_r  = mu*horizon_days - sigma*sqrt(horizon_days) * (phi / tail)

    # Convert to positive loss numbers
    var_loss = max(0.0, -(var_h_r))
    es_loss  = max(0.0, -(es_h_r))
    return var_loss, es_loss

import numpy as np

def ewma_vol(r, lam=0.94):
    r = np.asarray(r, float)
    s2 = np.zeros_like(r)
    s2[0] = np.var(r, ddof=1)
    for t in range(1, len(r)):
        s2[t] = lam * s2[t-1] + (1 - lam) * r[t-1]**2
    return np.sqrt(s2)

def fhs_var_es(returns, confidence=0.99, horizon_days=1, n_sims=20000, lam=0.94, seed=42):
    """
    Filtered Historical Simulation:
      1) Standardize returns by EWMA vol -> residuals z
      2) Resample z, re-scale by current EWMA vol and √h
    Returns positive VaR and ES (loss fractions).
    """
    rng = np.random.default_rng(seed)
    r = np.asarray(returns, float)
    r = r[np.isfinite(r)]
    if len(r) < 50:
        return np.nan, np.nan

    mu = float(np.mean(r))
    sig = ewma_vol(r, lam=lam)
    sig_t = float(sig[-1])
    z = np.divide(r, sig, out=np.zeros_like(r), where=sig>0)

    # simulate horizon sum (assume iid residuals)
    draws = rng.choice(z, size=(n_sims, horizon_days), replace=True)
    sim = mu*horizon_days + sig_t*np.sqrt(horizon_days) * draws.sum(axis=1)

    tail = 1.0 - confidence
    var_r = np.quantile(sim, tail)
    es_r  = sim[sim <= var_r].mean() if np.any(sim <= var_r) else var_r

    return max(0.0, -var_r), max(0.0, -es_r)

# src/risk/var.py


RiskMethod = Literal["parametric", "historical_es"]

def _to_vec(w: pd.Series, index: pd.Index) -> np.ndarray:
    return w.reindex(index).fillna(0.0).astype(float).values

def _to_cov(df: pd.DataFrame, shrinkage: bool = False, lam: float = 0.1) -> pd.DataFrame:
    S = df.cov()
    if not shrinkage:
        return S
    # Simple diagonal shrinkage toward variances (no external deps)
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
    """
    Δ-Normal σ / VaR / ES with Euler allocation:
      σ_p = sqrt(w'Σw)
      grad σ = Σw / σ_p
      comp_σ_i = w_i * gradσ_i
      comp_VaR_i = z_α * comp_σ_i
      comp_ES_i  = c_α * comp_σ_i,  c_α = φ(z_α)/(1-α)
    """
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
    c = float(norm.pdf(z) / (1 - alpha))

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
    var_level: float  # portfolio VaR level used to define the tail (return, not loss)

def historical_es_decomposition(returns: pd.DataFrame, weights: pd.Series, alpha: float = 0.99) -> HistoricalESRisk:
    """
    Historical ES (Acerbi–Tasche style allocation for linear portfolios):
      r_p = R @ w
      ES = E[ L | L >= VaR ], L = -r_p
      component ES_i = E[ w_i * (-r_i) | tail ]
    Sums exactly: sum_i comp_ES_i = ES_total
    """
    cols = returns.columns
    w = _to_vec(weights, cols)
    R = returns.astype(float).values  # T×N
    r_p = R @ w                       # T
    q = np.quantile(r_p, 1 - alpha)   # VaR level for returns
    tail_mask = r_p <= q
    if not np.any(tail_mask):
        raise ValueError("No tail observations at this alpha; increase window or lower alpha.")

    L_tail = -r_p[tail_mask]
    ES_total = float(L_tail.mean())
    comp = (-R[tail_mask] * w).mean(axis=0)  # average loss attribution by asset
    comp_es = pd.Series(comp, index=cols, name="component_es")

    return HistoricalESRisk(
        es_total=ES_total,
        comp_es=comp_es,
        tail_count=int(tail_mask.sum()),
        var_level=float(q),
    )

def percent_of_total(components: pd.Series, total: float, use_abs: bool = False) -> pd.Series:
    base = components.abs().sum() if use_abs else total
    if base == 0:
        return components * 0
    return (components / base * 100).rename("% of total ({}base)".format("abs " if use_abs else ""))

def build_returns_from_prices(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Helper: tidy prices ['date','symbol','price'] -> wide daily returns DataFrame.
    """
    # Expect tidy input; pivot to wide prices
    wide = price_df.pivot(index="date", columns="symbol", values="price").sort_index()
    rets = wide.pct_change().dropna(how="all")
    return rets

