from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional

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
