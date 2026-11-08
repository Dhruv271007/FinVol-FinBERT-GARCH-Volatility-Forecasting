"""Gaussian QMLE for GARCH and signed-sentiment GARCH-X.

All returns are percentage log returns and all h values are squared percentage
units. Sentiment is already aligned to the target return, bounded by [-1, 1].
The initial variance must be supplied when filtering beyond the training set.
"""
import numpy as np
from scipy.optimize import minimize, LinearConstraint
from scipy.signal import lfilter
from scipy.stats import norm
from project_config import PERSISTENCE_CAP, VARIANCE_MARGIN


def validate(returns, sentiment=None):
    r = np.asarray(returns, dtype=float)
    if r.ndim != 1 or len(r) < 2 or not np.isfinite(r).all():
        raise ValueError("Need at least two finite, one-dimensional returns")
    s = None if sentiment is None else np.asarray(sentiment, dtype=float)
    if s is not None and (s.shape != r.shape or not np.isfinite(s).all() or np.max(np.abs(s)) > 1 + 1e-10):
        raise ValueError("Sentiment must match returns, be finite and lie in [-1, 1]")
    return r, s


def variance_path(params, returns, sentiment=None, initial_variance=None):
    r, s = validate(returns, sentiment)
    p = np.asarray(params, dtype=float)
    if len(p) != (4 if s is None else 5):
        raise ValueError("Incorrect parameter count")
    mu, omega, alpha, beta = p[:4]
    h0 = max(float(np.var(r)), VARIANCE_MARGIN) if initial_variance is None else float(initial_variance)
    if not np.isfinite(h0) or h0 <= 0:
        raise ValueError("Initial variance must be finite and positive")
    forcing = omega + alpha * (r[:-1] - mu) ** 2
    if s is not None:
        forcing = forcing + p[4] * s[1:]
    # h[t] = forcing[t-1] + beta*h[t-1], computed in compiled scipy code.
    tail, _ = lfilter([1.0], [1.0, -beta], forcing, zi=[beta * h0])
    return np.r_[h0, tail]


def nll_contributions(params, returns, sentiment=None, initial_variance=None):
    h = variance_path(params, returns, sentiment, initial_variance)
    if not np.isfinite(h).all() or np.any(h <= 0):
        return np.full(len(returns), np.inf)
    return 0.5 * (np.log(2 * np.pi) + np.log(h) + (np.asarray(returns) - params[0]) ** 2 / h)


def fit_model(returns, sentiment=None, baseline=None):
    """Fit GARCH (no sentiment) or GARCH-X. Pass an existing GARCH fit as
    ``baseline`` to avoid refitting it for the nested GARCH-X start."""
    r, s = validate(returns, sentiment)
    h0 = max(float(np.var(r)), VARIANCE_MARGIN)
    n = 4 if s is None else 5
    # alpha+beta <= cap. For X also omega-gamma and omega+gamma >= margin.
    rows = [[0, 0, 1, 1] + ([0] if n == 5 else [])]
    lower, upper = [-np.inf], [PERSISTENCE_CAP]
    if n == 5:
        rows += [[0, 1, 0, 0, -1], [0, 1, 0, 0, 1]]
        lower += [VARIANCE_MARGIN, VARIANCE_MARGIN]
        upper += [np.inf, np.inf]
    constraint = LinearConstraint(np.array(rows), lower, upper)
    bounds = [(None, None), (VARIANCE_MARGIN, None), (0, 1), (0, 1)] + ([(None, None)] if n == 5 else [])
    starts = []
    for a, b in [(0.05, 0.90), (0.10, 0.70), (0.15, 0.30)]:
        w = max(h0 * (1 - a - b), 10 * VARIANCE_MARGIN)
        for g in ([0, -0.5 * w, 0.5 * w] if n == 5 else [None]):
            starts.append([float(r.mean()), w, a, b] + ([] if g is None else [g]))
    if n == 5:
        if baseline is None:
            baseline = fit_model(r)
        starts.append([*baseline.x, 0.0])

    def objective(p):
        values = nll_contributions(p, r, s, h0)
        return float(values.mean()) if np.isfinite(values).all() else 1e20

    candidates = []
    for start in starts:
        result = minimize(objective, start, method="SLSQP", bounds=bounds,
                          constraints=[constraint], options={"maxiter": 1000, "ftol": 1e-10})
        p = result.x
        feasible = p[1] > 0 and min(p[2:4]) >= -1e-9 and p[2] + p[3] <= PERSISTENCE_CAP + 1e-8
        if n == 5:
            feasible = feasible and p[1] - abs(p[4]) >= VARIANCE_MARGIN - 1e-9
        if result.success and feasible and np.isfinite(result.fun) and result.fun < 1e19:
            candidates.append(result)
    if not candidates:
        raise RuntimeError("No converged feasible solution among deterministic starts")
    result = min(candidates, key=lambda item: item.fun)
    result.fun *= len(r)  # Report the TOTAL NLL, although average NLL is optimized.
    result.initial_variance = h0
    result.n_starts = len(starts)
    result.n_converged = len(candidates)
    result.boundary = bool(result.x[2] < 1e-5 or result.x[3] < 1e-5 or
                           PERSISTENCE_CAP - sum(result.x[2:4]) < 1e-5 or
                           (n == 5 and result.x[1] - abs(result.x[4]) < 1e-5))
    return result


def qlike_loss(realized, forecast):
    y, h = np.asarray(realized, float), np.asarray(forecast, float)
    if y.shape != h.shape or not np.isfinite(y).all() or not np.isfinite(h).all() or np.any(y < 0) or np.any(h <= 0):
        raise ValueError("QLIKE requires matching finite nonnegative targets and positive forecasts")
    return y / h + np.log(h)


def diebold_mariano(loss_a, loss_b, h=1, max_lag=None):
    """Exploratory normal-reference HAC loss comparison, not a nested-model correction."""
    a, b = np.asarray(loss_a, float), np.asarray(loss_b, float)
    if a.ndim != 1 or a.shape != b.shape or len(a) < 10 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Need matching finite loss arrays with at least ten observations")
    d = a - b
    T = len(d)
    lag = max(h - 1, int(np.floor(4 * (T / 100) ** (2 / 9)))) if max_lag is None else int(max_lag)
    if not 0 <= lag < T:
        raise ValueError("Invalid HAC lag")
    u = d - d.mean()
    lrv = float(u @ u / T)
    for k in range(1, lag + 1):
        lrv += 2 * (1 - k / (lag + 1)) * float(u[k:] @ u[:-k] / T)
    if lrv <= np.finfo(float).eps * max(1, float(np.mean(d ** 2))):
        return np.nan, np.nan, float(d.mean())
    stat = float(d.mean() / np.sqrt(lrv / T))
    return stat, float(2 * norm.sf(abs(stat))), float(d.mean())


def holm_adjust(pvalues):
    p = np.asarray(pvalues, float)
    result = np.full_like(p, np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    order = valid[np.argsort(p[valid])]
    if len(order):
        result[order] = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(order), 0, -1)))
    return result
