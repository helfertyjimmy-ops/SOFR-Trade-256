"""Bachelier (normal) option pricing on SOFR futures prices.

Prices and strikes are IMM index points (e.g. 95.75); vol is in index points per
sqrt(year), so 0.53 = 53bp/yr normal vol. Discounting is ignored (<1% effect over
~2 months on premium-upfront options).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def call(F: float, K: float, T: float, vol: float) -> float:
    if T <= 0 or vol <= 0:
        return max(F - K, 0.0)
    v = vol * np.sqrt(T)
    d = (F - K) / v
    return (F - K) * norm.cdf(d) + v * norm.pdf(d)


def put(F: float, K: float, T: float, vol: float) -> float:
    return call(F, K, T, vol) - (F - K)


def implied_vol(price: float, F: float, K: float, T: float, kind: str = "call") -> float:
    f = call if kind == "call" else put
    return brentq(lambda s: f(F, K, T, s) - price, 1e-6, 5.0)


def straddle_implied_vol(straddle: float, F: float, K: float, T: float) -> float:
    return brentq(lambda s: call(F, K, T, s) + put(F, K, T, s) - straddle, 1e-6, 5.0)


def parity_forward(strikes, call_mids, put_mids) -> float:
    """Forward implied by put-call parity, C - P = F - K, averaged across strikes."""
    return float(np.mean(np.asarray(call_mids) - np.asarray(put_mids) + np.asarray(strikes)))


def butterfly(F: float, center: float, wing: float, T: float, vol: float) -> float:
    """Flat-vol value of a long call butterfly (center-wing / center / center+wing), in index points."""
    return call(F, center - wing, T, vol) - 2 * call(F, center, T, vol) + call(F, center + wing, T, vol)
