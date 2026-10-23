"""Monte Carlo of SFRZ6 at option expiry under discrete Fed scenarios.

State on Dec 11 = (hikes by Dec, January hike odds priced that day, SOFR basis noise),
plus a small tail of non-standard outcomes (50bp move, intermeeting action, cut).

WIRP gives expected hikes per meeting, not the joint probability of exactly one hike.
With hikes in {0,1,2} and a mean of 1.186, P(exactly 1) can be anywhere in [0.478, 0.814];
scenario probabilities below are therefore *forecasts*, labelled as such.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq

from .contract import MarketState, price_vectorized


@dataclass(frozen=True)
class Scenario:
    name: str
    p_hikes: tuple[float, float, float]                 # P(0, 1, 2 more hikes by Dec)
    jan_mean: tuple[float, float, float] = (0.6, 0.25, 0.4)   # mean Jan odds conditional on 0/1/2 hikes
    jan_sd: float = 0.20
    basis_sd_bp: float = 2.0
    tail_prob: float = 0.03
    tail_halfwidth: float = 0.50                         # index points, uniform around the forward

    def __post_init__(self):
        if not np.isclose(sum(self.p_hikes), 1.0):
            raise ValueError("p_hikes must sum to 1")


def _beta(rng, mean: float, sd: float, n: int) -> np.ndarray:
    if sd <= 0:
        return np.full(n, mean)
    m = float(np.clip(mean, 1e-3, 1 - 1e-3))
    var = min(sd ** 2, m * (1 - m) * 0.95)
    k = m * (1 - m) / var - 1
    return rng.beta(m * k, (1 - m) * k, n)


def wirp_bounds_p_exactly_one(mean_hikes: float, p_oct: float) -> tuple[float, float]:
    """Range of P(exactly one hike by Dec) consistent with WIRP, assuming 25bp steps (K in {0,1,2}).

    P(2) must lie in [max(0, E-1), min(P(Oct), P(Dec))] with P(Dec) = E - P(Oct); P(1) = E - 2 P(2).
    """
    p_dec = mean_hikes - p_oct
    p2_lo, p2_hi = max(0.0, mean_hikes - 1.0), min(p_oct, p_dec)
    return mean_hikes - 2 * p2_hi, mean_hikes - 2 * p2_lo


def calibrate_basis(forward: float, p_hikes, jan_odds: float, state: MarketState = MarketState()) -> float:
    """Basis (bp) such that the market-consistent expected price equals the observed forward."""
    k = np.array([0, 1, 2], dtype=float)
    p = np.asarray(p_hikes)

    def gap(b):
        return float(np.sum(p * price_vectorized(k, np.full(3, jan_odds), b, state))) - forward

    return brentq(gap, -50, 50)


def simulate(scn: Scenario, forward: float, basis_bp: float, n: int = 400_000, seed: int = 7,
             state: MarketState = MarketState()) -> np.ndarray:
    rng = np.random.default_rng(seed)
    hikes = rng.choice(3, size=n, p=scn.p_hikes)
    jan = np.empty(n)
    for k in range(3):
        idx = hikes == k
        jan[idx] = _beta(rng, scn.jan_mean[k], scn.jan_sd, int(idx.sum()))
    basis = basis_bp + scn.basis_sd_bp * rng.standard_normal(n)
    x = price_vectorized(hikes.astype(float), jan, basis, state)
    if scn.tail_prob > 0:
        t = rng.random(n) < scn.tail_prob
        x[t] = forward + rng.uniform(-scn.tail_halfwidth, scn.tail_halfwidth, int(t.sum()))
    return x


# Scenarios used in the deck (forecasts, not market data)
MARKET_LIKE = Scenario("Market-like (independent meetings)", (0.109, 0.596, 0.295), (0.528, 0.528, 0.528), jan_sd=0.25, tail_prob=0.0)
BASE_CASE = Scenario("Base case: one hike, then pause", (0.10, 0.80, 0.10), (0.6, 0.10, 0.4), jan_sd=0.15)
CENTRAL = Scenario("Central: one hike, data-dependent", (0.10, 0.65, 0.25), (0.6, 0.25, 0.4))
HAWKISH = Scenario("Hawkish: two-hike risk", (0.15, 0.50, 0.35), (0.6, 0.25, 0.4))
