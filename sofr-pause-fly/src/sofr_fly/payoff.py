"""Expiry payoffs and P&L for the pause fly and the one-hike condor (all in bp unless noted)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contract import TICK_VALUE_PER_BP


@dataclass(frozen=True)
class Butterfly:
    low: float = 95.75
    mid: float = 95.8125
    high: float = 95.875

    def payoff_bp(self, x):
        x = np.asarray(x, dtype=float)
        width = (self.mid - self.low) * 100
        return np.maximum(0.0, width - np.abs(x - self.mid) * 100)

    @property
    def max_payoff_bp(self) -> float:
        return (self.mid - self.low) * 100

    def breakevens(self, cost_bp: float) -> tuple[float, float]:
        return self.low + cost_bp / 100, self.high - cost_bp / 100


@dataclass(frozen=True)
class Condor:
    k1: float = 95.6875
    k2: float = 95.75
    k3: float = 95.8125
    k4: float = 95.875

    def payoff_bp(self, x):
        x = np.asarray(x, dtype=float)
        up = np.clip((x - self.k1) * 100, 0, (self.k2 - self.k1) * 100)
        down = np.clip((self.k4 - x) * 100, 0, (self.k4 - self.k3) * 100)
        return np.minimum(up, down)

    @property
    def max_payoff_bp(self) -> float:
        return (self.k2 - self.k1) * 100

    def breakevens(self, cost_bp: float) -> tuple[float, float]:
        return self.k1 + cost_bp / 100, self.k4 - cost_bp / 100


def pnl_dollars(payoff_bp, cost_bp: float, fees_bp: float = 0.0):
    """P&L per structure in dollars: (payoff - cost - fees) x $25 per bp."""
    return (np.asarray(payoff_bp) - cost_bp - fees_bp) * TICK_VALUE_PER_BP


def breakeven_probability(cost_bp: float, avg_payoff_in_zone_bp: float, fees_bp: float = 0.0) -> float:
    """Probability of landing in the zone needed for zero expected P&L (payoff outside the zone = 0)."""
    return (cost_bp + fees_bp) / avg_payoff_in_zone_bp
