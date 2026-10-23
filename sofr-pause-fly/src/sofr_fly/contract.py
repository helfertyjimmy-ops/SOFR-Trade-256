"""SFRZ6 contract mechanics: reference quarter, FOMC weights, and the price map.

SFRZ6 (CME 3M SOFR, Dec-26) settles to 100 - compounded daily SOFR over its
reference quarter, Dec 16 2026 (inclusive) to Mar 17 2027 (exclusive), ACT/360.
Its options expire Fri Dec 11 2026, two days after the Dec 9 FOMC, so on the
expiry date the futures price is the market's *forecast* of that quarter.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np

REF_START = date(2026, 12, 16)
REF_END = date(2027, 3, 17)          # exclusive
OPTION_EXPIRY = date(2026, 12, 11)
TICK_VALUE_PER_BP = 25.0             # $2,500 x IMM index -> $25 per bp per contract

# FOMC decision dates -> date the new rate takes effect (day after the meeting)
FOMC_EFFECTIVE = {
    "2026-10-28": date(2026, 10, 29),
    "2026-12-09": date(2026, 12, 10),
    "2027-01-27": date(2027, 1, 28),
    "2027-03-17": date(2027, 3, 18),
}


def reference_days() -> list[date]:
    n = (REF_END - REF_START).days
    return [REF_START + timedelta(days=i) for i in range(n)]


def meeting_weight(effective: date) -> float:
    """Share of the reference quarter on which a decision taking effect on `effective` applies."""
    days = reference_days()
    return sum(d >= effective for d in days) / len(days)


JAN_WEIGHT = meeting_weight(FOMC_EFFECTIVE["2027-01-27"])   # 48/91


@dataclass(frozen=True)
class MarketState:
    effr0: float = 3.88          # effective fed funds today, % (Bloomberg WIRP, 1 Oct 2026)
    hike_size: float = 0.25      # %, one 25bp move
    basis_bp: float = 3.0        # SOFR - EFFR + year-end turn + premium, bp (calibrated)


def compounded_rate(daily_rates_pct: np.ndarray) -> float:
    """ACT/360 compounded rate (%) over a sequence of daily overnight rates (%)."""
    growth = np.prod(1.0 + daily_rates_pct / 100.0 / 360.0)
    return (growth - 1.0) * 360.0 / len(daily_rates_pct) * 100.0


def sfrz6_price(hikes_by_dec: int | float, jan_odds: float, state: MarketState = MarketState()) -> float:
    """SFRZ6 price on Dec 11 given the Oct+Dec hike count and the January hike odds priced that day.

    Oct 28 and Dec 9 decisions take effect before Dec 16, so only the *count* matters.
    A January hike applies from Jan 28 onward (48 of 91 days); March has zero weight.
    The January expectation enters linearly (expected rate path), then compounds daily.
    """
    days = reference_days()
    base = state.effr0 + state.hike_size * hikes_by_dec + state.basis_bp / 100.0
    jan_eff = FOMC_EFFECTIVE["2027-01-27"]
    path = np.array([base + (state.hike_size * jan_odds if d >= jan_eff else 0.0) for d in days])
    return 100.0 - compounded_rate(path)


def price_vectorized(hikes: np.ndarray, jan_odds: np.ndarray, basis_bp: np.ndarray | float,
                     state: MarketState = MarketState()) -> np.ndarray:
    """Fast vectorized version of `sfrz6_price` for Monte Carlo (two-segment compounding)."""
    n = len(reference_days())
    n_jan = int(round(JAN_WEIGHT * n))
    n_pre = n - n_jan
    r1 = (state.effr0 + state.hike_size * hikes + np.asarray(basis_bp) / 100.0) / 100.0
    r2 = r1 + state.hike_size * jan_odds / 100.0
    growth = (1 + r1 / 360.0) ** n_pre * (1 + r2 / 360.0) ** n_jan
    return 100.0 - (growth - 1.0) * 360.0 / n * 100.0
