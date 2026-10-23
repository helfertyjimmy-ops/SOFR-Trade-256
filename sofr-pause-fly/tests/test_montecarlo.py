import numpy as np
import pytest

from sofr_fly.montecarlo import BASE_CASE, MARKET_LIKE, calibrate_basis, simulate, wirp_bounds_p_exactly_one


def test_wirp_bounds():
    lo, hi = wirp_bounds_p_exactly_one(1.186, 0.354)
    assert (lo, hi) == pytest.approx((0.478, 0.814), abs=1e-3)


def test_calibration_recovers_forward():
    F = 95.6964
    b = calibrate_basis(F, MARKET_LIKE.p_hikes, 0.528)
    x = simulate(MARKET_LIKE.__class__(**{**MARKET_LIKE.__dict__, "jan_sd": 0.0, "basis_sd_bp": 0.0}), F, b, n=200_000)
    assert x.mean() == pytest.approx(F, abs=0.003)


def test_base_case_concentrates_in_landing_zone():
    x = simulate(BASE_CASE, 95.6964, 3.45, n=100_000)
    assert np.mean((x > 95.75) & (x < 95.875)) > 0.6
