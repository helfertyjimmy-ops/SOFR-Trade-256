import numpy as np
import pytest

from sofr_fly.contract import JAN_WEIGHT, MarketState, price_vectorized, reference_days, sfrz6_price


def test_reference_quarter_is_91_days():
    assert len(reference_days()) == 91


def test_january_weight_is_48_of_91():
    assert JAN_WEIGHT == pytest.approx(48 / 91)


def test_one_hike_moves_price_about_25bp():
    s = MarketState()
    assert (sfrz6_price(0, 0.0, s) - sfrz6_price(1, 0.0, s)) * 100 == pytest.approx(25.3, abs=0.2)


def test_full_jan_hike_moves_price_about_13bp():
    s = MarketState()
    assert (sfrz6_price(1, 0.0, s) - sfrz6_price(1, 1.0, s)) * 100 == pytest.approx(13.3, abs=0.3)


def test_vectorized_matches_daily_compounding():
    s = MarketState(basis_bp=3.0)
    for k, j in [(0, 0.6), (1, 0.25), (2, 0.4)]:
        assert price_vectorized(np.array([k]), np.array([j]), 3.0, s)[0] == pytest.approx(sfrz6_price(k, j, s), abs=1e-4)
