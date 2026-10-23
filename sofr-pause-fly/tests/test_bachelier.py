import pytest

from sofr_fly import bachelier as bl


def test_put_call_parity():
    F, K, T, v = 95.70, 95.75, 0.16, 0.53
    assert bl.call(F, K, T, v) - bl.put(F, K, T, v) == pytest.approx(F - K)


def test_implied_vol_roundtrip():
    p = bl.call(95.70, 95.75, 0.16, 0.53)
    assert bl.implied_vol(p, 95.70, 95.75, 0.16) == pytest.approx(0.53, abs=1e-6)


def test_atm_straddle_is_0798_sd():
    F, T, v = 95.70, 0.25, 0.5
    sd = v * T ** 0.5
    assert bl.call(F, F, T, v) + bl.put(F, F, T, v) == pytest.approx(0.7979 * sd, rel=1e-3)
