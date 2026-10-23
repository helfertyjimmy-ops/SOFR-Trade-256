import pytest

from sofr_fly.payoff import Butterfly, Condor, breakeven_probability, pnl_dollars


def test_fly_peak_and_wings():
    f = Butterfly()
    assert f.payoff_bp(95.8125) == pytest.approx(6.25)
    assert f.payoff_bp(95.75) == pytest.approx(0.0)
    assert f.payoff_bp(95.875) == pytest.approx(0.0)
    assert f.payoff_bp(95.78) == pytest.approx(3.0)


def test_fly_breakevens_and_pnl():
    f = Butterfly()
    assert f.breakevens(1.3) == pytest.approx((95.763, 95.862))
    assert pnl_dollars(f.max_payoff_bp, 1.3) == pytest.approx(123.75)


def test_condor_flat_top():
    c = Condor()
    assert c.payoff_bp(95.75) == pytest.approx(6.25)
    assert c.payoff_bp(95.8125) == pytest.approx(6.25)
    assert c.payoff_bp(95.6875) == pytest.approx(0.0)


def test_breakeven_probability():
    assert breakeven_probability(1.3, 4.0) == pytest.approx(0.325)
