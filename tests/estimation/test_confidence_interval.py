"""Tests for rc.confidence_interval. Run with: python -m pytest tests/estimation/test_confidence_interval.py"""
import numpy as np
import pytest
from scipy.stats import norm

import rfcorr as rc


def test_confidence_interval_wald():
    lower, upper = rc.confidence_interval(0.3, 0.1, fisher_trans=False)
    z = norm.ppf(0.975)
    assert lower == pytest.approx(0.3 - z * 0.1)
    assert upper == pytest.approx(0.3 + z * 0.1)
    assert isinstance(lower, float)


def test_confidence_interval_wald_is_clipped():
    lower, upper = rc.confidence_interval(0.95, 0.2, fisher_trans=False)
    assert upper == 1.0
    lower, upper = rc.confidence_interval(-0.95, 0.2, fisher_trans=False)
    assert lower == -1.0


def test_confidence_interval_fisher():
    rho, se = 0.6, 0.1
    lower, upper = rc.confidence_interval(rho, se)
    z = norm.ppf(0.975)
    se_F = se / (1 - rho**2)
    assert lower == pytest.approx(np.tanh(np.arctanh(rho) - z * se_F))
    assert upper == pytest.approx(np.tanh(np.arctanh(rho) + z * se_F))
    assert -1 < lower < rho < upper < 1
    # the interval is symmetric on the Fisher scale, not on the rho scale
    assert np.arctanh(upper) - np.arctanh(rho) == pytest.approx(np.arctanh(rho) - np.arctanh(lower))


def test_confidence_interval_alpha_and_vectorized():
    narrow = rc.confidence_interval(0.2, 0.1, alpha=0.2)
    wide = rc.confidence_interval(0.2, 0.1, alpha=0.01)
    assert wide[0] < narrow[0] and wide[1] > narrow[1]
    rho = np.array([-0.5, 0.0, 0.5])
    lower, upper = rc.confidence_interval(rho, np.full(3, 0.1))
    assert lower.shape == upper.shape == (3,)
    np.testing.assert_allclose(lower, -upper[::-1])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
