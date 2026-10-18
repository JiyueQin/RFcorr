"""Tests for rc.pvalue. Run with: python -m pytest tests/estimation/test_pvalue.py"""
import numpy as np
import pytest
from scipy.stats import norm

import rfcorr as rc


def test_pvalue():
    assert rc.pvalue(0.0, 0.1) == pytest.approx(1.0)
    assert rc.pvalue(0.2, 0.1, fisher_trans=False) == pytest.approx(2 * (1 - norm.cdf(2)))
    assert rc.pvalue(0.2, 0.1) == pytest.approx(2 * (1 - norm.cdf(np.arctanh(0.2) / 0.1)))
    assert rc.pvalue(-0.2, 0.1) == pytest.approx(rc.pvalue(0.2, 0.1))
    # larger |rho| gives a smaller p value
    p = rc.pvalue(np.array([0.1, 0.3, 0.5]), np.full(3, 0.1))
    assert p.shape == (3,) and np.all(np.diff(p) < 0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
