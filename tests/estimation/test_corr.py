"""Tests for rc.corr. Run with: python -m pytest tests/estimation/test_corr.py"""
import numpy as np
import pytest

import rfcorr as rc


@pytest.mark.parametrize("se_method", ["general", "gauss", "indep", "iid"])
def test_matches_corr_dev_with_paper_rule(fields_2d, se_method):
    X, Y = fields_2d
    out = rc.corr(X, Y, se_method=se_method)
    out_dev = rc.corr_dev(X, Y, se_method=se_method, M="auto", kernel="bartlett")
    assert out[:3] == out_dev[:3]
    np.testing.assert_array_equal(out[4], out_dev[4])


def test_default_lag_is_optimal_rate(fields_2d):
    X, Y = fields_2d
    n = X.size
    np.testing.assert_array_equal(rc.corr(X, Y)[4], np.full(2, np.ceil(n**0.25)))
    np.testing.assert_array_equal(rc.corr(X, Y, constant=2)[4], np.full(2, np.ceil(2 * n**0.25)))


def test_fixed_lags(fields_2d):
    X, Y = fields_2d
    assert rc.corr(X, Y, M=3)[4] == (3, 3)
    assert rc.corr(X, Y, M=np.int64(3))[4] == (3, 3)
    assert rc.corr(X, Y, M=(2, 4))[4] == (2, 4)
    assert rc.corr(X, Y, M=3)[:3] == rc.corr_dev(X, Y, M=3)[:3]


def test_mask(fields_2d):
    X, Y = fields_2d
    mask = rc.gen_mask(X.shape, axes=(1, 0.7))
    out = rc.corr(X, Y, mask=mask)
    assert out[0] == pytest.approx(np.corrcoef(X[mask], Y[mask])[0, 1])
    assert out[:3] == rc.corr_dev(X, Y, M="optimal_rate", mask=mask)[:3]


@pytest.mark.parametrize("M", ["auto", "cubic_root", "linear", "plugin", 2.5, (2, 2.5), True])
def test_rejects_development_lag_rules(fields_2d, M):
    X, Y = fields_2d
    with pytest.raises(ValueError, match="optimal_rate"):
        rc.corr(X, Y, M=M)


@pytest.mark.parametrize("option", [{"kernel": "qs"}, {"true_rho": 0.4}, {"check": True}])
def test_development_options_not_accepted(fields_2d, option):
    X, Y = fields_2d
    with pytest.raises(TypeError):
        rc.corr(X, Y, **option)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
