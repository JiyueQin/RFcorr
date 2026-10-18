"""Tests for rc.get_se. Run with: python -m pytest tests/estimation/test_get_se.py"""
import numpy as np
import pytest

import rfcorr as rc


def test_get_se_negative_variance_warns():
    # an omega that makes vec @ omega @ vec negative
    dim = np.array([5])
    est_exp_ls = [-np.ones(9)] + [np.zeros(9)] * 8
    with pytest.warns(UserWarning, match="negative"):
        se, _ = rc.get_se(dim, est_exp_ls, rho_plugin=0, M=2, se_method="gauss")
    assert np.isnan(se)


def test_get_se_radial_bartlett():
    # only lag (0, 0) and the diagonal lag (1, 1) carry covariance, in the (1, 1) entry of omega
    dim = np.array([4, 4])
    cov = np.zeros((7, 7))
    cov[3, 3] = 1.0
    cov[4, 4] = 0.5
    est_exp_ls = [cov] + [np.zeros((7, 7))] * 8
    se_prod, _ = rc.get_se(dim, est_exp_ls, rho_plugin=0, M=1, se_method="gauss")
    se_rad, _ = rc.get_se(dim, est_exp_ls, rho_plugin=0, M=1, se_method="gauss",
                          kernel="bartlett_radial")
    # with M = 1 the weight at lag (1, 1) is (1/2)^2 for the product kernel and
    # 1 - sqrt(2)/2 for the radial kernel
    assert se_prod == pytest.approx(np.sqrt((1 + 0.5 * 0.25) / 16))
    assert se_rad == pytest.approx(np.sqrt((1 + 0.5 * (1 - np.sqrt(2) / 2)) / 16))
    # in 1D the radial and product Bartlett weights coincide
    est_1d = [np.array([0.2, 0.5, 1.0, 0.5, 0.2])] + [np.zeros(5)] * 8
    se1, _ = rc.get_se(np.array([3]), est_1d, rho_plugin=0, M=2, se_method="gauss")
    se2, _ = rc.get_se(np.array([3]), est_1d, rho_plugin=0, M=2, se_method="gauss",
                       kernel="bartlett_radial")
    assert se1 == pytest.approx(se2)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
