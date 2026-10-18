"""Tests for rc.plugin_bandwidth. Run with: python -m pytest tests/estimation/test_plugin_bandwidth.py"""
import numpy as np
import pytest

import rfcorr as rc


def test_plugin_bandwidth_matches_andrews_in_1d():
    # V with AR(1) autocovariance r^|h| in 1D: m = 1.1447 (a^2 n)^(1/3), a = 2r / (1 - r^2)
    n, r = 1000, 0.6
    h = np.arange(-(n - 1), n)
    cov_ls = [r ** np.abs(h)] + [0] * 8
    a = 2 * r / (1 - r**2)
    m = 1.1447 * (a**2 * n) ** (1 / 3)
    M = rc.plugin_bandwidth(cov_ls, np.array([1, 0, 0]), n, (n,))
    assert M[0] == np.ceil(m - 1)


def test_plugin_bandwidth_grows_with_smoothness():
    Ms = []
    for fwhm in (1, 2, 4):
        X, Y = rc.gen_corr_field(0, (60, 60), fwhm=fwhm, seed=0, method="gaussian_filter")
        Ms.append(rc.corr_dev(X, Y, M="plugin", true_rho=0)[4][0])
    assert Ms[0] < Ms[1] < Ms[2]


def test_plugin_bandwidth_capped_at_half_side():
    X, Y = rc.gen_corr_field(0, (12, 40), fwhm=8, seed=0, method="gaussian_filter")
    M = rc.corr_dev(X, Y, M="plugin", true_rho=0)[4]
    assert M[0] <= 5 and M[1] <= 19 and M[0] >= 1


@pytest.mark.parametrize("se_method", ["general", "gauss", "indep"])
def test_corr_plugin_runs_with_mask(se_method):
    X, Y = rc.gen_corr_field(0.4, (30, 40), fwhm=3, seed=1)
    mask = rc.gen_mask(X.shape, axes=(0.8, 1))
    rho, se, se0, _, M = rc.corr_dev(X, Y, M="plugin", se_method=se_method, mask=mask)
    assert np.isfinite(se) and np.isfinite(se0) and (M >= 1).all()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
