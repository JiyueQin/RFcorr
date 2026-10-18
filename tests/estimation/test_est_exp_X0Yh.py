"""Tests for rc.est_exp_X0Yh. Run with: python -m pytest tests/estimation/test_est_exp_X0Yh.py"""
import itertools
import numpy as np
import pytest

import rfcorr as rc


def _brute_force_lag_mean(X, Y):
    """E[h] = mean over valid l of X[l + h] * Y[l], for every lag h."""
    dim = X.shape
    lag_ranges = [range(-(n - 1), n) for n in dim]
    E = np.zeros([2 * n - 1 for n in dim])
    for h in itertools.product(*lag_ranges):
        total, count = 0.0, 0
        for l in itertools.product(*[range(n) for n in dim]):
            lh = tuple(li + hi for li, hi in zip(l, h))
            if all(0 <= v < n for v, n in zip(lh, dim)):
                total += X[lh] * Y[l]
                count += 1
        E[tuple(hi + n - 1 for hi, n in zip(h, dim))] = total / count
    return E


@pytest.mark.parametrize("dim", [(7,), (4, 5), (3, 3, 4)])
def test_est_exp_X0Yh_matches_brute_force(dim):
    rng = np.random.default_rng(0)
    X, Y = rng.normal(size=dim), rng.normal(size=dim)
    np.testing.assert_allclose(rc.est_exp_X0Yh(X, Y), _brute_force_lag_mean(X, Y))


def test_est_exp_X0Yh_counts_lags_and_symmetry():
    rng = np.random.default_rng(1)
    X, Y = rng.normal(size=(4, 6)), rng.normal(size=(4, 6))
    S, C, E, lag_ls = rc.est_exp_X0Yh(X, Y, return_counts_lags=True)
    assert E.shape == C.shape == S.shape == (7, 11)
    np.testing.assert_array_equal(lag_ls[0], np.arange(-3, 4))
    np.testing.assert_array_equal(lag_ls[1], np.arange(-5, 6))
    # zero lag uses all n pairs and equals mean(X * Y)
    assert C[3, 5] == X.size
    assert E[3, 5] == pytest.approx(np.mean(X * Y))
    # swapping X and Y reverses the lags
    np.testing.assert_allclose(rc.est_exp_X0Yh(Y, X), E[::-1, ::-1])


def test_est_exp_X0Yh_shape_mismatch():
    with pytest.raises(ValueError):
        rc.est_exp_X0Yh(np.zeros((3, 3)), np.zeros((3, 4)))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
