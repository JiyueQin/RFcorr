"""Tests for rc.kernel_weights. Run with: python -m pytest tests/estimation/test_kernel_weights.py"""
import numpy as np
import pytest

import rfcorr as rc


def test_kernel_weights_bartlett():
    x = np.arange(-5, 6)
    w = rc.kernel_weights(x, "bartlett", bandwidth=None, M=3)
    # Bartlett uses bandwidth M + 1, so weights are 1 - |h|/4 up to |h| = 3
    expected = np.where(np.abs(x) <= 3, 1 - np.abs(x) / 4, 0)
    np.testing.assert_allclose(w, expected)


def test_kernel_weights_constant_and_qs():
    x = np.arange(-4, 5)
    w_const = rc.kernel_weights(x, "constant", None, M=2)
    assert w_const.dtype == float
    np.testing.assert_array_equal(w_const, np.abs(x) <= 2)
    w = rc.kernel_weights(x, "qs", bandwidth=2, M=3)
    assert w[4] == 1.0
    assert np.all(w[np.abs(x) > 3] == 0)
    np.testing.assert_allclose(w, w[::-1])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
