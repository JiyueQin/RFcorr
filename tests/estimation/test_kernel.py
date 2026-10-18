"""Tests for rc.kernel. Run with: python -m pytest tests/estimation/test_kernel.py"""
import numpy as np
import pytest

import rfcorr as rc


def test_kernel_bartlett():
    x = np.array([-2, -1, -0.5, 0, 0.25, 1, 3])
    np.testing.assert_allclose(rc.kernel(x, "bartlett"), [0, 0, 0.5, 1, 0.75, 0, 0])


def test_kernel_qs():
    x = np.array([-2.0, -0.3, 0.0, 1e-10, 0.3, 2.0])
    k = rc.kernel(x, "QS")
    assert k[2] == 1.0 and k[3] == 1.0
    np.testing.assert_allclose(k, k[::-1])  # symmetric
    # continuous at 0
    assert rc.kernel(np.array([1e-4]), "qs")[0] == pytest.approx(1.0, abs=1e-6)
    # closed form at x = 1
    z = 6 * np.pi / 5
    expected = 25 / (12 * np.pi**2) * (np.sin(z) / z - np.cos(z))
    assert rc.kernel(np.array([1.0]), "qs")[0] == pytest.approx(expected)


def test_kernel_constant_and_unknown():
    assert rc.kernel(np.array([5.0]), "constant") == 1
    with pytest.raises(ValueError):
        rc.kernel(np.array([0.0]), "triangle")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
