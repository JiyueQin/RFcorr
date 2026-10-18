"""Tests for rc.gaussian_filter_acf. Run with: python -m pytest tests/fields/test_gaussian_filter_acf.py"""
import numpy as np
import pytest

import rfcorr as rc


def test_gaussian_filter_acf_values():
    r = [rc.gaussian_filter_acf((1, 0), f) for f in [1, 2, 3, 4]]
    np.testing.assert_allclose(r, [0.124, 0.705, 0.857, 0.917], atol=1e-3)
    assert rc.gaussian_filter_acf((0, 0), 3) == pytest.approx(1.0)
    # symmetric in the lag, and separable across axes
    assert rc.gaussian_filter_acf((-1, 0), 3) == pytest.approx(rc.gaussian_filter_acf((1, 0), 3))
    assert rc.gaussian_filter_acf((1, 1), 3) == pytest.approx(rc.gaussian_filter_acf(1, 3) ** 2)


def test_gaussian_filter_acf_matches_simulated_fields():
    Z = rc.gen_field(dim=(300, 300), method="gaussian_filter", fwhm=3, seed=0)
    for h, (a, b) in {(1, 0): (Z[:-1], Z[1:]), (0, 2): (Z[:, :-2], Z[:, 2:])}.items():
        assert np.corrcoef(a.ravel(), b.ravel())[0, 1] == pytest.approx(
            rc.gaussian_filter_acf(h, 3), abs=0.02)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
