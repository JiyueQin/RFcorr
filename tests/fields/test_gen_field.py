"""Tests for rc.gen_field. Run with: python -m pytest tests/fields/test_gen_field.py"""
import numpy as np
import pytest

import rfcorr as rc


@pytest.mark.parametrize("method", ["iid", "gstools", "gaussian_filter"])
@pytest.mark.parametrize("dim, shape", [(30, (30,)), ((10, 12), (10, 12)), ((6, 7, 8), (6, 7, 8))])
def test_gen_field_shape(method, dim, shape):
    X = rc.gen_field(dim=dim, method=method, seed=0)
    assert X.shape == shape


@pytest.mark.parametrize("method", ["iid", "gstools", "gaussian_filter"])
def test_gen_field_seed_reproducible(method):
    X1 = rc.gen_field(dim=(20, 20), method=method, seed=1)
    X2 = rc.gen_field(dim=(20, 20), method=method, seed=1)
    X3 = rc.gen_field(dim=(20, 20), method=method, seed=2)
    np.testing.assert_array_equal(X1, X2)
    assert not np.allclose(X1, X3)


@pytest.mark.parametrize("distribution", ["gaussian", "t", "chisq"])
def test_gen_field_gaussian_filter_variance(distribution):
    # the smoothed field is rescaled so each point has variance `var`
    X = rc.gen_field(dim=(300, 300), var=2.0, method="gaussian_filter", fwhm=3,
                     distribution=distribution, seed=0)
    assert abs(X.mean()) < 0.1
    assert X.var() == pytest.approx(2.0, rel=0.1)


def test_gen_field_iid_variance():
    X = rc.gen_field(dim=(200, 200), var=3.0, method="iid", seed=0)
    assert X.var() == pytest.approx(3.0, rel=0.03)


def test_gen_field_truncation_and_square():
    X = rc.gen_field(dim=(50, 50), method="iid", seed=0, M_trunc=0.5)
    assert X.min() >= -0.5 and X.max() <= 0.5
    X_sq = rc.gen_field(dim=(50, 50), method="iid", seed=0, trans="square")
    X_raw = rc.gen_field(dim=(50, 50), method="iid", seed=0)
    np.testing.assert_allclose(X_sq, X_raw**2)


def test_gen_field_return_coordinate():
    X, coords = rc.gen_field(dim=(5, 6), method="iid", seed=0, return_coordinate=True)
    assert X.shape == (5, 6)
    np.testing.assert_array_equal(coords[0], [-2, -1, 0, 1, 2])
    np.testing.assert_array_equal(coords[1], [-3, -2, -1, 0, 1, 2])


def test_gen_field_unknown_method():
    with pytest.raises(ValueError):
        rc.gen_field(dim=10, method="nope")


def test_gen_field_t_requires_df_above_2():
    with pytest.raises(SystemExit):
        rc.gen_field(dim=10, method="gaussian_filter", distribution="t", noise_df=2)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
