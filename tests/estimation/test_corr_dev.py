"""Tests for rc.corr_dev. Run with: python -m pytest tests/estimation/test_corr_dev.py"""
import itertools
import numpy as np
import pytest

import rfcorr as rc


def test_corr_estimate_matches_pearson(fields_2d):
    X, Y = fields_2d
    rho_est, se, se0, runtime, M_tuple = rc.corr_dev(X, Y)
    assert rho_est == pytest.approx(np.corrcoef(X.ravel(), Y.ravel())[0, 1])
    assert se > 0 and se0 > 0 and runtime >= 0


def test_corr_iid_se_formula(fields_2d):
    X, Y = fields_2d
    rho_est, se, se0, _, M_tuple = rc.corr_dev(X, Y, se_method="iid")
    n = X.size
    assert se == pytest.approx((1 - rho_est**2) / np.sqrt(n))
    assert se0 == pytest.approx(1 / np.sqrt(n))
    assert M_tuple == (None,)


def test_corr_true_rho_plugin(fields_2d):
    X, Y = fields_2d
    _, se, se0, _, _ = rc.corr_dev(X, Y, se_method="iid", true_rho=0.8)
    assert se == pytest.approx((1 - 0.8**2) / np.sqrt(X.size))
    assert se0 == pytest.approx(1 / np.sqrt(X.size))


def test_corr_indep_se_scales_null_se(fields_2d):
    # the heuristic SE is the null SE times (1 - rho_plugin^2)
    X, Y = fields_2d
    rho_est, se, se0, _, _ = rc.corr_dev(X, Y, se_method="indep")
    assert se == pytest.approx((1 - rho_est**2) * se0)


def test_corr_gauss_null_sets_lag0_cross_correlation_to_zero(fields_2d):
    # RFCorr-Gauss-0 imposes rho = 0 at lag 0, where the cross-correlation equals rho_hat.
    # Lag 0 has weight 1, so the null variance drops by exactly rho_hat^2 / n.
    X, Y = fields_2d
    rho_est, _, se0, _, M_tuple = rc.corr_dev(X, Y, se_method="gauss", M="auto")
    X_s = (X - X.mean()) / X.std()
    Y_s = (Y - Y.mean()) / Y.std()
    est_unrestricted = rc.get_est_exp_ls("gauss", X_s, Y_s)
    se0_unrestricted, _ = rc.get_se(np.array(X.shape), est_unrestricted, 0, M="auto",
                                    se_method="gauss")
    assert se0**2 == pytest.approx(se0_unrestricted**2 - rho_est**2 / X.size)


@pytest.mark.parametrize("se_method", ["general", "gauss", "indep"])
@pytest.mark.parametrize("kernel", ["bartlett", "qs", "constant"])
def test_corr_spatial_methods_finite(fields_2d, se_method, kernel):
    X, Y = fields_2d
    _, se, se0, _, _ = rc.corr_dev(X, Y, se_method=se_method, kernel=kernel)
    assert np.isfinite(se0) and se0 > 0
    assert np.isnan(se) or se > 0


@pytest.mark.parametrize("se_method", ["general", "gauss"])
def test_corr_constant_kernel_matches_bartlett_at_lag_0(fields_2d, se_method):
    # with M = 0 both kernels put weight 1 on lag 0 only, so the SEs must agree
    X, Y = fields_2d
    se_const = rc.corr_dev(X, Y, se_method=se_method, kernel="constant", M=(0, 0))[1]
    se_bart = rc.corr_dev(X, Y, se_method=se_method, kernel="bartlett", M=(0, 0))[1]
    assert se_const == pytest.approx(se_bart)


def test_corr_hac_close_to_iid_on_iid_data():
    X, Y = rc.gen_corr_field(rho=0.3, dim=(100, 100), method="iid", seed=0)
    _, se_iid, _, _, _ = rc.corr_dev(X, Y, se_method="iid")
    for method in ["general", "gauss"]:
        _, se, _, _, _ = rc.corr_dev(X, Y, se_method=method)
        assert se == pytest.approx(se_iid, rel=0.15)


def test_corr_smooth_fields_inflate_se(fields_2d):
    # spatial dependence should make the HAC SE larger than the iid SE
    X, Y = fields_2d
    _, se_iid, _, _, _ = rc.corr_dev(X, Y, se_method="iid")
    _, se_hac, _, _, _ = rc.corr_dev(X, Y, se_method="general", M="auto")
    assert se_hac > se_iid


def test_corr_M_options():
    X, Y = rc.gen_corr_field(rho=0.3, dim=(27, 64), seed=0)
    # the rules use the side of a square with the same number of points, sqrt(27 * 64)
    n = 27 * 64
    np.testing.assert_array_equal(rc.corr_dev(X, Y, M="cubic_root")[4], np.full(2, np.ceil(n**0.15)))
    np.testing.assert_array_equal(rc.corr_dev(X, Y, M="linear")[4], np.full(2, np.floor(n**0.5 / 5)))
    np.testing.assert_array_equal(rc.corr_dev(X, Y, M="auto")[4], np.full(2, np.ceil(n**0.25)))
    np.testing.assert_array_equal(
        rc.corr_dev(X, Y, M="auto", constant=2)[4], np.full(2, np.ceil(n**0.25 * 2))
    )
    assert rc.corr_dev(X, Y, M=(2, 3))[4] == (2, 3)
    with pytest.raises(ValueError):
        rc.corr_dev(X, Y, M="quadratic")


def test_corr_optimal_rate_is_auto(fields_2d):
    # 'auto' is the earlier name of 'optimal_rate'
    X, Y = fields_2d
    for se_method in ["general", "gauss"]:
        a = rc.corr_dev(X, Y, M="optimal_rate", se_method=se_method, constant=1.5)
        b = rc.corr_dev(X, Y, M="auto", se_method=se_method, constant=1.5)
        assert a[:3] == b[:3]
        np.testing.assert_array_equal(a[4], b[4])


@pytest.mark.parametrize("side, n_dim", [(37, 2), (12, 3), (225, 1)])
@pytest.mark.parametrize("constant", [1, 1.4])
def test_corr_M_rules_on_square_grids(side, n_dim, constant):
    # on squares and cubes, side = n ** (1 / n_dim) is snapped to the integer side length,
    # so floating-point error in the root cannot move the lag across an integer
    dim = np.full(n_dim, side)
    X, Y = rc.gen_corr_field(rho=0.3, dim=tuple(dim) if n_dim > 1 else side, seed=0)
    rules = {
        "cubic_root": np.ceil(dim**0.3 * constant),
        "linear": np.floor(dim / 5 * constant),
        "optimal_rate": np.ceil(dim ** (n_dim / (n_dim + 2)) * constant),
    }
    for M, expected in rules.items():
        np.testing.assert_array_equal(rc.corr_dev(X, Y, M=M, constant=constant)[4], expected)


@pytest.mark.parametrize("dim", [(200,), (20, 20), (8, 9, 10)])
def test_corr_integer_M(dim):
    # an integer M is the same as passing it for every dimension
    X, Y = rc.gen_corr_field(rho=0.3, dim=dim, seed=0)
    out_int = rc.corr_dev(X, Y, M=4)
    out_tuple = rc.corr_dev(X, Y, M=(4,) * len(dim))
    assert out_int[4] == (4,) * len(dim)
    assert out_int[1] == pytest.approx(out_tuple[1])
    assert out_int[2] == pytest.approx(out_tuple[2])


def test_corr_shape_mismatch():
    with pytest.raises(ValueError):
        rc.corr_dev(np.zeros((3, 3)), np.zeros((3, 4)))


@pytest.mark.parametrize("old, new", list(rc.get_defaults()["se_method_aliases"].items()))
def test_corr_earlier_se_method_names_still_work(fields_2d, old, new):
    X, Y = fields_2d
    assert rc.corr_dev(X, Y, se_method=old)[:3] == rc.corr_dev(X, Y, se_method=new)[:3]
    assert rc.corr_dev(X, Y, se_method=new.upper())[:3] == rc.corr_dev(X, Y, se_method=new)[:3]



@pytest.mark.parametrize("se_method", ["general", "gauss", "indep", "iid"])
@pytest.mark.parametrize("dim", [(20, 20), (27, 64), (8, 9, 10)])
def test_corr_all_true_mask_matches_no_mask(se_method, dim):
    X, Y = rc.gen_corr_field(rho=0.4, dim=dim, len_scale=3, seed=0)
    out = rc.corr_dev(X, Y, se_method=se_method, M="auto")
    out_mask = rc.corr_dev(X, Y, se_method=se_method, M="auto", mask=np.ones(dim, dtype=bool))
    for i in (0, 1, 2):
        assert out_mask[i] == out[i]
    np.testing.assert_array_equal(out_mask[4], out[4])


def _brute_force_masked_hac_se(X, Y, mask, M, rho_plugin=None):
    # Omega = sum_h w(h) * (mean over valid pairs at lag h of T_{s+h} T_s' - mu mu')
    pts = [tuple(p) for p in np.argwhere(mask)]
    valid = set(pts)
    xv, yv = X[mask], Y[mask]
    rho = np.corrcoef(xv, yv)[0, 1] if rho_plugin is None else rho_plugin
    xs, ys = (X - xv.mean()) / xv.std(), (Y - yv.mean()) / yv.std()
    T = {p: np.array([xs[p] * ys[p], xs[p] ** 2, ys[p] ** 2]) for p in pts}
    mu = np.array([rho, 1, 1])
    omega = np.zeros((3, 3))
    for h in itertools.product(*[range(-m, m + 1) for m in M]):
        pairs = [(p, tuple(np.add(p, h))) for p in pts if tuple(np.add(p, h)) in valid]
        if pairs:
            w = np.prod([1 - abs(hi) / (m + 1) for hi, m in zip(h, M)])
            omega += w * (np.mean([np.outer(T[q], T[p]) for p, q in pairs], axis=0) - np.outer(mu, mu))
    a = np.array([1, -0.5 * rho, -0.5 * rho])
    return np.sqrt(a @ omega @ a / len(pts))


def test_corr_mask_matches_brute_force():
    X, Y = rc.gen_corr_field(rho=0.5, dim=(14, 11), len_scale=2, seed=3)
    ii, jj = np.indices(X.shape)
    # an L-shape: the diagonal lags across the corner have no valid pair
    mask = (ii < 4) | (jj < 3)
    rho, se, se0, _, _ = rc.corr_dev(X, Y, mask=mask, M=(3, 3))
    assert rho == pytest.approx(np.corrcoef(X[mask], Y[mask])[0, 1])
    assert se == pytest.approx(_brute_force_masked_hac_se(X, Y, mask, (3, 3)))
    assert se0 == pytest.approx(_brute_force_masked_hac_se(X, Y, mask, (3, 3), rho_plugin=0))


def test_corr_mask_ignores_values_outside():
    X, Y = rc.gen_corr_field(rho=0.5, dim=(30, 30), len_scale=2, seed=0)
    ii, jj = np.indices(X.shape)
    mask = (ii - 15) ** 2 + (jj - 15) ** 2 <= 100
    out = rc.corr_dev(X, Y, mask=mask, M="auto")
    out_nan = rc.corr_dev(np.where(mask, X, np.nan), np.where(mask, Y, 1e6), mask=mask, M="auto")
    for i in (0, 1, 2):
        assert out_nan[i] == pytest.approx(out[i])
    assert np.isfinite(out[1])


def test_corr_mask_shape_mismatch():
    with pytest.raises(ValueError):
        rc.corr_dev(np.zeros((3, 3)), np.zeros((3, 3)), mask=np.ones((3, 4), dtype=bool))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
