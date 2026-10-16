"""Unit tests for src.py. Run from this folder with: python -m pytest test_src.py"""

import itertools

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

import src


# ---------------------------------------------------------------------------
# gen_field
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("method", ["iid", "gstools", "gaussian_filter"])
@pytest.mark.parametrize("dim, shape", [(30, (30,)), ((10, 12), (10, 12)), ((6, 7, 8), (6, 7, 8))])
def test_gen_field_shape(method, dim, shape):
    X = src.gen_field(dim=dim, method=method, seed=0)
    assert X.shape == shape


@pytest.mark.parametrize("method", ["iid", "gstools", "gaussian_filter"])
def test_gen_field_seed_reproducible(method):
    X1 = src.gen_field(dim=(20, 20), method=method, seed=1)
    X2 = src.gen_field(dim=(20, 20), method=method, seed=1)
    X3 = src.gen_field(dim=(20, 20), method=method, seed=2)
    np.testing.assert_array_equal(X1, X2)
    assert not np.allclose(X1, X3)


@pytest.mark.parametrize("distribution", ["gaussian", "t", "chisq"])
def test_gen_field_gaussian_filter_variance(distribution):
    # the smoothed field is rescaled so each point has variance `var`
    X = src.gen_field(dim=(300, 300), var=2.0, method="gaussian_filter", fwhm=3,
                      distribution=distribution, seed=0)
    assert abs(X.mean()) < 0.1
    assert X.var() == pytest.approx(2.0, rel=0.1)


def test_gen_field_iid_variance():
    X = src.gen_field(dim=(200, 200), var=3.0, method="iid", seed=0)
    assert X.var() == pytest.approx(3.0, rel=0.03)


def test_gen_field_truncation_and_square():
    X = src.gen_field(dim=(50, 50), method="iid", seed=0, M_trunc=0.5)
    assert X.min() >= -0.5 and X.max() <= 0.5
    X_sq = src.gen_field(dim=(50, 50), method="iid", seed=0, trans="square")
    X_raw = src.gen_field(dim=(50, 50), method="iid", seed=0)
    np.testing.assert_allclose(X_sq, X_raw**2)


def test_gen_field_return_coordinate():
    X, coords = src.gen_field(dim=(5, 6), method="iid", seed=0, return_coordinate=True)
    assert X.shape == (5, 6)
    np.testing.assert_array_equal(coords[0], [-2, -1, 0, 1, 2])
    np.testing.assert_array_equal(coords[1], [-3, -2, -1, 0, 1, 2])


def test_gen_field_unknown_method():
    with pytest.raises(ValueError):
        src.gen_field(dim=10, method="nope")


def test_gen_field_t_requires_df_above_2():
    with pytest.raises(SystemExit):
        src.gen_field(dim=10, method="gaussian_filter", distribution="t", noise_df=2)


# ---------------------------------------------------------------------------
# gen_corr_field
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dim", [300, (40, 50), (10, 12, 14)])
def test_gen_corr_field_shape_and_seed(dim):
    X1, Y1 = src.gen_corr_field(rho=0.3, dim=dim, seed=5)
    X2, Y2 = src.gen_corr_field(rho=0.3, dim=dim, seed=5)
    expected = (dim,) if isinstance(dim, int) else dim
    assert X1.shape == Y1.shape == expected
    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(Y1, Y2)


@pytest.mark.parametrize("rho", [-0.9, 0.0, 0.4, 0.9])
def test_gen_corr_field_iid_correlation(rho):
    X, Y = src.gen_corr_field(rho=rho, dim=(200, 200), method="iid", seed=0)
    # SE of the sample correlation is about (1 - rho^2)/sqrt(n) <= 0.005
    assert np.corrcoef(X.ravel(), Y.ravel())[0, 1] == pytest.approx(rho, abs=0.03)
    assert Y.var() == pytest.approx(1.0, rel=0.05)


def test_gen_corr_field_smooth_correlation():
    X, Y = src.gen_corr_field(rho=0.5, dim=(300, 300), fwhm=3, seed=0)
    assert np.corrcoef(X.ravel(), Y.ravel())[0, 1] == pytest.approx(0.5, abs=0.05)


def test_gen_corr_field_rho_one_gives_identical_fields():
    X, Y = src.gen_corr_field(rho=1.0, dim=(20, 20), seed=0)
    np.testing.assert_allclose(X, Y)


def test_gen_corr_field_truncation_and_coordinates():
    X, Y, coords = src.gen_corr_field(rho=0.5, dim=(9, 10), seed=0, M_trunc=0.3,
                                      return_coordinate=True)
    assert np.abs(X).max() <= 0.3 and np.abs(Y).max() <= 0.3
    assert [len(c) for c in coords] == [9, 10]


# ---------------------------------------------------------------------------
# gaussian_filter_acf and the shift design (uncorrelated but dependent fields)
# ---------------------------------------------------------------------------
def test_gaussian_filter_acf_values():
    r = [src.gaussian_filter_acf((1, 0), f) for f in [1, 2, 3, 4]]
    np.testing.assert_allclose(r, [0.124, 0.705, 0.857, 0.917], atol=1e-3)
    assert src.gaussian_filter_acf((0, 0), 3) == pytest.approx(1.0)
    # symmetric in the lag, and separable across axes
    assert src.gaussian_filter_acf((-1, 0), 3) == pytest.approx(src.gaussian_filter_acf((1, 0), 3))
    assert src.gaussian_filter_acf((1, 1), 3) == pytest.approx(src.gaussian_filter_acf(1, 3) ** 2)


def test_gaussian_filter_acf_matches_simulated_fields():
    Z = src.gen_field(dim=(300, 300), method="gaussian_filter", fwhm=3, seed=0)
    for h, (a, b) in {(1, 0): (Z[:-1], Z[1:]), (0, 2): (Z[:, :-2], Z[:, 2:])}.items():
        assert np.corrcoef(a.ravel(), b.ravel())[0, 1] == pytest.approx(
            src.gaussian_filter_acf(h, 3), abs=0.02)


SHIFT_MODELS = ["shift", "symmetric_shift"]


def _mean_corr(rho, trans, dep_model="shift", m=150, n=40, **kw):
    return np.mean([
        np.corrcoef(*[f.ravel() for f in src.gen_corr_field(
            rho, dim=(n, n), fwhm=3, fwhm_ratio=2, seed=s, trans=trans, dep_model=dep_model,
            **kw)])[0, 1]
        for s in range(m)
    ])


@pytest.mark.parametrize("dep_model", SHIFT_MODELS)
@pytest.mark.parametrize("trans", [None, "square"])
@pytest.mark.parametrize("rho", [0.0, 0.4])
def test_shift_design_keeps_target_correlation(rho, trans, dep_model):
    # the population r_h (and r_2h) make the correlation exactly rho, in both settings
    assert _mean_corr(rho, trans, dep_model) == pytest.approx(rho, abs=0.015)


@pytest.mark.parametrize("trans", [None, "square"])
def test_symmetric_shift_cross_correlation(trans):
    # with rho = 0, corr(X_s, Y_{s+h}) = sd(D) / sd(U) = sqrt((1 + r_2h) / 2 - r_h^2), and the
    # cross-correlation is even in the lag: corr(X_s, Y_{s-h}) is the same
    r, r2 = src.gaussian_filter_acf((1, 0), 3), src.gaussian_filter_acf((2, 0), 3)
    if trans == "square":
        r, r2 = r**2, r2**2
    plus, minus = [], []
    for s in range(50):
        X, Y = src.gen_corr_field(0, dim=(60, 60), fwhm=3, seed=s, trans=trans,
                                  dep_model="symmetric_shift")
        plus.append(np.corrcoef(X[:-1].ravel(), Y[1:].ravel())[0, 1])    # (X_s, Y_{s+h})
        minus.append(np.corrcoef(X[1:].ravel(), Y[:-1].ravel())[0, 1])   # (X_s, Y_{s-h})
    expected = np.sqrt((1 + r2) / 2 - r**2)
    assert np.mean(plus) == pytest.approx(expected, abs=0.03)
    assert np.mean(minus) == pytest.approx(expected, abs=0.03)


@pytest.mark.parametrize("trans", [None, "square"])
def test_shift_design_is_dependent(trans):
    # with rho = 0, Y_{s-h} contains U_s, so it is strongly correlated with X_s
    r = src.gaussian_filter_acf((1, 0), 3)
    r_U = r if trans is None else r**2
    vals = []
    for s in range(50):
        X, Y = src.gen_corr_field(0, dim=(60, 60), fwhm=3, seed=s, trans=trans, dep_model="shift")
        vals.append(np.corrcoef(X[1:].ravel(), Y[:-1].ravel())[0, 1])
    assert np.mean(vals) == pytest.approx(np.sqrt(1 - r_U**2), abs=0.03)


@pytest.mark.parametrize("dep_model", SHIFT_MODELS)
def test_shift_design_is_not_degenerate(dep_model):
    # unlike a central difference, the sample correlation keeps the usual sqrt(N) scaling
    sd_scaled = []
    for n in [30, 60]:
        rhos = [np.corrcoef(*[f.ravel() for f in src.gen_corr_field(
            0, dim=(n, n), fwhm=3, seed=s, dep_model=dep_model)])[0, 1] for s in range(200)]
        sd_scaled.append(np.std(rhos) * n)
    assert sd_scaled[0] > 0.5
    assert sd_scaled[1] / sd_scaled[0] == pytest.approx(1, abs=0.3)


@pytest.mark.parametrize("dep_model", SHIFT_MODELS)
@pytest.mark.parametrize("lag, dim", [(2, (20, 25)), ((0, -1), (20, 25)), ((1, 1), (20, 25)), (1, 30)])
def test_shift_design_lags_and_shapes(lag, dim, dep_model):
    X, Y = src.gen_corr_field(0.3, dim=dim, fwhm=2, seed=0, dep_model=dep_model, shift_lag=lag)
    expected = (dim,) if isinstance(dim, int) else dim
    assert X.shape == Y.shape == expected


@pytest.mark.parametrize("dep_model", SHIFT_MODELS)
def test_shift_design_invalid_settings(dep_model):
    with pytest.raises(ValueError):
        src.gen_corr_field(0, dim=(20, 20), method="iid", dep_model=dep_model)
    with pytest.raises(ValueError):
        src.gen_corr_field(0, dim=(20, 20), distribution="t", trans="square", dep_model=dep_model)
    with pytest.raises(ValueError):
        src.gen_corr_field(0, dim=(20, 20), dep_model=dep_model, shift_lag=0)
    with pytest.raises(ValueError):
        src.gen_corr_field(0, dim=(20, 20), dep_model=dep_model, shift_lag=(1, 0, 0))


# ---------------------------------------------------------------------------
# est_exp_X0Yh
# ---------------------------------------------------------------------------
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
    np.testing.assert_allclose(src.est_exp_X0Yh(X, Y), _brute_force_lag_mean(X, Y))


def test_est_exp_X0Yh_counts_lags_and_symmetry():
    rng = np.random.default_rng(1)
    X, Y = rng.normal(size=(4, 6)), rng.normal(size=(4, 6))
    S, C, E, lag_ls = src.est_exp_X0Yh(X, Y, return_counts_lags=True)
    assert E.shape == C.shape == S.shape == (7, 11)
    np.testing.assert_array_equal(lag_ls[0], np.arange(-3, 4))
    np.testing.assert_array_equal(lag_ls[1], np.arange(-5, 6))
    # zero lag uses all n pairs and equals mean(X * Y)
    assert C[3, 5] == X.size
    assert E[3, 5] == pytest.approx(np.mean(X * Y))
    # swapping X and Y reverses the lags
    np.testing.assert_allclose(src.est_exp_X0Yh(Y, X), E[::-1, ::-1])


def test_est_exp_X0Yh_shape_mismatch():
    with pytest.raises(ValueError):
        src.est_exp_X0Yh(np.zeros((3, 3)), np.zeros((3, 4)))


# ---------------------------------------------------------------------------
# kernel / kernel_weights
# ---------------------------------------------------------------------------
def test_kernel_bartlett():
    x = np.array([-2, -1, -0.5, 0, 0.25, 1, 3])
    np.testing.assert_allclose(src.kernel(x, "bartlett"), [0, 0, 0.5, 1, 0.75, 0, 0])


def test_kernel_qs():
    x = np.array([-2.0, -0.3, 0.0, 1e-10, 0.3, 2.0])
    k = src.kernel(x, "QS")
    assert k[2] == 1.0 and k[3] == 1.0
    np.testing.assert_allclose(k, k[::-1])  # symmetric
    # continuous at 0
    assert src.kernel(np.array([1e-4]), "qs")[0] == pytest.approx(1.0, abs=1e-6)
    # closed form at x = 1
    z = 6 * np.pi / 5
    expected = 25 / (12 * np.pi**2) * (np.sin(z) / z - np.cos(z))
    assert src.kernel(np.array([1.0]), "qs")[0] == pytest.approx(expected)


def test_kernel_constant_and_unknown():
    assert src.kernel(np.array([5.0]), "constant") == 1
    with pytest.raises(ValueError):
        src.kernel(np.array([0.0]), "triangle")


def test_kernel_weights_bartlett():
    x = np.arange(-5, 6)
    w = src.kernel_weights(x, "bartlett", bandwidth=None, M=3)
    # Bartlett uses bandwidth M + 1, so weights are 1 - |h|/4 up to |h| = 3
    expected = np.where(np.abs(x) <= 3, 1 - np.abs(x) / 4, 0)
    np.testing.assert_allclose(w, expected)


def test_kernel_weights_constant_and_qs():
    x = np.arange(-4, 5)
    w_const = src.kernel_weights(x, "constant", None, M=2)
    assert w_const.dtype == float
    np.testing.assert_array_equal(w_const, np.abs(x) <= 2)
    w = src.kernel_weights(x, "qs", bandwidth=2, M=3)
    assert w[4] == 1.0
    assert np.all(w[np.abs(x) > 3] == 0)
    np.testing.assert_allclose(w, w[::-1])


# ---------------------------------------------------------------------------
# corr / get_se
# ---------------------------------------------------------------------------
@pytest.fixture
def fields_2d():
    return src.gen_corr_field(rho=0.4, dim=(40, 50), fwhm=3, seed=0)


def test_corr_estimate_matches_pearson(fields_2d):
    X, Y = fields_2d
    rho_est, se, se0, runtime, M_tuple = src.corr(X, Y)
    assert rho_est == pytest.approx(np.corrcoef(X.ravel(), Y.ravel())[0, 1])
    assert se > 0 and se0 > 0 and runtime >= 0


def test_corr_iid_se_formula(fields_2d):
    X, Y = fields_2d
    rho_est, se, se0, _, M_tuple = src.corr(X, Y, se_method="iid")
    n = X.size
    assert se == pytest.approx((1 - rho_est**2) / np.sqrt(n))
    assert se0 == pytest.approx(1 / np.sqrt(n))
    assert M_tuple == (None,)


def test_corr_true_rho_plugin(fields_2d):
    X, Y = fields_2d
    _, se, se0, _, _ = src.corr(X, Y, se_method="iid", true_rho=0.8)
    assert se == pytest.approx((1 - 0.8**2) / np.sqrt(X.size))
    assert se0 == pytest.approx(1 / np.sqrt(X.size))


def test_corr_indep_se_scales_null_se(fields_2d):
    # the heuristic SE is the null SE times (1 - rho_plugin^2)
    X, Y = fields_2d
    rho_est, se, se0, _, _ = src.corr(X, Y, se_method="indep")
    assert se == pytest.approx((1 - rho_est**2) * se0)


def test_corr_gauss_null_sets_lag0_cross_correlation_to_zero(fields_2d):
    # RFCorr-Gauss-0 imposes rho = 0 at lag 0, where the cross-correlation equals rho_hat.
    # Lag 0 has weight 1, so the null variance drops by exactly rho_hat^2 / n.
    X, Y = fields_2d
    rho_est, _, se0, _, M_tuple = src.corr(X, Y, se_method="gauss", M="auto")
    X_s = (X - X.mean()) / X.std()
    Y_s = (Y - Y.mean()) / Y.std()
    est_unrestricted = src.get_est_exp_ls("gauss", X_s, Y_s)
    se0_unrestricted, _ = src.get_se(np.array(X.shape), est_unrestricted, 0, M="auto",
                                     se_method="gauss")
    assert se0**2 == pytest.approx(se0_unrestricted**2 - rho_est**2 / X.size)


def test_get_est_exp_ls_impose_null_only_changes_lag0():
    rng = np.random.default_rng(3)
    X, Y = rng.normal(size=(6, 7)), rng.normal(size=(6, 7))
    full = src.get_est_exp_ls("gauss", X, Y)
    null = src.get_est_exp_ls("gauss", X, Y, impose_null=True)
    center = (5, 6)
    for a, b in zip(full, null):
        diff = np.argwhere(~np.isclose(a, b))
        assert all(tuple(d) == center for d in diff)
    # independence and general methods ignore the option
    for method in ["indep", "general"]:
        for a, b in zip(src.get_est_exp_ls(method, X, Y),
                        src.get_est_exp_ls(method, X, Y, impose_null=True)):
            np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("se_method", ["general", "gauss", "indep"])
@pytest.mark.parametrize("kernel", ["bartlett", "qs", "constant"])
def test_corr_spatial_methods_finite(fields_2d, se_method, kernel):
    X, Y = fields_2d
    _, se, se0, _, _ = src.corr(X, Y, se_method=se_method, kernel=kernel)
    assert np.isfinite(se0) and se0 > 0
    assert np.isnan(se) or se > 0


@pytest.mark.parametrize("se_method", ["general", "gauss"])
def test_corr_constant_kernel_matches_bartlett_at_lag_0(fields_2d, se_method):
    # with M = 0 both kernels put weight 1 on lag 0 only, so the SEs must agree
    X, Y = fields_2d
    se_const = src.corr(X, Y, se_method=se_method, kernel="constant", M=(0, 0))[1]
    se_bart = src.corr(X, Y, se_method=se_method, kernel="bartlett", M=(0, 0))[1]
    assert se_const == pytest.approx(se_bart)


def test_corr_hac_close_to_iid_on_iid_data():
    X, Y = src.gen_corr_field(rho=0.3, dim=(100, 100), method="iid", seed=0)
    _, se_iid, _, _, _ = src.corr(X, Y, se_method="iid")
    for method in ["general", "gauss"]:
        _, se, _, _, _ = src.corr(X, Y, se_method=method)
        assert se == pytest.approx(se_iid, rel=0.15)


def test_corr_smooth_fields_inflate_se(fields_2d):
    # spatial dependence should make the HAC SE larger than the iid SE
    X, Y = fields_2d
    _, se_iid, _, _, _ = src.corr(X, Y, se_method="iid")
    _, se_hac, _, _, _ = src.corr(X, Y, se_method="general", M="auto")
    assert se_hac > se_iid


def test_corr_M_options():
    X, Y = src.gen_corr_field(rho=0.3, dim=(27, 64), seed=0)
    # the rules use the side of a square with the same number of points, sqrt(27 * 64)
    n = 27 * 64
    np.testing.assert_array_equal(src.corr(X, Y, M="cubic_root")[4], np.full(2, np.ceil(n**0.15)))
    np.testing.assert_array_equal(src.corr(X, Y, M="linear")[4], np.full(2, np.floor(n**0.5 / 5)))
    np.testing.assert_array_equal(src.corr(X, Y, M="auto")[4], np.full(2, np.ceil(n**0.25)))
    np.testing.assert_array_equal(
        src.corr(X, Y, M="auto", constant=2)[4], np.full(2, np.ceil(n**0.25 * 2))
    )
    assert src.corr(X, Y, M=(2, 3))[4] == (2, 3)
    with pytest.raises(ValueError):
        src.corr(X, Y, M="quadratic")


@pytest.mark.parametrize("side, n_dim", [(100, 2), (50, 2), (37, 2), (32, 3), (12, 3), (225, 1)])
@pytest.mark.parametrize("constant", [0.5, 1, 1.4, 2])
def test_corr_M_rules_match_side_length_rules_on_cubes(side, n_dim, constant):
    # on cubes the size-based rules equal the old per-side rules
    dim = np.full(n_dim, side)
    X, Y = src.gen_corr_field(rho=0.3, dim=tuple(dim) if n_dim > 1 else side, seed=0)
    old_rules = {
        "cubic_root": np.ceil(dim**0.3 * constant),
        "linear": np.floor(dim / 5 * constant),
        "auto": np.ceil(dim ** (n_dim / (n_dim + 2)) * constant),
    }
    for M, expected in old_rules.items():
        np.testing.assert_array_equal(src.corr(X, Y, M=M, constant=constant)[4], expected)


def test_corr_integer_M_1d():
    X, Y = src.gen_corr_field(rho=0.3, dim=200, seed=0)
    assert src.corr(X, Y, M=4)[4] == (4,)


@pytest.mark.parametrize("dim", [(20, 20), (8, 9, 10)])
def test_corr_integer_M_multi_dim(dim):
    # an integer M is the same as passing it for every dimension
    X, Y = src.gen_corr_field(rho=0.3, dim=dim, seed=0)
    out_int = src.corr(X, Y, M=4)
    out_tuple = src.corr(X, Y, M=(4,) * len(dim))
    assert out_int[4] == (4,) * len(dim)
    assert out_int[1] == pytest.approx(out_tuple[1])
    assert out_int[2] == pytest.approx(out_tuple[2])


def test_corr_shape_mismatch():
    with pytest.raises(ValueError):
        src.corr(np.zeros((3, 3)), np.zeros((3, 4)))


@pytest.mark.parametrize("old, new", list(src.SE_METHOD_ALIASES.items()))
def test_corr_earlier_se_method_names_still_work(fields_2d, old, new):
    X, Y = fields_2d
    assert src.corr(X, Y, se_method=old)[:3] == src.corr(X, Y, se_method=new)[:3]
    assert src.corr(X, Y, se_method=new.upper())[:3] == src.corr(X, Y, se_method=new)[:3]


def test_corr_unknown_se_method():
    with pytest.raises(ValueError):
        src.corr(np.zeros((3, 3)), np.zeros((3, 3)), se_method="newey")


# ---------------------------------------------------------------------------
# corr with a mask
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("se_method", ["general", "gauss", "indep", "iid"])
@pytest.mark.parametrize("dim", [(20, 20), (27, 64), (8, 9, 10)])
def test_corr_all_true_mask_matches_no_mask(se_method, dim):
    X, Y = src.gen_corr_field(rho=0.4, dim=dim, len_scale=3, seed=0)
    out = src.corr(X, Y, se_method=se_method, M="auto")
    out_mask = src.corr(X, Y, se_method=se_method, M="auto", mask=np.ones(dim, dtype=bool))
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
    X, Y = src.gen_corr_field(rho=0.5, dim=(14, 11), len_scale=2, seed=3)
    ii, jj = np.indices(X.shape)
    # an L-shape: the diagonal lags across the corner have no valid pair
    mask = (ii < 4) | (jj < 3)
    rho, se, se0, _, _ = src.corr(X, Y, mask=mask, M=(3, 3))
    assert rho == pytest.approx(np.corrcoef(X[mask], Y[mask])[0, 1])
    assert se == pytest.approx(_brute_force_masked_hac_se(X, Y, mask, (3, 3)))
    assert se0 == pytest.approx(_brute_force_masked_hac_se(X, Y, mask, (3, 3), rho_plugin=0))


def test_corr_mask_ignores_values_outside():
    X, Y = src.gen_corr_field(rho=0.5, dim=(30, 30), len_scale=2, seed=0)
    ii, jj = np.indices(X.shape)
    mask = (ii - 15) ** 2 + (jj - 15) ** 2 <= 100
    out = src.corr(X, Y, mask=mask, M="auto")
    out_nan = src.corr(np.where(mask, X, np.nan), np.where(mask, Y, 1e6), mask=mask, M="auto")
    for i in (0, 1, 2):
        assert out_nan[i] == pytest.approx(out[i])
    assert np.isfinite(out[1])


def test_corr_mask_shape_mismatch():
    with pytest.raises(ValueError):
        src.corr(np.zeros((3, 3)), np.zeros((3, 3)), mask=np.ones((3, 4), dtype=bool))


def test_get_se_negative_variance_warns():
    # an omega that makes vec @ omega @ vec negative
    dim = np.array([5])
    est_exp_ls = [-np.ones(9)] + [np.zeros(9)] * 8
    with pytest.warns(UserWarning, match="negative"):
        se, _ = src.get_se(dim, est_exp_ls, rho_plugin=0, M=2, se_method="gauss")
    assert np.isnan(se)


def test_get_se_radial_bartlett():
    # only lag (0, 0) and the diagonal lag (1, 1) carry covariance, in the (1, 1) entry of omega
    dim = np.array([4, 4])
    cov = np.zeros((7, 7))
    cov[3, 3] = 1.0
    cov[4, 4] = 0.5
    est_exp_ls = [cov] + [np.zeros((7, 7))] * 8
    se_prod, _ = src.get_se(dim, est_exp_ls, rho_plugin=0, M=1, se_method="gauss")
    se_rad, _ = src.get_se(dim, est_exp_ls, rho_plugin=0, M=1, se_method="gauss",
                           kernel="bartlett_radial")
    # with M = 1 the weight at lag (1, 1) is (1/2)^2 for the product kernel and
    # 1 - sqrt(2)/2 for the radial kernel
    assert se_prod == pytest.approx(np.sqrt((1 + 0.5 * 0.25) / 16))
    assert se_rad == pytest.approx(np.sqrt((1 + 0.5 * (1 - np.sqrt(2) / 2)) / 16))
    # in 1D the radial and product Bartlett weights coincide
    est_1d = [np.array([0.2, 0.5, 1.0, 0.5, 0.2])] + [np.zeros(5)] * 8
    se1, _ = src.get_se(np.array([3]), est_1d, rho_plugin=0, M=2, se_method="gauss")
    se2, _ = src.get_se(np.array([3]), est_1d, rho_plugin=0, M=2, se_method="gauss",
                        kernel="bartlett_radial")
    assert se1 == pytest.approx(se2)


# ---------------------------------------------------------------------------
# confidence_interval / pvalue
# ---------------------------------------------------------------------------
def test_confidence_interval_wald():
    lower, upper = src.confidence_interval(0.3, 0.1, fisher_trans=False)
    z = norm.ppf(0.975)
    assert lower == pytest.approx(0.3 - z * 0.1)
    assert upper == pytest.approx(0.3 + z * 0.1)
    assert isinstance(lower, float)


def test_confidence_interval_wald_is_clipped():
    lower, upper = src.confidence_interval(0.95, 0.2, fisher_trans=False)
    assert upper == 1.0
    lower, upper = src.confidence_interval(-0.95, 0.2, fisher_trans=False)
    assert lower == -1.0


def test_confidence_interval_fisher():
    rho, se = 0.6, 0.1
    lower, upper = src.confidence_interval(rho, se)
    z = norm.ppf(0.975)
    se_F = se / (1 - rho**2)
    assert lower == pytest.approx(np.tanh(np.arctanh(rho) - z * se_F))
    assert upper == pytest.approx(np.tanh(np.arctanh(rho) + z * se_F))
    assert -1 < lower < rho < upper < 1
    # the interval is symmetric on the Fisher scale, not on the rho scale
    assert np.arctanh(upper) - np.arctanh(rho) == pytest.approx(np.arctanh(rho) - np.arctanh(lower))


def test_confidence_interval_alpha_and_vectorized():
    narrow = src.confidence_interval(0.2, 0.1, alpha=0.2)
    wide = src.confidence_interval(0.2, 0.1, alpha=0.01)
    assert wide[0] < narrow[0] and wide[1] > narrow[1]
    rho = np.array([-0.5, 0.0, 0.5])
    lower, upper = src.confidence_interval(rho, np.full(3, 0.1))
    assert lower.shape == upper.shape == (3,)
    np.testing.assert_allclose(lower, -upper[::-1])


def test_pvalue():
    assert src.pvalue(0.0, 0.1) == pytest.approx(1.0)
    assert src.pvalue(0.2, 0.1, fisher_trans=False) == pytest.approx(2 * (1 - norm.cdf(2)))
    assert src.pvalue(0.2, 0.1) == pytest.approx(2 * (1 - norm.cdf(np.arctanh(0.2) / 0.1)))
    assert src.pvalue(-0.2, 0.1) == pytest.approx(src.pvalue(0.2, 0.1))
    # larger |rho| gives a smaller p value
    p = src.pvalue(np.array([0.1, 0.3, 0.5]), np.full(3, 0.1))
    assert p.shape == (3,) and np.all(np.diff(p) < 0)


# ---------------------------------------------------------------------------
# results_summary / sim_single / sim_multiple
# ---------------------------------------------------------------------------
def test_results_summary_no_group():
    results = pd.DataFrame({
        "bias": [0.1, -0.1], "est": [0.5, 0.3], "se": [0.1, 0.3],
        "cover": [1, 0], "cover_raw": [1, 1], "reject": [1, 1],
        "reject_raw": [0, 1], "runtime": [1.0, 3.0],
    })
    out = src.results_summary(results)
    assert out.shape[0] == 1
    row = out.iloc[0]
    assert row["bias_mean"] == pytest.approx(0.0)
    assert row["est_sd"] == pytest.approx(np.std([0.5, 0.3], ddof=1))
    assert row["se_mean"] == pytest.approx(0.2)
    assert row["cover_rate"] == 0.5 and row["cover_rate_raw"] == 1.0
    assert row["power"] == 1.0 and row["power_raw"] == 0.5
    assert row["runtime_mean"] == 2.0


@pytest.fixture
def method_df():
    return pd.DataFrame({
        "use_true_rho": [False, False, True],
        "method": ["iid", "general", "gauss"],
        "kernel": ["bartlett"] * 3,
        "M": ["auto"] * 3,
        "b_qs": [None] * 3,
        "constant": [1] * 3,
    })


def _run_sim_single(method_df, **kwargs):
    return src.sim_single(method_df, rho=0.4, m_sim=3, nx=20, n_dim=2,
                          gen_method="gaussian_filter", seed=0, **kwargs)


def test_sim_single_raw_results(method_df):
    raw = _run_sim_single(method_df, return_summary=False)
    assert len(raw) == 3 * len(method_df)
    for col in ["sim_id", "method_id", "est", "se", "se_under_null", "lower", "upper",
                "p", "p_raw", "bias", "cover", "cover_raw", "reject", "reject_raw",
                "rho", "nx", "n_dim", "method", "kernel"]:
        assert col in raw.columns
    np.testing.assert_allclose(raw["bias"], raw["est"] - 0.4)
    assert set(raw["cover"]).issubset({0, 1}) and set(raw["reject"]).issubset({0, 1})
    # the field is the same for every method within a replicate
    assert raw.groupby("sim_id")["est"].nunique().eq(1).all()


def test_sim_single_seed_reproducible(method_df):
    a = _run_sim_single(method_df, return_summary=False).drop(columns="runtime")
    b = _run_sim_single(method_df, return_summary=False).drop(columns="runtime")
    pd.testing.assert_frame_equal(a, b)


def test_sim_single_summary(method_df):
    summary = _run_sim_single(method_df, return_summary=True)
    assert len(summary) == len(method_df)
    assert list(summary["method"]) == ["iid", "general", "gauss"]
    assert summary["cover_rate"].between(0, 1).all()


@pytest.mark.parametrize("n_dim, expected", [(1, 20), (2, 400), (3, 8000)])
def test_sim_single_dimensions(method_df, n_dim, expected):
    raw = src.sim_single(method_df.iloc[:1], rho=0.0, m_sim=1, nx=20, n_dim=n_dim,
                         gen_method="iid", seed=0, return_summary=False)
    # under iid SE the null SE is 1/sqrt(n), which pins down the field size
    assert raw["se_under_null"].iloc[0] == pytest.approx(1 / np.sqrt(expected))


def test_sim_multiple(method_df, capsys):
    setting_df = pd.DataFrame({
        "rho": [0.0, 0.5], "nx": [15, 20], "n_dim": [1, 2], "len_scale": [None, None],
        "gen_method": ["gaussian_filter"] * 2, "fwhm": [2, 3],
        "distribution": ["gaussian"] * 2, "noise_df": [None, None], "trans": [None, "square"],
    })
    out = src.sim_multiple(setting_df, method_df, m_sim=2, return_summary=True)
    assert out.columns[0] == "setting_id"
    assert len(out) == len(setting_df) * len(method_df)
    assert list(out.groupby("setting_id")["rho"].first()) == [0.0, 0.5]
    assert "remaining settings" in capsys.readouterr().out
    # verbose=False prints nothing
    src.sim_multiple(setting_df, method_df, m_sim=2, verbose=False)
    assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# masks in simulations
# ---------------------------------------------------------------------------
def test_gen_mask_ellipse():
    mask = src.gen_mask((40, 60), axes=(1, 0.5))
    assert mask.shape == (40, 60) and mask.dtype == bool
    # centered and symmetric
    np.testing.assert_array_equal(mask, mask[::-1, ::-1])
    # area close to pi * a * b with semi-axes 20 and 15
    assert mask.sum() == pytest.approx(np.pi * 20 * 15, rel=0.05)
    # inscribed ellipse touches the middle of each side
    full = src.gen_mask((41, 41))
    assert full[0, 20] and full[20, 0] and not full[0, 0]
    assert src.gen_mask((9, 9, 9), axes=0.5).sum() < src.gen_mask((9, 9, 9)).sum()
    with pytest.raises(ValueError):
        src.gen_mask((10, 10), axes=(1, 1, 1))
    with pytest.raises(ValueError):
        src.gen_mask((10, 10), shape="square")


def test_sim_single_with_mask(method_df):
    raw = _run_sim_single(method_df, return_summary=False, mask="ellipse", mask_axes=(1, 0.5))
    mask = src.gen_mask((20, 20), axes=(1, 0.5))
    assert (raw["n_obs"] == mask.sum()).all() and (raw["mask"] == "ellipse").all()
    assert raw["mask_axes"].iloc[0] == (1, 0.5)
    # the same mask is used in every replicate: recompute replicate 0 by hand
    rng = np.random.default_rng(0)
    X, Y = src.gen_corr_field(0.4, (20, 20), 1.0, 10.0, seed=rng.integers(0, 1e9),
                              method="gaussian_filter", fwhm=2)
    row = raw[(raw["sim_id"] == 0) & (raw["method"] == "general")].iloc[0]
    est, se, se0, _, _ = src.corr(X, Y, se_method="general", M="auto", mask=mask)
    assert (row["est"], row["se"], row["se_under_null"]) == pytest.approx((est, se, se0))


def test_sim_single_no_mask_unchanged(method_df):
    a = _run_sim_single(method_df, return_summary=False).drop(columns="runtime")
    b = _run_sim_single(method_df, return_summary=False, mask=np.nan).drop(columns="runtime")
    cols = [c for c in a.columns if c not in ("mask", "mask_axes")]
    pd.testing.assert_frame_equal(a[cols], b[cols])
    assert (a["n_obs"] == 400).all()


def test_sim_multiple_with_and_without_mask_column(method_df):
    setting_df = pd.DataFrame({
        "rho": [0.0, 0.5], "nx": [15, 20], "n_dim": [2, 2], "len_scale": [None, None],
        "gen_method": ["gaussian_filter"] * 2, "fwhm": [2, 3],
        "distribution": ["gaussian"] * 2, "noise_df": [None, None], "trans": [None, None],
        "mask": [None, "ellipse"], "mask_axes": [None, (1, 0.6)],
    })
    out = src.sim_multiple(setting_df, method_df, m_sim=2, return_summary=True)
    n_obs = out.groupby("setting_id")["n_obs"].first()
    assert n_obs[0] == 225 and n_obs[1] == src.gen_mask((20, 20), axes=(1, 0.6)).sum()


# ---------------------------------------------------------------------------
# fwhm_ratio in simulations
# ---------------------------------------------------------------------------
def test_sim_single_fwhm_ratio(method_df):
    raw = _run_sim_single(method_df, return_summary=False, fwhm_ratio=2)
    assert (raw["fwhm_ratio"] == 2).all()
    # replicate 0 by hand with the same seed stream
    rng = np.random.default_rng(0)
    X, Y = src.gen_corr_field(0.4, (20, 20), 1.0, 10.0, seed=rng.integers(0, 1e9),
                              method="gaussian_filter", fwhm=2, fwhm_ratio=2)
    row = raw[(raw["sim_id"] == 0) & (raw["method"] == "general")].iloc[0]
    assert row["est"] == pytest.approx(np.corrcoef(X.ravel(), Y.ravel())[0, 1])
    # ratio 1 is the default and differs from ratio 2
    base = _run_sim_single(method_df, return_summary=False)
    assert (base["fwhm_ratio"] == 1).all() and not np.allclose(base["est"], raw["est"])


def test_sim_multiple_fwhm_ratio_column(method_df):
    setting_df = pd.DataFrame({
        "rho": [0.4, 0.4], "nx": [15, 15], "n_dim": [2, 2], "len_scale": [None, None],
        "gen_method": ["gaussian_filter"] * 2, "fwhm": [3, 3],
        "distribution": ["gaussian"] * 2, "noise_df": [None, None], "trans": [None, None],
        "fwhm_ratio": [1, 2],
    })
    out = src.sim_multiple(setting_df, method_df, m_sim=2, return_summary=True)
    assert list(out.groupby("setting_id")["fwhm_ratio"].first()) == [1, 2]
    # without the column, the ratio is 1
    out1 = src.sim_multiple(setting_df.drop(columns="fwhm_ratio"), method_df, m_sim=2)
    assert (out1["fwhm_ratio"] == 1).all()


def test_make_setting_df_columns_and_rho_filter():
    df = src.make_setting_df(nx=[20, 40], rho=[0, 0.4, -0.9], fwhm=[2, 3],
                             trans=[None, "square"], mask=[("ellipse", (1, 0.8))],
                             fwhm_ratio=[2], dep=[(None, None), ("shift", 1),
                                                  ("symmetric_shift", 1)])
    assert list(df.columns) == ["nx", "n_dim", "rho", "len_scale", "fwhm", "gen_method",
                                "distribution", "noise_df", "trans", "fwhm_ratio", "mask",
                                "mask_axes", "dep_model", "shift_lag"]
    # without a design: every rho; with a rho = 0 design: rho = 0 only
    base = 2 * 2 * 2  # nx x fwhm x trans
    assert len(df) == base * 3 + base * 1 * 2
    assert set(df.loc[df["dep_model"].isna(), "rho"]) == {0, 0.4, -0.9}
    assert set(df.loc[df["dep_model"].notna(), "rho"]) == {0}
    assert list(df.index) == list(range(len(df)))
    assert (df["mask"] == "ellipse").all() and all(a == (1, 0.8) for a in df["mask_axes"])


def test_make_setting_df_defaults_and_matches_notebook_construction():
    from itertools import product
    nx, rho, fwhm = [25, 50], [0, 0.4], [1, 2]
    df = src.make_setting_df(nx=nx, rho=rho, fwhm=fwhm)
    # the construction previously written out in main.ipynb
    ref = pd.DataFrame(product(nx, [2], rho, [None], fwhm, ["gaussian_filter"], ["gaussian"], [None],
                               [None], [(None, None)], [1], [(None, None)]),
                       columns=["nx", "n_dim", "rho", "len_scale", "fwhm", "gen_method",
                                "distribution", "noise_df", "trans", "mask_spec", "fwhm_ratio",
                                "dep_spec"])
    ref[["mask", "mask_axes"]] = pd.DataFrame(ref.pop("mask_spec").tolist(), index=ref.index)
    ref[["dep_model", "shift_lag"]] = pd.DataFrame(ref.pop("dep_spec").tolist(), index=ref.index)
    pd.testing.assert_frame_equal(df, ref)


def test_make_method_df():
    df = src.make_method_df(M=["auto", "plugin"])
    assert list(df.columns) == ["use_true_rho", "method", "kernel", "M", "b_qs", "constant"]
    assert len(df) == 4 * 2
    assert list(df["method"].unique()) == ["general", "gauss", "indep", "iid"]
    with pytest.raises(ValueError):
        src.make_method_df(method=["bartlett"])


def test_make_setting_and_method_df_run_in_sim_multiple():
    setting_df = src.make_setting_df(nx=[15], rho=[0, 0.4], fwhm=[2], trans=[None, "square"],
                                     dep=[(None, None), ("symmetric_shift", 1)])
    method_df = src.make_method_df(method=["general", "indep"])
    out = src.sim_multiple(setting_df, method_df, m_sim=2, return_summary=True)
    assert out["setting_id"].nunique() == len(setting_df)
    assert len(out) == len(setting_df) * len(method_df)


def test_sim_multiple_shift_design_columns(method_df):
    setting_df = pd.DataFrame({
        "rho": [0.0, 0.0, 0.0], "nx": [15, 15, 15], "n_dim": [2, 2, 2], "len_scale": [None] * 3,
        "gen_method": ["gaussian_filter"] * 3, "fwhm": [3, 3, 3],
        "distribution": ["gaussian"] * 3, "noise_df": [None] * 3, "trans": [None, None, "square"],
        "dep_model": [None, "shift", "symmetric_shift"], "shift_lag": [None, 1, 2],
    })
    out = src.sim_multiple(setting_df, method_df, m_sim=2, return_summary=True)
    first = out.groupby("setting_id").first()
    assert list(first["dep_model"].fillna("none")) == ["none", "shift", "symmetric_shift"]
    assert list(first["shift_lag"]) == [1, 1, 2]


# ---------------------------------------------------------------------------
# plugin bandwidth
# ---------------------------------------------------------------------------
def test_plugin_bandwidth_matches_andrews_in_1d():
    # V with AR(1) autocovariance r^|h| in 1D: m = 1.1447 (a^2 n)^(1/3), a = 2r / (1 - r^2)
    n, r = 1000, 0.6
    h = np.arange(-(n - 1), n)
    cov_ls = [r ** np.abs(h)] + [0] * 8
    a = 2 * r / (1 - r**2)
    m = 1.1447 * (a**2 * n) ** (1 / 3)
    M = src.plugin_bandwidth(cov_ls, np.array([1, 0, 0]), n, (n,))
    assert M[0] == np.ceil(m - 1)


def test_plugin_bandwidth_grows_with_smoothness():
    Ms = []
    for fwhm in (1, 2, 4):
        X, Y = src.gen_corr_field(0, (60, 60), fwhm=fwhm, seed=0, method="gaussian_filter")
        Ms.append(src.corr(X, Y, M="plugin", true_rho=0)[4][0])
    assert Ms[0] < Ms[1] < Ms[2]


def test_plugin_bandwidth_capped_at_half_side():
    X, Y = src.gen_corr_field(0, (12, 40), fwhm=8, seed=0, method="gaussian_filter")
    M = src.corr(X, Y, M="plugin", true_rho=0)[4]
    assert M[0] <= 5 and M[1] <= 19 and M[0] >= 1


@pytest.mark.parametrize("se_method", ["general", "gauss", "indep"])
def test_corr_plugin_runs_with_mask(se_method):
    X, Y = src.gen_corr_field(0.4, (30, 40), fwhm=3, seed=1)
    mask = src.gen_mask(X.shape, axes=(0.8, 1))
    rho, se, se0, _, M = src.corr(X, Y, M="plugin", se_method=se_method, mask=mask)
    assert np.isfinite(se) and np.isfinite(se0) and (M >= 1).all()
