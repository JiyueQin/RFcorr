"""Tests for rc.gen_corr_field. Run with: python -m pytest tests/fields/test_gen_corr_field.py"""
import numpy as np
import pytest

import rfcorr as rc


@pytest.mark.parametrize("dim", [300, (40, 50), (10, 12, 14)])
def test_gen_corr_field_shape_and_seed(dim):
    X1, Y1 = rc.gen_corr_field(rho=0.3, dim=dim, seed=5)
    X2, Y2 = rc.gen_corr_field(rho=0.3, dim=dim, seed=5)
    expected = (dim,) if isinstance(dim, int) else dim
    assert X1.shape == Y1.shape == expected
    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(Y1, Y2)


@pytest.mark.parametrize("rho", [-0.9, 0.0, 0.4, 0.9])
def test_gen_corr_field_iid_correlation(rho):
    X, Y = rc.gen_corr_field(rho=rho, dim=(200, 200), method="iid", seed=0)
    # SE of the sample correlation is about (1 - rho^2)/sqrt(n) <= 0.005
    assert np.corrcoef(X.ravel(), Y.ravel())[0, 1] == pytest.approx(rho, abs=0.03)
    assert Y.var() == pytest.approx(1.0, rel=0.05)


def test_gen_corr_field_smooth_correlation():
    X, Y = rc.gen_corr_field(rho=0.5, dim=(300, 300), fwhm=3, seed=0)
    assert np.corrcoef(X.ravel(), Y.ravel())[0, 1] == pytest.approx(0.5, abs=0.05)


def test_gen_corr_field_rho_one_gives_identical_fields():
    X, Y = rc.gen_corr_field(rho=1.0, dim=(20, 20), seed=0)
    np.testing.assert_allclose(X, Y)


def test_gen_corr_field_truncation_and_coordinates():
    X, Y, coords = rc.gen_corr_field(rho=0.5, dim=(9, 10), seed=0, M_trunc=0.3,
                                     return_coordinate=True)
    assert np.abs(X).max() <= 0.3 and np.abs(Y).max() <= 0.3
    assert [len(c) for c in coords] == [9, 10]


SHIFT_MODELS = ["shift", "symmetric_shift"]


def _mean_corr(rho, trans, dep_model="shift", m=150, n=40, **kw):
    return np.mean([
        np.corrcoef(*[f.ravel() for f in rc.gen_corr_field(
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
    r, r2 = rc.gaussian_filter_acf((1, 0), 3), rc.gaussian_filter_acf((2, 0), 3)
    if trans == "square":
        r, r2 = r**2, r2**2
    plus, minus = [], []
    for s in range(50):
        X, Y = rc.gen_corr_field(0, dim=(60, 60), fwhm=3, seed=s, trans=trans,
                                 dep_model="symmetric_shift")
        plus.append(np.corrcoef(X[:-1].ravel(), Y[1:].ravel())[0, 1])    # (X_s, Y_{s+h})
        minus.append(np.corrcoef(X[1:].ravel(), Y[:-1].ravel())[0, 1])   # (X_s, Y_{s-h})
    expected = np.sqrt((1 + r2) / 2 - r**2)
    assert np.mean(plus) == pytest.approx(expected, abs=0.03)
    assert np.mean(minus) == pytest.approx(expected, abs=0.03)


@pytest.mark.parametrize("trans", [None, "square"])
def test_shift_design_is_dependent(trans):
    # with rho = 0, Y_{s-h} contains U_s, so it is strongly correlated with X_s
    r = rc.gaussian_filter_acf((1, 0), 3)
    r_U = r if trans is None else r**2
    vals = []
    for s in range(50):
        X, Y = rc.gen_corr_field(0, dim=(60, 60), fwhm=3, seed=s, trans=trans, dep_model="shift")
        vals.append(np.corrcoef(X[1:].ravel(), Y[:-1].ravel())[0, 1])
    assert np.mean(vals) == pytest.approx(np.sqrt(1 - r_U**2), abs=0.03)


@pytest.mark.parametrize("dep_model", SHIFT_MODELS)
def test_shift_design_is_not_degenerate(dep_model):
    # unlike a central difference, the sample correlation keeps the usual sqrt(N) scaling
    sd_scaled = []
    for n in [30, 60]:
        rhos = [np.corrcoef(*[f.ravel() for f in rc.gen_corr_field(
            0, dim=(n, n), fwhm=3, seed=s, dep_model=dep_model)])[0, 1] for s in range(200)]
        sd_scaled.append(np.std(rhos) * n)
    assert sd_scaled[0] > 0.5
    assert sd_scaled[1] / sd_scaled[0] == pytest.approx(1, abs=0.3)


@pytest.mark.parametrize("dep_model", SHIFT_MODELS)
@pytest.mark.parametrize("lag, dim", [(2, (20, 25)), ((0, -1), (20, 25)), ((1, 1), (20, 25)), (1, 30)])
def test_shift_design_lags_and_shapes(lag, dim, dep_model):
    X, Y = rc.gen_corr_field(0.3, dim=dim, fwhm=2, seed=0, dep_model=dep_model, shift_lag=lag)
    expected = (dim,) if isinstance(dim, int) else dim
    assert X.shape == Y.shape == expected


@pytest.mark.parametrize("dep_model", SHIFT_MODELS)
def test_shift_design_invalid_settings(dep_model):
    with pytest.raises(ValueError):
        rc.gen_corr_field(0, dim=(20, 20), method="iid", dep_model=dep_model)
    with pytest.raises(ValueError):
        rc.gen_corr_field(0, dim=(20, 20), distribution="t", trans="square", dep_model=dep_model)
    with pytest.raises(ValueError):
        rc.gen_corr_field(0, dim=(20, 20), dep_model=dep_model, shift_lag=0)
    with pytest.raises(ValueError):
        rc.gen_corr_field(0, dim=(20, 20), dep_model=dep_model, shift_lag=(1, 0, 0))


@pytest.mark.parametrize("trans", [None, "square"])
@pytest.mark.parametrize("rho", [0.0, 0.4])
def test_sum_diff_keeps_target_correlation(rho, trans):
    # U and V have the same variance, so their sum and difference are uncorrelated
    assert _mean_corr(rho, trans, "sum_diff") == pytest.approx(rho, abs=0.015)


def test_sum_diff_keeps_variance():
    X, Y = rc.gen_corr_field(0, dim=(200, 200), fwhm=2, fwhm_ratio=2, seed=0,
                             dep_model="sum_diff")
    assert X.var() == pytest.approx(1.0, rel=0.1)
    assert Y.var() == pytest.approx(1.0, rel=0.1)


@pytest.mark.parametrize("fwhm_ratio", [1, 2])
def test_sum_diff_cross_correlation(fwhm_ratio):
    # with rho = 0 and Gaussian fields, corr(X_s, Y_{s+e1}) = (r_U - r_V) / 2: zero when U and V
    # are equally smooth (then X and Y are independent), positive when V is rougher
    r_U = rc.gaussian_filter_acf((1, 0), 3)
    r_V = rc.gaussian_filter_acf((1, 0), 3 / fwhm_ratio)
    vals = []
    for s in range(50):
        X, Y = rc.gen_corr_field(0, dim=(60, 60), fwhm=3, fwhm_ratio=fwhm_ratio, seed=s,
                                 dep_model="sum_diff")
        vals.append(np.corrcoef(X[:-1].ravel(), Y[1:].ravel())[0, 1])
    assert np.mean(vals) == pytest.approx((r_U - r_V) / 2, abs=0.02)


def test_sum_diff_non_gaussian_is_dependent():
    # chi^2_1 fields: uncorrelated at every lag when U and V are equally smooth, but dependent,
    # e.g. the squares of X and Y are correlated
    vals = []
    for s in range(30):
        X, Y = rc.gen_corr_field(0, dim=(60, 60), fwhm=2, fwhm_ratio=1, seed=s, trans="square",
                                 dep_model="sum_diff")
        Xc, Yc = X - X.mean(), Y - Y.mean()
        vals.append(np.corrcoef((Xc**2).ravel(), (Yc**2).ravel())[0, 1])
    assert abs(np.mean(vals)) > 0.1


def test_sum_diff_is_kept_only_with_rho_zero():
    setting_df = rc.make_setting_df(nx=[20], rho=[0, 0.4], fwhm=[2],
                                    dep=[(None, None), ("sum_diff", None)])
    assert (setting_df.loc[setting_df["dep_model"] == "sum_diff", "rho"] == 0).all()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
