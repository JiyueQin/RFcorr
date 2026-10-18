"""Tests for rc.sim_single. Run with: python -m pytest tests/simulation/test_sim_single.py"""
import numpy as np
import pandas as pd
import pytest

import rfcorr as rc


def _run_sim_single(method_df, **kwargs):
    return rc.sim_single(method_df, rho=0.4, m_sim=3, nx=20, n_dim=2,
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
    raw = rc.sim_single(method_df.iloc[:1], rho=0.0, m_sim=1, nx=20, n_dim=n_dim,
                        gen_method="iid", seed=0, return_summary=False)
    # under iid SE the null SE is 1/sqrt(n), which pins down the field size
    assert raw["se_under_null"].iloc[0] == pytest.approx(1 / np.sqrt(expected))


def test_sim_single_with_mask(method_df):
    raw = _run_sim_single(method_df, return_summary=False, mask="ellipse", mask_axes=(1, 0.5))
    mask = rc.gen_mask((20, 20), axes=(1, 0.5))
    assert (raw["n_obs"] == mask.sum()).all() and (raw["mask"] == "ellipse").all()
    assert raw["mask_axes"].iloc[0] == (1, 0.5)
    # the same mask is used in every replicate: recompute replicate 0 by hand
    rng = np.random.default_rng(0)
    X, Y = rc.gen_corr_field(0.4, (20, 20), 1.0, 10.0, seed=rng.integers(0, 1e9),
                             method="gaussian_filter", fwhm=2)
    row = raw[(raw["sim_id"] == 0) & (raw["method"] == "general")].iloc[0]
    est, se, se0, _, _ = rc.corr_dev(X, Y, se_method="general", M="auto", mask=mask)
    assert (row["est"], row["se"], row["se_under_null"]) == pytest.approx((est, se, se0))


def test_sim_single_no_mask_unchanged(method_df):
    a = _run_sim_single(method_df, return_summary=False).drop(columns="runtime")
    b = _run_sim_single(method_df, return_summary=False, mask=np.nan).drop(columns="runtime")
    cols = [c for c in a.columns if c not in ("mask", "mask_axes")]
    pd.testing.assert_frame_equal(a[cols], b[cols])
    assert (a["n_obs"] == 400).all()


def test_sim_single_fwhm_ratio(method_df):
    raw = _run_sim_single(method_df, return_summary=False, fwhm_ratio=2)
    assert (raw["fwhm_ratio"] == 2).all()
    # replicate 0 by hand with the same seed stream
    rng = np.random.default_rng(0)
    X, Y = rc.gen_corr_field(0.4, (20, 20), 1.0, 10.0, seed=rng.integers(0, 1e9),
                             method="gaussian_filter", fwhm=2, fwhm_ratio=2)
    row = raw[(raw["sim_id"] == 0) & (raw["method"] == "general")].iloc[0]
    assert row["est"] == pytest.approx(np.corrcoef(X.ravel(), Y.ravel())[0, 1])
    # ratio 1 is the default and differs from ratio 2
    base = _run_sim_single(method_df, return_summary=False)
    assert (base["fwhm_ratio"] == 1).all() and not np.allclose(base["est"], raw["est"])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
