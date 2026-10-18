"""Tests for rc.make_setting_df. Run with: python -m pytest tests/simulation/test_make_setting_df.py"""
import pytest

import rfcorr as rc


def test_make_setting_df_columns_and_rho_filter():
    df = rc.make_setting_df(nx=[20, 40], rho=[0, 0.4, -0.9], fwhm=[2, 3],
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


def test_make_setting_df_defaults():
    df = rc.make_setting_df(nx=[25, 50], rho=[0, 0.4], fwhm=[1, 2])
    assert len(df) == 8
    assert (df["n_dim"] == 2).all() and (df["fwhm_ratio"] == 1).all()
    assert (df["gen_method"] == "gaussian_filter").all() and (df["distribution"] == "gaussian").all()
    for col in ["len_scale", "noise_df", "trans", "mask", "mask_axes", "dep_model", "shift_lag"]:
        assert df[col].isna().all(), col


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
