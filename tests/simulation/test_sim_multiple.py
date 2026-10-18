"""Tests for rc.sim_multiple. Run with: python -m pytest tests/simulation/test_sim_multiple.py"""
import pandas as pd
import pytest

import rfcorr as rc


def test_sim_multiple(method_df, capsys):
    setting_df = pd.DataFrame({
        "rho": [0.0, 0.5], "nx": [15, 20], "n_dim": [1, 2], "len_scale": [None, None],
        "gen_method": ["gaussian_filter"] * 2, "fwhm": [2, 3],
        "distribution": ["gaussian"] * 2, "noise_df": [None, None], "trans": [None, "square"],
    })
    out = rc.sim_multiple(setting_df, method_df, m_sim=2, return_summary=True)
    assert out.columns[0] == "setting_id"
    assert len(out) == len(setting_df) * len(method_df)
    assert list(out.groupby("setting_id")["rho"].first()) == [0.0, 0.5]
    assert "remaining settings" in capsys.readouterr().out
    # verbose=False prints nothing
    rc.sim_multiple(setting_df, method_df, m_sim=2, verbose=False)
    assert capsys.readouterr().out == ""


def test_sim_multiple_with_and_without_mask_column(method_df):
    setting_df = pd.DataFrame({
        "rho": [0.0, 0.5], "nx": [15, 20], "n_dim": [2, 2], "len_scale": [None, None],
        "gen_method": ["gaussian_filter"] * 2, "fwhm": [2, 3],
        "distribution": ["gaussian"] * 2, "noise_df": [None, None], "trans": [None, None],
        "mask": [None, "ellipse"], "mask_axes": [None, (1, 0.6)],
    })
    out = rc.sim_multiple(setting_df, method_df, m_sim=2, return_summary=True)
    n_obs = out.groupby("setting_id")["n_obs"].first()
    assert n_obs[0] == 225 and n_obs[1] == rc.gen_mask((20, 20), axes=(1, 0.6)).sum()


def test_sim_multiple_fwhm_ratio_column(method_df):
    setting_df = pd.DataFrame({
        "rho": [0.4, 0.4], "nx": [15, 15], "n_dim": [2, 2], "len_scale": [None, None],
        "gen_method": ["gaussian_filter"] * 2, "fwhm": [3, 3],
        "distribution": ["gaussian"] * 2, "noise_df": [None, None], "trans": [None, None],
        "fwhm_ratio": [1, 2],
    })
    out = rc.sim_multiple(setting_df, method_df, m_sim=2, return_summary=True)
    assert list(out.groupby("setting_id")["fwhm_ratio"].first()) == [1, 2]
    # without the column, the ratio is 1
    out1 = rc.sim_multiple(setting_df.drop(columns="fwhm_ratio"), method_df, m_sim=2)
    assert (out1["fwhm_ratio"] == 1).all()


def test_make_setting_and_method_df_run_in_sim_multiple():
    setting_df = rc.make_setting_df(nx=[15], rho=[0, 0.4], fwhm=[2], trans=[None, "square"],
                                    dep=[(None, None), ("symmetric_shift", 1)])
    method_df = rc.make_method_df(method=["general", "indep"])
    out = rc.sim_multiple(setting_df, method_df, m_sim=2, return_summary=True)
    assert out["setting_id"].nunique() == len(setting_df)
    assert len(out) == len(setting_df) * len(method_df)


def test_sim_multiple_shift_design_columns(method_df):
    setting_df = pd.DataFrame({
        "rho": [0.0, 0.0, 0.0], "nx": [15, 15, 15], "n_dim": [2, 2, 2], "len_scale": [None] * 3,
        "gen_method": ["gaussian_filter"] * 3, "fwhm": [3, 3, 3],
        "distribution": ["gaussian"] * 3, "noise_df": [None] * 3, "trans": [None, None, "square"],
        "dep_model": [None, "shift", "symmetric_shift"], "shift_lag": [None, 1, 2],
    })
    out = rc.sim_multiple(setting_df, method_df, m_sim=2, return_summary=True)
    first = out.groupby("setting_id").first()
    assert list(first["dep_model"].fillna("none")) == ["none", "shift", "symmetric_shift"]
    assert list(first["shift_lag"]) == [1, 1, 2]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
