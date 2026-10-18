"""Tests for rc.make_method_df. Run with: python -m pytest tests/simulation/test_make_method_df.py"""
import pytest

import rfcorr as rc


def test_make_method_df():
    df = rc.make_method_df(M=["auto", "plugin"])
    assert list(df.columns) == ["use_true_rho", "method", "kernel", "M", "b_qs", "constant"]
    assert len(df) == 4 * 2
    assert list(df["method"].unique()) == ["general", "gauss", "indep", "iid"]
    with pytest.raises(ValueError):
        rc.make_method_df(method=["bartlett"])
    # the default lag rule is the paper's
    assert (rc.make_method_df()["M"] == "optimal_rate").all()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
