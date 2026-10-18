"""Tests for rc.get_est_exp_ls. Run with: python -m pytest tests/estimation/test_get_est_exp_ls.py"""
import numpy as np
import pytest

import rfcorr as rc


def test_get_est_exp_ls_impose_null_only_changes_lag0():
    rng = np.random.default_rng(3)
    X, Y = rng.normal(size=(6, 7)), rng.normal(size=(6, 7))
    full = rc.get_est_exp_ls("gauss", X, Y)
    null = rc.get_est_exp_ls("gauss", X, Y, impose_null=True)
    center = (5, 6)
    for a, b in zip(full, null):
        diff = np.argwhere(~np.isclose(a, b))
        assert all(tuple(d) == center for d in diff)
    # independence and general methods ignore the option
    for method in ["indep", "general"]:
        for a, b in zip(rc.get_est_exp_ls(method, X, Y),
                        rc.get_est_exp_ls(method, X, Y, impose_null=True)):
            np.testing.assert_array_equal(a, b)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
