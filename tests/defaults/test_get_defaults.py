"""Tests for rc.get_defaults. Run with: python -m pytest tests/defaults/test_get_defaults.py"""
import pytest

import rfcorr as rc


def test_values():
    defaults = rc.get_defaults()
    assert defaults["se_methods"] == ("general", "gauss", "indep", "iid")
    assert dict(defaults["se_method_aliases"]) == {
        "hac": "general", "hac_gaussian": "gauss", "huristic": "indep"}
    assert defaults["rho_zero_dep_models"] == ("shift", "symmetric_shift")


def test_aliases_point_to_known_methods():
    defaults = rc.get_defaults()
    assert set(defaults["se_method_aliases"].values()) <= set(defaults["se_methods"])


def test_cached_and_read_only():
    assert rc.get_defaults() is rc.get_defaults()
    with pytest.raises(TypeError):
        rc.get_defaults()["se_methods"] = ("iid",)
    with pytest.raises(TypeError):
        rc.get_defaults()["se_method_aliases"]["hac"] = "iid"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
