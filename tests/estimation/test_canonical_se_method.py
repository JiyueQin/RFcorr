"""Tests for rc.canonical_se_method. Run with: python -m pytest tests/estimation/test_canonical_se_method.py"""
import pytest

import rfcorr as rc


@pytest.mark.parametrize("name", ["general", "gauss", "indep", "iid"])
def test_paper_names_are_kept(name):
    assert rc.canonical_se_method(name) == name
    assert rc.canonical_se_method(name.upper()) == name


@pytest.mark.parametrize("old, new", [("hac", "general"), ("hac_gaussian", "gauss"),
                                      ("huristic", "indep"), ("HAC", "general")])
def test_earlier_names_are_mapped(old, new):
    assert rc.canonical_se_method(old) == new


def test_unknown_name():
    with pytest.raises(ValueError, match="Unknown se_method"):
        rc.canonical_se_method("newey")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
