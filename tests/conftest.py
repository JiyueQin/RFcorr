"""Fixtures shared by the tests in several subfolders."""
import pandas as pd
import pytest

import rfcorr as rc


@pytest.fixture
def fields_2d():
    return rc.gen_corr_field(rho=0.4, dim=(40, 50), fwhm=3, seed=0)


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
