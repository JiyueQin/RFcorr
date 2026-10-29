from itertools import product
from typing import Optional, Sequence, Tuple, Union

import pandas as pd

import rfcorr as rc


def make_method_df(method: Sequence[str] = ("general", "gauss", "indep", "iid"),
                   M: Sequence[Union[str, int, Tuple]] = ("optimal_rate",),
                   kernel: Sequence[str] = ("bartlett",),
                   use_true_rho: Sequence[bool] = (False,),
                   b_qs: Sequence[Optional[float]] = (None,),
                   constant: Sequence[float] = (1,)) -> pd.DataFrame:
    """
    Build the table of standard error methods for rc.sim_multiple, one row per combination.

    ---------------------------------------------------------------------------
    ARGUMENTS:
    method : list of str, optional (default=("general", "gauss", "indep", "iid"))
        Standard error methods (see rc.get_defaults()['se_methods']).
    M : list, optional (default=("optimal_rate",))
        Lag truncation rules, e.g. 'optimal_rate' or 'plugin' (see rc.get_se;
        'plugin' is still under development and not verified).
    kernel : list of str, optional (default=("bartlett",))
        Kernel types (see rc.corr_dev).
    use_true_rho : list of bool, optional (default=(False,))
        Whether to plug the true rho into the variance formula instead of the
        estimate.
    b_qs : list, optional (default=(None,))
        Bandwidths for the QS kernel.
    constant : list of float, optional (default=(1,))
        Scaling constants for the lag truncation.

    ---------------------------------------------------------------------------
    OUTPUT:
    method_df : pd.DataFrame
        Columns use_true_rho, method, kernel, M, b_qs, constant, with rows numbered
        from 0.
    ---------------------------------------------------------------------------
    AUTHORS: Jiyue Qin
    ---------------------------------------------------------------------------
    """
    for m in method:
        rc.canonical_se_method(m)  # fail early on an unknown method name
    return pd.DataFrame(
        product(use_true_rho, method, kernel, M, b_qs, constant),
        columns=["use_true_rho", "method", "kernel", "M", "b_qs", "constant"],
    )
