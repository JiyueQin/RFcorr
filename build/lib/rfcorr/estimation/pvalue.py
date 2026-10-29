from typing import Union

import numpy as np
from scipy.stats import norm


def pvalue(rho_est: Union[float, np.ndarray],
           se_under_null: Union[float, np.ndarray],
           fisher_trans: bool = True) -> Union[float, np.ndarray]:
    """
    Two-sided p-value for testing H0: rho = 0.

    ---------------------------------------------------------------------------
    ARGUMENTS:
    rho_est : float or array-like
        Estimated Pearson correlation(s).
    se_under_null : float or array-like
        Standard error(s) of `rho_est` computed under the null (rho = 0), on the
        original (untransformed) scale.
    fisher_trans : bool, optional (default=True)
        If True, build the test statistic on the Fisher z scale
        (F = arctanh(rho_est)); under the null the SE on that scale equals
        se_under_null, since 1 - rho^2 = 1. If False, use the plain Wald statistic
        Z = rho_est / se_under_null.

    ---------------------------------------------------------------------------
    OUTPUT:
    p : float or np.ndarray
        Two-sided p-value(s): 2 * (1 - Phi(|Z|)).
    ---------------------------------------------------------------------------
    AUTHORS: Jiyue Qin
    ---------------------------------------------------------------------------
    """
    rho_est = np.asarray(rho_est, dtype=float)
    se_under_null = np.asarray(se_under_null, dtype=float)

    if fisher_trans:
        F = np.arctanh(rho_est)
        # 1 - rho**2 = 1 under the null, so the SE on the Fisher scale is se_under_null
        F_se_under_null = se_under_null
        Z = F / F_se_under_null
    else:
        Z = rho_est / se_under_null

    p = 2 * (1 - norm.cdf(np.abs(Z)))

    if p.ndim == 0:
        return float(p)
    return p
