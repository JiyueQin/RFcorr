from typing import Tuple, Union

import numpy as np
from scipy.stats import norm


def confidence_interval(rho_est: Union[float, np.ndarray],
                        se: Union[float, np.ndarray],
                        alpha: float = 0.05,
                        fisher_trans: bool = True
                        ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
    Compute a confidence interval for a correlation estimate.

    ---------------------------------------------------------------------------
    ARGUMENTS:
    rho_est : float or array-like
        Estimated Pearson correlation(s).
    se : float or array-like
        Standard error(s) of `rho_est` on the original (untransformed) scale.
    alpha : float, optional (default=0.05)
        Significance level; the returned interval has coverage 1 - alpha.
    fisher_trans : bool, optional (default=True)
        If True, build the interval on the Fisher z scale (z = arctanh(rho)) and
        back-transform via tanh. The SE on the z scale is se / (1 - rho_est**2).
        If False, return the symmetric Wald interval rho_est +/- z * se, clipped
        to [-1, 1].

    ---------------------------------------------------------------------------
    OUTPUT:
    lower : float or np.ndarray
        Lower bound of the (1 - alpha) confidence interval.
    upper : float or np.ndarray
        Upper bound of the (1 - alpha) confidence interval.
    ---------------------------------------------------------------------------
    AUTHORS: Jiyue Qin
    ---------------------------------------------------------------------------
    """
    rho_est = np.asarray(rho_est, dtype=float)
    se = np.asarray(se, dtype=float)
    z_crit = norm.ppf(1 - alpha / 2)

    if fisher_trans:
        F = np.arctanh(rho_est)
        se_F = se / (1 - rho_est**2)
        lower = np.tanh(F - z_crit * se_F)
        upper = np.tanh(F + z_crit * se_F)
    else:
        lower = np.clip(rho_est - z_crit * se, -1, 1)
        upper = np.clip(rho_est + z_crit * se, -1, 1)

    if lower.ndim == 0:
        return float(lower), float(upper)
    return lower, upper
