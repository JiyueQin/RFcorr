import numbers
from typing import Optional, Tuple, Union

import numpy as np

import rfcorr as rc


def corr(X: np.ndarray,
         Y: np.ndarray,
         se_method: str = "general",
         M: Union[str, int, Tuple[int, ...]] = "optimal_rate",
         constant: float = 1,
         mask: Optional[np.ndarray] = None
         ) -> Tuple[float, float, float, float, tuple]:
    """
    Estimate the correlation of two fields and its standard errors.

    Returns the Pearson correlation over the valid grid points with two standard
    errors that account for spatial dependence: one at the estimated correlation
    (for confidence intervals, see rc.confidence_interval) and one at rho = 0 (for
    testing H0: rho = 0, see rc.pvalue). The variance is estimated with Bartlett
    weights over spatial lags up to the truncation lag M in every direction.

    ---------------------------------------------------------------------------
    ARGUMENTS:
    X : np.ndarray
        First input field, of any dimension.
    Y : np.ndarray
        Second input field, same shape as X.
    se_method : str, optional (default="general")
        Standard error method: 'general' (RFCorr), 'gauss' (RFCorr-Gauss, for
        jointly Gaussian fields), 'indep' (RFCorr-Indep, for independent fields),
        or 'iid' (ignores spatial dependence).
    M : str, int, or tuple of int, optional (default="optimal_rate")
        Truncation lag. 'optimal_rate' uses ceil(constant * n ** (1 / (d + 2))) in
        every direction, where n is the number of valid grid points and d the
        dimension; this rule grows at the rate that minimizes the mean squared
        error of the variance estimate. An int uses that lag in every direction, and
        a tuple gives one lag per dimension.
    constant : float, optional (default=1)
        Constant multiplying n ** (1 / (d + 2)) when M='optimal_rate'.
    mask : np.ndarray of bool or None, optional (default=None)
        Valid grid points, same shape as X and Y. Only these points enter the
        estimate and its standard errors; values outside the mask (including NaN)
        are ignored. If None, every grid point is valid.

    ---------------------------------------------------------------------------
    OUTPUT:
    rho_est : float
        Estimated Pearson correlation.
    se : float
        Standard error of rho_est at the estimated correlation.
    se_under_null : float
        Standard error of rho_est under the null hypothesis rho = 0.
    runtime_secs : float
        Computation time in seconds.
    M_tuple : tuple
        Truncation lag used in each dimension ((None,) for 'iid').
    ---------------------------------------------------------------------------
    AUTHORS: Jiyue Qin
    ---------------------------------------------------------------------------
    """
    # only the paper's rule or a fixed lag; the legacy and experimental rules are in
    # rc.corr_dev
    is_lag = isinstance(M, numbers.Integral) and not isinstance(M, bool)
    is_lags = isinstance(M, tuple) and all(isinstance(x, numbers.Integral) for x in M)
    is_rule = isinstance(M, str) and M == "optimal_rate"
    if not (is_rule or is_lag or is_lags):
        raise ValueError("M must be 'optimal_rate', an int, or a tuple of int.")
    # get_se treats an int M as the same lag in every dimension only for Python ints
    if is_lag:
        M = int(M)
    return rc.corr_dev(X, Y, M=M, se_method=se_method, kernel="bartlett",
                       constant=constant, mask=mask)
