import time
from typing import Optional, Tuple, Union

import numpy as np

import rfcorr as rc


def corr_dev(X: np.ndarray,
             Y: np.ndarray,
             M: Union[str, int, Tuple[float, ...]] = "cubic_root",
             true_rho: Optional[float] = None,
             se_method: str = "general",
             kernel: str = "bartlett",
             check: bool = False,
             b_qs: float = 5,
             constant: float = 1,
             standardize: bool = True,
             mask: Optional[np.ndarray] = None
             ) -> Tuple[float, float, float, float, tuple]:
    """
    Development version of rc.corr, with every option; the interface may change.

    Returns the Pearson correlation over the valid grid points with two standard
    errors that account for spatial dependence: one at the plugin rho (for
    confidence intervals) and one at rho = 0 (for testing H0: rho = 0). Besides the
    options of rc.corr, it accepts the legacy and experimental lag rules, other
    kernels, a known true rho, and diagnostics. It is used by the simulation
    functions. Users should call rc.corr, which uses only the methods of the paper.

    ---------------------------------------------------------------------------
    ARGUMENTS:
    X : np.ndarray
        First input field.
    Y : np.ndarray
        Second input field, same shape as X.
    M : str, int, or tuple of numeric, optional (default="cubic_root")
        Lag truncation: 'optimal_rate' (the paper's rule), the legacy rules 'auto'
        (same as 'optimal_rate'), 'cubic_root' and 'linear', 'plugin', an int, or a
        tuple with one value per dimension (see rc.get_se). 'plugin' is still under
        development and not verified.
    true_rho : float or None, optional (default=None)
        Known true correlation for plugin estimation. If None, the estimate is used.
    se_method : str, optional (default="general")
        Standard error method: 'general', 'gauss', 'indep', or 'iid' (see
        rc.get_defaults()['se_methods']). The earlier names 'hac', 'hac_gaussian',
        and 'huristic' are also accepted.
    kernel : str, optional (default="bartlett")
        Kernel type for HAC: 'bartlett', 'qs', 'constant', or 'bartlett_radial'
        (see rc.get_se).
    check : bool, optional (default=False)
        If True, print diagnostic information.
    b_qs : float, optional (default=5)
        Bandwidth parameter for the QS kernel.
    constant : float, optional (default=1)
        Scaling constant for the M calculation.
    standardize : bool, optional (default=True)
        If True, standardize X and Y (subtract mean, divide by std) before
        computing the standard error.
    mask : np.ndarray of bool or None, optional (default=None)
        Valid grid points, same shape as X and Y. Only these points enter the
        estimate and its standard error; values outside the mask (including NaN)
        are ignored. If None, every grid point is valid.

    ---------------------------------------------------------------------------
    OUTPUT:
    rho_est : float
        Estimated Pearson correlation.
    se : float
        Standard error of the correlation estimate (at the plugin rho), e.g. RFCorr.
    se_under_null : float
        Standard error computed under the null hypothesis rho = 0, e.g. RFCorr-0.
    runtime_secs : float
        Computation time in seconds.
    M_tuple : tuple
        Lag truncation values used, one per dimension ((None,) for 'iid').
    ---------------------------------------------------------------------------
    AUTHORS: Jiyue Qin
    ---------------------------------------------------------------------------
    """
    start_time = time.time()
    if X.shape != Y.shape:
        raise ValueError("X and Y must have the same shape")
    se_method = rc.canonical_se_method(se_method)
    dim = np.array(X.shape)
    if mask is None:
        rho_est = np.corrcoef(X.ravel(), Y.ravel())[0, 1]
        if standardize:
            X_standard = (X - X.mean()) / X.std()
            Y_standard = (Y - Y.mean()) / Y.std()
        else:
            X_standard = X
            Y_standard = Y
        n_obs = None
    else:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != X.shape:
            raise ValueError("mask must have the same shape as X and Y")
        X_valid, Y_valid = X[mask], Y[mask]
        rho_est = np.corrcoef(X_valid, Y_valid)[0, 1]
        if standardize:
            X_standard = (X - X_valid.mean()) / X_valid.std()
            Y_standard = (Y - Y_valid.mean()) / Y_valid.std()
        else:
            X_standard = X
            Y_standard = Y
        # zero outside the mask, so lag sums only pick up pairs of valid points
        X_standard = np.where(mask, X_standard, 0.0)
        Y_standard = np.where(mask, Y_standard, 0.0)
        n_obs = mask.sum()

    if true_rho is None:
        rho_plugin = rho_est
    else:
        rho_plugin = true_rho
    est_exp_ls = rc.get_est_exp_ls(se_method, X_standard, Y_standard, mask=mask)

    se, M_tuple = rc.get_se(
        dim,
        est_exp_ls,
        rho_plugin,
        M,
        se_method,
        kernel,
        check,
        b_qs,
        constant,
        n_obs=n_obs,
    )

    # under the null, the Gaussian method also sets the lag-0 cross-correlation to 0
    if se_method == "gauss":
        est_exp_ls_null = rc.get_est_exp_ls(
            se_method, X_standard, Y_standard, mask=mask, impose_null=True
        )
    else:
        est_exp_ls_null = est_exp_ls
    se_under_null, _ = rc.get_se(
        dim,
        est_exp_ls_null,
        0,
        M,
        se_method,
        kernel,
        check,
        b_qs,
        constant,
        n_obs=n_obs,
    )

    runtime_secs = time.time() - start_time
    return rho_est, se, se_under_null, runtime_secs, M_tuple
