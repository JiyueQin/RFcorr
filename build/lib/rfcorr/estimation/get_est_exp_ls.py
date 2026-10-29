from typing import List, Optional

import numpy as np

import rfcorr as rc


def get_est_exp_ls(se_method: str,
                   X_standard: np.ndarray,
                   Y_standard: np.ndarray,
                   mask: Optional[np.ndarray] = None,
                   impose_null: bool = False) -> List[np.ndarray]:
    """
    Compute the lag-wise expectation estimates entering the HAC variance.

    Development function: a building block of rc.corr_dev; its interface may change.

    For 'general' (and 'iid') these are the lag-wise means of products of the
    components of T = (X Y, X^2, Y^2). For 'gauss' and 'indep' they are the
    Gaussian (Isserlis) expressions in the lag-wise auto- and cross-correlations.

    ---------------------------------------------------------------------------
    ARGUMENTS:
    se_method : str
        Standard error method: 'general', 'gauss', 'indep', or 'iid' (see
        rc.get_defaults()['se_methods']); the earlier names 'hac', 'hac_gaussian',
        and 'huristic' are also accepted. Determines which set of cross-product
        expectations is computed.
    X_standard : np.ndarray
        The (optionally standardized) first field, 0 outside `mask`.
    Y_standard : np.ndarray
        The (optionally standardized) second field, same shape, 0 outside `mask`.
    mask : np.ndarray of bool or None, optional (default=None)
        Valid grid points. If None, every grid point is valid.
    impose_null : bool, optional (default=False)
        Only used for 'gauss'. If True, set the lag-0 cross-correlation (which equals
        the sample correlation) to 0, so that the null rho = 0 is imposed in the
        Gaussian covariance structure as well (RFCorr-Gauss-0 in the paper).

    ---------------------------------------------------------------------------
    OUTPUT:
    est_exp_ls : list of np.ndarray
        List of 9 lag-wise expectation arrays, ordered to fill the 3x3 omega matrix
        by row in rc.get_se.
    ---------------------------------------------------------------------------
    AUTHORS: Jiyue Qin
    ---------------------------------------------------------------------------
    """
    se_method = rc.canonical_se_method(se_method)
    if se_method in ["gauss", "indep"]:
        rho_xx_h = rc.est_exp_X0Yh(X_standard, X_standard, mask=mask)
        rho_yy_h = rc.est_exp_X0Yh(Y_standard, Y_standard, mask=mask)
        if se_method == "indep":
            rho_xy_h = 0
            rho_yx_h = 0
        else:
            rho_xy_h = rc.est_exp_X0Yh(X_standard, Y_standard, mask=mask)
            rho_yx_h = rc.est_exp_X0Yh(Y_standard, X_standard, mask=mask)
            if impose_null:
                # lag 0 sits at the center of the lag arrays; rho = 0 fixes only this entry
                center = tuple(n - 1 for n in X_standard.shape)
                rho_xy_h = rho_xy_h.copy()
                rho_yx_h = rho_yx_h.copy()
                rho_xy_h[center] = 0.0
                rho_yx_h[center] = 0.0
        # order the matrix by row
        est_exp_ls = [
            rho_xx_h * rho_yy_h + rho_xy_h * rho_yx_h,
            2 * rho_xx_h * rho_yx_h,
            2 * rho_yy_h * rho_xy_h,
            2 * rho_xx_h * rho_xy_h,
            2 * rho_xx_h * rho_xx_h,
            2 * rho_xy_h * rho_xy_h,
            2 * rho_yy_h * rho_yx_h,
            2 * rho_yx_h * rho_yx_h,
            2 * rho_yy_h * rho_yy_h,
        ]
    else:
        X_standard_square = X_standard**2
        Y_standard_square = Y_standard**2
        X_standard_Y_standard = X_standard * Y_standard
        est_exp_ls = [
            rc.est_exp_X0Yh(X_standard_Y_standard, X_standard_Y_standard, mask=mask),
            rc.est_exp_X0Yh(X_standard_Y_standard, X_standard_square, mask=mask),
            rc.est_exp_X0Yh(X_standard_Y_standard, Y_standard_square, mask=mask),
            rc.est_exp_X0Yh(X_standard_square, X_standard_Y_standard, mask=mask),
            rc.est_exp_X0Yh(X_standard_square, X_standard_square, mask=mask),
            rc.est_exp_X0Yh(X_standard_square, Y_standard_square, mask=mask),
            rc.est_exp_X0Yh(Y_standard_square, X_standard_Y_standard, mask=mask),
            rc.est_exp_X0Yh(Y_standard_square, X_standard_square, mask=mask),
            rc.est_exp_X0Yh(Y_standard_square, Y_standard_square, mask=mask),
        ]
    return est_exp_ls
