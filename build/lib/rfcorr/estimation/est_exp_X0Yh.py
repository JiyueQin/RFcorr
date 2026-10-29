from typing import List, Optional, Tuple, Union

import numpy as np
from scipy.signal import correlate


def est_exp_X0Yh(X: np.ndarray,
                 Y: np.ndarray,
                 return_counts_lags: bool = False,
                 mask: Optional[np.ndarray] = None
                 ) -> Union[np.ndarray,
                            Tuple[np.ndarray, np.ndarray, np.ndarray, List[np.ndarray]]]:
    """
    Estimate E[X_0 * Y_h] for all spatial integer lags h on a grid.

    Development function: a building block of rc.corr_dev; its interface may change.

    At each lag the sum of products over valid pairs (computed by FFT) is divided
    by the number of valid pairs, which corrects for the edge of the grid.

    ---------------------------------------------------------------------------
    ARGUMENTS:
    X : np.ndarray
        First input field, of any dimensionality. If `mask` is given, X must be 0
        outside it.
    Y : np.ndarray
        Second input field, same shape as X. If `mask` is given, Y must be 0
        outside it.
    return_counts_lags : bool, optional (default=False)
        If False, returns only E. If True, returns (S, C, E, lag_ls).
    mask : np.ndarray of bool or None, optional (default=None)
        Valid grid points. If None, every grid point is valid. Lags with no valid
        pair get E = NaN, since there is nothing to average.

    ---------------------------------------------------------------------------
    OUTPUT:
    S : np.ndarray
        Sum of products at each lag. Only returned if return_counts_lags=True.
    C : np.ndarray
        Number of valid pairs at each lag (edge correction). Only returned if
        return_counts_lags=True.
    E : np.ndarray
        Estimated E[X_0 Y_h] at each lag h. Shape is (2*d_i-1 for each dimension).
    lag_ls : list of np.ndarray
        Lag coordinates for each dimension. Only returned if return_counts_lags=True.
    ---------------------------------------------------------------------------
    AUTHORS: Jiyue Qin
    ---------------------------------------------------------------------------
    """
    if X.shape != Y.shape:
        raise ValueError("X and Y must have the same shape")

    dim = X.shape

    # todo: only have to do this once so could speed up here.
    if mask is None:
        mask = np.ones(dim)
    else:
        mask = mask.astype(float)

    # Sum of products at each lag
    S = correlate(X, Y, mode="full")
    # number of valid pairs at each lag (also known as edge correction for unbiased estimate)
    C = correlate(mask, mask, mode="full")
    # estimate of the expectaion; with a mask some lags have no valid pair (C is 0 up to
    # FFT round-off), and those are NaN so get_se can drop them
    E = np.divide(S, C, out=np.full_like(S, np.nan), where=C > 0.5)
    if not return_counts_lags:
        return E
    else:
        lag_ls = [np.arange(-(ni - 1), ni) for ni in dim]
        return S, C, E, lag_ls
