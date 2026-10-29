from typing import Optional

import numpy as np

import rfcorr as rc


def kernel_weights(x: np.ndarray,
                   type: str,
                   bandwidth: Optional[float],
                   M: Optional[float] = None) -> np.ndarray:
    """
    Compute kernel weights for lag weighting, truncated at lag M.

    Development function: a building block of rc.corr_dev; its interface may change.

    For the Bartlett kernel the bandwidth is set to M + 1, so the weights are
    1 - |h| / (M + 1) for |h| <= M. For the QS kernel `bandwidth` is used.

    ---------------------------------------------------------------------------
    ARGUMENTS:
    x : np.ndarray
        Lag values.
    type : str
        Kernel type: 'bartlett', 'qs', or 'constant'.
    bandwidth : float or None
        Bandwidth parameter for the QS kernel. Ignored for 'bartlett' and 'constant'.
    M : float or None, optional (default=None)
        Maximum lag cutoff.

    ---------------------------------------------------------------------------
    OUTPUT:
    weights : np.ndarray
        Kernel weights at each lag value (floats, 0 beyond M).
    ---------------------------------------------------------------------------
    AUTHORS: Jiyue Qin
    ---------------------------------------------------------------------------
    """
    mask = np.abs(x) <= M
    if type == "constant":
        # float weights, not booleans: np.ix_ treats a boolean array as indices
        return mask.astype(float)
    else:
        if type.lower() == "bartlett":
            bandwidth = M + 1
        elif type.lower() == "qs":
            if M is None:
                M = np.ceil(5 * bandwidth)
        y = x / float(bandwidth)
    return rc.kernel(y, type=type) * mask
