from typing import Union

import numpy as np


def kernel(x: Union[float, np.ndarray],
           type: str = "bartlett") -> Union[float, np.ndarray]:
    """
    Kernel function for lag weighting.

    Development function: a building block of rc.corr_dev; its interface may change.

    ---------------------------------------------------------------------------
    ARGUMENTS:
    x : float or np.ndarray
        Input value(s).
    type : str, optional (default="bartlett")
        Kernel type: 'bartlett', 'qs' (Quadratic Spectral), or 'constant'. Case
        insensitive.

    ---------------------------------------------------------------------------
    OUTPUT:
    out : float or np.ndarray
        Kernel value(s) at x. The constant kernel returns the scalar 1.
    ---------------------------------------------------------------------------
    AUTHORS: Jiyue Qin
    ---------------------------------------------------------------------------
    """
    x = np.asarray(x, dtype=float)

    if type.lower() == "bartlett":
        out = np.maximum(0, 1 - np.abs(x))

    elif type.lower() == "qs":
        z = (6.0 * np.pi * x) / 5.0

        # Numerically stable evaluation:
        # For very small |x|, use the exact limit K_QS(0) = 1.
        out = np.empty_like(x)
        small = np.abs(x) < 1e-8

        # General case
        num = np.sin(z) / np.where(small, 1.0, z) - np.cos(z)
        out[~small] = (25.0 / (12.0 * np.pi**2 * x[~small] ** 2)) * num[~small]

        # Limit at 0
        out[small] = 1.0
    elif type.lower() == "constant":
        out = 1

    else:
        raise ValueError("Unknown kernel type: choose 'bartlett' or 'QS'")

    return out
