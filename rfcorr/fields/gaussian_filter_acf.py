import numbers
from typing import Tuple, Union

import numpy as np
from scipy.ndimage import gaussian_filter1d


def gaussian_filter_acf(h: Union[int, Tuple[int, ...]],
                        fwhm: float,
                        truncate: float = 4) -> float:
    """
    Population autocorrelation at lag h of a field from rc.gen_field(method='gaussian_filter').

    The field is i.i.d. noise convolved with a discrete Gaussian kernel K, so its
    correlation at lag h is sum_u K(u) K(u + h) / sum_u K(u)^2, whatever the noise
    distribution. The kernel is a product of 1D kernels, so the correlation is the
    product of the 1D correlations over axes.

    ---------------------------------------------------------------------------
    ARGUMENTS:
    h : int or tuple of int
        Lag. An int is a lag along a 1D axis.
    fwhm : float
        Full width at half maximum of the Gaussian kernel, as in rc.gen_field.
    truncate : float, optional (default=4)
        Truncation of the kernel in standard deviations, as in rc.gen_field.

    ---------------------------------------------------------------------------
    OUTPUT:
    r : float
        Correlation between the field at a site and at the site shifted by h.
    ---------------------------------------------------------------------------
    AUTHORS: Jiyue Qin
    ---------------------------------------------------------------------------
    """
    if isinstance(h, numbers.Integral):
        h = (h,)
    sigma = fwhm / np.sqrt(8 * np.log(2))
    n = 2 * int(np.ceil(truncate * sigma)) + 2 * max(abs(int(x)) for x in h) + 3
    impulse = np.zeros(n)
    impulse[n // 2] = 1
    # the 1D kernel used by gaussian_filter in gen_field
    K = gaussian_filter1d(impulse, sigma, truncate=truncate)
    r = 1.0
    for lag in h:
        lag = abs(int(lag))
        r *= np.sum(K[: n - lag] * K[lag:]) / np.sum(K * K)
    return r
