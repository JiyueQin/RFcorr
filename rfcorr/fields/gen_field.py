import sys
from typing import List, Optional, Tuple, Union

import gstools as gs
import numpy as np
from scipy.ndimage import gaussian_filter


def gen_field(dim: Union[int, Tuple[int, ...]] = (50, 50),
              var: float = 1.0,
              len_scale: float = 1,
              seed: Optional[int] = None,
              M_trunc: Optional[float] = None,
              return_coordinate: bool = False,
              method: str = "gstools",
              fwhm: float = 2,
              truncate: float = 4,
              distribution: str = "gaussian",
              noise_df: Optional[int] = None,
              trans: Optional[str] = None
              ) -> Union[np.ndarray, Tuple[np.ndarray, List[np.ndarray]]]:
    """
    Generate a 0-mean random field with optional truncation.

    The field is i.i.d. noise ('iid'), a Gaussian random field with a Gaussian
    covariance ('gstools'), or i.i.d. noise smoothed with a Gaussian kernel and
    rescaled to the target variance ('gaussian_filter').

    ---------------------------------------------------------------------------
    ARGUMENTS:
    dim : int or tuple of int, optional (default=(50, 50))
        Dimensions of the field. If int, creates a 1D field. If tuple, creates a
        multi-dimensional field.
    var : float, optional (default=1.0)
        Target variance of the field.
    len_scale : float, optional (default=1)
        Length scale parameter of the Gaussian covariance (larger length scale is
        smoother). Only used when method='gstools'.
    seed : int or None, optional (default=None)
        Random seed for reproducibility. If None, uses a random seed.
    M_trunc : float or None, optional (default=None)
        Truncation threshold. If provided, the field is clipped to [-M_trunc, M_trunc].
    return_coordinate : bool, optional (default=False)
        If True, also returns the per-dimension coordinate lists.
    method : str, optional (default="gstools")
        Method to generate the field: 'iid', 'gstools', or 'gaussian_filter'.
    fwhm : float, optional (default=2)
        Full width at half maximum of the Gaussian smoothing kernel. Only used when
        method='gaussian_filter'.
    truncate : float, optional (default=4)
        Truncation parameter (in standard deviations) for the Gaussian filter. Only
        used when method='gaussian_filter'.
    distribution : str, optional (default="gaussian")
        Distribution of the raw noise before smoothing: 'gaussian', 't', or 'chisq'.
        Only used when method='gaussian_filter'.
    noise_df : int or None, optional (default=None)
        Degrees of freedom for the 't' (must be >2, default 3) or 'chisq' (default 5)
        noise distribution. Ignored for the Gaussian distribution.
    trans : str or None, optional (default=None)
        Optional pointwise transformation applied to the field. If 'square', the
        field is squared.

    ---------------------------------------------------------------------------
    OUTPUT:
    X : np.ndarray
        Random field of shape `dim`. If `M_trunc` is not None, values are clipped
        at +/-M_trunc.
    coordinate_ls : list of np.ndarray
        Per-dimension coordinate arrays (centered at 0). Only returned if
        return_coordinate=True.
    ---------------------------------------------------------------------------
    AUTHORS: Jiyue Qin
    ---------------------------------------------------------------------------
    """
    # structured grid with origin at the center
    if isinstance(dim, int):
        dim = (dim,)
    n_dim = len(dim)
    coordinate_ls = [np.arange(dim_i) - dim_i // 2 for dim_i in dim]

    # if seed = None, rng.integers will genereate a random number
    rng = np.random.default_rng(seed)

    if method == "iid":
        X = rng.normal(loc=0, scale=np.sqrt(var), size=dim)
    elif method == "gstools":
        model = gs.Gaussian(dim=n_dim, var=var, len_scale=len_scale)
        srf = gs.SRF(model)
        X = srf.structured(coordinate_ls, seed=rng.integers(0, 1e9))
    elif method == "gaussian_filter":
        std = np.sqrt(var)
        # fwmh = sqrt(8log2)*sigma
        sigma = fwhm / np.sqrt(8 * np.log(2))
        # start with a field of larger dimension for later cropping to the specified dimension
        padding = round(truncate * sigma)
        dim_larger = [dim_i + 2 * padding for dim_i in dim]
        if distribution == "gaussian":
            noise_raw = rng.standard_normal(dim_larger)
            sd_before_smoothing = 1
        elif distribution == "t":
            if noise_df is None:
                noise_df = 3
            noise_raw = rng.standard_t(noise_df, dim_larger)
            if noise_df > 2:
                sd_before_smoothing = np.sqrt(noise_df / (noise_df - 2))
            else:
                sys.exit("Noise degree of freedom needs to be >2 for a t distribution!")
        elif distribution == "chisq":
            if noise_df is None:
                noise_df = 5
            # center by subtracting the mean
            noise_raw = rng.chisquare(noise_df, dim_larger) - noise_df
            sd_before_smoothing = np.sqrt(2 * noise_df)
        # smoothing the noise with gaussian kernel
        noise_raw = gaussian_filter(noise_raw, sigma=sigma, truncate=truncate)
        # rescale the smoothed noise to have sd of the specified sd: std

        # calculate the sd of the field after smoothing for pixels in the center
        sig = np.zeros(dim)
        center = tuple(dim_i // 2 for dim_i in dim)
        sig[center] = 1
        # retrieved the Gaussian kernel used for smoothing
        kernel = gaussian_filter(sig, sigma=sigma, truncate=truncate)
        scale = np.sqrt(
            np.sum(kernel**2)
        )  # scale the smoothed noise field by sqrt(sum(kernel^2))
        sd_after_smoothing = scale * sd_before_smoothing
        slices = tuple(slice(padding, padding + dim_i) for dim_i in dim)
        noise = noise_raw[slices]
        X = noise / sd_after_smoothing * std
    else:
        raise ValueError("Unknown method")

    # optional: truncated field
    if M_trunc is not None:
        X = np.clip(X, -M_trunc, M_trunc)

    if trans == 'square':
        X = X**2
    if return_coordinate:
        return (X, coordinate_ls)
    else:
        return X
