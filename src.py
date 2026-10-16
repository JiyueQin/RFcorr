import numpy as np

import gstools as gs
import pandas as pd
import warnings
from scipy.ndimage import gaussian_filter, gaussian_filter1d
import sys
import numbers


def gen_field(
    dim=(50, 50),
    var=1.0,
    len_scale=1,
    seed=None,
    M_trunc=None,
    return_coordinate=False,
    method="gstools",
    fwhm=2,
    truncate=4,
    distribution="gaussian",
    noise_df = None,
    trans = None
):
    """
    Generate a 0-mean random field with optional truncation.

    Parameters
    ----------
    dim : int or tuple of int, optional (default=(50, 50))
        Dimensions of the field. If int, creates 1D field. If tuple, creates multi-dimensional field.
    var : float, optional (default=1.0)
        Target variance of the field.
    len_scale : float, optional (default=1)
        Length scale parameter of the Gaussian covariance (larger length scale is smoother).
        Only used when ``method='gstools'``.
    seed : int or None, optional (default=None)
        Random seed for reproducibility. If None, uses a random seed.
    M_trunc : float or None, optional (default=None)
        Truncation threshold. If provided, the field is clipped to [-M_trunc, M_trunc].
    return_coordinate : bool, optional (default=False)
        If True, also returns the per-dimension coordinate lists.
    method : str, optional (default="gstools")
        Method to generate the field: 'iid', 'gstools', or 'gaussian_filter'.
    fwhm : float, optional (default=2)
        Full width at half maximum of the Gaussian smoothing kernel.
        Only used when ``method='gaussian_filter'``.
    truncate : float, optional (default=4)
        Truncation parameter (in standard deviations) for the Gaussian filter.
        Only used when ``method='gaussian_filter'``.
    distribution : str, optional (default="gaussian")
        Distribution of the raw noise before smoothing: 'gaussian', 't', or 'chisq'.
        Only used when ``method='gaussian_filter'``.
    noise_df : int or None, optional (default=None)
        Degrees of freedom for the 't' (must be >2, default 3) or 'chisq' (default 5)
        noise distribution. Ignored for the Gaussian distribution.
    trans : str or None, optional (default=None)
        Optional pointwise transformation applied to the field. If 'square',
        the field is squared.

    Returns
    -------
    X : ndarray
        Random field of shape `dim`. If `M_trunc` is not None, values are clipped at ±M_trunc.
    coordinate_ls : list of ndarray
        Per-dimension coordinate arrays (centered at 0). Only returned if
        `return_coordinate=True`.

    Examples
    --------
    >>> X = gen_field(dim=(50, 50), len_scale=5.0, seed=42, M_trunc=2.0)
    >>> X.shape
    (50, 50)
    """
    # structured grid with origin at the center
    # if ny is None:
    #    ny = nx
    # origin = (nx//2, ny//2)
    # x = range(nx)
    # y = range(ny)
    if isinstance(dim, int):
        dim = (dim,)
    n_dim = len(dim)
    coordinate_ls = [np.arange(dim_i) - dim_i // 2 for dim_i in dim]
    # x = np.arange(nx) - origin[0]
    # y = np.arange(nx) - origin[1]

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
        # start with a 2D field of larger dimension for later cropping to the specified dimension
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
        # noise_raw_sd = noise_raw.std(0, ddof=1)
        # print('SD before smoothing across all subjects for each pixel,', 'mean:', noise_raw_sd.mean(), 'max:', noise_raw_sd.max(), 'min:', noise_raw_sd.min())
        # smoothing the noise with gaussian kernel
        noise_raw = gaussian_filter(noise_raw, sigma=sigma, truncate=truncate)
        # rescale the smoothed noise to have sd of the specified sd: std

        # calculate the sd of the 2d field after smoothing for pixels in the center
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
        # noise = noise_raw[:, padding:padding + dim[1], padding:padding + dim[2]]
        noise = noise_raw[slices]
        X = noise / sd_after_smoothing * std
    else:
        raise ValueError("Unknown method")

    # optional: correlated pair of truncated fields
    if M_trunc is not None:
        X = np.clip(X, -M_trunc, M_trunc)
    
    if trans == 'square':
        X = X**2 
    if return_coordinate:
        return (X, coordinate_ls)
    else:
        return X


def gaussian_filter_acf(h, fwhm, truncate=4):
    """
    Population autocorrelation at lag h of a field from ``gen_field(method='gaussian_filter')``.

    The field is i.i.d. noise convolved with a discrete Gaussian kernel K, so its correlation at
    lag h is sum_u K(u) K(u + h) / sum_u K(u)^2, whatever the noise distribution. The kernel is a
    product of 1D kernels, so the correlation is the product of the 1D correlations over axes.

    Parameters
    ----------
    h : int or tuple of int
        Lag. An int is a lag along a 1D axis.
    fwhm : float
        Full width at half maximum of the Gaussian kernel, as in `gen_field`.
    truncate : float, optional (default=4)
        Truncation of the kernel in standard deviations, as in `gen_field`.

    Returns
    -------
    r : float
        Correlation between the field at a site and at the site shifted by h.

    Examples
    --------
    >>> round(gaussian_filter_acf((1, 0), fwhm=3), 3)
    0.857
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


def gen_corr_field(
    rho,
    dim=(50, 50),
    var=1.0,
    len_scale=10.0,
    seed=None,
    M_trunc=None,
    return_coordinate=False,
    method="gaussian_filter",
    fwhm=4,
    truncate=4,
    distribution="gaussian",
    noise_df = None,
    fwhm_ratio = 1,
    trans = None,
    dep_model = None,
    dep_strength = 1.0,
    shift_lag = 1,
):
    """
    Generate a pair of correlated random fields with optional truncation.

    Parameters
    ----------
    rho : float
        Desired pointwise correlation between the two fields (between -1 and 1).
    dim : int or tuple of int, optional (default=(50, 50))
        Dimensions of the fields. If int, creates 1D fields. If tuple, creates multi-dimensional fields.
    var : float, optional (default=1.0)
        Target variance of each field.
    len_scale : float, optional (default=10.0)
        Length scale parameter of the Gaussian covariance (larger length scale is smoother).
    seed : int or None, optional (default=None)
        Random seed for reproducibility. If None, uses a random seed.
    M_trunc : float or None, optional (default=None)
        Truncation threshold. If provided, both fields are clipped to [-M_trunc, M_trunc].
    return_coordinate : bool, optional (default=False)
        If True, returns fields and coordinate lists.
    method : str, optional (default="gaussian_filter")
        Method to generate fields: 'iid', 'gstools', or 'gaussian_filter'.
    fwhm : float, optional (default=4)
        Full width at half maximum for the Gaussian filter method.
    truncate : float, optional (default=4)
        Truncation parameter for Gaussian filter.
    distribution : str, optional (default="gaussian")
        Distribution type: 'gaussian', 't', or 'chisq'.
    noise_df : int or None, optional (default=None)
        Degrees of freedom for the 't' or 'chisq' noise distribution.
    fwhm_ratio : float, optional (default=1)
        Ratio of the two fields' smoothing: the second field V is generated with
        ``fwhm / fwhm_ratio``, allowing X and Y to have different smoothness.
    trans : str or None, optional (default=None)
        Optional pointwise transformation applied to each field (e.g. 'square').
    dep_model : str or None, optional (default=None)
        If None, Y = rho U + sqrt(1 - rho^2) V with U, V independent. Otherwise V is replaced by
        a field W that is uncorrelated with U at every pair of locations but depends on U: W is
        the residual of a nonlinear field on X = U with the population slope, rescaled to unit
        variance. With lam = dep_strength and U, V standardized,
        'quad': W = lam (U^2 - 1) / sqrt(2) + sqrt(1 - lam^2) V, the residual of
        U + W on U (slope 1).
        'exp': W = lam (exp(U) - e^(1/2) - e^(1/2) U) / sqrt(e (e - 2)) + sqrt(1 - lam^2) V,
        the residual of exp(U) on U (slope e^(1/2)) mixed with V.
        'shift': W = lam D / sd(D) * sd(U) + sqrt(1 - lam^2) V, where
        D_s = U_{s+h} - r_h U_s, h is ``shift_lag``, r_h = corr(U_s, U_{s+h}) is the population
        lag-h correlation of U (from `gaussian_filter_acf`, squared for trans='square'), and sd(D)
        is the population standard deviation of D. Then cov(U_s, D_s) = 0 exactly, while D is
        correlated with U at other lags. Unlike 'quad' and 'exp', the dependence is linear in
        U, so with trans=None and Gaussian noise X and Y are jointly Gaussian. Works with
        trans=None (any noise distribution) or trans='square' (Gaussian noise, the chi^2_1
        setting), and requires method='gaussian_filter'.
        'symmetric_shift': as 'shift' with the symmetric D_s = (U_{s+h} + U_{s-h}) / 2 - r_h U_s,
        whose variance is var(U) ((1 + r_2h) / 2 - r_h^2). Again cov(U_s, D_s) = 0, but the
        cross-correlation of X and Y is even in the lag, so the cross term
        sum_k rho_xy,k rho_xy,-k of the variance of the sample correlation is positive and tests
        assuming independence (Clifford-type) are anti-conservative. With 'shift' the cross term
        is negative and those tests are conservative.
        So rho = 0 gives X and Y uncorrelated but dependent. 'quad' and 'exp' require Gaussian
        U and trans=None.
    dep_strength : float, optional (default=1.0)
        Weight lam in [0, 1] of the dependent part of W; lam = 0 gives W = V (independence).
        Only used when ``dep_model`` is not None.
    shift_lag : int or tuple of int, optional (default=1)
        Lag h for ``dep_model='shift'`` or ``'symmetric_shift'``. An int k means a shift of k
        grid points along the first axis. Only used for these two models.

    Returns
    -------
    X, Y : ndarray
        Two correlated random fields of shape `dim`.
        If `M_trunc` is not None, the arrays are clipped at ±M_trunc.
    coordinate_ls : list of ndarray
        Per-dimension coordinate arrays. Only returned if `return_coordinate=True`.

    Examples
    --------
    >>> X, Y = gen_corr_field(rho=0.7, dim=(50, 50), len_scale=5.0, seed=42)
    >>> X.shape, Y.shape
    ((50, 50), (50, 50))
    """
    # structured grid with origin at the center
    # if ny is None:
    #    ny = nx
    # origin = (nx//2, ny//2)
    # x = range(nx)
    # y = range(ny)
    if isinstance(dim, int):
        dim = (dim,)
    n_dim = len(dim)
    coordinate_ls = [np.arange(dim_i) - dim_i // 2 for dim_i in dim]
    # x = np.arange(nx) - origin[0]
    # y = np.arange(nx) - origin[1]

    # if seed = None, rng.integers will genereate a random number
    rng = np.random.default_rng(seed)

    # for the shift designs, U is generated on a larger grid so that the shifted values exist at
    # every site of the returned grid: by |h| on one side for 'shift' (U_{s+h}) and on both sides
    # for 'symmetric_shift' (U_{s+h} and U_{s-h})
    if dep_model in ("shift", "symmetric_shift"):
        if isinstance(shift_lag, numbers.Integral):
            h = (int(shift_lag),) + (0,) * (n_dim - 1)
        else:
            h = tuple(int(x) for x in shift_lag)
        if len(h) != n_dim or all(x == 0 for x in h):
            raise ValueError("shift_lag must be a nonzero lag with one entry per dimension")
        n_sides = 1 if dep_model == "shift" else 2
        dim_U = tuple(n + n_sides * abs(x) for n, x in zip(dim, h))
    else:
        dim_U = dim

    U = gen_field(
        dim=dim_U,
        var=var,
        len_scale=len_scale,
        seed=rng.integers(0, 1e9),
        M_trunc=None,
        return_coordinate=False,
        method=method,
        fwhm=fwhm,
        truncate=truncate,
        distribution=distribution,
        noise_df = noise_df, 
        trans = trans 
    )
    V = gen_field(
        dim=dim,
        var=var,
        len_scale=len_scale,
        seed=rng.integers(0, 1e9),
        M_trunc=None,
        return_coordinate=False,
        method=method,
        fwhm=fwhm/fwhm_ratio,
        truncate=truncate,
        distribution=distribution,
        noise_df = noise_df, 
        trans = trans
    )
    if dep_model in ("shift", "symmetric_shift"):
        # a missing value in a settings table (NaN) means no transformation
        if isinstance(trans, float) and np.isnan(trans):
            trans = None
        if method != "gaussian_filter":
            raise ValueError(f"dep_model='{dep_model}' requires method='gaussian_filter'")
        if trans not in (None, "square"):
            raise ValueError(f"dep_model='{dep_model}' requires trans=None or 'square'")
        if trans == "square" and distribution != "gaussian":
            raise ValueError(f"dep_model='{dep_model}' with trans='square' requires Gaussian noise")
        lam = dep_strength
        # population variance and autocorrelations of U at lags h and 2h; for U = Z^2 with
        # Gaussian Z, var(U) = 2 var^2 and corr(Z_0^2, Z_k^2) = r_k^2
        r_h = gaussian_filter_acf(h, fwhm, truncate)
        r_2h = gaussian_filter_acf(tuple(2 * x for x in h), fwhm, truncate)
        if trans == "square":
            var_U, r_U, r_U2 = 2 * var**2, r_h**2, r_2h**2
        else:
            var_U, r_U, r_U2 = var, r_h, r_2h
        # the population r_U makes cov(U_s, D_s) = 0 exactly; an estimate from the sample
        # would force the sample correlation towards 0
        if dep_model == "shift":
            # D_s = U_{s+h} - r U_s; base sites s and shifted sites s + h inside the enlarged
            # grid (handles negative lags)
            off = [max(0, -x) for x in h]
            base = tuple(slice(o, o + n) for o, n in zip(off, dim))
            shifted = tuple(slice(o + x, o + x + n) for o, x, n in zip(off, h, dim))
            D = U[shifted] - r_U * U[base]
            var_D = var_U * (1 - r_U**2)
        else:
            # D_s = (U_{s+h} + U_{s-h}) / 2 - r U_s, symmetric in the lag
            off = [abs(x) for x in h]
            base = tuple(slice(o, o + n) for o, n in zip(off, dim))
            plus = tuple(slice(o + x, o + x + n) for o, x, n in zip(off, h, dim))
            minus = tuple(slice(o - x, o - x + n) for o, x, n in zip(off, h, dim))
            D = (U[plus] + U[minus]) / 2 - r_U * U[base]
            var_D = var_U * ((1 + r_U2) / 2 - r_U**2)
        U = U[base]
        # D rescaled to the variance of U; V has the same variance as U, so W does too
        W = lam * D * np.sqrt(var_U / var_D)
        V = W + np.sqrt(max(0.0, 1.0 - lam**2)) * V
    elif dep_model is not None:
        if distribution != "gaussian" or trans is not None:
            raise ValueError("dep_model requires distribution='gaussian' and trans=None")
        # population moments below are for standard normal U, V
        sd = np.sqrt(var)
        U_std, V_std = U / sd, V / sd
        lam = dep_strength
        if dep_model == "quad":
            W = lam * (U_std**2 - 1) / np.sqrt(2)
        elif dep_model == "exp":
            e = np.exp(1)
            W = lam * (np.exp(U_std) - np.sqrt(e) - np.sqrt(e) * U_std) / np.sqrt(e * (e - 2))
        else:
            raise ValueError("dep_model must be None, 'quad', 'exp', 'shift', or 'symmetric_shift'")
        # unit-variance residual, uncorrelated with U but dependent on it when lam > 0
        V = (W + np.sqrt(max(0.0, 1.0 - lam**2)) * V_std) * sd

    beta = np.sqrt(max(0.0, 1.0 - rho**2))
    # correlated pair of fields
    X = U
    Y = rho * U + beta * V

    # optional: correlated pair of truncated fields
    if M_trunc is not None:
        X = np.clip(X, -M_trunc, M_trunc)
        Y = np.clip(Y, -M_trunc, M_trunc)
    if return_coordinate:
        return (X, Y, coordinate_ls)
    else:
        return (X, Y)


def gen_mask(dim, shape="ellipse", axes=None):
    """
    Generate a fixed mask of valid grid points.

    Parameters
    ----------
    dim : int or tuple of int
        Dimensions of the grid.
    shape : str, optional (default="ellipse")
        Mask shape. Only 'ellipse' is supported: an ellipse (ellipsoid for n_dim > 2) centered
        in the grid with its axes along the grid axes.
    axes : float, tuple of float, or None, optional (default=None)
        Semi-axis in each dimension as a fraction of half the side length, in (0, 1]. A float
        uses the same fraction in every dimension. If None, all fractions are 1, i.e. the
        ellipse inscribed in the grid.

    Returns
    -------
    mask : ndarray of bool
        True at valid grid points, shape `dim`.

    Examples
    --------
    >>> mask = gen_mask((50, 50), axes=(1, 0.5))
    >>> mask.shape
    (50, 50)
    """
    if isinstance(dim, numbers.Integral):
        dim = (dim,)
    n_dim = len(dim)
    if shape != "ellipse":
        raise ValueError("shape must be 'ellipse'")
    if axes is None:
        axes = (1.0,) * n_dim
    elif isinstance(axes, numbers.Number):
        axes = (axes,) * n_dim
    if len(axes) != n_dim:
        raise ValueError("axes must have one entry per dimension")

    # centered coordinates scaled by the semi-axes, so the ellipse is the unit ball
    coords = np.indices(dim, dtype=float)
    dist2 = sum(
        ((coords[i] - (dim[i] - 1) / 2) / (axes[i] * dim[i] / 2)) ** 2 for i in range(n_dim)
    )
    return dist2 <= 1


import numpy as np
from scipy.signal import correlate2d, correlate


def est_exp_X0Yh(X, Y, return_counts_lags=False, mask=None):
    """
    Estimate E[X_0 * Y_h] for all spatial integer lags h on a grid.

    Parameters
    ----------
    X, Y : ndarray (same shape)
        Input fields of any dimensionality. If `mask` is given, X and Y must be 0 outside it.
    return_counts_lags : bool, optional (default=False)
        If False, returns only E. If True, returns (S, C, E, lag_ls).
    mask : ndarray of bool or None, optional (default=None)
        Valid grid points. If None, every grid point is valid. Lags with no valid pair get
        E = NaN, since there is nothing to average.

    Returns
    -------
    E : ndarray
        Estimated E[X_0 Y_h] at each lag h. Shape is (2*d_i-1 for each dimension).
    S : ndarray (if return_counts_lags=True)
        Sum of products at each lag.
    C : ndarray (if return_counts_lags=True)
        Number of valid pairs at each lag (edge correction).
    lag_ls : list of ndarray (if return_counts_lags=True)
        Lag coordinates for each dimension.
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
    # max number of pairs
    # C = nx*ny
    # estimate of the expectaion; with a mask some lags have no valid pair (C is 0 up to
    # FFT round-off), and those are NaN so get_se can drop them
    E = np.divide(S, C, out=np.full_like(S, np.nan), where=C > 0.5)
    if not return_counts_lags:
        return E
    else:
        lag_ls = [np.arange(-(ni - 1), ni) for ni in dim]
        # lags_x = np.arange(-(nx-1), nx)
        # lags_y = np.arange(-(ny-1), ny)
        return S, C, E, lag_ls


def kernel(x, type="bartlett"):
    """
    Kernel function for lag weighting.

    Parameters
    ----------
    x : float or ndarray
        Input value(s).
    type : str, optional (default="bartlett")
        Kernel type: 'bartlett', 'qs' (Quadratic Spectral), or 'constant'.

    Returns
    -------
    out : float or ndarray
        Kernel value(s) at x.
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


def kernel_weights(x, type, bandwidth, M=None):
    """
    Compute kernel weights for lag weighting.

    Parameters
    ----------
    x : ndarray
        Lag values.
    type : str
        Kernel type: 'bartlett', 'qs', or 'constant'.
    bandwidth : float
        Bandwidth parameter for the kernel.
    M : float or None, optional (default=None)
        Maximum lag cutoff.

    Returns
    -------
    weights : ndarray
        Kernel weights at each lag value.
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
    return kernel(y, type=type) * mask


import time
import numbers
from functools import reduce


# se_method names follow the method names in the paper. Each gives two standard errors,
# `se` (at the plugin rho) and `se_under_null` (at rho = 0):
#   'general' -> RFCorr                                and RFCorr-0
#   'gauss'   -> RFCorr-Gauss                          and RFCorr-Gauss-0
#   'indep'   -> (1 - rho^2)^2 times RFCorr-Indep-0    and RFCorr-Indep-0
#   'iid'     -> (1 - rho^2)^2 / n, ignoring spatial dependence
SE_METHODS = ("general", "gauss", "indep", "iid")
# earlier names, still accepted so saved method tables keep working
SE_METHOD_ALIASES = {"hac": "general", "hac_gaussian": "gauss", "huristic": "indep"}


def canonical_se_method(se_method):
    """Return the se_method name used in the paper, mapping earlier names to it."""
    name = str(se_method).lower()
    name = SE_METHOD_ALIASES.get(name, name)
    if name not in SE_METHODS:
        raise ValueError(f"Unknown se_method {se_method!r}; choose from {SE_METHODS}.")
    return name


def corr(
    X,
    Y,
    M="cubic_root",
    true_rho=None,
    se_method="general",
    kernel="bartlett",
    check=False,
    b_qs=5,
    constant=1,
    standardize = True,
    mask=None,
):
    """
    Estimate correlation and its standard error for two fields.

    Parameters
    ----------
    X, Y : ndarray (same shape)
        Input fields to correlate.
    M : str or numeric, optional (default="cubic_root")
        Lag truncation method: 'cubic_root', 'linear', 'auto', or numeric value.
    true_rho : float or None, optional (default=None)
        Known true correlation for plugin estimation.
    se_method : str, optional (default="general")
        Standard error method: 'general', 'gauss', 'indep', or 'iid' (see
        SE_METHODS). The earlier names 'hac', 'hac_gaussian', and 'huristic' are also accepted.
    kernel : str, optional (default="bartlett")
        Kernel type for HAC: 'bartlett', 'qs', 'constant', or 'bartlett_radial' (see `get_se`).
    check : bool, optional (default=False)
        If True, print diagnostic information.
    b_qs : float, optional (default=5)
        Bandwidth parameter for QS kernel.
    constant : float, optional (default=1)
        Scaling constant for M calculation.
    standardize : bool, optional (default=True)
        If True, standardize X and Y (subtract mean, divide by std) before
        computing the standard error.
    mask : ndarray of bool or None, optional (default=None)
        Valid grid points, same shape as X and Y. Only these points enter the estimate and its
        standard error; values outside the mask (including NaN) are ignored. If None, every grid
        point is valid.

    Returns
    -------
    rho_est : float
        Estimated Pearson correlation.
    se : float
        Standard error of the correlation estimate (at the plugin rho), e.g. RFCorr.
    se_under_null : float
        Standard error computed under the null hypothesis rho = 0, e.g. RFCorr-0.
    runtime_secs : float
        Computation time in seconds.
    M_tuple : tuple
        Lag truncation values used.
    """
    start_time = time.time()
    if X.shape != Y.shape:
        raise ValueError("X and Y must have the same shape")
    se_method = canonical_se_method(se_method)
    dim = np.array(X.shape)
    if mask is None:
        rho_est = np.corrcoef(X.ravel(), Y.ravel())[0, 1]
        if standardize:
            X_standard =  (X-X.mean())/X.std()
            Y_standard =  (Y-Y.mean())/Y.std()
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
    # print('rho_plugin:', rho_plugin)
    est_exp_ls = get_est_exp_ls(se_method, X_standard, Y_standard, mask=mask)

    se, M_tuple = get_se(
        dim,
        est_exp_ls,
        rho_plugin,
        M,
        se_method,
        kernel,
        check,
        b_qs,
        constant,
        n_obs=n_obs,)


    # under the null, the Gaussian method also sets the lag-0 cross-correlation to 0
    if se_method == "gauss":
        est_exp_ls_null = get_est_exp_ls(
            se_method, X_standard, Y_standard, mask=mask, impose_null=True
        )
    else:
        est_exp_ls_null = est_exp_ls
    se_under_null, _ = get_se(
        dim,
        est_exp_ls_null,
        0,
        M,
        se_method,
        kernel,
        check,
        b_qs,
        constant,
        n_obs=n_obs,)
    

    runtime_secs = time.time() - start_time
    return rho_est, se, se_under_null,  runtime_secs, M_tuple


from scipy.stats import norm


def plugin_bandwidth(cov_ls, vec, n, dim, constant=1, r_max=0.97):
    """
    Data-driven lag truncation for the product Bartlett kernel (parametric plug-in).

    The variance being estimated is the long-run variance of the scalar series V = vec' T,
    whose lag-h autocovariance is vec' C(h) vec. Along each axis i, V is approximated by an
    AR(1) with lag-1 autocorrelation r_i (from the same lag estimates used for the variance),
    giving the bias constant a_i = 2 r_i / (1 - r_i^2) (Andrews 1991). With one bandwidth m for
    all axes, the asymptotic MSE of the Bartlett estimator, proportional to
    (sum_i a_i / m)^2 + (2 / n) (2 m / 3)^n_dim, is minimized at
    m = [(sum_i a_i)^2 n / (n_dim (2/3)^n_dim)]^(1 / (n_dim + 2)),
    which in 1D is Andrews' 1.1447 (a^2 n)^(1/3). Smoother fields give larger r_i and so a larger m.

    Parameters
    ----------
    cov_ls : list of ndarray
        The 9 lag-wise covariance arrays of T (the 3x3 entries by row), as in `get_se`.
    vec : ndarray
        Weights of the 3 components of T in V.
    n : int
        Number of valid grid points.
    dim : sequence of int
        Shape of the grid.
    constant : float, optional (default=1)
        Scaling constant for the bandwidth.
    r_max : float, optional (default=0.97)
        Upper bound on |r_i|, so that nearly nonstationary fields do not give an unbounded lag.

    Returns
    -------
    M_tuple : ndarray
        Lag truncation in each dimension. The Bartlett weights in `kernel_weights` are
        1 - |h| / (M + 1), so M = m - 1, rounded up, at least 1, and at most half the side.
    """
    n_dim = len(dim)
    center = [d - 1 for d in dim]

    def acov(idx):
        # vec' C(h) vec at the lag with array index idx; entries can be scalars (zero terms)
        return sum(
            vec[a] * vec[b] * (np.asarray(cov_ls[3 * a + b])[tuple(idx)]
                               if np.ndim(cov_ls[3 * a + b]) else cov_ls[3 * a + b])
            for a in range(3) for b in range(3)
        )

    s0 = acov(center)
    a_sum = 0.0
    for i in range(n_dim):
        if dim[i] < 2 or not s0 > 0:
            continue
        plus, minus = list(center), list(center)
        plus[i] += 1
        minus[i] -= 1
        r = np.nan_to_num((acov(plus) + acov(minus)) / (2 * s0))
        r = min(abs(r), r_max)
        a_sum += 2 * r / (1 - r**2)

    m = (a_sum**2 * n / (n_dim * (2 / 3) ** n_dim)) ** (1 / (n_dim + 2))
    M = max(1.0, np.ceil(constant * m - 1))
    return np.array([min(M, max(1, (d - 1) // 2)) for d in dim], dtype=float)


def get_se(
    dim,
    est_exp_ls,
    rho_plugin,
    M="cubic_root",
    se_method="general",
    kernel="bartlett",
    check=False,
    b_qs=5,
    constant=1,
    n_obs=None,
):
    """
    Compute the HAC standard error of the correlation estimate.

    Parameters
    ----------
    dim : ndarray
        Shape of the field (per-dimension sizes).
    est_exp_ls : list of ndarray
        Lag-wise expectation estimates produced by `get_est_exp_ls`.
    rho_plugin : float
        Plugin correlation used in the variance formula (use 0 for the null SE).
    M : str or numeric, optional (default="cubic_root")
        Lag truncation method: 'cubic_root', 'linear', 'auto', 'plugin', an int, or a tuple of
        numerics. The string rules use the same lag in every dimension, computed from the number
        of valid grid points n rather than the side lengths. With side = n ** (1 / n_dim),
        'cubic_root' is ceil(constant * side ** 0.3), 'linear' is floor(constant * side / 5), and
        'auto' is ceil(constant * side ** (n_dim / (n_dim + 2))) = ceil(constant * n ** (1 / (n_dim + 2))).
        'plugin' is data driven and grows with the smoothness of the fields (see `plugin_bandwidth`).
    se_method : str, optional (default="general")
        Standard error method: 'general', 'gauss', 'indep', or 'iid' (see
        SE_METHODS). The earlier names 'hac', 'hac_gaussian', and 'huristic' are also accepted.
    kernel : str, optional (default="bartlett")
        Kernel type for HAC weighting: 'bartlett', 'qs', or 'constant' (products of 1D weights
        over axes), or 'bartlett_radial', the radial Bartlett weights 1 - ||h / (M + 1)|| (with
        lags scaled per axis), as in Kim and Sun (2011).
    check : bool, optional (default=False)
        If True, print diagnostic information (omega and weights).
    b_qs : float, optional (default=5)
        Bandwidth parameter for the QS kernel.
    constant : float, optional (default=1)
        Scaling constant for the M calculation.
    n_obs : int or None, optional (default=None)
        Number of valid grid points (the size of the mask). If None, every grid point is valid.

    Returns
    -------
    se : float
        Standard error of the correlation estimate (np.nan if the estimated
        variance is negative).
    M_tuple : tuple
        Lag truncation values used.
    """
    n = np.prod(dim) if n_obs is None else n_obs
    n_dim = len(dim)
    se_method = canonical_se_method(se_method)
    if se_method == "indep":
        # pre-compute the factor before setting rho_plugin to be 0
        factor = (1 - rho_plugin**2) ** 2 
        rho_plugin = 0
        kernel = "bartlett"

    if se_method == "iid":
        var = (1 - rho_plugin**2) ** 2 / n
        M_tuple = (None,)
    else:
        if se_method in ["gauss", "indep"]:
            cov_ls = est_exp_ls
        else:
            offsets = [
                rho_plugin**2,
                rho_plugin,
                rho_plugin,
                rho_plugin,
                1,
                1,
                rho_plugin,
                1,
                1,
            ]
            cov_ls = [e - o for e, o in zip(est_exp_ls, offsets)]
        vec = np.array([1, -0.5 * rho_plugin, -0.5 * rho_plugin])

        # --------weight function and truncation lag numbder -----------
        # the rules below give the same lag in every dimension, from the domain size n rather
        # than side lengths: they apply the per-side rules to the side of a cube with n points.
        # snap that side to an integer when it is one (up to float error), so on cubes the
        # results are exactly those of the per-side rules
        side = n ** (1 / n_dim)
        if np.isclose(side, np.round(side), rtol=1e-12, atol=0):
            side = np.round(side)
        if M == "cubic_root":
            M_tuple = np.full(n_dim, np.ceil(side**0.3 * constant))
        elif M == "linear":
            M_tuple = np.full(n_dim, np.floor(side / 5 * constant))
            # M_tuple = np.floor((nx**0.33, ny**0.33))*1.4
        elif M == "auto":
            M_tuple = np.full(n_dim, np.ceil(side ** (n_dim / (n_dim + 2)) * constant))
        elif M == "plugin":
            M_tuple = plugin_bandwidth(cov_ls, vec, n, dim, constant)
        elif isinstance(M, int):
            # same truncation lag in every dimension
            M_tuple = (M,) * n_dim
        elif isinstance(M, tuple) and all(isinstance(x, numbers.Number) for x in M):
            M_tuple = M
        else:
            raise ValueError("M must be 'cubic_root', 'linear', 'auto', 'plugin', or numeric.")

        lag_ls = [np.arange(-(dim_i - 1), dim_i) for dim_i in dim]

        if kernel == "bartlett_radial":
            # radial Bartlett weights 1 - ||h / (M + 1)||, with lags scaled per axis as in the
            # product weights, so that the same M gives the same reach along each axis
            scaled = np.meshgrid(*[lag_ls[i] / (M_tuple[i] + 1) for i in range(n_dim)],
                                 indexing="ij")
            weights = np.maximum(0.0, 1 - np.sqrt(sum(s**2 for s in scaled)))
        else:
            weights_ls = [
                kernel_weights(lag_ls[i], type=kernel, bandwidth=b_qs, M=M_tuple[i])
                for i in range(n_dim)
            ]
            weights = reduce(np.multiply, np.ix_(*weights_ls))

        # print(weights.shape)
        # print(X_standard.mean())
        # print(X_standard.std())

        # nansum drops lags with no valid pair under a mask (NaN from est_exp_X0Yh)
        omega = np.array([np.nansum(weights * cov) for cov in cov_ls]).reshape((3, 3))
        var = (vec @ omega @ vec) / n
        if se_method == "indep":
            var = factor * var
        # print(X_standard_Y_standard.shape)
        # if true_rho == 0:
        #     T1= T2 = 0
        # else:
        #     # () needed for linebreak
        #     T1 = (est_exp_X0Yh(X_standard_square, X_standard_square) + est_exp_X0Yh(Y_standard_square, Y_standard_square) +
        #           est_exp_X0Yh(X_standard_square, Y_standard_square) + est_exp_X0Yh(Y_standard_square, X_standard_square))
        #     T2 = (est_exp_X0Yh(X_standard_square, X_standard_Y_standard) + est_exp_X0Yh(X_standard_Y_standard, X_standard_square) +
        #           est_exp_X0Yh(X_standard_Y_standard, Y_standard_square) + est_exp_X0Yh(Y_standard_square, X_standard_Y_standard))
        # T3 = est_exp_X0Yh(X_standard_Y_standard, X_standard_Y_standard)
        # print(T1,T2,T3)
        # print(np.sqrt(np.sum(T3)/(nx*ny)))

        # h  = weights*(1/4*(rho_est**2)*T1 -1/2*rho_est*T2 +T3)
        if check:
            print("omega:", omega)
            print("weights:", weights)
        #    print('T1:', T1, '\nT2', T2, '\nT3', T3)
        # print(h)
        # print(h.shape)
        # print(np.max(h))
        # var = np.sum(h)/(nx*ny)
    # print(var)
    if var < 0:
        warnings.warn("Estimated variance is negative. Check!")
        se = np.nan
    else:
        se = np.sqrt(var)
    return se, M_tuple



def get_est_exp_ls(se_method, X_standard, Y_standard, mask=None, impose_null=False):
    """
    Compute the lag-wise expectation estimates entering the HAC variance.

    Parameters
    ----------
    se_method : str
        Standard error method: 'general', 'gauss', 'indep', or 'iid' (see
        SE_METHODS); the earlier names 'hac', 'hac_gaussian', and 'huristic' are also accepted.
        Determines which set of cross-product expectations is computed.
    X_standard, Y_standard : ndarray (same shape)
        The (optionally standardized) input fields, 0 outside `mask`.
    mask : ndarray of bool or None, optional (default=None)
        Valid grid points. If None, every grid point is valid.
    impose_null : bool, optional (default=False)
        Only used for 'gauss'. If True, set the lag-0 cross-correlation (which equals the sample
        correlation) to 0, so that the null rho = 0 is imposed in the Gaussian covariance
        structure as well (RFCorr-Gauss-0 in the paper).

    Returns
    -------
    est_exp_ls : list of ndarray
        List of 9 lag-wise expectation arrays, ordered to fill the 3x3 omega
        matrix by row in `get_se`.
    """
    se_method = canonical_se_method(se_method)
    if se_method in ["gauss", "indep"]:
        rho_xx_h = est_exp_X0Yh(X_standard, X_standard, mask=mask)
        rho_yy_h = est_exp_X0Yh(Y_standard, Y_standard, mask=mask)
        if se_method == "indep":
            rho_xy_h = 0
            rho_yx_h = 0
        else:
            rho_xy_h = est_exp_X0Yh(X_standard, Y_standard, mask=mask)
            rho_yx_h = est_exp_X0Yh(Y_standard, X_standard, mask=mask)
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
            est_exp_X0Yh(X_standard_Y_standard, X_standard_Y_standard, mask=mask),
            est_exp_X0Yh(X_standard_Y_standard, X_standard_square, mask=mask),
            est_exp_X0Yh(X_standard_Y_standard, Y_standard_square, mask=mask),
            est_exp_X0Yh(X_standard_square, X_standard_Y_standard, mask=mask),
            est_exp_X0Yh(X_standard_square, X_standard_square, mask=mask),
            est_exp_X0Yh(X_standard_square, Y_standard_square, mask=mask),
            est_exp_X0Yh(Y_standard_square, X_standard_Y_standard, mask=mask),
            est_exp_X0Yh(Y_standard_square, X_standard_square, mask=mask),
            est_exp_X0Yh(Y_standard_square, Y_standard_square, mask=mask),
        ]
    return est_exp_ls


def confidence_interval(rho_est, se, alpha=0.05, fisher_trans=True):
    """
    Compute a confidence interval for a correlation estimate.

    Parameters
    ----------
    rho_est : float or array-like
        Estimated Pearson correlation(s).
    se : float or array-like
        Standard error(s) of `rho_est` on the original (untransformed) scale.
    alpha : float, optional (default=0.05)
        Significance level; the returned interval has coverage 1 - alpha.
    fisher_trans : bool, optional (default=True)
        If True, build the interval on the Fisher z scale (z = arctanh(rho))
        and back-transform via tanh. The SE on the z scale is se / (1 - rho_est**2).
        If False, return the symmetric Wald interval rho_est +/- z * se,
        clipped to [-1, 1].

    Returns
    -------
    lower, upper : float or ndarray
        Lower and upper bounds of the (1 - alpha) confidence interval.
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

def pvalue(rho_est, se_under_null, fisher_trans=True):
    """
    Two-sided p-value for testing H0: rho = 0.

    Parameters
    ----------
    rho_est : float or array-like
        Estimated Pearson correlation(s).
    se_under_null : float or array-like
        Standard error(s) of `rho_est` computed under the null (rho = 0),
        on the original (untransformed) scale.
    fisher_trans : bool, optional (default=True)
        If True, build the test statistic on the Fisher z scale
        (F = arctanh(rho_est)) with SE on that scale equal to
        se_under_null / (1 - rho_est**2). If False, use the plain Wald
        statistic Z = rho_est / se_under_null.

    Returns
    -------
    p : float or ndarray
        Two-sided p-value(s): 2 * (1 - Phi(|Z|)).
    """
    rho_est = np.asarray(rho_est, dtype=float)
    se_under_null = np.asarray(se_under_null, dtype=float)

    if fisher_trans:
        F = np.arctanh(rho_est)
        # 1 - rho_est**2 = 0 under null
        F_se_under_null = se_under_null 
        Z = F / F_se_under_null
    else:
        Z = rho_est / se_under_null

    p = 2 * (1 - norm.cdf(np.abs(Z)))

    if p.ndim == 0:
        return float(p)
    return p

def results_summary(results, group=None):
    """
    Summarize simulation results by aggregating across simulations.

    Parameters
    ----------
    results : pd.DataFrame
        Results DataFrame with columns: bias, est, se, cover, cover_raw,
        reject, reject_raw, runtime (and, when grouping, the per-method/setting
        identifier columns).
    group : list of str or None, optional (default=None)
        Columns to group by. If None, returns a single-row summary across all rows.

    Returns
    -------
    summary : pd.DataFrame
        Aggregated results with columns: bias_mean, est_sd, se_mean, cover_rate,
        cover_rate_raw, power, power_raw, runtime_mean. When `group` is provided,
        also includes n_se_na (count of NaN SEs) and the grouping columns.
    """
    if group is None:
        out = pd.DataFrame(
            {
                "bias_mean": results["bias"].mean(),
                "est_sd": results["est"].std(),
                "se_mean": results["se"].mean(),
                "cover_rate": results["cover"].mean(),
                "cover_rate_raw": results["cover_raw"].mean(),
                "power": results["reject"].mean(),
                "power_raw": results["reject_raw"].mean(),
                "runtime_mean": results["runtime"].mean(),
            },
            index=[0],
        )
    else:
        setting_method_df = results.drop(
            columns=[
                "sim_id",
                "lower",
                "upper",
                "lower_raw",
                "upper_raw",
                "reject",
                "reject_raw",
                "p",
                "p_raw",
                "bias",
                "est",
                "se",
                "se_under_null",
                "cover",
                "cover_raw",
                "runtime",
            ]
        ).drop_duplicates()
        # out = results.groupby(['setting_id', 'rho', 'nx', 'ny', 'len_scale', 'm_sim', 'method', 'M0', 'M1', 'b_qs', 'M_method'],
        #                       dropna=False).agg({'bias':'mean', 'est': 'std', 'se': 'mean', 'cover': 'mean', 'runtime': 'mean'})
        out = results.groupby(group).agg(
            {
                "bias": "mean",
                "est": "std",
                "se": ["mean", lambda x: x.isna().sum()],
                "cover": "mean",
                "cover_raw": "mean",
                "reject": "mean",
                "reject_raw": "mean",
                "runtime": "mean",
            }
        )
        out.columns = [
            "bias_mean",
            "est_sd",
            "se_mean",
            "n_se_na",
            "cover_rate",
            "cover_rate_raw",
            "power",
            "power_raw",
            "runtime_mean",
        ]
        out = out.reset_index()
        out = pd.merge(setting_method_df, out, on=group, how="inner")
    return out


def sim_single(
    method_df,
    rho,
    m_sim=5,
    nx=50,
    ny=None,
    n_dim=2,
    var=1.0,
    len_scale=10.0,
    seed=None,
    M_trunc=None,
    return_summary=True,
    gen_method="gstools",
    fwhm=2,
    truncate=4,
    distribution="gaussian",
    noise_df = None,
    trans = None,
    mask=None,
    mask_axes=None,
    fwhm_ratio=1,
    dep_model=None,
    dep_strength=1.0,
    shift_lag=1,
):
    """
    Run correlation estimation simulations for a single setting.

    Parameters
    ----------
    method_df : pd.DataFrame
        Methods to compare with columns: use_true_rho, method, kernel, M, b_qs, constant.
    rho : float
        True correlation coefficient.
    m_sim : int, optional (default=5)
        Number of simulations.
    nx : int, optional (default=50)
        First dimension size.
    ny : int or None, optional (default=None)
        Second dimension size. If None, equals nx.
    n_dim : int, optional (default=2)
        Number of dimensions.
    var : float, optional (default=1.0)
        Variance of the field.
    len_scale : float, optional (default=10.0)
        Length scale of the field.
    seed : int or None, optional (default=None)
        Random seed.
    M_trunc : float or None, optional (default=None)
        Truncation threshold for field values.
    return_summary : bool, optional (default=True)
        If True, returns summarized results by method.
    gen_method : str, optional (default="gstools")
        Field generation method.
    fwhm : float, optional (default=2)
        Full width at half maximum for Gaussian filter.
    truncate : float, optional (default=4)
        Truncation for Gaussian filter.
    distribution : str, optional (default="gaussian")
        Distribution type.
    noise_df : int or None, optional (default=None)
        Degrees of freedom for the 't' or 'chisq' noise distribution.
    trans : str or None, optional (default=None)
        Optional pointwise transformation applied to the fields (e.g. 'square').
    mask : str or None, optional (default=None)
        Shape of a fixed mask of valid grid points, the same in every simulation: 'ellipse'
        (see `gen_mask`). The fields are generated on the full grid and only the points in the
        mask are used. If None, every grid point is used.
    mask_axes : float, tuple of float, or None, optional (default=None)
        Semi-axes of the ellipse as fractions of half the side lengths (see `gen_mask`).
    fwhm_ratio : float, optional (default=1)
        Ratio of the smoothing of the two independent fields generating X and Y (see
        `gen_corr_field`): the field entering only Y is smoothed with ``fwhm / fwhm_ratio``.
        With 1, X and Y have the same smoothness; with 2, Y is partly rougher.
    dep_model : str or None, optional (default=None)
        Dependence beyond correlation between X and Y: None, 'quad', 'exp', 'shift', or
        'symmetric_shift' (see `gen_corr_field`). With rho = 0, X and Y are then uncorrelated but
        dependent.
    dep_strength : float, optional (default=1.0)
        Weight in [0, 1] of the dependent part (see `gen_corr_field`).
    shift_lag : int or tuple of int, optional (default=1)
        Lag of the shift for dep_model='shift' or 'symmetric_shift' (see `gen_corr_field`).

    Returns
    -------
    results : pd.DataFrame
        Simulation results. When `return_summary=False`, columns include: sim_id,
        method_id, est, se, se_under_null, runtime, M0, lower, upper, lower_raw,
        upper_raw, p, p_raw, bias, cover, cover_raw, reject, reject_raw, plus the
        setting columns (rho, nx, n_dim, len_scale, m_sim, gen_method, fwhm, fwhm_ratio,
        distribution, noise_df, trans, mask, mask_axes, n_obs) and the columns from
        method_df. n_obs is the number of grid points used.
        When `return_summary=True`, returns the per-method summary from
        `results_summary` joined with these setting/method columns.
    """
    # reset the index
    method_df = method_df.reset_index(drop=True)

    # if seed = None, rng.integers will genereate a random number
    rng = np.random.default_rng(seed)
    # turn pandas into array for faster access
    use_true_rho = method_df["use_true_rho"].values
    se_method = method_df["method"].values
    kernel = method_df["kernel"].values
    M = method_df["M"].values
    b_qs = method_df["b_qs"].values
    constant = method_df["constant"].values

    if n_dim == 1:
        dim = (nx,)
    elif n_dim == 2:
        if ny is None:
            ny = nx
        dim = (nx, ny)
    else:
        # use the same side length
        dim = (nx,) * n_dim

    # fixed mask, the same in every simulation; a missing value in setting_df (NaN) means none
    if isinstance(mask, float) and np.isnan(mask):
        mask = None
    if isinstance(mask_axes, float) and np.isnan(mask_axes):
        mask_axes = None
    mask_array = None if mask is None else gen_mask(dim, shape=mask, axes=mask_axes)
    if fwhm_ratio is None or (isinstance(fwhm_ratio, float) and np.isnan(fwhm_ratio)):
        fwhm_ratio = 1
    if isinstance(dep_model, float) and np.isnan(dep_model):
        dep_model = None
    if dep_strength is None or (isinstance(dep_strength, float) and np.isnan(dep_strength)):
        dep_strength = 1.0
    if shift_lag is None or (isinstance(shift_lag, float) and np.isnan(shift_lag)):
        shift_lag = 1
    if isinstance(shift_lag, float):
        shift_lag = int(shift_lag)
    n_obs =np.prod(dim) if mask_array is None else mask_array.sum()

    # if M_trunc is None:
    #     rho_final = rho
    # else:
    #     rho_final = get_rho_trunc(rho, nx, ny, var, len_scale, seed, M_trunc):

    results = []

    for i in range(m_sim):
        X, Y = gen_corr_field(
            rho,
            dim,
            var,
            len_scale,
            seed=rng.integers(0, 1e9),
            M_trunc=M_trunc,
            method=gen_method,
            fwhm=fwhm,
            truncate=truncate,
            distribution=distribution,
            noise_df = noise_df,
            trans = trans,
            fwhm_ratio=fwhm_ratio,
            dep_model=dep_model,
            dep_strength=dep_strength,
            shift_lag=shift_lag,
        )

        # print('shape of X:', X.shape, 'shape of Y:', Y.shape)
        for j in method_df.index:
            true_rho = None
            if use_true_rho[j]:
                true_rho = rho

            # Pearson correlation and se
            est, se, se_under_null, runtime_secs, M_tuple = corr(
                X,
                Y,
                true_rho=true_rho,
                se_method=se_method[j],
                kernel=kernel[j],
                M=M[j],
                b_qs=b_qs[j],
                constant=constant[j],
                mask=mask_array,
            )
            # python list append is similar to preallocate and then fill, faster than concat a row to a pandas dataframe
            results.append(
                {
                    "sim_id": i,
                    "method_id": j,
                    "est": est,
                    "se": se,
                    "se_under_null": se_under_null, 
                    "runtime": runtime_secs,
                    # M1 = M_tuple[1] is the same as M0
                    "M0": M_tuple[0],
                }
            )

    results = pd.DataFrame(results)
    lower, upper = confidence_interval(
        results["est"], results["se"], fisher_trans=True
    )
    lower_raw, upper_raw = confidence_interval(
        results["est"], results["se"], fisher_trans=False
    )
    p = pvalue(results["est"], results["se_under_null"], fisher_trans=True)
    p_raw  = pvalue(results["est"], results["se_under_null"], fisher_trans=False)
    results = results.assign(
        lower=lower,
        upper=upper,
        lower_raw=lower_raw,
        upper_raw=upper_raw,
        p = p, 
        p_raw = p_raw
    )
    results = results.assign(
        bias=results["est"] - rho,
        cover=1 * (rho <= results["upper"]) & (rho >= results["lower"]),
        cover_raw = 1 * (rho <= results["upper_raw"]) & (rho >= results["lower_raw"]),
        reject = 1*(p<0.05),
        reject_raw = 1*(p_raw<0.05)
    )

    if return_summary:
        results = results_summary(results, group=["method_id"])

    results = results.assign(
        rho=rho,
        nx=nx,
        n_dim=n_dim,
        len_scale=len_scale,
        m_sim=m_sim,
        gen_method=gen_method,
        fwhm=fwhm,
        fwhm_ratio=fwhm_ratio,
        distribution = distribution,
        noise_df = noise_df,
        trans = trans,
        mask=mask,
        # a list, so a tuple of axes is stored as one value per row
        mask_axes=[mask_axes] * len(results),
        n_obs=n_obs,
        dep_model=dep_model,
        dep_strength=dep_strength,
        # a list, so a tuple lag is stored as one value per row
        shift_lag=[shift_lag] * len(results),
    )
    # join using the index of method_df
    results = results.join(method_df, on="method_id")

    return results


def sim_multiple(setting_df, method_df, m_sim, return_summary=False, verbose=True):
    """
    Run correlation estimation simulations for multiple settings.

    Parameters
    ----------
    setting_df : pd.DataFrame
        Settings to iterate over with columns: rho, nx, n_dim, len_scale,
        gen_method, fwhm, distribution, noise_df, trans, and optionally mask,
        mask_axes, fwhm_ratio, dep_model, dep_strength, and shift_lag (see `sim_single`; without
        these columns no mask is used, fwhm_ratio is 1, and X, Y depend only through rho).
    method_df : pd.DataFrame
        Methods to compare with columns: use_true_rho, method, kernel, M, b_qs, constant.
    m_sim : int
        Number of simulations per setting.
    return_summary : bool, optional (default=False)
        If True, returns summarized results.
    verbose : bool, optional (default=True)
        If True, prints a progress message before each setting.

    Returns
    -------
    results : pd.DataFrame
        Combined results from all settings and methods, with a leading
        `setting_id` column.
    """
    results = []
    for i in setting_df.index:
        if verbose:
            print(
                f"----performing simulation, current setting number: {i + 1}, remaining settings: {setting_df.shape[0] - i - 1}----"
            )
        results_single = sim_single(
            method_df,
            rho=setting_df["rho"][i],
            m_sim=m_sim,
            nx=setting_df["nx"][i],
            n_dim=setting_df["n_dim"][i],
            len_scale=setting_df["len_scale"][i],
            return_summary=return_summary,
            gen_method=setting_df["gen_method"][i],
            fwhm=setting_df["fwhm"][i],
            distribution = setting_df["distribution"][i],
            noise_df = setting_df["noise_df"][i],
            trans = setting_df["trans"][i],
            mask=setting_df["mask"][i] if "mask" in setting_df else None,
            mask_axes=setting_df["mask_axes"][i] if "mask_axes" in setting_df else None,
            fwhm_ratio=setting_df["fwhm_ratio"][i] if "fwhm_ratio" in setting_df else 1,
            dep_model=setting_df["dep_model"][i] if "dep_model" in setting_df else None,
            dep_strength=setting_df["dep_strength"][i] if "dep_strength" in setting_df else 1.0,
            shift_lag=setting_df["shift_lag"][i] if "shift_lag" in setting_df else 1,
        )
        results_single.insert(0, "setting_id", i)
        # results_single = results_single.assign(setting_id=i + 1)
        results.append(results_single)
    results = pd.concat(results, ignore_index=True)
    return results


# dependence designs that are only meaningful with rho = 0 (see gen_corr_field)
RHO_ZERO_DEP_MODELS = ("shift", "symmetric_shift")


def make_setting_df(
    nx,
    rho,
    fwhm,
    n_dim=(2,),
    len_scale=(None,),
    gen_method=("gaussian_filter",),
    distribution=("gaussian",),
    noise_df=(None,),
    trans=(None,),
    mask=((None, None),),
    fwhm_ratio=(1,),
    dep=((None, None),),
):
    """
    Build the table of simulation settings for `sim_multiple`, one row per combination.

    Every argument is a list of values to cross. The settings table has one row for each
    combination, except that the dependence designs in RHO_ZERO_DEP_MODELS are kept only with
    rho = 0, since they make X and Y uncorrelated but dependent.

    Parameters
    ----------
    nx, rho, fwhm : list
        Side lengths, true correlations, and FWHMs of the smoothing kernel.
    n_dim, len_scale, gen_method, distribution, noise_df, trans, fwhm_ratio : list, optional
        Further settings passed to `sim_single` (see there); the defaults are single values.
    mask : list of tuple, optional (default=[(None, None)])
        Pairs (mask, mask_axes), e.g. ('ellipse', (1, 0.8)); (None, None) means no mask.
    dep : list of tuple, optional (default=[(None, None)])
        Pairs (dep_model, shift_lag), e.g. ('shift', 1) or ('symmetric_shift', 1);
        (None, None) means X and Y depend only through rho.

    Returns
    -------
    setting_df : pd.DataFrame
        Columns nx, n_dim, rho, len_scale, fwhm, gen_method, distribution, noise_df, trans,
        fwhm_ratio, mask, mask_axes, dep_model, shift_lag, with rows numbered from 0.

    Examples
    --------
    >>> setting_df = make_setting_df(nx=[25, 50], rho=[0, 0.4], fwhm=[2, 3],
    ...                              trans=[None, 'square'], dep=[(None, None), ('shift', 1)])
    """
    from itertools import product

    columns = ["nx", "n_dim", "rho", "len_scale", "fwhm", "gen_method", "distribution",
               "noise_df", "trans", "mask_spec", "fwhm_ratio", "dep_spec"]
    setting_df = pd.DataFrame(
        product(nx, n_dim, rho, len_scale, fwhm, gen_method, distribution, noise_df, trans,
                mask, fwhm_ratio, dep),
        columns=columns,
    )
    setting_df[["mask", "mask_axes"]] = pd.DataFrame(
        setting_df.pop("mask_spec").tolist(), index=setting_df.index)
    setting_df[["dep_model", "shift_lag"]] = pd.DataFrame(
        setting_df.pop("dep_spec").tolist(), index=setting_df.index)
    # the rho = 0 designs: keep all rho for rows without such a design, rho = 0 for the rest
    keep = ~setting_df["dep_model"].isin(RHO_ZERO_DEP_MODELS) | (setting_df["rho"] == 0)
    return setting_df[keep].reset_index(drop=True)


def make_method_df(
    method=("general", "gauss", "indep", "iid"),
    M=("auto",),
    kernel=("bartlett",),
    use_true_rho=(False,),
    b_qs=(None,),
    constant=(1,),
):
    """
    Build the table of standard error methods for `sim_multiple`, one row per combination.

    Parameters
    ----------
    method : list, optional (default=['general', 'gauss', 'indep', 'iid'])
        Standard error methods (see SE_METHODS).
    M : list, optional (default=['auto'])
        Lag truncation rules, e.g. 'auto' or 'plugin' (see `get_se`).
    kernel, use_true_rho, b_qs, constant : list, optional
        Further method settings (see `corr`); the defaults are single values.

    Returns
    -------
    method_df : pd.DataFrame
        Columns use_true_rho, method, kernel, M, b_qs, constant, with rows numbered from 0.
    """
    from itertools import product

    for m in method:
        canonical_se_method(m)  # fail early on an unknown method name
    return pd.DataFrame(
        product(use_true_rho, method, kernel, M, b_qs, constant),
        columns=["use_true_rho", "method", "kernel", "M", "b_qs", "constant"],
    )


# def mean(
#     X,
#     M="cubic_root",
#     method="bartlett",
#     check=False,
#     b_qs=5,
#     use_true_mean=False,
#     true_mean=None,
# ):
#     """
#     Estimate mean and standard error for autocorrelated 1D data.
#
#     Parameters
#     ----------
#     X : ndarray
#         1D input data.
#     M : str or int, optional (default="cubic_root")
#         Lag truncation method: 'cubic_root', 'linear', or numeric.
#     method : str, optional (default="bartlett")
#         HAC kernel type: 'iid', 'bartlett', 'constant', 'qs', or 'naive'.
#     check : bool, optional (default=False)
#         If True, print diagnostic information.
#     b_qs : float, optional (default=5)
#         Bandwidth parameter for QS kernel.
#     use_true_mean : bool, optional (default=False)
#         If True, use true_mean for calculations.
#     true_mean : float or None, optional (default=None)
#         Known true mean value.
#
#     Returns
#     -------
#     mean_est : float
#         Estimated mean.
#     se : float
#         Standard error of the mean estimate.
#     runtime_secs : float
#         Computation time in seconds.
#     M_int : int
#         Lag truncation value used.
#     """
#     start_time = time.time()
#     n = len(X)
#
#     mean_est = X.mean()
#     sd = X.std()
#     if use_true_mean:
#         mean_plugin = true_mean
#     else:
#         mean_plugin = mean_est
#     if method == "iid":
#         var = sd**2 / n
#     else:
#         # --------weight function and truncation lag numbder -----------
#         if M == "cubic_root":
#             M_int = np.ceil(n**0.33)
#         elif M == "linear":
#             M_int = np.floor(n / 5)
#             # M_tuple = np.floor((nx**0.33, ny**0.33))*1.4
#         elif isinstance(M, int):
#             M_int = M
#         else:
#             raise ValueError("M must be 'cubic_root', 'linear', or numeric tuple.")
#
#         lags = np.arange(-(n - 1), n)
#         if method == "bartlett":
#             weights = (np.abs(lags) < M_int) * (1 - np.abs(lags) / M_int)
#         if method == "constant":
#             weights = (np.abs(lags_x_all) < M_tuple[0]) * (
#                 np.abs(lags_y_all) < M_tuple[1]
#             )
#         if method == "qs":
#             weights = (
#                 (np.abs(lags_x_all) < M_tuple[0])
#                 * (np.abs(lags_y_all) < M_tuple[1])
#                 * qs_kernel(np.abs(lags_x_all) / b_qs)
#                 * qs_kernel(np.abs(lags_y_all) / b_qs)
#             )
#         if method == "naive":
#             # known to perform poorly, testing only!!
#             weights = np.ones((len(lags_x), len(lags_y)))
#         # print(weights)
#         # print(weights.shape)
#         # print(X_standard.mean())
#         # print(X_standard.std())
#
#         cov = est_exp_X0Yh(X, X) - mean_plugin**2
#
#         var = np.sum(weights * cov) / n
#
#     # print(var)
#     if var < 0:
#         warnings.warn("Estimated variance is negative. Check!")
#         se = np.nan
#     else:
#         se = np.sqrt(var)
#     runtime_secs = time.time() - start_time
#     return mean_est, se, runtime_secs, M_int
#
#
# def sim_single_mean(
#     m_sim=5,
#     n=64,
#     var=1.0,
#     len_scale=10.0,
#     seed=None,
#     return_summary=True,
#     method="bartlett",
#     iid=False,
#     M=None,
#     b_qs=None,
#     square=False,
# ):
#     """
#     Run mean estimation simulations for a single setting.
#
#     Parameters
#     ----------
#     m_sim : int, optional (default=5)
#         Number of simulations.
#     n : int, optional (default=64)
#         Length of the time series.
#     var : float, optional (default=1.0)
#         Variance of the field.
#     len_scale : float, optional (default=10.0)
#         Length scale of spatial correlation.
#     seed : int or None, optional (default=None)
#         Random seed.
#     return_summary : bool, optional (default=True)
#         If True, returns summarized results.
#     method : str, optional (default="bartlett")
#         HAC kernel type.
#     iid : bool, optional (default=False)
#         If True, generates iid data instead of correlated.
#     M : int or None, optional (default=None)
#         Lag truncation value.
#     b_qs : float or None, optional (default=None)
#         Bandwidth for QS kernel.
#     square : bool, optional (default=False)
#         If True, estimates mean of squared field.
#
#     Returns
#     -------
#     results : pd.DataFrame
#         Simulation results with columns: sim_id, est, se, runtime, lower, upper, bias,
#         cover, n, len_scale, m_sim, method, M, b_qs, M_method.
#     """
#     # if seed = None, rng.integers will genereate a random number
#     rng = np.random.default_rng(seed)
#
#     results = []
#     x = np.arange(n) - n // 2
#     if square:
#         truth = var
#     else:
#         truth = 0
#     if not iid:
#         model = gs.Gaussian(dim=1, var=var, len_scale=len_scale)
#         srf = gs.SRF(model)
#     for i in range(1, m_sim + 1):
#         if iid:
#             X = np.random.normal(loc=0, scale=np.sqrt(var), size=n)
#         else:
#             X = srf.structured(x, seed=rng.integers(0, 1e9))
#         if square:
#             X = X**2
#         est, se, runtime_secs, M_int = mean(X, method=method, M=M, b_qs=b_qs)
#         results.append({"sim_id": i, "est": est, "se": se, "runtime": runtime_secs})
#
#     results = pd.DataFrame(results)
#     results = results.assign(
#         lower=results["est"] - 1.96 * results["se"],
#         upper=results["est"] + 1.96 * results["se"],
#     )
#     results = results.assign(
#         bias=est, cover=1 * (truth <= results["upper"]) & (truth >= results["lower"])
#     )
#     if return_summary:
#         results = results_summary(results)
#     results = results.assign(
#         n=n,
#         len_scale=len_scale,
#         m_sim=m_sim,
#         method=method,
#         M=M_int,
#         b_qs=b_qs,
#         M_method=str(M),
#     )
#     return results
#
#
# def sim_multiple_mean(
#     setting_df, m_sim, return_summary=False, method="bartlett", iid=False, square=False
# ):
#     """
#     Run mean estimation simulations for multiple settings.
#
#     Parameters
#     ----------
#     setting_df : pd.DataFrame
#         Settings to iterate over with columns: nx, len_scale, M, b_qs.
#     m_sim : int
#         Number of simulations per setting.
#     return_summary : bool, optional (default=False)
#         If True, returns summarized results.
#     method : str, optional (default="bartlett")
#         HAC kernel type.
#     iid : bool, optional (default=False)
#         If True, generates iid data.
#     square : bool, optional (default=False)
#         If True, estimates mean of squared field.
#
#     Returns
#     -------
#     results : pd.DataFrame
#         Combined results from all settings.
#     """
#     results = []
#     for i in setting_df.index:
#         print(
#             f"----performing simulation, current setting number: {i + 1}, remaining settings: {setting_df.shape[0] - i - 1}----"
#         )
#         results_single = sim_single_mean(
#             m_sim=m_sim,
#             n=setting_df["nx"][i],
#             len_scale=setting_df["len_scale"][i],
#             return_summary=return_summary,
#             method=method,
#             iid=iid,
#             M=setting_df["M"][i],
#             b_qs=setting_df["b_qs"][i],
#             square=square,
#         )
#         results_single = results_single.assign(setting_id=i + 1)
#         results.append(results_single)
#     results = pd.concat(results, ignore_index=True)
#     return results


# sim_multiple(pd.DataFrame({'nx': [10],'n_dim': [1], 'rho':[0.4], 'len_scale':[None],
#                             'fwhm':[1], 'gen_method': ['gaussian_filter']}),
#     pd.DataFrame({'use_true_rho': [False], 'method': ['iid'], 'kernel': ['bartlett'], 'M': ['auto'], 'b_qs': [None],
#                   'constant': [1]}),
#                               m_sim=2, return_summary = False)
