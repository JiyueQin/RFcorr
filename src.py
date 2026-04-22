import numpy as np

# import gstools as gs
import pandas as pd
import warnings
from scipy.ndimage import gaussian_filter
import sys


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
):
    """
    Generate a 0-mean random field with optional truncation.

    Parameters
    ----------
    dim : int or tuple of int, optional (default=(50, 50))
        Dimensions of the field. If int, creates 1D field. If tuple, creates multi-dimensional field.
    var : float, optional (default=1.0)
        Variance parameter for the Gaussian covariance model.
    len_scale : float, optional (default=1)
        Length scale parameter of the Gaussian covariance (larger length scale is smoother).
    seed : int or None, optional (default=None)
        Random seed for reproducibility. If None, uses a random seed.
    M_trunc : float or None, optional (default=None)
        Truncation threshold. If provided, both fields are clipped to [-M_trunc, M_trunc].
    Returns
    -------
    X : ndarray
        A 2D NumPy arrays of shape (nx, ny), representing correlated Gaussian fields.
        If `M_trunc` is not None, the arrays are truncated at ±M_trunc.

    Examples
    --------
    >>> X = gen_field(rho=0.7, nx=50, len_scale=5.0, seed=42, M_trunc=2.0)
    >>> X.shape
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
        X = np.random.normal(loc=0, scale=np.sqrt(var), size=dim)
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
            noise_raw = np.random.randn(*dim_larger)
            sd_before_smoothing = 1
        elif distribution == "t":
            if noise_df is None:
                noise_df = 3
                noise_raw = np.random.standard_t(noise_df, dim_larger)
            if noise_df > 2:
                sd_before_smoothing = np.sqrt(noise_df / (noise_df - 2))
            else:
                sys.exit("Noise degree of freedom needs to be >2 for a t distribution!")
        elif distribution == "chisq":
            if noise_df is None:
                noise_df = 5
                # center by subtracting the mean
                noise_raw = np.random.chisquare(noise_df, dim_larger) - noise_df
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
    if return_coordinate:
        return (X, coordinate_ls)
    else:
        return X


def gen_corr_field(
    rho,
    dim=(50, 50),
    var=1.0,
    len_scale=10.0,
    seed=None,
    M_trunc=None,
    return_coordinate=False,
    method="gstools",
    fwhm=2,
    truncate=4,
    distribution="gaussian",
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
        Variance parameter for the Gaussian covariance model.
    len_scale : float, optional (default=10.0)
        Length scale parameter of the Gaussian covariance (larger length scale is smoother).
    seed : int or None, optional (default=None)
        Random seed for reproducibility. If None, uses a random seed.
    M_trunc : float or None, optional (default=None)
        Truncation threshold. If provided, both fields are clipped to [-M_trunc, M_trunc].
    return_coordinate : bool, optional (default=False)
        If True, returns fields and coordinate lists.
    method : str, optional (default="gstools")
        Method to generate fields: 'iid', 'gstools', or 'gaussian_filter'.
    fwhm : float, optional (default=2)
        Full width at half maximum for Gaussian filter method.
    truncate : float, optional (default=4)
        Truncation parameter for Gaussian filter.
    distribution : str, optional (default="gaussian")
        Distribution type: 'gaussian', 't', or 'chisq'.

    Returns
    -------
    X, Y : ndarray
        Two 2D NumPy arrays of shape (nx, ny), representing correlated Gaussian fields.
        If `M_trunc` is not None, the arrays are truncated at ±M_trunc.

    Examples
    --------
    >>> X, Y = gen_field(rho=0.7, nx=50, len_scale=5.0, seed=42, M_trunc=2.0)
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

    U = gen_field(
        dim=dim,
        var=var,
        len_scale=len_scale,
        seed=rng.integers(0, 1e9),
        M_trunc=None,
        return_coordinate=False,
        method=method,
        fwhm=fwhm,
        truncate=truncate,
        distribution=distribution,
    )
    V = gen_field(
        dim=dim,
        var=var,
        len_scale=len_scale,
        seed=rng.integers(0, 1e9),
        M_trunc=None,
        return_coordinate=False,
        method=method,
        fwhm=fwhm,
        truncate=truncate,
        distribution=distribution,
    )
    beta = np.sqrt(max(0.0, 1.0 - rho**2))
    # correlated pair of fields
    X = U
    Y = rho * U + beta * V

    # optional: correlated pair of truncated fields
    if M_trunc is not None:
        X = np.clip(X, -M_trunc, M_trunc)
        Y = np.clip(X, -M_trunc, M_trunc)
    if return_coordinate:
        return (X, Y, coordinate_ls)
    else:
        return (X, Y)


import numpy as np
from scipy.signal import correlate2d, correlate


def est_exp_X0Yh(X, Y, return_counts_lags=False):
    """
    Estimate E[X_0 * Y_h] for all spatial integer lags h on a grid.

    Parameters
    ----------
    X, Y : ndarray (same shape)
        Input fields of any dimensionality.
    return_counts_lags : bool, optional (default=False)
        If False, returns only E. If True, returns (S, C, E, lag_ls).

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
    mask = np.ones(dim)

    # Sum of products at each lag
    S = correlate(X, Y, mode="full")
    # number of valid pairs at each lag (also known as edge correction for unbiased estimate)
    C = correlate(mask, mask, mode="full")
    # max number of pairs
    # C = nx*ny
    # estimate of the expectaion
    E = S / C
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
        return mask
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


def corr_old(
    X, Y, M="cubic_root", true_rho=None, method="bartlett", check=False, b_qs=5
):
    start_time = time.time()
    if X.shape != Y.shape:
        raise ValueError("X and Y must have the same shape")
    nx, ny = X.shape
    dim = np.array(X.shape)
    rho_est = np.corrcoef(X.ravel(), Y.ravel())[0, 1]
    X_standard = X
    Y_standard = Y
    # X_standard =  (X-X.mean())/X.std()
    # Y_standard =  (Y-Y.mean())/Y.std()
    if true_rho is None:
        rho_plugin = rho_est
    else:
        rho_plugin = true_rho
    # print('rho_plugin:', rho_plugin)
    if method == "iid":
        var = (1 - rho_plugin**2) ** 2 / n
    else:
        # --------weight function and truncation lag numbder -----------
        if M == "cubic_root":
            M_tuple = np.ceil(dim**0.3)
        elif M == "linear":
            M_tuple = np.floor(dim / 5)
            # M_tuple = np.floor((nx**0.33, ny**0.33))*1.4
        elif (
            isinstance(M, tuple) and all(isinstance(x, numbers.Number) for x in M)
        ) or (isinstance(M, int)):
            M_tuple = M
        else:
            raise ValueError("M must be 'cubic_root', 'linear', or numeric.")

        lags_x = np.arange(-(nx - 1), nx)
        lags_y = np.arange(-(ny - 1), ny)
        lags_x_all, lags_y_all = np.meshgrid(lags_x, lags_y)
        if method == "bartlett":
            weights = (
                (np.abs(lags_x_all) <= M_tuple[0])
                * (np.abs(lags_y_all) <= M_tuple[1])
                * (1 - np.abs(lags_x_all) / (M_tuple[0] + 1))
                * (1 - np.abs(lags_y_all) / (1 + M_tuple[1]))
            )
        if method == "constant":
            weights = (np.abs(lags_x_all) <= M_tuple[0]) * (
                np.abs(lags_y_all) <= M_tuple[1]
            )
        if method == "qs":
            weights = (
                (np.abs(lags_x_all) <= M_tuple[0])
                * (np.abs(lags_y_all) <= M_tuple[1])
                * kernel(np.abs(lags_x_all) / b_qs, type="qs")
                * kernel(np.abs(lags_y_all) / b_qs, type="qs")
            )
        if method == "naive":
            # known to perform poorly, testing only!!
            weights = np.ones((len(lags_x), len(lags_y)))
        print("weights:", weights)
        # print(weights.shape)
        # print(X_standard.mean())
        # print(X_standard.std())

        X_standard_square = X_standard**2
        Y_standard_square = Y_standard**2
        X_standard_Y_standard = X_standard * Y_standard
        cov_ls = [
            est_exp_X0Yh(X_standard_Y_standard, X_standard_Y_standard) - rho_plugin**2,
            est_exp_X0Yh(X_standard_Y_standard, X_standard_square) - rho_plugin,
            est_exp_X0Yh(X_standard_Y_standard, Y_standard_square) - rho_plugin,
            est_exp_X0Yh(X_standard_square, X_standard_Y_standard) - rho_plugin,
            est_exp_X0Yh(X_standard_square, X_standard_square) - 1,
            est_exp_X0Yh(X_standard_square, Y_standard_square) - 1,
            est_exp_X0Yh(Y_standard_square, X_standard_Y_standard) - rho_plugin,
            est_exp_X0Yh(Y_standard_square, X_standard_square) - 1,
            est_exp_X0Yh(Y_standard_square, Y_standard_square) - 1,
        ]

        omega = np.array([np.sum(weights.T * cov) for cov in cov_ls]).reshape((3, 3))
        print("omega", omega)
        vec = np.array([1, -0.5 * rho_plugin, -0.5 * rho_plugin])
        var = (vec @ omega @ vec) / (nx * ny)
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
        # if check:
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
    runtime_secs = time.time() - start_time
    return rho_est, se, runtime_secs, M_tuple


from functools import reduce


def corr(
    X,
    Y,
    M="cubic_root",
    true_rho=None,
    se_method="hac",
    kernel="bartlett",
    check=False,
    b_qs=5,
    constant=1,
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
    se_method : str, optional (default="hac")
        Standard error method: 'iid', 'hac', 'hac_gaussian', or 'huristic'.
    kernel : str, optional (default="bartlett")
        Kernel type for HAC: 'bartlett', 'qs', or 'constant'.
    check : bool, optional (default=False)
        If True, print diagnostic information.
    b_qs : float, optional (default=5)
        Bandwidth parameter for QS kernel.
    constant : float, optional (default=1)
        Scaling constant for M calculation.

    Returns
    -------
    rho_est : float
        Estimated Pearson correlation.
    se : float
        Standard error of the correlation estimate.
    runtime_secs : float
        Computation time in seconds.
    M_tuple : tuple
        Lag truncation values used.
    """
    start_time = time.time()
    if X.shape != Y.shape:
        raise ValueError("X and Y must have the same shape")
    dim = np.array(X.shape)
    n = np.prod(dim)
    n_dim = len(dim)
    rho_est = np.corrcoef(X.ravel(), Y.ravel())[0, 1]
    X_standard = X
    Y_standard = Y
    # todo: add this
    # X_standard =  (X-X.mean())/X.std()
    # Y_standard =  (Y-Y.mean())/Y.std()

    if true_rho is None:
        rho_plugin = rho_est
    else:
        rho_plugin = true_rho
    # print('rho_plugin:', rho_plugin)

    if se_method == "huristic":
        rho_plugin = 0
        kernel = "bartlett"

    if se_method == "iid":
        var = (1 - rho_plugin**2) ** 2 / n
        M_tuple = (None,)
    else:
        # --------weight function and truncation lag numbder -----------
        if M == "cubic_root":
            M_tuple = np.ceil(dim**0.3 * constant)
        elif M == "linear":
            M_tuple = np.floor(dim / 5 * constant)
            # M_tuple = np.floor((nx**0.33, ny**0.33))*1.4
        elif M == "auto":
            M_tuple = np.ceil(dim ** (n_dim / (n_dim + 2)) * constant)
        elif isinstance(M, int):
            M_tuple = (M,)
        elif isinstance(M, tuple) and all(isinstance(x, numbers.Number) for x in M):
            M_tuple = M
        else:
            raise ValueError("M must be 'cubic_root', 'linear', or numeric.")

        lag_ls = [np.arange(-(dim_i - 1), dim_i) for dim_i in dim]

        weights_ls = [
            kernel_weights(lag_ls[i], type=kernel, bandwidth=b_qs, M=M_tuple[i])
            for i in range(n_dim)
        ]
        weights = reduce(np.multiply, np.ix_(*weights_ls))

        # print(weights.shape)
        # print(X_standard.mean())
        # print(X_standard.std())

        if se_method in ["hac_gaussian", "huristic"]:
            rho_xx_h = est_exp_X0Yh(X_standard, X_standard)
            rho_yy_h = est_exp_X0Yh(Y_standard, Y_standard)
            if se_method == "huristic":
                rho_xy_h = 0
                rho_yx_h = 0
            else:
                rho_xy_h = est_exp_X0Yh(X_standard, Y_standard)
                rho_yx_h = est_exp_X0Yh(Y_standard, X_standard)
            # order the matrix by row
            cov_ls = [
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
            cov_ls = [
                est_exp_X0Yh(X_standard_Y_standard, X_standard_Y_standard)
                - rho_plugin**2,
                est_exp_X0Yh(X_standard_Y_standard, X_standard_square) - rho_plugin,
                est_exp_X0Yh(X_standard_Y_standard, Y_standard_square) - rho_plugin,
                est_exp_X0Yh(X_standard_square, X_standard_Y_standard) - rho_plugin,
                est_exp_X0Yh(X_standard_square, X_standard_square) - 1,
                est_exp_X0Yh(X_standard_square, Y_standard_square) - 1,
                est_exp_X0Yh(Y_standard_square, X_standard_Y_standard) - rho_plugin,
                est_exp_X0Yh(Y_standard_square, X_standard_square) - 1,
                est_exp_X0Yh(Y_standard_square, Y_standard_square) - 1,
            ]

        omega = np.array([np.sum(weights * cov) for cov in cov_ls]).reshape((3, 3))
        vec = np.array([1, -0.5 * rho_plugin, -0.5 * rho_plugin])
        var = (vec @ omega @ vec) / n
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
    runtime_secs = time.time() - start_time
    return rho_est, se, runtime_secs, M_tuple


def results_summary(results, group=None):
    """
    Summarize simulation results by aggregating across simulations.

    Parameters
    ----------
    results : pd.DataFrame
        Results DataFrame with columns: bias, est, se, cover, runtime.
    group : list of str or None, optional (default=None)
        Columns to group by. If None, returns summary across all rows.

    Returns
    -------
    summary : pd.DataFrame
        Aggregated results with columns: bias_mean, est_sd, se_mean, n_se_na, cover_rate, runtime_mean.
    """
    if group is None:
        out = pd.DataFrame(
            {
                "bias_mean": results["bias"].mean(),
                "est_sd": results["est"].std(),
                "se_mean": results["se"].mean(),
                "cover_rate": results["cover"].mean(),
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
                "bias",
                "est",
                "se",
                "cover",
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
                "runtime": "mean",
            }
        )
        out.columns = [
            "bias_mean",
            "est_sd",
            "se_mean",
            "n_se_na",
            "cover_rate",
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

    Returns
    -------
    results : pd.DataFrame
        Simulation results with columns: sim_id, method_id, est, se, runtime, M0,
        lower, upper, bias, cover, rho, nx, n_dim, len_scale, m_sim, gen_method, fwhm,
        and columns from method_df.
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
        )

        # print('shape of X:', X.shape, 'shape of Y:', Y.shape)
        for j in method_df.index:
            true_rho = None
            if use_true_rho[j]:
                true_rho = rho

            # Pearson correlation and se
            est, se, runtime_secs, M_tuple = corr(
                X,
                Y,
                true_rho=true_rho,
                se_method=se_method[j],
                kernel=kernel[j],
                M=M[j],
                b_qs=b_qs[j],
                constant=constant[j],
            )
            # python list append is similar to preallocate and then fill, faster than concat a row to a pandas dataframe
            results.append(
                {
                    "sim_id": i,
                    "method_id": j,
                    "est": est,
                    "se": se,
                    "runtime": runtime_secs,
                    # M1 = M_tuple[1] is the same as M0
                    "M0": M_tuple[0],
                }
            )

    results = pd.DataFrame(results)
    results = results.assign(
        lower=results["est"] - 1.96 * results["se"],
        upper=results["est"] + 1.96 * results["se"],
    )
    results = results.assign(
        bias=results["est"] - rho,
        cover=1 * (rho <= results["upper"]) & (rho >= results["lower"]),
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
    )
    # join using the index of method_df
    results = results.join(method_df, on="method_id")

    return results


def sim_multiple(setting_df, method_df, m_sim, return_summary=False):
    """
    Run correlation estimation simulations for multiple settings.

    Parameters
    ----------
    setting_df : pd.DataFrame
        Settings to iterate over with columns: rho, nx, n_dim, len_scale, gen_method, fwhm.
    method_df : pd.DataFrame
        Methods to compare with columns: use_true_rho, method, kernel, M, b_qs, constant.
    m_sim : int
        Number of simulations per setting.
    return_summary : bool, optional (default=False)
        If True, returns summarized results.

    Returns
    -------
    results : pd.DataFrame
        Combined results from all settings and methods.
    """
    results = []
    for i in setting_df.index:
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
        )
        results_single.insert(0, "setting_id", i)
        # results_single = results_single.assign(setting_id=i + 1)
        results.append(results_single)
    results = pd.concat(results, ignore_index=True)
    return results


def mean(
    X,
    M="cubic_root",
    method="bartlett",
    check=False,
    b_qs=5,
    use_true_mean=False,
    true_mean=None,
):
    """
    Estimate mean and standard error for autocorrelated 1D data.

    Parameters
    ----------
    X : ndarray
        1D input data.
    M : str or int, optional (default="cubic_root")
        Lag truncation method: 'cubic_root', 'linear', or numeric.
    method : str, optional (default="bartlett")
        HAC kernel type: 'iid', 'bartlett', 'constant', 'qs', or 'naive'.
    check : bool, optional (default=False)
        If True, print diagnostic information.
    b_qs : float, optional (default=5)
        Bandwidth parameter for QS kernel.
    use_true_mean : bool, optional (default=False)
        If True, use true_mean for calculations.
    true_mean : float or None, optional (default=None)
        Known true mean value.

    Returns
    -------
    mean_est : float
        Estimated mean.
    se : float
        Standard error of the mean estimate.
    runtime_secs : float
        Computation time in seconds.
    M_int : int
        Lag truncation value used.
    """
    start_time = time.time()
    n = len(X)

    mean_est = X.mean()
    sd = X.std()
    if use_true_mean:
        mean_plugin = true_mean
    else:
        mean_plugin = mean_est
    if method == "iid":
        var = sd**2 / n
    else:
        # --------weight function and truncation lag numbder -----------
        if M == "cubic_root":
            M_int = np.ceil(n**0.33)
        elif M == "linear":
            M_int = np.floor(n / 5)
            # M_tuple = np.floor((nx**0.33, ny**0.33))*1.4
        elif isinstance(M, int):
            M_int = M
        else:
            raise ValueError("M must be 'cubic_root', 'linear', or numeric tuple.")

        lags = np.arange(-(n - 1), n)
        if method == "bartlett":
            weights = (np.abs(lags) < M_int) * (1 - np.abs(lags) / M_int)
        if method == "constant":
            weights = (np.abs(lags_x_all) < M_tuple[0]) * (
                np.abs(lags_y_all) < M_tuple[1]
            )
        if method == "qs":
            weights = (
                (np.abs(lags_x_all) < M_tuple[0])
                * (np.abs(lags_y_all) < M_tuple[1])
                * qs_kernel(np.abs(lags_x_all) / b_qs)
                * qs_kernel(np.abs(lags_y_all) / b_qs)
            )
        if method == "naive":
            # known to perform poorly, testing only!!
            weights = np.ones((len(lags_x), len(lags_y)))
        # print(weights)
        # print(weights.shape)
        # print(X_standard.mean())
        # print(X_standard.std())

        cov = est_exp_X0Yh(X, X) - mean_plugin**2

        var = np.sum(weights * cov) / n

    # print(var)
    if var < 0:
        warnings.warn("Estimated variance is negative. Check!")
        se = np.nan
    else:
        se = np.sqrt(var)
    runtime_secs = time.time() - start_time
    return mean_est, se, runtime_secs, M_int


def sim_single_mean(
    m_sim=5,
    n=64,
    var=1.0,
    len_scale=10.0,
    seed=None,
    return_summary=True,
    method="bartlett",
    iid=False,
    M=None,
    b_qs=None,
    square=False,
):
    """
    Run mean estimation simulations for a single setting.

    Parameters
    ----------
    m_sim : int, optional (default=5)
        Number of simulations.
    n : int, optional (default=64)
        Length of the time series.
    var : float, optional (default=1.0)
        Variance of the field.
    len_scale : float, optional (default=10.0)
        Length scale of spatial correlation.
    seed : int or None, optional (default=None)
        Random seed.
    return_summary : bool, optional (default=True)
        If True, returns summarized results.
    method : str, optional (default="bartlett")
        HAC kernel type.
    iid : bool, optional (default=False)
        If True, generates iid data instead of correlated.
    M : int or None, optional (default=None)
        Lag truncation value.
    b_qs : float or None, optional (default=None)
        Bandwidth for QS kernel.
    square : bool, optional (default=False)
        If True, estimates mean of squared field.

    Returns
    -------
    results : pd.DataFrame
        Simulation results with columns: sim_id, est, se, runtime, lower, upper, bias,
        cover, n, len_scale, m_sim, method, M, b_qs, M_method.
    """
    # if seed = None, rng.integers will genereate a random number
    rng = np.random.default_rng(seed)

    results = []
    x = np.arange(n) - n // 2
    if square:
        truth = var
    else:
        truth = 0
    if not iid:
        model = gs.Gaussian(dim=1, var=var, len_scale=len_scale)
        srf = gs.SRF(model)
    for i in range(1, m_sim + 1):
        if iid:
            X = np.random.normal(loc=0, scale=np.sqrt(var), size=n)
        else:
            X = srf.structured(x, seed=rng.integers(0, 1e9))
        if square:
            X = X**2
        est, se, runtime_secs, M_int = mean(X, method=method, M=M, b_qs=b_qs)
        results.append({"sim_id": i, "est": est, "se": se, "runtime": runtime_secs})

    results = pd.DataFrame(results)
    results = results.assign(
        lower=results["est"] - 1.96 * results["se"],
        upper=results["est"] + 1.96 * results["se"],
    )
    results = results.assign(
        bias=est, cover=1 * (truth <= results["upper"]) & (truth >= results["lower"])
    )
    if return_summary:
        results = results_summary(results)
    results = results.assign(
        n=n,
        len_scale=len_scale,
        m_sim=m_sim,
        method=method,
        M=M_int,
        b_qs=b_qs,
        M_method=str(M),
    )
    return results


def sim_multiple_mean(
    setting_df, m_sim, return_summary=False, method="bartlett", iid=False, square=False
):
    """
    Run mean estimation simulations for multiple settings.

    Parameters
    ----------
    setting_df : pd.DataFrame
        Settings to iterate over with columns: nx, len_scale, M, b_qs.
    m_sim : int
        Number of simulations per setting.
    return_summary : bool, optional (default=False)
        If True, returns summarized results.
    method : str, optional (default="bartlett")
        HAC kernel type.
    iid : bool, optional (default=False)
        If True, generates iid data.
    square : bool, optional (default=False)
        If True, estimates mean of squared field.

    Returns
    -------
    results : pd.DataFrame
        Combined results from all settings.
    """
    results = []
    for i in setting_df.index:
        print(
            f"----performing simulation, current setting number: {i + 1}, remaining settings: {setting_df.shape[0] - i - 1}----"
        )
        results_single = sim_single_mean(
            m_sim=m_sim,
            n=setting_df["nx"][i],
            len_scale=setting_df["len_scale"][i],
            return_summary=return_summary,
            method=method,
            iid=iid,
            M=setting_df["M"][i],
            b_qs=setting_df["b_qs"][i],
            square=square,
        )
        results_single = results_single.assign(setting_id=i + 1)
        results.append(results_single)
    results = pd.concat(results, ignore_index=True)
    return results


# sim_multiple(pd.DataFrame({'nx': [10],'n_dim': [1], 'rho':[0.4], 'len_scale':[None],
#                             'fwhm':[1], 'gen_method': ['gaussian_filter']}),
#     pd.DataFrame({'use_true_rho': [False], 'method': ['iid'], 'kernel': ['bartlett'], 'M': ['auto'], 'b_qs': [None],
#                   'constant': [1]}),
#                               m_sim=2, return_summary = False)
