import numbers
import warnings
from functools import reduce
from typing import List, Optional, Tuple, Union

import numpy as np

import rfcorr as rc


def get_se(dim: np.ndarray,
           est_exp_ls: List[np.ndarray],
           rho_plugin: float,
           M: Union[str, int, Tuple[float, ...]] = "cubic_root",
           se_method: str = "general",
           kernel: str = "bartlett",
           check: bool = False,
           b_qs: float = 5,
           constant: float = 1,
           n_obs: Optional[int] = None) -> Tuple[float, tuple]:
    """
    Compute the HAC standard error of the correlation estimate.

    Development function: a building block of rc.corr_dev; its interface may change.
    The variance is vec' Omega vec / n, where Omega is the kernel-weighted sum over
    lags of the lag-wise covariances of T = (X Y, X^2, Y^2) and
    vec = (1, -rho/2, -rho/2).

    ---------------------------------------------------------------------------
    ARGUMENTS:
    dim : np.ndarray
        Shape of the field (per-dimension sizes).
    est_exp_ls : list of np.ndarray
        Lag-wise expectation estimates produced by rc.get_est_exp_ls.
    rho_plugin : float
        Plugin correlation used in the variance formula (use 0 for the null SE).
    M : str, int, or tuple of numeric, optional (default="cubic_root")
        Lag truncation method: 'optimal_rate', 'plugin', an int, or a tuple of
        numerics, or the legacy rules 'auto', 'cubic_root', and 'linear'. The string
        rules use the same lag in every dimension, computed from the number of valid
        grid points n rather than the side lengths. 'optimal_rate' (the paper's
        rule) is ceil(constant * n ** (1 / (n_dim + 2))), the MSE-optimal rate;
        'auto' is its earlier name. With side = n ** (1 / n_dim), 'cubic_root' is
        ceil(constant * side ** 0.3) and 'linear' is floor(constant * side / 5).
        'plugin' is data driven and grows
        with the smoothness of the fields (see rc.plugin_bandwidth); it is still
        under development and not verified.
    se_method : str, optional (default="general")
        Standard error method: 'general', 'gauss', 'indep', or 'iid' (see
        rc.get_defaults()['se_methods']). The earlier names 'hac', 'hac_gaussian',
        and 'huristic' are also accepted.
    kernel : str, optional (default="bartlett")
        Kernel type for HAC weighting: 'bartlett', 'qs', or 'constant' (products of
        1D weights over axes), or 'bartlett_radial', the radial Bartlett weights
        1 - ||h / (M + 1)|| (with lags scaled per axis), as in Kim and Sun (2011).
    check : bool, optional (default=False)
        If True, print diagnostic information (omega and weights).
    b_qs : float, optional (default=5)
        Bandwidth parameter for the QS kernel.
    constant : float, optional (default=1)
        Scaling constant for the M calculation.
    n_obs : int or None, optional (default=None)
        Number of valid grid points (the size of the mask). If None, every grid
        point is valid.

    ---------------------------------------------------------------------------
    OUTPUT:
    se : float
        Standard error of the correlation estimate (np.nan if the estimated
        variance is negative).
    M_tuple : tuple
        Lag truncation values used.
    ---------------------------------------------------------------------------
    AUTHORS: Jiyue Qin
    ---------------------------------------------------------------------------
    """
    n = np.prod(dim) if n_obs is None else n_obs
    n_dim = len(dim)
    se_method = rc.canonical_se_method(se_method)
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

        # --------weight function and truncation lag number -----------
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
        elif M in ("optimal_rate", "auto"):
            # 'auto' is the earlier name of 'optimal_rate'; ceil(side^(n_dim/(n_dim+2))) is
            # ceil(n^(1/(n_dim+2)))
            M_tuple = np.full(n_dim, np.ceil(side ** (n_dim / (n_dim + 2)) * constant))
        elif M == "plugin":
            M_tuple = rc.plugin_bandwidth(cov_ls, vec, n, dim, constant)
        elif isinstance(M, int):
            # same truncation lag in every dimension
            M_tuple = (M,) * n_dim
        elif isinstance(M, tuple) and all(isinstance(x, numbers.Number) for x in M):
            M_tuple = M
        else:
            raise ValueError("M must be 'optimal_rate', 'plugin', 'auto', 'cubic_root', "
                             "'linear', or numeric.")

        lag_ls = [np.arange(-(dim_i - 1), dim_i) for dim_i in dim]

        if kernel == "bartlett_radial":
            # radial Bartlett weights 1 - ||h / (M + 1)||, with lags scaled per axis as in the
            # product weights, so that the same M gives the same reach along each axis
            scaled = np.meshgrid(*[lag_ls[i] / (M_tuple[i] + 1) for i in range(n_dim)],
                                 indexing="ij")
            weights = np.maximum(0.0, 1 - np.sqrt(sum(s**2 for s in scaled)))
        else:
            weights_ls = [
                rc.kernel_weights(lag_ls[i], type=kernel, bandwidth=b_qs, M=M_tuple[i])
                for i in range(n_dim)
            ]
            weights = reduce(np.multiply, np.ix_(*weights_ls))

        # nansum drops lags with no valid pair under a mask (NaN from est_exp_X0Yh)
        omega = np.array([np.nansum(weights * cov) for cov in cov_ls]).reshape((3, 3))
        var = (vec @ omega @ vec) / n
        if se_method == "indep":
            var = factor * var
        if check:
            print("omega:", omega)
            print("weights:", weights)
    if var < 0:
        warnings.warn("Estimated variance is negative. Check!")
        se = np.nan
    else:
        se = np.sqrt(var)
    return se, M_tuple
