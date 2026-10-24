import numbers
from typing import List, Optional, Tuple, Union

import numpy as np

import rfcorr as rc


def gen_corr_field(rho: float,
                   dim: Union[int, Tuple[int, ...]] = (50, 50),
                   var: float = 1.0,
                   len_scale: float = 10.0,
                   seed: Optional[int] = None,
                   M_trunc: Optional[float] = None,
                   return_coordinate: bool = False,
                   method: str = "gaussian_filter",
                   fwhm: float = 4,
                   truncate: float = 4,
                   distribution: str = "gaussian",
                   noise_df: Optional[int] = None,
                   fwhm_ratio: float = 1,
                   trans: Optional[str] = None,
                   dep_model: Optional[str] = None,
                   dep_strength: float = 1.0,
                   shift_lag: Union[int, Tuple[int, ...]] = 1
                   ) -> Union[Tuple[np.ndarray, np.ndarray],
                              Tuple[np.ndarray, np.ndarray, List[np.ndarray]]]:
    """
    Generate a pair of correlated random fields with optional truncation.

    X = U and Y = rho U + sqrt(1 - rho^2) V, where U and V are fields from
    rc.gen_field. With `dep_model`, V is replaced by a field that is uncorrelated
    with U at every location but depends on U.

    ---------------------------------------------------------------------------
    ARGUMENTS:
    rho : float
        Desired pointwise correlation between the two fields (between -1 and 1).
    dim : int or tuple of int, optional (default=(50, 50))
        Dimensions of the fields. If int, creates 1D fields. If tuple, creates
        multi-dimensional fields.
    var : float, optional (default=1.0)
        Target variance of each field.
    len_scale : float, optional (default=10.0)
        Length scale parameter of the Gaussian covariance (larger length scale is
        smoother).
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
        Truncation parameter for the Gaussian filter.
    distribution : str, optional (default="gaussian")
        Distribution type: 'gaussian', 't', or 'chisq'.
    noise_df : int or None, optional (default=None)
        Degrees of freedom for the 't' or 'chisq' noise distribution.
    fwhm_ratio : float, optional (default=1)
        Ratio of the two fields' smoothing: the second field V is generated with
        fwhm / fwhm_ratio, allowing X and Y to have different smoothness.
    trans : str or None, optional (default=None)
        Optional pointwise transformation applied to each field (e.g. 'square').
    dep_model : str or None, optional (default=None)
        If None, Y = rho U + sqrt(1 - rho^2) V with U, V independent. Otherwise V is
        replaced by a field W that is uncorrelated with U at every pair of locations
        but depends on U: W is the residual of a nonlinear field on X = U with the
        population slope, rescaled to unit variance. With lam = dep_strength and
        U, V standardized,
        'quad': W = lam (U^2 - 1) / sqrt(2) + sqrt(1 - lam^2) V, the residual of
        U + W on U (slope 1).
        'exp': W = lam (exp(U) - e^(1/2) - e^(1/2) U) / sqrt(e (e - 2))
        + sqrt(1 - lam^2) V, the residual of exp(U) on U (slope e^(1/2)) mixed with V.
        'shift': W = lam D / sd(D) * sd(U) + sqrt(1 - lam^2) V, where
        D_s = U_{s+h} - r_h U_s, h is `shift_lag`, r_h = corr(U_s, U_{s+h}) is the
        population lag-h correlation of U (from rc.gaussian_filter_acf, squared for
        trans='square'), and sd(D) is the population standard deviation of D. Then
        cov(U_s, D_s) = 0 exactly, while D is correlated with U at other lags. Unlike
        'quad' and 'exp', the dependence is linear in U, so with trans=None and
        Gaussian noise X and Y are jointly Gaussian. Works with trans=None (any noise
        distribution) or trans='square' (Gaussian noise, the chi^2_1 setting), and
        requires method='gaussian_filter'.
        'symmetric_shift': as 'shift' with the symmetric
        D_s = (U_{s+h} + U_{s-h}) / 2 - r_h U_s, whose variance is
        var(U) ((1 + r_2h) / 2 - r_h^2). Again cov(U_s, D_s) = 0, but the
        cross-correlation of X and Y is even in the lag, so the cross term
        sum_k rho_xy,k rho_xy,-k of the variance of the sample correlation is
        positive and tests assuming independence (Clifford-type) are
        anti-conservative. With 'shift' the cross term is negative and those tests
        are conservative.
        'sum_diff': U and V are replaced by their normalized sum and difference,
        (U + V) / sqrt(2) and (U - V) / sqrt(2), so that X = (U + V) / sqrt(2) and, with
        rho = 0, Y = (U - V) / sqrt(2). U and V have the same variance, so X and Y keep it
        and are uncorrelated at every location. Their lag-k cross-correlation is
        (r_U(k) - r_V(k)) / 2, so with fwhm_ratio != 1 they are dependent, the cross
        term above is positive, and tests assuming independence are anti-conservative.
        With fwhm_ratio = 1 and Gaussian fields X and Y are independent, while
        non-Gaussian fields (e.g. trans='square') are dependent even then. Works with any
        distribution and trans.
        So rho = 0 gives X and Y uncorrelated but dependent. 'quad' and 'exp'
        require Gaussian U and trans=None.
    dep_strength : float, optional (default=1.0)
        Weight lam in [0, 1] of the dependent part of W; lam = 0 gives W = V
        (independence). Used with 'quad', 'exp', 'shift' and 'symmetric_shift', not
        with 'sum_diff'.
    shift_lag : int or tuple of int, optional (default=1)
        Lag h for dep_model='shift' or 'symmetric_shift'. An int k means a shift of
        k grid points along the first axis. Only used for these two models.

    ---------------------------------------------------------------------------
    OUTPUT:
    X : np.ndarray
        First field, of shape `dim`. Clipped at +/-M_trunc if `M_trunc` is not None.
    Y : np.ndarray
        Second field, of shape `dim`. Clipped at +/-M_trunc if `M_trunc` is not None.
    coordinate_ls : list of np.ndarray
        Per-dimension coordinate arrays. Only returned if return_coordinate=True.
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

    U = rc.gen_field(
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
        noise_df=noise_df,
        trans=trans
    )
    V = rc.gen_field(
        dim=dim,
        var=var,
        len_scale=len_scale,
        seed=rng.integers(0, 1e9),
        M_trunc=None,
        return_coordinate=False,
        method=method,
        fwhm=fwhm / fwhm_ratio,
        truncate=truncate,
        distribution=distribution,
        noise_df=noise_df,
        trans=trans
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
        r_h = rc.gaussian_filter_acf(h, fwhm, truncate)
        r_2h = rc.gaussian_filter_acf(tuple(2 * x for x in h), fwhm, truncate)
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
    elif dep_model == "sum_diff":
        # normalized sum and difference of the independent fields U and V, which have the
        # same variance: both keep that variance and are uncorrelated at every location, but
        # are dependent when U and V differ in smoothness (fwhm_ratio != 1) or are non-Gaussian
        U, V = (U + V) / np.sqrt(2), (U - V) / np.sqrt(2)
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
            raise ValueError("dep_model must be None, 'quad', 'exp', 'shift', 'symmetric_shift', "
                             "or 'sum_diff'")
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
