from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd

import rfcorr as rc


def sim_single(method_df: pd.DataFrame,
               rho: float,
               m_sim: int = 5,
               nx: int = 50,
               ny: Optional[int] = None,
               n_dim: int = 2,
               var: float = 1.0,
               len_scale: float = 10.0,
               seed: Optional[int] = None,
               M_trunc: Optional[float] = None,
               return_summary: bool = True,
               gen_method: str = "gstools",
               fwhm: float = 2,
               truncate: float = 4,
               distribution: str = "gaussian",
               noise_df: Optional[int] = None,
               trans: Optional[str] = None,
               mask: Optional[str] = None,
               mask_axes: Optional[Union[float, Tuple[float, ...]]] = None,
               fwhm_ratio: float = 1,
               dep_model: Optional[str] = None,
               dep_strength: float = 1.0,
               shift_lag: Union[int, Tuple[int, ...]] = 1) -> pd.DataFrame:
    """
    Run correlation estimation simulations for a single setting.

    In each of `m_sim` replications a pair of fields is generated with
    rc.gen_corr_field, and every method in `method_df` is applied to it with
    rc.corr. Confidence intervals, p-values, coverage, and rejections are added.

    ---------------------------------------------------------------------------
    ARGUMENTS:
    method_df : pd.DataFrame
        Methods to compare with columns: use_true_rho, method, kernel, M, b_qs,
        constant (see rc.make_method_df).
    rho : float
        True correlation coefficient.
    m_sim : int, optional (default=5)
        Number of simulations.
    nx : int, optional (default=50)
        First dimension size.
    ny : int or None, optional (default=None)
        Second dimension size. If None, equals nx.
    n_dim : int, optional (default=2)
        Number of dimensions. For n_dim > 2 every side has length nx.
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
        Full width at half maximum for the Gaussian filter.
    truncate : float, optional (default=4)
        Truncation for the Gaussian filter.
    distribution : str, optional (default="gaussian")
        Distribution type.
    noise_df : int or None, optional (default=None)
        Degrees of freedom for the 't' or 'chisq' noise distribution.
    trans : str or None, optional (default=None)
        Optional pointwise transformation applied to the fields (e.g. 'square').
    mask : str or None, optional (default=None)
        Shape of a fixed mask of valid grid points, the same in every simulation:
        'ellipse' (see rc.gen_mask). The fields are generated on the full grid and
        only the points in the mask are used. If None, every grid point is used.
    mask_axes : float, tuple of float, or None, optional (default=None)
        Semi-axes of the ellipse as fractions of half the side lengths (see
        rc.gen_mask).
    fwhm_ratio : float, optional (default=1)
        Ratio of the smoothing of the two independent fields generating X and Y (see
        rc.gen_corr_field): the field entering only Y is smoothed with
        fwhm / fwhm_ratio. With 1, X and Y have the same smoothness; with 2, Y is
        partly rougher.
    dep_model : str or None, optional (default=None)
        Dependence beyond correlation between X and Y: None, 'quad', 'exp', 'shift',
        or 'symmetric_shift' (see rc.gen_corr_field). With rho = 0, X and Y are then
        uncorrelated but dependent.
    dep_strength : float, optional (default=1.0)
        Weight in [0, 1] of the dependent part (see rc.gen_corr_field).
    shift_lag : int or tuple of int, optional (default=1)
        Lag of the shift for dep_model='shift' or 'symmetric_shift' (see
        rc.gen_corr_field).

    ---------------------------------------------------------------------------
    OUTPUT:
    results : pd.DataFrame
        Simulation results. When return_summary=False, columns include: sim_id,
        method_id, est, se, se_under_null, runtime, M0, lower, upper, lower_raw,
        upper_raw, p, p_raw, bias, cover, cover_raw, reject, reject_raw, plus the
        setting columns (rho, nx, n_dim, len_scale, m_sim, gen_method, fwhm,
        fwhm_ratio, distribution, noise_df, trans, mask, mask_axes, n_obs,
        dep_model, dep_strength, shift_lag) and the columns from method_df. n_obs is
        the number of grid points used. When return_summary=True, returns the
        per-method summary from rc.results_summary joined with these setting and
        method columns.
    ---------------------------------------------------------------------------
    AUTHORS: Jiyue Qin
    ---------------------------------------------------------------------------
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
    mask_array = None if mask is None else rc.gen_mask(dim, shape=mask, axes=mask_axes)
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
    n_obs = np.prod(dim) if mask_array is None else mask_array.sum()

    results = []

    for i in range(m_sim):
        X, Y = rc.gen_corr_field(
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
            noise_df=noise_df,
            trans=trans,
            fwhm_ratio=fwhm_ratio,
            dep_model=dep_model,
            dep_strength=dep_strength,
            shift_lag=shift_lag,
        )

        for j in method_df.index:
            true_rho = None
            if use_true_rho[j]:
                true_rho = rho

            # Pearson correlation and se
            est, se, se_under_null, runtime_secs, M_tuple = rc.corr_dev(
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
    lower, upper = rc.confidence_interval(
        results["est"], results["se"], fisher_trans=True
    )
    lower_raw, upper_raw = rc.confidence_interval(
        results["est"], results["se"], fisher_trans=False
    )
    p = rc.pvalue(results["est"], results["se_under_null"], fisher_trans=True)
    p_raw = rc.pvalue(results["est"], results["se_under_null"], fisher_trans=False)
    results = results.assign(
        lower=lower,
        upper=upper,
        lower_raw=lower_raw,
        upper_raw=upper_raw,
        p=p,
        p_raw=p_raw
    )
    results = results.assign(
        bias=results["est"] - rho,
        cover=1 * (rho <= results["upper"]) & (rho >= results["lower"]),
        cover_raw=1 * (rho <= results["upper_raw"]) & (rho >= results["lower_raw"]),
        reject=1 * (p < 0.05),
        reject_raw=1 * (p_raw < 0.05)
    )

    if return_summary:
        results = rc.results_summary(results, group=["method_id"])

    results = results.assign(
        rho=rho,
        nx=nx,
        n_dim=n_dim,
        len_scale=len_scale,
        m_sim=m_sim,
        gen_method=gen_method,
        fwhm=fwhm,
        fwhm_ratio=fwhm_ratio,
        distribution=distribution,
        noise_df=noise_df,
        trans=trans,
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
