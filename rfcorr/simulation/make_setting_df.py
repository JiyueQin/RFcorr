from itertools import product
from typing import Optional, Sequence, Tuple

import pandas as pd

import rfcorr as rc


def make_setting_df(nx: Sequence[int],
                    rho: Sequence[float],
                    fwhm: Sequence[float],
                    n_dim: Sequence[int] = (2,),
                    len_scale: Sequence[Optional[float]] = (None,),
                    gen_method: Sequence[str] = ("gaussian_filter",),
                    distribution: Sequence[str] = ("gaussian",),
                    noise_df: Sequence[Optional[int]] = (None,),
                    trans: Sequence[Optional[str]] = (None,),
                    mask: Sequence[Tuple] = ((None, None),),
                    fwhm_ratio: Sequence[float] = (1,),
                    dep: Sequence[Tuple] = ((None, None),)) -> pd.DataFrame:
    """
    Build the table of simulation settings for rc.sim_multiple, one row per combination.

    Every argument is a list of values to cross. The settings table has one row for
    each combination, except that the dependence designs in
    rc.get_defaults()['rho_zero_dep_models'] are kept only with rho = 0, since they
    make X and Y uncorrelated but dependent.

    ---------------------------------------------------------------------------
    ARGUMENTS:
    nx : list of int
        Side lengths.
    rho : list of float
        True correlations.
    fwhm : list of float
        FWHMs of the smoothing kernel.
    n_dim : list of int, optional (default=(2,))
        Numbers of dimensions.
    len_scale : list, optional (default=(None,))
        Length scales (only used by gen_method='gstools').
    gen_method : list of str, optional (default=("gaussian_filter",))
        Field generation methods (see rc.gen_field).
    distribution : list of str, optional (default=("gaussian",))
        Noise distributions (see rc.gen_field).
    noise_df : list, optional (default=(None,))
        Degrees of freedom of the noise distribution.
    trans : list, optional (default=(None,))
        Pointwise transformations, e.g. None or 'square'.
    mask : list of tuple, optional (default=((None, None),))
        Pairs (mask, mask_axes), e.g. ('ellipse', (1, 0.8)); (None, None) means no
        mask.
    fwhm_ratio : list of float, optional (default=(1,))
        Ratios of the smoothing of the two fields (see rc.gen_corr_field).
    dep : list of tuple, optional (default=((None, None),))
        Pairs (dep_model, shift_lag), e.g. ('shift', 1), ('symmetric_shift', 1) or
        ('sum_diff', None);
        (None, None) means X and Y depend only through rho.

    ---------------------------------------------------------------------------
    OUTPUT:
    setting_df : pd.DataFrame
        Columns nx, n_dim, rho, len_scale, fwhm, gen_method, distribution, noise_df,
        trans, fwhm_ratio, mask, mask_axes, dep_model, shift_lag, with rows numbered
        from 0.
    ---------------------------------------------------------------------------
    AUTHORS: Jiyue Qin
    ---------------------------------------------------------------------------
    """
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
    rho_zero_dep_models = rc.get_defaults()["rho_zero_dep_models"]
    keep = ~setting_df["dep_model"].isin(rho_zero_dep_models) | (setting_df["rho"] == 0)
    return setting_df[keep].reset_index(drop=True)
