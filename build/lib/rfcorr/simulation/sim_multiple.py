import pandas as pd

import rfcorr as rc


def sim_multiple(setting_df: pd.DataFrame,
                 method_df: pd.DataFrame,
                 m_sim: int,
                 return_summary: bool = False,
                 verbose: bool = True) -> pd.DataFrame:
    """
    Run correlation estimation simulations for multiple settings.

    Calls rc.sim_single for each row of `setting_df` and stacks the results.

    ---------------------------------------------------------------------------
    ARGUMENTS:
    setting_df : pd.DataFrame
        Settings to iterate over with columns: rho, nx, n_dim, len_scale,
        gen_method, fwhm, distribution, noise_df, trans, and optionally mask,
        mask_axes, fwhm_ratio, dep_model, dep_strength, and shift_lag (see
        rc.sim_single; without these columns no mask is used, fwhm_ratio is 1, and
        X, Y depend only through rho). Usually built with rc.make_setting_df.
    method_df : pd.DataFrame
        Methods to compare with columns: use_true_rho, method, kernel, M, b_qs,
        constant (see rc.make_method_df).
    m_sim : int
        Number of simulations per setting.
    return_summary : bool, optional (default=False)
        If True, returns summarized results.
    verbose : bool, optional (default=True)
        If True, prints a progress message before each setting.

    ---------------------------------------------------------------------------
    OUTPUT:
    results : pd.DataFrame
        Combined results from all settings and methods, with a leading setting_id
        column holding the row label in `setting_df`.
    ---------------------------------------------------------------------------
    AUTHORS: Jiyue Qin
    ---------------------------------------------------------------------------
    """
    results = []
    for i in setting_df.index:
        if verbose:
            print(
                f"----performing simulation, current setting number: {i + 1}, remaining settings: {setting_df.shape[0] - i - 1}----"
            )
        results_single = rc.sim_single(
            method_df,
            rho=setting_df["rho"][i],
            m_sim=m_sim,
            nx=setting_df["nx"][i],
            n_dim=setting_df["n_dim"][i],
            len_scale=setting_df["len_scale"][i],
            return_summary=return_summary,
            gen_method=setting_df["gen_method"][i],
            fwhm=setting_df["fwhm"][i],
            distribution=setting_df["distribution"][i],
            noise_df=setting_df["noise_df"][i],
            trans=setting_df["trans"][i],
            mask=setting_df["mask"][i] if "mask" in setting_df else None,
            mask_axes=setting_df["mask_axes"][i] if "mask_axes" in setting_df else None,
            fwhm_ratio=setting_df["fwhm_ratio"][i] if "fwhm_ratio" in setting_df else 1,
            dep_model=setting_df["dep_model"][i] if "dep_model" in setting_df else None,
            dep_strength=setting_df["dep_strength"][i] if "dep_strength" in setting_df else 1.0,
            shift_lag=setting_df["shift_lag"][i] if "shift_lag" in setting_df else 1,
        )
        results_single.insert(0, "setting_id", i)
        results.append(results_single)
    results = pd.concat(results, ignore_index=True)
    return results
