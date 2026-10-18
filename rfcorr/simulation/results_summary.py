from typing import List, Optional

import pandas as pd


def results_summary(results: pd.DataFrame,
                    group: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Summarize simulation results by aggregating across simulations.

    ---------------------------------------------------------------------------
    ARGUMENTS:
    results : pd.DataFrame
        Results DataFrame with columns: bias, est, se, cover, cover_raw, reject,
        reject_raw, runtime (and, when grouping, the per-method/setting identifier
        columns, as returned by rc.sim_single or rc.sim_multiple with
        return_summary=False).
    group : list of str or None, optional (default=None)
        Columns to group by. If None, returns a single-row summary across all rows.

    ---------------------------------------------------------------------------
    OUTPUT:
    summary : pd.DataFrame
        Aggregated results with columns: bias_mean, est_sd, se_mean, cover_rate,
        cover_rate_raw, power, power_raw, runtime_mean. When `group` is provided,
        also includes n_se_na (count of NaN SEs) and the setting and method columns.
    ---------------------------------------------------------------------------
    AUTHORS: Jiyue Qin
    ---------------------------------------------------------------------------
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
