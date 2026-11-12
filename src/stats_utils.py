"""Small bootstrap helpers (numpy/pandas only).

Units are the resampling unit shared across configs (a (prompt, seed) pair for per-image
metrics, a prompt for per-group metrics). Using the SAME resampled units for every config
makes differences between configs paired, which is what matched initial noise allows.
Missing values (NaN, e.g. safety-checker-flagged images) are ignored via nanmean.
"""
import warnings
import numpy as np
import pandas as pd


def _boot_idx(n, B, seed):
    return np.random.default_rng(seed).integers(0, n, size=(B, n))


def boot_means(pivot: pd.DataFrame, B=2000, seed=0) -> pd.DataFrame:
    """pivot: rows = units, columns = configs. Mean per config with 95% percentile CI."""
    X = pivot.to_numpy(dtype=float)
    idx = _boot_idx(X.shape[0], B, seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        bm = np.nanmean(X[idx], axis=1)                  # B x configs
        lo, hi = np.nanpercentile(bm, [2.5, 97.5], axis=0)
        mean = np.nanmean(X, axis=0)
    return pd.DataFrame(dict(config=list(pivot.columns), mean=mean, ci_lo=lo, ci_hi=hi,
                             n_units=(~np.isnan(X)).sum(axis=0)))


def boot_diff(pivot: pd.DataFrame, a: str, b: str, B=2000, seed=0) -> dict:
    """Paired difference a - b (mean over units) with 95% percentile CI."""
    d = (pivot[a] - pivot[b]).to_numpy(dtype=float)
    idx = _boot_idx(len(d), B, seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        bm = np.nanmean(d[idx], axis=1)
        lo, hi = np.nanpercentile(bm, [2.5, 97.5])
        mean = np.nanmean(d)
    return dict(comparison=f"{a} - {b}", mean_diff=mean, ci_lo=lo, ci_hi=hi,
                n_units=int((~np.isnan(d)).sum()))
