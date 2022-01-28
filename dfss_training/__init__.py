from typing import Optional, Tuple

import numpy as np
import scipy.stats as sp_stats


def pp(dist: sp_stats.rv_continuous, spec_limits: Tuple[float, float]) -> float:
    """Calculate the process capability potential of a distribution.

    Parameters
    ----------
    dist
        Scipy statitics continuous distribution.
    spec_limits
        Tuple representing the lower and upper spec limits.

    """
    six_sigma_bound = 0.9973
    six_sigma_range = dist.ppf(six_sigma_bound) - dist.ppf(1.0 - six_sigma_bound)
    return (spec_limits[1] - spec_limits[0]) / six_sigma_range


def ppk(
    dist: sp_stats.rv_continuous, spec_limits: Tuple[Optional[float], Optional[float]]
) -> Tuple[Optional[float], Optional[float], float]:
    """Calculate the process capability of a given distribution.

    Parameters
    ----------
    dist
        Scipy-style statistics distribution to calculate statistics on.
    spec_limits
        Tuple representing the lower and upper spec limits.

    Returns
    -------
    lower_ppk, upper_ppk, ppk
        The upper bounded ppk, the lower bounded ppk, and the ppk
        for this distribution.
    """
    if all(lim is None for lim in spec_limits):
        raise ValueError("Cannot have both spec_limits set to None")
    median = dist.median()
    three_sigma_approx = 0.5 * (1 - 0.9973)
    if spec_limits[0] is not None:
        lower_quantile = dist.ppf(three_sigma_approx)
        lower_range = median - lower_quantile
        lower_ppk = (median - spec_limits[0]) / lower_range
    else:
        lower_ppk = None
    if spec_limits[1] is not None:
        upper_quantile = dist.ppf(1 - three_sigma_approx)
        upper_range = upper_quantile - median
        upper_ppk = (spec_limits[1] - median) / upper_range
    else:
        upper_ppk = None

    if lower_ppk is None:
        ppk_val = upper_ppk
    elif upper_ppk is None:
        ppk_val = lower_ppk
    else:
        ppk_val = min(lower_ppk, upper_ppk)
    return lower_ppk, upper_ppk, ppk_val


def defect_probability(
    dist: sp_stats.rv_continuous, spec_limits: Tuple[float, float]
) -> Tuple[float, float, float]:
    """Calculate the mass of defects below and above the spec_limits.

    Parameters
    ----------
    dist
        Scipy statitics continuous distribution.
    spec_limits
        Tuple representing the lower and upper spec limits.

    Returns
    -------
    lower_prob, upper_prob, total_prob
        The lower, upper and total defect probability.
    """
    upper_end = 1.0 - dist.cdf(spec_limits[1])
    lower_end = dist.cdf(spec_limits[0])
    return lower_end, upper_end, lower_end + upper_end


def _moving_range(array: np.ndarray) -> np.ndarray:
    """Calculate a moving range"""
    view = np.lib.stride_tricks.sliding_window_view(array, 2)
    return np.max(view, axis=-1) - np.min(view, axis=-1)


def calculate_d2(n: int) -> float:
    """Calculate the d2 constant.

    Parameters
    ----------
    n
        Number of elements in the windowed range.

    Returns
    -------
    d2
        Value of the d2 constant.

    """
    standard_normal = sp_stats.norm()
    approx_x_pos = np.geomspace(1e-5, 1e2, 501)
    approx_x = np.concatenate((approx_x_pos[::-1], approx_x_pos))
    cdf_samples = standard_normal.cdf(approx_x)
    sample_vals = 1.0 - (1.0 - cdf_samples) ** n - cdf_samples ** n
    return np.trapz(sample_vals, x=approx_x)


def calculate_sigma_within(data: np.ndarray):
    """Calculate the within sample sigma approximation.

    Parameters
    ----------
    data
        Array of shape of at most 2. If 2 dimensional, each
        row is assumed to be the samples for a logical grouping.
        If of length 1, a 2 element rolling average is used for the
        sub groups.

    Returns
    -------
    sigma_within
        This is Rbar / d2 where Rbar is the average of the within
        subgroup ranges.
    """
    if data.dims == 1:
        return np.mean(_moving_range(data)) / calculate_d2(2)
    elif data.dims == 2:
        range_array = np.max(data, axis=-1) - np.min(data, axis=-1)
        rbar = np.mean(range_array)
        return rbar / calculate_d2(data.shape[1])
    else:
        raise ValueError


def cpk(array, spec_limits):
    sigma_within_val = calculate_sigma_within(array)
    median = np.median(array)
    return min(
        (median - spec_limits[0]) / (3 * sigma_within_val),
        (spec_limits[1] - median) / (3 * sigma_within_val),
    )
