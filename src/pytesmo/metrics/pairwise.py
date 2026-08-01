"""
Pairwise metrics and analytical confidence intervals.

Metrics
-------

The metrics function implemented here all have the signature::

    def metric(x : np.ndarray, y : np.ndarray) -> float

Confidence intervals
--------------------

Formulas for confidence intervals have in general been taken from from
Gilleland (2010), 10.5065/D6WD3XJM,
https://opensky.ucar.edu/islandora/object/technotes:491

Other references are cited in the docstring of the respective function.

Analytical confidence interval functions implemented here are named
``<metric>_ci``, e.g. for ``bias``, the CI function is ``bias_ci``.
The signature is be::

    def metric_ci(x : np.ndarray, y : np.ndarray, m : float,
                  alpha=0.05 : float) -> float, float

where m is the metric value that has been calculated for x and y.

Typically, you should use
:py:func:`pytesmo.metrics.confidence_intervals.with_analytical_ci` for
calculating a metric CI.
"""

# IMPORTANT: DEVELOPERS NOTES
#
# Some of the RMSD methods take ddof as keyword argument. Although there are
# some warnings in the docstrings (and also when using it) that it is
# deprecated, we want to keep it for now, in order to reproduce previous
# results.


import numpy as np
from scipy import stats
import warnings

# Try to import GPU-accelerated versions first
try:
    from pytesmo.gpu import is_gpu_available
    from pytesmo.gpu.pairwise import (
        bias as _gpu_bias,
        mse_decomposition as _gpu_mse_decomposition,
        rmsd as _gpu_rmsd,
        ubrmsd as _gpu_ubrmsd,
        pearson_r as _gpu_pearson_r,
        spearman_r as _gpu_spearman_r,
        kendall_tau as _gpu_kendall_tau,
        rolling_pr_rmsd as _gpu_rolling_pr_rmsd,
    )
    _GPU_AVAILABLE = is_gpu_available()
except ImportError:
    _GPU_AVAILABLE = False

# Fallback to CPU implementations
from pytesmo.metrics._fast_pairwise import (  # noqa: F401
    bias as _cpu_bias,
    mse_bias as _cpu_mse_bias,
    mse_var as _cpu_mse_var,
    mse_corr as _cpu_mse_corr,
    mse_decomposition as _cpu_mse_decomposition,
    RSS as _cpu_RSS,
    _ubrmsd as _cpu_ubrmsd,
    rolling_pr_rmsd as _cpu_rolling_pr_rmsd,
)


def _select_impl(gpu_func, cpu_func):
    """Select GPU or CPU implementation based on availability."""
    if _GPU_AVAILABLE:
        return gpu_func
    return cpu_func


# CPU implementations for pearson, spearman, kendall (use scipy.stats)
def _cpu_pearsonr(x, y):
    return stats.pearsonr(x, y)


def _cpu_spearmanr(x, y):
    return stats.spearmanr(x, y)


def _cpu_kendalltau(x, y):
    return stats.kendalltau(x, y)


# Public API - automatically uses GPU if available
bias = _select_impl(_gpu_bias, _cpu_bias)
mse_decomposition = _select_impl(_gpu_mse_decomposition, _cpu_mse_decomposition)
rmsd = _select_impl(_gpu_rmsd, lambda x, y, ddof=0: np.sqrt(_cpu_mse_decomposition(x, y)[0]) if ddof == 0 else np.sqrt(_cpu_RSS(x, y) / (len(x) - ddof)))
ubrmsd = _select_impl(_gpu_ubrmsd, _cpu_ubrmsd)
pearson_r = _select_impl(_gpu_pearson_r, _cpu_pearsonr)
spearman_r = _select_impl(_gpu_spearman_r, _cpu_spearmanr)
kendall_tau = _select_impl(_gpu_kendall_tau, _cpu_kendalltau)
rolling_pr_rmsd = _select_impl(_gpu_rolling_pr_rmsd, _cpu_rolling_pr_rmsd)

# Re-export CPU-only functions
mse_bias = _cpu_mse_bias
mse_var = _cpu_mse_var
mse_corr = _cpu_mse_corr
RSS = _cpu_RSS
_ubrmsd = _cpu_ubrmsd


has_ci = [
    "bias",
    "ubrmsd",
    "pearson_r",
    "spearman_r",
    "kendall_tau",
]

no_ci = [
    "aad",
    "mad",
    "mse_bias",
    "msd",
    "rmsd",
    "nrmsd",
    "mse_corr",
    "mse_var",
    "nash_sutcliffe",
    "index_of_agreement",
]

no_ci = [
    "aad",
    "mad",
    "mse_bias",
    "msd",
    "rmsd",
    "nrmsd",
    "mse_corr",
    "mse_var",
    "nash_sutcliffe",
    "index_of_agreement",
]


def bias_ci(x, y, b, alpha=0.05):
    """
    Confidence interval for bias.

    The confidence interval is the same as the confidence interval for a mean.

    Parameters
    ----------
    x : numpy.ndarray
        First input vector.
    y : numpy.ndarray
        Second input vector.
    b : float
        bias
    alpha : float, optional
        1 - confidence level, default is 0.05

    Returns
    -------
    lower, upper : float
        Lower and upper confidence interval bounds.
    """
    n = len(x)
    delta = np.std(x - y, ddof=1) / np.sqrt(n) * stats.t.ppf(1 - alpha / 2, n - 1)
    return b - delta, b + delta


def _bias_ci_from_moments(alpha, mx, my, varx, vary, cov, n):
    # This is based on the fact that:
    # var(x - y) = var(x) + var(y) - 2*cov(x,y)
    # and therefore
    # std(x - y, ddof=1) = sqrt(var(x - y, ddof=1))
    #                    = sqrt(n/(n-1) * var(x-y))
    std = np.sqrt(n / (n - 1) * (varx + vary - 2 * cov))
    delta = std / np.sqrt(n) * stats.t.ppf(1 - alpha / 2, n - 1)
    b = mx - my
    return b - delta, b + delta


def aad(x, y):
    """
    Average (=mean) absolute deviation (AAD).

    Parameters
    ----------
    x : numpy.ndarray
        First input vector.
    y : numpy.ndarray
        Second input vector.

    Returns
    -------
    d : float
        Mean absolute deviation.
    """
    return np.mean(np.abs(x - y))


def mad(x, y):
    """
    Median absolute deviation (MAD).

    Parameters
    ----------
    x : numpy.ndarray
        First input vector.
    y : numpy.ndarray
        Second input vector.

    Returns
    -------
    d : float
        Median absolute deviation.
    """
    return np.median(np.abs(x - y))


def msd(x, y):
    r"""
    Mean square deviation/mean square error.

    For validation, MSD (same as MSE) is defined as

    ..math::

        MSD = \frac{1}{n}\sum\limits_{i=1}^n (x_i - y_i)^2

    MSD can be decomposed into a term describing the deviation of x and y
    attributable to non-perfect correlation (r < 1), a term depending on the
    difference in variances between x and y, and the difference in means
    between x and y (bias).

    ..math::

        MSD &= MSD_{corr} + MSD_{var} + MSD_{bias}\\
            &= 2\sigma_x\sigma_y (1-r) + (\sigma_x - \sigma_y)^2
               + (\mu_x - \mu_y)^2

    This function calculates the full MSD, the function `msd_corr`, `msd_var`,
    and `msd_bias` can be used to calculate the individual components.

    Parameters
    ----------
    x : numpy.ndarray
        First input vector
    y : numpy.ndarray
        Second input vector

    Returns
    -------
    msd : float
        Mean square deviation
    """
    return RSS(x, y) / len(x)


def rmsd(x, y, ddof=0):
    """
    Root-mean-square deviation (RMSD).

    This is the root of MSD (see :py:func:`pytesmo.metrics.msd`).  If `x` and
    `y` have the same mean (i.e. `mean(x - y = 0`) RMSD corresponds to the
    square root of the variance of `x - y`.

    Parameters
    ----------
    x : numpy.ndarray
        First input vector.
    y : numpy.ndarray
        Second input vector.
    ddof : int, optional, DEPRECATED
        Delta degree of freedom.The divisor used in calculations is N - ddof,
        where N represents the number of elements. By default ddof is zero.
        DEPRECATED: ddof is deprecated and might be removed in future versions.

    Returns
    -------
    rmsd : float
        Root-mean-square deviation.
    """
    if ddof == 0:
        return np.sqrt(msd(x, y))
    else:
        warnings.warn(
            "ddof is deprecated and might be removed in future versions of"
            " pytesmo.",
            category=DeprecationWarning,
        )
        return np.sqrt(RSS(x, y) / (len(x) - ddof))


def nrmsd(x, y, ddof=0):
    """
    Normalized root-mean-square deviation (nRMSD).

    This is normalizes RMSD by ``max(max(x), max(y)) - min(min(x), min(y))``.

    Parameters
    ----------
    x : numpy.ndarray
        First input vector.
    y : numpy.ndarray
        Second input vector.
    ddof : int, optional
        Delta degree of freedom.The divisor used in calculations is N - ddof,
        where N represents the number of elements. By default ddof is zero.
        DEPRECATED: ddof is deprecated and might be removed in future versions.

    Returns
    -------
    nrmsd : float
        Normalized root-mean-square deviation (nRMSD).
    """
    return rmsd(x, y, ddof) / (np.max([x, y]) - np.min([x, y]))


def ubrmsd(x, y, ddof=0):
    r"""
    Unbiased root-mean-square deviation (uRMSD).

    This corresponds to RMSD with mean biases removed beforehand, that is

    ..math::

        ubRMSD = \sqrt{\frac{1}{n}\sum\limits_{i=1}^n
                           \left((x - \bar{x}) - (y - \bar{y}))^2}

    NOTE: If you are scaling the data beforehand to have zero mean bias, this
    is exactly the same as RMSD.

    Parameters
    ----------
    x : numpy.ndarray
        First input vector.
    y : numpy.ndarray
        Second input vector.
    ddof : int, optional
        Delta degree of freedom.The divisor used in calculations is N - ddof,
        where N represents the number of elements. By default ddof is zero.
        DEPRECATED: ddof is deprecated and might be removed in future versions.

    Returns
    -------
    ubrmsd : float
        Unbiased root-mean-square deviation (uRMSD).
    """
    if ddof != 0:
        warnings.warn(
            "ddof is deprecated and might be removed in future versions of"
            " pytesmo.",
            DeprecationWarning,
        )
        return np.sqrt(RSS(x - np.mean(x), y - np.mean(y)) / (len(x) - ddof))
    else:
        return _ubrmsd(x, y)


def ubrmsd_ci(x, y, ubrmsd, alpha=0.05):
    """
    Confidende interval for unbiased root-mean-square deviation (uRMSD).

    Parameters
    ----------
    x : numpy.ndarray
        First input vector
    y : numpy.ndarray
        Second input vector
    ubrmsd : float
        ubRMSD for this data
    alpha : float, optional
        1 - confidence level, default is 0.05

    Returns
    -------
    lower, upper : float
        Lower and upper confidence interval bounds.
    """
    n = len(x)
    ubMSD = ubrmsd ** 2
    lb_ubMSD = n * ubMSD / stats.chi2.ppf(1 - alpha / 2, n - 1)
    ub_ubMSD = n * ubMSD / stats.chi2.ppf(alpha / 2, n - 1)
    return np.sqrt(lb_ubMSD), np.sqrt(ub_ubMSD)


def pearson_r(x, y):
    """
    Pearson's linear correlation coefficient.

    Parameters
    ----------
    x : numpy.ndarray
        First input vector.
    y : numpy.ndarray
        Second input vector.

    Returns
    -------
    r : float
        Pearson's correlation coefficent.

    See Also
    --------
    scipy.stats.pearsonr
    """
    return stats.pearsonr(x, y)[0]


def pearson_r_ci(x, y, r, alpha=0.05):
    """
    Confidence interval for Pearson correlation coefficient.

    Parameters
    ----------
    x : numpy.ndarray
        First input vector
    y : numpy.ndarray
        Second input vector
    r : float
        Pearson r for this data
    alpha : float, optional
        1 - confidence level, default is 0.05

    Returns
    -------
    lower, upper : float
        Lower and upper confidence interval bounds.

    References
    ----------
    Bonett, D. G., & Wright, T. A. (2000). Sample size requirements for
    estimating Pearson, Kendall and Spearman correlations. Psychometrika,
    65(1), 23-28.
    """
    n = len(x)
    v = np.arctanh(r)
    z = stats.norm.ppf(1 - alpha / 2)
    cl = v - z / np.sqrt(n - 3)
    cu = v + z / np.sqrt(n - 3)
    return np.tanh(cl), np.tanh(cu)


def spearman_r(x, y):
    """
    Spearman's rank correlation coefficient.

    Parameters
    ----------
    x : numpy.array
        First input vector.
    y : numpy.array
        Second input vector.

    Returns
    -------
    rho : float
        Spearman correlation coefficient

    See Also
    --------
    scipy.stats.spearmenr
    """
    return stats.spearmanr(x, y)[0]


def spearman_r_ci(x, y, r, alpha=0.05):
    """
    Confidence interval for Spearman rank correlation coefficient.

    Parameters
    ----------
    x : numpy.ndarray
        First input vector
    y : numpy.ndarray
        Second input vector
    r : float
        Spearman's r for this data
    alpha : float, optional
        1 - confidence level, default is 0.05

    Returns
    -------
    lower, upper : float
        Lower and upper confidence interval bounds.

    References
    ----------
    Bonett, D. G., & Wright, T. A. (2000). Sample size requirements for
    estimating Pearson, Kendall and Spearman correlations. Psychometrika,
    65(1), 23-28.
    """
    n = len(x)
    v = np.arctanh(r)
    z = stats.norm.ppf(1 - alpha / 2)
    # see reference for this formula
    cl = v - z * np.sqrt(1 + r ** 2 / 2) / np.sqrt(n - 3)
    cu = v + z * np.sqrt(1 + r ** 2 / 2) / np.sqrt(n - 3)
    return np.tanh(cl), np.tanh(cu)


def kendall_tau(x, y):
    """
    Wrapper for scipy.stats.kendalltau

    Parameters
    ----------
    x : numpy.array
        First input vector.
    y : numpy.array
        Second input vector.

    Returns
    -------
    tau : float
        Kendall's tau statistic

    See Also
    --------
    scipy.stats.kendalltau
    """
    return stats.kendalltau(x, y)[0]


def kendall_tau_ci(x, y, tau, alpha=0.05):
    r"""
    Confidence intervall for Kendall's rank coefficient.

    Parameters
    ----------
    x : numpy.ndarray
        First input vector
    y : numpy.ndarray
        Second input vector
    tau : float
        Kendall tau for this data
    alpha : float, optional
        1 - confidence level, default is 0.05

    Returns
    -------
    lower, upper : float
        Lower and upper confidence interval bounds.

    References
    ----------
    Bonett, D. G., & Wright, T. A. (2000). Sample size requirements for
    estimating Pearson, Kendall and Spearman correlations. Psychometrika,
    65(1), 23-28.
    """
    n = len(x)
    v = np.arctanh(tau)
    z = stats.norm.ppf(1 - alpha / 2)
    # see reference for this formula
    cl = v - z * 0.431 / np.sqrt(n - 3)
    cu = v + z * 0.431 / np.sqrt(n - 3)
    return np.tanh(cl), np.tanh(cu)


def index_of_agreement(o, p):
    """
    Index of agreement was proposed by Willmot (1981), to overcome the
    insenstivity of Nash-Sutcliffe efficiency E and R^2 to differences in the
    observed and predicted means and variances (Legates and McCabe, 1999).
    The index of agreement represents the ratio of the mean square error and
    the potential error (Willmot, 1984). The potential error in the denominator
    represents the largest value that the squared difference of each pair can
    attain. The range of d is similar to that of R^2 and lies between
    0 (no correlation) and 1 (perfect fit).

    Parameters
    ----------
    o : numpy.ndarray
        Observations.
    p : numpy.ndarray
        Predictions.

    Returns
    -------
    d : float
        Index of agreement.
    """
    denom = np.sum((np.abs(p - np.mean(o)) + np.abs(o - np.mean(o))) ** 2)
    d = 1 - RSS(o, p) / denom
    return d


def nash_sutcliffe(o, p):
    """
    Nash Sutcliffe model efficiency coefficient E. The range of E lies between
    1.0 (perfect fit) and -inf.

    Parameters
    ----------
    o : numpy.ndarray
        Observations.
    p : numpy.ndarray
        Predictions.

    Returns
    -------
    E : float
        Nash Sutcliffe model efficiency coefficient E.
    """
    return 1 - (np.sum((o - p) ** 2)) / (np.sum((o - np.mean(o)) ** 2))


# Re-apply GPU dispatch AFTER all function definitions.
# The def statements above would otherwise override the dispatch
# assignments at the top of this module.
bias = _select_impl(_gpu_bias, bias)
mse_decomposition = _select_impl(_gpu_mse_decomposition, mse_decomposition)


from functools import wraps


@wraps(_gpu_rmsd)
def _gpu_rmsd_wrap(x, y, ddof=0, *args, **kwargs):
    if ddof != 0:
        warnings.warn(
            "ddof is deprecated and might be removed in future versions of"
            " pytesmo.",
            category=DeprecationWarning,
        )
        return np.sqrt(_cpu_RSS(x, y) / (len(x) - ddof))
    return _gpu_rmsd(x, y)


@wraps(_gpu_ubrmsd)
def _gpu_ubrmsd_wrap(x, y, ddof=0, *args, **kwargs):
    if ddof != 0:
        warnings.warn(
            "ddof is deprecated and might be removed in future versions of"
            " pytesmo.",
            category=DeprecationWarning,
        )
        return np.sqrt(_cpu_RSS(x - np.mean(x), y - np.mean(y)) / (len(x) - ddof))
    return _gpu_ubrmsd(x, y)


rmsd = _select_impl(_gpu_rmsd_wrap, rmsd)
ubrmsd = _select_impl(_gpu_ubrmsd_wrap, ubrmsd)


def _gpu_scalar(f):
    """Extract scalar metric value from a GPU function returning (value, p)."""
    from functools import wraps
    @wraps(f)
    def wrapped(x, y, *args, **kwargs):
        result = f(x, y, *args, **kwargs)
        if isinstance(result, tuple):
            return result[0]
        return result
    return wrapped


pearson_r = _select_impl(_gpu_scalar(_gpu_pearson_r), pearson_r)
spearman_r = _select_impl(_gpu_scalar(_gpu_spearman_r), spearman_r)
kendall_tau = _select_impl(_gpu_scalar(_gpu_kendall_tau), kendall_tau)


def _gpu_rolling_pr_rmsd_to_numpy(*args, **kwargs):
    pr, rmsd = _gpu_rolling_pr_rmsd(*args, **kwargs)
    if hasattr(pr, 'get'):
        pr = pr.get()
    if hasattr(rmsd, 'get'):
        rmsd = rmsd.get()
    return np.asarray(pr), np.asarray(rmsd)


rolling_pr_rmsd = _select_impl(_gpu_rolling_pr_rmsd_to_numpy, rolling_pr_rmsd)
