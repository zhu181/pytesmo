"""
GPU-accelerated bootstrapping for confidence intervals.

Implements vectorized bootstrap resampling with percentile, basic, and BCa methods.
Supports both pairwise and triple collocation metrics.
"""

from .backend import get_gpu_module
from .array_ops import (
    mean, std, var, sqrt, square, sum, abs as xp_abs,
    nanquantile, clip, asarray, float64, nan, isnan, isfinite,
    empty, zeros, ones, concatenate, stack, random_choice, random_normal
)
from .tcol import tcol_metrics


def _jackknife(metric_func, x, y):
    """
    Jackknife resampling for BCa acceleration parameter.
    
    Parameters
    ----------
    metric_func : callable
        Function taking (x, y) returning scalar
    x, y : array, shape (n_samples,)
    
    Returns
    -------
    jk : array, shape (n_samples,)
        Jackknife replicates
    """
    xp = get_gpu_module()
    n = len(x)
    jk = xp.empty(n, dtype=float64())
    
    mask = ones(n, dtype=bool)
    for i in range(n):
        mask[i] = False
        jk[i] = metric_func(x[mask], y[mask])
        mask[i] = True
    
    return jk


def _jackknife_tcol(x, y, z, ref_ind=0):
    """Jackknife for triple collocation metrics."""
    from .tcol import tcol_metrics
    xp = get_gpu_module()
    n = len(x)
    
    snr_jk = xp.empty((n, 3), dtype=float64())
    err_std_jk = xp.empty((n, 3), dtype=float64())
    beta_jk = xp.empty((n, 3), dtype=float64())
    
    mask = ones(n, dtype=bool)
    for i in range(n):
        mask[i] = False
        snr, err_std, beta = tcol_metrics(x[mask], y[mask], z[mask], ref_ind=ref_ind)
        snr_jk[i] = snr
        err_std_jk[i] = err_std
        beta_jk[i] = beta
        mask[i] = True
    
    return snr_jk, err_std_jk, beta_jk


def _call_metric(metric_func, xi, yi):
    """Call a metric function, falling back to numpy inputs if needed."""
    xp = get_gpu_module()
    try:
        return asarray(metric_func(xi, yi))
    except TypeError:
        # metric function only accepts numpy arrays
        xi_np = xi.get() if hasattr(xi, 'get') else xi
        yi_np = yi.get() if hasattr(yi, 'get') else yi
        return asarray(metric_func(xi_np, yi_np))


def _percentile_ci(bs_metrics, orig_metric, alpha):
    """Percentile bootstrap confidence interval."""
    lower = nanquantile(bs_metrics, alpha / 2, axis=0)
    upper = nanquantile(bs_metrics, 1 - alpha / 2, axis=0)
    return lower, upper


def _basic_ci(bs_metrics, orig_metric, alpha):
    """Basic bootstrap confidence interval."""
    lower = 2 * orig_metric - nanquantile(bs_metrics, 1 - alpha / 2, axis=0)
    upper = 2 * orig_metric - nanquantile(bs_metrics, alpha / 2, axis=0)
    return lower, upper


def _bca_ci(bs_metrics, orig_metric, alpha, jk_metrics):
    """BCa (bias-corrected and accelerated) bootstrap confidence interval."""
    from scipy import stats
    import numpy as np
    xp = get_gpu_module()
    
    # Bias correction
    bias_correction = mean(bs_metrics <= orig_metric, axis=0)
    bias_correction = asarray(bias_correction)
    if hasattr(bias_correction, 'get'):
        bias_correction = bias_correction.get()
    z0 = stats.norm.ppf(np.clip(bias_correction, 1e-8, 1 - 1e-8))
    
    # Acceleration
    jk_mean = mean(jk_metrics, axis=0)
    diff = jk_mean - jk_metrics
    num = sum(diff ** 3, axis=0)
    den = 6 * (sum(diff ** 2, axis=0)) ** 1.5
    a = num / den
    a = clip(a, -0.99, 0.99)  # Prevent extreme values
    if hasattr(a, 'get'):
        a = a.get()
    
    # Adjusted percentiles
    z_alpha = stats.norm.ppf(alpha)
    z_1_alpha = stats.norm.ppf(1 - alpha)
    
    alpha_lower = stats.norm.cdf(z0 + (z0 + z_alpha) / (1 - a * (z0 + z_alpha)))
    alpha_upper = stats.norm.cdf(z0 + (z0 + z_1_alpha) / (1 - a * (z0 + z_1_alpha)))
    
    # Compute per-column quantiles when alpha is an array (2D bootstrap case).
    alpha_lower = np.atleast_1d(alpha_lower)
    alpha_upper = np.atleast_1d(alpha_upper)
    alpha_lower = xp.asarray(alpha_lower)
    alpha_upper = xp.asarray(alpha_upper)
    if bs_metrics.ndim == 2 and alpha_lower.size > 1:
        lower = []
        upper = []
        for col in range(bs_metrics.shape[1]):
            lower.append(nanquantile(bs_metrics[:, col], alpha_lower[col]))
            upper.append(nanquantile(bs_metrics[:, col], alpha_upper[col]))
        lower = xp.asarray(lower)
        upper = xp.asarray(upper)
    else:
        lower = nanquantile(bs_metrics, alpha_lower, axis=0)
        upper = nanquantile(bs_metrics, alpha_upper, axis=0)
    
    # Squeeze singleton quantile dimension for 1D input
    if bs_metrics.ndim == 1:
        lower = lower.squeeze() if hasattr(lower, 'squeeze') else lower
        upper = upper.squeeze() if hasattr(upper, 'squeeze') else upper
    
    return lower, upper


def with_bootstrapped_ci(metric_func, x, y, alpha=0.05, method='percentile',
                         nsamples=1000, minimum_data_length=100):
    """
    Bootstrap confidence interval for a pairwise metric.
    
    Vectorized across bootstrap samples for GPU efficiency.
    
    Parameters
    ----------
    metric_func : callable
        Function taking (x, y) returning scalar or array
    x, y : array, shape (n_samples,)
        Input data
    alpha : float, default 0.05
        Confidence level
    method : str, default 'percentile'
        'percentile', 'basic', or 'BCa'
    nsamples : int, default 1000
        Number of bootstrap samples
    minimum_data_length : int, default 100
        Minimum samples required
    
    Returns
    -------
    orig_metric : float or array
        Original metric value
    lower : float or array
        Lower CI bound
    upper : float or array
        Upper CI bound
    """
    xp = get_gpu_module()
    
    # Ensure inputs are on the correct device
    x = asarray(x, dtype=xp.float64)
    y = asarray(y, dtype=xp.float64)
    
    n = len(x)
    
    if n < minimum_data_length:
        raise ValueError(
            f"Not enough data for bootstrapping. Need at least "
            f"{minimum_data_length} samples, got {n}."
        )
    
    # Original metric
    orig_metric = _call_metric(metric_func, x, y)
    orig_metric = asarray(orig_metric)
    
    # Generate bootstrap indices (batch_size, n)
    # Using vectorized random choice
    idx = random_choice(n, size=(nsamples, n), replace=True)
    
    # Vectorized bootstrap - process in chunks to avoid memory issues
    chunk_size = min(100, nsamples)
    bs_metrics = []
    
    for start in range(0, nsamples, chunk_size):
        end = min(start + chunk_size, nsamples)
        idx_chunk = idx[start:end]
        
        # Apply metric to each bootstrap sample
        chunk_results = []
        for i in range(len(idx_chunk)):
            idx_i = idx_chunk[i]
            # Use advanced indexing - idx_i is already on GPU
            xi = x[idx_i]
            yi = y[idx_i]
            result = _call_metric(metric_func, xi, yi)
            # Ensure result is an array
            chunk_results.append(asarray(result))
        
        bs_metrics.append(stack(chunk_results, axis=0))
    
    bs_metrics = concatenate(bs_metrics, axis=0)
    
    # Compute CI
    if method == 'percentile':
        lower, upper = _percentile_ci(bs_metrics, orig_metric, alpha)
    elif method == 'basic':
        lower, upper = _basic_ci(bs_metrics, orig_metric, alpha)
    elif method == 'BCa':
        jk = _jackknife(metric_func, x, y)
        lower, upper = _bca_ci(bs_metrics, orig_metric, alpha, jk)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return orig_metric, lower, upper


def tcol_metrics_with_bootstrapped_ci(x, y, z, ref_ind=0, alpha=0.05,
                                        method='percentile', nsamples=1000,
                                        minimum_data_length=100):
    """
    Bootstrap CI for triple collocation metrics.
    
    Parameters
    ----------
    x, y, z : array, shape (n_samples,)
        Three input datasets
    ref_ind : int, default 0
        Reference index
    alpha : float, default 0.05
        Confidence level
    method : str, default 'percentile'
        'percentile', 'basic', or 'BCa'
    nsamples : int, default 1000
        Number of bootstrap samples
    minimum_data_length : int, default 100
        Minimum samples required
    
    Returns
    -------
    snr_result : tuple (snr, lower, upper)
    err_std_result : tuple (err_std, lower, upper)
    beta_result : tuple (beta, lower, upper)
    """
    xp = get_gpu_module()
    
    # Ensure inputs are on the correct device
    x = asarray(x, dtype=xp.float64)
    y = asarray(y, dtype=xp.float64)
    z = asarray(z, dtype=xp.float64)
    
    n = len(x)
    
    if n < minimum_data_length:
        raise ValueError(
            f"Not enough data for bootstrapping. Need at least "
            f"{minimum_data_length} samples, got {n}."
        )
    
    # Original metrics
    orig_snr, orig_err_std, orig_beta = tcol_metrics(x, y, z, ref_ind=ref_ind)
    
    # Squeeze to handle single batch case
    if orig_snr.ndim > 1:
        orig_snr = orig_snr.squeeze(0)
        orig_err_std = orig_err_std.squeeze(0)
        orig_beta = orig_beta.squeeze(0)
    
    # Generate bootstrap indices
    idx = random_choice(n, size=(nsamples, n), replace=True)
    
    # Vectorized bootstrap
    chunk_size = min(100, nsamples)
    bs_snr = []
    bs_err_std = []
    bs_beta = []
    
    for start in range(0, nsamples, chunk_size):
        end = min(start + chunk_size, nsamples)
        idx_chunk = idx[start:end]
        
        chunk_snr = []
        chunk_err = []
        chunk_beta = []
        
        for i in range(len(idx_chunk)):
            idx_i = idx_chunk[i]
            snr, err, beta = tcol_metrics(x[idx_i], y[idx_i], z[idx_i], ref_ind=ref_ind)
            # Squeeze batch dimension if present
            if snr.ndim > 1:
                snr = snr.squeeze(0)
                err = err.squeeze(0)
                beta = beta.squeeze(0)
            chunk_snr.append(asarray(snr))
            chunk_err.append(asarray(err))
            chunk_beta.append(asarray(beta))
        
        bs_snr.append(stack(chunk_snr, axis=0))
        bs_err_std.append(stack(chunk_err, axis=0))
        bs_beta.append(stack(chunk_beta, axis=0))
    
    bs_snr = concatenate(bs_snr, axis=0)
    bs_err_std = concatenate(bs_err_std, axis=0)
    bs_beta = concatenate(bs_beta, axis=0)
    
    # Compute CIs per metric
    def compute_ci(bs, orig, method, jk=None):
        if method == 'percentile':
            return _percentile_ci(bs, orig, alpha)
        elif method == 'basic':
            return _basic_ci(bs, orig, alpha)
        elif method == 'BCa':
            return _bca_ci(bs, orig, alpha, jk)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    # Jackknife for BCa
    if method == 'BCa':
        snr_jk, err_jk, beta_jk = _jackknife_tcol(x, y, z, ref_ind)
    else:
        snr_jk = err_jk = beta_jk = None
    
    lower_snr, upper_snr = compute_ci(bs_snr, orig_snr, method, snr_jk)
    lower_err, upper_err = compute_ci(bs_err_std, orig_err_std, method, err_jk)
    
    # Beta CI (skip reference dataset)
    lower_beta = []
    upper_beta = []
    for i in range(3):
        if i == ref_ind:
            lb, ub = xp.asarray(1.0), xp.asarray(1.0)
        else:
            lb, ub = compute_ci(bs_beta[:, i], orig_beta[i], method,
                                beta_jk[:, i] if beta_jk is not None else None)
        lower_beta.append(xp.asarray(lb, dtype=xp.float64))
        upper_beta.append(xp.asarray(ub, dtype=xp.float64))
    
    lower_beta = xp.asarray(lower_beta)
    upper_beta = xp.asarray(upper_beta)
    
    return (
        (orig_snr, lower_snr, upper_snr),
        (orig_err_std, lower_err, upper_err),
        (orig_beta, lower_beta, upper_beta),
    )