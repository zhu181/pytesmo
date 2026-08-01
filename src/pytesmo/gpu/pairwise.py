"""
GPU-accelerated pairwise metrics.

Implements vectorized Welford's algorithm for online mean/variance/covariance
computation across batches. All operations use float64 precision.

Key functions:
- bias: Mean difference
- mse_decomposition: MSE + correlation/variance/bias components
- rmsd: Root mean square deviation
- ubrmsd: Unbiased RMSD
- pearson_r: Pearson correlation with p-value
- spearman_r: Spearman rank correlation
- kendall_tau: Kendall's tau
- rolling_pr_rmsd: Rolling window Pearson R and RMSD
"""

from .backend import get_gpu_module
from .array_ops import (
    mean, std, var, cov, sqrt, square, sum, abs as xp_abs,
    arctanh, tanh, log10, maximum, minimum, clip,
    empty, zeros, asarray, float64, nan, isnan, isfinite,
    dot, matmul, concatenate, stack
)


def _check_arrays(x, y):
    """Validate and convert input arrays."""
    xp = get_gpu_module()
    x = asarray(x, dtype=float64())
    y = asarray(y, dtype=float64())
    
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if y.ndim == 1:
        y = y.reshape(1, -1)
    
    if x.shape != y.shape:
        raise ValueError(f"Shape mismatch: {x.shape} vs {y.shape}")
    
    return x, y


def _welford_batch(x, y):
    """
    Vectorized moment computation for batched pairs.
    
    Computes means, variances, and covariance for multiple pairs simultaneously
    using fully vectorized reductions (single kernel per reduction).
    
    Parameters
    ----------
    x : array, shape (batch_size, n_samples)
    y : array, shape (batch_size, n_samples)
    
    Returns
    -------
    mx, my : array, shape (batch_size,)
        Means
    varx, vary : array, shape (batch_size,)
        Variances
    cov_xy : array, shape (batch_size,)
        Covariance
    """
    mx = mean(x, axis=1)
    my = mean(y, axis=1)
    
    xc = x - mx[:, None]
    yc = y - my[:, None]
    
    varx = mean(xc * xc, axis=1)
    vary = mean(yc * yc, axis=1)
    cov_xy = mean(xc * yc, axis=1)
    
    return mx, my, varx, vary, cov_xy


def _welford_single(x, y):
    """
    Moment computation for a single pair (vectorized).
    """
    mx, my, varx, vary, cov_xy = _welford_batch(
        x.reshape(1, -1), y.reshape(1, -1))
    return mx[0], my[0], varx[0], vary[0], cov_xy[0]


def bias(x, y):
    """
    Bias (mean difference) between x and y.
    
    Parameters
    ----------
    x, y : array-like, shape (n_samples,) or (batch_size, n_samples)
    
    Returns
    -------
    bias : float or array, shape (batch_size,)
        Mean(x) - Mean(y)
    """
    x, y = _check_arrays(x, y)
    
    if x.shape[0] == 1:
        return float(mean(x) - mean(y))
    
    mx = mean(x, axis=1)
    my = mean(y, axis=1)
    return mx - my


def mse_decomposition(x, y):
    """
    MSE decomposition into correlation, variance, and bias components.
    
    MSE = MSE_corr + MSE_var + MSE_bias
    = 2*σx*σy*(1-r) + (σx-σy)^2 + (μx-μy)^2
    
    Parameters
    ----------
    x, y : array-like, shape (n_samples,) or (batch_size, n_samples)
    
    Returns
    -------
    mse, mse_corr, mse_bias, mse_var : float or arrays
        MSE and its components
    """
    x, y = _check_arrays(x, y)
    batch_size = x.shape[0]
    
    if batch_size == 1:
        mx, my, varx, vary, cov_xy = _welford_single(x.ravel(), y.ravel())
    else:
        mx, my, varx, vary, cov_xy = _welford_batch(x, y)
    
    stdx = sqrt(varx)
    stdy = sqrt(vary)
    
    # Correlation coefficient
    # Avoid NaN when either variance is zero (constant input): the
    # correlation component of MSE is then 0 regardless of r.
    xp = get_gpu_module()
    denom = stdx * stdy
    safe_denom = xp.where(denom == 0, xp.asarray(1.0), denom)
    r = cov_xy / safe_denom
    r = clip(r, -1.0, 1.0)
    mse_corr = 2 * denom * (1 - r)
    mse_var = (stdx - stdy) ** 2
    mse_bias = (mx - my) ** 2
    mse = mse_corr + mse_var + mse_bias
    
    if batch_size == 1:
        return float(mse), float(mse_corr), float(mse_bias), float(mse_var)
    
    return mse, mse_corr, mse_bias, mse_var


def rmsd(x, y):
    """
    Root Mean Square Deviation.
    
    Parameters
    ----------
    x, y : array-like
    
    Returns
    -------
    rmsd : float or array
    """
    mse, _, _, _ = mse_decomposition(x, y)
    result = sqrt(mse)
    if isinstance(mse, float):
        return float(result)
    return result


def ubrmsd(x, y):
    """
    Unbiased Root Mean Square Deviation (bias removed).
    
    uRMSD = sqrt(MSE - MSE_bias) = sqrt(MSE_corr + MSE_var)
    
    Parameters
    ----------
    x, y : array-like
    
    Returns
    -------
    ubrmsd : float or array
    """
    _, mse_corr, _, mse_var = mse_decomposition(x, y)
    result = sqrt(mse_corr + mse_var)
    if isinstance(mse_corr, float):
        return float(result)
    return result


def pearson_r(x, y):
    """
    Pearson correlation coefficient with p-value.
    
    Parameters
    ----------
    x, y : array-like, shape (n_samples,) or (batch_size, n_samples)
    
    Returns
    -------
    r : float or array
        Pearson correlation coefficient
    p : float or array
        Two-tailed p-value
    """
    x, y = _check_arrays(x, y)
    batch_size = x.shape[0]
    n = x.shape[1]
    
    if batch_size == 1:
        mx, my, varx, vary, cov_xy = _welford_single(x.ravel(), y.ravel())
    else:
        mx, my, varx, vary, cov_xy = _welford_batch(x, y)
    
    stdx = sqrt(varx)
    stdy = sqrt(vary)
    
    r = cov_xy / (stdx * stdy)
    r = clip(r, -1.0, 1.0)
    
    # p-value using t-distribution
    df = n - 2
    t_squared = r * r * (df / ((1.0 - r) * (1.0 + r)))
    t_squared = clip(t_squared, 0, None)
    
    # Use regularized incomplete beta function for p-value
    # p = betainc(df/2, 0.5, df/(df + t^2))
    from scipy.special import betainc
    z = df / (df + t_squared)
    p = betainc(0.5 * df, 0.5, z)
    
    if batch_size == 1:
        return float(r), float(p)
    
    return r, p


def spearman_r(x, y):
    """
    Spearman rank correlation coefficient.
    
    Note: This falls back to CPU (scipy) for ranking, then computes on GPU.
    
    Parameters
    ----------
    x, y : array-like
    
    Returns
    -------
    rho : float or array
        Spearman correlation
    p : float or array
        p-value
    """
    from scipy import stats
    xp = get_gpu_module()
    
    x, y = _check_arrays(x, y)
    batch_size = x.shape[0]
    
    results_r = []
    results_p = []
    
    for i in range(batch_size):
        xi = x[i] if batch_size > 1 else x.ravel()
        yi = y[i] if batch_size > 1 else y.ravel()
        
        # Convert to numpy for scipy ranking
        xi_np = xp.asnumpy(xi) if hasattr(xi, 'get') else xi
        yi_np = xp.asnumpy(yi) if hasattr(yi, 'get') else yi
        
        rho, p = stats.spearmanr(xi_np, yi_np)
        results_r.append(rho)
        results_p.append(p)
    
    if batch_size == 1:
        return float(results_r[0]), float(results_p[0])
    
    return asarray(results_r), asarray(results_p)


def kendall_tau(x, y):
    """
    Kendall's tau rank correlation.
    
    Note: Falls back to CPU (scipy) for computation.
    
    Parameters
    ----------
    x, y : array-like
    
    Returns
    -------
    tau : float or array
        Kendall's tau
    p : float or array
        p-value
    """
    from scipy import stats
    xp = get_gpu_module()
    
    x, y = _check_arrays(x, y)
    batch_size = x.shape[0]
    
    results_tau = []
    results_p = []
    
    for i in range(batch_size):
        xi = x[i] if batch_size > 1 else x.ravel()
        yi = y[i] if batch_size > 1 else y.ravel()
        
        xi_np = xp.asnumpy(xi) if hasattr(xi, 'get') else xi
        yi_np = xp.asnumpy(yi) if hasattr(yi, 'get') else yi
        
        tau, p = stats.kendalltau(xi_np, yi_np)
        results_tau.append(tau)
        results_p.append(p)
    
    if batch_size == 1:
        return float(results_tau[0]), float(results_p[0])
    
    return asarray(results_tau), asarray(results_p)


def rolling_pr_rmsd(timestamps, x, y, window_size, center=True, min_periods=2):
    """
    Rolling window Pearson R and RMSD.
    
    Vectorized Welford-based rolling computation across batch dimension.
    
    Parameters
    ----------
    timestamps : array, shape (n_samples,) or (batch_size, n_samples)
        Julian date timestamps
    x, y : array, shape (n_samples,) or (batch_size, n_samples)
        Time series data
    window_size : float
        Window size in days
    center : bool
        Center window on current point
    min_periods : int
        Minimum observations in window
    
    Returns
    -------
    pr_arr : array, shape (n_samples, 2) or (batch_size, n_samples, 2)
        Rolling Pearson R and p-value
    rmsd_arr : array, shape (n_samples,) or (batch_size, n_samples)
        Rolling RMSD
    """
    xp = get_gpu_module()
    
    # Handle single series vs batch
    if x.ndim == 1:
        timestamps = timestamps.reshape(1, -1)
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        squeeze_output = True
    else:
        squeeze_output = False
    
    batch_size, n_ts = x.shape
    
    # Allocate output arrays
    pr_arr = xp.empty((batch_size, n_ts, 2), dtype=float64())
    rmsd_arr = xp.empty((batch_size, n_ts), dtype=float64())
    
    # Process each batch independently (could be parallelized)
    for b in range(batch_size):
        ts = timestamps[b]
        xb = x[b]
        yb = y[b]
        
        # Rolling Welford algorithm
        mx = my = msd = M2x = M2y = C = 0.0
        rolling_nobs = 0.0
        lower = 0
        upper = -1
        
        for i in range(n_ts):
            lold = lower
            uold = upper
            
            # Find window bounds
            if center:
                # Find new start
                for j in range(lower, n_ts):
                    lower = j
                    if ts[j] >= ts[i] - window_size:
                        break
                
                # Find new end
                if ts[n_ts - 1] > ts[i] + window_size:
                    if i == 0:
                        upper = 1
                    for j in range(upper, n_ts):
                        upper = j - 1
                        if ts[j] > ts[i] + window_size:
                            break
                else:
                    upper = n_ts - 1
            else:
                for j in range(lower, n_ts):
                    lower = j
                    if ts[j] > ts[i] - window_size:
                        break
                upper = i
            
            # Add new observations
            for j in range(uold + 1, upper + 1):
                mxold = mx
                myold = my
                rolling_nobs += 1
                mx += (xb[j] - mx) / rolling_nobs
                my += (yb[j] - my) / rolling_nobs
                msd += ((xb[j] - yb[j])**2 - msd) / rolling_nobs
                M2x += (xb[j] - mx) * (xb[j] - mxold)
                M2y += (yb[j] - my) * (yb[j] - myold)
                C += (xb[j] - mx) * (yb[j] - myold)
            
            # Remove old observations
            for j in range(lold, lower):
                mxold = mx
                myold = my
                rolling_nobs -= 1
                if rolling_nobs > 0:
                    mx -= (xb[j] - mx) / rolling_nobs
                    my -= (yb[j] - my) / rolling_nobs
                    msd -= ((xb[j] - yb[j])**2 - msd) / rolling_nobs
                    M2x -= (xb[j] - mxold) * (xb[j] - mx)
                    M2y -= (yb[j] - myold) * (yb[j] - my)
                    C -= (xb[j] - mxold) * (yb[j] - my)
            
            num_obs = upper - lower + 1
            if num_obs == 0 or num_obs < min_periods:
                pr_arr[b, i, 0] = nan()
                pr_arr[b, i, 1] = nan()
                rmsd_arr[b, i] = nan()
            else:
                # Pearson R from moments
                if M2x == 0 or M2y == 0:
                    r = nan()
                    p = nan()
                else:
                    r = C / sqrt(M2x * M2y)
                    r = clip(r, -1.0, 1.0)
                    
                    df = num_obs - 2
                    if abs(r) == 1.0:
                        p = 0.0
                    else:
                        t_squared = r * r * (df / ((1.0 - r) * (1.0 + r)))
                        from scipy.special import betainc
                        z = min(float(df) / (df + t_squared), 1.0)
                        p = betainc(0.5 * df, 0.5, z)
                
                pr_arr[b, i, 0] = r
                pr_arr[b, i, 1] = p
                rmsd_arr[b, i] = sqrt(msd)
    
    if squeeze_output:
        return pr_arr[0], rmsd_arr[0]
    
    return pr_arr, rmsd_arr


# Confidence interval functions (analytical)
def bias_ci(x, y, b, alpha=0.05):
    """Confidence interval for bias."""
    from scipy import stats
    xp = get_gpu_module()
    
    x, y = _check_arrays(x, y)
    n = x.shape[1] if x.ndim > 1 else len(x)
    
    if x.ndim == 1:
        diff = x - y
    else:
        diff = x - y
    
    std_diff = std(diff, axis=-1, ddof=1)
    delta = std_diff / sqrt(n) * stats.t.ppf(1 - alpha / 2, n - 1)
    return b - delta, b + delta


def ubrmsd_ci(x, y, ubrmsd_val, alpha=0.05):
    """Confidence interval for uRMSD."""
    from scipy import stats
    xp = get_gpu_module()
    
    n = x.shape[1] if x.ndim > 1 else len(x)
    ubMSD = ubrmsd_val ** 2
    lb = n * ubMSD / stats.chi2.ppf(1 - alpha / 2, n - 1)
    ub = n * ubMSD / stats.chi2.ppf(alpha / 2, n - 1)
    return sqrt(lb), sqrt(ub)


def pearson_r_ci(x, y, r, alpha=0.05):
    """Confidence interval for Pearson r."""
    from scipy import stats
    xp = get_gpu_module()
    
    n = x.shape[1] if x.ndim > 1 else len(x)
    v = arctanh(r)
    z = stats.norm.ppf(1 - alpha / 2)
    cl = v - z / sqrt(n - 3)
    cu = v + z / sqrt(n - 3)
    return tanh(cl), tanh(cu)


def spearman_r_ci(x, y, r, alpha=0.05):
    """Confidence interval for Spearman rho."""
    from scipy import stats
    xp = get_gpu_module()
    
    n = x.shape[1] if x.ndim > 1 else len(x)
    v = arctanh(r)
    z = stats.norm.ppf(1 - alpha / 2)
    cl = v - z * sqrt(1 + r**2 / 2) / sqrt(n - 3)
    cu = v + z * sqrt(1 + r**2 / 2) / sqrt(n - 3)
    return tanh(cl), tanh(cu)


def kendall_tau_ci(x, y, tau, alpha=0.05):
    """Confidence interval for Kendall's tau."""
    from scipy import stats
    xp = get_gpu_module()
    
    n = x.shape[1] if x.ndim > 1 else len(x)
    v = arctanh(tau)
    z = stats.norm.ppf(1 - alpha / 2)
    cl = v - z * 0.431 / sqrt(n - 3)
    cu = v + z * 0.431 / sqrt(n - 3)
    return tanh(cl), tanh(cu)