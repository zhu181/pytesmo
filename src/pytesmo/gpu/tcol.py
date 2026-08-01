"""
GPU-accelerated Triple Collocation Analysis.

Implements TCA metrics computation from covariance matrices with batch support.
All operations use float64 precision.
"""

from .backend import get_gpu_module
from .array_ops import (
    mean, std, var, cov, sqrt, square, sum, abs as xp_abs,
    log10, maximum, minimum, clip, asarray, float64, nan,
    concatenate, stack, empty, zeros, dot, matmul, linalg_inv
)


def _check_triple(x, y, z):
    """Validate and convert three input arrays to batch format."""
    xp = get_gpu_module()
    x = asarray(x, dtype=float64())
    y = asarray(y, dtype=float64())
    z = asarray(z, dtype=float64())
    
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if y.ndim == 1:
        y = y.reshape(1, -1)
    if z.ndim == 1:
        z = z.reshape(1, -1)
    
    if x.shape != y.shape or x.shape != z.shape:
        raise ValueError(f"Shape mismatch: {x.shape}, {y.shape}, {z.shape}")
    
    return x, y, z


def _batch_covariance(x, y, z):
    """
    Compute covariance matrices for batched triplets.
    
    Parameters
    ----------
    x, y, z : array, shape (batch_size, n_samples)
    
    Returns
    -------
    cov : array, shape (batch_size, 3, 3)
        Covariance matrices for each batch
    """
    xp = get_gpu_module()
    batch_size, n = x.shape
    
    # Stack into (batch_size, 3, n)
    data = stack([x, y, z], axis=1)
    
    # Center the data
    means = mean(data, axis=2, keepdims=True)
    centered = data - means
    
    # Compute covariance: (batch, 3, n) @ (batch, n, 3) / (n-1) -> (batch, 3, 3)
    cov = matmul(centered, centered.transpose(0, 2, 1)) / (n - 1)
    
    return cov


def _tcol_from_cov(cov, ref_ind=0):
    """
    Compute TCA metrics from covariance matrices.
    
    Parameters
    ----------
    cov : array, shape (batch_size, 3, 3) or (3, 3)
        Covariance matrix/matrices
    ref_ind : int
        Reference index (0, 1, or 2)
    
    Returns
    -------
    snr : array, shape (batch_size, 3) or (3,)
        Signal-to-noise ratio [dB]
    err_std : array, shape (batch_size, 3) or (3,)
        Scaled error standard deviation
    beta : array, shape (batch_size, 3) or (3,)
        Scaling coefficients
    """
    xp = get_gpu_module()
    
    # Handle single vs batch
    if cov.ndim == 2:
        cov = cov.reshape(1, 3, 3)
        squeeze = True
    else:
        squeeze = False
    
    batch_size = cov.shape[0]
    ind = (0, 1, 2, 0, 1, 2)
    
    snr_list = []
    err_std_list = []
    beta_list = []
    
    for b in range(batch_size):
        c = cov[b]
        no_ref_ind = [i for i in range(3) if i != ref_ind]
        
        # SNR
        snr_b = []
        for i in range(3):
            num = abs(c[i, i] * c[ind[i + 1], ind[i + 2]])
            den = abs(c[i, ind[i + 1]] * c[i, ind[i + 2]])
            ratio = num / den
            snr_val = 10 * log10(abs(ratio - 1) ** (-1))
            snr_b.append(snr_val)
        
        # Error variance
        err_var_b = []
        for i in range(3):
            err_var = c[i, i] - (c[i, ind[i + 1]] * c[i, ind[i + 2]]) / c[ind[i + 1], ind[i + 2]]
            err_var_b.append(err_var)
        
        # Beta (scaling coefficients)
        beta_b = []
        for i in range(3):
            if i == ref_ind:
                beta_b.append(1.0)
            else:
                other = no_ref_ind[0] if no_ref_ind[0] != i else no_ref_ind[1]
                beta_val = float(c[ref_ind, other] / c[i, other])
                beta_b.append(beta_val)
        
        snr_list.append(xp.asarray(snr_b, dtype=xp.float64))
        err_std_list.append(xp.asarray([sqrt(float(ev)) * b for ev, b in zip(err_var_b, beta_b)], dtype=xp.float64))
        beta_list.append(xp.asarray(beta_b, dtype=xp.float64))
    
    snr = xp.stack(snr_list)
    err_std = xp.stack(err_std_list)
    beta = xp.stack(beta_list)
    
    if squeeze:
        return snr[0], err_std[0], beta[0]
    
    return snr, err_std, beta


def tcol_metrics(x, y, z, ref_ind=0):
    """
    Triple collocation metrics: SNR, error std, and scaling coefficients.
    
    Parameters
    ----------
    x, y, z : array-like, shape (n_samples,) or (batch_size, n_samples)
        Three input datasets
    ref_ind : int, default 0
        Index of reference dataset for scaling
    
    Returns
    -------
    snr : array, shape (3,) or (batch_size, 3)
        Signal-to-noise ratio [dB]
    err_std : array, shape (3,) or (batch_size, 3)
        Scaled error standard deviation
    beta : array, shape (3,) or (batch_size, 3)
        Scaling coefficients (beta[ref_ind] = 1)
    """
    x, y, z = _check_triple(x, y, z)
    
    # Compute batched covariance
    cov = _batch_covariance(x, y, z)
    
    # Compute metrics from covariance
    return _tcol_from_cov(cov, ref_ind)


def ecol(data, correlated=None, err_cov=None, abs_est=True):
    """
    Extended collocation analysis for N > 3 datasets.
    
    Note: This implementation falls back to CPU for the linear algebra
    solve step due to complexity of batched linear system solving.
    
    Parameters
    ----------
    data : array, shape (n_samples, n_datasets) or (batch_size, n_samples, n_datasets)
        Input data
    correlated : list of tuples, optional
        Pairs of dataset indices with correlated errors
    err_cov : tuple, optional
        (idx1, idx2, covariance) for known error covariance
    abs_est : bool, default True
        Force absolute values for variance estimates
    
    Returns
    -------
    dict
        Dictionary with signal/error variances, SNR, and error covariances
    """
    xp = get_gpu_module()
    
    # For now, fall back to CPU implementation for ecol
    # due to complex batched linear system construction
    from pytesmo.metrics.tcol import ecol as cpu_ecol
    
    if data.ndim == 3:
        # Process each batch independently
        results = []
        for b in range(data.shape[0]):
            results.append(cpu_ecol(data[b], correlated, err_cov, abs_est))
        return results
    else:
        return cpu_ecol(data, correlated, err_cov, abs_est)