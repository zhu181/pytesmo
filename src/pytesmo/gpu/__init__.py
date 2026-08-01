"""
GPU acceleration module for pytesmo.

This module provides CuPy-based GPU acceleration for core numerical computations
including pairwise metrics, triple collocation, bootstrapping, and rolling window
calculations. It automatically falls back to NumPy if CuPy is not available.

Usage:
    from pytesmo.gpu import get_gpu_module, is_gpu_available
    
    xp = get_gpu_module()  # Returns cupy or numpy
    if is_gpu_available():
        print("GPU acceleration enabled")
"""

from .backend import GPUContext, get_gpu_module, is_gpu_available, to_device, to_host, get_device_info
from .array_ops import (
    mean, std, var, cov, corrcoef, sqrt, square, sum, abs,
    arctanh, tanh, log10, maximum, minimum, clip,
    concatenate, stack, empty, zeros, ones, full,
    asarray, asnumpy, float64, float32, int32, int64,
    nan, nanquantile, isnan, isfinite, all, any
)
from .pairwise import (
    bias, mse_decomposition, rmsd, ubrmsd, pearson_r,
    spearman_r, kendall_tau, rolling_pr_rmsd
)
from .tcol import tcol_metrics
from .bootstrap import (
    with_bootstrapped_ci, tcol_metrics_with_bootstrapped_ci
)
from .rolling import rolling_pr_rmsd as gpu_rolling_pr_rmsd

__all__ = [
    'GPUContext',
    'get_gpu_module',
    'is_gpu_available',
    'to_device',
    'to_host',
    'mean', 'std', 'var', 'cov', 'corrcoef', 'sqrt', 'square', 'sum', 'abs',
    'arctanh', 'tanh', 'log10', 'maximum', 'minimum', 'clip',
    'concatenate', 'stack', 'empty', 'zeros', 'ones', 'full',
    'asarray', 'asnumpy', 'float64', 'float32', 'int32', 'int64',
    'nan', 'nanquantile', 'isnan', 'isfinite', 'all', 'any',
    'bias', 'mse_decomposition', 'rmsd', 'ubrmsd', 'pearson_r',
    'spearman_r', 'kendall_tau', 'rolling_pr_rmsd',
    'tcol_metrics',
    'with_bootstrapped_ci', 'tcol_metrics_with_bootstrapped_ci',
    'gpu_rolling_pr_rmsd',
]