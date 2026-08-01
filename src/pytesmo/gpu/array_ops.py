"""
Unified array operations for GPU (CuPy) and CPU (NumPy).

This module provides a consistent API for array operations that works
with both CuPy and NumPy, defaulting to float64 precision.
"""

from .backend import get_gpu_module


def _get_module():
    """Get the active array module."""
    return get_gpu_module()


# Type aliases (module-dependent)
def float64():
    return _get_module().float64


def float32():
    return _get_module().float32


def int32():
    return _get_module().int32


def int64():
    return _get_module().int64


def nan():
    return _get_module().nan


# Array creation
def empty(shape, dtype=None):
    xp = _get_module()
    if dtype is None:
        dtype = xp.float64
    return xp.empty(shape, dtype=dtype)


def zeros(shape, dtype=None):
    xp = _get_module()
    if dtype is None:
        dtype = xp.float64
    return xp.zeros(shape, dtype=dtype)


def ones(shape, dtype=None):
    xp = _get_module()
    if dtype is None:
        dtype = xp.float64
    return xp.ones(shape, dtype=dtype)


def full(shape, fill_value, dtype=None):
    xp = _get_module()
    if dtype is None:
        dtype = xp.float64
    return xp.full(shape, fill_value, dtype=dtype)


def asarray(arr, dtype=None):
    xp = _get_module()
    if dtype is None:
        dtype = xp.float64
    return xp.asarray(arr, dtype=dtype)


def asnumpy(arr):
    """Convert to NumPy array (always returns numpy.ndarray)."""
    xp = _get_module()
    if hasattr(arr, 'get'):
        return arr.get()
    return xp.asarray(arr)


# Basic operations
def mean(a, axis=None, keepdims=False):
    return _get_module().mean(a, axis=axis, keepdims=keepdims)


def std(a, axis=None, keepdims=False, ddof=0):
    return _get_module().std(a, axis=axis, keepdims=keepdims, ddof=ddof)


def var(a, axis=None, keepdims=False, ddof=0):
    return _get_module().var(a, axis=axis, keepdims=keepdims, ddof=ddof)


def sum(a, axis=None, keepdims=False):
    return _get_module().sum(a, axis=axis, keepdims=keepdims)


def abs(a):
    return _get_module().abs(a)


def sqrt(a):
    return _get_module().sqrt(a)


def square(a):
    return _get_module().square(a)


def maximum(a, b):
    return _get_module().maximum(a, b)


def minimum(a, b):
    return _get_module().minimum(a, b)


def clip(a, a_min, a_max):
    return _get_module().clip(a, a_min, a_max)


# Transcendental functions
def arctanh(a):
    return _get_module().arctanh(a)


def tanh(a):
    return _get_module().tanh(a)


def log10(a):
    return _get_module().log10(a)


# Covariance and correlation
def cov(m, y=None, rowvar=True, bias=False, ddof=None):
    return _get_module().cov(m, y=y, rowvar=rowvar, bias=bias, ddof=ddof)


def corrcoef(x, y=None, rowvar=True):
    return _get_module().corrcoef(x, y=y, rowvar=rowvar)


# Stacking and concatenation
def concatenate(arrays, axis=0):
    return _get_module().concatenate(arrays, axis=axis)


def stack(arrays, axis=0):
    return _get_module().stack(arrays, axis=axis)


# NaN handling
def isnan(a):
    return _get_module().isnan(a)


def isfinite(a):
    return _get_module().isfinite(a)


def nanquantile(a, q, axis=None):
    xp = _get_module()
    if hasattr(xp, 'nanquantile'):
        return xp.nanquantile(a, q, axis=axis)
    # Fallback for older versions
    return xp.percentile(a, q * 100, axis=axis)


# Reduction operations
def all(a, axis=None):
    return _get_module().all(a, axis=axis)


def any(a, axis=None):
    return _get_module().any(a, axis=axis)


# Advanced indexing helpers
def where(condition, x=None, y=None):
    return _get_module().where(condition, x, y)


def argsort(a, axis=-1):
    return _get_module().argsort(a, axis=axis)


def sort(a, axis=-1):
    return _get_module().sort(a, axis=axis)


def unique(a, return_index=False, return_inverse=False, return_counts=False):
    return _get_module().unique(a, return_index=return_index, 
                                 return_inverse=return_inverse, return_counts=return_counts)


# Linear algebra
def dot(a, b):
    return _get_module().dot(a, b)


def matmul(a, b):
    return _get_module().matmul(a, b)


def linalg_inv(a):
    return _get_module().linalg.inv(a)


def linalg_svd(a, full_matrices=True):
    return _get_module().linalg.svd(a, full_matrices=full_matrices)


def linalg_eigh(a):
    return _get_module().linalg.eigh(a)


# Random number generation (for bootstrapping)
def random_choice(a, size=None, replace=True, p=None):
    xp = _get_module()
    if hasattr(xp, 'random') and hasattr(xp.random, 'choice'):
        return xp.random.choice(a, size=size, replace=replace, p=p)
    # Fallback
    return xp.random.choice(a, size=size, replace=replace, p=p)


def random_normal(loc=0.0, scale=1.0, size=None):
    xp = _get_module()
    return xp.random.normal(loc, scale, size)


def random_uniform(low=0.0, high=1.0, size=None):
    xp = _get_module()
    return xp.random.uniform(low, high, size)


def random_standard_normal(size=None):
    xp = _get_module()
    return xp.random.standard_normal(size)


# Device synchronization
def synchronize():
    from .backend import synchronize
    synchronize()


# Memory management
def free_memory():
    from .backend import free_memory
    free_memory()