"""
GPU acceleration tests: CPU vs GPU numerical equivalence.

These tests verify that the GPU (CuPy) implementations of the core
metrics produce results numerically equivalent to the CPU (NumPy)
implementations. They are skipped when no GPU is available.
"""

import numpy as np
import numpy.testing as nptest
import pytesmo.metrics
import pytest
from pytesmo.gpu import is_gpu_available
from pytesmo.gpu.bootstrap import (
    tcol_metrics_with_bootstrapped_ci as gpu_tcol_metrics_with_bootstrapped_ci,
)
from pytesmo.gpu.bootstrap import (
    with_bootstrapped_ci as gpu_with_bootstrapped_ci,
)
from pytesmo.gpu.pairwise import (
    bias as gpu_bias,
)
from pytesmo.gpu.pairwise import (
    kendall_tau as gpu_kendall_tau,
)
from pytesmo.gpu.pairwise import (
    mse_decomposition as gpu_mse_decomposition,
)
from pytesmo.gpu.pairwise import (
    pearson_r as gpu_pearson_r,
)
from pytesmo.gpu.pairwise import (
    rmsd as gpu_rmsd,
)
from pytesmo.gpu.pairwise import (
    spearman_r as gpu_spearman_r,
)
from pytesmo.gpu.pairwise import (
    ubrmsd as gpu_ubrmsd,
)
from pytesmo.gpu.tcol import tcol_metrics as gpu_tcol_metrics


def _numpy(arr):
    """Convert array to numpy, handling CuPy arrays."""
    if hasattr(arr, 'get'):
        arr = arr.get()
    return np.asarray(arr).squeeze()

requires_gpu = pytest.mark.skipif(
    not is_gpu_available(), reason="GPU (CuPy) not available"
)


@pytest.fixture
def correlated_data():
    """Correlated random data with r ~ 0.8."""
    np.random.seed(42)
    cov = np.array([[1, 0.8], [0.8, 1]])
    X = np.linalg.cholesky(cov) @ np.random.randn(2, 500)  # noqa: N806
    x, y = X[0, :], X[1, :]
    y = 1.1 * y + 0.5
    return x, y


@pytest.fixture
def triple_data():
    """Three datasets with a shared signal."""
    np.random.seed(42)
    n = 500
    signal = np.random.randn(n)
    x = signal + 0.2 * np.random.randn(n)
    y = signal + 0.3 * np.random.randn(n)
    z = signal + 0.25 * np.random.randn(n)
    return x, y, z


@requires_gpu
def test_gpu_available():
    """GPU context should be available and report device info."""
    from pytesmo.gpu import get_device_info
    info = get_device_info()
    assert info['available'] is True
    assert info['name']
    assert info['compute_capability'] != 'unknown'


@requires_gpu
def test_bias_equivalence(correlated_data):
    x, y = correlated_data
    gpu_val = gpu_bias(x, y)
    cpu_val = pytesmo.metrics.bias(x, y)
    assert isinstance(gpu_val, float)
    nptest.assert_almost_equal(gpu_val, cpu_val, decimal=12)


@requires_gpu
def test_mse_decomposition_equivalence(correlated_data):
    x, y = correlated_data
    gpu_mse, gpu_corr, gpu_bias, gpu_var = gpu_mse_decomposition(x, y)
    cpu_mse, cpu_corr, cpu_bias, cpu_var = pytesmo.metrics.mse_decomposition(x, y)
    nptest.assert_almost_equal(gpu_mse, cpu_mse, decimal=12)
    nptest.assert_almost_equal(gpu_corr, cpu_corr, decimal=12)
    nptest.assert_almost_equal(gpu_bias, cpu_bias, decimal=12)
    nptest.assert_almost_equal(gpu_var, cpu_var, decimal=12)


@requires_gpu
def test_rmsd_equivalence(correlated_data):
    x, y = correlated_data
    gpu_val = gpu_rmsd(x, y)
    cpu_val = pytesmo.metrics.rmsd(x, y)
    assert isinstance(gpu_val, float)
    nptest.assert_almost_equal(gpu_val, cpu_val, decimal=12)


@requires_gpu
def test_ubrmsd_equivalence(correlated_data):
    x, y = correlated_data
    gpu_val = gpu_ubrmsd(x, y)
    cpu_val = pytesmo.metrics.ubrmsd(x, y)
    assert isinstance(gpu_val, float)
    nptest.assert_almost_equal(gpu_val, cpu_val, decimal=12)


@requires_gpu
def test_pearson_r_equivalence(correlated_data):
    x, y = correlated_data
    gpu_r, gpu_p = gpu_pearson_r(x, y)
    cpu_r = pytesmo.metrics.pearson_r(x, y)
    nptest.assert_almost_equal(gpu_r, cpu_r, decimal=12)


@requires_gpu
def test_spearman_r_equivalence(correlated_data):
    x, y = correlated_data
    gpu_rho, gpu_p = gpu_spearman_r(x, y)
    cpu_rho = pytesmo.metrics.spearman_r(x, y)
    nptest.assert_almost_equal(gpu_rho, cpu_rho, decimal=12)


@requires_gpu
def test_kendall_tau_equivalence(correlated_data):
    x, y = correlated_data
    gpu_tau, gpu_p = gpu_kendall_tau(x, y)
    cpu_tau = pytesmo.metrics.kendall_tau(x, y)
    nptest.assert_almost_equal(gpu_tau, cpu_tau, decimal=12)


@requires_gpu
def test_constant_data_rmsd():
    """Constant (zero-variance) input must not produce NaN in RMSD."""
    n = 100
    x = np.full(n, 0.5)
    y = np.full(n, 0.5) + 0.2
    gpu_val = gpu_rmsd(x, y)
    cpu_val = pytesmo.metrics.rmsd(x, y)
    assert not np.isnan(gpu_val)
    nptest.assert_almost_equal(gpu_val, cpu_val, decimal=12)


@requires_gpu
def test_tcol_metrics_equivalence(triple_data):
    x, y, z = triple_data
    gpu_snr, gpu_err, gpu_beta = gpu_tcol_metrics(x, y, z, ref_ind=0)
    cpu_snr, cpu_err, cpu_beta = pytesmo.metrics.tcol_metrics(x, y, z, ref_ind=0)
    nptest.assert_almost_equal(_numpy(gpu_snr), cpu_snr, decimal=10)
    nptest.assert_almost_equal(_numpy(gpu_err), cpu_err, decimal=10)
    nptest.assert_almost_equal(_numpy(gpu_beta), cpu_beta, decimal=10)


@requires_gpu
def test_tcol_metrics_beta_ref():
    """Reference dataset beta should be exactly 1."""
    np.random.seed(7)
    n = 200
    signal = np.random.randn(n)
    x = signal + 0.1 * np.random.randn(n)
    y = signal + 0.2 * np.random.randn(n)
    z = signal + 0.15 * np.random.randn(n)
    for ref_ind in (0, 1, 2):
        snr, err, beta = gpu_tcol_metrics(x, y, z, ref_ind=ref_ind)
        snr_c, err_c, beta_c = pytesmo.metrics.tcol_metrics(x, y, z, ref_ind=ref_ind)
        nptest.assert_almost_equal(_numpy(beta)[ref_ind], 1.0, decimal=12)
        nptest.assert_almost_equal(_numpy(beta), beta_c, decimal=10)


@requires_gpu
def test_pairwise_bootstrap_percentile(correlated_data):
    x, y = correlated_data
    from pytesmo.metrics import pairwise
    func = pytesmo.metrics.pairwise.bias
    m, lb, ub = gpu_with_bootstrapped_ci(
        func, x, y, nsamples=100, method='percentile'
    )
    m, lb, ub = _numpy(m), _numpy(lb), _numpy(ub)
    assert lb < m < ub
    m_cpu, lb_cpu, ub_cpu = pytesmo.metrics.with_bootstrapped_ci(
        pairwise.bias, x, y, nsamples=100, method='percentile'
    )
    nptest.assert_almost_equal(m, m_cpu, decimal=6)


@requires_gpu
@pytest.mark.parametrize("method", ["percentile", "basic", "BCa"])
def test_tcol_bootstrap_methods(triple_data, method):
    x, y, z = triple_data
    (snr, snr_l, snr_u), (err, err_l, err_u), (beta, beta_l, beta_u) = (
        gpu_tcol_metrics_with_bootstrapped_ci(
            x, y, z, ref_ind=0, nsamples=50, method=method
        )
    )
    snr, snr_l, snr_u = _numpy(snr), _numpy(snr_l), _numpy(snr_u)
    beta, beta_l, beta_u = _numpy(beta), _numpy(beta_l), _numpy(beta_u)
    assert snr.shape == (3,)
    assert snr_l.shape == (3,)
    assert np.all(snr_l <= snr)
    assert np.all(snr <= snr_u)
    assert np.all(beta_l[1:] <= beta[1:])
    assert np.all(beta[1:] <= beta_u[1:])


@requires_gpu
def test_public_metrics_return_numpy():
    """Public API metrics must return NumPy/Python scalars, not CuPy arrays."""
    np.random.seed(3)
    x = np.random.randn(100)
    y = x + 0.5 * np.random.randn(100)
    for func in [pytesmo.metrics.bias, pytesmo.metrics.rmsd,
                 pytesmo.metrics.ubrmsd, pytesmo.metrics.pearson_r,
                 pytesmo.metrics.spearman_r, pytesmo.metrics.kendall_tau]:
        val = func(x, y)
        assert not isinstance(val, np.ndarray) or val.ndim == 0, func.__name__

    snr, err, beta = pytesmo.metrics.tcol_metrics(x, y, x + 0.1 * np.random.randn(100))
    assert isinstance(snr, np.ndarray)
    assert snr.dtype == np.float64
