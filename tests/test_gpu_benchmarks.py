"""
Benchmark tests validating GPU speedup for pytesmo metric functions.

These tests verify that GPU-accelerated code paths are at least as fast as
the CPU reference implementations. They do not assert strict speedup ratios
(system-dependent), but fail if the GPU path is pathologically slow or broken.
"""

import time

import numpy as np
import pytest

gpu_available = True
try:
    import cupy as cp  # noqa: F401
except ImportError:
    gpu_available = False


def _bench(func, repeats=3):
    """Time a function over several repeats, returning min elapsed seconds."""
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        func()
        best = min(best, time.perf_counter() - t0)
    return best


@pytest.mark.skipif(not gpu_available, reason="cupy not installed")
class TestGPUSpeedup:
    def test_bootstrap_speedup(self):
        """GPU bootstrap should not be dramatically slower than CPU."""
        from pytesmo.gpu.bootstrap import (
            with_bootstrapped_ci as gpu_with_bootstrapped_ci,
        )
        from pytesmo.gpu.pairwise import pearson_r as gpu_pearson
        from pytesmo.metrics import with_bootstrapped_ci
        from pytesmo.metrics.pairwise import pearson_r as cpu_pearson

        rng = np.random.RandomState(42)
        x = rng.randn(2000)
        y = rng.randn(2000)

        cpu_t = _bench(
            lambda: with_bootstrapped_ci(
                cpu_pearson, x, y, nsamples=200, method="percentile",
            ),
            repeats=2,
        )
        gpu_t = _bench(
            lambda: gpu_with_bootstrapped_ci(
                gpu_pearson, x, y, nsamples=200, method="percentile",
            ),
            repeats=2,
        )

        assert gpu_t < max(cpu_t * 5, 0.5), (
            f"GPU bootstrap ({gpu_t:.3f}s) unexpectedly slower than "
            f"CPU ({cpu_t:.3f}s)"
        )

    def test_welford_batch_speedup(self):
        """Vectorized GPU batch moments should beat per-point CPU loops."""
        from pytesmo.gpu.pairwise import _welford_batch
        from pytesmo.metrics.pairwise import pearson_r as cpu_pearson_r

        rng = np.random.RandomState(1)
        n_pairs, n_samples = 100, 1000
        x = rng.randn(n_pairs, n_samples)
        y = rng.randn(n_pairs, n_samples)

        # CPU reference: process 100 pairs serially
        def cpu_loop():
            for i in range(n_pairs):
                cpu_pearson_r(x[i], y[i])
        cpu_t = _bench(cpu_loop, repeats=2)

        # GPU: full batch vectorized Welford (all 100 pairs at once)
        gpu_t = _bench(lambda: _welford_batch(x, y), repeats=2)

        assert gpu_t < max(cpu_t * 5, 0.5), (
            f"GPU batch ({gpu_t:.3f}s) unexpectedly slower than "
            f"CPU loop ({cpu_t:.3f}s)"
        )

    def test_tcol_batch_speedup(self):
        """GPU tcol metrics should be at least as fast as CPU tcol."""
        from pytesmo.gpu.tcol import tcol_metrics as gpu_tcol
        from pytesmo.metrics.tcol import tcol_metrics as cpu_tcol

        rng = np.random.RandomState(7)
        x = rng.randn(1000)
        y = rng.randn(1000)
        z = rng.randn(1000)

        cpu_t = _bench(lambda: cpu_tcol(x, y, z), repeats=2)
        gpu_t = _bench(lambda: gpu_tcol(x, y, z), repeats=2)

        assert gpu_t < max(cpu_t * 5, 0.5), (
            f"GPU tcol ({gpu_t:.3f}s) unexpectedly slower than "
            f"CPU ({cpu_t:.3f}s)"
        )
