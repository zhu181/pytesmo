# coding: utf-8
"""
Integration tests for the full validation workflow with GPU/parallel support.

Runs the complete Validation pipeline (DataManager -> temporal matching ->
metric calculators) with GPU enabled and verifies results match the CPU
baseline.
"""

import numpy as np
import numpy.testing as nptest
import pytest

from pytesmo.validation_framework.data_manager import DataManager
from pytesmo.validation_framework.validation import Validation
from pytesmo.validation_framework.metric_calculators import (
    PairwiseIntercomparisonMetrics,
)
from pytesmo.validation_framework.temporal_matchers import (
    BasicTemporalMatching,
)

from tests.test_validation_framework.test_datasets import setup_TestDatasets

gpu_available = True
try:
    import cupy as cp  # noqa: F401
except ImportError:
    gpu_available = False

dask_available = True
try:
    import dask.distributed  # noqa: F401
except ImportError:
    dask_available = False


def _run_validation(use_gpu=False, **kwargs):
    datasets = setup_TestDatasets()
    dm = DataManager(
        datasets,
        "DS1",
        read_ts_names={d: "read" for d in ["DS1", "DS2", "DS3"]},
    )

    process = Validation(
        dm,
        "DS1",
        temporal_matcher=BasicTemporalMatching(
            window=1 / 24.0).combinatory_matcher,
        metrics_calculators={
            (3, 2): PairwiseIntercomparisonMetrics(
                min_obs=10).calc_metrics
        },
    )

    results = {}
    for gpi, lon, lat in [(1, 1, 1), (2, 2, 2), (3, 3, 2)]:
        res = process.calc([gpi], [lon], [lat], use_gpu=use_gpu, **kwargs)
        results[gpi] = res

    return results


def _flatten(results):
    """Flatten nested results into a dict keyed by (gpi, combo, metric)."""
    flat = {}
    for gpi, combos in results.items():
        for combo, metrics in combos.items():
            for name, val in metrics.items():
                flat[(gpi, combo, name)] = val
    return flat


@pytest.mark.skipif(not gpu_available, reason="cupy not installed")
class TestGPUValidationWorkflow:
    def test_gpu_matches_cpu(self):
        cpu = _flatten(_run_validation(use_gpu=False))
        gpu = _flatten(_run_validation(use_gpu=True))

        assert set(cpu.keys()) == set(gpu.keys())

        for key, cpu_val in cpu.items():
            gpu_val = gpu[key]
            assert cpu_val.shape == gpu_val.shape, f"shape mismatch for {key}"
            nptest.assert_allclose(
                cpu_val, gpu_val, rtol=1e-6, atol=1e-8,
                err_msg=f"value mismatch for {key}",
            )

    def test_gpu_bootstrap_matches_cpu(self):
        """Bootstrap CI results should match between CPU and GPU paths."""
        cpu = _run_validation_bootstrap(use_gpu=False)
        gpu = _run_validation_bootstrap(use_gpu=True)

        assert set(cpu.keys()) == set(gpu.keys())
        for key, cpu_val in cpu.items():
            nptest.assert_allclose(
                cpu_val, gpu[key], rtol=1e-5, atol=1e-6,
                err_msg=f"value mismatch for {key}",
            )


def _run_validation_bootstrap(use_gpu=False):
    datasets = setup_TestDatasets()
    dm = DataManager(
        datasets,
        "DS1",
        read_ts_names={d: "read" for d in ["DS1", "DS2", "DS3"]},
    )

    process = Validation(
        dm,
        "DS1",
        temporal_matcher=BasicTemporalMatching(
            window=1 / 24.0).combinatory_matcher,
        metrics_calculators={
            (3, 2): PairwiseIntercomparisonMetrics(
                min_obs=10, bootstrap_cis=True,
                bootstrap_min_obs=10, bootstrap_alpha=0.05
            ).calc_metrics
        },
    )

    results = _flatten({1: process.calc([1], [1], [1], use_gpu=use_gpu)})
    return results


class TestSequentialIntegration:
    def test_cpu_baseline(self):
        results = _run_validation(use_gpu=False)
        flat = _flatten(results)
        assert len(flat) > 0
        assert any("bias" in k[2] or "RMSD" in k[2] for k in flat)


class TestDaskIntegration:
    """Dask parallel path must produce the same results as the sequential one."""

    @pytest.mark.skipif(not dask_available, reason="dask not installed")
    def test_dask_matches_sequential(self):
        cpu = _flatten(_run_validation(use_gpu=False))
        dask = _flatten(
            _run_validation(
                use_gpu=False, parallel="dask", n_workers=1,
                parallel_kwargs={"dashboard": False},
            )
        )

        assert set(cpu.keys()) == set(dask.keys())

        for key, cpu_val in cpu.items():
            dask_val = dask[key]
            assert cpu_val.shape == dask_val.shape, f"shape mismatch for {key}"
            nptest.assert_allclose(
                cpu_val, dask_val, rtol=1e-6, atol=1e-8,
                err_msg=f"value mismatch for {key}",
            )
