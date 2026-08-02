# AGENTS.md - Pytesmo GPU Acceleration & Parallel Support Implementation

## Project Overview
Implementing CuPy-based GPU acceleration and Dask-based parallel processing for pytesmo validation framework.

## Implementation Status

### ✅ Completed

#### 1. GPU Module Structure (`src/pytesmo/gpu/`)
- **`__init__.py`** - Public API exports
- **`backend.py`** - GPUContext singleton with:
  - Thread-safe singleton pattern
  - CuPy/NumPy auto-detection
  - CUDA 12.x support with memory pool
  - Non-blocking CUDA stream
  - float64 precision
  - Device management
- **`array_ops.py`** - Unified array operations (CuPy/NumPy compatible)
- **`pairwise.py`** - GPU-accelerated pairwise metrics:
  - Vectorized Welford's algorithm for batched moments
  - bias, mse_decomposition, rmsd, ubrmsd
  - pearson_r, spearman_r, kendall_tau
  - rolling_pr_rmsd (adapted Welford)
  - Analytical confidence intervals
- **`tcol.py`** - GPU-accelerated triple collocation:
  - Batched covariance computation
  - tcol_metrics (SNR, err_std, beta)
  - ecol (extended collocation) - CPU fallback
- **`bootstrap.py`** - GPU-accelerated bootstrapping:
  - Vectorized percentile/basic/BCa methods
  - with_bootstrapped_ci (pairwise)
  - tcol_metrics_with_bootstrapped_ci (triple collocation)
  - Batched processing with chunking
- **`rolling.py`** - Rolling window metrics (re-export)

#### 2. Parallel Processing (`src/pytesmo/parallel.py`)
- **`ParallelExecutor`** - Base class
- **`SequentialExecutor`** - Fallback for no Dask
- **`DaskParallelExecutor`** - Dask LocalCluster with:
  - Multi-process workers (1 GPU per process)
  - Progress bars via tqdm
  - Batch processing
  - Retry logic
  - Intermediate Zarr/Parquet output
  - Dashboard support
- **`get_executor()`** - Factory function
- **`parallel_map()`** - Convenience function

#### 3. I/O Layer (`src/pytesmo/io/`)
- **`zarr_writer.py`** - Chunked Zarr storage
- **`parquet_writer.py`** - Columnar Parquet storage
- **`netcdf_writer.py`** - Final netCDF output with conversion from Zarr/Parquet

#### 4. Integration Points
- **`validation.py`** - Modified `Validation.calc()` with:
  - `use_gpu`, `parallel`, `n_workers` parameters
  - `batch_size`, `output_format`, `output_path`, `progress`
  - DaskParallelExecutor integration
  - Sequential fallback with tqdm
- **`metrics/pairwise.py`** - Auto GPU/CPU dispatch
- **`metrics/tcol.py`** - Auto GPU/CPU dispatch
- **`metrics/confidence_intervals.py`** - Auto GPU/CPU dispatch
- **`metric_calculators.py`** - GPU-aware metric calculators:
  - PairwiseIntercomparisonMetrics
  - TripleCollocationMetrics
- **`start_validation.py`** - Deprecated with redirect

#### 5. Configuration
- **`setup.py`** - GPU extras: `pip install pytesmo[gpu]`
  - cupy-cuda12x>=13.0
  - dask[distributed]>=2024.0
  - zarr>=2.16
  - fastparquet>=2024.0
  - tqdm>=4.66
- **`__init__.py`** - Exports gpu, parallel, io modules

### 🧪 Tested & Working (2026-08-01)

- **Full test suite passes**: 247 passed (7 notebook tests deselected — pre-existing Windows GBK encoding issue, unrelated to GPU work).
- Verified end-to-end on RTX 5070 Laptop GPU (compute capability 12.0):
  - `tcol_metrics_with_bootstrapped_ci` (TCA bootstrap) — all 3 methods
  - Pairwise GPU bootstrap via `PairwiseIntercomparisonMetrics(bootstrap_cis=True)`
  - TripleCollocationMetrics (with and without bootstrap CIs)
  - `DaskParallelExecutor` with real LocalCluster (auto-limits workers to GPU count)
- **Key fixes applied during debugging:**
  - `_welford_batch`/`_welford_single` rewritten from per-element Python loops (each a GPU kernel launch — 1000-sample bootstrap took >180s) to fully vectorized reductions (now ~1.7s).
  - `mse_decomposition` handles zero-variance inputs (constant data): `r = 0/0` → NaN was poisoning RMSD; now uses `where(denom==0, 1, denom)`.
  - GPU dispatch in `metrics/pairwise.py` is re-applied at the END of the module — later `def` statements were overriding the `_select_impl` assignments.
  - Public API (`metrics/pairwise.py`, `metrics/tcol.py`, `metrics/confidence_intervals.py`) converts CuPy output back to NumPy/Python scalars; the GPU `pearson_r`/`spearman_r`/`kendall_tau` return `(value, p)` but public API must return scalar — wrapped via `_gpu_scalar` + `functools.wraps` (preserves `__name__` for CI lookup).
  - `_bca_ci` converts intermediate scalars to NumPy before `scipy.stats` calls (CuPy arrays can't implicitly convert).
  - `gpu/bootstrap.py::_bca_ci` converts `alpha_lower`/`alpha_upper` to the active module (`xp.asarray`) before `nanquantile` (NumPy percentile arrays triggered "Implicit conversion" TypeError), and squeezes the singleton quantile dim for 1D input. `tcol_metrics_with_bootstrapped_ci` now builds `lower_beta`/`upper_beta` with consistent `xp.asarray(lb, dtype=xp.float64)` entries (was mixing numpy `1.0` with CuPy arrays).
  - `gpu/bootstrap.py::_call_metric` falls back to NumPy inputs for CPU-only metric functions.
  - `_gpu_rmsd_wrap`/`_gpu_ubrmsd_wrap` preserve the deprecated `ddof` kwarg + DeprecationWarning.
  - `metric_calculators.py` TripleCollocationMetrics squeezes the `(1,3)` batch dim from GPU `tcol_metrics`.
  - `parallel.py::_init_worker_gpu` uses the `client` property (lazy cluster start).
  - `validation.py::calc::_process_job` now accepts the `(gpi, lon, lat)` tuple either as a single argument (sequential path) or splatted into separate arguments (Dask executor path, which calls `func(*job)`); previously the Dask path failed every job with "takes 1 positional argument but 3 were given".
  - `validation.py::calc` result merge now skips executor-level error dicts (`{'error': ..., 'job': ...}`) instead of crashing with "can only concatenate list (not str) to list".
- **I/O layer bug fixes** (found via new round-trip tests):
  - zarr 3.x API: `zarr.Blosc` → `zarr.codecs.BloscCodec`; `DirectoryStore` → plain path string; `create_dataset` → `create_array`; `consolidate_metadata` wrapped in try/except.
  - netCDF4: `createVariable` uses `zlib=` + `complevel=` (not `compression`/`compression_level`); missing integer metadata values filled with `0` (not NaN) to avoid "cannot convert float NaN to integer".
  - `zarr_writer.py::_get_next_index` fixed `self._chunk_size` → `self.chunk_size`; write position now tracks actual rows written (was advancing by `chunk_size` per append, leaving sparse holes); `close()` compacts arrays to actual written rows *before* `consolidate_metadata`.
  - `netcdf_writer.py::from_zarr` uses `arr.shape[0]` instead of `len(arr)` (zarr 3 Array has no `len()`).

### ✅ Test Suite (2026-08-01)

- **Full suite passes**: 296 passed (7 notebook tests deselected — pre-existing Windows GBK encoding issue, unrelated to GPU work).
- **`tests/test_gpu.py`** (16 tests) — GPU/CPU numerical equivalence: tcol metrics (incl. beta/SNR/err), tcol bootstrap (percentile/basic/BCa), pairwise metrics, pairwise bootstrap. Uses `_numpy(arr)` helper that `.get()`s CuPy arrays and `.squeeze()`s to align `(1,3)` GPU batch dims with CPU `(3,)`.
- **`tests/test_parallel.py`** (15 tests) — `SequentialExecutor` + `DaskParallelExecutor` (real LocalCluster, `dashboard=False` for test hygiene): simple/tuple/empty/error-handling maps, context manager, lazy `client` start, `get_executor`, `parallel_map`.
- **`tests/test_io_formats.py`** (12 tests) — Zarr/Parquet/netCDF round-trips, batch append, partition by metric type, Zarr→netCDF and Parquet→netCDF conversion.
- **`tests/test_gpu_validation_integration.py`** (4 tests) — full `Validation` workflow (DataManager → temporal matcher → `PairwiseIntercomparisonMetrics`) comparing GPU vs CPU, incl. bootstrap CIs; plus `TestDaskIntegration::test_dask_matches_sequential` verifying the Dask parallel path (`parallel="dask"`, `parallel_kwargs={"dashboard": False}`) yields results identical to the sequential path.
- **`tests/test_gpu_benchmarks.py`** (3 tests) — sanity-check GPU not pathologically slower than CPU (bootstrap, `_welford_batch` vs serial loop, tcol).

### 🔄 In Progress / To Do

#### Tests
- [x] Unit tests for GPU numerical equivalence (CPU vs GPU)
- [x] Benchmark tests for speedup validation
- [x] Integration tests for full validation workflow
- [x] Parallel processing tests with Dask
- [x] I/O format round-trip tests

#### Documentation
- [ ] Update examples with GPU/parallel usage
- [ ] API documentation for new modules
- [ ] Migration guide from IPython parallel

### ⚠️ Environment Notes (this machine)
- Windows + MSYS2/UCRT64 + Python 3.14 venv at `.venv`; use `uv run python` / `uv pip install`.
- `distutils` is absent on 3.14 — `uv pip install setuptools` provides the shim (required by `validation.py`).
- CuPy conflict risk: `cupy-cuda12x` FAILED to install its files (only dist-info) on this system; `cupy-cuda13x==14.1.1` works. Do NOT install both — they share the `cupy` namespace directory and uninstalling one removes the other's files (reinstall with `--reinstall` afterwards).
- Numba requires numpy ≤ 2.4 — installing cupy can bump numpy to 2.5; pin `numpy==2.4.6` after.
- Notebook tests (`test_docs/test_examples.py`) fail on GBK decoding — pre-existing, deselect them.

### Usage Example

```python
from pytesmo.validation_framework.validation import Validation
from pytesmo.validation_framework.metric_calculators import PairwiseIntercomparisonMetrics

# GPU-accelerated validation
process = Validation(
    datasets=datasets,
    spatial_ref='ISMN',
    temporal_ref='ASCAT',
    metrics_calculators={(2, 2): metrics.calc_metrics},
    period=period
)

# Sequential with GPU
results = process.calc(gpis, lons, lats, use_gpu=True)

# Parallel with Dask + GPU
results = process.calc(
    gpis, lons, lats,
    use_gpu=True,
    parallel='dask',
    n_workers=4,
    batch_size=1000,
    output_format='zarr',
    output_path='/tmp/validation_output',
    progress=True
)
```

### GPU Requirements
- CUDA 12.x compatible GPU
- `pip install pytesmo[gpu]` installs cupy-cuda12x
- Falls back to NumPy if CuPy unavailable

### Parallel Requirements
- Dask with distributed
- LocalCluster for multi-process (1 GPU per process)
- Optional: existing Dask scheduler for cluster deployment