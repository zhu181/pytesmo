GPU Acceleration & Parallel Processing
======================================

pytesmo supports GPU-accelerated metric computation via CuPy and parallel
processing via Dask. Both features are optional and are enabled per-call on
the :class:`pytesmo.validation_framework.validation.Validation` workflow
(and the underlying metric functions).

Installation
------------

Install the optional dependencies with:

.. code-block:: bash

    pip install pytesmo[gpu]

This installs:

- ``cupy`` (CUDA-aware NumPy replacement) for GPU-accelerated metrics
- ``dask[distributed]`` for parallel job processing
- ``zarr`` and ``fastparquet`` for intermediate chunked/columnar output
- ``tqdm`` for progress bars

.. note::

    CuPy is only required if you want GPU acceleration. If CuPy is not
    installed, pytesmo transparently falls back to NumPy and all code
    continues to work on the CPU.

GPU Acceleration
----------------

The following metric functions are GPU-accelerated when CuPy is available:

- Pairwise metrics: ``bias``, ``mse_decomposition``, ``rmsd``, ``ubrmsd``,
  ``pearson_r``, ``spearman_r``, ``kendall_tau`` and rolling variants
- Triple collocation: ``tcol_metrics`` (SNR, error standard deviation,
  scaling factor ``beta``)
- Confidence intervals: analytical and bootstrapped (percentile, basic, BCa)

Enable GPU acceleration in the validation workflow with ``use_gpu=True``:

.. code-block:: python

    from pytesmo.validation_framework.validation import Validation
    from pytesmo.validation_framework.metric_calculators import (
        PairwiseIntercomparisonMetrics,
    )

    process = Validation(
        datasets=datasets,
        spatial_ref='ISMN',
        temporal_ref='ASCAT',
        metrics_calculators={
            (2, 2): PairwiseIntercomparisonMetrics().calc_metrics
        },
        period=period,
    )

    # Sequential with GPU
    results = process.calc(gpis, lons, lats, use_gpu=True)

The GPU module is also usable directly:

.. code-block:: python

    from pytesmo.metrics import pearson_r
    from pytesmo.metrics import tcol_metrics

    # Automatic dispatch: uses GPU if cupy is available, else NumPy
    r, p = pearson_r(x, y)
    snr, err_std, beta = tcol_metrics(x, y, z)

Parallel Processing
-------------------

The validation workflow can be parallelized across grid points with Dask's
``LocalCluster`` (multi-process, one GPU per worker):

.. code-block:: python

    results = process.calc(
        gpis, lons, lats,
        use_gpu=True,
        parallel='dask',          # or 'sequential' / None
        n_workers=4,              # -1 uses all CPUs (limited to GPU count)
        batch_size=1000,
        output_format='zarr',     # 'netcdf', 'zarr' or 'parquet'
        output_path='/tmp/validation_output',
        progress=True,
    )

Options
~~~~~~~

- ``parallel``: ``None`` (sequential), ``'sequential'`` or ``'dask'``.
- ``n_workers``: number of Dask workers. Default ``-1`` auto-selects the
  number of CPUs, capped by the number of available GPUs when
  ``use_gpu=True``.
- ``batch_size``: number of grid points processed per Dask batch.
- ``output_format``: intermediate output format. ``'zarr'`` and ``'parquet'``
  write partial results to disk after each batch to bound memory usage;
  ``'netcdf'`` writes a final netCDF file.
- ``output_path``: directory for intermediate/final output.
- ``progress``: show a ``tqdm`` progress bar.

I/O Layer
---------

pytesmo provides chunked writers for intermediate and final results:

- :mod:`pytesmo.io.zarr_writer` - chunked, compressed Zarr storage (zarr 3)
- :mod:`pytesmo.io.parquet_writer` - columnar Parquet storage, partitioned by
  metric type
- :mod:`pytesmo.io.netcdf_writer` - final netCDF4 output, including
  conversion from Zarr/Parquet intermediate stores

These are used internally by the parallel executor and can also be used
directly:

.. code-block:: python

    from pytesmo.io import write_results_zarr, write_results_parquet

    results = [...]   # list of result dicts from Validation.calc()

    write_results_zarr(results, "/tmp/zarr_out")
    write_results_parquet(results, "/tmp/pq_out")

Requirements
------------

- CUDA-compatible GPU for GPU acceleration. pytesmo is tested against CUDA 12.x
  (``cupy-cuda12x``) and CUDA 13.x (``cupy-cuda13x``).
- Dask with ``distributed`` for parallel processing (CPU-only workers work
  too).

Both features fall back gracefully: without CuPy, metrics run on NumPy;
without Dask, jobs run sequentially.
