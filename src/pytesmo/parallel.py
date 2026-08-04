"""
Dask-based parallel execution for validation framework.

Provides a modern replacement for IPython parallel with:
- LocalCluster for multi-process parallelism
- Progress bars via tqdm
- Batch processing for GPU efficiency
- Zarr/Parquet intermediate storage
- Error handling and retries
"""

import json
import logging
import tempfile
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _make_batch_processor(func: Callable):
    """Return a Dask-serializable closure that runs one batch of jobs."""

    def _process_batch(batch):
        results = []
        for job in batch:
            try:
                result = func(*job) if isinstance(job, tuple) else func(job)
                results.append(result)
            except Exception as e:
                logger.error(f"Job {job} failed: {e}")
                results.append({"error": str(e), "job": job})
        # Release GPU memory pool blocks back to CUDA and break any Python
        # reference cycles accumulated from the complex Validation / reader /
        # adapter object graph deserialized per-batch.  Without this, the
        # CuPy pool grows to the high-water mark and never shrinks, and
        # reference cycles delay GC of large intermediate objects — both
        # appear as "unmanaged memory" to the Dask worker monitor.
        import gc
        import time

        try:
            from pytesmo.gpu.backend import free_memory as _gpu_free

            _gpu_free()
        except Exception:
            pass
        # Multiple GC passes with short pauses between them:
        # pass 1 breaks reference cycles, pass 2 collects newly-unreachable
        # objects revealed by pass 1, pass 3 is a confirmation no-op.
        # The sleep gives the OS a chance to reclaim freed pages.
        for _ in range(3):
            gc.collect()
            time.sleep(0.01)
        return results

    return _process_batch


def _sanitize_combo(combo) -> str:
    return "_with_".join(".".join(str(ds) for ds in key) for key in combo)


def _safe_array_name(name: str) -> str:
    """Map an arbitrary string to a zarr group/array name safe for all OSes."""
    import re

    safe = re.sub(r"[^\w.\-]", "_", name)
    safe = safe.strip("_")
    if not safe:
        safe = f"metric_{abs(hash(name)) % (10**8)}"
    return safe


def _batch_dir(output_path: str, batch_idx: int) -> Path:
    return Path(output_path) / f"batch_{batch_idx:06d}.zarr"


def _batch_complete(batch_dir: Path) -> bool:
    return (batch_dir / ".complete").exists()


def _save_batch_zarr(batch_dir: Path, results: list[dict]) -> None:
    """Persist one batch of per-gpi result dicts to a zarr store.

    ``results`` is a list of per-gpi dicts mapping a dataset-combination key to a
    *list of per-window metric dicts* (``{metric: scalar | np.array(1,)}``), the
    same structure ``Validation.calc``'s job processor produces. Windows are
    flattened across the batch (mirroring ``_compact_batch``) and stored
    columnar: one array per metric plus a ``_gpi`` index array mapping each
    window row back to its gpi. A ``.complete`` sentinel is written last so a
    crashed/interrupted batch is never mistaken for a finished one.
    """
    import zarr

    batch_dir.mkdir(parents=True, exist_ok=True)

    combo_order = []
    for r in results:
        if not isinstance(r, dict) or "error" in r:
            continue
        for combo in r:
            if combo not in [c[0] for c in combo_order]:
                combo_order.append((combo, [list(ds) for ds in combo]))

    with open(batch_dir / "combos.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "n_gpis": len(results),
                "combos": [[c, ds_list, _sanitize_combo(c)] for c, ds_list in combo_order],
            },
            f,
        )

    for combo, _ in combo_order:
        window_rows = []  # (gpi_index, metric dict)
        for gpi_idx, r in enumerate(results):
            if not isinstance(r, dict) or "error" in r or combo not in r:
                continue
            for wd in r[combo]:
                if isinstance(wd, dict):
                    window_rows.append((gpi_idx, wd))
        if not window_rows:
            continue

        group = zarr.group(store=str(batch_dir / _sanitize_combo(combo)), overwrite=True)
        n_rows = len(window_rows)
        gpi_arr = group.create_array("_gpi", shape=(n_rows,), chunks=(n_rows,), dtype="int64")
        gpi_arr[:] = [g for g, _ in window_rows]

        metrics = sorted({m for _, wd in window_rows for m in wd})
        metric_map = {}  # sanitized -> original
        for metric in metrics:
            col = [_scalarize(wd.get(metric)) for _, wd in window_rows]
            if not any(c is not None for c in col):
                continue
            dtype = None
            for c in col:
                if c is not None:
                    dtype = np.asarray(c).dtype
                    break
            if any(c is None for c in col):
                dtype = np.float64
                col = [np.nan if c is None else c for c in col]
            safe_name = _safe_array_name(metric)
            metric_map[safe_name] = metric
            try:
                arr = group.create_array(safe_name, shape=(n_rows,), chunks=(n_rows,), dtype=dtype)
                arr[:] = col
            except Exception:
                # Any failure (OSError, TypeError, ValueError, etc.): skip this
                # metric rather than failing the whole batch.
                try:
                    del group[safe_name]
                except Exception:
                    pass

        # Write metric name mapping so load can restore original names
        with open(batch_dir / _sanitize_combo(combo) / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metric_map, f)

    (batch_dir / ".complete").touch()


def _scalarize(v) -> Any:
    """Reduce a per-window metric value to a single scalar (or None if empty)."""
    if v is None:
        return None
    arr = np.asarray(v)
    if arr.ndim == 0:
        return v
    if arr.size == 0:
        return None
    return arr.reshape(-1)[0]


def _load_batch_zarr(batch_dir: Path) -> list[dict]:
    """Reconstruct one batch of per-gpi result dicts from a zarr store."""
    import zarr

    with open(batch_dir / "combos.json", encoding="utf-8") as f:
        meta = json.load(f)
    n_gpis = meta.get("n_gpis", 0)
    combos = [(tuple(tuple(ds) for ds in combo), sanitized) for combo, ds_list, sanitized in meta["combos"]]

    results = [{} for _ in range(n_gpis)]
    for combo, sanitized in combos:
        group = zarr.group(store=str(batch_dir / sanitized))
        if "_gpi" not in group.array_keys():
            continue
        gpi_idx = np.asarray(group["_gpi"][:])
        metric_names = [name for name in group.array_keys() if name != "_gpi"]

        # Restore original metric names from metrics.json if present
        metric_map_path = batch_dir / sanitized / "metrics.json"
        metric_map = {}
        if metric_map_path.exists():
            with open(metric_map_path, encoding="utf-8") as f:
                metric_map = json.load(f)

        windows = [{} for _ in range(len(gpi_idx))]
        for safe_name in metric_names:
            col = np.asarray(group[safe_name][:])
            orig_name = metric_map.get(safe_name, safe_name)
            for j in range(len(col)):
                windows[j][orig_name] = np.array([col[j]])
        for j, g in enumerate(gpi_idx):
            results[int(g)].setdefault(combo, []).append(windows[j])
    return results


class ParallelExecutor:
    """Base class for parallel executors."""

    def __init__(self, n_workers: int = -1, **kwargs):
        self.n_workers = n_workers
        self.kwargs = kwargs

    def map(self, func: Callable, jobs: list[tuple], batch_size: int = 100, progress: bool = True) -> list[Any]:
        raise NotImplementedError

    def close(self):
        pass


class SequentialExecutor(ParallelExecutor):
    """Sequential executor (no parallelism)."""

    def map(self, func: Callable, jobs: list[tuple], batch_size: int = 100, progress: bool = True) -> list[Any]:
        from tqdm import tqdm

        results = []
        iterator = tqdm(jobs, desc="Processing", disable=not progress)
        for job in iterator:
            try:
                result = func(*job) if isinstance(job, tuple) else func(job)
                results.append(result)
            except Exception as e:
                logger.error(f"Job {job} failed: {e}")
                results.append({"error": str(e), "job": job})
        return results


class DaskParallelExecutor(ParallelExecutor):
    """
    Dask-based parallel executor with LocalCluster.

    Features:
    - Multi-process workers (one GPU per process)
    - Progress bars with tqdm
    - Automatic batching
    - Error handling with retries
    - Intermediate Zarr/Parquet output
    """

    def __init__(
        self,
        n_workers: int = -1,
        scheduler_port: int = 0,
        dashboard: bool = True,
        dashboard_port: int = 8787,
        processes: bool = True,
        threads_per_worker: int = 1,
        memory_limit: str = "auto",
        memory_target_fraction: float = 0.7,
        memory_spill_fraction: float = 0.7,
        memory_pause_fraction: float = 0.9,
        memory_terminate_fraction: float = 0.95,
        local_directory: str | None = None,
        **kwargs,
    ):
        super().__init__(n_workers, **kwargs)

        self.scheduler_port = scheduler_port
        self.dashboard = dashboard
        self.dashboard_port = dashboard_port
        self.processes = processes
        self.threads_per_worker = threads_per_worker
        self.memory_limit = memory_limit
        self.memory_target_fraction = memory_target_fraction
        self.memory_spill_fraction = memory_spill_fraction
        self.memory_pause_fraction = memory_pause_fraction
        self.memory_terminate_fraction = memory_terminate_fraction
        self.local_directory = local_directory

        self._cluster = None
        self._client = None
        self._worker_gpu_contexts = {}

    @property
    def client(self):
        if self._client is None:
            self._start_cluster()
        return self._client

    def _start_cluster(self):
        """Start Dask LocalCluster."""
        import dask
        from dask.distributed import Client, LocalCluster

        # Determine number of workers
        if self.n_workers == -1:
            import os

            n_workers = os.cpu_count() or 1
        else:
            n_workers = self.n_workers

        # Limit to available GPUs if using GPU
        try:
            import cupy as cp

            n_gpus = cp.cuda.runtime.getDeviceCount()
            if n_gpus > 0 and n_workers > n_gpus:
                logger.warning(
                    f"Requested {n_workers} workers but only {n_gpus} GPUs available. Limiting to {n_gpus} workers."
                )
                n_workers = n_gpus
        except ImportError:
            pass  # No GPU, use CPU workers

        # Worker memory fractions are no longer accepted as LocalCluster kwargs
        # in modern distributed; set them via dask.config, which propagates to
        # the scheduler/worker subprocesses at startup.
        memory_config = {
            "distributed.worker.memory.target": self.memory_target_fraction,
            "distributed.worker.memory.spill": self.memory_spill_fraction,
            "distributed.worker.memory.pause": self.memory_pause_fraction,
            "distributed.worker.memory.terminate": self.memory_terminate_fraction,
        }
        with dask.config.set(memory_config):
            self._cluster = LocalCluster(
                n_workers=n_workers,
                processes=self.processes,
                threads_per_worker=self.threads_per_worker,
                memory_limit=self.memory_limit,
                scheduler_port=self.scheduler_port,
                dashboard=self.dashboard,
                dashboard_address=f":{self.dashboard_port}" if self.dashboard else None,
                local_directory=self.local_directory,
                silence_logs=logging.WARNING,
            )

        self._client = Client(self._cluster)
        logger.info(f"Dask cluster started: {self._cluster.dashboard_link}")
        logger.info(f"Workers: {len(self._client.scheduler_info()['workers'])}")

    def _init_worker_gpu(self):
        """Initialize GPU context on each worker."""

        def _init():
            from pytesmo.gpu import GPUContext

            ctx = GPUContext()
            return ctx.get_device_info()

        # Run on all workers
        futures = self.client.run(_init)
        for worker, info in futures.items():
            logger.info(f"Worker {worker} GPU: {info}")

    def map(
        self,
        func: Callable,
        jobs: list[tuple],
        batch_size: int = 100,
        progress: bool = True,
        progress_callback: Callable[[int, int], None] | None = None,
        retries: int = 2,
        output_format: str = "netcdf",
        output_path: str | None = None,
        **kwargs,
    ) -> list[Any]:
        """
        Execute jobs in parallel with batching and progress tracking.

        Parameters
        ----------
        func : callable
            Function to execute per job (or batch of jobs)
        jobs : list of tuples
            Job arguments
        batch_size : int, default 100
            Number of jobs per batch
        progress : bool, default True
            Show progress bar
        progress_callback : callable, optional
            Callable ``(done, total)`` invoked after each completed batch in the
            client process. Useful for logging progress into a structured log.
        retries : int, default 2
            Number of retries on failure
        output_format : str, default 'netcdf'
            Output format for intermediate results
        output_path : str, optional
            Path for intermediate output

        Returns
        -------
        results : list
            Results from all jobs
        """
        from dask.distributed import as_completed
        from tqdm import tqdm

        if not jobs:
            return []

        # Initialize worker GPU contexts
        self._init_worker_gpu()

        # Create batches
        batches = [jobs[i : i + batch_size] for i in range(0, len(jobs), batch_size)]

        # Submit all batches
        processor = _make_batch_processor(func)
        futures = self._client.map(processor, batches, retries=retries)
        futures_to_batches = dict(zip(futures, batches))

        # Collect results with progress bar
        results = []
        done = 0
        with tqdm(total=len(jobs), desc="Processing", disable=not progress) as pbar:
            for future in as_completed(futures_to_batches):
                try:
                    batch_result = future.result()
                except Exception as e:
                    batch = futures_to_batches[future]
                    logger.error(f"Batch of {len(batch)} jobs failed permanently after {retries} retries: {e}")
                    batch_result = [{"error": str(e), "job": j} for j in batch]
                results.extend(batch_result)
                done += len(batch_result)
                pbar.update(len(batch_result))
                if progress_callback is not None:
                    progress_callback(done, len(jobs))

        # Flatten results
        flat_results = []
        for batch_result in results:
            if isinstance(batch_result, list):
                flat_results.extend(batch_result)
            else:
                flat_results.append(batch_result)

        return flat_results

    def stream_batches(
        self,
        func: Callable,
        jobs: list[tuple],
        batch_size: int = 100,
        progress: bool = True,
        progress_callback: Callable[[int, int], None] | None = None,
        retries: int = 2,
    ) -> Iterator[list[Any]]:
        """
        Execute jobs in batches and yield each batch's results as it completes.

        The client only ever holds one batch of results in memory at a time,
        unlike ``map()`` which accumulates everything. ``progress_callback``
        (``(done, total)``) is invoked after each batch.
        """
        from dask.distributed import as_completed
        from tqdm import tqdm

        if not jobs:
            return

        self._init_worker_gpu()
        batches = [jobs[i : i + batch_size] for i in range(0, len(jobs), batch_size)]
        processor = _make_batch_processor(func)
        futures = self._client.map(processor, batches, retries=retries)
        futures_to_batches = dict(zip(futures, batches))

        done = 0
        with tqdm(total=len(jobs), desc="Processing", disable=not progress) as pbar:
            for future in as_completed(futures_to_batches):
                try:
                    batch_result = future.result()
                except Exception as e:
                    batch = futures_to_batches[future]
                    logger.error(f"Batch of {len(batch)} jobs failed permanently after {retries} retries: {e}")
                    batch_result = [{"error": str(e), "job": j} for j in batch]
                yield batch_result
                done += len(batch_result)
                pbar.update(len(batch_result))
                if progress_callback is not None:
                    progress_callback(done, len(jobs))

    def map_batches_streaming(
        self,
        func: Callable,
        jobs: list[tuple],
        batch_size: int = 100,
        progress: bool = True,
        progress_callback: Callable[[int, int], None] | None = None,
        output_format: str = "zarr",
        output_path: str | None = None,
        resume: bool = True,
        retries: int = 2,
    ) -> Iterator[list[Any]]:
        """
        Stream batches with optional crash-resume via a client-side zarr store.

        Each batch's results are yielded as they complete and persisted to
        ``output_path/batch_<i>.zarr`` (with a ``.complete`` sentinel written
        last). On a later call with the same ``output_path`` and ``resume=True``,
        already-completed batches are loaded from disk instead of being
        recomputed. The client only holds one batch at a time.
        """
        from dask.distributed import as_completed
        from tqdm import tqdm

        if output_path is None:
            output_path = tempfile.mkdtemp(prefix="pytesmo_batches_")
            resume = False
        Path(output_path).mkdir(parents=True, exist_ok=True)

        if not jobs:
            if progress_callback is not None:
                progress_callback(0, 0)
            return

        self._init_worker_gpu()
        batches = [jobs[i : i + batch_size] for i in range(0, len(jobs), batch_size)]

        done_indices = set()
        if resume and output_format == "zarr":
            for i in range(len(batches)):
                if _batch_complete(_batch_dir(output_path, i)):
                    done_indices.add(i)

        resumed = 0
        for i in sorted(done_indices):
            bdir = _batch_dir(output_path, i)
            try:
                yield _load_batch_zarr(bdir)
                resumed += len(batches[i])
            except Exception as e:
                logger.warning(f"Could not load cached batch {i} ({e}); recomputing.")
                done_indices.discard(i)

        pending = [i for i in range(len(batches)) if i not in done_indices]
        if not pending:
            if progress_callback is not None:
                progress_callback(len(jobs), len(jobs))
            return

        processor = _make_batch_processor(func)
        pending_batches = [batches[i] for i in pending]
        futures = self._client.map(processor, pending_batches, retries=retries)
        futures_to_idx = dict(zip(futures, pending))

        done = resumed
        with tqdm(total=len(jobs), desc="Processing", disable=not progress) as pbar:
            pbar.update(resumed)
            for future in as_completed(futures_to_idx):
                i = futures_to_idx[future]
                try:
                    batch_result = future.result()
                except Exception as e:
                    batch = batches[i]
                    logger.error(f"Batch {i} (n={len(batch)}) failed permanently after {retries} retries: {e}")
                    batch_result = [{"error": str(e), "job": j} for j in batch]
                if output_format == "zarr":
                    try:
                        _save_batch_zarr(_batch_dir(output_path, i), batch_result)
                    except Exception as e:
                        logger.warning(f"Could not persist batch {i} to zarr: {e}")
                yield batch_result
                done += len(batch_result)
                pbar.update(len(batch_result))
                if progress_callback is not None:
                    progress_callback(done, len(jobs))

    def map_with_intermediate_output(
        self,
        func: Callable,
        jobs: list[tuple],
        batch_size: int = 100,
        progress: bool = True,
        progress_callback: Callable[[int, int], None] | None = None,
        output_format: str = "zarr",
        output_path: str | None = None,
        resume: bool = True,
        retries: int = 2,
        **kwargs,
    ) -> list[Any]:
        """
        Execute jobs with intermediate Zarr output.

        Writes partial results to disk after each batch (client-side) and
        returns the flattened list of all results. With ``resume=True`` and a
        stable ``output_path``, completed batches are loaded from disk instead
        of being recomputed on a re-run.
        """
        if output_path is None:
            output_path = tempfile.mkdtemp(prefix="pytesmo_batches_")
            resume = False
        Path(output_path).mkdir(parents=True, exist_ok=True)

        results = []
        for batch_result in self.map_batches_streaming(
            func,
            jobs,
            batch_size=batch_size,
            progress=progress,
            progress_callback=progress_callback,
            output_format=output_format,
            output_path=output_path,
            resume=resume,
            retries=retries,
        ):
            results.extend(batch_result)

        flat_results = []
        for batch_result in results:
            if isinstance(batch_result, list):
                flat_results.extend(batch_result)
            else:
                flat_results.append(batch_result)
        return flat_results

    def close(self):
        """Shutdown cluster.

        Teardown errors (e.g. a worker that is slow to terminate) are logged
        but not raised: by the time ``close()`` is called the computation has
        already finished, so a shutdown failure must not be mistaken for a
        compute failure by callers.
        """
        if self._client:
            try:
                self._client.close()
            except Exception as e:
                logger.warning(f"Error closing distributed client during shutdown: {e}")
            self._client = None
        if self._cluster:
            try:
                self._cluster.close()
            except Exception as e:
                logger.warning(f"Error closing cluster during shutdown: {e}")
            self._cluster = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def get_executor(backend: str = "dask", **kwargs) -> ParallelExecutor:
    """
    Factory function to get parallel executor.

    Parameters
    ----------
    backend : str
        'dask', 'sequential', or 'auto'
    **kwargs : dict
        Backend-specific arguments

    Returns
    -------
    ParallelExecutor
    """
    if backend == "sequential":
        return SequentialExecutor(**kwargs)
    elif backend == "dask":
        return DaskParallelExecutor(**kwargs)
    elif backend == "auto":
        import importlib.util

        if importlib.util.find_spec("dask") is not None:
            return DaskParallelExecutor(**kwargs)
        else:
            logger.info("Dask not available, using sequential executor")
            return SequentialExecutor(**kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}")


# Convenience function for simple parallel execution
def parallel_map(
    func: Callable,
    jobs: list[tuple],
    n_workers: int = -1,
    batch_size: int = 100,
    progress: bool = True,
    backend: str = "auto",
) -> list[Any]:
    """
    Simple parallel map function.

    Parameters
    ----------
    func : callable
        Function to apply
    jobs : list
        List of job arguments
    n_workers : int
        Number of workers
    batch_size : int
        Batch size
    progress : bool
        Show progress bar
    backend : str
        Backend to use

    Returns
    -------
    results : list
    """
    executor = get_executor(backend=backend, n_workers=n_workers)
    try:
        return executor.map(func, jobs, batch_size=batch_size, progress=progress)
    finally:
        executor.close()
