"""
Dask-based parallel execution for validation framework.

Provides a modern replacement for IPython parallel with:
- LocalCluster for multi-process parallelism
- Progress bars via tqdm
- Batch processing for GPU efficiency
- Zarr/Parquet intermediate storage
- Error handling and retries
"""

import os
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ParallelExecutor:
    """Base class for parallel executors."""
    
    def __init__(self, n_workers: int = -1, **kwargs):
        self.n_workers = n_workers
        self.kwargs = kwargs
    
    def map(self, func: Callable, jobs: List[Tuple], 
            batch_size: int = 100, progress: bool = True) -> List[Any]:
        raise NotImplementedError
    
    def close(self):
        pass


class SequentialExecutor(ParallelExecutor):
    """Sequential executor (no parallelism)."""
    
    def map(self, func: Callable, jobs: List[Tuple], 
            batch_size: int = 100, progress: bool = True) -> List[Any]:
        from tqdm import tqdm
        
        results = []
        iterator = tqdm(jobs, desc="Processing", disable=not progress)
        for job in iterator:
            try:
                result = func(*job) if isinstance(job, tuple) else func(job)
                results.append(result)
            except Exception as e:
                logger.error(f"Job {job} failed: {e}")
                results.append({'error': str(e), 'job': job})
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
    
    def __init__(self, n_workers: int = -1, 
                 scheduler_port: int = 0,
                 dashboard: bool = True,
                 dashboard_port: int = 8787,
                 processes: bool = True,
                 threads_per_worker: int = 1,
                 memory_limit: str = 'auto',
                 local_directory: Optional[str] = None,
                 **kwargs):
        super().__init__(n_workers, **kwargs)
        
        self.scheduler_port = scheduler_port
        self.dashboard = dashboard
        self.dashboard_port = dashboard_port
        self.processes = processes
        self.threads_per_worker = threads_per_worker
        self.memory_limit = memory_limit
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
        from dask.distributed import LocalCluster, Client
        
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
                    f"Requested {n_workers} workers but only {n_gpus} GPUs available. "
                    f"Limiting to {n_gpus} workers."
                )
                n_workers = n_gpus
        except ImportError:
            pass  # No GPU, use CPU workers
        
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
    
    def map(self, func: Callable, jobs: List[Tuple], 
            batch_size: int = 100, progress: bool = True,
            retries: int = 2,
            output_format: str = 'netcdf',
            output_path: Optional[str] = None,
            **kwargs) -> List[Any]:
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
        from dask.distributed import as_completed, wait
        from tqdm import tqdm
        
        if not jobs:
            return []
        
        # Initialize worker GPU contexts
        self._init_worker_gpu()
        
        # Create batches
        batches = [jobs[i:i + batch_size] for i in range(0, len(jobs), batch_size)]
        
        # Submit batch jobs
        def _process_batch(batch):
            results = []
            for job in batch:
                try:
                    result = func(*job) if isinstance(job, tuple) else func(job)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Job {job} failed: {e}")
                    results.append({'error': str(e), 'job': job})
            return results
        
        # Submit all batches
        futures = self._client.map(_process_batch, batches, retries=retries)
        
        # Collect results with progress bar
        results = []
        with tqdm(total=len(jobs), desc="Processing", disable=not progress) as pbar:
            for future in as_completed(futures):
                batch_result = future.result()
                results.extend(batch_result)
                pbar.update(len(batch_result))
        
        # Flatten results
        flat_results = []
        for batch_result in results:
            if isinstance(batch_result, list):
                flat_results.extend(batch_result)
            else:
                flat_results.append(batch_result)
        
        return flat_results
    
    def map_with_intermediate_output(self, func: Callable, jobs: List[Tuple],
                                      batch_size: int = 100,
                                      progress: bool = True,
                                      output_format: str = 'zarr',
                                      output_path: Optional[str] = None,
                                      **kwargs) -> List[Any]:
        """
        Execute jobs with intermediate Zarr/Parquet output.
        
        This writes partial results to disk after each batch to avoid
        memory issues with large job counts.
        """
        from dask.distributed import as_completed
        from tqdm import tqdm
        
        if output_path is None:
            output_path = os.path.join(os.getcwd(), 'validation_output')
        
        Path(output_path).mkdir(parents=True, exist_ok=True)
        
        # Initialize worker GPU contexts
        self._init_worker_gpu()
        
        # Create batches
        batches = [jobs[i:i + batch_size] for i in range(0, len(jobs), batch_size)]
        
        def _process_batch_with_output(batch, batch_idx):
            results = []
            for job in batch:
                try:
                    result = func(*job) if isinstance(job, tuple) else func(job)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Job {job} failed: {e}")
                    results.append({'error': str(e), 'job': job})
            
            # Write intermediate output
            if output_format == 'zarr':
                self._write_zarr_batch(results, output_path, batch_idx)
            elif output_format == 'parquet':
                self._write_parquet_batch(results, output_path, batch_idx)
            
            return results
        
        # Submit all batches
        futures = self._client.map(_process_batch_with_output, batches, 
                                   range(len(batches)), retries=2)
        
        # Collect results
        results = []
        with tqdm(total=len(jobs), desc="Processing", disable=not progress) as pbar:
            for future in as_completed(futures):
                batch_result = future.result()
                results.extend(batch_result)
                pbar.update(len(batch_result))
        
        return results
    
    def _write_zarr_batch(self, results: List[Dict], output_path: str, batch_idx: int):
        """Write batch results to Zarr format."""
        import zarr
        import numpy as np
        
        batch_dir = Path(output_path) / f'batch_{batch_idx:06d}.zarr'
        batch_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert results to arrays
        for i, result in enumerate(results):
            if isinstance(result, dict):
                for key, value in result.items():
                    if isinstance(value, np.ndarray):
                        arr = value
                    else:
                        arr = np.array([value])
                    
                    zarr.save_array(str(batch_dir / f'{key}_{i}.zarr'), arr)
    
    def _write_parquet_batch(self, results: List[Dict], output_path: str, batch_idx: int):
        """Write batch results to Parquet format."""
        import pandas as pd
        
        batch_dir = Path(output_path) / f'batch_{batch_idx:06d}.parquet'
        
        # Flatten results to DataFrame
        rows = []
        for i, result in enumerate(results):
            if isinstance(result, dict):
                row = {'job_idx': i}
                for key, value in result.items():
                    if isinstance(value, np.ndarray):
                        row[key] = value.tolist() if value.size > 1 else value.item()
                    else:
                        row[key] = value
                rows.append(row)
        
        if rows:
            df = pd.DataFrame(rows)
            df.to_parquet(batch_dir, index=False)
    
    def close(self):
        """Shutdown cluster."""
        if self._client:
            self._client.close()
            self._client = None
        if self._cluster:
            self._cluster.close()
            self._cluster = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def get_executor(backend: str = 'dask', **kwargs) -> ParallelExecutor:
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
    if backend == 'sequential':
        return SequentialExecutor(**kwargs)
    elif backend == 'dask':
        return DaskParallelExecutor(**kwargs)
    elif backend == 'auto':
        try:
            import dask
            return DaskParallelExecutor(**kwargs)
        except ImportError:
            logger.info("Dask not available, using sequential executor")
            return SequentialExecutor(**kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}")


# Convenience function for simple parallel execution
def parallel_map(func: Callable, jobs: List[Tuple], 
                 n_workers: int = -1, batch_size: int = 100,
                 progress: bool = True, backend: str = 'auto') -> List[Any]:
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