"""
Parquet writer for columnar intermediate output.

Parquet provides efficient columnar storage with good compression,
ideal for tabular validation results.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from pathlib import Path


class ParquetWriter:
    """
    Parquet-based writer for validation results.
    
    Features:
    - Columnar storage for efficient queries
    - Partitioning by metric type
    - Compression (snappy, gzip, zstd)
    - Append mode with batching
    """
    
    def __init__(self, output_path: str,
                 compression: str = 'snappy',
                 partition_cols: Optional[List[str]] = None,
                 batch_size: int = 10000,
                 mode: str = 'append'):
        """
        Initialize Parquet writer.
        
        Parameters
        ----------
        output_path : str
            Root directory for Parquet dataset
        compression : str, default 'snappy'
            Compression algorithm ('snappy', 'gzip', 'zstd', 'lz4')
        partition_cols : list of str, optional
            Columns to partition by
        batch_size : int, default 10000
            Batch size for writing
        mode : str, default 'append'
            Write mode
        """
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        self.compression = compression
        self.partition_cols = partition_cols or ['metric_type']
        self.batch_size = batch_size
        self.mode = mode
        
        self._buffers = {}  # metric_type -> list of rows
        self._schemas = {}  # metric_type -> set of columns
    
    def _get_metric_type(self, key: str) -> str:
        """Extract metric type from key."""
        # Split by common separators
        for sep in ['_between_', '_ci_', '_']:
            if sep in key:
                return key.split(sep)[0]
        return key
    
    def _flatten_result(self, result: Dict[str, Any], 
                        job_idx: int) -> List[Dict[str, Any]]:
        """
        Flatten a result dict into rows for Parquet.
        
        Each metric becomes a row with columns:
        - job_idx
        - metric_name
        - metric_type
        - value (or array as list)
        """
        rows = []
        
        for key, value in result.items():
            if key in ('gpi', 'lon', 'lat', 'status', 'n_obs'):
                # Metadata columns
                continue
            
            metric_type = self._get_metric_type(key)
            
            if isinstance(value, np.ndarray):
                if value.size == 1:
                    val = value.item()
                else:
                    val = value.tolist()
            elif np.isscalar(value):
                val = value
            else:
                val = str(value)
            
            row = {
                'job_idx': job_idx,
                'metric_name': key,
                'metric_type': metric_type,
                'value': val,
            }
            
            # Add metadata if available
            for meta_key in ('gpi', 'lon', 'lat', 'status', 'n_obs'):
                if meta_key in result:
                    row[meta_key] = result[meta_key]
            
            rows.append(row)
        
        return rows
    
    def append(self, result: Dict[str, Any], job_idx: int):
        """
        Add a result to the buffer.
        
        Parameters
        ----------
        result : dict
            Single job result
        job_idx : int
            Job index
        """
        rows = self._flatten_result(result, job_idx)
        
        for row in rows:
            metric_type = row['metric_type']
            if metric_type not in self._buffers:
                self._buffers[metric_type] = []
                self._schemas[metric_type] = set()
            
            self._buffers[metric_type].append(row)
            self._schemas[metric_type].update(row.keys())
            
            # Flush if buffer is full
            if len(self._buffers[metric_type]) >= self.batch_size:
                self._flush_metric_type(metric_type)
    
    def _flush_metric_type(self, metric_type: str):
        """Flush buffered rows for a metric type to Parquet."""
        if metric_type not in self._buffers or not self._buffers[metric_type]:
            return
        
        df = pd.DataFrame(self._buffers[metric_type])
        
        # Ensure consistent column order
        cols = sorted(self._schemas[metric_type])
        df = df.reindex(columns=cols)
        
        # Write to partitioned Parquet
        partition_path = self.output_path / metric_type
        partition_path.mkdir(parents=True, exist_ok=True)
        
        # Generate filename with timestamp for uniqueness
        import time
        filename = f'part_{int(time.time() * 1000000)}.parquet'
        filepath = partition_path / filename
        
        df.to_parquet(
            filepath,
            compression=self.compression,
            index=False
        )
        
        # Clear buffer
        self._buffers[metric_type].clear()
    
    def flush_all(self):
        """Flush all buffered data."""
        for metric_type in list(self._buffers.keys()):
            self._flush_metric_type(metric_type)
    
    def write_batch(self, batch_results: List[Dict[str, Any]], 
                    batch_idx: int):
        """
        Write a batch of results.
        
        Parameters
        ----------
        batch_results : list of dict
            Results from a batch of jobs
        batch_idx : int
            Batch index (used for job_idx offset)
        """
        for job_idx, result in enumerate(batch_results):
            if isinstance(result, dict):
                self.append(result, batch_idx * 10000 + job_idx)
    
    def close(self):
        """Close writer and flush remaining data."""
        self.flush_all()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class ParquetBatchWriter:
    """
    High-level batch writer for validation results.
    """
    
    def __init__(self, output_path: str, **kwargs):
        self.writer = ParquetWriter(output_path, **kwargs)
    
    def write(self, results: List[Dict[str, Any]]):
        """Write a batch of results."""
        for i, result in enumerate(results):
            if isinstance(result, dict):
                self.writer.append(result, i)
    
    def close(self):
        self.writer.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def write_results_parquet(results: List[Dict[str, Any]], 
                          output_path: str,
                          **kwargs) -> str:
    """
    Convenience function to write results to Parquet.
    
    Parameters
    ----------
    results : list of dict
        Validation results
    output_path : str
        Output directory
    **kwargs : dict
        ParquetWriter options
    
    Returns
    -------
    str
        Path to Parquet dataset
    """
    with ParquetBatchWriter(output_path, **kwargs) as writer:
        writer.write(results)
    return output_path