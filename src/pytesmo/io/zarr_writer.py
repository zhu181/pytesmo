"""
Zarr writer for chunked intermediate output.

Zarr provides efficient chunked, compressed array storage that works well
for intermediate results in parallel processing.
"""

import os
import numpy as np
import zarr
from typing import Dict, Any, List, Optional
from pathlib import Path


class ZarrWriter:
    """
    Zarr-based chunked writer for validation results.
    
    Features:
    - Chunked storage for large datasets
    - Compression (blosc, gzip, lz4)
    - Append-only mode for streaming writes
    - Automatic chunk size optimization
    """
    
    def __init__(self, output_path: str, 
                 compression: str = 'blosc',
                 compression_level: int = 5,
                 chunk_size: int = 10000,
                 mode: str = 'a'):
        """
        Initialize Zarr writer.
        
        Parameters
        ----------
        output_path : str
            Root directory for Zarr store
        compression : str, default 'blosc'
            Compression algorithm ('blosc', 'gzip', 'lz4', 'zstd')
        compression_level : int, default 5
            Compression level (1-9)
        chunk_size : int, default 10000
            Chunk size along first dimension
        mode : str, default 'a'
            File mode ('a' for append, 'w' for overwrite)
        """
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        try:
            from zarr.codecs import BloscCodec
            cname = compression
            if cname == 'blosc':
                cname = 'blosclz'
            self.compressor = BloscCodec(
                cname=cname,
                clevel=compression_level,
                shuffle='shuffle',
            )
        except ImportError:
            self.compressor = None
        self.chunk_size = chunk_size
        self.mode = mode
        
        self._store = None
        self._root = None
        self._arrays = {}
        self._shapes = {}
        self._dtypes = {}
    
    @property
    def root(self):
        if self._root is None:
            self._store = str(self.output_path)
            self._root = zarr.group(store=self._store, overwrite=(self.mode == 'w'))
        return self._root
    
    def create_array(self, name: str, shape: tuple, dtype: np.dtype, 
                     chunks: Optional[tuple] = None):
        """
        Create a new Zarr array.
        
        Parameters
        ----------
        name : str
            Array name
        shape : tuple
            Full array shape
        dtype : np.dtype
            Data type
        chunks : tuple, optional
            Chunk shape
        """
        if chunks is None:
            # Default chunking: chunk_size along first dim, full size for others
            chunks = (min(self.chunk_size, shape[0]),) + shape[1:]

        kwargs = dict(
            shape=shape,
            chunks=chunks,
            dtype=dtype,
            fill_value=np.nan if np.issubdtype(dtype, np.floating) else 0,
            overwrite=True,
        )
        if self.compressor is not None:
            kwargs['compressor'] = self.compressor

        create = getattr(self.root, 'create_array', None) or self.root.create_array
        arr = create(name, **kwargs)
        
        self._arrays[name] = arr
        self._shapes[name] = shape
        self._dtypes[name] = dtype
        
        return arr
    
    def append(self, name: str, data: np.ndarray, 
               index: Optional[int] = None):
        """
        Append data to an existing array.
        
        Parameters
        ----------
        name : str
            Array name
        data : np.ndarray
            Data to append
        index : int, optional
            Starting index (auto-increment if None)
        """
        if name not in self._arrays:
            # Create array with estimated size
            total_size = data.shape[0] * 10  # Guess
            full_shape = (total_size,) + data.shape[1:]
            self.create_array(name, full_shape, data.dtype)
        
        arr = self._arrays[name]
        
        if index is None:
            # Find first NaN/zero row
            # This is a simple approach; in practice, track write position
            index = self._get_next_index(name)

        # Ensure array is large enough
        end_idx = index + data.shape[0]
        if end_idx > arr.shape[0]:
            # Resize array
            new_shape = (max(end_idx, arr.shape[0] * 2),) + arr.shape[1:]
            arr.resize(new_shape)

        arr[index:end_idx] = data

        # Track write position
        self._write_positions[name] = end_idx

        return index
    
    def _get_next_index(self, name: str) -> int:
        """Get next write index for array."""
        if not hasattr(self, '_write_positions'):
            self._write_positions = {}
        
        return self._write_positions.get(name, 0)
    
    def write_batch(self, batch_results: List[Dict[str, Any]], 
                    batch_idx: int):
        """
        Write a batch of results.
        
        Parameters
        ----------
        batch_results : list of dict
            Results from a batch of jobs
        batch_idx : int
            Batch index
        """
        for job_idx, result in enumerate(batch_results):
            if not isinstance(result, dict):
                continue
            
            for key, value in result.items():
                if isinstance(value, np.ndarray):
                    arr_name = f'{key}'
                    data = value.reshape(1, -1) if value.ndim == 1 else value
                elif np.isscalar(value):
                    arr_name = f'{key}'
                    data = np.array([[value]])
                else:
                    # Skip non-array, non-scalar values
                    continue
                
                self.append(arr_name, data)
    
    def consolidate(self):
        """Consolidate metadata for faster reads."""
        if self._store:
            try:
                zarr.consolidate_metadata(self._store)
            except Exception:
                pass

    def close(self):
        """Close the Zarr store."""
        if self._store:
            # Compact arrays to the actual number of written rows
            for name, end_idx in getattr(self, '_write_positions', {}).items():
                arr = self._arrays.get(name)
                if arr is not None and end_idx < arr.shape[0]:
                    new_shape = (end_idx,) + arr.shape[1:]
                    arr.resize(new_shape)
            self.consolidate()
            self._store = None
            self._root = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class ZarrBatchWriter:
    """
    High-level batch writer that organizes results by metric type.
    """
    
    def __init__(self, output_path: str, **kwargs):
        self.writer = ZarrWriter(output_path, **kwargs)
        self._batch_idx = 0
    
    def write(self, results: List[Dict[str, Any]]):
        """Write a batch of results."""
        self.writer.write_batch(results, self._batch_idx)
        self._batch_idx += 1
    
    def close(self):
        self.writer.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def write_results_zarr(results: List[Dict[str, Any]], 
                       output_path: str,
                       **kwargs) -> str:
    """
    Convenience function to write results to Zarr.
    
    Parameters
    ----------
    results : list of dict
        Validation results
    output_path : str
        Output directory
    **kwargs : dict
        ZarrWriter options
    
    Returns
    -------
    str
        Path to Zarr store
    """
    with ZarrBatchWriter(output_path, **kwargs) as writer:
        writer.write(results)
    return output_path