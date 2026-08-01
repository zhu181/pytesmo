"""
NetCDF writer for final output.

Enhanced netCDF writer that can read from Zarr/Parquet intermediate
formats and produce final netCDF files.
"""

import os
import numpy as np
from typing import Dict, Any, List, Optional, Union
from pathlib import Path


class NetCDFWriter:
    """
    NetCDF writer for final validation results.
    
    Can write directly from results or convert from Zarr/Parquet.
    """
    
    def __init__(self, output_path: str,
                 format: str = 'NETCDF4',
                 compression: str = 'zlib',
                 compression_level: int = 4,
                 chunk_size: int = 1000):
        """
        Initialize NetCDF writer.
        
        Parameters
        ----------
        output_path : str
            Output directory for netCDF files
        format : str, default 'NETCDF4'
            NetCDF format ('NETCDF4', 'NETCDF4_CLASSIC')
        compression : str, default 'zlib'
            Compression algorithm
        compression_level : int, default 4
            Compression level (1-9)
        chunk_size : int, default 1000
            Chunk size for variables
        """
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        self.format = format
        self.compression = compression
        self.compression_level = compression_level
        self.chunk_size = chunk_size
    
    def write_results(self, results: List[Dict[str, Any]], 
                      filename: str = 'validation_results.nc'):
        """
        Write results to a single netCDF file.
        
        Parameters
        ----------
        results : list of dict
            Validation results
        filename : str
            Output filename
        """
        from netCDF4 import Dataset
        
        filepath = self.output_path / filename
        
        with Dataset(filepath, 'w', format=self.format) as ds:
            # Create dimensions
            n_jobs = len(results)
            ds.createDimension('job', n_jobs)
            
            # Collect all metric names
            all_metrics = set()
            for result in results:
                if isinstance(result, dict):
                    all_metrics.update(result.keys())
            
            # Separate metadata and metrics
            meta_keys = {'gpi', 'lon', 'lat', 'status', 'n_obs'}
            metric_keys = all_metrics - meta_keys
            
            # Write metadata variables
            for key in meta_keys:
                if any(key in r for r in results if isinstance(r, dict)):
                    values = []
                    for r in results:
                        if isinstance(r, dict) and key in r:
                            val = r[key]
                            if isinstance(val, np.ndarray):
                                values.append(val.item() if val.size == 1 else val[0])
                            else:
                                values.append(val)
                        else:
                            values.append(0)
                    
                    var = ds.createVariable(
                        key, 'f8' if key in ('lon', 'lat') else 'i4',
                        ('job',),
                        zlib=(self.compression == 'zlib'),
                        complevel=self.compression_level,
                        chunksizes=(min(self.chunk_size, n_jobs),)
                    )
                    var[:] = values
            
            # Write metric variables
            for key in sorted(metric_keys):
                values = []
                for r in results:
                    if isinstance(r, dict) and key in r:
                        val = r[key]
                        if isinstance(val, np.ndarray):
                            if val.size == 1:
                                values.append(val.item())
                            else:
                                # Multi-value metric - store as array
                                values.append(val)
                        else:
                            values.append(val)
                    else:
                        values.append(np.nan)
                
                # Check if all values are scalar
                all_scalar = all(np.isscalar(v) or (isinstance(v, np.ndarray) and v.size == 1) 
                                for v in values)
                
                if all_scalar:
                    scalar_values = [v.item() if isinstance(v, np.ndarray) else v for v in values]
                    var = ds.createVariable(
                        key, 'f8', ('job',),
                        zlib=(self.compression == 'zlib'),
                        complevel=self.compression_level,
                        chunksizes=(min(self.chunk_size, n_jobs),)
                    )
                    var[:] = scalar_values
                else:
                    # Variable-length arrays - use object array or separate handling
                    pass
            
            # Add global attributes
            ds.setncattr('title', 'Pytesmo Validation Results')
            ds.setncattr('n_jobs', n_jobs)
            ds.setncattr('format', self.format)
    
    def write_split(self, results: List[Dict[str, Any]],
                    group_by: str = 'metric_combination'):
        """
        Write results split into multiple files by combination.
        
        Parameters
        ----------
        results : list of dict
            Validation results
        group_by : str
            Grouping strategy ('metric_combination', 'dataset_pair')
        """
        from netCDF4 import Dataset
        from collections import defaultdict
        
        # Group results by combination
        groups = defaultdict(list)
        
        for i, result in enumerate(results):
            if not isinstance(result, dict):
                continue
            
            # Create group key from result keys
            metric_keys = [k for k in result.keys() 
                          if k not in ('gpi', 'lon', 'lat', 'status', 'n_obs')]
            
            if metric_keys:
                # Use first metric as group identifier
                group_key = metric_keys[0].split('_between_')[0] if '_between_' in metric_keys[0] else 'default'
                groups[group_key].append((i, result))
        
        # Write each group
        for group_key, group_results in groups.items():
            indices, group_data = zip(*group_results)
            
            filename = f'{group_key}.nc'
            filepath = self.output_path / filename
            
            with Dataset(filepath, 'w', format=self.format) as ds:
                n_jobs = len(group_data)
                ds.createDimension('job', n_jobs)
                
                # Write metadata
                for key in ('gpi', 'lon', 'lat', 'status', 'n_obs'):
                    values = []
                    for r in group_data:
                        if key in r:
                            val = r[key]
                            if isinstance(val, np.ndarray):
                                values.append(val.item() if val.size == 1 else val[0])
                            else:
                                values.append(val)
                        else:
                            values.append(0)
                    
                    dtype = 'f8' if key in ('lon', 'lat') else 'i4'
                    var = ds.createVariable(
                        key, dtype, ('job',),
                        zlib=(self.compression == 'zlib'),
                        complevel=self.compression_level
                    )
                    var[:] = values
                
                # Write metrics
                all_metrics = set()
                for r in group_data:
                    all_metrics.update(k for k in r.keys() 
                                     if k not in ('gpi', 'lon', 'lat', 'status', 'n_obs'))
                
                for key in sorted(all_metrics):
                    values = []
                    for r in group_data:
                        if key in r:
                            val = r[key]
                            if isinstance(val, np.ndarray):
                                if val.size == 1:
                                    values.append(val.item())
                                else:
                                    values.append(val)
                            else:
                                values.append(val)
                        else:
                            values.append(np.nan)
                    
                    all_scalar = all(np.isscalar(v) or (isinstance(v, np.ndarray) and v.size == 1) 
                                    for v in values)
                    
                    if all_scalar:
                        scalar_values = [v.item() if isinstance(v, np.ndarray) else v for v in values]
                        var = ds.createVariable(
                            key, 'f8', ('job',),
                            zlib=(self.compression == 'zlib'),
                            complevel=self.compression_level
                        )
                        var[:] = scalar_values
    
    @classmethod
    def from_zarr(cls, zarr_path: str, output_path: str, **kwargs):
        """
        Convert Zarr store to netCDF.
        
        Parameters
        ----------
        zarr_path : str
            Path to Zarr store
        output_path : str
            Output directory for netCDF
        **kwargs : dict
            NetCDFWriter options
        """
        import zarr
        
        writer = cls(output_path, **kwargs)
        
        # Read Zarr arrays
        store = str(zarr_path)
        root = zarr.group(store=store)
        
        # Convert each array to netCDF
        results = []
        max_len = 0
        
        # Find maximum length
        for name, arr in root.arrays():
            if arr.shape[0] > max_len:
                max_len = arr.shape[0]
        
        # Reconstruct results
        for i in range(max_len):
            result = {}
            for name, arr in root.arrays():
                if i < arr.shape[0]:
                    val = arr[i]
                    if hasattr(val, 'get'):
                        val = val.get()
                    result[name] = val
            if result:
                results.append(result)
        
        writer.write_results(results)
        return writer
    
    @classmethod
    def from_parquet(cls, parquet_path: str, output_path: str, **kwargs):
        """
        Convert Parquet dataset to netCDF.
        
        Parameters
        ----------
        parquet_path : str
            Path to Parquet dataset (directory)
        output_path : str
            Output directory for netCDF
        **kwargs : dict
            NetCDFWriter options
        """
        import pandas as pd
        
        writer = cls(output_path, **kwargs)
        
        # Read all Parquet files
        parquet_dir = Path(parquet_path)
        dfs = []
        
        for partition_dir in parquet_dir.iterdir():
            if partition_dir.is_dir():
                for pq_file in partition_dir.glob('*.parquet'):
                    df = pd.read_parquet(pq_file)
                    dfs.append(df)
        
        if not dfs:
            return writer
        
        # Combine and pivot
        combined = pd.concat(dfs, ignore_index=True)
        
        # Pivot to wide format
        # This is a simplified conversion; real implementation depends on schema
        results = []
        for job_idx in combined['job_idx'].unique():
            job_data = combined[combined['job_idx'] == job_idx]
            result = {'job_idx': job_idx}
            
            for _, row in job_data.iterrows():
                result[row['metric_name']] = row['value']
            
            results.append(result)
        
        writer.write_results(results)
        return writer


def write_results_netcdf(results: List[Dict[str, Any]], 
                         output_path: str,
                         split: bool = False,
                         **kwargs) -> str:
    """
    Convenience function to write results to netCDF.
    
    Parameters
    ----------
    results : list of dict
        Validation results
    output_path : str
        Output directory
    split : bool, default False
        Split into multiple files by combination
    **kwargs : dict
        NetCDFWriter options
    
    Returns
    -------
    str
        Output path
    """
    writer = NetCDFWriter(output_path, **kwargs)
    
    if split:
        writer.write_split(results)
    else:
        writer.write_results(results)
    
    return output_path