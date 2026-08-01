"""
I/O module for pytesmo validation results.

Provides writers for intermediate (Zarr, Parquet) and final (netCDF) output formats.
"""

from .zarr_writer import ZarrWriter, ZarrBatchWriter, write_results_zarr
from .parquet_writer import ParquetWriter, ParquetBatchWriter, write_results_parquet
from .netcdf_writer import NetCDFWriter, write_results_netcdf

__all__ = [
    'ZarrWriter',
    'ZarrBatchWriter',
    'write_results_zarr',
    'ParquetWriter',
    'ParquetBatchWriter',
    'write_results_parquet',
    'NetCDFWriter',
    'write_results_netcdf',
]