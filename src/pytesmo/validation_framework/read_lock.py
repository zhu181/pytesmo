"""HDF5 / netCDF4 read-serialisation lock.

libhdf5 and netCDF-C are not thread-safe in the PyPI wheels. All pytesmo
netCDF4 reads **must** hold this lock to avoid 0xC0000005 access
violations (Windows) and silent memory corruption elsewhere.

The lock is reentrant so that the same worker thread can trigger
recursive reads (e.g. upscaling Lut lookups) without deadlocking.
See ``DataManager.get_data`` / ``get_other_data`` for usage.
"""
import threading

HDF5_READ_LOCK = threading.RLock()
