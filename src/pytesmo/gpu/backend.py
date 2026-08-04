"""
GPU Context singleton for CuPy backend management.

Provides a thread-safe singleton that manages:
- CuPy/NumPy module selection
- CUDA memory pool
- Non-blocking CUDA stream
- Device/host array transfers
"""

import threading
import warnings
from typing import Any, Optional, Union


class GPUContext:
    """
    Singleton GPU context manager.
    
    Automatically detects CuPy availability and configures:
    - Memory pool for efficient allocations
    - Non-blocking CUDA stream for async operations
    - float64 as default precision
    
    Falls back to NumPy if CuPy is not available.
    """
    
    _instance: Optional['GPUContext'] = None
    _lock = threading.Lock()
    
    def __new__(cls) -> 'GPUContext':
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._init()
            self._initialized = True
    
    def _init(self):
        """Initialize GPU context with CuPy or NumPy fallback."""
        self._cp = None
        self._memory_pool = None
        self._stream = None
        self._available = False
        self._device_id = 0
        
        try:
            import cupy as cp
            
            # Verify CUDA 12+ compatibility
            cuda_version = cp.cuda.runtime.runtimeGetVersion()
            if cuda_version < 12000:
                warnings.warn(
                    f"CUDA version {cuda_version/1000:.1f} detected. "
                    "CuPy CUDA 12.x recommended for best performance.",
                    UserWarning
                )
            
            # Configure memory pool — store the actual pool instance so
            # free_memory() frees the allocator that is actively used, not
            # the default pool which is a separate object.
            self._memory_pool = cp.cuda.MemoryPool()
            cp.cuda.set_allocator(self._memory_pool.malloc)
            
            # Create non-blocking stream for async operations
            self._stream = cp.cuda.Stream(non_blocking=True)
            
            self._cp = cp
            self._available = True
            
        except ImportError:
            import numpy as cp
            self._cp = cp
            self._memory_pool = None
            self._stream = None
            self._available = False
        except Exception as e:
            warnings.warn(
                f"Failed to initialize CuPy: {e}. Falling back to NumPy.",
                UserWarning
            )
            import numpy as cp
            self._cp = cp
            self._memory_pool = None
            self._stream = None
            self._available = False
    
    @property
    def cp(self):
        """Return the array module (CuPy or NumPy)."""
        return self._cp
    
    @property
    def available(self) -> bool:
        """Return True if GPU acceleration is available."""
        return self._available
    
    @property
    def memory_pool(self):
        """Return the memory pool (CuPy only)."""
        return self._memory_pool
    
    @property
    def stream(self):
        """Return the CUDA stream (CuPy only)."""
        return self._stream
    
    @property
    def device_id(self) -> int:
        """Return the current GPU device ID."""
        return self._device_id
    
    def set_device(self, device_id: int):
        """Set the active GPU device."""
        if self._available:
            self._cp.cuda.Device(device_id).use()
        self._device_id = device_id
    
    def to_device(self, arr: Any) -> Any:
        """Transfer array to GPU device."""
        if self._available:
            return self._cp.asarray(arr)
        return arr
    
    def to_host(self, arr: Any) -> Any:
        """Transfer array to host (CPU)."""
        if self._available and hasattr(arr, 'get'):
            return arr.get()
        return self._cp.asarray(arr) if hasattr(self._cp, 'asarray') else arr
    
    def synchronize(self):
        """Synchronize the CUDA stream."""
        if self._available and self._stream is not None:
            self._stream.synchronize()
    
    def free_memory(self):
        """Free unused memory in the pool."""
        if self._available and self._memory_pool is not None:
            self._memory_pool.free_all_blocks()
    
    def get_device_info(self) -> dict:
        """Get GPU device information."""
        if not self._available:
            return {'available': False}
        
        dev = self._cp.cuda.Device(self._device_id)
        attrs = dev.attributes
        
        # Get device name from runtime
        try:
            props = self._cp.cuda.runtime.getDeviceProperties(self._device_id)
            name = props['name']
            if isinstance(name, bytes):
                name = name.decode()
            total_mem = props.get('totalGlobalMem', 0)
            major = props.get('major', 0)
            minor = props.get('minor', 0)
            multiprocessors = props.get('multiProcessorCount', 0)
        except:
            name = f"CUDA Device {self._device_id}"
            total_mem = 0
            major = minor = multiprocessors = 0
        
        cc = f"{major}.{minor}" if major or minor else "unknown"
        
        return {
            'available': True,
            'device_id': self._device_id,
            'name': name,
            'compute_capability': cc,
            'total_memory_gb': total_mem / 1e9,
            'multiprocessors': multiprocessors,
            'cuda_version': self._cp.cuda.runtime.runtimeGetVersion() / 1000,
        }


# Global singleton instance
_gpu_context = None


def get_gpu_context() -> GPUContext:
    """Get the global GPU context singleton."""
    global _gpu_context
    if _gpu_context is None:
        _gpu_context = GPUContext()
    return _gpu_context


def get_gpu_module():
    """Get the array module (CuPy if available, else NumPy)."""
    return get_gpu_context().cp


def is_gpu_available() -> bool:
    """Check if GPU acceleration is available."""
    return get_gpu_context().available


def to_device(arr: Any) -> Any:
    """Transfer array to GPU device."""
    return get_gpu_context().to_device(arr)


def to_host(arr: Any) -> Any:
    """Transfer array to host (CPU)."""
    return get_gpu_context().to_host(arr)


def synchronize():
    """Synchronize GPU operations."""
    get_gpu_context().synchronize()


def free_memory():
    """Free unused GPU memory."""
    get_gpu_context().free_memory()


def get_device_info() -> dict:
    """Get GPU device information."""
    return get_gpu_context().get_device_info()


# For backward compatibility
GPU_AVAILABLE = property(lambda self: is_gpu_available())