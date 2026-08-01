"""Lazy thread-safe CuPy backend singleton with NumPy fallback.

Attribution note: this module is part of the QA4SM GPU acceleration
contribution, vendored inside ``pytesmo``.
"""
from __future__ import annotations

import logging
import threading
from typing import Any

import numpy as np

_LOGGER = logging.getLogger(__name__)

try:
    import cupy as cp  # type: ignore[import-untyped]

    _CUPY_AVAILABLE = True
except ImportError:
    cp = None  # type: ignore[assignment,misc]
    _CUPY_AVAILABLE = False


class _GPUBackend:
    """Process-wide singleton wrapping a CuPy device with a NumPy fallback.

    The first call to :func:`get_gpu_backend` triggers lazy initialisation
    of the active CuPy device; a reentrant lock serialises any concurrent
    calls from worker threads.
    """

    _instance: "_GPUBackend | None" = None
    _lock = threading.RLock()

    def __init__(self, device_id: int = 0) -> None:
        self.device_id = int(device_id)
        self._available: bool = False
        self._xp: Any = np  # module
        self._device_obj: Any = None
        if not _CUPY_AVAILABLE:
            _LOGGER.debug("CuPy not installed; GPU backend uses NumPy.")
            return
        try:
            self._device_obj = cp.cuda.Device(self.device_id)
            self._device_obj.use()
            cp.arange(3, dtype=cp.float64)  # smoke-test
            self._available = True
            _LOGGER.info(
                "GPU backend ready on device %s (%s).",
                self.device_id,
                self._device_obj.name,
            )
        except Exception as exc:
            self._available = False
            self._device_obj = None
            _LOGGER.warning(
                "GPU backend unavailable on device %s (%s). "
                "Falling back to NumPy. %s: %s",
                self.device_id,
                type(exc).__name__,
                exc,
            )

    @property
    def available(self) -> bool:
        return self._available

    @property
    def xp(self) -> Any:
        return self._xp

    @property
    def torch(self) -> None:
        return None

    @property
    def device(self) -> Any:
        return self._device_obj

    def get_device_info(self) -> dict[str, object]:
        if not self._available:
            return {"name": "numpy-cpu", "memory_total": 0, "memory_free": 0}
        try:
            name = self._device_obj.name  # type: ignore[union-attr]
        except Exception:
            name = f"cuda:{self.device_id}"
        free, total = self._device_obj.mem_info  # type: ignore[union-attr]
        return {"name": name, "memory_total": int(total), "memory_free": int(free)}

    def mem_info(self) -> tuple[int, int]:
        if not self._available:
            return 0, 0
        free, total = self._device_obj.mem_info  # type: ignore[union-attr]
        return int(free), int(total)

    def to_gpu(self, arr: Any) -> Any:
        if isinstance(arr, cp.ndarray):  # type: ignore[possibly-undefined]
            return arr
        return cp.asarray(arr) if _CUPY_AVAILABLE else np.asarray(arr)

    def to_cpu(self, arr: Any) -> np.ndarray:
        if isinstance(arr, np.ndarray):
            return arr
        return cp.asnumpy(arr) if _CUPY_AVAILABLE else np.asarray(arr)

    def to_tensor(self, arr: Any) -> Any:
        return self.to_gpu(arr)

    def synchronize(self) -> None:
        if self._available:
            cp.cuda.Stream.null.synchronize()

    def empty_cache(self) -> None:
        if not self._available:
            return
        try:
            cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass
        try:
            cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass

    def __repr__(self) -> str:
        return (
            f"<_GPUBackend device={self.device_id} "
            f"available={self._available} xp={self._xp.__name__}>"
        )

    @classmethod
    def _get(cls, device_id: int = 0) -> "_GPUBackend":
        with cls._lock:
            if cls._instance is None or cls._instance.device_id != int(device_id):
                cls._instance = cls(device_id=int(device_id))
            return cls._instance


def get_gpu_backend(device_id: int = 0) -> _GPUBackend:
    """Return the process-wide :class:`_GPUBackend` singleton."""
    return _GPUBackend._get(device_id=device_id)
