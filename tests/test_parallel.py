# -*- coding: utf-8 -*-
"""
Parallel processing tests (Dask + Sequential executors).
"""

import numpy as np
import pytest

from pytesmo.parallel import (
    SequentialExecutor,
    get_executor,
    parallel_map,
)

dask_available = True
try:
    import dask.distributed  # noqa: F401
    from pytesmo.parallel import DaskParallelExecutor
except ImportError:
    dask_available = False


def _square(x):
    return x * x


def _square_tuple(a, b):
    return a * b


def _failing(x):
    if x == 3:
        raise ValueError("intentional failure")
    return x


class TestSequentialExecutor:
    def test_map_simple(self):
        executor = SequentialExecutor()
        results = executor.map(_square, list(range(10)), progress=False)
        assert results == [i * i for i in range(10)]

    def test_map_tuple_jobs(self):
        executor = SequentialExecutor()
        jobs = [(i, i + 1) for i in range(5)]
        results = executor.map(_square_tuple, jobs, progress=False)
        assert results == [i * (i + 1) for i in range(5)]

    def test_map_empty(self):
        executor = SequentialExecutor()
        assert executor.map(_square, [], progress=False) == []

    def test_map_error_handling(self):
        executor = SequentialExecutor()
        results = executor.map(_failing, [1, 3, 5], progress=False)
        assert results[0] == 1
        assert 'error' in results[1]
        assert results[2] == 5

    def test_close(self):
        executor = SequentialExecutor()
        executor.close()  # should be a no-op


@pytest.mark.skipif(not dask_available, reason="dask[distributed] not installed")
class TestDaskParallelExecutor:
    def test_map_simple(self):
        executor = DaskParallelExecutor(n_workers=2, dashboard=False)
        try:
            results = executor.map(_square, list(range(20)), batch_size=5, progress=False)
            assert sorted(results) == [i * i for i in range(20)]
        finally:
            executor.close()

    def test_map_tuple_jobs(self):
        executor = DaskParallelExecutor(n_workers=2, dashboard=False)
        try:
            jobs = [(i, i + 1) for i in range(10)]
            results = executor.map(_square_tuple, jobs, batch_size=4, progress=False)
            assert sorted(results) == sorted(i * (i + 1) for i in range(10))
        finally:
            executor.close()

    def test_map_empty(self):
        executor = DaskParallelExecutor(n_workers=2, dashboard=False)
        try:
            assert executor.map(_square, [], progress=False) == []
        finally:
            executor.close()

    def test_map_error_handling(self):
        executor = DaskParallelExecutor(n_workers=2, dashboard=False)
        try:
            results = executor.map(_failing, [1, 3, 5], batch_size=2, progress=False)
            assert 1 in results
            assert 5 in results
            assert any(isinstance(r, dict) and 'error' in r for r in results)
        finally:
            executor.close()

    def test_context_manager(self):
        with DaskParallelExecutor(n_workers=2, dashboard=False) as executor:
            results = executor.map(_square, [1, 2, 3], progress=False)
            assert results == [1, 4, 9]

    def test_client_lazy_start(self):
        """The client property should start the cluster lazily."""
        executor = DaskParallelExecutor(n_workers=1, dashboard=False)
        assert executor._client is None
        client = executor.client
        assert client is not None
        executor.close()


class TestExecutors:
    def test_get_executor_sequential(self):
        executor = get_executor(backend='sequential')
        assert isinstance(executor, SequentialExecutor)

    def test_get_executor_invalid(self):
        with pytest.raises(ValueError):
            get_executor(backend='invalid')

    def test_parallel_map_sequential(self):
        results = parallel_map(_square, list(range(8)), backend='sequential', progress=False)
        assert results == [i * i for i in range(8)]

    @pytest.mark.skipif(not dask_available, reason="dask[distributed] not installed")
    def test_parallel_map_auto(self):
        results = parallel_map(_square, list(range(8)), backend='auto', n_workers=1, progress=False)
        assert sorted(results) == sorted(i * i for i in range(8))
