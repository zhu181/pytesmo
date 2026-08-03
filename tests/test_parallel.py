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


def _explode(x):
    raise RuntimeError("must never run on resume")


def _square_dict(x):
    return {(("ds_a",), ("ds_b",)): [{"bias": np.float64(x * x), "n": np.float64(1.0)}]}


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
        assert "error" in results[1]
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
            assert any(isinstance(r, dict) and "error" in r for r in results)
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

    def test_progress_callback(self):
        """The progress callback should report monotonically increasing counts."""
        executor = DaskParallelExecutor(n_workers=2, dashboard=False)
        try:
            calls = []
            executor.map(
                _square,
                list(range(12)),
                batch_size=4,
                progress=False,
                progress_callback=lambda done, total: calls.append((done, total)),
            )
            assert calls, "progress callback was never invoked"
            assert calls[-1][0] == 12
            assert all(total == 12 for _, total in calls)
            dones = [done for done, _ in calls]
            assert dones == sorted(dones)
        finally:
            executor.close()

    def test_stream_batches_yield(self, tmp_path):
        """stream_batches should yield every batch and invoke the callback."""
        executor = DaskParallelExecutor(n_workers=2, dashboard=False)
        try:
            calls = []
            got = []
            for batch in executor.stream_batches(
                _square,
                list(range(10)),
                batch_size=3,
                progress=False,
                progress_callback=lambda done, total: calls.append((done, total)),
            ):
                got.extend(batch)
            assert sorted(got) == [i * i for i in range(10)]
            assert calls[-1][0] == 10
            assert all(total == 10 for _, total in calls)
        finally:
            executor.close()

    def test_map_batches_streaming_resume(self, tmp_path):
        """Completed zarr batches must be reused, not recomputed, on resume."""
        out = tmp_path / "batch_store"
        jobs = list(range(12))
        first = DaskParallelExecutor(n_workers=2, dashboard=False)
        try:
            results = first.map_batches_streaming(
                _square_dict,
                jobs,
                batch_size=4,
                progress=False,
                output_format="zarr",
                output_path=str(out),
            )
            got = [v for b in results for v in b]
            assert len(got) == 12
            bias = [v[(("ds_a",), ("ds_b",))][0]["bias"] for v in got]
            assert sorted(bias) == [float(i * i) for i in range(12)]
        finally:
            first.close()

        batch_dirs = sorted(p.name for p in out.glob("batch_*.zarr"))
        assert batch_dirs == ["batch_000000.zarr", "batch_000001.zarr", "batch_000002.zarr"]
        assert all((out / name / ".complete").exists() for name in batch_dirs)

        # A second executor with an exploding func and the same output_path must
        # reconstruct everything from the cache without ever invoking the func.
        second = DaskParallelExecutor(n_workers=2, dashboard=False)
        try:
            results = second.map_batches_streaming(
                _explode,
                jobs,
                batch_size=4,
                progress=False,
                output_format="zarr",
                output_path=str(out),
            )
            got = [v for b in results for v in b]
            assert len(got) == 12
            bias = [v[(("ds_a",), ("ds_b",))][0]["bias"] for v in got]
            assert sorted(bias) == [float(i * i) for i in range(12)]
        finally:
            second.close()

    def test_map_with_intermediate_output_resume(self, tmp_path):
        """map_with_intermediate_output should return identical results on resume."""
        out = tmp_path / "intermediate_store"
        jobs = list(range(12))

        def run(func):
            with DaskParallelExecutor(n_workers=2, dashboard=False) as executor:
                return executor.map_with_intermediate_output(
                    func,
                    jobs,
                    batch_size=5,
                    progress=False,
                    output_format="zarr",
                    output_path=str(out),
                )

        first = run(_square_dict)
        second = run(_explode)
        assert len(first) == len(second) == 12
        key = (("ds_a",), ("ds_b",))
        assert sorted(v[key][0]["bias"] for v in first) == sorted(v[key][0]["bias"] for v in second)
        assert sorted(v[key][0]["bias"] for v in first) == [float(i * i) for i in range(12)]

    def test_memory_fraction_forwarding(self):
        """Worker memory fractions should reach the worker processes via config."""
        executor = DaskParallelExecutor(
            n_workers=1,
            dashboard=False,
            memory_target_fraction=0.5,
            memory_spill_fraction=0.45,
            memory_pause_fraction=0.8,
            memory_terminate_fraction=0.9,
        )
        try:
            client = executor.client

            def _target():
                import dask

                return dask.config.get("distributed.worker.memory.target")

            def _spill():
                import dask

                return dask.config.get("distributed.worker.memory.spill")

            target = client.run(_target)
            spill = client.run(_spill)
            assert all(abs(v - 0.5) < 1e-9 for v in target.values())
            assert all(abs(v - 0.45) < 1e-9 for v in spill.values())
        finally:
            executor.close()


class TestExecutors:
    def test_get_executor_sequential(self):
        executor = get_executor(backend="sequential")
        assert isinstance(executor, SequentialExecutor)

    def test_get_executor_invalid(self):
        with pytest.raises(ValueError):
            get_executor(backend="invalid")

    def test_parallel_map_sequential(self):
        results = parallel_map(_square, list(range(8)), backend="sequential", progress=False)
        assert results == [i * i for i in range(8)]

    @pytest.mark.skipif(not dask_available, reason="dask[distributed] not installed")
    def test_parallel_map_auto(self):
        results = parallel_map(_square, list(range(8)), backend="auto", n_workers=1, progress=False)
        assert sorted(results) == sorted(i * i for i in range(8))
