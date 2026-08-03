"""
I/O format round-trip tests for Zarr, Parquet, and netCDF writers.
"""

import numpy as np
import pytest
from pytesmo.io import (
    NetCDFWriter,
    ParquetBatchWriter,
    ZarrBatchWriter,
    write_results_netcdf,
    write_results_parquet,
    write_results_zarr,
)

netcdf_available = True
try:
    import netCDF4  # noqa: F401
except ImportError:
    netcdf_available = False


def _sample_results(n=3):
    return [
        {
            "gpi": i,
            "lon": float(i) + 0.5,
            "lat": float(i) - 0.5,
            "n_obs": 100 + i,
            "bias": float(i),
            "rmsd": np.float64(i + 1),
            "pearson_r": np.array([0.5 + 0.1 * i]),
        }
        for i in range(n)
    ]


@pytest.fixture
def tmp_path_factory():
    import tempfile
    from pathlib import Path

    d = tempfile.mkdtemp(prefix="pytesmo_io_")
    return Path(d)


class TestZarrWriter:
    def test_roundtrip(self, tmp_path_factory):
        out = tmp_path_factory / "zarr"
        results = _sample_results(3)

        with ZarrBatchWriter(str(out)) as writer:
            writer.write(results)

        import zarr

        root = zarr.group(store=str(out))

        # bias column should have 3 rows
        assert "bias" in root
        arr = root["bias"]
        assert arr.shape[0] == 3
        assert list(arr[:, 0]) == [0.0, 1.0, 2.0]

    def test_append_multiple_batches(self, tmp_path_factory):
        out = tmp_path_factory / "zarr_append"
        results = _sample_results(2)

        with ZarrBatchWriter(str(out)) as writer:
            writer.write(results[:1])
            writer.write(results[1:])

        import zarr

        root = zarr.group(store=str(out))
        assert root["rmsd"].shape[0] == 2
        assert list(root["rmsd"][:, 0]) == [1.0, 2.0]

    def test_scalar_and_array(self, tmp_path_factory):
        out = tmp_path_factory / "zarr_types"
        results = [
            {"bias": 1.0, "pearson_r": np.array([0.5, 0.6])},
            {"bias": 2.0, "pearson_r": np.array([0.7, 0.8])},
        ]

        with ZarrBatchWriter(str(out)) as writer:
            writer.write(results)

        import zarr

        root = zarr.group(store=str(out))
        assert root["bias"].shape[0] == 2
        assert root["pearson_r"].shape == (2, 2)
        np.testing.assert_allclose(root["pearson_r"][:], [[0.5, 0.6], [0.7, 0.8]])

    def test_convenience_function(self, tmp_path_factory):
        out = tmp_path_factory / "zarr_fn"
        path = write_results_zarr(_sample_results(2), str(out))
        import zarr

        assert zarr.group(store=str(path)) is not None


class TestParquetWriter:
    def test_roundtrip(self, tmp_path_factory):
        out = tmp_path_factory / "parquet"
        results = _sample_results(3)

        with ParquetBatchWriter(str(out)) as writer:
            writer.write(results)

        import pandas as pd

        files = list(out.rglob("*.parquet"))
        assert len(files) >= 1

        combined = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        metric_names = set(combined["metric_name"])
        assert "bias" in metric_names
        assert "rmsd" in metric_names
        assert "pearson_r" in metric_names

    def test_partition_by_metric_type(self, tmp_path_factory):
        out = tmp_path_factory / "parquet_part"
        results = _sample_results(2)

        with ParquetBatchWriter(str(out)) as writer:
            writer.write(results)

        dirs = [d.name for d in out.iterdir() if d.is_dir()]
        assert "bias" in dirs or "rmsd" in dirs

    def test_convenience_function(self, tmp_path_factory):
        out = tmp_path_factory / "parquet_fn"
        path = write_results_parquet(_sample_results(2), str(out))
        assert len(list(out.rglob("*.parquet"))) >= 1
        assert path == str(out)


@pytest.mark.skipif(not netcdf_available, reason="netCDF4 not installed")
class TestNetCDFWriter:
    def test_write_results(self, tmp_path_factory):
        out = tmp_path_factory / "netcdf"
        results = _sample_results(3)

        writer = NetCDFWriter(str(out))
        writer.write_results(results)

        import netCDF4

        f = netCDF4.Dataset(str(out / "validation_results.nc"))
        assert f.dimensions["job"].size == 3
        np.testing.assert_allclose(f.variables["bias"][:], [0.0, 1.0, 2.0])
        np.testing.assert_allclose(f.variables["lon"][:], [0.5, 1.5, 2.5])
        f.close()

    def test_write_split(self, tmp_path_factory):
        out = tmp_path_factory / "netcdf_split"
        results = _sample_results(3)

        writer = NetCDFWriter(str(out))
        writer.write_split(results)

        import netCDF4

        ncs = list(out.glob("*.nc"))
        assert len(ncs) >= 1
        for nc in ncs:
            f = netCDF4.Dataset(str(nc))
            assert "bias" in f.variables
            f.close()

    def test_convenience_function(self, tmp_path_factory):
        out = tmp_path_factory / "netcdf_fn"
        write_results_netcdf(_sample_results(2), str(out))
        assert (out / "validation_results.nc").exists()

    def test_roundtrip_from_zarr(self, tmp_path_factory):
        zarr_path = tmp_path_factory / "zarr_src"
        results = _sample_results(2)
        with ZarrBatchWriter(str(zarr_path)) as writer:
            writer.write(results)

        out = tmp_path_factory / "netcdf_from_zarr"
        NetCDFWriter.from_zarr(str(zarr_path), str(out))

        import netCDF4

        f = netCDF4.Dataset(str(out / "validation_results.nc"))
        np.testing.assert_allclose(f.variables["bias"][:], [0.0, 1.0])
        f.close()

    def test_roundtrip_from_parquet(self, tmp_path_factory):
        pq_path = tmp_path_factory / "parquet_src"
        results = _sample_results(2)
        with ParquetBatchWriter(str(pq_path)) as writer:
            writer.write(results)

        out = tmp_path_factory / "netcdf_from_pq"
        NetCDFWriter.from_parquet(str(pq_path), str(out))

        import netCDF4

        f = netCDF4.Dataset(str(out / "validation_results.nc"))
        assert f.dimensions["job"].size == 2
        np.testing.assert_allclose(f.variables["bias"][:], [0.0, 1.0])
        f.close()


class TestClientBatchStore:
    """Client-side streaming batch store (save/load used by map_batches_streaming)."""

    def test_roundtrip(self, tmp_path_factory):
        from pytesmo.parallel import _load_batch_zarr, _save_batch_zarr

        bdir = tmp_path_factory / "batch_000000.zarr"
        combos = [
            (("dataset_a",), ("dataset_b",)),
            (("ds_1", "ds_2"), ("ds_3",)),
        ]
        results = [
            {c: [{"bias": np.float64(0.1), "rmsd": np.array([1.0])}] for c in combos},
            {combos[0]: [{"bias": np.float64(0.2), "rmsd": np.array([2.0])}]},
            {"error": "boom"},
        ]

        _save_batch_zarr(bdir, results)
        assert (bdir / ".complete").exists()
        assert (bdir / "combos.json").exists()

        loaded = _load_batch_zarr(bdir)
        assert len(loaded) == 3
        # windows round-trip; the error gpi is positionally absent for both combos
        assert loaded[0][combos[0]][0]["bias"][0] == pytest.approx(0.1)
        assert loaded[0][combos[1]][0]["bias"][0] == pytest.approx(0.1)
        assert loaded[1][combos[0]][0]["rmsd"][0] == pytest.approx(2.0)
        assert combos[0] not in loaded[2]

    def test_roundtrip_multi_window(self, tmp_path_factory):
        from pytesmo.parallel import _load_batch_zarr, _save_batch_zarr

        bdir = tmp_path_factory / "batch_000000.zarr"
        combo = (("ds_a",), ("ds_b",))
        results = [
            {combo: [{"bias": np.float64(1.0)}, {"bias": np.float64(2.0)}]},
            {combo: [{"bias": np.float64(3.0)}]},
            {combo: [{"bias": np.float64(4.0)}, {"bias": np.float64(5.0)}, {"bias": np.float64(6.0)}]},
        ]

        _save_batch_zarr(bdir, results)
        loaded = _load_batch_zarr(bdir)
        assert [w["bias"][0] for w in loaded[0][combo]] == [1.0, 2.0]
        assert [w["bias"][0] for w in loaded[1][combo]] == [3.0]
        assert [w["bias"][0] for w in loaded[2][combo]] == [4.0, 5.0, 6.0]

    def test_sanitize_combo_unique(self):
        from pytesmo.parallel import _sanitize_combo

        combo = (("dataset_a",), ("dataset_b",))
        assert _sanitize_combo(combo) == "dataset_a_with_dataset_b"
        # deterministic and a valid path component
        assert _sanitize_combo(combo) == _sanitize_combo(combo)
        assert _sanitize_combo(combo) == _sanitize_combo(tuple(tuple(ds) for ds in combo))
        assert "/" not in _sanitize_combo(combo) and "\\" not in _sanitize_combo(combo)
