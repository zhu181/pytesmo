"""GPU-accelerated batched pytesmo Validation.

Subclasses pytesmo Validation, reusing data-management / scaling /
temporal-matching machinery and overriding calc() to batch metric
computation across grid points through GPU (CuPy) or NumPy.

Attribution note: part of the QA4SM GPU acceleration contribution,
vendored inside pytesmo.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from pytesmo.gpu_backend import get_gpu_backend
from pytesmo.validation_framework.data_manager import (
    DataManager,
    get_result_combinations,
    get_result_names,
)
from pytesmo.validation_framework.data_scalers import DefaultScaler
from pytesmo.validation_framework.metric_calculators import (
    PairwiseIntercomparisonMetrics,
    TripleCollocationMetrics,
)
from pytesmo.validation_framework.read_lock import HDF5_READ_LOCK
from pytesmo.validation_framework.temporal_matchers import (
    make_combined_temporal_matcher,
)
from pytesmo.validation_framework.validation import Validation
import pytesmo.validation_framework.error_handling as eh
from pytesmo.validation_framework.gpu_metrics import (
    BatchedPairwiseMetrics,
    BatchedTCAMetrics,
    STATUS_INSUFFICIENT,
    STATUS_METRICS_FAILED,
    STATUS_OK,
    _fill_meta,
    _pairwise_template,
)

_LOGGER = logging.getLogger(__name__)

try:
    from pytesmo.validation_framework.metric_calculators_adapters import (
        SubsetsMetricsAdapter,
    )
    from validator.adapters import StabilityMetricsAdapter
except ImportError:
    try:
        from pytesmo.validation_framework.metric_calculators_adapters import (
            SubsetsMetricsAdapter,
        )
    except ImportError:
        SubsetsMetricsAdapter = None  # type: ignore[misc,assignment]
    StabilityMetricsAdapter = None  # type: ignore[misc,assignment]


def _detect_calc_type(calc_obj):
    if StabilityMetricsAdapter is not None and isinstance(
        calc_obj, StabilityMetricsAdapter
    ):
        return _detect_calc_type(calc_obj.cls)[0], calc_obj.cls, calc_obj
    if SubsetsMetricsAdapter is not None and isinstance(
        calc_obj, SubsetsMetricsAdapter
    ):
        return _detect_calc_type(calc_obj.cls)[0], calc_obj.cls, calc_obj
    if isinstance(calc_obj, PairwiseIntercomparisonMetrics):
        return "pairwise", calc_obj, None
    if isinstance(calc_obj, TripleCollocationMetrics):
        return "tcol", calc_obj, None
    return "unknown", calc_obj, None


class _RKSpec:
    __slots__ = ("columns", "spec")

    def __init__(self, spec, columns):
        self.spec = spec
        self.columns = list(columns)


class _CalcSpec:
    __slots__ = (
        "kind", "n", "k", "inner", "adapter", "batched",
        "_pairwise_tmpl", "_tcol_tmpl_cache",
    )

    def __init__(self, kind, n, k, inner, adapter=None):
        self.kind = kind
        self.n = n
        self.k = k
        self.inner = inner
        self.adapter = adapter
        self._pairwise_tmpl = None
        self._tcol_tmpl_cache = {}
        if kind == "pairwise":
            self.batched = BatchedPairwiseMetrics(
                metadata_template=getattr(inner, "metadata_template", None),
                calc_spearman=getattr(inner, "calc_spearman", True),
                calc_kendall=getattr(inner, "calc_kendall", False),
                min_obs=getattr(inner, "min_obs", 10),
                analytical_cis=getattr(inner, "analytical_cis", True),
            )
        elif kind == "tcol":
            self.batched = BatchedTCAMetrics(
                refname=getattr(inner, "refname", ""),
                min_obs=getattr(inner, "min_obs", 10),
                metadata_template=getattr(inner, "metadata_template", None),
                bootstrap_cis=getattr(inner, "bootstrap_cis", False),
            )
        else:
            raise ValueError(f"Unsupported calc kind: {kind!r}")

    def pairwise_template(self):
        if self._pairwise_tmpl is None:
            self._pairwise_tmpl = _pairwise_template(
                getattr(self.inner, "metadata_template", None)
            )
        return self._pairwise_tmpl

    def tcol_template(self, columns):
        from pytesmo.validation_framework.gpu_metrics import _tcol_template
        key = tuple(columns)
        if key not in self._tcol_tmpl_cache:
            self._tcol_tmpl_cache[key] = _tcol_template(
                list(columns),
                getattr(self.inner, "refname", ""),
                getattr(self.inner, "bootstrap_cis", False),
                getattr(self.inner, "metadata_template", None),
            )
        return self._tcol_tmpl_cache[key]

    def empty_result(self, gpi_info, n_obs=0, status=STATUS_METRICS_FAILED):
        if self.kind == "pairwise":
            t = self.pairwise_template()
        else:
            cols = [getattr(self.inner, "refname", "")]
            try:
                dummy = pd.DataFrame([], columns=cols)
                self.inner.calc_metrics(dummy, gpi_info or (0, np.nan, np.nan))
                t = dict(self.inner.result_template)
            except Exception:
                t = self.tcol_template(cols)
        return _meta_fill(t, gpi_info,
            getattr(self.inner, "metadata_template", None),
            n_obs, status)


def _meta_fill(t, gpi_info, meta, n_obs, status):
    r = {k: np.array(v, copy=True) for k, v in t.items()}
    r["gpi"][0] = int(gpi_info[0])
    r["lon"][0] = float(gpi_info[1])
    r["lat"][0] = float(gpi_info[2])
    r["n_obs"][0] = int(n_obs)
    r["status"][0] = int(status)
    if meta and len(gpi_info) > 3:
        for k in meta:
            if k in gpi_info[3]:
                r[k][0] = gpi_info[3][k]
    return r


class GPUBatchedValidation(Validation):
    def __init__(
        self,
        datamanager,
        temporal_matcher,
        temporal_ref,
        spatial_ref,
        scaling,
        scaling_ref,
        metrics_calculators,
        batch_size: int = 1000,
        **kwargs: Any,
    ):
        super().__init__(
            datasets=datamanager,
            spatial_ref=spatial_ref,
            metrics_calculators=metrics_calculators,
            temporal_matcher=temporal_matcher,
            temporal_ref=temporal_ref,
            scaling=scaling,
            scaling_ref=scaling_ref,
            **{k: v for k, v in kwargs.items()
               if k in ("masking_datasets", "period")},
        )
        self.batch_size = int(batch_size)
        self._rk_specs: dict = {}
        self._calc_specs: dict = {}
        self._init_specs()

    def _init_specs(self):
        for n, k in self.metrics_c:
            if (n, k) in self._calc_specs:
                continue
            bound = self.metrics_c[(n, k)]
            calc_obj = bound.__self__
            kind, inner, adapter = _detect_calc_type(calc_obj)
            self._calc_specs[(n, k)] = _CalcSpec(kind, n, k, inner, adapter)
            result_names = get_result_combinations(
                self.data_manager.ds_dict, n=k
            )
            for rk in result_names:
                self._rk_specs[rk] = _RKSpec(
                    self._calc_specs[(n, k)],
                    [r[0] for r in rk],
                )

    def calc(self, gpis, lons, lats, *args,
             rename_cols=True, only_with_reference=False,
             handle_errors="raise"):
        handle_errors = handle_errors.lower()
        if args:
            gpis, lons, lats, args = Validation.args_to_iterable(
                gpis, lons, lats, *args, n=3
            )
        else:
            gpis, lons, lats = Validation.args_to_iterable(gpis, lons, lats)
        gpi_list = list(zip(gpis, lons, lats, *args)) if args else list(
            zip(gpis, lons, lats))

        acc: dict = {}
        results: dict = {}

        for gpi_idx, gpi_info in enumerate(gpi_list):
            try:
                df_dict = self.data_manager.get_data(*gpi_info[:3])
            except Exception as exc:
                if handle_errors == "raise":
                    raise eh.DataManagerError(
                        f"Data read failed for gpi {gpi_info}: {exc}"
                    ) from exc
                self._append_dummy(gpi_info, results, only_with_reference)
                continue

            if len(df_dict) == 0:
                if handle_errors == "raise":
                    raise eh.NoGpiDataError(f"No data for gpi {gpi_info}")
                self._append_dummy(gpi_info, results, only_with_reference)
                continue

            data_df_dict = {}
            for ds in df_dict:
                data_df_dict[ds] = df_dict[ds][
                    self.data_manager.datasets[ds]["columns"]
                ]

            if self.masking_dm is not None:
                ref_df = data_df_dict.get(self.temporal_ref)
                if ref_df is None:
                    self._append_dummy(gpi_info, results, only_with_reference)
                    continue
                masked = self.mask_dataset(ref_df, gpi_info)
                if len(masked) == 0:
                    self._append_dummy(gpi_info, results, only_with_reference)
                    continue
                data_df_dict[self.temporal_ref] = masked

            try:
                matched_n = self.temporal_match_datasets(data_df_dict)
            except Exception as exc:
                if handle_errors == "raise":
                    raise eh.TemporalMatchingError(
                        f"Temporal matching failed for gpi {gpi_info}: {exc}"
                    ) from exc
                self._append_dummy(gpi_info, results, only_with_reference)
                continue

            for n, k in self.metrics_c:
                spec = self._calc_specs[(n, k)]
                rk_list = get_result_combinations(
                    self.data_manager.ds_dict, n=k
                )
                if only_with_reference:
                    rk_list = [
                        rk for rk in rk_list
                        if self.data_manager.reference_name in
                        [r[0] for r in rk]
                    ]

                n_matched = matched_n.get((n, k), {})
                if len(n_matched) == 0:
                    for rk in rk_list:
                        rk_ds = [r[0] for r in rk]
                        if only_with_reference:
                            if self.data_manager.reference_name not in rk_ds:
                                continue
                        r = spec.empty_result(
                            gpi_info, n_obs=0,
                            status=eh.NO_TEMP_MATCHED_DATA,
                        )
                        results.setdefault(rk, []).append(r)
                        acc.setdefault(rk, []).append(
                            (gpi_info, None, gpi_idx))
                    continue

                rk_data_map: dict = {}
                for data_unscaled, rk in self.k_datasets_from(n_matched, rk_list):
                    rk_ds = [r[0] for r in rk]
                    if only_with_reference:
                        if self.data_manager.reference_name not in rk_ds:
                            continue
                    rk_data_map[rk] = data_unscaled

                for rk in rk_list:
                    rk_ds = [r[0] for r in rk]
                    if only_with_reference:
                        if self.data_manager.reference_name not in rk_ds:
                            continue
                    data_unscaled = rk_data_map.get(rk)
                    if data_unscaled is None or len(data_unscaled) == 0:
                        r = spec.empty_result(
                            gpi_info, n_obs=0,
                            status=eh.NO_TEMP_MATCHED_DATA,
                        )
                        results.setdefault(rk, []).append(r)
                        acc.setdefault(rk, []).append(
                            (gpi_info, None, gpi_idx))
                        continue

                    data_sc = data_unscaled.rename(columns=lambda x: x[0])
                    if self.scaling is not None:
                        try:
                            si = data_sc.columns.tolist().index(
                                self.scaling_ref)
                            data_sc = self.scaling.scale(
                                data_sc, si, gpi_info)
                        except Exception as exc:
                            if handle_errors == "raise":
                                raise eh.ScalingError(
                                    f"Scaling failed {rk} gpi {gpi_info}: "
                                    f"{exc}"
                                ) from exc
                            r = spec.empty_result(
                                gpi_info, n_obs=0,
                                status=eh.SCALING_FAILED,
                            )
                            results.setdefault(rk, []).append(r)
                            acc.setdefault(rk, []).append(
                                (gpi_info, None, gpi_idx))
                            continue
                    if self.scaling_ref not in rk_ds:
                        data_sc = data_sc.drop(
                            columns=[self.scaling_ref], errors="ignore",
                        )

                    try:
                        data_sc = data_sc[[d for d in rk_ds]]
                    except KeyError:
                        r = spec.empty_result(
                            gpi_info, n_obs=0,
                            status=STATUS_METRICS_FAILED,
                        )
                        results.setdefault(rk, []).append(r)
                        acc.setdefault(rk, []).append(
                            (gpi_info, None, gpi_idx))
                        continue

                    acc.setdefault(rk, []).append(
                        (gpi_info, data_sc, gpi_idx))

            if sum(len(v) for v in acc.values()) >= self.batch_size:
                self._flush(acc, results)
                acc.clear()

        if acc:
            self._flush(acc, results)

        compact: dict = {}
        for key in results:
            entries = results[key]
            first = next((e for e in entries if e is not None), None)
            if first is None:
                compact[key] = {}
                continue
            fnames = list(first.keys())
            compact[key] = {}
            for fn in fnames:
                vals = []
                for e in entries:
                    if e is not None and fn in e:
                        vals.append(e[fn][0])
                    elif fn in ("n_obs", "status"):
                        vals.append(0)
                    else:
                        vals.append(np.nan)
                compact[key][fn] = np.array(vals, dtype=first[fn].dtype)
        return compact

    def _append_dummy(self, gpi_info, results, only_with_reference):
        dummy = self.dummy_validation_result(
            gpi_info, rename_cols=False,
            only_with_reference=only_with_reference,
        )
        for rk, rlist in dummy.items():
            if rk not in results:
                results[rk] = []
            results[rk].append(rlist[0] if rlist else None)

    def _flush(self, acc, results):
        per_gpi_by_rk: dict = {}
        for rk, entries in list(acc.items()):
            spec = self._rk_specs[rk].spec
            real = [
                (entry_idx, info, df)
                for entry_idx, (info, df, _) in enumerate(entries)
                if df is not None
            ]
            if not real:
                continue
            infos = [info for _, info, _ in real]
            dfs = [df for _, _, df in real]
            if spec.kind == "pairwise" and spec.adapter is None:
                computed = spec.batched.calc_batch(dfs, infos)
            elif spec.kind == "tcol" and spec.adapter is None:
                computed = spec.batched.calc_batch(
                    dfs, infos, columns=self._rk_specs[rk].columns,
                )
            else:
                computed = self._flush_subset(spec, rk, dfs, infos, entries)
            per_gpi_by_rk[rk] = {}
            for j, (entry_idx, _, _) in enumerate(real):
                per_gpi_by_rk[rk][entry_idx] = computed[j]

        for rk, entries in acc.items():
            rlist = results.setdefault(rk, [])
            real_map = per_gpi_by_rk.get(rk, {})
            for entry_idx, (info, df, gpi_pos) in enumerate(entries):
                if entry_idx in real_map:
                    rlist[gpi_pos] = real_map[entry_idx]

    def _flush_subset(self, spec, rk, dfs, infos, entries):
        subsets = getattr(spec.adapter, "subsets", {})
        is_stability = (
            StabilityMetricsAdapter is not None
            and spec.adapter is not None
            and isinstance(spec.adapter, StabilityMetricsAdapter)
        )
        n = len(dfs)
        subset_df_map: dict = {s: [None] * n for s in subsets}
        s_valid: dict = {s: [] for s in subsets}
        for j, df in enumerate(dfs):
            info = infos[j]
            for sname, distr in subsets.items():
                try:
                    sel = distr.select(df) if len(df) > 0 else df
                except Exception:
                    sel = df.iloc[0:0]
                subset_df_map[sname][j] = sel

        # Per-gpi merged result dicts
        merged: dict = {j: {} for j in range(n)}
        # Get common keys from first valid subset of first gpi
        for sname, sdfs in subset_df_map.items():
            valid_idx = [j for j, d in enumerate(sdfs) if d is not None and not d.empty]
            if not valid_idx:
                continue
            s_infos = [infos[j] for j in valid_idx]
            sdfs_valid = [sdfs[j] for j in valid_idx]
            if spec.kind == "pairwise":
                sub_res = spec.batched.calc_batch(sdfs_valid, s_infos)
            else:
                sub_res = spec.batched.calc_batch(
                    sdfs_valid, s_infos, columns=self._rk_specs[rk].columns,
                )
            for local_j, gpi_j in enumerate(valid_idx):
                sr = sub_res[local_j]
                if sr is None:
                    continue
                if sname == sorted(subsets.keys())[0]:
                    merged[gpi_j]["gpi"] = sr["gpi"]
                    merged[gpi_j]["lon"] = sr["lon"]
                    merged[gpi_j]["lat"] = sr["lat"]
                    for mk in getattr(spec.inner, "metadata_template", {}) or {}:
                        if mk in sr:
                            merged[gpi_j][mk] = sr[mk]
                prefix = f"{sname}|"
                for sk, sv in sr.items():
                    if sk in ("gpi", "lon", "lat"):
                        continue
                    merged[gpi_j][prefix + sk] = sv

        if is_stability:
            try:
                from scipy.stats import theilslopes as _theilslopes
                sup = {"R", "BIAS", "urmsd"}
                for j, r in merged.items():
                    by_metric: dict = {}
                    for k, v in r.items():
                        if "|" not in str(k):
                            continue
                        parts = str(k).rsplit("|", 1)
                        if len(parts) != 2:
                            continue
                        year_s, mname = parts
                        if mname not in sup or year_s == "bulk":
                            continue
                        try:
                            by_metric[mname].append((int(year_s), v[0]))
                        except (ValueError, KeyError):
                            pass
                    for mname, entries in by_metric.items():
                        entries.sort(key=lambda x: x[0])
                        vs = np.array([e[1] for e in entries])
                        valid = ~np.isnan(vs)
                        if valid.sum() < 2:
                            r[f"bulk|slope{mname.upper()}"] = np.array([np.nan])
                            continue
                        try:
                            slope, _, _, _ = _theilslopes(
                                vs[valid],
                                np.array([e[0] for e in entries])[valid],
                            )
                            r[f"bulk|slope{mname.upper()}"] = np.array(
                                [slope * 10.0])
                        except Exception:
                            r[f"bulk|slope{mname.upper()}"] = np.array([np.nan])
            except Exception as exc:
                _LOGGER.warning("Stability metrics failed: %s", exc)

        return [merged[j] for j in range(n)]
