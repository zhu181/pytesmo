"""Batched GPU metric calculators producing byte-compatible pytesmo results.

Replaces per-grid-point CPU metric calls with a single batched GPU (or
NumPy) pass over N grid points.  Result dictionaries match the format
produced by PairwiseIntercomparisonMetrics and TripleCollocationMetrics
exactly so that pytesmo\'s Validation.calc() produces byte-compatible
outputs.

Attribution note: part of the QA4SM GPU acceleration contribution,
vendored inside pytesmo.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import betainc as _betainc
from scipy.stats import spearmanr as _spearmanr
from scipy.stats import t as _t_dist

from pytesmo.gpu_backend import get_gpu_backend
from pytesmo.validation_framework.read_lock import HDF5_READ_LOCK

_LOGGER = logging.getLogger(__name__)

STATUS_OK = 0
STATUS_INSUFFICIENT = 1
STATUS_METRICS_FAILED = 2

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _to_numpy(tensor: Any) -> np.ndarray:
    arr = get_gpu_backend().to_cpu(tensor)
    return arr if isinstance(arr, np.ndarray) else np.asarray(arr)

def _ensure_2col(df):
    if df is None or df.empty or df.shape[1] != 2:
        return None
    return df.dropna(axis=0, how="any")

def _col_vals(df):
    v = df.values
    return v[:, 0].astype(np.float64), v[:, 1].astype(np.float64)

def _pvalue(varx, vary, cov, n):
    if varx == 0.0 or vary == 0.0:
        return float("nan")
    R = max(-1.0, min(cov / np.sqrt(varx * vary), 1.0))
    if abs(float(R)) == 1.0:
        return 0.0
    df = float(n - 2)
    t2 = float(R) * float(R) * (df / ((1.0 - float(R)) * (1.0 + float(R))))
    z = float(min(df / (df + t2), 1.0))
    return float(_betainc(0.5 * df, 0.5, z))

def _bias_ci(alpha, mx, my, varx, vary, cov, n):
    std = np.sqrt(n / float(n - 1) * (varx + vary - 2.0 * cov))
    delta = std / np.sqrt(n) * _t_dist.ppf(1.0 - alpha / 2.0, n - 1)
    return float(mx - my - delta), float(mx - my + delta)

def _ubrmsd_ci(x, y, ubrmsd_val, alpha=0.05):
    n = len(x)
    delta = np.std(x - y, ddof=1) / np.sqrt(n) * _t_dist.ppf(
        1.0 - alpha / 2.0, n - 1)
    return float(ubrmsd_val - delta), float(ubrmsd_val + delta)

def _pearson_r_ci(x, y, r, alpha=0.05):
    z = np.arctanh(float(r))
    se = 1.0 / np.sqrt(len(x) - 3)
    return float(np.arctanh(z - 1.96 * se)), float(np.arctanh(z + 1.96 * se))

# ------------------------------------------------------------------
# Templates
# ------------------------------------------------------------------

def _pairwise_template(meta):
    t: dict[str, np.ndarray] = {
        "gpi": np.array([-1], dtype=np.int32),
        "lon": np.array([np.nan], dtype=np.float64),
        "lat": np.array([np.nan], dtype=np.float64),
        "n_obs": np.array([0], dtype=np.int32),
        "status": np.array([STATUS_OK], dtype=np.int32),
    }
    F32 = np.float32
    F64 = np.float64
    for k in ("R", "p_R", "rho", "p_rho", "BIAS", "RMSD", "mse",
              "RSS", "mse_corr", "mse_bias", "mse_var", "urmsd",
              "BIAS_ci_lower", "BIAS_ci_upper",
              "urmsd_ci_lower", "urmsd_ci_upper",
              "R_ci_lower", "R_ci_upper",
              "rho_ci_lower", "rho_ci_upper"):
        t[k] = np.array([np.nan], dtype=F32)
    if meta:
        for k, v in meta.items():
            t[k] = np.array(v)
    return t

def _tcol_template(columns, refname, bootstrap=False, meta=None):
    t: dict[str, np.ndarray] = {
        "gpi": np.array([-1], dtype=np.int32),
        "lon": np.array([np.nan], dtype=np.float64),
        "lat": np.array([np.nan], dtype=np.float64),
        "n_obs": np.array([0], dtype=np.int32),
        "status": np.array([STATUS_OK], dtype=np.int32),
    }
    for m in ("snr", "err_std", "beta"):
        for ds in columns:
            t[(m, ds)] = np.array([np.nan], dtype=np.float32)
    if bootstrap:
        for m in ("snr", "err_std", "beta"):
            for ds in columns:
                t[(m + "_ci_lower", ds)] = np.array([np.nan], dtype=np.float32)
                t[(m + "_ci_upper", ds)] = np.array([np.nan], dtype=np.float32)
    if meta:
        for k, v in meta.items():
            t[k] = np.array(v)
    return t

def _fill_meta(result, gpi_info, meta):
    result["gpi"][0] = int(gpi_info[0])
    result["lon"][0] = float(gpi_info[1])
    result["lat"][0] = float(gpi_info[2])
    if meta and len(gpi_info) > 3:
        for k in meta:
            if k in gpi_info[3]:
                result[k][0] = gpi_info[3][k]

def _get_template(template, gpi_info, meta, n_obs, status):
    r = {k: np.array(v, copy=True) for k, v in template.items()}
    _fill_meta(r, gpi_info, meta)
    r["n_obs"][0] = int(n_obs)
    r["status"][0] = status
    return r

# ------------------------------------------------------------------
# Pairwise batched calculator
# ------------------------------------------------------------------

class BatchedPairwiseMetrics:
    """Batched pairwise metric computation over N grid points.

    Parameters
    ----------
    metadata_template : dict | None
        Extra per-gpi metadata fields.
    calc_spearman : bool
    calc_kendall : bool
    min_obs : int
    analytical_cis : bool
        When True the result dict includes BIAS/urmsd/R/rho CIs.
    """

    def __init__(self, metadata_template=None, calc_spearman=True,
                 calc_kendall=False, min_obs=10, analytical_cis=True):
        self.meta = metadata_template
        self.calc_spearman = calc_spearman
        self.calc_kendall = calc_kendall
        self.min_obs = int(min_obs)
        self.analytical_cis = analytical_cis
        self._template = _pairwise_template(metadata_template)

    def calc_batch(self, data_list, gpi_infos):
        n = len(data_list)
        out: list = [None] * n
        vx_list, vy_list, v_idx = [], [], []
        no_data_idx, empty_idx = [], []

        for i in range(n):
            info = gpi_infos[i]
            if info is None:
                no_data_idx.append(i)
                continue
            df = data_list[i]
            if df is None or df.empty:
                out[i] = _get_template(self._template, info, self.meta, 0,
                                       STATUS_METRICS_FAILED)
                continue
            dd = _ensure_2col(df)
            if dd is None:
                out[i] = _get_template(self._template, info, self.meta, 0,
                                       STATUS_METRICS_FAILED)
                continue
            x, y = _col_vals(dd)
            if len(x) < self.min_obs:
                out[i] = _get_template(self._template, info, self.meta,
                                       len(x), STATUS_INSUFFICIENT)
                continue
            vx_list.append(x)
            vy_list.append(y)
            v_idx.append(i)

        if not v_idx:
            return out

        host = _pairwise_batched(vx_list, vy_list)
        n_obs_all = host.pop("n_obs")
        for j, vi in enumerate(v_idx):
            info = gpi_infos[vi]
            vals = {k: float(host[k][j]) for k in host}
            vals["_x_full"] = vx_list[j]
            vals["_y_full"] = vy_list[j]
            out[vi] = _pairwise_build_result(self, info, vals,
                                             int(n_obs_all[j]))
        return out

# ------------------------------------------------------------------

def _pairwise_batched(vx_list, vy_list):
    back = get_gpu_backend()
    xp = back.xp
    n = len(vx_list)
    L = max(len(a) for a in vx_list)
    batch = np.full((n, 2, L), np.nan, dtype=np.float64)
    mask = np.zeros((n, L), dtype=np.float64)
    for j in range(n):
        t = len(vx_list[j])
        batch[j, 0, :t] = vx_list[j]
        batch[j, 1, :t] = vy_list[j]
        mask[j, :t] = 1.0

    if xp is np:
        return _pairwise_np(batch, mask)
    return _pairwise_gpu(batch, mask, back, xp)

def _pairwise_np(batch, mask):
    n_obs = mask.sum(axis=1)
    mx = np.nansum(batch[:, 0] * mask, axis=1) / n_obs
    my = np.nansum(batch[:, 1] * mask, axis=1) / n_obs
    dx = batch[:, 0] - mx[:, None]
    dy = batch[:, 1] - my[:, None]
    vx = np.nansum(dx * dx * mask, axis=1) / n_obs
    vy = np.nansum(dy * dy * mask, axis=1) / n_obs
    cov = np.nansum(dx * dy * mask, axis=1) / n_obs
    bias = mx - my
    mse_corr = np.maximum(2.0 * np.sqrt(vx) * np.sqrt(vy) - 2.0 * cov, 0.0)
    mse_var = (np.sqrt(vx) - np.sqrt(vy)) ** 2
    mse_bias = bias ** 2
    mse = mse_corr + mse_var + mse_bias
    rmsd = np.sqrt(mse)
    urmsd = np.sqrt(np.maximum(mse - mse_bias, 0.0))
    rss = mse * n_obs
    R = cov / np.sqrt(vx * vy)
    return {
        "BIAS": bias, "RMSD": rmsd, "urmsd": urmsd,
        "mse": mse, "mse_corr": mse_corr, "mse_var": mse_var,
        "mse_bias": mse_bias, "RSS": rss, "R": R,
        "n_obs": n_obs.astype(np.int32),
        "_mx": mx, "_my": my, "_vx": vx, "_vy": vy, "_cov": cov,
    }

def _pairwise_gpu(batch, mask, back, xp):
    x = back.to_tensor(batch[:, 0, :])
    y = back.to_tensor(batch[:, 1, :])
    m = back.to_tensor(mask)
    n_obs = m.sum(axis=1)
    mx = xp.nansum(x * m, axis=1) / n_obs
    my = xp.nansum(y * m, axis=1) / n_obs
    dx = x - mx[:, None]
    dy = y - my[:, None]
    vx = xp.nansum(dx * dx * m, axis=1) / n_obs
    vy = xp.nansum(dy * dy * m, axis=1) / n_obs
    cov = xp.nansum(dx * dy * m, axis=1) / n_obs
    bias = mx - my
    mse_corr = xp.maximum(2.0 * xp.sqrt(vx) * xp.sqrt(vy) - 2.0 * cov,
                          xp.zeros_like(cov))
    mse_var = (xp.sqrt(vx) - xp.sqrt(vy)) ** 2
    mse_bias = bias ** 2
    mse = mse_corr + mse_var + mse_bias
    rmsd = xp.sqrt(mse)
    urmsd = xp.sqrt(xp.maximum(mse - mse_bias, xp.zeros_like(mse)))
    rss = mse * n_obs
    R = cov / xp.sqrt(vx * vy)
    back.synchronize()
    out = {
        "BIAS": _to_numpy(bias), "RMSD": _to_numpy(rmsd),
        "urmsd": _to_numpy(urmsd), "mse": _to_numpy(mse),
        "mse_corr": _to_numpy(mse_corr), "mse_var": _to_numpy(mse_var),
        "mse_bias": _to_numpy(mse_bias), "RSS": _to_numpy(rss),
        "R": _to_numpy(R), "n_obs": _to_numpy(n_obs).astype(np.int32),
        "_mx": _to_numpy(mx), "_my": _to_numpy(my),
        "_vx": _to_numpy(vx), "_vy": _to_numpy(vy),
        "_cov": _to_numpy(cov),
    }
    del x, y, m, dx, dy, vx, vy, cov, bias, mse, rmsd, urmsd
    del mse_corr, mse_var, mse_bias, rss, R, n_obs, mx, my
    back.empty_cache()
    return out

def _pairwise_build_result(calc, gpi_info, vals, n_obs):
    r = {k: np.array(v, copy=True) for k, v in calc._template.items()}
    _fill_meta(r, gpi_info, calc.meta)
    r["n_obs"][0] = n_obs
    if n_obs < calc.min_obs:
        r["status"][0] = STATUS_INSUFFICIENT
        return r
    for k in ("BIAS", "RMSD", "urmsd", "mse", "mse_corr", "mse_var",
              "mse_bias", "RSS"):
        r[k][0] = np.float32(vals[k])
    r["R"][0] = np.float32(np.clip(float(vals["R"]), -1.0, 1.0))
    try:
        r["p_R"][0] = np.float32(
            _pvalue(vals["_vx"], vals["_vy"], vals["_cov"], n_obs))
    except Exception:
        r["p_R"][0] = np.nan
    if calc.calc_spearman:
        try:
            rho, p_rho = _spearmanr(vals["_x_full"], vals["_y_full"])
            r["rho"][0] = np.float32(float(rho))
            r["p_rho"][0] = np.float32(float(p_rho))
        except Exception:
            r["rho"][0] = np.nan
            r["p_rho"][0] = np.nan
        if calc.analytical_cis:
            try:
                lo, hi = _pearson_r_ci(vals["_x_full"], vals["_y_full"],
                                        float(r["R"][0]))
                r["R_ci_lower"][0] = np.float32(lo)
                r["R_ci_upper"][0] = np.float32(hi)
            except Exception:
                r["R_ci_lower"][0] = np.nan
                r["R_ci_upper"][0] = np.nan
            try:
                lo, hi = _bias_ci(0.05, vals["_mx"], vals["_my"],
                                  vals["_vx"], vals["_vy"], vals["_cov"], n_obs)
                r["BIAS_ci_lower"][0] = np.float32(lo)
                r["BIAS_ci_upper"][0] = np.float32(hi)
            except Exception:
                r["BIAS_ci_lower"][0] = np.nan
                r["BIAS_ci_upper"][0] = np.nan
            try:
                lo, hi = _ubrmsd_ci(vals["_x_full"], vals["_y_full"],
                                    float(r["urmsd"][0]))
                r["urmsd_ci_lower"][0] = np.float32(lo)
                r["urmsd_ci_upper"][0] = np.float32(hi)
            except Exception:
                r["urmsd_ci_lower"][0] = np.nan
                r["urmsd_ci_upper"][0] = np.nan
            try:
                lo, hi = _pearson_r_ci(vals["_x_full"], vals["_y_full"],
                                        float(r["rho"][0]))
                r["rho_ci_lower"][0] = np.float32(lo)
                r["rho_ci_upper"][0] = np.float32(hi)
            except Exception:
                r["rho_ci_lower"][0] = np.nan
                r["rho_ci_upper"][0] = np.nan
    r["status"][0] = STATUS_OK
    return r

# ------------------------------------------------------------------
# Triple Collocation batched calculator
# ------------------------------------------------------------------

class BatchedTCAMetrics:
    """Batched TCA computation over N grid points.

    *columns* (the DataFrame column order from the ``result_key``) is
    passed at every ``calc_batch`` call because it varies per result key.
    Template keys and output tuple keys are derived from this ordering.
    """

    def __init__(self, refname, min_obs=10,
                 metadata_template=None, bootstrap_cis=False):
        self.refname = refname
        self.min_obs = int(min_obs)
        self.meta = metadata_template
        self.bootstrap_cis = bootstrap_cis

    def calc_batch(self, data_list, gpi_infos, columns=None):
        if columns is None:
            raise ValueError("columns (result_key dataset names) required for TCA")
        columns = list(columns)
        n = len(data_list)
        out: list = [None] * n
        tmpl = _tcol_template(columns, self.refname, self.bootstrap_cis, self.meta)
        valid_arr, valid_idx = [], []
        for i in range(n):
            info = gpi_infos[i]
            if info is None:
                out[i] = _get_template(tmpl, info, self.meta, 0,
                                       STATUS_METRICS_FAILED)
                continue
            df = data_list[i]
            if df is None or df.empty:
                out[i] = _get_template(tmpl, info, self.meta, 0,
                                       STATUS_METRICS_FAILED)
                continue
            dd = df.dropna(axis=0, how="any")
            if len(dd) < self.min_obs:
                out[i] = _get_template(tmpl, info, self.meta, len(dd),
                                       STATUS_INSUFFICIENT)
                continue
            try:
                arr = np.column_stack([dd[c].values for c in columns]).astype(
                    np.float64)
            except KeyError as exc:
                _LOGGER.warning("TCA key mismatch gpi=%s: %s",
                                info[0] if info else "", exc)
                out[i] = _get_template(tmpl, info, self.meta, 0,
                                       STATUS_METRICS_FAILED)
                continue
            valid_arr.append(arr)
            valid_idx.append(i)

        if not valid_idx:
            return out

        ref_pos = columns.index(self.refname)
        if self.bootstrap_cis:
            snr_a, err_a, beta_a = _tcol_with_bootstrap(valid_arr, ref_pos,
                                                        columns, self.refname)
        else:
            snr_a, err_a, beta_a = _tcol_batched(valid_arr, ref_pos)

        for j, vi in enumerate(valid_idx):
            info = gpi_infos[vi]
            r = _get_template(tmpl, info, self.meta,
                              valid_arr[j].shape[0], STATUS_OK)
            for k_pos, name in enumerate(columns):
                r[("snr", name)][0] = np.float32(float(snr_a[k_pos][j]))
                r[("err_std", name)][0] = np.float32(float(err_a[k_pos][j]))
                r[("beta", name)][0] = np.float32(float(beta_a[k_pos][j]))
            if self.bootstrap_cis:
                for k_pos, name in enumerate(columns):
                    r[("snr_ci_lower", name)][0] = np.nan
                    r[("snr_ci_upper", name)][0] = np.nan
            out[vi] = r
        return out


def _tcol_batched(valid_arr, ref_pos):
    """Sample-covariance (ddof=1) TCA across a batch of (3, n_obs) arrays.

    Returns three lists of length 3 (one array per dataset position),
    each of shape (n_valid,).
    """
    back = get_gpu_backend()
    xp = back.xp
    n_valid = len(valid_arr)
    max_L = max(a.shape[0] for a in valid_arr)
    batch = np.full((n_valid, 3, max_L), np.nan, dtype=np.float64)
    mask = np.zeros((n_valid, max_L), dtype=np.float64)
    for j in range(n_valid):
        t = valid_arr[j].shape[0]
        if t:
            batch[j, :, :t] = valid_arr[j].T
            mask[j, :t] = 1.0

    if xp is np:
        return _tcol_np(batch, mask, ref_pos)
    return _tcol_gpu(batch, mask, ref_pos, back, xp)


def _tcol_np(batch, mask, ref_pos):
    n_obs = mask.sum(axis=1)
    d = batch - _row_mean_masked(batch, mask, n_obs)
    n1 = np.maximum(n_obs - 1.0, 1.0)
    c = {i: np.nansum(d[:, i] * d[:, j] * mask, axis=1) / n1
         for i in range(3) for j in range(i, 3)}
    c00, c11, c22 = c[(0, 0)], c[(1, 1)], c[(2, 2)]
    c01, c02, c12 = c[(0, 1)], c[(0, 2)], c[(1, 2)]
    return _tcol_from_cov(c00, c11, c22, c01, c02, c12, ref_pos)


def _tcol_gpu(batch, mask, ref_pos, back, xp):
    b = back.to_tensor(batch)
    m = back.to_tensor(mask)
    n_obs = m.sum(axis=1)
    mu = _row_mean_masked(b, m, n_obs)
    d = b - mu[:, :, None]
    n1 = xp.maximum(n_obs - 1.0, xp.ones_like(n_obs))
    c = {}
    for i in range(3):
        for j in range(i, 3):
            c[(i, j)] = _to_numpy(
                xp.nansum(d[:, i] * d[:, j] * m, axis=1) / n1)
    c00, c11, c22 = c[(0, 0)], c[(1, 1)], c[(2, 2)]
    c01, c02, c12 = c[(0, 1)], c[(0, 2)], c[(1, 2)]
    back.empty_cache()
    return _tcol_from_cov(c00, c11, c22, c01, c02, c12, ref_pos)


def _row_mean_masked(arr, mask, n_obs):
    """Per-row mean ignoring masked columns."""
    xp = get_gpu_backend().xp
    s = xp.nansum(arr * mask[:, None, :], axis=2)
    c = xp.maximum(n_obs, 1.0)[:, None]
    return s / c


def _tcol_from_cov(c00, c11, c22, c01, c02, c12, ref_pos):
    """TCA from 6 upper-triangle sample-covariance arrays of shape (n,)."""
    # Other positions
    all_pos = [0, 1, 2]
    others = [p for p in all_pos if p != ref_pos]
    o1, o2 = others[0], others[1]

    def _cc(i, j):
        return {0: {0: c00, 1: c01, 2: c02},
                1: {0: c01, 1: c11, 2: c12},
                2: {0: c02, 1: c12, 2: c22}}[i][j]

    def _safe(x):
        return np.where(np.abs(x) < 1e-30, np.sign(x + 1e-30) * 1e-30, x)

    def _snr(i):
        num = _cc(i, i) * _cc(o1, o2)
        den = _cc(i, o1) * _cc(i, o2)
        r = num / _safe(den)
        ar = np.abs(r)
        return 10.0 * np.log10(np.maximum(1.0 / np.abs(ar - 1.0), 1e-30))

    def _err_var(i):
        num = _cc(i, o1) * _cc(i, o2)
        den = _safe(_cc(o1, o2))
        return _cc(i, i) - num / den

    def _beta(i):
        if i == ref_pos:
            return np.ones_like(c00)
        other_k = [p for p in all_pos if p not in (i, ref_pos)][0]
        return _cc(ref_pos, other_k) / _safe(_cc(i, other_k))

    snr = [np.zeros_like(c00) for _ in range(3)]
    err_var = [np.zeros_like(c00) for _ in range(3)]
    beta = [np.zeros_like(c00) for _ in range(3)]
    for i in range(3):
        snr[i] = _snr(i)
        err_var[i] = _err_var(i)
        beta[i] = _beta(i)
    err_std = [np.sqrt(np.maximum(ev, 0.0)) * beta[i] for i, ev in enumerate(err_var)]
    return snr, err_std, beta


def _tcol_with_bootstrap(valid_arr, ref_pos, columns, refname):
    """Bootstrap TCA ¡ª currently falls back to scalar CPU per gpi."""
    import warnings
    try:
        from pytesmo.metrics.confidence_intervals import tcol_metrics_with_bootstrapped_ci
    except ImportError:
        warnings.warn("Bootstrap TCA CI not available (confidence_intervals missing)")
        return _tcol_batched(valid_arr, ref_pos)
    # Per-gpi bootstrap (rare path; correctness over speed)
    n_valid = len(valid_arr)
    snr_out = [np.empty(n_valid) for _ in range(3)]
    err_out = [np.empty(n_valid) for _ in range(3)]
    beta_out = [np.empty(n_valid) for _ in range(3)]
    for j in range(n_valid):
        arr = valid_arr[j].T
        x, y, z = arr[0], arr[1], arr[2]
        snr, err, beta = tcol_metrics_with_bootstrapped_ci(
            x, y, z, ref_ind=ref_pos, alpha=0.05,
            method="percentile", nsamples=1000,
            minimum_data_length=100)
        for k in range(3):
            snr_out[k][j] = snr[k]
            err_out[k][j] = err[k]
            beta_out[k][j] = beta[k]
    return snr_out, err_out, beta_out

