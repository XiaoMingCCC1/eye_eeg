# -*- coding: utf-8 -*-
"""
MI robustness re-analysis (equal-length / surrogate / sensitivity) for R4-2.

Implements the procedure fixed in f:/eye_eeg/mi_robustness_analysis_protocol.md.
Stages are explicit so that thresholds are frozen before any condition MI is seen:

    python mi_robustness_analysis.py inventory   # parse inputs, build manifest
    python mi_robustness_analysis.py probe       # contiguity / n_i / ACF diagnostics only
    python mi_robustness_analysis.py run         # full MI + surrogate + stats

No condition-level MI is computed in the `inventory` or `probe` stages.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import digamma

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------
EEG_LOG_DIR = Path(r"F:\eye_eeg_1\jiequ\jiequ_eeg_ave")
EEG_CSV_DIR = Path(r"F:\eye_eeg_1\jiequ\data_jiequ_ave")
PUPIL_DIR = Path(r"F:\eye_eeg\tongkong")
PROTOCOL = Path(r"f:\eye_eeg\mi_robustness_analysis_protocol.md")
REGION_TXT = Path(r"f:\eye_eeg\EGI_HydroCel_129_brain_regions.txt")
OUT_ROOT = Path(r"f:\eye_eeg\ouhe\results_mi")

CONDITIONS = ["negative", "neutral", "positive"]
COND_VIDEO = {"negative": 1, "neutral": 2, "positive": 3}
VIDEO_COND = {v: k for k, v in COND_VIDEO.items()}

REGION_ORDER = ["prefrontal", "frontal", "central", "parietal", "temporal", "occipital"]

# --------------------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------------------
_HEADER_FILE_RE = re.compile(r"^文件:\s*(.+?)\s*$", re.M)
_HEADER_SID_RE = re.compile(r"^被试ID:\s*(.+?)\s*$", re.M)
_CHAN_BLOCK_RE = re.compile(r"=== 通道 (\d+) 参数提取开始 ===")
_TP_RE = re.compile(
    r"时间点\s*(\d+):\s*\n\s*-\s*偏移\(offset\):\s*([-+]?\d*\.?\d+)"
    r"\s*\n\s*-\s*指数\(exponent\):\s*([-+]?\d*\.?\d+)"
)
_PUPIL_HDR_RE = re.compile(r"^#\s*([^:]+):\s*(.+?)\s*$", re.M)


@dataclass
class SprintLog:
    path: Path
    csv_name: str
    subject: str
    videos: Tuple[int, ...]
    batch: str
    offsets: Dict[int, np.ndarray]      # channel -> array(n_tp)
    exponents: Dict[int, np.ndarray]
    n_tp: int
    n_freq: int

    @property
    def video(self) -> Optional[int]:
        return self.videos[0] if len(self.videos) == 1 else None


def parse_sprint_log(path: Path) -> SprintLog:
    text = path.read_text(encoding="utf-8", errors="replace")
    m = _HEADER_FILE_RE.search(text)
    if not m:
        raise ValueError(f"no 文件 header in {path}")
    csv_name = m.group(1).strip()
    sid = csv_name.split("_")[0]
    videos = tuple(sorted({int(v) for v in re.findall(r"video(\d+)", csv_name.lower())}))
    m2 = _HEADER_SID_RE.search(text)
    batch = m2.group(1).strip() if m2 else ""

    # frequency bins from the shape line: (n_timepoints, n_freqs)
    ms = re.search(r"数据加载成功，形状:\s*\((\d+),\s*(\d+)\)", text)
    n_tp = int(ms.group(1)) if ms else -1
    n_freq = int(ms.group(2)) if ms else -1

    offsets: Dict[int, np.ndarray] = {}
    exponents: Dict[int, np.ndarray] = {}
    pos = [(int(mm.group(1)), mm.end()) for mm in _CHAN_BLOCK_RE.finditer(text)]
    for i, (ch, start) in enumerate(pos):
        end = pos[i + 1][1] if i + 1 < len(pos) else len(text)
        block = text[start:end]
        # cut the per-channel peak listing to avoid the "峰值 k" table
        cut = block.find("峰值 k:")
        if cut > 0:
            block = block[:cut]
        trips = _TP_RE.findall(block)
        if not trips:
            continue
        idx = np.array([int(t[0]) for t in trips], dtype=int)
        off = np.array([float(t[1]) for t in trips], dtype=float)
        exp = np.array([float(t[2]) for t in trips], dtype=float)
        n = int(idx.max()) + 1
        o = np.full(n, np.nan)
        e = np.full(n, np.nan)
        o[idx] = off
        e[idx] = exp
        offsets[ch] = o
        exponents[ch] = e
    return SprintLog(path, csv_name, sid, videos, batch, offsets, exponents, n_tp, n_freq)


@dataclass
class PupilLog:
    path: Path
    subject: str
    video: int
    declared_dur: Optional[int]
    t: np.ndarray
    y: np.ndarray
    header: Dict[str, str]

    @property
    def n(self) -> int:
        return int(self.t.size)


_PUPIL_NAME_RE = re.compile(r"pupil_size_data_([A-Za-z0-9]+)_Video(\d+)_SMP_(\d+)s", re.I)


def parse_pupil_log(path: Path) -> Optional[PupilLog]:
    m = _PUPIL_NAME_RE.search(path.name)
    if not m:
        return None
    sid, vid, dur = m.group(1), int(m.group(2)), int(m.group(3))
    header = {k.strip(): v.strip() for k, v in _PUPIL_HDR_RE.findall(
        path.read_text(encoding="utf-8", errors="replace")[:2000])}
    t, y = [], []
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 2:
                continue
            try:
                t.append(float(parts[0]))
                y.append(float(parts[1]))
            except ValueError:
                continue
    return PupilLog(path, sid, vid, dur, np.array(t), np.array(y), header)


# --------------------------------------------------------------------------------------
# Regions
# --------------------------------------------------------------------------------------
def load_regions() -> Dict[int, str]:
    """Parse EGI_HydroCel_129_brain_regions.txt into {E-number(int): region}."""
    mapping: Dict[int, str] = {}
    if not REGION_TXT.exists():
        return mapping
    name_map = {
        "前额叶": "prefrontal", "额叶": "frontal", "中央区": "central",
        "顶叶": "parietal", "颞叶": "temporal", "枕叶": "occipital",
    }
    text = REGION_TXT.read_text(encoding="utf-8", errors="replace")
    cur = None
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        for zh, en in name_map.items():
            if zh in s and "E" not in s:
                cur = en
                break
        if cur is None:
            continue
        for e in re.findall(r"E(\d+)", s):
            mapping.setdefault(int(e), cur)
    return mapping


def build_region_mapping() -> Dict[int, str]:
    """Fallback: hard-coded mapping identical to pac_region_pairwise_stats_violin.py."""
    return {
        **{c: "prefrontal" for c in [1, 8, 14, 15, 21, 25, 22, 26, 32, 33, 38, 39, 43, 44, 49,
                                     2, 9, 113, 114, 115, 120, 121, 122]},
        **{c: "frontal" for c in [6, 11, 16, 18, 23, 27, 34, 40, 3, 10, 116, 123]},
        **{c: "central" for c in [7, 31, 55, 80, 106, 12, 13, 19, 20, 24, 28, 29, 30, 35, 36,
                                  37, 4, 5, 104, 105, 110, 111, 112, 117, 118, 124]},
        **{c: "parietal" for c in [62, 72, 41, 42, 46, 47, 51, 52, 53, 54, 59, 60, 61, 84, 85,
                                   86, 87, 91, 92, 93, 97, 98, 102, 103]},
        **{c: "temporal" for c in [45, 50, 56, 57, 58, 63, 64, 65, 68, 69, 94, 95, 96, 99, 100,
                                   101, 107, 108, 109]},
        **{c: "occipital" for c in [70, 71, 75, 76, 77, 78, 79, 81, 82, 83, 66, 67, 88, 89, 90]},
    }


# --------------------------------------------------------------------------------------
# Inventory
# --------------------------------------------------------------------------------------
def sha1_of(paths: Sequence[Path]) -> str:
    h = hashlib.sha1()
    for p in sorted(paths):
        h.update(p.name.encode("utf-8"))
        h.update(str(p.stat().st_size).encode("utf-8"))
    return h.hexdigest()


def stage_inventory(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    logs = sorted(EEG_LOG_DIR.glob("SPRiNT_parameters_*.log"))
    recs = []
    parsed: List[SprintLog] = []
    for p in logs:
        sl = parse_sprint_log(p)
        parsed.append(sl)
        recs.append({
            "log_file": p.name, "csv_name": sl.csv_name, "subject_id": sl.subject,
            "video_tokens": "+".join(map(str, sl.videos)), "batch": sl.batch,
            "n_channels": len(sl.offsets), "n_timepoints": sl.n_tp, "n_freq_bins": sl.n_freq,
            "n_tp_parsed": (len(next(iter(sl.offsets.values()))) if sl.offsets else 0),
        })
    eeg = pd.DataFrame(recs)

    pupil_recs = []
    pupils: Dict[Tuple[str, int], PupilLog] = {}
    for p in sorted(PUPIL_DIR.glob("*.log")):
        pl = parse_pupil_log(p)
        if pl is None:
            continue
        pupils[(pl.subject, pl.video)] = pl
        dt = np.diff(pl.t)
        pupil_recs.append({
            "pupil_file": p.name, "subject_id": pl.subject, "video": pl.video,
            "declared_dur_s": pl.declared_dur, "n_samples": pl.n,
            "t_min": float(pl.t.min()) if pl.n else np.nan,
            "t_max": float(pl.t.max()) if pl.n else np.nan,
            "dt_median": float(np.median(dt)) if dt.size else np.nan,
            "dt_max": float(dt.max()) if dt.size else np.nan,
            "n_nan": int(np.isnan(pl.y).sum()),
        })
    pup = pd.DataFrame(pupil_recs)
    eeg.to_csv(out_dir / "inventory_eeg.csv", index=False, encoding="utf-8-sig")
    pup.to_csv(out_dir / "inventory_pupil.csv", index=False, encoding="utf-8-sig")

    print(f"SPRiNT logs: {len(eeg)}  subjects: {eeg.subject_id.nunique()}")
    print(eeg.groupby("video_tokens").size().to_string())
    amb = eeg[eeg.video_tokens.str.contains(r"\+")]
    print(f"ambiguous video tokens: {len(amb)}")
    if len(amb):
        print(amb[["csv_name", "video_tokens", "n_channels", "n_timepoints"]].to_string(index=False))
    print(f"pupil logs: {len(pup)}  subjects: {pup.subject_id.nunique()}")
    print(pup.groupby("video").size().to_string())
    print(f"elapsed {time.time()-t0:.1f}s")


# --------------------------------------------------------------------------------------
# Cohort assembly / native pairing
# --------------------------------------------------------------------------------------
def load_sprint_logs() -> Dict[Tuple[str, int], SprintLog]:
    """Return {(subject, video): log}, resolving the 3 mislabelled 'video3_video1' files."""
    by_subject: Dict[str, List[SprintLog]] = {}
    for p in sorted(EEG_LOG_DIR.glob("SPRiNT_parameters_*.log")):
        sl = parse_sprint_log(p)
        by_subject.setdefault(sl.subject, []).append(sl)

    out: Dict[Tuple[str, int], SprintLog] = {}
    problems = []
    for sid, logs in by_subject.items():
        fixed = [sl for sl in logs if sl.video is not None]
        amb = [sl for sl in logs if sl.video is None]
        used = {sl.video for sl in fixed}
        for sl in fixed:
            key = (sid, sl.video)
            if key in out:
                problems.append(f"duplicate {key} in {sl.csv_name}")
            out[key] = sl
        for sl in amb:
            missing = [v for v in (1, 2, 3) if v not in used]
            if len(missing) != 1:
                problems.append(f"cannot resolve {sl.csv_name} tokens={sl.videos}")
                continue
            used.add(missing[0])
            sl.videos = (missing[0],)
            out[(sid, missing[0])] = sl
    if problems:
        for msg in problems:
            print("WARN:", msg)
    return out


def load_pupils() -> Dict[Tuple[str, int], PupilLog]:
    out: Dict[Tuple[str, int], PupilLog] = {}
    for p in sorted(PUPIL_DIR.glob("*.log")):
        pl = parse_pupil_log(p)
        if pl is None:
            continue
        out[(pl.subject, pl.video)] = pl
    return out


def pair_pupil_to_params(pl: PupilLog, n_tp: int,
                         tol_direct: float = 0.5, max_gap: float = 2.0
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Put pupil on the native SPRiNT parameter grid.

    Param timepoint k (0-based) is centred at (k+1) s: Lwin=2 s, 50 % overlap,
    WinAverage=0 -> window centres 1, 2, 3, ... s (SPRiNT_py_onlyo1.py).

    Returns (values, valid, interpolated).  A direct sample is used when a pupil
    sample lies within +-tol_direct s; otherwise linear interpolation is used
    only when the bracketing gap is <= max_gap s; otherwise the point is invalid.
    """
    centers = np.arange(n_tp, dtype=float) + 1.0
    t, y = pl.t, pl.y
    ok = np.isfinite(t) & np.isfinite(y)
    t, y = t[ok], y[ok]
    vals = np.full(n_tp, np.nan)
    valid = np.zeros(n_tp, dtype=bool)
    interp = np.zeros(n_tp, dtype=bool)
    if t.size == 0:
        return vals, valid, interp
    right = np.searchsorted(t, centers, side="left")
    left = np.clip(right - 1, 0, t.size - 1)
    right = np.clip(right, 0, t.size - 1)
    d_left = np.abs(centers - t[left])
    d_right = np.abs(centers - t[right])
    use_left = d_left <= d_right
    d_near = np.where(use_left, d_left, d_right)
    v_near = np.where(use_left, y[left], y[right])
    direct = d_near <= tol_direct
    vals[direct] = v_near[direct]
    valid[direct] = True

    need = ~direct
    if np.any(need):
        gap = t[right] - t[left]
        brack = need & (right != left) & (gap <= max_gap)
        if np.any(brack):
            w = (centers[brack] - t[left[brack]]) / (t[right[brack]] - t[left[brack]])
            vals[brack] = y[left[brack]] + w * (y[right[brack]] - y[left[brack]])
            valid[brack] = True
            interp[brack] = True
    # outside the observed pupil range -> invalid
    outside = (centers < t[0] - tol_direct) | (centers > t[-1] + tol_direct)
    valid[outside] = False
    return vals, valid, interp


def run_info(mask: np.ndarray) -> Tuple[int, List[Tuple[int, int]]]:
    best, cur, runs, start = 0, 0, [], None
    for i, v in enumerate(mask):
        if v:
            if cur == 0:
                start = i
            cur += 1
            best = max(best, cur)
        else:
            if cur:
                runs.append((start, cur))
            cur = 0
    if cur:
        runs.append((start, cur))
    return best, runs


def acf_decay(x: np.ndarray, max_lag: int, thr: float = 1.0 / math.e) -> float:
    """First lag whose autocorrelation drops below thr; np.nan if it never does."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 5:
        return np.nan
    x = x - x.mean()
    denom = float(np.dot(x, x))
    if denom <= 0:
        return np.nan
    for lag in range(1, min(max_lag, x.size - 2) + 1):
        r = float(np.dot(x[:-lag], x[lag:]) / denom)
        if r < thr:
            return float(lag)
    return np.nan


def build_cohort(logs: Dict[Tuple[str, int], SprintLog],
                 pupils: Dict[Tuple[str, int], PupilLog],
                 pair_kw: Optional[dict] = None) -> Tuple[List[str], List[dict]]:
    """Subjects with all three conditions in both modalities."""
    pair_kw = pair_kw or {}
    sids = sorted({s for s, _ in logs})
    rows, keep = [], []
    for sid in sids:
        has = all((sid, v) in logs for v in (1, 2, 3))
        has_p = all((sid, v) in pupils for v in (1, 2, 3))
        rec = {"subject_id": sid, "eeg_complete": has, "pupil_complete": has_p}
        if has and has_p:
            recs = []
            for v in (1, 2, 3):
                sl, pl = logs[(sid, v)], pupils[(sid, v)]
                n_tp = len(next(iter(sl.offsets.values())))
                _, valid, interp = pair_pupil_to_params(pl, n_tp, **pair_kw)
                best, _ = run_info(valid)
                recs.append({"video": v, "n_tp": n_tp, "n_valid": int(valid.sum()),
                             "n_interp": int(interp.sum()), "max_run": best,
                             "pupil_n": pl.n, "pupil_dur": pl.declared_dur})
            rec["n_i"] = min(r["max_run"] for r in recs)
            for r in recs:
                rec[f"v{r['video']}_n_tp"] = r["n_tp"]
                rec[f"v{r['video']}_valid"] = r["n_valid"]
                rec[f"v{r['video']}_interp"] = r["n_interp"]
                rec[f"v{r['video']}_maxrun"] = r["max_run"]
            keep.append(sid)
        rows.append(rec)
    return keep, rows


# --------------------------------------------------------------------------------------
# Probe: geometry / quality only (no MI)
# --------------------------------------------------------------------------------------
def stage_probe(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    logs = load_sprint_logs()
    pupils = load_pupils()
    sids, rows = build_cohort(logs, pupils)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "probe_cohort.csv", index=False, encoding="utf-8-sig")

    tri = df[df.eeg_complete & df.pupil_complete].copy()
    print(f"complete tri-condition EEG+pupil: {len(tri)} subjects")
    print("\nn_tp per condition:")
    print(pd.DataFrame({f"v{v}": tri[f"v{v}_n_tp"].describe() for v in (1, 2, 3)}).to_string())
    print("\nn_i distribution:")
    print(tri.n_i.describe().to_string())
    for thr in (20, 25, 30, 40, 50, 60, 70, 80):
        print(f"  n_i >= {thr}: {int((tri.n_i >= thr).sum())}")
    print("\nmax_run per condition:")
    for v in (1, 2, 3):
        print(f"  v{v}: {tri[f'v{v}_maxrun'].describe().to_dict()}")
    print("\ninterpolated points per condition (median):")
    for v in (1, 2, 3):
        print(f"  v{v}: {tri[f'v{v}_interp'].median():.0f} / {tri[f'v{v}_n_tp'].median():.0f}")

    # ACF decay of pupil and of EEG parameters on the longest valid run
    acf_rows = []
    for sid in sids:
        for v in (1, 2, 3):
            sl, pl = logs[(sid, v)], pupils[(sid, v)]
            n_tp = len(next(iter(sl.offsets.values())))
            vals, valid, _ = pair_pupil_to_params(pl, n_tp)
            best, runs = run_info(valid)
            if best < 5:
                continue
            st = max(runs, key=lambda r: r[1])[0]
            sl_idx = valid.copy()
            mask = np.zeros_like(valid)
            mask[st:st + best] = True
            lag_p = acf_decay(vals[mask], max_lag=min(60, best - 2))
            lags_o, lags_e = [], []
            for ch, off in sl.offsets.items():
                o = off[st:st + best]
                e = sl.exponents[ch][st:st + best]
                if np.isfinite(o).all():
                    lags_o.append(acf_decay(o, max_lag=min(60, best - 2)))
                if np.isfinite(e).all():
                    lags_e.append(acf_decay(e, max_lag=min(60, best - 2)))
            acf_rows.append({
                "subject_id": sid, "video": v, "max_run": best,
                "lag_pupil": lag_p,
                "lag_offset_med": float(np.nanmedian(lags_o)) if lags_o else np.nan,
                "lag_exponent_med": float(np.nanmedian(lags_e)) if lags_e else np.nan,
            })
    acf = pd.DataFrame(acf_rows)
    acf.to_csv(out_dir / "probe_acf.csv", index=False, encoding="utf-8-sig")
    print("\nACF decay lag (1/e), median across cohort:")
    print(acf[["lag_pupil", "lag_offset_med", "lag_exponent_med"]].describe().to_string())
    print("\nquantiles:")
    for c in ["lag_pupil", "lag_offset_med", "lag_exponent_med"]:
        q = acf[c].quantile([0.5, 0.75, 0.9, 0.95, 1.0]).round(2).to_dict()
        print(f"  {c}: {q}")

    # channel completeness
    ch_rows = []
    for sid in sids:
        sets = [set(logs[(sid, v)].offsets) for v in (1, 2, 3)]
        inter = set.intersection(*sets)
        ch_rows.append({"subject_id": sid, "n_ch_common": len(inter),
                        "ch": ",".join(map(str, sorted(inter)))})
    chdf = pd.DataFrame(ch_rows)
    print("\ncommon channels across three conditions:")
    print(chdf.n_ch_common.value_counts().to_string())
    chdf.to_csv(out_dir / "probe_channels.csv", index=False, encoding="utf-8-sig")
    print(f"elapsed {time.time()-t0:.1f}s")


# ======================================================================================
# Frozen configuration -- written to analysis_config.json BEFORE any condition MI exists
# ======================================================================================
NOMINAL_VIDEO_S = {1: 100.0, 2: 280.0, 3: 120.0}
REGION_INDEX = {r: i for i, r in enumerate(REGION_ORDER)}

ANALYSIS_SPECS = [
    # analysis_id, window rule, k, estimator, surrogate?
    ("native_equal", "earliest_ni", 3, "ksg", True),
    ("native_equal_k5", "earliest_ni", 5, "ksg", True),
    ("native_equal_k10", "earliest_ni", 10, "ksg", True),
    ("position_early", "position_early", 3, "ksg", True),
    ("position_middle", "position_middle", 3, "ksg", True),
    ("position_late", "position_late", 3, "ksg", True),
    ("native_global_equal", "earliest_global", 3, "ksg", True),
    ("legacy_same_cohort", "earliest_ni", 3, "hist10", False),
]
POSITION_IDS = ["position_early", "position_middle", "position_late"]

CONFIG = {
    "analysis_version": "mi_robustness_v1",
    "protocol_file": str(PROTOCOL),
    "protocol_date": "2026-10-27",
    "conditions": {"negative": 1, "neutral": 2, "positive": 3},
    "n_min": 20,
    "n_min_rule": (
        "max(4 x median pupil ACF 1/e lag, ceil(2 x spectral_support/step)) = max(4x5, 4) = 20; "
        "exceeds every k; the shortest n_i in the cohort is 23, so no subject is dropped on "
        "length alone. Fixed before any condition MI was computed."
    ),
    "n_position": 80,
    "n_position_rule": (
        "protocol 5.1 nominal-80-s candidate expressed in native points (1 s step); >= n_min; "
        "evaluable in 33/38 tri-condition subjects (availability table in sensitivity_summary)"
    ),
    "n_global_rule": "n_global = min(n_i) over the native_equal cohort",
    "estimator_main": "KSG1 kNN (chebyshev metric); raw value kept, no clipping at 0",
    "estimator_legacy": (
        "10-bin 2D histogram on z-scored signals, log2, replicating "
        "pac_coupling_analysis_real_data.py::calculate_coupling_metrics"
    ),
    "k_main": 3,
    "k_sensitivity": [5, 10],
    "pairing": {"param_center_offset_s": 1.0, "tol_direct_s": 0.5, "max_gap_s": 2.0},
    "time_grid": {"step_s": 1.0, "window_s": 2.0, "spectral_support_s": 2.0},
    "min_shift_rule": (
        "m_c = max(ceil(spectral_support_s/step_s), ceil(ACF lag at which the ACF first falls "
        "below 1/e)) on the windowed pupil series; m = max over the three conditions of one "
        "subject; S = {s: 1<=s<=n-1 and min(s,n-s)>=m}; every unique shift is enumerated"
    ),
    "acf_fallback_rule": (
        "if the ACF never falls below 1/e within min(60, n-2) lags, "
        "m_c = min(n//2-1, max(2, ceil(n/4))) and acf_fallback=True is recorded"
    ),
    "tts_radius_rule": "r_i = floor((n_i - n_min)/2), identical for the three conditions",
    "tts_stationarity_rule": (
        "supported if |mean(first half) - mean(second half)| <= 1 SD and |OLS slope| * n <= 1 SD "
        "of the windowed pupil series"
    ),
    "target_mid_rule": (
        "nominal video midpoint (100/280/120 s -> 50/140/60 s); event records are unavailable and "
        "the duration token in pupil filenames is not a reliable video length (82-593 s observed)"
    ),
    "event_records_available": False,
    "negative_mi_rule": "kept as returned by each estimator; identical rule for real and surrogate",
    "region_statistic": "per-channel MI first, then arithmetic mean within region",
    "n_bootstrap": 2000,
    "seed": 20261027,
    "alpha": 0.05,
    "families": {
        "orig48_overall": {
            "planned": 48, "computed": 12,
            "note": "6 regions x 2 parameters x 4 metrics. Only mutual_information was computed; "
                    "p_raw is saved for the 12 computed tests and p_adjusted is left empty with "
                    "status=family_pending (protocol L174: unrun metrics are NOT filled with 1).",
        },
        "orig144_paired": {
            "planned": 144, "computed": 36,
            "note": "48 x 3 contrasts. p_raw saved, p_adjusted left empty (family_pending).",
        },
        "mi_only_12_overall": {
            "planned": 12, "computed": 12,
            "note": "BH-FDR over the 12 mutual-information overall tests of one setting; this is a "
                    "separate, explicitly labelled family, not the original 48-test family.",
        },
        "mi_only_36_paired": {
            "planned": 36, "computed": 36,
            "note": "BH-FDR over the 36 mutual-information paired tests of one setting; separate "
                    "from the original 144-test family.",
        },
        "mi_excess12_overall": {"planned": 12, "computed": 12, "note": "complete family"},
        "mi_excess36_paired": {"planned": 36, "computed": 36, "note": "complete family"},
    },
    "family_rule": (
        "Original paper families (48 overall / 144 paired) are reported as p_raw only and marked "
        "family_pending; BH is additionally applied within the computed MI-only 12/36 subfamilies "
        "and labelled as such. Corrections are within one setting; cross-setting selection is not "
        "corrected. Rows that cannot be computed stay empty (NaN), never 1, in the delivered tables."
    ),
    "bh_nan_rule": (
        "for a family that is actually computed, non-computable p values are internally replaced "
        "by 1 so that m is preserved; the number of such placeholders is written into the reason "
        "field of every row of that family. Families with no computable p at all are labelled "
        "<family>_not_computed and their adjusted p stays empty."
    ),
    "analysis_ids": [s[0] for s in ANALYSIS_SPECS],
}

# ======================================================================================
# MI estimators
# ======================================================================================
def zscore(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    s = v.std()
    return v - v.mean() if s == 0 else (v - v.mean()) / s


def legacy_hist_mi_channels(X: np.ndarray, y: np.ndarray, bins: int = 10) -> np.ndarray:
    """Vectorised replication of the submitted 10-bin histogram MI (all channels at once).

    The original code z-scores both signals, calls
    ``np.digitize(v, np.linspace(v.min(), v.max(), bins))`` (which yields the values
    1..bins) and then ``np.histogram2d(..., bins=bins)`` on those integers, so the
    histogram range is re-derived from the digitised values.  That re-binning step is
    reproduced exactly below.
    """
    C, n = X.shape
    out = np.full(C, np.nan)
    if n < 10:
        return out
    Xz = np.vstack([zscore(r) for r in X])
    yz = zscore(y)
    ok = (Xz.max(1) > Xz.min(1)) & (yz.max() > yz.min())
    if not ok.any():
        return out

    frac = np.linspace(0.0, 1.0, bins)
    xlo = Xz.min(1)[:, None]
    xhi = Xz.max(1)[:, None]
    ex = xlo + (xhi - xlo) * frac[None, :]                            # (C, bins)
    xb = (Xz[:, :, None] >= ex[:, None, :]).sum(2).astype(float)      # 1..bins
    ey = yz.min() + (yz.max() - yz.min()) * frac
    yb = (yz[:, None] >= ey[None, :]).sum(1).astype(float)            # 1..bins

    def rebin(v: np.ndarray) -> np.ndarray:
        lo = v.min(1, keepdims=True)
        hi = v.max(1, keepdims=True)
        rng = np.where(hi > lo, hi - lo, 1.0)
        return np.clip(np.floor((v - lo) / rng * bins).astype(int), 0, bins - 1)

    bx = rebin(xb)
    by = rebin(yb[None, :])[0]
    codes = bx * bins + by[None, :]
    off = (np.arange(C) * bins * bins)[:, None]
    flat = np.bincount((codes + off).ravel(), minlength=C * bins * bins).reshape(C, bins, bins)
    p = flat / float(n)
    pi = p.sum(2)
    pj = p.sum(1)
    den = pi[:, :, None] * pj[:, None, :]
    nz = p > 0
    ratio = np.ones_like(p)
    ratio[nz] = p[nz] / den[nz]
    mi = np.where(nz, p * np.log2(ratio), 0.0).sum((1, 2))
    out[ok] = mi[ok]
    return out


def _ksg_from_Dx(Dx: np.ndarray, y: np.ndarray, k: int,
                 Dy: Optional[np.ndarray] = None) -> np.ndarray:
    """KSG estimator 1 (max-norm joint radius, strict-inequality marginal counts)."""
    C, n, _ = Dx.shape
    if Dy is None:
        Dy = np.abs(y[:, None] - y[None, :])
    D = np.maximum(Dx, Dy[None, :, :])
    i = np.arange(n)
    D[:, i, i] = np.inf
    eps = np.partition(D, k - 1, axis=2)[:, :, k - 1]
    nx = (Dx < eps[:, :, None]).sum(2) - 1
    ny = (Dy[None, :, :] < eps[:, :, None]).sum(2) - 1
    return digamma(k) - np.mean(digamma(nx + 1) + digamma(ny + 1), axis=1) + digamma(n)


def mi_channels(X: np.ndarray, y: np.ndarray, estimator: str, k: int,
                shifts: Sequence[int] = ()) -> Tuple[np.ndarray, np.ndarray]:
    """Per-channel observed MI and per-shift surrogate MI.

    Returns ``(obs (C,), sur (len(shifts), C))``.  Constant channels and points shorter
    than the estimator minimum return NaN: never 0.
    """
    C, n = X.shape
    obs = np.full(C, np.nan)
    sur = np.full((len(shifts), C), np.nan)
    if n < max(k + 1, 11) or np.ptp(y) == 0:
        return obs, sur
    if estimator == "hist10":
        obs[:] = legacy_hist_mi_channels(X, y)
        for j, s in enumerate(shifts):
            sur[j] = legacy_hist_mi_channels(X, np.roll(y, int(s)))
        return obs, sur

    ok = (X.max(1) - X.min(1)) > 0
    if not ok.any():
        return obs, sur
    Xo = X[ok]
    Dx = np.abs(Xo[:, :, None] - Xo[:, None, :])
    Dy = np.abs(y[:, None] - y[None, :])
    kk = min(k, n - 1)
    obs[ok] = _ksg_from_Dx(Dx, y, kk, Dy)
    for j, s in enumerate(shifts):
        sur[j, ok] = _ksg_from_Dx(Dx, np.roll(y, int(s)), kk, Dy=None)
    return obs, sur


# ======================================================================================
# Window selection
# ======================================================================================
def valid_runs(valid: np.ndarray, n: int) -> List[Tuple[int, int]]:
    _, runs = run_info(valid)
    return [(s, l) for s, l in runs if l >= n]


def pick_start(valid: np.ndarray, n: int, mode: str,
               target_mid: float = np.nan) -> Optional[int]:
    rs = valid_runs(valid, n)
    if not rs:
        return None
    if mode in ("early", "earliest_ni", "earliest_global"):
        return int(rs[0][0])
    if mode == "late":
        s, l = rs[-1]
        return int(s + l - n)
    cands = np.concatenate([np.arange(s, s + l - n + 1) for s, l in rs])
    if cands.size == 0:
        return None
    mids = cands + n / 2.0
    j = int(np.argmin(np.abs(mids - target_mid)))   # argmin -> earliest on ties
    return int(cands[j])


def count_windows(valid: np.ndarray, n: int) -> int:
    return int(sum(l - n + 1 for _, l in valid_runs(valid, n)))


# ======================================================================================
# Subject-level input assembly
# ======================================================================================
def load_subject_block(sid: str, logs: Dict[Tuple[str, int], SprintLog],
                       pupils: Dict[Tuple[str, int], PupilLog],
                       chans: Sequence[int]) -> Dict[str, dict]:
    block: Dict[str, dict] = {}
    chans = list(chans)
    for cond in CONDITIONS:
        v = COND_VIDEO[cond]
        sl, pl = logs[(sid, v)], pupils[(sid, v)]
        n_tp = int(next(iter(sl.offsets.values())).size)
        vals, pvalid, pinterp = pair_pupil_to_params(pl, n_tp)
        off = np.vstack([sl.offsets[c] for c in chans])
        exp = np.vstack([sl.exponents[c] for c in chans])
        finite = np.isfinite(off).all(0) & np.isfinite(exp).all(0)
        block[cond] = {
            "video": v, "n_tp": n_tp, "pupil": vals, "pvalid": pvalid,
            "pinterp": pinterp, "valid": pvalid & finite, "off": off, "exp": exp,
        }
    return block


def shift_set(n: int, m: int) -> List[int]:
    return [s for s in range(1, n) if min(s, n - s) >= m]


def subject_min_shift(y: np.ndarray) -> Tuple[int, bool]:
    """m_c from the windowed pupil series (ACF 1/e lag) and the spectral support."""
    n = y.size
    base = int(math.ceil(CONFIG["time_grid"]["spectral_support_s"] /
                         CONFIG["time_grid"]["step_s"]))
    lag = acf_decay(y, max_lag=min(60, max(n - 2, 1)))
    if not np.isfinite(lag):
        fallback = int(min(n // 2 - 1, max(2, math.ceil(n / 4))))
        return max(base, fallback), True
    return max(base, int(math.ceil(lag))), False


def stationarity_ok(y: np.ndarray) -> Tuple[bool, float, float]:
    y = np.asarray(y, float)
    n = y.size
    sd = float(np.std(y))
    if n < 6 or sd == 0:
        return False, np.nan, np.nan
    h = n // 2
    split = abs(float(np.mean(y[:h]) - np.mean(y[h:]))) / sd
    t = np.arange(n, dtype=float)
    slope = float(np.polyfit(t, y, 1)[0])
    drift = abs(slope) * n / sd
    return bool(split <= 1.0 and drift <= 1.0), split, drift


# ======================================================================================
# Statistics helpers
# ======================================================================================
def bh_fdr(pvals: Sequence[float], alpha: float = 0.05) -> np.ndarray:
    """BH-FDR; NaN entries are replaced by 1.0 so that m is preserved."""
    p = np.asarray(pvals, dtype=float)
    m = p.size
    if m == 0:
        return p
    p = np.where(np.isfinite(p), p, 1.0)
    order = np.argsort(p)
    ranked = p[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(ranked, 1.0)
    return out


def boot_median_ci(d: np.ndarray, rng: np.random.Generator,
                   B: int = 2000) -> Tuple[float, float]:
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if d.size < 3:
        return np.nan, np.nan
    idx = rng.integers(0, d.size, size=(B, d.size))
    meds = np.median(d[idx], axis=1)
    lo, hi = np.percentile(meds, [2.5, 97.5])
    return float(lo), float(hi)


def paired_test(a: np.ndarray, b: np.ndarray, rng: np.random.Generator
                ) -> Dict[str, object]:
    m = np.isfinite(a) & np.isfinite(b)
    a, b = np.asarray(a, float)[m], np.asarray(b, float)[m]
    d = a - b
    res = {
        "n_subjects": int(d.size), "test": "wilcoxon_signed_rank",
        "statistic": np.nan, "p_raw": np.nan, "median_paired_difference": np.nan,
        "ci_low": np.nan, "ci_high": np.nan, "n_zero_difference": int((d == 0).sum()),
        "n_ties_abs": int(d.size - np.unique(np.abs(d)).size),
        "method": "auto", "status": "ok", "reason": "",
    }
    if d.size < 3:
        res.update(status="not_evaluable", reason="fewer than 3 complete pairs")
        return res
    if np.all(d == 0):
        res.update(status="all_zero_differences",
                   reason="every paired difference is exactly zero")
        res["median_paired_difference"] = 0.0
        res["ci_low"] = res["ci_high"] = 0.0
        return res
    try:
        w = stats.wilcoxon(a, b, zero_method="wilcox", correction=False,
                           alternative="two-sided", method="auto")
        res["statistic"] = float(w.statistic)
        res["p_raw"] = float(w.pvalue)
    except Exception as exc:  # pragma: no cover - defensive
        res.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
    res["median_paired_difference"] = float(np.median(d))
    lo, hi = boot_median_ci(d, rng)
    res["ci_low"], res["ci_high"] = lo, hi
    return res


def friedman_test(groups: Sequence[np.ndarray]) -> Dict[str, object]:
    cols = [np.asarray(g, float) for g in groups]
    m = np.ones(cols[0].size, bool)
    for c in cols:
        m &= np.isfinite(c)
    cols = [c[m] for c in cols]
    res = {"n_subjects": int(cols[0].size), "test": "friedman",
           "statistic": np.nan, "p_raw": np.nan, "effect_name": "kendall_W",
           "effect_value": np.nan, "status": "ok", "reason": ""}
    if cols[0].size < 5:
        res.update(status="not_evaluable", reason="fewer than 5 complete subjects")
        return res
    stacked = np.vstack(cols)
    if np.all(stacked == stacked[0, 0]):
        res.update(status="constant", reason="all values identical across conditions")
        return res
    try:
        st = stats.friedmanchisquare(*cols)
        res["statistic"] = float(st.statistic)
        res["p_raw"] = float(st.pvalue)
        res["effect_value"] = float(st.statistic / (cols[0].size * (len(cols) - 1)))
    except Exception as exc:  # pragma: no cover
        res.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
    return res


# ======================================================================================
# run stage
# ======================================================================================
SURROGATE_COLS = ["analysis_id", "subject_id", "condition", "region", "parameter",
                  "shift_points", "shift_s", "mi_value", "n_points", "n_channels",
                  "k_neighbors"]
CHANNEL_COLS = ["analysis_id", "subject_id", "condition", "region", "channel",
                "parameter", "metric", "value", "n_points", "k_neighbors",
                "status", "reason"]
SEGMENT_COLS = ["analysis_id", "subject_id", "condition", "video", "channel_set_id",
                "start_index", "start_center_s", "end_center_s", "support_start_s",
                "support_end_s", "center_span_s", "n_points", "step_s",
                "valid_fraction", "interpolated_fraction", "position", "target_mid_s",
                "actual_mid_s", "duplicate_window", "shared_frac_vs_early",
                "shared_frac_vs_middle", "shared_frac_vs_late", "n_channels",
                "min_shift_points", "acf_fallback", "stationarity_supported",
                "half_mean_diff_sd", "drift_sd", "n_shifts", "status", "reason"]
POSITION_NAME = {"earliest_ni": "earliest", "earliest_global": "earliest",
                 "position_early": "early", "position_middle": "middle",
                 "position_late": "late"}
CONTRASTS = [("negative", "neutral"), ("negative", "positive"), ("neutral", "positive")]

SUR_SUM_COLS = ["analysis_id", "subject_id", "condition", "region", "parameter", "metric",
                "n_points", "n_channels", "n_unique_shifts", "mi_observed",
                "surrogate_mean", "surrogate_median", "surrogate_q025", "surrogate_q975",
                "mi_excess", "status", "reason"]
SUR_QC_COLS = ["analysis_id", "subject_id", "condition", "n_points", "acf_lag_pupil",
               "min_shift_points_condition", "min_shift_points_subject", "acf_fallback",
               "n_unique_shifts", "shift_min", "shift_max", "stationarity_supported",
               "half_mean_diff_sd", "drift_sd", "status", "reason"]
THETA_COLS = ["analysis_id", "subject_id", "condition", "region", "parameter", "d",
              "theta", "is_zero_shift"]
TTS_COLS = ["analysis_id", "subject_id", "condition", "region", "parameter", "radius",
            "n_points_used", "theta_zero", "theta_min", "theta_max", "count_ge", "p_tts",
            "min_attainable_p", "compute_status", "data_valid", "assumptions_supported",
            "inference_status", "reason"]
COND_SUM_COLS = ["analysis_id", "condition", "region", "parameter", "metric", "n_subjects",
                 "mean", "sd", "median", "min", "max"]
TEST_COLS = ["analysis_id", "metric", "parameter", "region", "test", "contrast",
             "n_subjects", "statistic", "effect_value", "median_paired_difference",
             "ci_low", "ci_high", "n_zero_difference", "n_ties_abs", "p_raw",
             "family_id_orig", "family_planned_orig", "p_adjusted_orig", "status_orig",
             "family_id_mi", "p_adjusted_mi", "status", "reason"]
SENS_COLS = ["analysis_id", "baseline", "condition", "region", "parameter", "metric",
             "test", "n_subjects", "median_paired_difference", "ci_low", "ci_high",
             "p_raw", "p_adjusted", "status", "reason", "note"]
COUPLING_COLS = ["analysis_id", "subject_id", "condition", "region", "parameter", "metric",
                 "value", "n_points", "n_channels", "channel_set_id", "k_neighbors",
                 "status", "reason"]
COHORT_COLS = ["analysis_id", "subject_id", "cohort_id", "available", "n_points",
               "n_channels", "channel_set_id", "age", "sex", "age_group", "reason"]
MATCH_COLS = ["subject_id", "channel", "region", "channel_set_id", "n_channels_in_set"]


def _write_rows(path: Path, cols: Sequence[str], rows: Iterable[dict]) -> None:
    pd.DataFrame(list(rows), columns=list(cols)).to_csv(
        path, index=False, encoding="utf-8-sig")


def reg_mean(v: np.ndarray, masks: Sequence[np.ndarray],
             ok: Optional[np.ndarray] = None) -> np.ndarray:
    """Region statistic: per-channel value first, then arithmetic mean in the region."""
    out = np.full(len(masks), np.nan)
    v = np.asarray(v, float)
    for r, msk in enumerate(masks):
        use = msk if ok is None else (msk & ok)
        if use.any():
            vals = v[use]
            vals = vals[np.isfinite(vals)]
            if vals.size:
                out[r] = float(vals.mean())
    return out


def reg_mean_many(V: np.ndarray, masks: Sequence[np.ndarray],
                  ok: Optional[np.ndarray] = None) -> np.ndarray:
    V = np.asarray(V, float)
    out = np.full((V.shape[0], len(masks)), np.nan)
    for j in range(V.shape[0]):
        out[j] = reg_mean(V[j], masks, ok)
    return out


def tts_subject(ys: Dict[str, np.ndarray], offs: Dict[str, np.ndarray],
                exps: Dict[str, np.ndarray], masks: Sequence[np.ndarray],
                r: int, k: int) -> Dict[str, dict]:
    """TTS as fixed by protocol L101-114, computed at region level.

    ``theta[d]`` for d in -r..r uses x0 = x[r:n-r] and y[r+d:n-r+d]; theta[0] is the
    zero-shift MI on the truncated (n-2r) series.  The count uses >= so d=0 and ties are
    included, and the denominator is r+1 (not 2r+1).
    """
    out: Dict[str, dict] = {}
    for cond in CONDITIONS:
        y = np.asarray(ys[cond], float)
        n = y.size
        nd = 2 * r + 1
        theta = {}
        per_ch = {}
        for pname, X in (("offset", offs[cond]), ("exponent", exps[cond])):
            X = np.asarray(X, float)
            per = np.full((nd, X.shape[0]), np.nan)
            for j, d in enumerate(range(-r, r + 1)):
                o, _ = mi_channels(X[:, r:n - r], y[r + d:n - r + d], "ksg", k)
                per[j] = o
            per_ch[pname] = per
            theta[pname] = reg_mean_many(per, masks, np.all(np.isfinite(per), axis=0))
        out[cond] = {"theta": theta, "r": r, "n_used": n - 2 * r,
                     "stationarity": stationarity_ok(y[r:n - r])}
    return out


PUBLIC_FILE = Path(r"f:\eye_eeg\MIPDB_PublicFile.csv")


def load_demographics(path: Path = PUBLIC_FILE) -> Dict[str, dict]:
    """Age / sex lookup; missing values stay NaN, never imputed."""
    out: Dict[str, dict] = {}
    if not path.exists():
        return out
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        return out
    if "ID" not in df.columns:
        return out
    for _, r in df.iterrows():
        sid = str(r["ID"]).strip()
        age = pd.to_numeric(r.get("Age"), errors="coerce")
        sex = pd.to_numeric(r.get("Sex"), errors="coerce")
        rec = {"age": float(age) if pd.notna(age) else np.nan,
               "sex": float(sex) if pd.notna(sex) else np.nan}
        rec["age_group"] = ("under_18" if rec["age"] < 18 else
                            "18_or_older" if np.isfinite(rec["age"]) else "")
        out[sid] = rec
    return out


def stage_run(out_dir: Path, only_settings: Optional[Sequence[str]] = None,
              only_subjects: Optional[Sequence[str]] = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    rng = np.random.default_rng(CONFIG["seed"])

    logs = load_sprint_logs()
    pupils = load_pupils()
    cohort, manifest_rows = build_cohort(logs, pupils)
    if only_subjects:
        cohort = [s for s in cohort if s in set(only_subjects)]
    specs = [s for s in ANALYSIS_SPECS if not only_settings or s[0] in set(only_settings)]
    print(f"[run] subjects: {len(cohort)}  settings: {[s[0] for s in specs]}", flush=True)

    regions = build_region_mapping()
    region_idx_of = lambda ch: REGION_INDEX[regions[ch]]
    region_masks: Dict[int, List[np.ndarray]] = {}

    # ---- per subject: channel set and loaded block ----
    blocks: Dict[str, dict] = {}
    chans_of: Dict[str, List[int]] = {}
    for sid in cohort:
        ch = sorted(set(logs[(sid, 1)].offsets) & set(logs[(sid, 2)].offsets)
                    & set(logs[(sid, 3)].offsets) & set(regions))
        chans_of[sid] = ch
        blocks[sid] = load_subject_block(sid, logs, pupils, ch)
        region_masks[sid] = [np.array([region_idx_of(c) == r for c in ch]) for r in range(6)]
        nitems = [int(run_info(blocks[sid][c]["valid"])[0]) for c in CONDITIONS]
        blocks[sid]["n_i"] = int(min(nitems))
        blocks[sid]["m_cond"] = dict(zip(CONDITIONS, nitems))
    n_global = int(min(blocks[s]["n_i"] for s in cohort)) if cohort else 0
    print(f"[run] n_global (min n_i) = {n_global}", flush=True)

    cfg = dict(CONFIG)
    cfg.update({
        "n_global": n_global,
        "n_subjects_tri_condition": len(cohort),
        "input_hash_sha1": sha1_of(sorted(EEG_LOG_DIR.glob("SPRiNT_parameters_*.log"))
                                   + sorted(PUPIL_DIR.glob("*.log"))),
        "python": sys.version.split()[0], "numpy": np.__version__,
        "pandas": pd.__version__, "scipy": __import__("scipy").__version__,
        "platform": platform.platform(),
        "code_file": str(Path(__file__).resolve()),
        "code_sha1": hashlib.sha1(Path(__file__).read_bytes()).hexdigest(),
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
    })
    (out_dir / "analysis_config.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- streaming writers for the two large tables ----
    f_sur = open(out_dir / "surrogate_values.csv", "w", newline="", encoding="utf-8-sig")
    w_sur = csv.writer(f_sur)
    w_sur.writerow(SURROGATE_COLS)
    f_ch = open(out_dir / "coupling_channel_values.csv", "w", newline="",
                encoding="utf-8-sig")
    w_ch = csv.writer(f_ch)
    w_ch.writerow(CHANNEL_COLS)
    f_th = open(out_dir / "tts_theta.csv", "w", newline="", encoding="utf-8-sig")
    w_th = csv.writer(f_th)
    w_th.writerow(THETA_COLS)
    f_log = open(out_dir / "run_log.txt", "w", encoding="utf-8")

    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        f_log.write(line + "\n")
        f_log.flush()

    segments, coupling_values, sur_qc, sur_summary = [], [], [], []
    matched_channels, cohort_rows, tts_rows = [], [], []
    results: Dict[str, Dict[Tuple[str, str], dict]] = {}
    obs_by_setting: Dict[str, dict] = {}
    ids_by_setting: Dict[str, List[str]] = {}
    excl_by_setting: Dict[str, List[Tuple[str, str]]] = {}
    pos_starts: Dict[Tuple[str, str, str], int] = {}
    starts_all: Dict[Tuple[str, str, str], int] = {}
    npz_payload: Dict[str, np.ndarray] = {}
    sur_draw: Dict[Tuple[str, str], List[float]] = {}   # figure only (native_equal, prefrontal)

    def push_segment(aid, sid, cond, n_use, mode, status, reason, **kw):
        rows = {
            "analysis_id": aid, "subject_id": sid, "condition": cond,
            "video": COND_VIDEO[cond], "channel_set_id": f"{sid}_{len(chans_of[sid])}ch",
            "start_index": np.nan, "start_center_s": np.nan, "end_center_s": np.nan,
            "support_start_s": np.nan, "support_end_s": np.nan, "center_span_s": np.nan,
            "n_points": n_use, "step_s": CONFIG["time_grid"]["step_s"],
            "valid_fraction": np.nan, "interpolated_fraction": np.nan,
            "position": POSITION_NAME.get(mode, mode),
            "target_mid_s": NOMINAL_VIDEO_S[COND_VIDEO[cond]] / 2.0,
            "actual_mid_s": np.nan, "duplicate_window": "", "n_channels": len(chans_of[sid]),
            "status": status, "reason": reason,
        }
        rows.update(kw)
        segments.append(rows)

    for spec in specs:
        aid, mode, k, est, do_sur = spec
        t_set = time.time()
        results[aid] = {}
        eval_ids, excl = [], []
        for si, sid in enumerate(cohort):
            blk = blocks[sid]
            ch = chans_of[sid]
            masks = region_masks[sid]
            n_i = blk["n_i"]
            if mode == "earliest_ni":
                n_use = n_i
            elif mode == "earliest_global":
                n_use = n_global
            else:
                n_use = CONFIG["n_position"]
            reason = ""
            if n_use < CONFIG["n_min"]:
                reason = f"n required ({n_use}) < n_min ({CONFIG['n_min']})"
            starts = {}
            if not reason:
                for cond in CONDITIONS:
                    v = COND_VIDEO[cond]
                    tgt = NOMINAL_VIDEO_S[v] / 2.0
                    st = pick_start(blk[cond]["valid"], n_use,
                                    {"earliest_ni": "early",
                                     "earliest_global": "early"}.get(mode, mode), tgt)
                    if st is None:
                        reason = f"{cond}: no contiguous {n_use}-point valid window"
                        break
                    starts[cond] = st
            if reason:
                for cond in CONDITIONS:
                    push_segment(aid, sid, cond, n_use, mode, "not_evaluable", reason)
                cohort_rows.append({"analysis_id": aid, "subject_id": sid,
                                    "cohort_id": f"{aid}_cohort", "available": False,
                                    "n_points": n_use, "n_channels": len(ch),
                                    "channel_set_id": f"{sid}_{len(ch)}ch",
                                    "reason": reason})
                excl.append((sid, reason))
                continue
            eval_ids.append(sid)
            cohort_rows.append({"analysis_id": aid, "subject_id": sid,
                                "cohort_id": f"{aid}_cohort", "available": True,
                                "n_points": n_use, "n_channels": len(ch),
                                "channel_set_id": f"{sid}_{len(ch)}ch", "reason": ""})
            for cond in CONDITIONS:
                starts_all[(aid, sid, cond)] = starts[cond]
                if mode in POSITION_IDS:
                    pos_starts[(aid, sid, cond)] = starts[cond]

            # ---- per-condition extraction; n_use and the shift set are shared ----
            cond_dat: Dict[str, dict] = {}
            m_used, acf_fb = 0, False
            for cond in CONDITIONS:
                v = COND_VIDEO[cond]
                st = starts[cond]
                sl_idx = slice(st, st + n_use)
                y = blk[cond]["pupil"][sl_idx]
                m_c, fb = subject_min_shift(y)
                m_used = max(m_used, m_c)
                acf_fb = acf_fb or fb
                lag = acf_decay(y, max_lag=min(60, max(n_use - 2, 1)))
                ok_st, split, drift = stationarity_ok(y)
                cond_dat[cond] = {
                    "video": v, "start": st, "y": y, "acf_lag": lag, "m_c": m_c,
                    "fallback": fb, "ok_st": ok_st, "split": split, "drift": drift,
                    "off": blk[cond]["off"][:, sl_idx],
                    "exp": blk[cond]["exp"][:, sl_idx],
                    "pvalid": blk[cond]["pvalid"][sl_idx],
                    "pinterp": blk[cond]["pinterp"][sl_idx],
                }

            shifts = shift_set(n_use, m_used) if do_sur else []
            if do_sur and not shifts:
                reason = f"no admissible cyclic shift (m={m_used}, n={n_use})"
                for cond in CONDITIONS:
                    d = cond_dat[cond]
                    push_segment(aid, sid, cond, n_use, mode, "not_evaluable", reason,
                                 start_index=d["start"], start_center_s=d["start"] + 1.0,
                                 end_center_s=d["start"] + n_use,
                                 support_start_s=d["start"],
                                 support_end_s=d["start"] + n_use + 1.0,
                                 center_span_s=n_use - 1,
                                 valid_fraction=float(np.mean(d["pvalid"])),
                                 interpolated_fraction=float(np.mean(d["pinterp"])),
                                 actual_mid_s=d["start"] + n_use / 2.0,
                                 min_shift_points=m_used, acf_fallback=bool(acf_fb),
                                 stationarity_supported=bool(d["ok_st"]),
                                 half_mean_diff_sd=d["split"], drift_sd=d["drift"],
                                 n_shifts=0)
                cohort_rows[-1].update({"available": False, "reason": reason})
                for cond in CONDITIONS:
                    starts_all.pop((aid, sid, cond), None)
                    pos_starts.pop((aid, sid, cond), None)
                eval_ids.pop()
                excl.append((sid, reason))
                continue

            obs_by, sur_by = {}, {}
            for cond in CONDITIONS:
                d = cond_dat[cond]
                st = d["start"]
                o_off, s_off = mi_channels(d["off"], d["y"], est, k, shifts)
                o_exp, s_exp = mi_channels(d["exp"], d["y"], est, k, shifts)
                obs_by[cond] = {"offset": reg_mean(o_off, masks),
                                "exponent": reg_mean(o_exp, masks)}
                sur_by[cond] = {"offset": reg_mean_many(s_off, masks),
                                "exponent": reg_mean_many(s_exp, masks)}

                push_segment(aid, sid, cond, n_use, mode, "ok", "",
                             start_index=st, start_center_s=st + 1.0,
                             end_center_s=st + n_use, support_start_s=st,
                             support_end_s=st + n_use + 1.0, center_span_s=n_use - 1,
                             valid_fraction=float(np.mean(d["pvalid"])),
                             interpolated_fraction=float(np.mean(d["pinterp"])),
                             actual_mid_s=st + n_use / 2.0,
                             min_shift_points=m_used, acf_fallback=bool(acf_fb),
                             stationarity_supported=bool(d["ok_st"]),
                             half_mean_diff_sd=d["split"], drift_sd=d["drift"],
                             n_shifts=len(shifts))

                for pname, chv in (("offset", o_off), ("exponent", o_exp)):
                    for ci, c in enumerate(ch):
                        val = chv[ci]
                        w_ch.writerow([aid, sid, cond, REGION_ORDER[region_idx_of(c)],
                                       c, pname, "mutual_information",
                                       f"{val:.10g}" if np.isfinite(val) else "",
                                       n_use, k if est == "ksg" else "",
                                       "ok" if np.isfinite(val) else "not_evaluable",
                                       "" if np.isfinite(val) else "constant_or_short"])
                    for r in range(6):
                        val = obs_by[cond][pname][r]
                        coupling_values.append({
                            "analysis_id": aid, "subject_id": sid, "condition": cond,
                            "region": REGION_ORDER[r], "parameter": pname,
                            "metric": "mutual_information", "value": val,
                            "n_points": n_use, "n_channels": int(masks[r].sum()),
                            "channel_set_id": f"{sid}_{len(ch)}ch",
                            "k_neighbors": k if est == "ksg" else "hist10",
                            "status": "ok" if np.isfinite(val) else "not_evaluable",
                            "reason": "" if np.isfinite(val)
                            else "no finite channel value in region"})

                for pname, SV in (("offset", s_off), ("exponent", s_exp)):
                    RM = sur_by[cond][pname]
                    for j, s in enumerate(shifts):
                        for r in range(6):
                            val = RM[j, r]
                            w_sur.writerow([aid, sid, cond, REGION_ORDER[r], pname,
                                            int(s), s * CONFIG["time_grid"]["step_s"],
                                            f"{val:.10g}" if np.isfinite(val) else "",
                                            n_use, int(masks[r].sum()),
                                            k if est == "ksg" else "hist10"])

                if aid == "native_equal":
                    tag = f"{sid}|{cond}"
                    npz_payload[f"y|{tag}"] = d["y"]
                    npz_payload[f"t|{tag}"] = np.arange(n_use) + st + 1.0
                    npz_payload[f"off|{tag}"] = d["off"]
                    npz_payload[f"exp|{tag}"] = d["exp"]
                    npz_payload[f"pvalid|{tag}"] = d["pvalid"]
                    npz_payload[f"pinterp|{tag}"] = d["pinterp"]
                    npz_payload[f"start|{tag}"] = np.array([st], float)
                    npz_payload[f"ch|{sid}"] = np.array(ch)

                results[aid][(sid, cond)] = {
                    "obs": obs_by[cond], "sur": sur_by[cond], "n": n_use,
                    "m": m_used, "shifts": shifts}

            # ---- per-subject surrogate QC and observed-vs-surrogate summary ----
            for cond in CONDITIONS:
                d = cond_dat[cond]
                sur_qc.append({
                    "analysis_id": aid, "subject_id": sid, "condition": cond,
                    "n_points": n_use, "acf_lag_pupil": d["acf_lag"],
                    "min_shift_points_condition": d["m_c"],
                    "min_shift_points_subject": m_used,
                    "acf_fallback": bool(d["fallback"]),
                    "n_unique_shifts": len(shifts),
                    "shift_min": min(shifts) if shifts else np.nan,
                    "shift_max": max(shifts) if shifts else np.nan,
                    "stationarity_supported": bool(d["ok_st"]),
                    "half_mean_diff_sd": d["split"], "drift_sd": d["drift"],
                    "status": "ok" if (do_sur and shifts) else (
                        "no_surrogate_requested" if not do_sur else "not_evaluable"),
                    "reason": "" if (do_sur and shifts) else (
                        "surrogate not requested for this setting" if not do_sur
                        else "empty shift set")})
                for pname in ("offset", "exponent"):
                    O, S = obs_by[cond][pname], sur_by[cond][pname]
                    for r in range(6):
                        v = S[:, r]
                        v = v[np.isfinite(v)]
                        if v.size:
                            sm, med = float(v.mean()), float(np.median(v))
                            q025, q975 = (float(x) for x in np.percentile(v, [2.5, 97.5]))
                            if aid == "native_equal" and REGION_ORDER[r] == "prefrontal":
                                sur_draw.setdefault((cond, pname), []).extend(v.tolist())
                            exc = float(O[r] - sm) if np.isfinite(O[r]) else np.nan
                            stt, rsn = "ok", ""
                        else:
                            sm = med = q025 = q975 = exc = np.nan
                            stt = "not_evaluable"
                            rsn = ("surrogate not requested for this setting" if not do_sur
                                   else "no finite surrogate value")
                        sur_summary.append({
                            "analysis_id": aid, "subject_id": sid, "condition": cond,
                            "region": REGION_ORDER[r], "parameter": pname,
                            "metric": "mi_excess",
                            "n_points": n_use, "n_channels": int(masks[r].sum()),
                            "n_unique_shifts": int(v.size), "mi_observed": O[r],
                            "surrogate_mean": sm, "surrogate_median": med,
                            "surrogate_q025": q025, "surrogate_q975": q975,
                            "mi_excess": exc, "status": stt, "reason": rsn})

            # ---- TTS (participant level, only for the main equal-length setting) ----
            if aid == "native_equal":
                rr = int((n_use - CONFIG["n_min"]) // 2)
                if rr < 1 or (n_use - 2 * rr) < CONFIG["n_min"]:
                    rsn = (f"radius r={rr} leaves {n_use - 2 * rr} points "
                           f"(needs >= n_min={CONFIG['n_min']} and r>=1)")
                    for cond in CONDITIONS:
                        for pname in ("offset", "exponent"):
                            for r in range(6):
                                tts_rows.append({
                                    "analysis_id": "tts_native_equal", "subject_id": sid,
                                    "condition": cond, "region": REGION_ORDER[r],
                                    "parameter": pname, "radius": rr,
                                    "n_points_used": n_use - 2 * rr, "theta_zero": np.nan,
                                    "theta_min": np.nan, "theta_max": np.nan,
                                    "count_ge": np.nan, "p_tts": np.nan,
                                    "min_attainable_p": np.nan,
                                    "compute_status": "not_evaluable", "data_valid": False,
                                    "assumptions_supported": False,
                                    "inference_status": "not_evaluable", "reason": rsn})
                else:
                    tt = tts_subject({c: cond_dat[c]["y"] for c in CONDITIONS},
                                     {c: cond_dat[c]["off"] for c in CONDITIONS},
                                     {c: cond_dat[c]["exp"] for c in CONDITIONS},
                                     masks, rr, k)
                    for cond in CONDITIONS:
                        TH_all = tt[cond]["theta"]
                        for pname in ("offset", "exponent"):
                            TH = TH_all[pname]
                            for r in range(6):
                                col = TH[:, r]
                                zero = float(TH[rr, r])
                                ok_col = bool(np.isfinite(col).all())
                                if ok_col:
                                    cnt = int(np.sum(col >= zero))
                                    pval = min(1.0, cnt / (rr + 1.0))
                                    cs, dv, rsn = "ok", True, ""
                                else:
                                    cnt, pval = np.nan, np.nan
                                    cs, dv = "not_evaluable", False
                                    rsn = "non-finite theta for at least one shift"
                                ok_st = bool(tt[cond]["stationarity"][0])
                                tts_rows.append({
                                    "analysis_id": "tts_native_equal", "subject_id": sid,
                                    "condition": cond, "region": REGION_ORDER[r],
                                    "parameter": pname, "radius": rr,
                                    "n_points_used": tt[cond]["n_used"],
                                    "theta_zero": zero if ok_col else np.nan,
                                    "theta_min": float(np.nanmin(col)),
                                    "theta_max": float(np.nanmax(col)),
                                    "count_ge": cnt, "p_tts": pval,
                                    "min_attainable_p": 1.0 / (rr + 1.0),
                                    "compute_status": cs, "data_valid": dv,
                                    "assumptions_supported": ok_st,
                                    "inference_status": "not_evaluable" if cs != "ok"
                                    else ("p_le_alpha" if pval <= CONFIG["alpha"]
                                          else "p_gt_alpha"),
                                    "reason": rsn})
                                for dd in range(2 * rr + 1):
                                    w_th.writerow([
                                        "tts_native_equal", sid, cond, REGION_ORDER[r],
                                        pname, dd - rr,
                                        f"{col[dd]:.10g}" if np.isfinite(col[dd]) else "",
                                        int(dd == rr)])
        log(f"{aid}: evaluable {len(eval_ids)}/{len(cohort)}, excluded {len(excl)}, "
            f"elapsed {time.time() - t_set:.1f}s")

    # ==================================================================================
    # Position windows: duplicate flag and shared fraction (protocol L128-140)
    # ==================================================================================
    pos_idx: Dict[Tuple[str, str], Dict[str, int]] = {}
    for i, row in enumerate(segments):
        if (row["analysis_id"] in POSITION_IDS
                and np.isfinite(row.get("start_index", np.nan))):
            pos_idx.setdefault((row["subject_id"], row["condition"]), {})[row["analysis_id"]] = i
    n_dup = 0
    for (sid_, cond_), d in pos_idx.items():
        for aid in POSITION_IDS:
            i = d.get(aid)
            if i is None:
                continue
            st = float(segments[i]["start_index"])
            nwin = float(segments[i]["n_points"])
            dup = False
            for other in POSITION_IDS:
                j = d.get(other)
                if j is None:
                    continue
                st2 = float(segments[j]["start_index"])
                ov = max(0.0, min(st + nwin, st2 + nwin) - max(st, st2))
                segments[i][f"shared_frac_vs_{POSITION_NAME[other]}"] = ov / nwin
                if other != aid and st2 == st:
                    dup = True
            segments[i]["duplicate_window"] = bool(dup)
            n_dup += int(dup)
    log(f"position windows: {n_dup} rows share an identical INDEX with another position")

    # ==================================================================================
    # Matched channels and cohort manifest
    # ==================================================================================
    for sid_ in cohort:
        for c in chans_of[sid_]:
            matched_channels.append({
                "subject_id": sid_, "channel": int(c), "region": regions[c],
                "channel_set_id": f"{sid_}_{len(chans_of[sid_])}ch",
                "n_channels_in_set": len(chans_of[sid_])})
    demo = load_demographics()
    for row in cohort_rows:
        d = demo.get(row["subject_id"], {})
        row["age"] = d.get("age", np.nan)
        row["sex"] = d.get("sex", np.nan)
        row["age_group"] = d.get("age_group", "")

    # ==================================================================================
    # Condition summaries and tests (unit = subject)
    # ==================================================================================
    aids_done = [s[0] for s in specs if results.get(s[0])]
    subs_by: Dict[str, List[str]] = {
        aid: sorted({sid_ for (sid_, _c) in results[aid]}) for aid in aids_done}
    idx_by = {aid: {s: i for i, s in enumerate(subs_by[aid])} for aid in aids_done}

    exc_map: Dict[Tuple[str, str, str, str, str], float] = {}
    for row in sur_summary:
        exc_map[(row["analysis_id"], row["subject_id"], row["condition"],
                 row["parameter"], row["region"])] = row["mi_excess"]

    V: Dict[Tuple[str, str, str, str, str], np.ndarray] = {}
    for aid in aids_done:
        subs_ = subs_by[aid]
        for metric in ("mutual_information", "mi_excess"):
            for pname in ("offset", "exponent"):
                for region in REGION_ORDER:
                    ri = REGION_INDEX[region]
                    for cond in CONDITIONS:
                        arr = np.full(len(subs_), np.nan)
                        for i, sid_ in enumerate(subs_):
                            rec = results[aid].get((sid_, cond))
                            if rec is None:
                                continue
                            arr[i] = (float(rec["obs"][pname][ri])
                                      if metric == "mutual_information"
                                      else exc_map.get((aid, sid_, cond, pname, region),
                                                       np.nan))
                        V[(aid, metric, pname, region, cond)] = arr

    cond_sum = []
    for aid in aids_done:
        for cond in CONDITIONS:
            for pname in ("offset", "exponent"):
                for region in REGION_ORDER:
                    for metric in ("mutual_information", "mi_excess"):
                        v = V[(aid, metric, pname, region, cond)]
                        v = v[np.isfinite(v)]
                        cond_sum.append({
                            "analysis_id": aid, "condition": cond, "region": region,
                            "parameter": pname, "metric": metric,
                            "n_subjects": int(v.size),
                            "mean": float(v.mean()) if v.size else np.nan,
                            "sd": float(v.std(ddof=1)) if v.size > 1 else np.nan,
                            "median": float(np.median(v)) if v.size else np.nan,
                            "min": float(v.min()) if v.size else np.nan,
                            "max": float(v.max()) if v.size else np.nan})

    test_rows: List[dict] = []

    def add_test(aid, metric, pname, region, test, contrast, res, kind):
        row = {
            "analysis_id": aid, "metric": metric, "parameter": pname, "region": region,
            "test": test, "contrast": contrast,
            "n_subjects": res.get("n_subjects", np.nan),
            "statistic": res.get("statistic", np.nan),
            "effect_value": res.get("effect_value", np.nan),
            "median_paired_difference": res.get("median_paired_difference", np.nan),
            "ci_low": res.get("ci_low", np.nan), "ci_high": res.get("ci_high", np.nan),
            "n_zero_difference": res.get("n_zero_difference", np.nan),
            "n_ties_abs": res.get("n_ties_abs", np.nan),
            "p_raw": res.get("p_raw", np.nan),
            "family_id_orig": "", "family_planned_orig": np.nan,
            "p_adjusted_orig": np.nan, "status_orig": "",
            "family_id_mi": "", "p_adjusted_mi": np.nan,
            "status": res.get("status", ""), "reason": res.get("reason", "")}
        if metric == "mutual_information" and aid == "native_equal":
            if kind == "overall":
                row.update(family_id_orig="orig48_overall", family_planned_orig=48,
                           status_orig="family_pending")
            else:
                row.update(family_id_orig="orig144_paired", family_planned_orig=144,
                           status_orig="family_pending")
        test_rows.append(row)

    for aid in aids_done:
        for metric in ("mutual_information", "mi_excess"):
            for pname in ("offset", "exponent"):
                for region in REGION_ORDER:
                    fr = friedman_test([V[(aid, metric, pname, region, c)]
                                        for c in CONDITIONS])
                    add_test(aid, metric, pname, region, "friedman", "", fr, "overall")
                    for c1, c2 in CONTRASTS:
                        pt = paired_test(V[(aid, metric, pname, region, c1)],
                                         V[(aid, metric, pname, region, c2)], rng)
                        add_test(aid, metric, pname, region, "wilcoxon_signed_rank",
                                 f"{c1}_vs_{c2}", pt, "paired")

    fam_defs = [("mutual_information", "mi_only_12_overall", "mi_only_36_paired"),
                ("mi_excess", "mi_excess12_overall", "mi_excess36_paired")]
    for aid in aids_done:
        for metric, f_over, f_pair in fam_defs:
            for tname, fam in (("friedman", f_over),
                               ("wilcoxon_signed_rank", f_pair)):
                rows = [r for r in test_rows
                        if r["analysis_id"] == aid and r["metric"] == metric
                        and r["test"] == tname]
                ps = [r["p_raw"] for r in rows]
                n_bad = int(sum(1 for p in ps if not np.isfinite(p)))
                if not rows or n_bad == len(rows):
                    for r in rows:
                        r["family_id_mi"] = f"{fam}_not_computed"
                        r["p_adjusted_mi"] = np.nan
                    continue
                adj = bh_fdr(ps)
                for r, a in zip(rows, adj):
                    r["family_id_mi"] = fam
                    r["p_adjusted_mi"] = float(a)
                    if n_bad:
                        extra = (f"BH over {len(rows)} tests, {n_bad} non-computable p "
                                 f"replaced by 1 (m preserved)")
                        r["reason"] = f"{r['reason']}; {extra}" if r["reason"] else extra

    # ==================================================================================
    # Sensitivity: each alternative setting vs native_equal (subject-level pairing)
    # ==================================================================================
    sens_rows: List[dict] = []
    baseline = "native_equal"
    if baseline in aids_done:
        for other in [a for a in aids_done if a != baseline]:
            common = sorted(set(subs_by[baseline]) & set(subs_by[other]))
            ia = [idx_by[baseline][s] for s in common]
            ib = [idx_by[other][s] for s in common]
            n_missing = len(subs_by[baseline]) - len(common)
            for metric in ("mutual_information", "mi_excess"):
                bucket = []
                for cond in CONDITIONS:
                    for pname in ("offset", "exponent"):
                        for region in REGION_ORDER:
                            a = V[(baseline, metric, pname, region, cond)][ia]
                            b = V[(other, metric, pname, region, cond)][ib]
                            pt = paired_test(a, b, rng)
                            note = (f"{pt['n_subjects']} common subjects; "
                                    f"{n_missing} native_equal subjects absent from "
                                    f"{other}")
                            if other in POSITION_IDS:
                                note += (f"; excluded there if n_i < "
                                         f"{CONFIG['n_position']} points")
                            if other in ("native_equal_k5", "native_equal_k10"):
                                note += "; same windows, only k differs"
                            if other == "native_global_equal":
                                note += (f"; common length forced to n_global="
                                         f"{n_global}")
                            bucket.append({
                                "analysis_id": other, "baseline": baseline,
                                "condition": cond, "region": region,
                                "parameter": pname, "metric": metric,
                                "test": pt["test"], "n_subjects": pt["n_subjects"],
                                "median_paired_difference":
                                    pt["median_paired_difference"],
                                "ci_low": pt["ci_low"], "ci_high": pt["ci_high"],
                                "p_raw": pt["p_raw"], "p_adjusted": np.nan,
                                "status": pt["status"], "reason": pt["reason"],
                                "note": note})
                ps = [r["p_raw"] for r in bucket]
                if bucket and any(np.isfinite(ps)):
                    for r, a in zip(bucket, bh_fdr(ps)):
                        r["p_adjusted"] = float(a)
                        if any(not np.isfinite(p) for p in ps):
                            r["note"] += ("; BH over "
                                          f"{len(bucket)} tests with NaN->1 placeholder")
                sens_rows.extend(bucket)
        log(f"sensitivity rows: {len(sens_rows)}")

    # ==================================================================================
    # Deliverables
    # ==================================================================================
    _write_rows(out_dir / "segments.csv", SEGMENT_COLS, segments)
    _write_rows(out_dir / "coupling_values.csv", COUPLING_COLS, coupling_values)
    _write_rows(out_dir / "surrogate_qc.csv", SUR_QC_COLS, sur_qc)
    _write_rows(out_dir / "surrogate_summary.csv", SUR_SUM_COLS, sur_summary)
    _write_rows(out_dir / "tts_results.csv", TTS_COLS, tts_rows)
    _write_rows(out_dir / "condition_summary.csv", COND_SUM_COLS, cond_sum)
    _write_rows(out_dir / "condition_tests.csv", TEST_COLS, test_rows)
    _write_rows(out_dir / "sensitivity_summary.csv", SENS_COLS, sens_rows)
    _write_rows(out_dir / "matched_channels.csv", MATCH_COLS, matched_channels)
    _write_rows(out_dir / "cohort_manifest.csv", COHORT_COLS, cohort_rows)
    np.savez_compressed(out_dir / "paired_inputs.npz", **npz_payload)

    log(f"done: {len(aids_done)} settings, {len(segments)} segment rows, "
        f"{len(test_rows)} test rows, {len(tts_rows)} TTS rows, "
        f"total {time.time() - t_start:.1f}s")
    for fh in (f_sur, f_ch, f_th, f_log):
        fh.flush()
        fh.close()

    try:
        make_figures(out_dir, segments, blocks, sur_draw, sur_summary, sens_rows,
                     n_global)
    except Exception as exc:  # figures must not invalidate the tables
        print(f"[figures] skipped: {type(exc).__name__}: {exc}", flush=True)


# ======================================================================================
# Figures
# ======================================================================================
def make_figures(out_dir: Path, segments: List[dict], blocks: Dict[str, dict],
                 sur_draw: Dict[Tuple[str, str], List[float]],
                 sur_summary: List[dict], sens_rows: List[dict],
                 n_global: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ---- (1) real timeline and the selected windows for one subject ----
    nat = [r for r in segments if r["analysis_id"] == "native_equal"
           and r["status"] == "ok"]
    if nat:
        per_sid = {}
        for r in nat:
            per_sid.setdefault(r["subject_id"], 0)
            per_sid[r["subject_id"]] += 1
        sid = sorted([s for s, n in per_sid.items() if n == 3])[0]
        conds = CONDITIONS
        fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=False)
        colors = {"native_equal": "black", "position_early": "#1f77b4",
                  "position_middle": "#2ca02c", "position_late": "#ff7f0e"}
        for ax, cond in zip(axes, conds):
            blk = blocks[sid][cond]
            valid = blk["valid"]
            ax.fill_between(np.arange(valid.size) + 1.0, 0, 1, where=valid,
                            step="mid", color="#cccccc", alpha=0.7,
                            label="paired sample valid")
            y0 = 1.0
            for aid, col in colors.items():
                for r in segments:
                    if (r["analysis_id"] == aid and r["subject_id"] == sid
                            and r["condition"] == cond
                            and np.isfinite(r.get("start_center_s", np.nan))):
                        ax.plot([r["start_center_s"], r["end_center_s"]], [y0, y0],
                                lw=6, color=col, solid_capstyle="butt",
                                label=f"{aid} (n={int(r['n_points'])})")
                        y0 += 0.22
            ax.set_ylim(0, 2.3)
            ax.set_yticks([])
            ax.set_ylabel(cond)
            ax.set_xlabel("time after video onset (s); center of each 2 s window")
            ax.grid(axis="x", alpha=0.3)
        axes[0].legend(fontsize=7, ncol=3, loc="upper right")
        fig.suptitle(f"Subject {sid}: valid pupil/parameter samples and selected windows "
                     f"(native_equal n points shown in legend; n_global={n_global})")
        fig.tight_layout()
        fig.savefig(out_dir / "fig_segments_timeline.png", dpi=150)
        plt.close(fig)

    # ---- (2) observed MI against the surrogate distribution (prefrontal, offset) ----
    if sur_draw:
        fig, ax = plt.subplots(figsize=(8, 5))
        data, obs_all = [], []
        for cond in CONDITIONS:
            data.append(np.asarray(sur_draw.get((cond, "offset"), []), float))
            obs = [r["mi_observed"] for r in sur_summary
                   if r["analysis_id"] == "native_equal" and r["condition"] == cond
                   and r["region"] == "prefrontal" and r["parameter"] == "offset"
                   and np.isfinite(r["mi_observed"])]
            obs_all.append(np.asarray(obs, float))
        pos = np.arange(1, 4)
        bp = ax.boxplot([d[np.isfinite(d)] for d in data], positions=pos, widths=0.5,
                        whis=(2.5, 97.5), showfliers=False, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("#dddddd")
        for i, cond in enumerate(CONDITIONS):
            o = obs_all[i]
            ax.scatter(np.full(o.size, pos[i]) + 0.08, o, s=12, color="crimson",
                       zorder=3, label="observed MI" if i == 0 else None)
        ax.set_xticks(pos)
        ax.set_xticklabels([f"{c}\n(n={np.isfinite(o).sum()} subjects)" 
                            for c, o in zip(CONDITIONS, obs_all)])
        ax.set_ylabel("MI (bit), prefrontal / offset")
        ax.set_title("Observed MI (red) against the cyclic-shift surrogate distribution "
                     "(grey: 2.5-97.5 percentile, all unique shifts)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "fig_surrogate_position.png", dpi=150)
        plt.close(fig)

    # ---- (3) paired differences between settings (negative condition) ----
    if sens_rows:
        sel = [r for r in sens_rows if r["condition"] == "negative"
               and r["metric"] == "mutual_information"]
        comps = sorted({r["analysis_id"] for r in sel})
        fig, axes = plt.subplots(1, len(comps), figsize=(4.2 * len(comps), 5.2),
                                 sharey=True)
        if len(comps) == 1:
            axes = [axes]
        for ax, comp in zip(axes, comps):
            rows = [r for r in sel if r["analysis_id"] == comp]
            rows.sort(key=lambda r: (r["region"], r["parameter"]))
            yy = np.arange(len(rows))
            md = np.array([r["median_paired_difference"] for r in rows], float)
            lo = np.array([r["ci_low"] for r in rows], float)
            hi = np.array([r["ci_high"] for r in rows], float)
            ax.errorbar(md, yy, xerr=[md - lo, hi - md], fmt="o", ms=3, lw=1)
            ax.axvline(0, color="grey", lw=1)
            ax.set_yticks(yy)
            ax.set_yticklabels([f"{r['region'][:6]}/{r['parameter'][:3]}" for r in rows],
                               fontsize=6)
            ax.set_xlabel("median paired difference (bit)")
            ax.set_title(f"{comp}\nvs native_equal", fontsize=8)
        nrep = sel[0]["n_subjects"] if sel else 0
        fig.suptitle(f"Setting sensitivity, negative condition, observed MI "
                     f"(n={nrep} common subjects; bars = 95% bootstrap CI of median)")
        fig.tight_layout()
        fig.savefig(out_dir / "fig_sensitivity_forest.png", dpi=150)
        plt.close(fig)


# ======================================================================================
# Self-check (protocol L205)
# ======================================================================================
def stage_selfcheck(out_dir: Path) -> None:
    t0 = time.time()
    rep: Dict[str, object] = {"out_dir": str(out_dir), "checks": []}

    def chk(name: str, ok: bool, detail: object) -> None:
        rep["checks"].append({"check": name, "pass": bool(ok), "detail": detail})
        print(f"[selfcheck] {'PASS' if ok else 'FAIL'}  {name}: {detail}")

    man = pd.read_csv(out_dir / "cohort_manifest.csv")
    avail = man["available"].astype(str).str.lower().isin(["true", "1"])
    nat = man[(man["analysis_id"] == "native_equal") & avail]
    seg = pd.read_csv(out_dir / "segments.csv")
    npz = np.load(out_dir / "paired_inputs.npz")
    keys = set(npz.files)
    sid = None
    for s in nat["subject_id"].astype(str):
        if all(f"y|{s}|{c}" in keys for c in CONDITIONS):
            sid = s
            break
    chk("subject with three native_equal arrays", sid is not None, sid)
    if sid is None:
        rep["elapsed_s"] = time.time() - t0
        (out_dir / "selfcheck_report.json").write_text(
            json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
        return

    ns, starts, ts_ok, finite_ok = [], {}, True, True
    for cond in CONDITIONS:
        y = npz[f"y|{sid}|{cond}"]
        t = npz[f"t|{sid}|{cond}"]
        off = npz[f"off|{sid}|{cond}"]
        exp = npz[f"exp|{sid}|{cond}"]
        pv = npz[f"pvalid|{sid}|{cond}"]
        pi = npz[f"pinterp|{sid}|{cond}"]
        ns.append(int(y.size))
        starts[cond] = float(npz[f"start|{sid}|{cond}"][0])
        ts_ok &= bool(np.allclose(np.diff(t), CONFIG["time_grid"]["step_s"]))
        ts_ok &= bool(abs(t[0] - (starts[cond] + 1.0)) < 1e-9)
        finite_ok &= bool(np.isfinite(y).all() and np.isfinite(off).all()
                          and np.isfinite(exp).all() and pv.size == y.size
                          and pi.size == y.size)
    chk("three conditions share one n_points", len(set(ns)) == 1, ns)
    chk("center times are 1 s apart and start at index+1", ts_ok, starts)
    chk("x/y/t complete (finite, same length)", finite_ok,
        f"{off.shape[0]} channels")  # type: ignore[name-defined]
    ch = npz[f"ch|{sid}"]
    chk("one common channel set for the subject", len(ch) == len(set(ch.tolist())),
        f"{len(ch)} channels")

    # region mean recomputed from the per-channel table
    ccv = pd.read_csv(out_dir / "coupling_channel_values.csv")
    sub = ccv[(ccv["analysis_id"] == "native_equal") & (ccv["subject_id"].astype(str) == sid)
              & (ccv["condition"] == "negative") & (ccv["region"] == "prefrontal")
              & (ccv["parameter"] == "offset")]
    vals = pd.to_numeric(sub["value"], errors="coerce").dropna()
    cv = pd.read_csv(out_dir / "coupling_values.csv")
    row = cv[(cv["analysis_id"] == "native_equal") & (cv["subject_id"].astype(str) == sid)
             & (cv["condition"] == "negative") & (cv["region"] == "prefrontal")
             & (cv["parameter"] == "offset")]
    stored = float(row["value"].iloc[0]) if len(row) else np.nan
    chk("region mean recomputed from channel values",
        bool(np.isfinite(stored) and abs(vals.mean() - stored) < 1e-8),
        {"n_channels": int(vals.size), "recomputed": float(vals.mean()),
         "stored": stored})

    # surrogate mean and mi_excess recomputed from every saved surrogate value
    target = ("native_equal", sid, "negative", "prefrontal", "offset")
    hits = []
    for chunk in pd.read_csv(out_dir / "surrogate_values.csv", chunksize=250000):
        m = ((chunk["analysis_id"] == target[0])
             & (chunk["subject_id"].astype(str) == target[1])
             & (chunk["condition"] == target[2]) & (chunk["region"] == target[3])
             & (chunk["parameter"] == target[4]))
        if m.any():
            hits.append(chunk.loc[m, ["shift_points", "mi_value"]])
        if hits and chunk["subject_id"].astype(str).iloc[-1] != target[1]:
            break
    sv = pd.concat(hits) if hits else pd.DataFrame(columns=["shift_points", "mi_value"])
    sv["mi_value"] = pd.to_numeric(sv["mi_value"], errors="coerce")
    sv = sv.dropna(subset=["mi_value"])
    ss = pd.read_csv(out_dir / "surrogate_summary.csv")
    srow = ss[(ss["analysis_id"] == target[0]) & (ss["subject_id"].astype(str) == target[1])
              & (ss["condition"] == target[2]) & (ss["region"] == target[3])
              & (ss["parameter"] == target[4])]
    sm_stored = float(srow["surrogate_mean"].iloc[0]) if len(srow) else np.nan
    exc_stored = float(srow["mi_excess"].iloc[0]) if len(srow) else np.nan
    exc_re = float(srow["mi_observed"].iloc[0]) - sv["mi_value"].mean() if len(row) else np.nan
    chk("surrogate mean recomputed from all unique shifts",
        bool(len(sv) and abs(sv["mi_value"].mean() - sm_stored) < 1e-8),
        {"n_unique_shifts": int(sv["shift_points"].nunique()), "n_values": int(len(sv)),
         "recomputed": float(sv["mi_value"].mean()), "stored": sm_stored})
    chk("mi_excess = observed - mean(surrogate)",
        bool(np.isfinite(exc_re) and abs(exc_re - exc_stored) < 1e-8),
        {"recomputed": exc_re, "stored": exc_stored})

    # TTS: B, denominator r+1 and n-2r from the saved theta
    tr = pd.read_csv(out_dir / "tts_results.csv")
    trow = tr[(tr["subject_id"].astype(str) == sid) & (tr["condition"] == "negative")
              & (tr["region"] == "prefrontal") & (tr["parameter"] == "offset")]
    th = pd.read_csv(out_dir / "tts_theta.csv")
    ths = th[(th["subject_id"].astype(str) == sid) & (th["condition"] == "negative")
             & (th["region"] == "prefrontal") & (th["parameter"] == "offset")]
    ths = ths.sort_values("d")
    if len(trow) and len(ths):
        r = int(trow["radius"].iloc[0])
        zero = float(trow["theta_zero"].iloc[0])
        b = int((ths["theta"].to_numpy(float) >= zero).sum())
        p = min(1.0, b / (r + 1.0))
        chk("TTS count/denominator recomputed",
            bool(b == int(trow["count_ge"].iloc[0])
                 and abs(p - float(trow["p_tts"].iloc[0])) < 1e-12
                 and len(ths) == 2 * r + 1),
            {"r": r, "B": b, "denominator": r + 1, "n_theta": int(len(ths)),
             "n_points_used": int(trow["n_points_used"].iloc[0])})
    else:
        chk("TTS rows present", False, "missing TTS row or theta")

    rep["status"] = ("all_pass" if all(c["pass"] for c in rep["checks"])  # type: ignore[index]
                     else "some_failed")
    rep["elapsed_s"] = round(time.time() - t0, 1)
    (out_dir / "selfcheck_report.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[selfcheck] status={rep['status']}  elapsed={rep['elapsed_s']}s")


# ======================================================================================
# CLI
# ======================================================================================
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", nargs="?", default="run",
                    choices=["inventory", "probe", "run", "selfcheck"])
    ap.add_argument("--out", default=str(OUT_ROOT))
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--settings", nargs="*", default=None)
    a = ap.parse_args(argv)
    out = Path(a.out)
    if a.stage == "inventory":
        stage_inventory(out)
    elif a.stage == "probe":
        stage_probe(out)
    elif a.stage == "run":
        stage_run(out, a.settings, a.subjects)
    else:
        stage_selfcheck(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
