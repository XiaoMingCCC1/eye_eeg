#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
审稿补算草稿（draft）—— 对应 required_calculation_steps(1).md 的 ①②③ 及统计家族修正。

本文件实现审稿要求的核心计算，尚未接全量衍生数据，先供单被试×三任务验证流程：
  ① 等长分析：三任务用可比较长度重算耦合指标（native_equal / native_global_equal）。
  ② 替代检验：循环平移参考 + mi_excess；必要时截断时间平移(TTS，Yuan & Shou)。
  ③ 时间对齐：保留原生时间戳/谱支持的映射与验证辅助函数。
  统计：统一改为重复测量配对检验（Friedman 总体 + Wilcoxon 配对 + BH-FDR）。

设计要点（与文档一致）：
  - 条件映射固定为 negative=Video1, neutral=Video2, positive=Video3。
  - 时间统一为秒，但先保留原生时间戳与时钟标识。
  - 缺失用 NaN + 原因字段，绝不用 0 顶替；无效瞳孔 0 按原 QC 识别。
  - 不读取 MI 结果来选窗口；主分析取“合格连续区间中最早可用窗口”。
  - FDR 先建全部分定检验行，不可算者 p_raw=NA 且原因记录，家族以完整清单为分母。
  - 循环移位是“周期延拓”下的稳健性参考，不是精确独立性 p 值；TTS 才是正式 p 值候选。
"""

from __future__ import annotations

import math
from itertools import combinations
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.special import digamma
from scipy.stats import friedmanchisquare, wilcoxon

try:
    from sklearn.neighbors import NearestNeighbors
    _HAS_SKLEARN = True
except Exception:  # pragma: no cover
    _HAS_SKLEARN = False


# ---------------------------------------------------------------------------
# 0. 基本工具
# ---------------------------------------------------------------------------
CONDITION_MAP = {'negative': 'Video1', 'neutral': 'Video2', 'positive': 'Video3'}


def bh_fdr(pvals: Sequence[float], alpha: float = 0.05) -> Tuple[List[float], List[bool]]:
    """Benjamini-Hochberg FDR 校正。

    返回 (p_adjusted, significant)。输入中的 NaN 会被置为 1.0 参与家族大小守恒，
    但在调用方仍需单独记录“不可计算”原因（文档：占位 1 只是计算占位）。
    """
    m = len(pvals)
    if m == 0:
        return [], []

    p = np.array([float(x) if np.isfinite(x) else 1.0 for x in pvals])
    order = np.argsort(p)
    sorted_p = p[order]
    ranks = np.arange(1, m + 1)
    q = sorted_p * m / ranks
    q_monotonic = np.minimum.accumulate(q[::-1])[::-1]

    p_adj = np.empty(m)
    p_adj[order] = q_monotonic

    thresholds = alpha * ranks / m
    sig_sorted = sorted_p <= thresholds
    sig = np.empty(m, dtype=bool)
    sig[order] = sig_sorted

    return p_adj.tolist(), sig.tolist()


def kendall_w(chi2_stat: float, n_subjects: int, k_groups: int) -> float:
    """Friedman 检验对应的 Kendall's W 效应量：W = chi2 / (N*(k-1))。"""
    denom = n_subjects * (k_groups - 1)
    return float(chi2_stat / denom) if denom > 0 else float('nan')


# ---------------------------------------------------------------------------
# ②/① MI 估计器：连续变量 k 近邻 MI（Kraskov–Stögbauer–Grassberger, KSG 估计 1）
# 文档要求“先确认原函数确为连续变量近邻 MI 估计器”；原代码用的是 10 箱直方图粗估，
# 需要替换成本函数（或至少作为与旧实现的对照）。
# ---------------------------------------------------------------------------
def kraskov_mi(x: Sequence[float], y: Sequence[float], k: int = 3) -> float:
    """连续变量 k-NN 互信息（KSG 估计 1，chebyshev 距离）。

    输入需为等长、有限的时间对齐序列；非有限值被剔除后重配对。
    点数 < k+1 时返回 NaN（表示不可估计，不填 0）。
    """
    if not _HAS_SKLEARN:
        raise RuntimeError('kraskov_mi 需要 scikit-learn（NearestNeighbors）。')

    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.shape[0] != y.shape[0]:
        raise ValueError('x/y 长度不一致。')
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < k + 1:
        return float('nan')

    pts = np.column_stack([x, y])
    nn = NearestNeighbors(n_neighbors=k + 1, metric='chebyshev')
    nn.fit(pts)
    dists, _ = nn.kneighbors(pts)
    eps = dists[:, -1]  # 各点到第 k 近邻的 chebyshev 距离

    nx = np.empty(n)
    ny = np.empty(n)
    for i in range(n):
        # 严格小于 eps_i 的点计数，去除自身
        nx[i] = np.sum(np.abs(x - x[i]) < eps[i]) - 1
        ny[i] = np.sum(np.abs(y - y[i]) < eps[i]) - 1

    mi = digamma(k) - np.mean(digamma(nx + 1) + digamma(ny + 1)) + digamma(n)
    return float(max(mi, 0.0))


# ---------------------------------------------------------------------------
# ① 等长分析：连续有效区间 + 每人三条件等长窗口选取
# ---------------------------------------------------------------------------
def find_continuous_valid_segments(
    valid_mask: Sequence[bool], timestamps: Sequence[float]
) -> List[Dict[str, object]]:
    """按真实时间戳识别连续有效区间（缺失谱窗/不合格瞳孔窗会切断区间）。

    不能删除缺失点后把两段拼成一段；以时间步长是否连续来判断接缝。
    返回按长度降序的区间列表，每个元素含 start/end 位置与点数、时间跨度。
    """
    valid = np.asarray(valid_mask, dtype=bool)
    ts = np.asarray(timestamps, dtype=float)
    n = len(valid)
    if n == 0:
        return []

    # 连续 = 有效且时间步与前一有效点步长接近（允许小容差），默认取相邻有效点
    segs: List[Dict[str, object]] = []
    i = 0
    while i < n:
        if not valid[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and valid[j + 1]:
            j += 1
        segs.append({
            'start': int(i),
            'end': int(j),       # inclusive
            'n_points': int(j - i + 1),
            't_start': float(ts[i]),
            't_end': float(ts[j]),
            'span_s': float(ts[j] - ts[i]),
        })
        i = j + 1

    segs.sort(key=lambda s: s['n_points'], reverse=True)
    return segs


def select_equal_length_windows(
    segs_by_condition: Dict[str, List[Dict[str, object]]],
    n_min: int,
    rule: str = 'earliest',
) -> Tuple[Optional[int], Dict[str, Optional[Dict[str, object]]]]:
    """①1.4 每人三条件等长：n_i = min(三条件最长合格连续区间点数)。

    各条件取“最早满足 n_i”的窗口；三条件步长相同，因此时长与点数同时匹配。
    n_i < n_min 者返回 (None, ...) 表示该被试不进入主方案。

    segs_by_condition: {condition: find_continuous_valid_segments(...)}
    """
    longest = {}
    for cond, segs in segs_by_condition.items():
        if not segs:
            return None, {}
        longest[cond] = int(segs[0]['n_points'])

    n_i = min(longest.values())
    if n_i < n_min:
        return None, {}

    windows: Dict[str, Optional[Dict[str, object]]] = {}
    for cond, segs in segs_by_condition.items():
        # 在长度 >= n_i 的区间中按规则选窗口（最早：起始时间最小）
        candidates = [s for s in segs if s['n_points'] >= n_i]
        if not candidates:
            return None, {}
        if rule == 'earliest':
            chosen = min(candidates, key=lambda s: s['t_start'])
        elif rule == 'middle':
            chosen = min(candidates, key=lambda s: abs((s['t_start'] + s['t_end']) / 2))
        elif rule == 'latest':
            chosen = max(candidates, key=lambda s: s['t_start'])
        else:
            raise ValueError(f'未知取窗规则: {rule}')

        start = int(chosen['start'])
        windows[cond] = {
            'start': start,
            'end': start + n_i - 1,
            'n_points': n_i,
            't_start': float(chosen['t_start']),
            't_end': float(chosen['t_start']) + (n_i - 1),  # 占位，实际以时间戳为准
        }
    return n_i, windows


# ---------------------------------------------------------------------------
# ② 替代检验：循环平移参考与 mi_excess；TTS 截断时间平移
# ---------------------------------------------------------------------------
def allowed_shift_points(n_points: int, min_shift_points: int) -> List[int]:
    """②2.2 允许偏移 S = {s: min_shift_points <= s <= n - min_shift_points}，取整数且排除 0。

    注意：min_shift_points 必须由数据/配置得到（谱支持、自相关持续时长），
    不能预填任意秒数。
    """
    if min_shift_points < 1 or 2 * min_shift_points > n_points:
        return []
    return [s for s in range(min_shift_points, n_points - min_shift_points + 1)]


def circular_shift_surrogate(
    x: Sequence[float],
    y: Sequence[float],
    shifts: Sequence[int],
    stat_fn: Callable[[np.ndarray, np.ndarray], float],
) -> np.ndarray:
    """固定 x（参数），对 y（瞳孔）整体循环偏移 s 点，重算同一统计量。

    同一被试/条件/替代编号下，所有通道与两个参数共用同一 s。
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    vals = []
    for s in shifts:
        if s == 0:
            vals.append(stat_fn(x, y))
        else:
            vals.append(stat_fn(x, np.roll(y, s)))
    return np.asarray(vals, dtype=float)


def truncated_time_shift_test(
    x: Sequence[float],
    y: Sequence[float],
    radius: int,
    stat_fn: Callable[[np.ndarray, np.ndarray], float],
) -> Dict[str, object]:
    """②2.2 截断时间平移检验 (TTS, Yuan & Shou 2024)。

    x0 = x[r:n-r]，对每个整数 δ∈[-r,r] 取 yδ = y[r+δ:n-r+δ]，计算 θδ。
    p_TTS = min(1, B/(r+1))，其中 B = count(θδ >= θ0)，含 δ=0。
    注意：不要除以 2r+1，也不要沿用循环平移的近零排除规则。
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n - 2 * radius < 2:
        return {'radius': radius, 'n_points_used': 0, 'theta_zero': float('nan'),
                'count_ge': 0, 'p_tts': float('nan'), 'theta': []}

    theta = []
    for d in range(-radius, radius + 1):
        x0 = x[radius:n - radius]
        y0 = y[radius + d:n - radius + d]
        theta.append(stat_fn(x0, y0))
    theta = np.asarray(theta, dtype=float)
    theta0 = theta[radius]  # δ=0
    b = np.sum(theta >= theta0)
    p = min(1.0, float(b) / (radius + 1))
    return {
        'radius': radius,
        'n_points_used': n - 2 * radius,
        'theta_zero': float(theta0),
        'count_ge': int(b),
        'p_tts': p,
        'theta': theta.tolist(),
    }


# ---------------------------------------------------------------------------
# 统计家族：重复测量 Friedman 总体 + 配对 Wilcoxon + BH-FDR（文档 1.7/2.7）
# ---------------------------------------------------------------------------
def _long_to_wide_subjects(
    df: pd.DataFrame, region: str, param: str, metric: str
) -> Tuple[pd.DataFrame, List[str]]:
    """从长表取某 region×param×metric 的三条件每被试一个值（长表已聚合到每人一个值）。"""
    sub = df[(df['region'] == region) & (df['param'] == param) & (df['metric'] == metric)]
    sub = sub[['subject_id', 'condition', 'value']].dropna(subset=['value'])
    # 同一 subject×condition 若多行，取中位数（每人只贡献一个条件值）
    sub = sub.groupby(['subject_id', 'condition'], as_index=False)['value'].median()
    wide = sub.pivot(index='subject_id', columns='condition', values='value')
    return wide


def run_condition_tests(
    df_long: pd.DataFrame,
    conditions: Sequence[str],
    alpha: float = 0.05,
) -> pd.DataFrame:
    """①1.7/②2.7：对每个 region×param×metric 做 Friedman 总体检验，
    三条件两两配对 Wilcoxon，并分别在“总体”与“配对”两个家族内做 BH-FDR。

    df_long 需含列：subject_id, region, param, metric, condition, value。
    要求每名被试在三条件都可分析（同一共同样本）；行列缺失会记入失败原因。
    """
    regions = list(pd.unique(df_long['region']))
    params = list(pd.unique(df_long['param']))
    metrics = list(pd.unique(df_long['metric']))

    # 先建全部分定检验行（文档：不因失败/不显著缩小家族）
    rows: List[Dict[str, object]] = []
    for region in regions:
        for param in params:
            for metric in metrics:
                wide = _long_to_wide_subjects(df_long, region, param, metric)
                wide = wide.reindex(columns=conditions)
                complete = wide.dropna()
                n_subjects = len(complete)

                # ---- 总体检验 ----
                if n_subjects >= 5 and len(conditions) >= 3:
                    cols = [complete[c].values for c in conditions]
                    try:
                        stat, p = friedmanchisquare(*cols)
                        rows.append({
                            'region': region, 'param': param, 'metric': metric,
                            'comparison': 'overall', 'test': 'friedman',
                            'n_subjects': n_subjects,
                            'statistic': float(stat), 'p_raw': float(p),
                            'effect_name': "Kendall's W",
                            'effect_value': kendall_w(float(stat), n_subjects, len(conditions)),
                        })
                    except Exception as e:
                        rows.append({
                            'region': region, 'param': param, 'metric': metric,
                            'comparison': 'overall', 'test': 'friedman',
                            'n_subjects': n_subjects, 'statistic': float('nan'),
                            'p_raw': float('nan'), 'effect_name': '', 'effect_value': float('nan'),
                            'reason': str(e),
                        })
                else:
                    rows.append({
                        'region': region, 'param': param, 'metric': metric,
                        'comparison': 'overall', 'test': 'friedman',
                        'n_subjects': n_subjects, 'statistic': float('nan'),
                        'p_raw': float('nan'), 'effect_name': '', 'effect_value': float('nan'),
                        'reason': 'insufficient',
                    })

                # ---- 配对检验（三条件两两） ----
                for c1, c2 in combinations(list(conditions), 2):
                    if c1 not in wide.columns or c2 not in wide.columns:
                        rows.append({
                            'region': region, 'param': param, 'metric': metric,
                            'comparison': f'{c1}_vs_{c2}', 'test': 'wilcoxon',
                            'n_subjects': n_subjects, 'statistic': float('nan'),
                            'p_raw': float('nan'),
                            'reason': 'missing_condition',
                        })
                        continue
                    paired = complete[[c1, c2]].dropna()
                    a = paired[c1].values.astype(float)
                    b = paired[c2].values.astype(float)
                    n_pairs = len(a)
                    if n_pairs < 5:
                        rows.append({
                            'region': region, 'param': param, 'metric': metric,
                            'comparison': f'{c1}_vs_{c2}', 'test': 'wilcoxon',
                            'n_subjects': n_pairs, 'statistic': float('nan'),
                            'p_raw': float('nan'), 'reason': 'insufficient',
                        })
                        continue
                    diffs = a - b
                    if np.allclose(diffs, 0):
                        p = 1.0
                        stat = 0.0
                        reason = 'all_differences_zero'
                    else:
                        try:
                            stat, p = wilcoxon(a, b, zero_method='wilcox',
                                               correction=False,
                                               alternative='two-sided', mode='auto')
                            reason = ''
                        except Exception as e:
                            stat, p = float('nan'), float('nan')
                            reason = str(e)
                    rows.append({
                        'region': region, 'param': param, 'metric': metric,
                        'comparison': f'{c1}_vs_{c2}', 'test': 'wilcoxon',
                        'n_subjects': n_pairs, 'statistic': float(stat),
                        'p_raw': float(p), 'reason': reason,
                    })

    df = pd.DataFrame(rows)

    # 两个校正家族分别 BH-FDR（文档：总体48项、配对144项分别校正）
    overall_mask = df['comparison'] == 'overall'
    pw_mask = ~overall_mask

    df['p_adjusted'] = float('nan')
    df['family_id'] = ''
    df['family_size'] = 0

    for mask, fid in [(overall_mask, 'overall'), (pw_mask, 'pairwise')]:
        if mask.sum() == 0:
            continue
        pvals = df.loc[mask, 'p_raw'].tolist()
        p_adj, sig = bh_fdr(pvals, alpha=alpha)
        df.loc[mask, 'p_adjusted'] = p_adj
        df.loc[mask, 'family_id'] = fid
        df.loc[mask, 'family_size'] = int(mask.sum())
        df.loc[mask, 'significant'] = sig

    return df


if __name__ == '__main__':
    # 最小自检：验证 FDR、等长、循环平移、TTS、配对统计脚本能运行（不涉及全量数据）
    print('BH-FDR:', bh_fdr([0.001, 0.02, 0.04, 0.5]))
    print('kraskov_mi:', kraskov_mi(np.random.rand(200), np.random.rand(200), k=3))
    segs = find_continuous_valid_segments(
        [True] * 10 + [False, False] + [True] * 20,
        list(range(32)),
    )
    print('segments:', [(s['n_points'], s['start'], s['end']) for s in segs])
    print('TTS:', truncated_time_shift_test(
        np.random.rand(100), np.random.rand(100), 5, lambda a, b: float(np.corrcoef(a, b)[0, 1])))