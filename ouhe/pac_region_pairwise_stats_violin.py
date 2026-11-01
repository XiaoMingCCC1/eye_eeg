#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按脑区进行PAC耦合指标的“同一脑区不同视频”配对统计检验，并在显著时绘制三视频散点图。

更新后的功能概述：
- 使用用户提供的电极清单将通道映射到六个标准脑区：前额叶、额叶、中央区、顶叶、颞叶、枕叶；
- 从CSV载入PAC耦合详细结果（包含所有被试、所有视频、所有通道）；
- 将数据聚合到“被试-视频-脑区”层级，按脑区对每位被试的各耦合指标取中位数；
- 对六个耦合指标（offset_spearman_r、offset_cross_corr、offset_mutual_info、exponent_spearman_r、exponent_cross_corr、exponent_mutual_info），
-  在每个脑区内对所有视频两两做配对检验（Wilcoxon优先，失败退回配对t检验）；
-  仅当“任意两个视频之间原始p值<alpha”时，绘制该脑区该指标的三视频散点图；
- 记录中文日志，并保存统计结果CSV。

使用示例：
python f:/eye_eeg/ouhe/pac_region_pairwise_stats_violin.py \
  --data f:/eye_eeg/ouhe/results_old/pac_coupling_detailed_20251030_153128.csv \
  --alpha 0.05 \
  --outdir f:/eye_eeg/ouhe/results_2

作者：AI Assistant
时间：2026-01-11
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon, ttest_rel, mannwhitneyu

# 中文显示设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def _to_int(ch):
    """将'E123'或'Ch23'或纯数字转换为int通道号。"""
    if pd.isna(ch):
        return None
    s = str(ch)
    digits = ''.join([c for c in s if c.isdigit()])
    return int(digits) if digits else None


def build_region_mapping():
    """根据用户提供的电极清单构建通道号->脑区映射字典。"""
    regions = {
        'prefrontal': [
            # 中央区域
            'E1','E8','E14','E15','E21','E25',
            # 左前额叶
            'E22','E26','E32','E33','E38','E39','E43','E44','E49',
            # 右前额叶
            'E2','E9','E113','E114','E115','E120','E121','E122'
        ],
        'frontal': [
            # 中央区域
            'E6','E11','E16',
            # 左额叶
            'E18','E23','E27','E34','E40',
            # 右额叶
            'E3','E10','E116','E123'
        ],
        'central': [
            # 中线
            'E7','E31','E55','E80','E106',
            # 左中央
            'E12','E13','E19','E20','E24','E28','E29','E30','E35','E36','E37',
            # 右中央
            'E4','E5','E104','E105','E110','E111','E112','E117','E118','E124'
        ],
        'parietal': [
            # 中央区域
            'E62','E72',
            # 左顶叶
            'E41','E42','E46','E47','E51','E52','E53','E54','E59','E60','E61',
            # 右顶叶
            'E84','E85','E86','E87','E91','E92','E93','E97','E98','E102','E103'
        ],
        'temporal': [
            # 左颞叶
            'E45','E50','E56','E57','E58','E63','E64','E65','E68','E69',
            # 右颞叶
            'E94','E95','E96','E99','E100','E101','E107','E108','E109'
        ],
        'occipital': [
            # 中央区域
            'E70','E71','E75','E76','E77','E78','E79','E81','E82','E83',
            # 左枕叶
            'E66','E67','E88','E89',
            # 右枕叶
            'E90'
        ],
    }

    mapping = {}
    for region, labels in regions.items():
        for lab in labels:
            ch = _to_int(lab)
            if ch is not None:
                mapping[ch] = region
    return mapping


REGION_ORDER = ['prefrontal', 'frontal', 'central', 'parietal', 'temporal', 'occipital']
REGION_CN = {
    'prefrontal': '前额叶',
    'frontal': '额叶',
    'central': '中央区',
    'parietal': '顶叶',
    'temporal': '颞叶',
    'occipital': '枕叶',
}

# 英文显示名称（用于图像）
REGION_EN = {
    'prefrontal': 'Prefrontal',
    'frontal': 'Frontal',
    'central': 'Central',
    'parietal': 'Parietal',
    'temporal': 'Temporal',
    'occipital': 'Occipital',
}

METRICS = [
    'offset_spearman_r',
    'offset_cross_corr',
    'offset_mutual_info',
    'exponent_spearman_r',
    'exponent_cross_corr',
    'exponent_mutual_info',
]


def _subject_video_column_names(df: pd.DataFrame) -> tuple:
    """智能检测subject和video列名（兼容subject/subject_id、video_id/video/Video等）。"""
    subj_candidates = ['subject_id', 'subject', 'Subj', 'subj', 'ID']
    video_candidates = ['video_id', 'video', 'Video', 'stimulus', 'condition']
    subj_col = None
    video_col = None
    for c in subj_candidates:
        if c in df.columns:
            subj_col = c
            break
    for c in video_candidates:
        if c in df.columns:
            video_col = c
            break
    if subj_col is None or video_col is None:
        raise RuntimeError('数据中缺少subject/subject_id或video/video_id字段，请检查CSV列名。')
    return subj_col, video_col


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def setup_logger(outdir: str) -> logging.Logger:
    ensure_dir(outdir)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    logfile = os.path.join(outdir, f'pac_region_pairwise_stats_{ts}.log')

    logger = logging.getLogger('PAC_REGION_PAIRWISE')
    logger.setLevel(logging.INFO)
    for h in list(logger.handlers):
        logger.removeHandler(h)

    fh = logging.FileHandler(logfile, encoding='utf-8')
    ch = logging.StreamHandler()
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info('按脑区进行视频间PAC耦合配对统计检验开始')
    logger.info(f'日志文件: {logfile}')
    return logger


def significance_stars(p: float) -> str:
    if p < 1e-4:
        return '***'
    if p < 1e-3:
        return '***'
    if p < 1e-2:
        return '**'
    if p < 0.05:
        return '*'
    return ''


def cohens_d_paired(a: np.ndarray, b: np.ndarray) -> float:
    """配对Cohen's d（differences的均值/标准差）。"""
    mask = np.isfinite(a) & np.isfinite(b)
    diffs = (a[mask] - b[mask]).astype(float)
    if len(diffs) < 2:
        return np.nan
    sd = float(np.std(diffs, ddof=1))
    if sd == 0.0:
        return 0.0
    return float(np.mean(diffs) / sd)


def paired_stat(a: np.ndarray, b: np.ndarray):
    """做配对统计检验：优先Wilcoxon，失败时退回配对t检验。返回(test_name, stat, p, n)。"""
    mask = np.isfinite(a) & np.isfinite(b)
    a_valid = a[mask]
    b_valid = b[mask]
    n = int(len(a_valid))
    if n < 5:
        return ('insufficient', np.nan, np.nan, n)
    diffs = a_valid - b_valid
    if np.allclose(diffs, 0):
        return ('all_zero', 0.0, 1.0, n)
    try:
        stat, p = wilcoxon(a_valid, b_valid, zero_method='wilcox', correction=False, alternative='two-sided', mode='auto')
        return ('wilcoxon', float(stat), float(p), n)
    except Exception:
        stat, p = ttest_rel(a_valid, b_valid, nan_policy='omit')
        return ('ttest_rel', float(stat), float(p), n)


def aggregate_subject_region_video(df: pd.DataFrame, region_map: dict, metrics: list, subj_col: str, video_col: str) -> pd.DataFrame:
    """将通道级数据聚合到被试-视频-脑区层级（各指标取中位数）。"""
    if 'channel' not in df.columns:
        raise ValueError('CSV缺少列: channel')
    df = df.copy()
    df['channel_id'] = df['channel'].apply(_to_int)
    df['region'] = df['channel_id'].map(region_map)
    # 仅保留映射到脑区的记录
    df = df[df['region'].notna()]

    # 分组聚合
    agg = df.groupby([subj_col, video_col, 'region'])[metrics].median().reset_index()
    return agg


def build_video_subject_values(df_agg: pd.DataFrame, region: str, metric: str, subj_col: str, video_col: str) -> dict:
    """按视频汇总为每被试一个值（同一subject同一视频在该脑区该指标的中位数）。"""
    sub_df = df_agg[df_agg['region'] == region][[subj_col, video_col, metric]].dropna(subset=[metric])
    vid_map = {}
    for v, g in sub_df.groupby(video_col):
        s = g.groupby(subj_col)[metric].median()
        vid_map[str(v)] = s
    return vid_map


def get_paired_arrays(vid_map: dict, v1: str, v2: str) -> tuple:
    """返回按被试配对的两组数组及配对数量。"""
    s1 = vid_map.get(str(v1), pd.Series(dtype=float))
    s2 = vid_map.get(str(v2), pd.Series(dtype=float))
    common = s1.index.intersection(s2.index)
    a = s1.loc[common].values.astype(float)
    b = s2.loc[common].values.astype(float)
    return a, b, int(len(common))


def plot_scatter_three_groups(group_values, group_labels, title, outfile):
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ['#4C78A8', '#F58518', '#54A24B']
    centers = np.linspace(1.0, 1.0 + 0.35 * (len(group_labels) - 1), len(group_labels))
    for i, vals in enumerate(group_values):
        vals = np.asarray(vals)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue
        x = np.random.normal(centers[i], 0.03, len(vals))
        ax.scatter(x, vals, s=30, alpha=0.85, color=colors[i % len(colors)], edgecolors='none')
        m = float(np.nanmean(vals)) if len(vals) > 0 else np.nan
        if np.isfinite(m):
            ax.hlines(m, centers[i] - 0.12, centers[i] + 0.12, colors=colors[i % len(colors)], linewidth=3, alpha=0.9)

    ax.set_xticks(centers)
    ax.set_xticklabels(group_labels, fontsize=16)
    ax.set_xlabel('Video', fontsize=18)
    ax.set_ylabel('Value', fontsize=18)
    ax.tick_params(axis='both', labelsize=16)
    ax.set_title(title, fontsize=20, fontweight='bold')
    ax.grid(True, alpha=0.25, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_pair_violin(values_a, values_b, name_a, name_b, metric, outdir, logger):
    title_metric = metric.replace('_', ' ').title()
    fig, ax = plt.subplots(figsize=(8, 6))

    data = [values_a.dropna().values, values_b.dropna().values]
    parts = ax.violinplot(data, positions=[0, 1], showmeans=True, showmedians=True, showextrema=True)

    colors = ['#CEE0EC', '#F7E8C1']
    for i, b in enumerate(parts['bodies']):
        b.set_facecolor(colors[i % len(colors)])
        b.set_alpha(0.75)
        b.set_edgecolor('black')
        b.set_linewidth(1)
    parts['cmeans'].set_color('red')
    parts['cmedians'].set_color('blue')

    # 抖动散点
    for i, vals in enumerate(data):
        if len(vals) > 0:
            x = np.random.normal(i, 0.04, len(vals))
            ax.scatter(x, vals, s=15, alpha=0.7, color=colors[i % len(colors)], edgecolors='black', linewidths=0.3)

    ax.set_xticks([0, 1])
    ax.set_xticklabels([name_a, name_b])
    ax.set_ylabel(title_metric)
    ax.set_title(f'{title_metric}: {name_a} vs {name_b}')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ensure_dir(outdir)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    fname = f'violin_{metric}_{name_a}_vs_{name_b}_{ts}.png'.replace(' ', '')
    fpath = os.path.join(outdir, fname)
    plt.tight_layout()
    plt.savefig(fpath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logger.info(f'小提琴图已保存: {fpath}')


def main(args):
    outdir = args.outdir
    ensure_dir(outdir)
    logger = setup_logger(outdir)

    # 载入数据
    csv_path = args.data
    logger.info(f'正在加载数据文件: {csv_path}')
    df = pd.read_csv(csv_path)
    logger.info(f'数据加载成功: {len(df)} 行, {len(df.columns)} 列')

    # 构建脑区映射
    region_map = build_region_mapping()
    logger.info(f'脑区映射条目数: {len(region_map)} (由电极清单直接构建)')

    # 检查指标列
    missing = [m for m in METRICS if m not in df.columns]
    if missing:
        logger.error(f'缺少指标列: {missing}')
        sys.exit(1)

    # 识别被试与视频列
    subj_col, video_col = _subject_video_column_names(df)
    videos = list(pd.unique(df[video_col].dropna()))
    alpha = float(args.alpha)
    logger.info(f'将在每个脑区内对6个指标进行视频两两配对检验，alpha={alpha}')

    # 聚合到被试-视频-脑区层级
    df_agg = aggregate_subject_region_video(df, region_map, METRICS, subj_col, video_col)
    cov = df_agg['region'].value_counts().reindex(REGION_ORDER).fillna(0).astype(int)
    logger.info('脑区（被试-视频聚合后）数据覆盖度:')
    for r in REGION_ORDER:
        logger.info(f'  {REGION_CN[r]}({r}): {cov.get(r, 0)} 条')

    # 统计检验与结果记录
    stats_records = []
    for region in REGION_ORDER:
        for metric in METRICS:
            # 构造视频->被试值映射
            vid_map = build_video_subject_values(df_agg, region, metric, subj_col, video_col)
            # 两两视频配对检验
            for v1, v2 in combinations(videos, 2):
                a, b, n_pairs = get_paired_arrays(vid_map, str(v1), str(v2))
                test_name, stat, p, n = paired_stat(a, b)
                # 组内均值、样本数报告（用于日志）
                m1 = float(np.mean(a)) if len(a) > 0 else float('nan')
                s1 = float(np.std(a, ddof=1)) if len(a) > 1 else 0.0
                m2 = float(np.mean(b)) if len(b) > 0 else float('nan')
                s2 = float(np.std(b, ddof=1)) if len(b) > 1 else 0.0
                d = cohens_d_paired(a, b)

                stats_records.append({
                    'region': region,
                    'metric': metric,
                    'video_a': str(v1),
                    'video_b': str(v2),
                    'test': test_name,
                    'stat': stat,
                    'p_raw': p,
                    'n_pairs': n,
                    'mean_a': m1,
                    'sd_a': s1,
                    'mean_b': m2,
                    'sd_b': s2,
                    'cohens_d_paired': d,
                })

    stats_df = pd.DataFrame.from_records(stats_records)
    stats_path = os.path.join(outdir, 'pac_region_video_stats.csv')
    stats_df['significant_raw'] = stats_df['p_raw'] < alpha
    stats_df.to_csv(stats_path, index=False)
    logger.info(f'统计结果CSV已保存: {stats_path}')

    # 写日志（中文）
    for metric in METRICS:
        logger.info(f'=== 指标（被试配对，脑区内视频对比）: {metric} ===')
        for region in REGION_ORDER:
            sub = stats_df[(stats_df['metric'] == metric) & (stats_df['region'] == region)]
            if len(sub) == 0:
                continue
            logger.info(f'-- 脑区：{REGION_CN[region]}({region}) --')
            for _, row in sub.iterrows():
                v1 = row['video_a']
                v2 = row['video_b']
                logger.info(f'视频 {v1} vs {v2}:')
                logger.info(f'{v1}: Mean={row["mean_a"]:.4f}, SD={row["sd_a"]:.4f}, n={int(row["n_pairs"])}')
                logger.info(f'{v2}: Mean={row["mean_b"]:.4f}, SD={row["sd_b"]:.4f}, n={int(row["n_pairs"])}')
                logger.info(f'检验: {row["test"]}, stat={row["stat"] if pd.notna(row["stat"]) else "NA"}')
                stars = significance_stars(float(row['p_raw']) if pd.notna(row['p_raw']) else 1.0)
                logger.info(f'p值(raw): {float(row["p_raw"]) if pd.notna(row["p_raw"]) else 1.0:.6f} {stars}')
                d = row['cohens_d_paired']
                if pd.isna(d):
                    logger.info('效应量(Cohen\'s d): NA')
                else:
                    logger.info(f'效应量(Cohen\'s d): {float(d):.4f}')

    # 绘图：仅当任意视频对raw显著时，绘制三视频小提琴图（英文标签）
    videos_sorted = [str(v) for v in videos]
    for region in REGION_ORDER:
        for metric in METRICS:
            sub = stats_df[(stats_df['region'] == region) & (stats_df['metric'] == metric)]
            if len(sub) == 0:
                continue
            if not any(sub['significant_raw']):
                continue
            # 组装三视频数据
            vid_map = build_video_subject_values(df_agg, region, metric, subj_col, video_col)
            group_values = []
            group_labels = []
            for v in videos_sorted:
                s = vid_map.get(v, pd.Series(dtype=float))
                group_values.append(s.values.astype(float))
                group_labels.append(v)
            title_metric = metric.replace('_', ' ').title()
            title = f'{title_metric} - {REGION_EN[region]} '
            fname = f'scatter_{metric}_{region}_videos.png'
            fpath = os.path.join(outdir, fname)
            plot_scatter_three_groups(group_values, group_labels, title, fpath)
            logger.info(f'显著结果图已保存: {fpath}')

    logger.info('分析完成。')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='区域两两检验与显著小提琴图绘制')
    parser.add_argument('--data', type=str, required=False,
                        default='f:/eye_eeg/ouhe/results_old/pac_coupling_detailed_20251030_153128.csv',
                        help='PAC耦合CSV路径')
    parser.add_argument('--outdir', type=str, required=False,
                        default='f:/eye_eeg/ouhe/results_3',
                        help='输出目录（图与CSV保存到该目录）')
    parser.add_argument('--alpha', type=float, required=False, default=0.05,
                        help='显著性水平alpha（默认0.05）')

    args = parser.parse_args()
    main(args)
