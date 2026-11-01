#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PAC耦合统计分析脚本
Phase-Amplitude Coupling (PAC) Statistical Analysis

功能：
1. 加载PAC耦合详细结果数据
2. 异常值检测和去除
3. 绘制offset和exponent参数的小提琴图（按视频分组）
4. 进行两两统计检验
5. 生成详细的统计分析日志

作者：AI Assistant
创建时间：2025-01-28
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import mannwhitneyu, levene, shapiro
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class PACCouplingStatisticalAnalysis:
    """PAC耦合统计分析类"""
    
    def __init__(self, data_file_path, output_dir="f:/eye_eeg/ouhe/results"):
        """
        初始化分析类
        
        Parameters:
        -----------
        data_file_path : str
            PAC耦合详细结果CSV文件路径
        output_dir : str
            输出目录路径
        """
        self.data_file_path = data_file_path
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 设置日志
        self.setup_logging()
        
        # 初始化数据
        self.raw_data = None
        self.processed_data = None
        
        # 定义要分析的相关性指标
        self.correlation_metrics = {
            'pearson_r': 'Pearson相关系数',
            'spearman_r': 'Spearman相关系数', 
            'cross_corr': '互相关系数',
            'mutual_info': '互信息'
        }
        
        # 定义参数类型
        self.parameters = ['offset', 'exponent']
        
        # 存储所有指标列名
        self.metric_columns = []
        
    def setup_logging(self):
        """设置日志记录"""
        log_filename = f"{self.output_dir}/pac_coupling_statistical_analysis_{self.timestamp}.log"
        
        # 创建logger
        self.logger = logging.getLogger('PAC_Statistical_Analysis')
        self.logger.setLevel(logging.INFO)
        
        # 清除已有的handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # 创建文件handler
        file_handler = logging.FileHandler(log_filename, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # 创建控制台handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 创建formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 添加handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.logger.info("PAC耦合统计分析开始")
        
    def load_data(self):
        """加载PAC耦合数据"""
        self.logger.info("开始加载PAC耦合数据...")
        self.logger.info(f"正在加载数据文件: {self.data_file_path}")
        
        try:
            self.raw_data = pd.read_csv(self.data_file_path)
            self.logger.info(f"数据加载成功，共{len(self.raw_data)}行数据")
            self.logger.info(f"数据列数: {len(self.raw_data.columns)}")
            
            # 自动识别所有相关性指标列
            self.metric_columns = []
            for param in self.parameters:
                for metric in self.correlation_metrics.keys():
                    col_name = f"{param}_{metric}"
                    if col_name in self.raw_data.columns:
                        self.metric_columns.append(col_name)
            
            self.logger.info(f"识别到的相关性指标列: {self.metric_columns}")
            
            # 显示各视频数据量
            video_counts = self.raw_data['video_id'].value_counts().sort_index()
            self.logger.info(f"各视频数据量: {dict(video_counts)}")
            
            # 显示被试数量
            subject_count = self.raw_data['subject_id'].nunique()
            self.logger.info(f"被试数量: {subject_count}")
            
            # 显示每个指标的数据完整性
            self.logger.info("\\n各指标数据完整性:")
            for col in self.metric_columns:
                valid_count = self.raw_data[col].notna().sum()
                valid_percent = valid_count / len(self.raw_data) * 100
                self.logger.info(f"  {col}: {valid_count}/{len(self.raw_data)} ({valid_percent:.1f}%)")
            
            return True
            
        except Exception as e:
            self.logger.error(f"数据加载失败: {str(e)}")
            return False
    
    def detect_outliers(self, data, method='iqr', threshold=1.5):
        """
        检测异常值
        
        Parameters:
        -----------
        data : array-like
            输入数据
        method : str
            异常值检测方法 ('iqr', 'zscore')
        threshold : float
            阈值
            
        Returns:
        --------
        outlier_mask : boolean array
            异常值掩码
        """
        if method == 'iqr':
            Q1 = np.percentile(data, 25)
            Q3 = np.percentile(data, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outlier_mask = (data < lower_bound) | (data > upper_bound)
            
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(data))
            outlier_mask = z_scores > threshold
            
        return outlier_mask
    
    def remove_outliers(self, outlier_method='iqr', threshold=1.5):
        """
        去除异常值
        
        Parameters:
        -----------
        outlier_method : str
            异常值检测方法
        threshold : float
            阈值
        """
        self.logger.info("开始数据预处理...")
        
        # 复制原始数据
        self.processed_data = self.raw_data.copy()
        
        self.logger.info(f"开始异常值检测和去除，使用方法: {outlier_method}")
        
        # 存储所有异常值掩码
        all_outlier_masks = {}
        
        # 对每个相关性指标进行异常值检测
        for col in self.metric_columns:
            self.logger.info(f"\\n处理 {col} 的异常值...")
            
            # 获取有效数据
            valid_data = self.processed_data[col].dropna()
            
            if len(valid_data) == 0:
                self.logger.warning(f"  {col} 没有有效数据，跳过")
                continue
            
            # 检测异常值
            outliers = self.detect_outliers(valid_data, method=outlier_method, threshold=threshold)
            
            # 统计异常值信息
            n_outliers = outliers.sum()
            
            self.logger.info(f"{col} 异常值统计:")
            self.logger.info(f"  原始数据点数: {len(valid_data)}")
            self.logger.info(f"  检测到异常值: {n_outliers} ({n_outliers/len(valid_data)*100:.2f}%)")
            
            if n_outliers > 0:
                outlier_values = valid_data[outliers]
                normal_values = valid_data[~outliers]
                self.logger.info(f"  异常值范围: [{outlier_values.min():.4f}, {outlier_values.max():.4f}]")
                if len(normal_values) > 0:
                    self.logger.info(f"  正常值范围: [{normal_values.min():.4f}, {normal_values.max():.4f}]")
            
            # 创建异常值掩码
            outlier_mask = pd.Series(False, index=self.processed_data.index)
            valid_idx = self.processed_data[col].dropna().index
            outlier_mask.loc[valid_idx[outliers]] = True
            all_outlier_masks[col] = outlier_mask
        
        # 合并所有异常值掩码（任何一个指标有异常值就移除该行）
        total_outliers = pd.Series(False, index=self.processed_data.index)
        for mask in all_outlier_masks.values():
            total_outliers = total_outliers | mask
        
        # 去除异常值
        self.processed_data = self.processed_data[~total_outliers]
        
        self.logger.info("\\n异常值去除汇总:")
        self.logger.info(f"  原始数据总数: {len(self.raw_data)}")
        self.logger.info(f"  移除异常值总数: {total_outliers.sum()} ({total_outliers.sum()/len(self.raw_data)*100:.2f}%)")
        self.logger.info(f"  剩余数据总数: {len(self.processed_data)}")
        
        # 显示每个指标移除的异常值数量
        self.logger.info("\\n各指标移除的异常值数量:")
        for col, mask in all_outlier_masks.items():
            self.logger.info(f"  {col}: {mask.sum()}")
        
        # 显示去除异常值后的数据分布
        self.logger.info("\\n去除异常值后各视频的数据分布:")
        for video_id in sorted(self.processed_data['video_id'].unique()):
            video_data = self.processed_data[self.processed_data['video_id'] == video_id]
            self.logger.info(f"  Video {video_id}: {len(video_data)} 个数据点")
            
            # 显示每个指标的有效数据量
            for col in self.metric_columns:
                valid_count = video_data[col].notna().sum()
                self.logger.info(f"    {col}: {valid_count}")
        
        # 显示去除异常值后的数据统计
        self.logger.info("\\n去除异常值后的数据统计:")
        for col in self.metric_columns:
            valid_data = self.processed_data[col].dropna()
            if len(valid_data) > 0:
                self.logger.info(f"  {col}: Mean={valid_data.mean():.4f}, SD={valid_data.std():.4f}, n={len(valid_data)}")
    
    def plot_violin_plots(self):
        """绘制小提琴图 - 每个指标单独输出一张图"""
        self.logger.info("开始绘制小提琴图...")
        
        # 定义指定的颜色方案
        colors = ['#BAD3E4', '#F5DFA7', '#F1E4E6']
        
        # 为每个指标单独绘制一张图
        for idx, col in enumerate(self.metric_columns):
            self.logger.info(f"正在绘制 {col} 的小提琴图...")
            
            # 创建单独的图形
            plt.figure(figsize=(12, 8))
            
            # 准备数据并进行异常值检测
            plot_data = []
            videos = sorted(self.processed_data['video_id'].unique())
            
            # 对每个视频的数据进行异常值检测和去除
            cleaned_data = {}
            for video_id in videos:
                video_data = self.processed_data[self.processed_data['video_id'] == video_id]
                values = video_data[col].dropna()
                
                if len(values) > 0:
                    # 使用IQR方法检测异常值
                    Q1 = values.quantile(0.25)
                    Q3 = values.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 2.0 * IQR  # 使用2.0倍IQR，更宽松的阈值
                    upper_bound = Q3 + 2.0 * IQR
                    
                    # 去除异常值
                    clean_values = values[(values >= lower_bound) & (values <= upper_bound)]
                    cleaned_data[video_id] = clean_values
                    
                    # 记录异常值信息
                    n_outliers = len(values) - len(clean_values)
                    if n_outliers > 0:
                        self.logger.info(f"  Video {video_id}: 移除 {n_outliers} 个异常值 ({n_outliers/len(values)*100:.1f}%)")
                    
                    # 添加到绘图数据
                    for val in clean_values:
                        plot_data.append({'Video': f'Video {video_id}', 'Value': val})
            
            if not plot_data:
                self.logger.warning(f"{col} 没有有效数据，跳过绘图")
                continue
            
            df = pd.DataFrame(plot_data)
            
            # 绘制小提琴图
            ax = plt.gca()
            
            # 使用指定颜色绘制小提琴图
            violin_parts = plt.violinplot([cleaned_data[video_id] for video_id in videos if video_id in cleaned_data], 
                                        positions=range(len([v for v in videos if v in cleaned_data])), 
                                        showmeans=True, showmedians=True, showextrema=True)
            
            # 设置小提琴图颜色
            for i, pc in enumerate(violin_parts['bodies']):
                pc.set_facecolor(colors[i % len(colors)])
                pc.set_alpha(0.7)
                pc.set_edgecolor('black')
                pc.set_linewidth(1)
            
            # 设置均值线和中位数线的颜色
            violin_parts['cmeans'].set_color('red')
            violin_parts['cmeans'].set_linewidth(2)
            violin_parts['cmedians'].set_color('blue')
            violin_parts['cmedians'].set_linewidth(2)
            
            # 添加散点图层
            for i, video_id in enumerate([v for v in videos if v in cleaned_data]):
                values = cleaned_data[video_id]
                if len(values) > 0:
                    # 添加抖动效果的散点，颜色与对应的小提琴图一致
                    x_jitter = np.random.normal(i, 0.04, len(values))
                    scatter_color = colors[i % len(colors)]
                    plt.scatter(x_jitter, values, alpha=0.8, s=20, color=scatter_color, edgecolors='black', linewidth=0.5)
            
            # 设置x轴标签
            video_labels = [f'Video {v}' for v in videos if v in cleaned_data]
            plt.xticks(range(len(video_labels)), video_labels)
            
            # 设置标题和标签
            metric_name = col.replace('_', ' ').title()
            plt.title(f'PAC Coupling Analysis - {metric_name} Distribution by Video', 
                     fontsize=16, fontweight='bold', pad=20)
            # plt.xlabel('Video', fontsize=14, fontweight='bold')
            plt.ylabel(f'{metric_name}', fontsize=14, fontweight='bold')
            
            # 添加网格
            plt.grid(True, alpha=0.3, linestyle='--')
            
            # 添加统计信息框
            # stats_text = []
            # for i, video_id in enumerate([v for v in videos if v in cleaned_data]):
            #     values = cleaned_data[video_id]
            #     if len(values) > 0:
            #         mean_val = values.mean()
            #         median_val = values.median()
            #         std_val = values.std()
            #         stats_text.append(f'Video {video_id}: n={len(values)}, μ={mean_val:.3f}, Med={median_val:.3f}')
            
            # 在图上添加统计信息
            # stats_str = '\\n'.join(stats_text)
            # plt.text(0.02, 0.98, stats_str, transform=ax.transAxes, 
            #         verticalalignment='top', fontsize=10,
            #         bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'))
            
            # 添加图例
            legend_elements = [
                plt.Line2D([0], [0], color='red', linewidth=2, label='Mean'),
                plt.Line2D([0], [0], color='blue', linewidth=2, label='Median')
            ]
            plt.legend(handles=legend_elements, loc='upper right', framealpha=0.9)
            
            # 移除顶部和右侧边框
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # 调整布局
            plt.tight_layout()
            
            # 保存图形
            plot_filename = f"{self.output_dir}/pac_coupling_violin_{col}_{self.timestamp}.png"
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight', facecolor='white')
            self.logger.info(f"{col} 小提琴图已保存: {plot_filename}")
            
            plt.close()  # 关闭当前图形以释放内存
    
    def plot_overall_distribution(self):
        """绘制所有指标的总体分布图"""
        self.logger.info("绘制总体分布图...")
        
        # 计算需要的子图数量
        n_metrics = len(self.metric_columns)
        n_cols = 3
        n_rows = (n_metrics + n_cols - 1) // n_cols  # 向上取整
        
        # 创建图形
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        fig.suptitle('PAC Coupling Analysis - Overall Distribution (All Metrics)', fontsize=16, fontweight='bold')
        
        # 定义颜色
        colors = ['lightcoral', 'lightblue', 'lightgreen', 'lightyellow', 'lightpink', 'lightgray']
        
        # 为每个指标绘制总体分布
        for idx, col in enumerate(self.metric_columns):
            row = idx // n_cols
            col_idx = idx % n_cols
            ax = axes[row, col_idx]
            
            # 获取所有数据
            all_data = self.processed_data[col].dropna()
            
            if len(all_data) > 0:
                # 准备数据
                plot_data = [{'Category': 'All Videos', 'Value': val} for val in all_data]
                df = pd.DataFrame(plot_data)
                
                # 绘制小提琴图
                sns.violinplot(data=df, x='Category', y='Value', ax=ax, 
                              color=colors[idx % len(colors)])
                
                # 设置标题和标签
                metric_name = col.replace('_', ' ').title()
                ax.set_title(f'{metric_name} - Overall', fontsize=14, fontweight='bold')
                ax.set_xlabel('')
                ax.set_ylabel(f'{metric_name}', fontsize=12)
                ax.grid(True, alpha=0.3)
                
                # 添加统计信息
                mean_val = all_data.mean()
                std_val = all_data.std()
                median_val = all_data.median()
                ax.text(0.02, 0.98, f'Mean: {mean_val:.3f}\\nMedian: {median_val:.3f}\\nSD: {std_val:.3f}\\nn: {len(all_data)}',
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            else:
                ax.text(0.5, 0.5, f'No data for {col}', 
                       transform=ax.transAxes, ha='center', va='center')
                ax.set_title(f'{col} (No Data)')
        
        # 隐藏多余的子图
        for idx in range(n_metrics, n_rows * n_cols):
            row = idx // n_cols
            col_idx = idx % n_cols
            axes[row, col_idx].set_visible(False)
        
        plt.tight_layout()
        
        # 保存图形
        plot_filename = f"{self.output_dir}/pac_coupling_overall_distribution_{self.timestamp}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        self.logger.info(f"总体分布图已保存: {plot_filename}")
        
        plt.show()
    
    def calculate_effect_size(self, group1, group2):
        """计算Cohen's d效应量"""
        n1, n2 = len(group1), len(group2)
        pooled_std = np.sqrt(((n1 - 1) * group1.var() + (n2 - 1) * group2.var()) / (n1 + n2 - 2))
        cohens_d = (group1.mean() - group2.mean()) / pooled_std
        return cohens_d
    
    def interpret_effect_size(self, d):
        """解释效应量大小"""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "极小"
        elif abs_d < 0.5:
            return "小"
        elif abs_d < 0.8:
            return "中等"
        else:
            return "大"
    
    def perform_statistical_tests(self):
        """进行统计检验"""
        self.logger.info("开始统计检验...")
        
        videos = sorted(self.processed_data['video_id'].unique())
        
        # 对所有相关性指标进行检验
        for col in self.metric_columns:
            self.logger.info(f"\\n=== {col.upper()} 统计检验 ===")
            
            # 检查数据可用性
            available_videos = []
            for video in videos:
                video_data = self.processed_data[self.processed_data['video_id'] == video][col].dropna()
                if len(video_data) > 0:
                    available_videos.append(video)
            
            if len(available_videos) < 2:
                self.logger.warning(f"{col} 可用视频数量不足，跳过统计检验")
                continue
            
            self.logger.info(f"可用视频: {available_videos}")
            
            # 两两比较
            for i in range(len(available_videos)):
                for j in range(i + 1, len(available_videos)):
                    video1, video2 = available_videos[i], available_videos[j]
                    
                    self.logger.info(f"\\nvideo{video1} vs video{video2}:")
                    
                    # 获取数据
                    data1 = self.processed_data[self.processed_data['video_id'] == video1][col].dropna()
                    data2 = self.processed_data[self.processed_data['video_id'] == video2][col].dropna()
                    
                    if len(data1) == 0 or len(data2) == 0:
                        self.logger.warning(f"数据不足，跳过比较")
                        continue
                    
                    # 描述性统计
                    self.logger.info(f"video{video1}: Mean={data1.mean():.4f}, SD={data1.std():.4f}, n={len(data1)}")
                    self.logger.info(f"video{video2}: Mean={data2.mean():.4f}, SD={data2.std():.4f}, n={len(data2)}")
                    
                    # 正态性检验
                    if len(data1) >= 3 and len(data2) >= 3:
                        try:
                            # 对于大样本，使用Kolmogorov-Smirnov检验
                            if len(data1) > 5000 or len(data2) > 5000:
                                from scipy.stats import kstest
                                _, p_norm1 = kstest(data1, 'norm', args=(data1.mean(), data1.std()))
                                _, p_norm2 = kstest(data2, 'norm', args=(data2.mean(), data2.std()))
                            else:
                                _, p_norm1 = shapiro(data1[:5000])  # shapiro最多支持5000个样本
                                _, p_norm2 = shapiro(data2[:5000])
                            self.logger.info(f"Normality test - video{video1}: p={p_norm1:.4f}, video{video2}: p={p_norm2:.4f}")
                        except Exception as e:
                            self.logger.info(f"正态性检验失败: {str(e)}")
                    
                    # 方差齐性检验
                    try:
                        _, p_levene = levene(data1, data2)
                        self.logger.info(f"Levene's test for equal variances: p={p_levene:.4f}")
                    except Exception as e:
                        self.logger.info(f"方差齐性检验失败: {str(e)}")
                    
                    # Mann-Whitney U检验（非参数检验）
                    try:
                        statistic, p_value = mannwhitneyu(data1, data2, alternative='two-sided')
                        
                        # 显著性标记
                        if p_value < 0.001:
                            sig_mark = "***"
                        elif p_value < 0.01:
                            sig_mark = "**"
                        elif p_value < 0.05:
                            sig_mark = "*"
                        else:
                            sig_mark = "ns"
                        
                        self.logger.info("Mann-Whitney U test results:")
                        self.logger.info(f"Test statistic: {statistic:.4f}")
                        self.logger.info(f"p-value: {p_value:.6f} {sig_mark}")
                        
                        # 计算效应量
                        try:
                            cohens_d = self.calculate_effect_size(data1, data2)
                            effect_interpretation = self.interpret_effect_size(cohens_d)
                            
                            self.logger.info(f"Effect size (Cohen's d): {cohens_d:.4f}")
                            self.logger.info(f"Effect size interpretation: {effect_interpretation}")
                        except Exception as e:
                            self.logger.warning(f"效应量计算失败: {str(e)}")
                        
                    except Exception as e:
                        self.logger.error(f"统计检验失败: {str(e)}")
            
            # 如果有多个视频，进行Kruskal-Wallis检验（多组比较）
            if len(available_videos) > 2:
                self.logger.info(f"\\n=== {col} Kruskal-Wallis 多组比较 ===")
                try:
                    groups = []
                    group_names = []
                    for video in available_videos:
                        video_data = self.processed_data[self.processed_data['video_id'] == video][col].dropna()
                        if len(video_data) > 0:
                            groups.append(video_data)
                            group_names.append(f'Video {video}')
                    
                    if len(groups) > 2:
                        from scipy.stats import kruskal
                        statistic, p_value = kruskal(*groups)
                        
                        # 显著性标记
                        if p_value < 0.001:
                            sig_mark = "***"
                        elif p_value < 0.01:
                            sig_mark = "**"
                        elif p_value < 0.05:
                            sig_mark = "*"
                        else:
                            sig_mark = "ns"
                        
                        self.logger.info("Kruskal-Wallis test results:")
                        self.logger.info(f"Test statistic: {statistic:.4f}")
                        self.logger.info(f"p-value: {p_value:.6f} {sig_mark}")
                        self.logger.info(f"Groups compared: {', '.join(group_names)}")
                        
                except Exception as e:
                    self.logger.error(f"Kruskal-Wallis检验失败: {str(e)}")
        
        self.logger.info("\\n=== 统计检验完成 ===")
    
    def run_analysis(self):
        """运行完整分析流程"""
        try:
            # 1. 加载数据
            if not self.load_data():
                return False
            
            # 2. 去除异常值
            self.remove_outliers()
            
            # 3. 绘制小提琴图
            self.plot_violin_plots()
            
            # 4. 进行统计检验
            self.perform_statistical_tests()
            
            self.logger.info("PAC耦合统计分析完成！")
            return True
            
        except Exception as e:
            self.logger.error(f"分析过程中出现错误: {str(e)}")
            return False

def main():
    """主函数"""
    # 设置文件路径
    data_file = "F:/eye_eeg/ouhe/results/pac_coupling_detailed_20251030_153128.csv"
    output_dir = "f:/eye_eeg/ouhe/results"
    
    # 创建分析实例
    analyzer = PACCouplingStatisticalAnalysis(data_file, output_dir)
    
    # 运行分析
    success = analyzer.run_analysis()
    
    if success:
        print("\\n=== PAC耦合统计分析完成 ===")
        print(f"结果已保存到: {output_dir}")
        print(f"日志文件: pac_coupling_statistical_analysis_{analyzer.timestamp}.log")
        print(f"图形文件: pac_coupling_violin_plots_{analyzer.timestamp}.png")
    else:
        print("分析失败，请检查日志文件")

if __name__ == "__main__":
    main()