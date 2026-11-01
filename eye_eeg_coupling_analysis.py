#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
眼动-脑电数据耦合分析 (改进版)
基于Cross Frequency Coupling方法分析眼动数据与脑电数据之间的耦合性
使用PLV、MVL、MI三种方法计算耦合强度

改进内容：
1. 添加数据预处理（去趋势、标准化、滤波）
2. 改进相位计算方法
3. 提高时间分辨率
4. 优化频率匹配
5. 添加统计显著性检验

作者: AI Assistant
创建时间: 2025-01-12
修改时间: 2025-01-12
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from scipy.signal import hilbert, butter, filtfilt, detrend
from scipy import signal, stats
from sklearn.preprocessing import StandardScaler
import logging
from datetime import datetime

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EyeEEGCouplingAnalyzer:
    """眼动-脑电耦合分析器 (改进版)"""
    
    def __init__(self, sampling_rate=120.0):  # 使用真实眼动数据的120Hz采样率
        """初始化分析器"""
        self.eye_data = {}  # 存储眼动数据
        self.eeg_data = {}  # 存储脑电数据
        self.coupling_results = {}  # 存储耦合分析结果
        self.sampling_rate = sampling_rate  # 采样率 (Hz)
        
    def preprocess_signal(self, signal_data, method='detrend_standardize'):
        """
        信号预处理
        
        参数:
        signal_data: 输入信号
        method: 预处理方法
        
        返回:
        预处理后的信号
        """
        # 去除NaN值
        signal_clean = signal_data[~np.isnan(signal_data)]
        
        if len(signal_clean) < 10:  # 数据太少
            return signal_data
        
        # 去趋势
        if 'detrend' in method:
            signal_clean = detrend(signal_clean)
        
        # 标准化
        if 'standardize' in method:
            if np.std(signal_clean) > 1e-10:  # 避免除零
                signal_clean = (signal_clean - np.mean(signal_clean)) / np.std(signal_clean)
        
        return signal_clean
    
    def bandpass_filter(self, data, lowcut, highcut, fs, order=4):
        """
        带通滤波器
        
        参数:
        data: 输入数据
        lowcut: 低频截止频率
        highcut: 高频截止频率
        fs: 采样频率
        order: 滤波器阶数
        
        返回:
        滤波后的数据
        """
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        
        if high >= 1.0:
            high = 0.99
        if low <= 0:
            low = 0.01
            
        b, a = butter(order, [low, high], btype='band')
        filtered_data = filtfilt(b, a, data)
        return filtered_data
    
    def extract_instantaneous_phase(self, signal_data, fs, freq_band=None):
        """
        提取瞬时相位
        
        参数:
        signal_data: 输入信号
        fs: 采样频率
        freq_band: 频率带 [low, high]，如果为None则不滤波
        
        返回:
        瞬时相位
        """
        # 预处理
        processed_signal = self.preprocess_signal(signal_data)
        
        # 如果指定了频率带，进行滤波
        if freq_band is not None:
            processed_signal = self.bandpass_filter(
                processed_signal, freq_band[0], freq_band[1], fs
            )
        
        # 希尔伯特变换提取瞬时相位
        analytic_signal = hilbert(processed_signal)
        instantaneous_phase = np.angle(analytic_signal)
        
        return instantaneous_phase
    
    def extract_instantaneous_amplitude(self, signal_data, fs, freq_band=None):
        """
        提取瞬时振幅
        
        参数:
        signal_data: 输入信号
        fs: 采样频率
        freq_band: 频率带 [low, high]，如果为None则不滤波
        
        返回:
        瞬时振幅
        """
        # 预处理
        processed_signal = self.preprocess_signal(signal_data)
        
        # 如果指定了频率带，进行滤波
        if freq_band is not None:
            processed_signal = self.bandpass_filter(
                processed_signal, freq_band[0], freq_band[1], fs
            )
        
        # 希尔伯特变换提取瞬时振幅
        analytic_signal = hilbert(processed_signal)
        instantaneous_amplitude = np.abs(analytic_signal)
        
        return instantaneous_amplitude
    
    def load_eye_data_from_existing_files(self, video_name):
        """
        从STFT处理后的眼动数据文件中读取数据
        
        参数:
        video_name: 视频名称 (Video1, Video2, Video3)
        """
        logger.info(f"正在读取{video_name}的STFT处理后眼动数据...")
        
        # 构建STFT处理后的眼动数据文件路径
        if video_name == "Video1":
            eye_data_path = "f:/eye_eeg/pupil_size_data_A00053375_Video1_SMP_100s_20251014_212035.log"
        elif video_name == "Video2":
            eye_data_path = "f:/eye_eeg/pupil_size_data_A00053375_Video2_SMP_280s_20251014_212036.log"
        elif video_name == "Video3":
            eye_data_path = "f:/eye_eeg/pupil_size_data_A00053375_Video3_SMP_274s_20251014_212036.log"
        else:
            logger.error(f"不支持的视频名称: {video_name}")
            return False
        
        if not os.path.exists(eye_data_path):
            logger.error(f"STFT处理后的眼动数据文件不存在: {eye_data_path}")
            return False
        
        try:
            # 读取STFT处理后的眼动数据文件
            with open(eye_data_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # 找到数据开始的行（跳过头部信息）
            data_start_idx = None
            for i, line in enumerate(lines):
                if line.strip().startswith('# 时间(秒)'):
                    data_start_idx = i + 1
                    break
            
            if data_start_idx is None:
                logger.error(f"无法找到数据开始行在文件: {eye_data_path}")
                return False
            
            # 解析数据
            timestamps = []
            pupil_sizes = []
            
            for line in lines[data_start_idx:]:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split('\t')
                if len(parts) < 2:
                    continue
                
                try:
                    # 提取时间（秒）和瞳孔大小估计值
                    time_sec = float(parts[0].strip())
                    pupil_size = float(parts[1].strip())
                    
                    timestamps.append(time_sec)
                    pupil_sizes.append(pupil_size)
                    
                except (ValueError, IndexError):
                    continue
            
            if len(timestamps) == 0:
                logger.error(f"没有找到有效的STFT眼动数据在文件: {eye_data_path}")
                return False
            
            # 转换为numpy数组
            time_seconds = np.array(timestamps)
            pupil_size = np.array(pupil_sizes)
            
            # 处理缺失值（如果有的话）
            valid_indices = ~np.isnan(pupil_size)
            if np.sum(valid_indices) < len(pupil_size) * 0.5:
                logger.warning(f"{video_name}的瞳孔数据缺失值过多，可能影响分析结果")
            
            # 插值填补缺失值
            if np.any(~valid_indices):
                pupil_size = pd.Series(pupil_size).interpolate(method='linear').fillna(method='bfill').fillna(method='ffill').values
            
            # 数据已经是STFT处理后的结果，直接使用
            self.eye_data[video_name] = {
                'time': time_seconds,
                'pupil_size': pupil_size,
                'original_time': time_seconds,
                'original_pupil_size': pupil_size
            }
            
            logger.info(f"{video_name}STFT处理后眼动数据读取成功，数据点: {len(timestamps)}个时间点，采样率{self.sampling_rate}Hz")
            return True
            
        except Exception as e:
            logger.error(f"读取STFT处理后眼动数据文件时出错: {e}")
            return False
        
    def load_eeg_data(self, file_path, video_name):
        """
        加载脑电数据 (保持原有逻辑)
        """
        try:
            logger.info(f"正在加载{video_name}的脑电数据: {file_path}")
            
            if not os.path.exists(file_path):
                logger.error(f"脑电数据文件不存在: {file_path}")
                return False
                
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            channels_data = {}
            
            for channel in [25, 68]:
                logger.info(f"解析通道{channel}数据...")
                
                channel_pattern = f"=== 通道 {channel} 参数提取开始 ==="
                if channel_pattern not in content:
                    logger.warning(f"未找到通道{channel}的数据")
                    continue
                
                channel_start = content.find(channel_pattern)
                next_channel_pattern = f"=== 通道 {25 if channel == 68 else 68} 参数提取开始 ==="
                if next_channel_pattern in content:
                    next_channel_start = content.find(next_channel_pattern)
                    if next_channel_start > channel_start:
                        channel_content = content[channel_start:next_channel_start]
                    else:
                        channel_content = content[channel_start:]
                else:
                    channel_content = content[channel_start:]
                
                time_points = []
                offsets = []
                exponents = []
                
                time_point_pattern = r'时间点 (\d+):'
                offset_pattern = r'偏移\(offset\): ([\d.]+)'
                exponent_pattern = r'指数\(exponent\): ([\d.]+)'
                
                time_matches = re.finditer(time_point_pattern, channel_content)
                for time_match in time_matches:
                    time_point = int(time_match.group(1))
                    
                    start_pos = time_match.end()
                    next_time_match = re.search(time_point_pattern, channel_content[start_pos:])
                    if next_time_match:
                        end_pos = start_pos + next_time_match.start()
                        section = channel_content[start_pos:end_pos]
                    else:
                        section = channel_content[start_pos:]
                    
                    offset_match = re.search(offset_pattern, section)
                    exponent_match = re.search(exponent_pattern, section)
                    
                    if offset_match and exponent_match:
                        time_points.append(time_point)
                        offsets.append(float(offset_match.group(1)))
                        exponents.append(float(exponent_match.group(1)))
                
                if len(time_points) > 0:
                    channels_data[channel] = {
                        'time_points': np.array(time_points),
                        'offsets': np.array(offsets),
                        'exponents': np.array(exponents)
                    }
                    logger.info(f"通道{channel}数据解析成功，共{len(time_points)}个时间点")
                else:
                    logger.warning(f"通道{channel}未找到有效数据")
            
            if len(channels_data) == 0:
                logger.error(f"未能从文件中解析到任何通道数据: {file_path}")
                return False
                
            self.eeg_data[video_name] = channels_data
            logger.info(f"{video_name}脑电数据加载成功，包含通道: {list(channels_data.keys())}")
            return True
            
        except Exception as e:
            logger.error(f"加载脑电数据时出错: {e}")
            return False
    
    def PLV_CFC_improved(self, phase_low, amp_high):
        """
        改进的相位锁定值(PLV)计算
        """
        # 提取高频信号的瞬时相位
        phase_high = np.angle(hilbert(amp_high))
        
        # 计算相位差
        phase_diff = phase_low - phase_high
        
        # 计算PLV
        plv = np.abs(np.mean(np.exp(1j * phase_diff)))
        
        return plv
    
    def MVL_CFC_improved(self, phase_low, amp_high):
        """
        改进的平均向量长度(MVL)计算
        """
        # 振幅归一化 (使用z-score标准化)
        if np.std(amp_high) > 1e-10:
            amp_normalized = (amp_high - np.mean(amp_high)) / np.std(amp_high)
        else:
            amp_normalized = amp_high - np.mean(amp_high)
        
        # 将振幅转换为正值并归一化到[0,1]
        amp_positive = amp_normalized - np.min(amp_normalized)
        if np.max(amp_positive) > 1e-10:
            amp_positive = amp_positive / np.max(amp_positive)
        
        # 计算MVL
        mvl = np.abs(np.mean(amp_positive * np.exp(1j * phase_low)))
        
        return mvl
    
    def MI_CFC_improved(self, phase_low, amp_high, nbins=18):
        """
        改进的互信息(MI)计算
        """
        # 相位分箱
        phase_bins = np.linspace(-np.pi, np.pi, nbins + 1)
        
        # 计算每个相位区间的振幅均值
        mean_amp_per_bin = np.zeros(nbins)
        
        for i in range(nbins):
            # 找到在当前相位区间内的数据点
            in_bin = (phase_low >= phase_bins[i]) & (phase_low < phase_bins[i + 1])
            
            if np.sum(in_bin) > 0:
                mean_amp_per_bin[i] = np.mean(amp_high[in_bin])
            else:
                mean_amp_per_bin[i] = 0
        
        # 避免零值
        mean_amp_per_bin = mean_amp_per_bin + 1e-10
        
        # 归一化为概率分布
        prob_dist = mean_amp_per_bin / np.sum(mean_amp_per_bin)
        prob_dist = np.maximum(prob_dist, 1e-10)  # 避免log(0)
        
        # 计算熵
        entropy = -np.sum(prob_dist * np.log(prob_dist))
        
        # 计算KL散度和归一化MI
        max_entropy = np.log(nbins)
        kl_divergence = max_entropy - entropy
        mi = kl_divergence / max_entropy
        
        return mi, mean_amp_per_bin
    
    def statistical_significance_test(self, phase_low, amp_high, n_surrogates=100):
        """
        统计显著性检验
        
        参数:
        phase_low: 低频相位
        amp_high: 高频振幅
        n_surrogates: 代理数据数量
        
        返回:
        p值字典
        """
        # 计算原始耦合值
        original_plv = self.PLV_CFC_improved(phase_low, amp_high)
        original_mvl = self.MVL_CFC_improved(phase_low, amp_high)
        original_mi, _ = self.MI_CFC_improved(phase_low, amp_high)
        
        # 生成代理数据并计算耦合值
        surrogate_plv = []
        surrogate_mvl = []
        surrogate_mi = []
        
        for _ in range(n_surrogates):
            # 随机打乱相位，保持振幅不变
            shuffled_phase = np.random.permutation(phase_low)
            
            plv_surr = self.PLV_CFC_improved(shuffled_phase, amp_high)
            mvl_surr = self.MVL_CFC_improved(shuffled_phase, amp_high)
            mi_surr, _ = self.MI_CFC_improved(shuffled_phase, amp_high)
            
            surrogate_plv.append(plv_surr)
            surrogate_mvl.append(mvl_surr)
            surrogate_mi.append(mi_surr)
        
        # 计算p值
        p_plv = np.sum(np.array(surrogate_plv) >= original_plv) / n_surrogates
        p_mvl = np.sum(np.array(surrogate_mvl) >= original_mvl) / n_surrogates
        p_mi = np.sum(np.array(surrogate_mi) >= original_mi) / n_surrogates
        
        return {
            'p_plv': p_plv,
            'p_mvl': p_mvl,
            'p_mi': p_mi,
            'surrogate_plv': surrogate_plv,
            'surrogate_mvl': surrogate_mvl,
            'surrogate_mi': surrogate_mi
        }
    
    def calculate_coupling(self, video_name):
        """
        计算改进的眼动-脑电耦合
        """
        if video_name not in self.eye_data or video_name not in self.eeg_data:
            logger.error(f"{video_name}的眼动或脑电数据未加载")
            return False
        
        logger.info(f"开始计算{video_name}的改进眼动-脑电耦合...")
        
        eye_data = self.eye_data[video_name]
        eeg_data = self.eeg_data[video_name]
        
        results = {}
        
        for channel in eeg_data.keys():
            logger.info(f"计算通道{channel}的耦合...")
            
            channel_data = eeg_data[channel]
            
            # 获取时间对齐的数据
            eeg_time_points = channel_data['time_points']
            eye_time_points = eye_data['time']
            
            # 找到时间范围的交集
            min_time = max(np.min(eeg_time_points), np.min(eye_time_points))
            max_time = min(np.max(eeg_time_points), np.max(eye_time_points))
            
            # 创建高分辨率的公共时间轴
            dt = 1.0 / self.sampling_rate
            common_time = np.arange(min_time, max_time, dt)
            
            # 插值对齐数据
            eye_pupil_interp = np.interp(common_time, eye_time_points, eye_data['pupil_size'])
            eeg_offset_interp = np.interp(common_time, eeg_time_points, channel_data['offsets'])
            eeg_exponent_interp = np.interp(common_time, eeg_time_points, channel_data['exponents'])
            
            # 提取瞳孔的低频相位 (0.01-2 Hz)
            pupil_phase = self.extract_instantaneous_phase(
                eye_pupil_interp, self.sampling_rate, freq_band=[0.01, 2.0]
            )
            
            channel_results = {}
            
            # 分析不同的EEG信号类型
            eeg_signals = {
                'offset': eeg_offset_interp,
                'exponent': eeg_exponent_interp,
                'offset_derivative': np.gradient(eeg_offset_interp),  # 添加导数信号
                'exponent_derivative': np.gradient(eeg_exponent_interp)
            }
            
            for signal_type, signal_data in eeg_signals.items():
                logger.info(f"  分析{signal_type}信号...")
                
                # 确保信号长度一致
                min_len = min(len(pupil_phase), len(signal_data))
                phase_aligned = pupil_phase[:min_len]
                signal_aligned = signal_data[:min_len]
                
                if min_len < 50:  # 数据太少
                    logger.warning(f"  {signal_type}数据点太少，跳过")
                    continue
                
                # 计算改进的耦合指标
                plv_value = self.PLV_CFC_improved(phase_aligned, signal_aligned)
                mvl_value = self.MVL_CFC_improved(phase_aligned, signal_aligned)
                mi_value, mi_amp_dist = self.MI_CFC_improved(phase_aligned, signal_aligned)
                
                # 统计显著性检验
                significance = self.statistical_significance_test(
                    phase_aligned, signal_aligned, n_surrogates=50
                )
                
                channel_results[signal_type] = {
                    'PLV': plv_value,
                    'MVL': mvl_value,
                    'MI': mi_value,
                    'MI_amp_distribution': mi_amp_dist,
                    'p_values': significance,
                    'time_points': common_time[:min_len],
                    'pupil_phase': phase_aligned,
                    'eeg_signal': signal_aligned,
                    'data_length': min_len
                }
                
                # 判断显著性
                significance_markers = []
                if significance['p_plv'] < 0.05:
                    significance_markers.append('PLV*')
                if significance['p_mvl'] < 0.05:
                    significance_markers.append('MVL*')
                if significance['p_mi'] < 0.05:
                    significance_markers.append('MI*')
                
                sig_str = ', '.join(significance_markers) if significance_markers else '无显著性'
                
                logger.info(f"  通道{channel}-{signal_type}: PLV={plv_value:.4f}(p={significance['p_plv']:.3f}), "
                           f"MVL={mvl_value:.4f}(p={significance['p_mvl']:.3f}), "
                           f"MI={mi_value:.4f}(p={significance['p_mi']:.3f}) [{sig_str}]")
            
            results[f'channel_{channel}'] = channel_results
        
        self.coupling_results[video_name] = results
        logger.info(f"{video_name}改进耦合计算完成")
        return True
    
    def save_results(self, output_dir='f:/eye_eeg/coupling_results'):
        """
        保存改进的分析结果
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存详细结果
        results_summary = []
        
        for video_name, video_results in self.coupling_results.items():
            for channel_name, channel_results in video_results.items():
                for signal_type, metrics in channel_results.items():
                    if isinstance(metrics, dict) and 'PLV' in metrics:
                        p_values = metrics.get('p_values', {})
                        results_summary.append({
                            'Video': video_name,
                            'Channel': channel_name,
                            'Signal_Type': signal_type,
                            'PLV': metrics['PLV'],
                            'MVL': metrics['MVL'],
                            'MI': metrics['MI'],
                            'PLV_pvalue': p_values.get('p_plv', 1.0),
                            'MVL_pvalue': p_values.get('p_mvl', 1.0),
                            'MI_pvalue': p_values.get('p_mi', 1.0),
                            'PLV_significant': p_values.get('p_plv', 1.0) < 0.05,
                            'MVL_significant': p_values.get('p_mvl', 1.0) < 0.05,
                            'MI_significant': p_values.get('p_mi', 1.0) < 0.05,
                            'Data_Length': metrics.get('data_length', 0)
                        })
        
        # 保存为CSV
        df = pd.DataFrame(results_summary)
        csv_path = os.path.join(output_dir, f'improved_coupling_results_{timestamp}.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        logger.info(f"改进结果已保存到: {csv_path}")
        
        # 生成可视化图表
        self.plot_improved_results(output_dir, timestamp)
        
        return csv_path
    
    def plot_improved_results(self, output_dir, timestamp):
        """
        生成改进的可视化图表
        """
        # 创建综合对比图
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('改进的眼动-脑电耦合分析结果', fontsize=16)
        
        videos = list(self.coupling_results.keys())
        metrics = ['PLV', 'MVL', 'MI']
        
        # 准备数据
        all_data = []
        for video_name, video_results in self.coupling_results.items():
            for channel_name, channel_results in video_results.items():
                for signal_type, metrics_dict in channel_results.items():
                    if isinstance(metrics_dict, dict) and 'PLV' in metrics_dict:
                        p_values = metrics_dict.get('p_values', {})
                        all_data.append({
                            'video': video_name,
                            'channel_signal': f"{channel_name}-{signal_type}",
                            'PLV': metrics_dict['PLV'],
                            'MVL': metrics_dict['MVL'],
                            'MI': metrics_dict['MI'],
                            'PLV_sig': p_values.get('p_plv', 1.0) < 0.05,
                            'MVL_sig': p_values.get('p_mvl', 1.0) < 0.05,
                            'MI_sig': p_values.get('p_mi', 1.0) < 0.05
                        })
        
        df_plot = pd.DataFrame(all_data)
        
        # 绘制每个指标的对比图
        for i, metric in enumerate(metrics):
            # 耦合强度对比
            ax1 = axes[0, i]
            
            # 按视频分组绘制
            x_labels = []
            x_pos = []
            pos_counter = 0
            
            for video in videos:
                video_data = df_plot[df_plot['video'] == video]
                values = video_data[metric].values
                significant = video_data[f'{metric}_sig'].values
                
                # 绘制柱状图
                bars = ax1.bar(range(pos_counter, pos_counter + len(values)), values, 
                              alpha=0.7, label=video)
                
                # 标记显著性
                for j, (bar, is_sig) in enumerate(zip(bars, significant)):
                    if is_sig:
                        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                                '*', ha='center', va='bottom', fontsize=12, color='red')
                
                x_labels.extend(video_data['channel_signal'].values)
                x_pos.extend(range(pos_counter, pos_counter + len(values)))
                pos_counter += len(values) + 1  # 添加间隔
            
            ax1.set_xlabel('通道-信号类型')
            ax1.set_ylabel(f'{metric}值')
            ax1.set_title(f'{metric}耦合强度对比 (*表示p<0.05)')
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(x_labels, rotation=45, ha='right')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 显著性统计
            ax2 = axes[1, i]
            sig_counts = df_plot.groupby('video')[f'{metric}_sig'].sum()
            total_counts = df_plot.groupby('video')[f'{metric}_sig'].count()
            sig_ratios = sig_counts / total_counts
            
            bars = ax2.bar(sig_ratios.index, sig_ratios.values, alpha=0.7)
            ax2.set_ylabel('显著耦合比例')
            ax2.set_title(f'{metric}显著性统计')
            ax2.set_ylim(0, 1)
            
            # 添加数值标签
            for bar, ratio in zip(bars, sig_ratios.values):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{ratio:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f'improved_coupling_comparison_{timestamp}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"改进的可视化图表已保存到: {plot_path}")

def main():
    """主函数"""
    logger.info("开始改进的眼动-脑电耦合分析...")
    
    # 创建改进的分析器
    analyzer = EyeEEGCouplingAnalyzer(sampling_rate=120.0)  # 120Hz采样率
    
    # 定义脑电数据文件路径
    eeg_files = {
        'Video1': r'f:\eye_eeg\logs\channel_parameters_A00053375_Video1_EEG_20251012_111452.log',
        'Video2': r'f:\eye_eeg\logs\channel_parameters_A00053375_Video2_EEG_20251012_113659.log',
        'Video3': r'f:\eye_eeg\logs\channel_parameters_A00053375_Video3_EEG_20251012_114641.log'
    }
    
    # 加载数据并计算耦合
    success_count = 0
    for video_name in ['Video1', 'Video2', 'Video3']:
        logger.info(f"\n处理{video_name}...")
        
        # 生成改进的模拟眼动数据
        if analyzer.load_eye_data_from_existing_files(video_name):
            # 加载脑电数据
            if analyzer.load_eeg_data(eeg_files[video_name], video_name):
                # 计算改进的耦合
                if analyzer.calculate_coupling(video_name):
                    success_count += 1
                    logger.info(f"{video_name}处理成功")
                else:
                    logger.error(f"{video_name}耦合计算失败")
            else:
                logger.error(f"{video_name}脑电数据加载失败")
        else:
            logger.error(f"{video_name}眼动数据生成失败")
    
    if success_count > 0:
        # 保存结果
        output_path = analyzer.save_results()
        logger.info(f"\n改进分析完成！成功处理{success_count}个视频")
        logger.info(f"结果已保存到: {output_path}")
        
        # 打印总结
        print("\n=== 改进的眼动-脑电耦合分析结果总结 ===")
        for video_name, video_results in analyzer.coupling_results.items():
            print(f"\n{video_name}:")
            for channel_name, channel_results in video_results.items():
                for signal_type, metrics in channel_results.items():
                    if isinstance(metrics, dict) and 'PLV' in metrics:
                        p_vals = metrics.get('p_values', {})
                        sig_markers = []
                        if p_vals.get('p_plv', 1.0) < 0.05:
                            sig_markers.append('PLV*')
                        if p_vals.get('p_mvl', 1.0) < 0.05:
                            sig_markers.append('MVL*')
                        if p_vals.get('p_mi', 1.0) < 0.05:
                            sig_markers.append('MI*')
                        
                        sig_str = f" [{', '.join(sig_markers)}]" if sig_markers else " [无显著性]"
                        
                        print(f"  {channel_name}-{signal_type}:")
                        print(f"    PLV: {metrics['PLV']:.4f} (p={p_vals.get('p_plv', 1.0):.3f})")
                        print(f"    MVL: {metrics['MVL']:.4f} (p={p_vals.get('p_mvl', 1.0):.3f})")
                        print(f"    MI:  {metrics['MI']:.4f} (p={p_vals.get('p_mi', 1.0):.3f})")
                        print(f"    显著性: {sig_str}")
        
        print(f"\n详细结果已保存到: {output_path}")
        print("* 表示 p < 0.05 (统计显著)")
        
    else:
        logger.error("所有视频处理失败")

if __name__ == "__main__":
    main()