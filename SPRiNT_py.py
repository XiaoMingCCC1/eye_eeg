# Python SPRiNT
# Author: Luc Wilson (2023)

from math import floor
from statistics import median
from itertools import compress
import numpy as np
import os
import csv
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import fooof
from fooof.data import FOOOFResults
from fooof.sim.gen import gen_periodic
from fooof.sim.gen import gen_aperiodic
from fooof.objs.utils import combine_fooofs
from copy import deepcopy
import scipy.signal

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def SPRiNT_stft_py(F, opt):
    ''' SPRiNT_stft_py: Compute a locally averaged short-time Fourier transform
    (for use in SPRiNT)

    Inputs
    F - Time series (nxm numpy array),
    where n is number of channels, m is number of samples
    opt - Model settings/hyperparameters (dict)

    Author: Luc Wilson (2023)
    '''
    # Get sampling frequency
    n_chan = F.shape
    if not len(n_chan) > 1:
        n_sample = n_chan[0]
    else:
        n_sample = n_chan[1]

    sfreq = opt['sfreq']
    avgWin = opt['WinAverage']

    # Initialize returned values
    ind_good = 1  # index for kept data

    # WINDOWING
    Lwin = round(opt['WinLength'] * opt['sfreq'])  # n data points per window
    Loverlap = round(Lwin * opt["WinOverlap"] / 100)  # n data points in overlap

    # If window is too small
    if Lwin < 50:
        print("Time window too small, please increase and run process again.")
        return
    # If window is bigger than the data
    elif Lwin > n_sample:
        Lwin = len(F[0])
        Lwin = Lwin - (Lwin % 2)  # Make sure the number of samples is even
        Loverlap = 0
        Nwin = 1
        print("Time window too large, using entire recording for spectrum.")
    # Else: there is at least one full time window
    else:
        Lwin = Lwin - (Lwin % 2)  # Make sure the number of samples is even
        Nwin = (n_sample - Loverlap) // (Lwin - Loverlap)

    # Positive frequency bins spanned by FFT
    FreqVector = sfreq / 2 * np.linspace(0, 1, round(Lwin / 2 + 1))
    # Determine hamming window shape/power
    Win = np.hanning(Lwin)
    WinNoisePowerGain = sum(Win**2)
    # Initialize STFT, time matrices
    ts = np.full((Nwin-(avgWin-1)),np.nan)
    if len(n_chan) > 1:
        TF = np.full((len(F), Nwin - (avgWin - 1), len(FreqVector)), np.nan)
        TFtmp = np.full((len(F), avgWin, len(FreqVector)), np.nan)
    else:
        TF = np.full((1, Nwin - (avgWin - 1), len(FreqVector)), np.nan)
        TFtmp = np.full((1, avgWin, len(FreqVector)), np.nan)

    # Calculate FFT for each window
    TFfull = np.zeros((len(F), Nwin, len(FreqVector)))
    for iWin in range(Nwin):
        # Build indices
        iTimes = list(range((iWin)*(Lwin-Loverlap),Lwin+(iWin)*(Lwin-Loverlap)))
        center_time = floor(median(np.add(np.array(iTimes),1))-\
            (avgWin-1)/2*(Lwin-Loverlap))/200
        if len(n_chan) > 1:
            Fwin = F[:, iTimes]
            Fwin = Fwin- Fwin.mean(axis=1, keepdims=True)
        else:
            Fwin = F[iTimes]
            Fwin = Fwin - Fwin.mean()
        # Apply a Hann window to signal
        Fw = np.multiply(Fwin,Win)
        # Compute FFT
        Ffft = np.fft.fft(Fw, Lwin)
        if len(n_chan) > 1:
            TFwin = Ffft[:, :Lwin//2+1] * \
                np.sqrt(2 / (sfreq * WinNoisePowerGain))
            TFwin[:, [0, -1]] = TFwin[:, [0, -1]] / np.sqrt(2)
            TFwin = np.abs(TFwin)**2
            TFfull[:,iWin,:] = TFwin
            TFtmp[:, iWin % avgWin, :] = TFwin
        else:
            TFwin = Ffft[:Lwin//2+1] * np.sqrt(2 / (sfreq * WinNoisePowerGain))
            TFwin[[0, -1]] = TFwin[[0, -1]] / np.sqrt(2)
            TFwin = np.abs(TFwin)**2
            TFfull[:,iWin,:] = TFwin
            TFtmp[:, iWin % avgWin, :] = TFwin
        if np.isnan(TFtmp[-1, -1, -1]):
            pass
        else:
            # Save STFTs for window
            TF[:, iWin - (avgWin - 1), :] = np.mean(TFtmp, axis=1)
            ts[iWin - (avgWin - 1)] = center_time

    output = {
        "TF": TF,
        "freqs": FreqVector,
        "ts": ts,
        "options": opt
    }

    return output

def SPRiNT_remove_outliers(fooof_chan, ts, opt):
    ''' SPRiNT_remove_outliers: helper function to remove outlier peaks
    according to user-defined specifications

    Input
    fooof_chan - fooof model for a given channel (FOOOFObject)
    ts - sampled times (numpy array)
    opt - Model settings/hyperparameters (dict)

    Author: Luc Wilson (2023)
    '''
    n_peaks_changed = True
    peaks = fooof_chan.get_params('gaussian_params')
    n_peaks = len(peaks)
    npeak_bytime = fooof_chan.n_peaks_

    peaks_tmp = peaks
    n_peaks_tmp = n_peaks
    npeak_bytime_tmp = deepcopy(npeak_bytime)

    while n_peaks_changed:
        n_peaks_changed = False
        remove = [False for _ in range(n_peaks_tmp)]
        n_peaks_tmp2 = n_peaks_tmp

        for p in range(n_peaks_tmp):
            n_close = 0
            close_t = [(np.abs(peaks_tmp[p,3] - peaks_tmp[r,3])\
                <= opt['maxTime']) for r in range(n_peaks_tmp2)]
            close_f = [(np.abs(peaks_tmp[p,0] - peaks_tmp[r,0])\
                <= opt['maxFreq']) for r in range(n_peaks_tmp2)]

            for r in range(n_peaks_tmp2):
                if close_t[r] and close_f[r]:
                    n_close +=1

            if n_close < (opt['minNear']+1):
                remove[p] = True
                npeak_bytime_tmp[int(peaks_tmp[p,3])] -= 1
                n_peaks_tmp -= 1
                n_peaks_changed = True

        peaks_tmp = peaks_tmp[[not bln for bln in remove]]

    fg = list([])
    for t in range(len(ts)):
        tmp = fooof_chan.get_fooof(t)
        if npeak_bytime_tmp[t] != npeak_bytime[t]:
            if npeak_bytime_tmp[t] == 0:
                gaus_pars = []
                pk_fit = gen_periodic(tmp.freqs, [])
                ap_pars = tmp._simple_ap_fit(tmp.freqs,tmp.power_spectrum)
                ap_fit = gen_aperiodic(tmp.freqs, ap_pars)
            else:
                gaus_pars = peaks_tmp[np.where(peaks_tmp[:,3] == t)[0],:3]
                pk_fit = gen_periodic(tmp.freqs, np.ndarray.flatten(gaus_pars))
                ap_pars = tmp._simple_ap_fit(tmp.freqs,tmp.power_spectrum-pk_fit)
                ap_fit = gen_aperiodic(tmp.freqs, ap_pars)
            error = np.abs(tmp.power_spectrum - ap_fit - pk_fit).mean()
            r_squared = np.corrcoef(tmp.power_spectrum, ap_fit + pk_fit)[0][1]**2
            tmp.add_results(FOOOFResults(ap_pars,\
                tmp._create_peak_params(gaus_pars), r_squared, error, gaus_pars))
        fg.append(tmp)
    fg = combine_fooofs(fg)
    return fg

# 读取脑电数据
with open('F:/眼动/data/A00053440/EEG/raw/csv_format/A00053440006.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    n = []
    for row in csv_reader:
        n.append([float(x) for x in row])

F = np.array(n)

# 设置超参数
opt = {
    "sfreq": 500,
    "WinLength": 2,
    "WinOverlap": 50,
    "WinAverage": 2,
    "rmoutliers": 1,
    "maxTime": 6,
    "maxFreq": 2.5,
    "minNear": 3,
}

# 生成numpy数组
if len(n) == 1:
    n = n[0]
    n = [float(x) for x in n]
    F = np.array(n)
else:
    n = [[float(x) for x in n[i]] for i in range(len(n))]
    F = np.array(n)

# 计算STFT
output = SPRiNT_stft_py(F, opt)

# 检查并处理功率谱
TF = output['TF']
TF = np.nan_to_num(TF, nan=0.0, posinf=0.0, neginf=0.0)
TF[TF < 0] = 0

# 检查是否为log10形式
if np.max(TF) < 10 and np.min(TF) > -10:
    print('检测到功率谱可能为log10形式，自动还原为线性功率谱...')
    TF = 10 ** TF

# 剔除无效数据
for ch in range(TF.shape[0]):
    for t in range(TF.shape[1]):
        if np.std(TF[ch, t, :]) == 0:
            TF[ch, t, :] = np.nan

mask_ch = ~np.any(np.isnan(TF), axis=(1,2))
mask_time = ~np.any(np.isnan(TF), axis=(0,2))
TF = TF[mask_ch][:, mask_time]
output['TF'] = TF
if 'ts' in output:
    output['ts'] = output['ts'][mask_time]

print(f'剔除后TF形状: {TF.shape}')

# 检查数据有效性
if TF.size == 0 or TF.shape[1] == 0:
    print('错误：所有功率谱均为无效，请检查原始数据或参数设置。')
    import sys
    sys.exit(1)

# 替换0值
TF[TF == 0] = 1e-12
output['TF'] = TF

# FOOOF拟合
fg = fooof.FOOOFGroup(peak_width_limits=[2, 6], min_peak_height=0.5, max_n_peaks=3)
fgs = fooof.fit_fooof_3d(fg, output['freqs'], output['TF'], freq_range=[1, 40])

# 修改这里：处理所有通道，而不仅仅是前两个通道
print(f"开始处理所有 {len(fgs)} 个通道...")
fgs = [SPRiNT_remove_outliers(fgs[i], output['ts'], opt) for i in range(len(fgs))]

# RSA分析
print("\n开始表征相似性分析...")

try:
    # 获取数据维度
    n_channels = len(fgs)
    n_timepoints = len(fgs[0]) if n_channels > 0 else 0
    max_freqs = len(output['freqs'])
    print(f"最大频率点数: {max_freqs}")
    print(f"通道数: {n_channels}")
    print(f"时间点数: {n_timepoints}")
    
    # 提取特征
    power_features = np.array([output['TF'][i] for i in range(n_channels)])
    peak_features = []
    for ch in range(n_channels):
        for t in range(n_timepoints):
            try:
                peaks = fgs[ch].get_fooof(t).peak_params_
                if peaks is not None and len(peaks) > 0:
                    peak_features.extend(peaks)
            except Exception as e:
                print(f"警告: 处理通道{ch}, 时间点{t}时出错: {str(e)}")
                continue
    
    # 打印原始特征数值统计信息
    print("\n原始特征数值统计:")
    print(f"功率谱特征原始值范围: [{np.nanmin(power_features):.6f}, {np.nanmax(power_features):.6f}]")
    print(f"功率谱特征均值: {np.nanmean(power_features):.6f}")
    print(f"功率谱特征标准差: {np.nanstd(power_features):.6f}")
    print(f"功率谱特征中位数: {np.nanmedian(power_features):.6f}")
    
    # 打印前10个功率谱特征值示例
    print("\n功率谱特征值示例(第一个通道，第一个时间点，前10个频率点):")
    if power_features.size > 0:
        sample_values = power_features[0, 0, :10] if power_features.shape[2] >= 10 else power_features[0, 0, :]
        for i, val in enumerate(sample_values):
            print(f"频率点 {i}: {val:.6f}")
    
    # 打印峰值特征示例
    if len(peak_features) > 0:
        print("\n峰值特征示例(前5个):")
        for i, peak in enumerate(peak_features[:5]):
            print(f"峰值 {i}: 中心频率={peak[0]:.2f}Hz, 幅度={peak[1]:.6f}, 带宽={peak[2]:.2f}")
    else:
        print("\n未检测到峰值特征")
    
    # 预处理特征
    power_features = np.nan_to_num(power_features, nan=0.0, posinf=0.0, neginf=0.0)
    power_features = np.log10(power_features + 1e-10)
    power_features = (power_features - np.nanmean(power_features)) / (np.nanstd(power_features) + 1e-10)
    
    # 打印预处理后的特征数值统计信息
    print("\n预处理后特征数值统计:")
    print(f"功率谱特征预处理后值范围: [{np.nanmin(power_features):.6f}, {np.nanmax(power_features):.6f}]")
    print(f"功率谱特征预处理后均值: {np.nanmean(power_features):.6f}")
    print(f"功率谱特征预处理后标准差: {np.nanstd(power_features):.6f}")
    print(f"功率谱特征预处理后中位数: {np.nanmedian(power_features):.6f}")
    
    # 打印预处理后的前10个功率谱特征值示例
    print("\n预处理后功率谱特征值示例(第一个通道，第一个时间点，前10个频率点):")
    if power_features.size > 0:
        sample_values = power_features[0, 0, :10] if power_features.shape[2] >= 10 else power_features[0, 0, :]
        for i, val in enumerate(sample_values):
            print(f"频率点 {i}: {val:.6f}")
    
    # 计算时间点相似性（分块处理）
    block_size = 50
    n_timepoints = power_features.shape[1]
    n_blocks = int(np.ceil(n_timepoints / block_size))
    time_similarity = np.zeros((n_timepoints, n_timepoints))
    
    for i in range(n_blocks):
        start_i = i * block_size
        end_i = min((i + 1) * block_size, n_timepoints)
        features_i = power_features[:, start_i:end_i].reshape(-1, end_i - start_i)
        
        for j in range(i, n_blocks):
            start_j = j * block_size
            end_j = min((j + 1) * block_size, n_timepoints)
            features_j = power_features[:, start_j:end_j].reshape(-1, end_j - start_j)
            
            # 计算相似度（使用余弦相似度）
            norm_i = np.linalg.norm(features_i, axis=0)
            norm_j = np.linalg.norm(features_j, axis=0)
            cosine_sim = np.dot(features_i.T, features_j) / (norm_i[:, np.newaxis] @ norm_j[np.newaxis, :] + 1e-10)
            
            time_similarity[start_i:end_i, start_j:end_j] = cosine_sim
            if i != j:  # 填充对称部分
                time_similarity[start_j:end_j, start_i:end_i] = cosine_sim.T
    
    # 计算通道相似性
    features_reshaped = power_features.reshape(n_channels, -1)
    channel_similarity = np.corrcoef(features_reshaped)
    
    # 打印相似性矩阵的数值统计
    print("\n相似性矩阵数值统计:")
    print(f"时间点相似性矩阵值范围: [{np.min(time_similarity):.6f}, {np.max(time_similarity):.6f}]")
    print(f"时间点相似性矩阵均值: {np.mean(time_similarity):.6f}")
    print(f"时间点相似性矩阵标准差: {np.std(time_similarity):.6f}")
    print(f"时间点相似性矩阵中位数: {np.median(time_similarity):.6f}")
    print(f"时间点相似性矩阵第5百分位数: {np.percentile(time_similarity, 5):.6f}")
    print(f"时间点相似性矩阵第95百分位数: {np.percentile(time_similarity, 95):.6f}")
    
    print(f"\n通道相似性矩阵值范围: [{np.min(channel_similarity):.6f}, {np.max(channel_similarity):.6f}]")
    print(f"通道相似性矩阵均值: {np.mean(channel_similarity):.6f}")
    print(f"通道相似性矩阵标准差: {np.std(channel_similarity):.6f}")
    print(f"通道相似性矩阵中位数: {np.median(channel_similarity):.6f}")
    
    # 打印分析结果
    print(f"\n有效时间点数: {n_timepoints} / {n_timepoints}")
    
    print(f"\n特征提取完成:")
    print(f"- 功率谱特征形状: {power_features.shape}")
    print(f"- 峰值特征数量: {len(peak_features)}")
    print(f"- 功率谱非零元素比例: {np.mean(power_features != 0):.3f}")
    
    print(f"\n时间点间相似性矩阵形状: {time_similarity.shape}")
    print(f"相似性范围: [{np.min(time_similarity):.3f}, {np.max(time_similarity):.3f}]")
    print(f"相似性均值: {np.mean(time_similarity):.3f}")
    print(f"相似性标准差: {np.std(time_similarity):.3f}")
    print(f"非零相似性比例: {np.mean(time_similarity != 0):.3f}")
    
    print(f"\n通道间相似性矩阵形状: {channel_similarity.shape}")
    print(f"相似性范围: [{np.min(channel_similarity):.3f}, {np.max(channel_similarity):.3f}]")
    print(f"相似性均值: {np.mean(channel_similarity):.3f}")
    print(f"相似性标准差: {np.std(channel_similarity):.3f}")
    print(f"非零相似性比例: {np.mean(channel_similarity != 0):.3f}")
    
    # 将特征值保存到txt文件中
    try:
        # 保存功率谱特征
        with open('power_features.txt', 'w', encoding='utf-8') as f:
            f.write(f"功率谱特征形状: {power_features.shape}\n")
            f.write(f"通道数: {power_features.shape[0]}\n")
            f.write(f"时间点数: {power_features.shape[1]}\n")
            f.write(f"频率点数: {power_features.shape[2]}\n\n")
            
            f.write("频率值(Hz):\n")
            for i, freq in enumerate(output['freqs']):
                f.write(f"{i}\t{freq:.2f}\n")
            f.write("\n")
            
            f.write("时间点值(s):\n")
            for i, t in enumerate(output['ts']):
                f.write(f"{i}\t{t:.4f}\n")
            f.write("\n")
            
            f.write("功率谱特征值 (通道, 时间点, 频率):\n")
            for ch in range(power_features.shape[0]):
                f.write(f"\n通道 {ch}:\n")
                for t in range(power_features.shape[1]):
                    f.write(f"\n时间点 {t} ({output['ts'][t]:.4f}s):\n")
                    for freq_idx in range(power_features.shape[2]):
                        f.write(f"{freq_idx}\t{output['freqs'][freq_idx]:.2f}Hz\t{power_features[ch, t, freq_idx]:.6f}\n")
        
        print("\n功率谱特征已保存到 'power_features.txt'")
        
        # 保存峰值特征
        if len(peak_features) > 0:
            with open('peak_features.txt', 'w', encoding='utf-8') as f:
                f.write(f"峰值特征数量: {len(peak_features)}\n\n")
                f.write("峰值特征 (中心频率, 幅度, 带宽):\n")
                for i, peak in enumerate(peak_features):
                    f.write(f"{i}\t{peak[0]:.2f}Hz\t{peak[1]:.6f}\t{peak[2]:.2f}\n")
            
            print("峰值特征已保存到 'peak_features.txt'")
        
        # 保存时间点相似性矩阵
        with open('time_similarity.txt', 'w', encoding='utf-8') as f:
            f.write(f"时间点相似性矩阵形状: {time_similarity.shape}\n\n")
            f.write("时间点相似性矩阵:\n")
            for i in range(time_similarity.shape[0]):
                for j in range(time_similarity.shape[1]):
                    f.write(f"{time_similarity[i, j]:.6f}\t")
                f.write("\n")
        
        print("时间点相似性矩阵已保存到 'time_similarity.txt'")
        
        # 保存通道相似性矩阵
        with open('channel_similarity.txt', 'w', encoding='utf-8') as f:
            f.write(f"通道相似性矩阵形状: {channel_similarity.shape}\n\n")
            f.write("通道相似性矩阵:\n")
            for i in range(channel_similarity.shape[0]):
                for j in range(channel_similarity.shape[1]):
                    f.write(f"{channel_similarity[i, j]:.6f}\t")
                f.write("\n")
        
        print("通道相似性矩阵已保存到 'channel_similarity.txt'")
        
    except Exception as e:
        print(f"保存特征值到txt文件失败: {str(e)}")
    
    # 可视化结果
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # 设置seaborn样式
        sns.set_style("white")
        sns.set_context("talk")
        
        # 创建图形
        fig = plt.figure(figsize=(20, 15))
        
        # 1. 时间点相似性矩阵（采样显示）
        ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2)
        sample_rate = max(1, time_similarity.shape[0] // 1000)  # 采样率
        sampled_matrix = time_similarity[::sample_rate, ::sample_rate]
        
        # 调整热图的值域范围
        vmin = np.percentile(sampled_matrix[~np.isnan(sampled_matrix)], 5)
        vmax = np.percentile(sampled_matrix[~np.isnan(sampled_matrix)], 95)
        sns.heatmap(sampled_matrix, cmap='RdBu_r', center=0, vmin=vmin, vmax=vmax, ax=ax1)
        ax1.set_title('时间点相似性矩阵 (采样显示)')
        
        # 2. 通道相似性矩阵
        ax2 = plt.subplot2grid((3, 3), (0, 2))
        vmin = np.percentile(channel_similarity[~np.isnan(channel_similarity)], 5)
        vmax = np.percentile(channel_similarity[~np.isnan(channel_similarity)], 95)
        sns.heatmap(channel_similarity, cmap='RdBu_r', center=0, vmin=vmin, vmax=vmax, ax=ax2)
        ax2.set_title('通道相似性矩阵')
        
        # 3. 平均功率谱热图（采样显示）
        ax3 = plt.subplot2grid((3, 3), (1, 0), colspan=2)
        mean_power = np.mean(power_features, axis=0)
        sample_rate_time = max(1, mean_power.shape[0] // 500)
        sample_rate_freq = max(1, mean_power.shape[1] // 100)
        sampled_power = mean_power[::sample_rate_time, ::sample_rate_freq]
        
        vmin = np.percentile(sampled_power[~np.isnan(sampled_power)], 5)
        vmax = np.percentile(sampled_power[~np.isnan(sampled_power)], 95)
        sns.heatmap(sampled_power, cmap='viridis', vmin=vmin, vmax=vmax, ax=ax3)
        ax3.set_title('平均功率谱 (采样显示)')
        
        # 4. 相似性分布直方图
        ax4 = plt.subplot2grid((3, 3), (1, 2))
        
        # 采样数据以减少内存使用
        time_sim_sample = time_similarity.flatten()[::100]
        channel_sim_sample = channel_similarity.flatten()
        
        # 绘制直方图
        sns.histplot(data=time_sim_sample, label='时间点相似性', alpha=0.5, ax=ax4)
        sns.histplot(data=channel_sim_sample, label='通道相似性', alpha=0.5, ax=ax4)
        ax4.set_title('相似性分布')
        ax4.legend()
        
        # 5. 统计信息文本框
        ax5 = plt.subplot2grid((3, 3), (2, 0), colspan=3)
        ax5.axis('off')
        
        # 计算统计信息
        stats_text = [
            "数据统计信息:",
            f"时间点相似性矩阵: 形状={time_similarity.shape}, 范围=[{np.min(time_similarity):.3f}, {np.max(time_similarity):.3f}]",
            f"时间点相似性: 均值={np.mean(time_similarity):.3f}, 标准差={np.std(time_similarity):.3f}",
            f"通道相似性矩阵: 形状={channel_similarity.shape}, 范围=[{np.min(channel_similarity):.3f}, {np.max(channel_similarity):.3f}]",
            f"通道相似性: 均值={np.mean(channel_similarity):.3f}, 标准差={np.std(channel_similarity):.3f}",
            f"功率谱: 形状={power_features.shape}, 非零比例={np.mean(power_features != 0):.3f}",
            f"采样率: 时间点={sample_rate}, 频率={sample_rate_freq}"
        ]
        
        ax5.text(0.05, 0.5, '\n'.join(stats_text), fontsize=10, va='center')
        
        # 调整布局并保存
        plt.tight_layout()
        plt.savefig('RSA_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("可视化结果已保存到 RSA_analysis.png")
        
    except Exception as e:
        print(f"可视化出错: {str(e)}")
        
        # 如果发生错误，尝试只保存统计信息
        try:
            with open('RSA_analysis_stats.txt', 'w', encoding='utf-8') as f:
                f.write(f"时间点相似性矩阵: 形状={time_similarity.shape}\n")
                f.write(f"时间点相似性范围: [{np.min(time_similarity):.3f}, {np.max(time_similarity):.3f}]\n")
                f.write(f"时间点相似性均值: {np.mean(time_similarity):.3f}\n")
                f.write(f"时间点相似性标准差: {np.std(time_similarity):.3f}\n\n")
                
                f.write(f"通道相似性矩阵: 形状={channel_similarity.shape}\n")
                f.write(f"通道相似性范围: [{np.min(channel_similarity):.3f}, {np.max(channel_similarity):.3f}]\n")
                f.write(f"通道相似性均值: {np.mean(channel_similarity):.3f}\n")
                f.write(f"通道相似性标准差: {np.std(channel_similarity):.3f}\n\n")
                
                f.write(f"功率谱形状: {power_features.shape}\n")
                f.write(f"功率谱非零比例: {np.mean(power_features != 0):.3f}\n")
            
            print("由于可视化错误，统计信息已保存到 RSA_analysis_stats.txt")
        except Exception as e2:
            print(f"保存统计信息也失败了: {str(e2)}")
    
    # 保存结果
    try:
        results = {
            'power_features': power_features,
            'peak_features': peak_features,
            'time_similarity': time_similarity,
            'channel_similarity': channel_similarity,
            'analysis_params': {
                'n_channels': n_channels,
                'n_timepoints': n_timepoints,
                'max_freqs': max_freqs
            }
        }
        np.save('rsa_results.npy', results)
        print("\nRSA分析结果已保存到 'rsa_results.npy'")
    except Exception as e:
        print(f"保存结果失败: {e}")

except Exception as e:
    print(f"RSA分析失败: {e}")
    raise

print("\nRSA分析完成！主要改进:")
print("1. 数据预处理（对数变换、标准化）")
print("2. 分块处理减少内存使用")
print("3. 更丰富的统计信息和可视化")
print("4. 添加特征数值的详细打印")
print("5. 将特征值保存到txt文件中")
print("6. 处理所有通道而不仅仅是前两个通道")
