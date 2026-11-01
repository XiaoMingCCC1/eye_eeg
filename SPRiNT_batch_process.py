import numpy as np
import os
import csv
import logging
from datetime import datetime
import fooof
from fooof.data import FOOOFResults
from fooof.sim.gen import gen_periodic, gen_aperiodic
from fooof.objs.utils import combine_fooofs
from copy import deepcopy
from math import floor
from statistics import median
import matplotlib.pyplot as plt
import pandas as pd
import glob

def setup_logging(subject_id, data_type, task_num=None):
    """为每个被试设置日志记录器"""
    # 过滤numpy警告
    import warnings
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    
    # 根据数据类型确定保存路径
    if data_type == "rest":
        log_dir = os.path.join('H:', 'EEG_EyeTracking', '静息态', subject_id)
    else:
        log_dir = os.path.join('H:', 'EEG_EyeTracking', '任务态', subject_id, f'任务{task_num}')
    
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'SPRiNT_analysis_{timestamp}.log')
    
    # 创建唯一logger
    logger_name = f'{subject_id}_{data_type}_{task_num or "rest"}_{timestamp}'
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.handlers = []
    
    # 文件handler
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)
    
    # 控制台handler
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(ch)
    
    logger.propagate = False
    return logger

def find_file_recursive(start_path, pattern):
    """递归查找文件"""
    matches = []
    for root, dirnames, filenames in os.walk(start_path):
        for filename in filenames:
            if pattern in filename:
                matches.append(os.path.join(root, filename))
    return matches

def find_dir_recursive(start_path, dir_name):
    """递归查找文件夹"""
    matches = []
    for root, dirnames, filenames in os.walk(start_path):
        if dir_name in dirnames:
            matches.append(os.path.join(root, dir_name))
    return matches

def check_output_exists(subject_id, data_type, task_num=None):
    """检查输出文件是否已存在"""
    if data_type == "rest":
        output_dir = os.path.join(r'H:\EEG_EyeTracking\静息态', subject_id)
    else:  # task
        output_dir = os.path.join(r'H:\EEG_EyeTracking\任务态', subject_id, f'任务{task_num}')
    
    # 检查目录是否存在且包含log文件
    if os.path.exists(output_dir):
        log_files = [f for f in os.listdir(output_dir) if f.endswith('.log')]
        if log_files:
            return True, output_dir
    return False, output_dir

def process_subject_data(subject_path, subject_id):
    """处理单个被试的所有数据"""
    logger = logging.getLogger(__name__)
    logger.info(f"开始处理被试 {subject_id} 的数据")
    
    try:
        # 检查并处理静息态数据
        rest_exists, rest_output_dir = check_output_exists(subject_id, "rest")
        if rest_exists:
            logger.info(f"静息态数据已存在，跳过处理: {rest_output_dir}")
        else:
            rest_files = find_file_recursive(subject_path, f'{subject_id}001.csv')
            if rest_files:
                rest_path = rest_files[0]  # 使用找到的第一个匹配文件
                logger.info(f"处理静息态数据: {rest_path}")
                process_single_file(rest_path, subject_id, "rest")
            else:
                logger.warning(f"未找到静息态数据文件: {subject_id}001.csv")
        
        # 检查并处理任务态数据
        pipei_dirs = find_dir_recursive(subject_path, 'pipei')
        if pipei_dirs:
            pipei_path = pipei_dirs[0]  # 使用找到的第一个匹配文件夹
            logger.info(f"找到pipei文件夹: {pipei_path}")
            
            # 处理每个视频任务
            
            # 查找所有video文件夹
            video_dirs = []
            for root, dirnames, filenames in os.walk(pipei_path):
                for dirname in dirnames:
                    if dirname.startswith('video'):
                        video_dirs.append(os.path.join(root, dirname))
            
            if not video_dirs:
                logger.warning(f"在pipei文件夹中未找到任何video文件夹")
                return
            
            # 处理每个找到的video文件夹
            for video_idx, video_path in enumerate(video_dirs, 1):
                video_name = os.path.basename(video_path)
                logger.info(f"找到{video_name}文件夹: {video_path}")
                
                # 检查任务态数据是否已存在
                task_exists, task_output_dir = check_output_exists(subject_id, "task", video_idx)
                if task_exists:
                    logger.info(f"任务{video_idx}({video_name})数据已存在，跳过处理: {task_output_dir}")
                    continue
                
                # 查找video文件夹下的所有csv文件
                csv_files = []
                for file in os.listdir(video_path):
                    if file.endswith('.csv'):
                        csv_files.append(os.path.join(video_path, file))
                
                # 过滤掉events和channels结尾的文件，保留EEG数据文件
                eeg_files = []
                for csv_file in csv_files:
                    filename = os.path.basename(csv_file).lower()
                    # 排除events和channels结尾的文件
                    if not (filename.endswith('_events.csv') or 
                           filename.endswith('_channels.csv') or
                           filename.endswith('_interpolated_channels.csv') or
                           'events' in filename or 
                           'channels' in filename):
                        eeg_files.append(csv_file)
                
                logger.info(f"在{video_name}文件夹中找到{len(csv_files)}个CSV文件，过滤后剩余{len(eeg_files)}个EEG数据文件")
                
                if eeg_files:
                    for csv_file in eeg_files:
                        filename = os.path.basename(csv_file)
                        logger.info(f"处理任务{video_idx}({video_name})数据: {filename}")
                        process_single_file(csv_file, subject_id, "task", video_idx)
                else:
                    logger.warning(f"在{video_name}文件夹中未找到可分析的EEG数据文件")
        else:
            logger.info(f"未找到pipei文件夹，跳过任务态数据处理")
    except Exception as e:
        logger.error(f"处理被试 {subject_id} 数据时出错: {str(e)}", exc_info=True)

def process_single_file(file_path, subject_id, data_type, task_num=None):
    """处理单个CSV文件"""
    # 设置该文件的日志记录器
    logger = setup_logging(subject_id, data_type, task_num)
    logger.info(f"开始处理文件: {file_path}")
    
    try:
        # 读取CSV数据
        with open(file_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            data = []
            for row in csv_reader:
                data.append(row)
        
        logger.info(f"原始数据行数: {len(data)}")
        
        # 检查是否为空文件
        if len(data) == 0:
            logger.error("文件为空")
            return
            
        # 跳过标题行（如果存在）
        if data[0][0].lower() in ['channel', 'time', 'sample', 'type']:
            data = data[1:]
            logger.info("跳过标题行")
            
        # 再次检查数据
        if len(data) == 0:
            logger.error("跳过标题行后数据为空")
            return
            
        # 检查第一行数据是否为数值
        try:
            test_row = [float(x) for x in data[0]]
        except ValueError as e:
            logger.error(f"数据包含非数值内容: {e}")
            logger.error(f"第一行数据: {data[0][:5]}...")
            return
        
        # 转换数据为numpy数组
        try:
            if len(data) == 1:
                # 单行数据（可能是interpolated_channels文件）
                data = data[0]
                data = [float(x) for x in data]
                F = np.array(data)
                logger.warning("检测到单行数据，可能是interpolated_channels文件，跳过处理")
                return
            else:
                # 多行数据（正常的EEG数据）
                data = [[float(x) for x in row] for row in data]
                F = np.array(data)
        except ValueError as e:
            logger.error(f"数据转换失败: {e}")
            return
        
        logger.info(f"数据加载成功，形状: {F.shape}")
        
        # 设置SPRiNT参数
        opt = {
            "sfreq": 500,
            "WinLength": 5,
            "WinOverlap": 50,
            "WinAverage": 5,
            "rmoutliers": 1,
            "maxTime": 6,
            "maxFreq": 2.5,
            "minNear": 3,
        }
        
        # 运行STFT分析
        output = SPRiNT_stft_py(F, opt)
        logger.info("STFT分析完成")
        
        # 运行FOOOF分析
        fg = fooof.FOOOFGroup(
            peak_width_limits=[1, 8],
            min_peak_height=1.0,
            max_n_peaks=2
        )
        
        # 确保数据为正值
        TF_positive = np.maximum(output['TF'], np.finfo(float).eps)
        freq_mask = (output['freqs'] >= 1) & (output['freqs'] <= 40)
        freqs = output['freqs'][freq_mask]
        TF_positive = TF_positive[:, :, freq_mask]
        
        # 执行FOOOF分析
        fgs = fooof.fit_fooof_3d(fg, freqs, TF_positive, freq_range=[1, 40])
        logger.info("FOOOF分析完成")
        
        # 记录参数
        for chan_idx in range(len(fgs)):
            logger.info(f"\n通道 {chan_idx} 的参数:")
            log_channel_parameters(fgs[chan_idx], logger)
        
        # 移除异常值
        if opt['rmoutliers']:
            logger.info("开始移除异常值")
            fgs = [SPRiNT_remove_outliers(fgs[i], output['ts'], opt) for i in range(len(fgs))]
            logger.info("异常值移除完成")
            
            # 记录移除异常值后的参数
            logger.info("\n异常值移除后的参数:")
            for chan_idx in range(len(fgs)):
                logger.info(f"\n通道 {chan_idx} 的最终参数:")
                log_channel_parameters(fgs[chan_idx], logger)
        
        # 保存结果
        save_results(output, fgs, subject_id, data_type, task_num)
        
    except Exception as e:
        logger.error(f"处理文件时出错: {str(e)}", exc_info=True)

def log_channel_parameters(fg, logger):
    """记录通道参数"""
    # 获取非周期性分量参数
    aperiodic = fg.get_params('aperiodic_params')
    logger.info("非周期性分量:")
    if aperiodic.ndim > 1:
        for t, ap in enumerate(aperiodic):
            logger.info(f"  时间点 {t}:")
            logger.info(f"    - 偏移(offset): {ap[0]:.3f}")
            logger.info(f"    - 指数(exponent): {ap[1]:.3f}")
    else:
        logger.info(f"  - 偏移(offset): {aperiodic[0]:.3f}")
        logger.info(f"  - 指数(exponent): {aperiodic[1]:.3f}")
    
    # 获取周期性分量参数
    peaks = fg.get_params('peak_params')
    if len(peaks) > 0:
        logger.info("周期性分量:")
        if peaks.ndim > 2:
            for t, time_peaks in enumerate(peaks):
                if len(time_peaks) > 0:
                    logger.info(f"  时间点 {t}:")
                    for i, peak in enumerate(time_peaks):
                        log_peak_parameters(peak, i, logger, indent="    ")
        else:
            for i, peak in enumerate(peaks):
                log_peak_parameters(peak, i, logger)
    else:
        logger.info("未检测到显著的周期性分量")

def log_peak_parameters(peak, index, logger, indent="  "):
    """记录峰值参数"""
    logger.info(f"{indent}峰值 {index+1}:")
    logger.info(f"{indent}  - 中心频率(CF): {peak[0]:.2f} Hz")
    logger.info(f"{indent}  - 幅度(PW): {peak[1]:.3f}")
    logger.info(f"{indent}  - 标准差(BW): {peak[2]:.3f}")

def save_results(output, fgs, subject_id, data_type, task_num=None):
    """保存分析结果"""
    # 确定保存路径
    if data_type == "rest":
        save_dir = os.path.join('H:', 'EEG_EyeTracking', '静息态', subject_id)
    else:
        save_dir = os.path.join('H:', 'EEG_EyeTracking', '任务态', subject_id, f'任务{task_num}')
    
    os.makedirs(save_dir, exist_ok=True)
    # 注意：不再保存.npy文件，所有结果都记录在日志中

def main():
    # 设置要处理的年龄组文件夹
    age_groups = [
         # r'H:\EEG_EyeTracking\6-9岁（n=24）'
         # r'H:\EEG_EyeTracking\Ages 10-11 (n=21)'
         # r'H:\EEG_EyeTracking\Ages 12-13 (n=22)'
         r'H:\EEG_EyeTracking\Ages 14-17 (n=26)'
    ]
    
    # 处理每个年龄组
    for age_group in age_groups:
        print(f"\n开始处理年龄组: {os.path.basename(age_group)}")
        
        # 获取该年龄组下的所有被试文件夹
        subject_folders = [f for f in os.listdir(age_group) 
                         if os.path.isdir(os.path.join(age_group, f))]
        
        # 处理每个被试
        for subject_folder in subject_folders:
            subject_path = os.path.join(age_group, subject_folder)
            process_subject_data(subject_path, subject_folder)

if __name__ == "__main__":
    # 从原始代码文件导入所需函数
    from SPRiNT_py_ground_truth import SPRiNT_stft_py, SPRiNT_remove_outliers
    main()
