import numpy as np
import matplotlib.pyplot as plt
import re
import os
from datetime import datetime
import logging

def setup_logging():
    """设置日志记录器"""
    log_dir = os.path.join('f:', 'eye_eeg', 'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'sprint_comparison_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def parse_log_file(log_file_path, target_channel=72):
    """
    解析SPRiNT分析的log文件，提取指定通道的参数
    
    参数:
    - log_file_path: log文件路径
    - target_channel: 目标通道号（默认72）
    
    返回:
    - dict: 包含offset和exponent数据的字典
    """
    logger = logging.getLogger(__name__)
    
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 查找目标通道的数据
        channel_pattern = rf'通道 {target_channel} 的参数:'
        channel_match = re.search(channel_pattern, content)
        
        if not channel_match:
            logger.warning(f"未找到通道 {target_channel} 的数据")
            return None
        
        # 从匹配位置开始查找参数
        start_pos = channel_match.end()
        
        # 查找下一个通道的开始位置（作为结束标记）
        next_channel_pattern = r'通道 \d+ 的参数:'
        next_channel_match = re.search(next_channel_pattern, content[start_pos:])
        
        if next_channel_match:
            end_pos = start_pos + next_channel_match.start()
            channel_content = content[start_pos:end_pos]
        else:
            channel_content = content[start_pos:]
        
        # 提取offset和exponent数据
        offset_data = []
        exponent_data = []
        time_points = []
        
        # 查找所有时间点的数据
        time_pattern = r'时间点 (\d+):'
        offset_pattern = r'- 偏移\(offset\): ([+-]?\d*\.?\d+)'
        exponent_pattern = r'- 指数\(exponent\): ([+-]?\d*\.?\d+)'
        
        time_matches = re.finditer(time_pattern, channel_content)
        
        for time_match in time_matches:
            time_point = int(time_match.group(1))
            time_points.append(time_point)
            
            # 在当前时间点后查找offset和exponent
            time_start = time_match.end()
            
            # 查找下一个时间点的位置（作为搜索范围的结束）
            next_time_match = re.search(time_pattern, channel_content[time_start:])
            if next_time_match:
                time_end = time_start + next_time_match.start()
                time_content = channel_content[time_start:time_end]
            else:
                time_content = channel_content[time_start:]
            
            # 提取offset
            offset_match = re.search(offset_pattern, time_content)
            if offset_match:
                offset_data.append(float(offset_match.group(1)))
            else:
                offset_data.append(np.nan)
            
            # 提取exponent
            exponent_match = re.search(exponent_pattern, time_content)
            if exponent_match:
                exponent_data.append(float(exponent_match.group(1)))
            else:
                exponent_data.append(np.nan)
        
        logger.info(f"成功解析通道 {target_channel} 的数据，共 {len(time_points)} 个时间点")
        
        return {
            'time_points': np.array(time_points),
            'offset': np.array(offset_data),
            'exponent': np.array(exponent_data),
            'channel': target_channel
        }
        
    except Exception as e:
        logger.error(f"解析log文件时出错: {str(e)}")
        return None

def plot_sprint_comparison(rest_data, task_data, save_path=None):
    """
    绘制SPRiNT静息态与任务态对比图
    
    参数:
    - rest_data: 静息态数据字典
    - task_data: 任务态数据字典
    - save_path: 保存路径（可选）
    """
    logger = logging.getLogger(__name__)
    
    # 设置图形样式
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'lines.linewidth': 2.0,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.facecolor': 'white',
        'figure.facecolor': 'white',
    })
    
    # 创建图形
    fig, ax1 = plt.subplots(figsize=(12, 8))
    ax2 = ax1.twinx()
    
    # 创建时间轴（假设每个时间点间隔3秒，基于STFT窗口设置）
    if task_data:
        time_axis = task_data['time_points'] * 3.0  # 转换为秒
        
        # 绘制任务态数据的填充区域 - 使用稍深的颜色
        ax1.fill_between(time_axis, task_data['exponent']-0.15, task_data['exponent']+0.15, 
                        color='#9AC5FF', alpha=0.4, label='Exponent', zorder=1)
        ax2.fill_between(time_axis, task_data['offset']-0.3, task_data['offset']+0.3, 
                        color='#FFD699', alpha=0.4, label='Offset', zorder=1)
        
        # 绘制任务态数据的线条 - 不添加到图例中
        ax1.plot(time_axis, task_data['exponent'], color='#4472C4', linewidth=2.0, zorder=3)
        ax2.plot(time_axis, task_data['offset'], color='#E1A84A', linewidth=2.0, zorder=3)
    
    # 绘制静息态数据作为ground truth
    if rest_data:
        rest_time_axis = rest_data['time_points'] * 3.0  # 转换为秒
        
        # 直接使用静息态数据，显示为变化的线条
        # 黑实线表示静息态的offset
        ax2.plot(rest_time_axis, rest_data['offset'], 
                color='black', linewidth=2.0, linestyle='-', 
                label='Ground truth', zorder=4)
        
        # 黑虚线表示静息态的exponent - 不添加到图例中
        ax1.plot(rest_time_axis, rest_data['exponent'], 
                color='black', linewidth=2.0, linestyle='--', zorder=4)
    
    # 设置轴标签
    ax1.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Exponent (Hz)', fontsize=12, fontweight='bold', color='#4472C4')
    ax2.set_ylabel('Offset (a.u.)', fontsize=12, fontweight='bold', color='#E1A84A')
    
    # 设置轴刻度颜色
    ax1.tick_params(axis='y', labelcolor='#4472C4')
    ax2.tick_params(axis='y', labelcolor='#E1A84A')
    
    # 设置标题
    plt.title('SPRiNT', fontsize=16, fontweight='bold', pad=20)
    
    # 移除图框顶部边框
    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    
    # 创建图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', 
              frameon=True, fontsize=10)
    
    # 设置X轴范围为0-300秒
    ax1.set_xlim(0, 300)
    
    # 设置网格
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax1.set_axisbelow(True)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"图片已保存到: {save_path}")
        plt.close()
    else:
        plt.show()

def main():
    """
    主函数 - 执行SPRiNT静息态与任务态对比分析
    """
    # 设置日志记录
    logger = setup_logging()
    logger.info("开始SPRiNT对比分析")
    
    # 文件路径
    rest_log_path = r'H:\EEG_EyeTracking\静息态\A00053375\SPRiNT_analysis_20250808_172619.log'
    task_log_path = r'H:\EEG_EyeTracking\任务态\A00053375\任务1\SPRiNT_analysis_20250808_173321.log'
    
    # 检查文件是否存在
    if not os.path.exists(rest_log_path):
        logger.warning(f"静息态log文件不存在: {rest_log_path}")
        rest_log_path = None
    
    if not os.path.exists(task_log_path):
        logger.warning(f"任务态log文件不存在: {task_log_path}")
        task_log_path = None
    
    if rest_log_path is None and task_log_path is None:
        logger.error("无法找到任何log文件")
        return
    
    # 目标通道
    target_channel = 72
    
    try:
        # 解析静息态数据
        rest_data = None
        if rest_log_path:
            logger.info(f"解析静息态数据: {rest_log_path}")
            rest_data = parse_log_file(rest_log_path, target_channel)
        
        # 解析任务态数据
        task_data = None
        if task_log_path:
            logger.info(f"解析任务态数据: {task_log_path}")
            task_data = parse_log_file(task_log_path, target_channel)
        
        if rest_data is None and task_data is None:
            logger.error("无法解析任何数据文件")
            return
        
        # 生成保存路径
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = f'f:/eye_eeg/sprint_comparison_channel_{target_channel}_{timestamp}.png'
        
        # 绘制对比图
        logger.info("开始绘制对比图")
        plot_sprint_comparison(rest_data, task_data, save_path)
        
        logger.info("SPRiNT对比分析完成")
        
    except Exception as e:
        logger.error(f"分析过程中出现错误: {str(e)}")

if __name__ == "__main__":
    main()