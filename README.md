# 情绪视频观看任务的 EEG–眼动耦合分析：论文代码与说明

本仓库包含论文《探究不同情绪状态下眼动脑电的耦合》相关分析的全部代码与说明文档。
**仓库不包含任何原始数据**：EEG 参数日志、瞳孔日志与公众人口学表均不在版本库内。

## 目录结构

```
.
├── mi_robustness_analysis_protocol.md      # MI 稳健性分析的唯一方法学规范（预注册口径）
├── MI稳健性分析_修改与结果总结.txt              # 代码改动明细 + 结果落点 + 运行日志
├── 文字说明_时间对齐验证与样本可用性.txt          # 时间对齐方法与样本可用性漏斗（可直接引用的正文）
├── requirements.txt                        # Python 依赖
├── fig_time_alignment.png                  # 时间对齐示意图（EEG 网格 / 瞳孔点 / 对齐结果）
├── ouhe/
│   ├── mi_robustness_analysis.py           # 主脚本：8 个设置的 MI 稳健性分析
│   └── make_alignment_diagram.py           # 生成 fig_time_alignment.png
├── eye_eeg_coupling_analysis.py            # EEG–眼动耦合分析
├── ouhe/pac_coupling_analysis_real_data.py # PAC 耦合计算
├── ouhe/pac_coupling_statistical_analysis.py   # PAC 耦合统计检验
├── ouhe/pac_region_pairwise_stats_violin.py    # 区域两两比较与提琴图
├── ouhe/pac_revision_supplement.py         # 审稿补充分析
├── SPRiNT_batch_process.py                 # SPRiNT 参数拟合批处理
├── SPRiNT_py.py                            # SPRiNT 拟合调用
├── sprint_huatu.py                         # 参数结果绘图
└── gonglvpu.py                             # 功率谱计算
```

## 运行环境

- Python 3.9 及以上
- `pip install -r requirements.txt`

## 主脚本用法

`ouhe/mi_robustness_analysis.py` 有四个阶段，按顺序运行：

```bash
python ouhe/mi_robustness_analysis.py inventory    # 清点可用被试与日志覆盖，写 inventory_*.csv
python ouhe/mi_robustness_analysis.py probe        # 探测自相关衰减、平稳性、通道集合，写 probe_*.csv
python ouhe/mi_robustness_analysis.py run          # 主分析：8 个设置 × 被试，写 segments/surrogate/tts
python ouhe/mi_robustness_analysis.py selfcheck    # 9 项数值自检，写 selfcheck_report.json
```

可选参数：

- `--out <目录>`：结果输出目录，默认 `ouhe/results_mi`
- `--subjects A00053375 A00053398`：只跑指定被试（调试用）
- `--settings native_equal k10`：只跑指定设置

## 分析口径要点

- **统计单位是被试**，不是试次、不是通道；被试级 p 值不做平均。
- **MI 估计器**：主分析用 kNN（KSG），`k = 3`；k = 5 / 10 为敏感性设置。
  另保留 `legacy_same_cohort`（直方图法、10 bin）以复现旧结果，单独列示、不与其他设置混用。
- **等长片段**：同一被试三个条件取共同有效连续区间的最大点数 `n_i = min(...)`，
  每条件取最早满足 `n_i` 的窗口，不做拼接。
- **替代数据**：循环移位构造零分布，`mi_excess = observed − mean(surrogate)`，
  负值保留、不裁剪到 0。
- **TTS**：`d` 从 `−r` 到 `r` 含两端，`p = min(1, B/(r+1))`，分母为 `r+1` 而非 `2r+1`。
- **时间对齐**：以 EEG 参数点中心 `(k+1) s` 为目标网格，瞳孔序列 `|Δt| ≤ 0.5 s` 取最近实测，
  缺口 `≤ 2.0 s` 线性插值，超范围记无效并切断连续有效区间。详见 `文字说明_时间对齐验证与样本可用性.txt`。

## 数据来源

- 公开数据集：MIPDB（Multimodal Infant/Individual Psychological Database）公开文件，用于被试人口学信息与可用性核对。
- 本地预处理产物：SPRiNT 拟合得到的分频段参数日志（`n_tp × 121`，121 = 0–60 Hz @ 0.5 Hz）、
  眼动 STFT 得到的分频段瞳孔参数日志。**两者均不随本仓库分发。**

脚本顶部的 `EEG_LOG_DIR` / `PUPIL_DIR` 为本地绝对路径，复现时需按自己的数据位置修改。

## 说明

仓库中的说明文档同时记录了方法选择理由、已验证与未验证的边界，以及样本从公开表
到最终纳入队列的逐层排除原因，便于审稿核查。
