"""Schematic of the EEG <-> pupil time alignment used in mi_robustness_analysis.py.

Not an analysis script: it draws one explanatory figure
(``fig_time_alignment.png``) from the documented time grids.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(r"f:\eye_eeg\fig_time_alignment.png")

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

fig, axes = plt.subplots(3, 1, figsize=(11, 7.6), sharex=True,
                         gridspec_kw={"height_ratios": [1.0, 1.0, 1.25]})

L = 2.0      # SPRiNT window length (s)
HOP = 1.0    # SPRiNT step (s)
N = 6        # number of EEG time points shown

centers = np.arange(N) * HOP + 1.0          # k+1 s, k = 0..N-1
pupil_t = np.arange(1, N + 1) * 1.001       # 2 s STFT, 50 % overlap, 120 Hz log

# ---------------------------------------------------------------- (A) EEG grid
ax = axes[0]
for k in range(N):
    ax.add_patch(plt.Rectangle((k * HOP, 0.25), L, 0.5, facecolor="#4C72B0",
                               alpha=0.18 + 0.04 * (k % 2), edgecolor="#2F4B7C",
                               linewidth=0.8))
    ax.text(k * HOP + L / 2, 0.5, f"window {k}", ha="center", va="center",
            fontsize=8, color="#2F4B7C")
ax.plot(centers, np.full(N, 0.08), "o", color="#2F4B7C", ms=6, zorder=5)
for c in centers:
    ax.annotate(f"{c:.0f} s", (c, 0.08), textcoords="offset points",
                xytext=(0, -12), ha="center", fontsize=8)
ax.set_ylim(-0.05, 0.95)
ax.set_yticks([])
ax.set_ylabel("EEG", rotation=0, labelpad=18, va="center")
ax.set_title("EEG（SPRiNT 参数序列）：2 s 窗、50 % 重叠、步长 1 s；"
             "第 k 个时间点中心 = k+1 s", fontsize=11, loc="left")

# -------------------------------------------------------------- (B) pupil grid
ax = axes[1]
keep = [True, True, True, False, True, True]      # 第 4 个时间点缺失
ax.plot(pupil_t[keep], np.full(sum(keep), 0.55), "o", color="#DD8452", ms=6, zorder=5)
for c in centers:
    ax.axvline(c, color="#4C72B0", ls="--", lw=0.8, alpha=0.7, zorder=1)
ax.axvspan(2.0 - 0.5, 2.0 + 0.5, color="#4C72B0", alpha=0.08, zorder=0)
ax.annotate("±0.5 s：取最近实测样本", (2.0, 0.85), xytext=(2.15, 0.95),
            fontsize=9, color="#2F4B7C",
            arrowprops=dict(arrowstyle="->", color="#2F4B7C", lw=0.8))
ax.annotate("缺失（> 2 s 间隔）", (pupil_t[3], 0.55), xytext=(pupil_t[3] - 1.3, 0.2),
            fontsize=9, color="#C44E52",
            arrowprops=dict(arrowstyle="->", color="#C44E52", lw=0.8))
ax.set_ylim(0.0, 1.15)
ax.set_yticks([])
ax.set_ylabel("瞳孔", rotation=0, labelpad=18, va="center")
ax.set_title("瞳孔（STFT 时间序列）：120 Hz 日志、2 s 窗、50 % 重叠；"
             "首个时间点 ≈1.001 s、步长 ≈1.001 s，允许缺口", fontsize=11, loc="left")

# ------------------------------------------------------------ (C) merged series
ax = axes[2]
state = ["direct", "direct", "interp", "invalid", "interp", "direct"]
style = {"direct": ("o", "#4C72B0", "实测样本（|Δt| ≤ 0.5 s）"),
         "interp": ("s", "#55A868", "线性插值（缺口 ≤ 2 s）"),
         "invalid": ("x", "#C44E52", "无效（缺口 > 2 s 或超出范围）")}
vals = [0.62, 0.70, 0.42, np.nan, 0.58, 0.66]
for c, st, v in zip(centers, state, vals):
    m, col, _ = style[st]
    ax.plot(c, 0.2 if st == "invalid" else v, m, color=col, ms=8, zorder=5)
for c in centers:
    ax.axvline(c, color="#4C72B0", ls="--", lw=0.7, alpha=0.5, zorder=1)
ax.set_ylim(0.0, 1.0)
ax.set_yticks([])
ax.set_xlim(0.2, centers[-1] + 0.9)
ax.set_xlabel("时间（s）")
ax.set_ylabel("对齐后", rotation=0, labelpad=18, va="center")
ax.set_title("对齐结果：x = EEG 参数（119 个通道）、y = 同一时间索引的瞳孔值；"
             "两序列逐点一一对应后计算 MI", fontsize=11, loc="left")

handles = [plt.Line2D([], [], ls="", marker=m, color=col, ms=8, label=lab)
           for m, col, lab in style.values()]
axes[2].legend(handles=handles, loc="upper right", fontsize=8, frameon=False)

fig.suptitle("图 1  时间对齐示意（不是按数据缩放的真实波形）", fontsize=12, y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.97))
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=180)
print(f"saved {OUT}")
