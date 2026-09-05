"""Plot Figure 3-5: Median Pruner acceleration efficiency and trial survival analysis.
"""

import sqlite3
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from plot_config import (
    RESULTS_DIR,
    THEME_BLUE,
    THEME_BLUE_DARK,
    THEME_BLUE_LIGHT,
    THEME_GRAY,
    THEME_ORANGE,
    THEME_GREEN,
    setup_plt_style,
    save_fig,
)


def get_trial_durations_from_db(db_path: Path):
    if not db_path.exists():
        return []
    conn = sqlite3.connect(str(db_path))
    c = conn.cursor()
    c.execute("SELECT (strftime('%s', datetime_complete) - strftime('%s', datetime_start)) FROM trials WHERE datetime_complete IS NOT NULL")
    rows = c.fetchall()
    conn.close()
    return [r[0] for r in rows if r[0] is not None and r[0] >= 0]


def main():
    setup_plt_style()
    
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
    
    # Data for Subplot A: Total wall-clock duration (Hours)
    models = ["薄板件\n(bracket_planar)", "连续对称螺栓\n(screw_black)", "离散对称把手\n(star)"]
    dur_nop = [9.97, 8.16, 9.06]
    dur_med = [7.61, 5.98, 8.90]
    speedups = [f"{n/m:.2f}×" for n, m in zip(dur_nop, dur_med)]
    
    x = np.arange(len(models))
    width = 0.35
    
    rects1 = axes[0].bar(x - width/2, dur_nop, width, label="无剪枝全量评估 (Nop)", color=THEME_BLUE_LIGHT, edgecolor=THEME_BLUE)
    rects2 = axes[0].bar(x + width/2, dur_med, width, label="中位数早停剪枝 (Median)", color=THEME_BLUE, edgecolor=THEME_BLUE_DARK)
    
    # Add speedup annotations
    for idx, (r1, r2, sp) in enumerate(zip(rects1, rects2, speedups)):
        h1 = r1.get_height()
        h2 = r2.get_height()
        axes[0].annotate(sp, xy=(x[idx] + width/2, h2 + 0.18), ha="center", va="bottom", fontsize=9.5, fontweight="bold", color=THEME_BLUE_DARK)
    
    axes[0].set_title("(a) 500 轮优化全流程搜索耗时对比 (小时)")
    axes[0].set_ylabel("总耗时 (h)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models)
    axes[0].set_ylim(0, 11.8)
    axes[0].grid(True, axis="y", linestyle="--", alpha=0.5)
    axes[0].legend(loc="upper right", frameon=True)
    
    # Data for Subplot B: Trial composition (Completed vs Pruned)
    completed_counts = [364, 363, 500]
    pruned_counts = [136, 137, 0]
    
    axes[1].bar(x, completed_counts, width * 1.3, label="完整评估试验 (Completed)", color=THEME_BLUE)
    axes[1].bar(x, pruned_counts, width * 1.3, bottom=completed_counts, label="提前截断试验 (Pruned)", color=THEME_ORANGE)
    
    for idx in range(len(models)):
        total = completed_counts[idx] + pruned_counts[idx]
        p_pct = pruned_counts[idx] / total * 100
        txt = f"剪枝率: {p_pct:.1f}%" if p_pct > 0 else "未触发剪枝"
        axes[1].annotate(txt, xy=(x[idx], total + 12), ha="center", va="bottom", fontsize=9.5, color=THEME_ORANGE if p_pct > 0 else THEME_GRAY)
        
    axes[1].set_title("(b) 500 轮优化试验状态分布与早停触发率")
    axes[1].set_ylabel("试验数量 (Trials)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models)
    axes[1].set_ylim(0, 580)
    axes[1].grid(True, axis="y", linestyle="--", alpha=0.5)
    axes[1].legend(loc="upper right", frameon=True)
    
    plt.tight_layout()
    save_fig(fig, "ch3_fig5_pruning_analysis")
    plt.close()


if __name__ == "__main__":
    main()
