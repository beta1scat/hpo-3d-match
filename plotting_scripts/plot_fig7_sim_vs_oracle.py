"""Plot Figure 3-7: Sim2Real synthetic domain vs Real-world Oracle performance ceiling.
"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from plot_config import (
    RESULTS_DIR,
    THEME_BLUE,
    THEME_BLUE_DARK,
    THEME_BLUE_LIGHT,
    THEME_GREEN,
    THEME_ORANGE,
    THEME_GRAY,
    THEME_RED,
    setup_plt_style,
    save_fig,
)


def load_oracle_checkpoints(model_name: str):
    p = RESULTS_DIR / f"itoddmv_val_{model_name}_tpe_median_lexrecall_b10000_s42" / "experiment_summary.json"
    if not p.exists():
        return [], []
    d = json.loads(p.read_text(encoding="utf-8"))
    ckpts = d.get("checkpoints", [])
    x = [c["checkpoint"] for c in ckpts]
    f1 = [c.get("f1", 0.0) for c in ckpts]
    return x, f1


def main():
    setup_plt_style()
    
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))
    
    # Subplot A: Multi-stage Performance Leap (Default -> Sim2Real -> Real Oracle)
    models = ["薄板件\n(bracket_planar)", "连续对称螺栓\n(screw_black)", "离散对称把手\n(star)"]
    f1_default = [0.0000, 0.0875, 0.0170]
    f1_sim2real = [0.0378, 0.1481, 0.0346]
    f1_oracle = [0.7500, 1.0000, 0.9231]
    
    x = np.arange(len(models))
    width = 0.25
    
    r1 = axes[0].bar(x - width, f1_default, width, label="默认基线参数 (Default)", color=THEME_GRAY, edgecolor=THEME_GRAY)
    r2 = axes[0].bar(x, f1_sim2real, width, label="仿真域迁移参数 (Sim2Real PBR)", color=THEME_BLUE_LIGHT, edgecolor=THEME_BLUE)
    r3 = axes[0].bar(x + width, f1_oracle, width, label="真实域理论天花板 (Real Oracle)", color=THEME_BLUE, edgecolor=THEME_BLUE_DARK)
    
    # Value annotations
    for idx in range(len(models)):
        axes[0].annotate(f"{f1_default[idx]:.3f}", xy=(x[idx]-width, f1_default[idx]+0.02), ha="center", va="bottom", fontsize=8.5)
        axes[0].annotate(f"{f1_sim2real[idx]:.3f}", xy=(x[idx], f1_sim2real[idx]+0.02), ha="center", va="bottom", fontsize=8.5)
        axes[0].annotate(f"{f1_oracle[idx]:.3f}", xy=(x[idx]+width, f1_oracle[idx]+0.02), ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=THEME_BLUE_DARK)
    
    axes[0].set_title("(a) 三阶段算法性能阶跃对比 ($F_1$ 分数)")
    axes[0].set_ylabel("测试集 $F_1$ 分数")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models)
    axes[0].set_ylim(0, 1.15)
    axes[0].grid(True, axis="y", linestyle="--", alpha=0.5)
    axes[0].legend(loc="upper left", frameon=True)
    
    # Subplot B: Oracle Convergence Trajectory over 10,000 trials
    ck_b, f1_b = load_oracle_checkpoints("bracket_planar")
    ck_s, f1_s = load_oracle_checkpoints("screw_black")
    ck_st, f1_st = load_oracle_checkpoints("star")
    
    if ck_b:
        axes[1].plot(ck_b, f1_b, marker="o", color=THEME_BLUE, label="薄板件 (bracket_planar) Oracle", lw=1.8)
    if ck_s:
        axes[1].plot(ck_s, f1_s, marker="^", color=THEME_GREEN, label="连续对称螺栓 (screw_black) Oracle", lw=1.8)
    if ck_st:
        axes[1].plot(ck_st, f1_st, marker="s", color=THEME_ORANGE, label="离散对称把手 (star) Oracle", lw=1.8)
        
    axes[1].set_title("(b) 真实域超长预算 (10000 轮) 性能收敛曲线")
    axes[1].set_xlabel("优化试验轮数 (Trials)")
    axes[1].set_ylabel("验证集 $F_1$ 演化轨迹")
    axes[1].set_ylim(0.2, 1.05)
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(loc="lower right", frameon=True)
    
    plt.tight_layout()
    save_fig(fig, "ch3_fig7_sim_vs_oracle")
    plt.close()


if __name__ == "__main__":
    main()
