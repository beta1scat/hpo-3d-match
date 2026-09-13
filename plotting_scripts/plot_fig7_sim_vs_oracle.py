"""Plot Figure 3-7: Real-world industrial domain baseline vs Real-world Oracle performance ceiling.
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


def load_oracle_data(model_name: str):
    p = RESULTS_DIR / f"itodd_manual_annotated_{model_name}_tpe_median_lexrecall_b5000_s42" / "experiment_summary.json"
    if not p.exists():
        return 0.0, [], []
    d = json.loads(p.read_text(encoding="utf-8"))
    base_f1 = d.get("default_baseline", {}).get("f1", 0.0)
    ckpts = d.get("checkpoints", [])
    x = [c["checkpoint"] for c in ckpts]
    f1 = [c.get("f1", 0.0) for c in ckpts]
    return base_f1, x, f1


def main():
    setup_plt_style()
    
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.0))
    
    # Subplot A: Performance Leap (Default Baseline vs Real Oracle b5000)
    models = ["薄板件", "连续对称螺栓", "离散对称把手"]
    model_keys = ["bracket_planar", "screw_black", "star"]
    
    f1_default = []
    f1_oracle = []
    checkpoints_dict = {}
    
    for k in model_keys:
        base_f1, x_ck, f1_ck = load_oracle_data(k)
        f1_default.append(base_f1)
        f1_oracle.append(f1_ck[-1] if f1_ck else 0.0)
        checkpoints_dict[k] = (x_ck, f1_ck)
        
    x = np.arange(len(models))
    width = 0.32
    
    r1 = axes[0].bar(x - width/2, f1_default, width, label="默认基线 (Baseline)", color=THEME_GRAY, edgecolor=THEME_GRAY)
    r2 = axes[0].bar(x + width/2, f1_oracle, width, label="真实域 5000 轮最优 (Oracle)", color=THEME_BLUE, edgecolor=THEME_BLUE_DARK)
    
    # Value annotations
    for idx in range(len(models)):
        axes[0].annotate(f"{f1_default[idx]:.3f}", xy=(x[idx]-width/2, f1_default[idx]+0.02), ha="center", va="bottom", fontsize=16.0)
        axes[0].annotate(f"{f1_oracle[idx]:.3f}", xy=(x[idx]+width/2, f1_oracle[idx]+0.02), ha="center", va="bottom", fontsize=16.5, fontweight="bold", color=THEME_BLUE_DARK)
        
        # Improvement percentage
        if f1_default[idx] > 0:
            rel_gain = (f1_oracle[idx] - f1_default[idx]) / f1_default[idx] * 100.0
            axes[0].text(x[idx] + width/2, f1_oracle[idx] + 0.11, f"+{rel_gain:.1f}%", ha="center", va="bottom", fontsize=16.0, color=THEME_RED, fontweight="bold")
    
    axes[0].set_title("(a) 默认基线与 5000 轮 HPO 性能对比", pad=12)
    axes[0].set_ylabel("真实标注验证集 $F_1$ 分数")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models)
    axes[0].set_ylim(0, 1.28)
    axes[0].grid(True, axis="y", linestyle="--", alpha=0.5)
    axes[0].legend(loc="upper left", frameon=True, fontsize=15.0)
    
    # Subplot B: Oracle Convergence Trajectory over 5,000 trials
    colors = {"bracket_planar": THEME_BLUE, "screw_black": THEME_GREEN, "star": THEME_ORANGE}
    labels = {
        "bracket_planar": "薄板件",
        "screw_black": "连续对称螺栓",
        "star": "离散对称把手",
    }
    markers = {"bracket_planar": "o", "screw_black": "^", "star": "s"}
    
    for k in model_keys:
        x_ck, f1_ck = checkpoints_dict[k]
        if x_ck:
            axes[1].plot(x_ck, f1_ck, marker=markers[k], color=colors[k], label=labels[k], lw=2.2, markersize=6.5)
        
    axes[1].set_title("(b) 真实物理域 5000 轮寻优收敛曲线", pad=12)
    axes[1].set_xlabel("优化试验轮数 (Trials)")
    axes[1].set_ylabel("真实标注集 $F_1$ 演化轨迹")
    axes[1].set_ylim(0.15, 0.95)
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(loc="lower right", frameon=True, fontsize=15.0)
    
    plt.tight_layout()
    save_fig(fig, "ch3_fig7_sim_vs_oracle")
    plt.close()


if __name__ == "__main__":
    main()
