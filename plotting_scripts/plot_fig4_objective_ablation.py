"""Plot Figure 3-4: Objective formulation ablation (Lexicographical Recall-First vs Fixed Penalty).
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt

from plot_config import (
    RESULTS_DIR,
    THEME_BLUE,
    THEME_RED,
    THEME_GREEN,
    THEME_ORANGE,
    setup_plt_style,
    save_fig,
)


def load_checkpoints(exp_name: str):
    p = RESULTS_DIR / exp_name / "experiment_summary.json"
    if not p.exists():
        return [], [], []
    d = json.loads(p.read_text(encoding="utf-8"))
    ckpts = d.get("checkpoints", [])
    x = [c["checkpoint"] for c in ckpts]
    tp = [c.get("tp", 0) for c in ckpts]
    f1 = [c.get("f1", 0.0) for c in ckpts]
    return x, tp, f1


def main():
    setup_plt_style()
    
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.8))
    
    # 1. bracket_planar TP evolution
    x_lex, tp_lex, f1_lex = load_checkpoints("bracket_planar_tpe_nop_lexrecall_b500_s42")
    x_fix, tp_fix, f1_fix = load_checkpoints("bracket_planar_tpe_nop_fixedpen_b500_s42")
    
    axes[0].plot(x_lex, tp_lex, marker="o", color=THEME_BLUE, label="分层字典序召回优先", lw=2.2, markersize=6.5)
    axes[0].plot(x_fix, tp_fix, marker="s", color=THEME_RED, linestyle="--", label="传统固定常数惩罚", lw=2.2, markersize=6.5)
    axes[0].set_title("(a) 薄板件检出数演化", pad=12)
    axes[0].set_xlabel("检查点试验轮数 (Trials)", labelpad=6)
    axes[0].set_ylabel("正确检出数 (TP)", labelpad=6)
    axes[0].set_ylim(-0.5, 15.5)
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(loc="upper left", fontsize=15)
    
    # 2. screw_black TP evolution
    x_lex_s, tp_lex_s, f1_lex_s = load_checkpoints("screw_black_tpe_nop_lexrecall_b500_s42")
    x_fix_s, tp_fix_s, f1_fix_s = load_checkpoints("screw_black_tpe_nop_fixedpen_b500_s42")
    
    axes[1].plot(x_lex_s, tp_lex_s, marker="o", color=THEME_BLUE, label="分层字典序召回优先", lw=2.2, markersize=6.5)
    axes[1].plot(x_fix_s, tp_fix_s, marker="s", color=THEME_RED, linestyle="--", label="传统固定常数惩罚", lw=2.2, markersize=6.5)
    axes[1].set_title("(b) 螺栓件检出数演化", pad=12)
    axes[1].set_xlabel("检查点试验轮数 (Trials)", labelpad=6)
    axes[1].set_ylabel("正确检出数 (TP)", labelpad=6)
    axes[1].set_ylim(20.5, 45.0)
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(loc="lower right", fontsize=15)
    
    # 3. F1 score comparison across checkpoints for star & bracket
    axes[2].plot(x_lex, f1_lex, marker="o", color=THEME_BLUE, label="薄板件 (字典序)", lw=2.2, markersize=6.5)
    axes[2].plot(x_fix, f1_fix, marker="o", color="#8FAADC", linestyle="--", label="薄板件 (固定惩罚)", lw=2.2, markersize=6.5)
    axes[2].plot(x_lex_s, f1_lex_s, marker="^", color=THEME_GREEN, label="螺栓件 (字典序)", lw=2.2, markersize=6.5)
    axes[2].plot(x_fix_s, f1_fix_s, marker="^", color=THEME_ORANGE, linestyle="--", label="螺栓件 (固定惩罚)", lw=2.2, markersize=6.5)
    axes[2].set_title(r"(c) 综合 $F_1$ 分数演化轨迹", pad=12)
    axes[2].set_xlabel("检查点试验轮数 (Trials)", labelpad=6)
    axes[2].set_ylabel(r"测试集 $F_1$ 分数", labelpad=6)
    axes[2].set_ylim(-0.01, 0.38)
    axes[2].grid(True, linestyle="--", alpha=0.5)
    axes[2].legend(loc="upper right", fontsize=14, ncol=1, framealpha=0.9)
    
    plt.tight_layout()
    save_fig(fig, "ch3_fig4_objective_ablation")
    plt.close()


if __name__ == "__main__":
    main()
