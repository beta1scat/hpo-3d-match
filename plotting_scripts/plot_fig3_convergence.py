"""Plot Figure 3-3: Optimization convergence curves comparing TPE, CMA-ES, and Random.
"""

import sqlite3
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from plot_config import (
    RESULTS_DIR,
    THEME_BLUE,
    THEME_ORANGE,
    THEME_GRAY,
    setup_plt_style,
    save_fig,
)


def extract_convergence_from_db(db_path: Path):
    """Extract (trial_numbers, best_so_far_values) from an Optuna SQLite DB."""
    if not db_path.exists():
        return [], []
    conn = sqlite3.connect(str(db_path))
    c = conn.cursor()
    c.execute("""
        SELECT t.number, tv.value 
        FROM trials t 
        JOIN trial_values tv ON t.trial_id = tv.trial_id 
        WHERE t.state = 'COMPLETE' 
        ORDER BY t.number ASC
    """)
    rows = c.fetchall()
    conn.close()
    if not rows:
        return [], []
    
    trials = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    # Compute cumulative best
    best_vals = []
    current_best = float("inf")
    for v in vals:
        if v < current_best:
            current_best = v
        best_vals.append(current_best)
    return trials, best_vals


def main():
    setup_plt_style()
    
    models = [
        ("bracket_planar", "(a) 薄板零件 (bracket_planar)"),
        ("screw_black", "(b) 连续对称螺栓 (screw_black)"),
        ("star", "(c) 离散对称把手 (star)"),
    ]
    
    samplers = [
        ("TPE", "tpe_nop_lexrecall_b500_s42", THEME_BLUE, "-", "TPE 贝叶斯优化"),
        ("CmaEs", "cmaes_nop_lexrecall_b500_s42", THEME_ORANGE, "--", "CMA-ES 进化策略"),
        ("Random", "random_nop_lexrecall_b500_s42", THEME_GRAY, ":", "Random 随机搜索基线"),
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), sharex=True)
    
    for ax, (model_name, title) in zip(axes, models):
        for s_name, exp_suffix, color, ls, label in samplers:
            exp_dir = RESULTS_DIR / f"{model_name}_{exp_suffix}"
            db_path = exp_dir / "studies" / f"{model_name}.db"
            trials, best_vals = extract_convergence_from_db(db_path)
            if trials:
                ax.plot(trials, best_vals, label=label, color=color, linestyle=ls, lw=1.8)
        
        ax.set_title(title, pad=8)
        ax.set_xlabel("搜索试验轮数 (Trials)")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_xlim(0, 500)
    
    axes[0].set_ylabel("当前历史最佳目标损失 $J_{\mathrm{train}}(\\boldsymbol{\\theta})$")
    # Place unified legend on top
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.06), ncol=3, frameon=True)
    
    plt.tight_layout()
    save_fig(fig, "ch3_fig3_convergence")
    plt.close()


if __name__ == "__main__":
    main()
