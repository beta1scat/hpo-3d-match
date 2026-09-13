"""Plot Figure 3-6: Hyperparameter importance and sensitivity analysis.
"""

import optuna
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from plot_config import (
    RESULTS_DIR,
    THEME_BLUE,
    THEME_BLUE_DARK,
    THEME_BLUE_LIGHT,
    THEME_ORANGE,
    THEME_GREEN,
    setup_plt_style,
    save_fig,
)


def compute_importances(model_name: str):
    db_path = RESULTS_DIR / f"{model_name}_tpe_nop_lexrecall_b500_s42" / "studies" / f"{model_name}.db"
    if not db_path.exists():
        return {}
    study = optuna.load_study(
        study_name=f"{model_name}_lexicographical-recall-first_TPE_Nop_repeat0_seed42",
        storage=f"sqlite:///{db_path}"
    )
    try:
        importances = optuna.importance.get_param_importances(study)
    except Exception as e:
        print(f"Error computing importance for {model_name}: {e}")
        importances = {}
    return importances


def main():
    setup_plt_style()
    
    models = [
        ("bracket_planar", "(a) 薄板件"),
        ("screw_black", "(b) 连续对称螺栓"),
        ("star", "(c) 离散对称把手"),
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.2), sharey=True)
    
    param_labels = {
        "RelSamplingDistance": "空间采样距离 (RelSamplingDistance)",
        "KeyPointFraction": "关键点比例 (KeyPointFraction)",
        "min_score": "候选最小得分 (min_score)",
        "max_overlap_dist_rel": "重叠距离阈值 (max_overlap_dist_rel)",
        "pose_ref_num_steps": "ICP 迭代步数 (pose_ref_num_steps)",
        "pose_ref_sub_sampling": "ICP 降采样率 (pose_ref_sub_sampling)",
        "pose_ref_dist_threshold_rel": "ICP 距离门限 (pose_ref_dist_threshold_rel)",
        "pose_ref_scoring_dist_rel": "ICP 评分距离 (pose_ref_scoring_dist_rel)",
        "pose_ref_use_scene_normals": "场景法线开关 (use_scene_normals)",
    }
    
    for ax, (model_name, title) in zip(axes, models):
        imps = compute_importances(model_name)
        if not imps:
            # Fallback heuristic if fANOVA fails due to single-eval cases
            continue
        
        # Sort top params
        sorted_items = sorted(imps.items(), key=lambda x: x[1], reverse=True)
        keys = [k for k, v in sorted_items]
        vals = [v * 100.0 for k, v in sorted_items]
        labels = [param_labels.get(k, k) for k in keys]
        
        y_pos = np.arange(len(keys))
        colors = [THEME_BLUE if i < 3 else THEME_BLUE_LIGHT for i in range(len(keys))]
        
        bars = ax.barh(y_pos, vals, color=colors, edgecolor=THEME_BLUE_DARK, height=0.6)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=17.0)
        ax.invert_yaxis()  # top-down
        ax.set_title(title, pad=10)
        ax.set_xlabel("参数重要度贡献率 (%)")
        ax.set_xlim(0, 105)
        ax.grid(True, axis="x", linestyle="--", alpha=0.5)
        
        for bar in bars:
            w = bar.get_width()
            if w > 1.5:
                ax.text(w + 1.2, bar.get_y() + bar.get_height()/2, f"{w:.1f}%", va="center", fontsize=16.0, fontweight="bold")
    
    plt.tight_layout()
    save_fig(fig, "ch3_fig6_param_importance")
    plt.close()


if __name__ == "__main__":
    main()
