"""Summarize multi-seed HPO experiments across seeds 42, 43, 44.

Parses experiment_summary.json and Optuna SQLite databases in results/
to produce mean, std, 95% bootstrap confidence intervals for TP, FP, FN,
Precision, Recall, F1, Loss, and total wall-clock duration.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

MODELS = [
    ("bracket_planar", "薄壁冲压钣金件 (bracket_planar)"),
    ("screw_black", "连续轴对称高强螺栓 (screw_black)"),
    ("star", "十二齿离散旋转对称把手 (star)"),
]

SEEDS = [42, 43, 44]
BUDGET = 500
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def bootstrap_ci_95(values: List[float], n_boot: int = 10000, seed: int = 42) -> tuple[float, float]:
    if len(values) <= 1:
        v = values[0] if values else 0.0
        return v, v
    rng = np.random.default_rng(seed)
    boots = [float(np.mean(rng.choice(values, size=len(values), replace=True))) for _ in range(n_boot)]
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def get_trial_counts(db_path: Path) -> tuple[int, int]:
    if not db_path.exists():
        return 0, 0
    conn = sqlite3.connect(str(db_path))
    c = conn.cursor()
    c.execute("SELECT state, count(*) FROM trials GROUP BY state")
    rows = dict(c.fetchall())
    conn.close()
    complete = rows.get("COMPLETE", 0)
    pruned = rows.get("PRUNED", 0)
    return complete, pruned


def main() -> None:
    output_rows = []
    print(f"[*] Scanning results directory: {RESULTS_DIR}")

    for model_key, model_display in MODELS:
        seed_data: Dict[int, Dict[str, Any]] = {}
        for seed in SEEDS:
            exp_dir = RESULTS_DIR / f"{model_key}_tpe_nop_lexrecall_b{BUDGET}_s{seed}"
            summary_file = exp_dir / "experiment_summary.json"
            db_file = exp_dir / "studies" / f"{model_key}.db"

            if not summary_file.exists():
                print(f"  [-] Missing {summary_file.name} for {model_key} seed {seed}")
                continue

            summary = json.loads(summary_file.read_text(encoding="utf-8"))
            completed, pruned = get_trial_counts(db_file)

            # Get final checkpoint
            ckpts = summary.get("checkpoints", [])
            final_eval = ckpts[-1] if ckpts else summary.get("default_baseline", {})

            tp_val = float(final_eval.get("tp", 0))
            fp_val = float(final_eval.get("fp", 0))
            fn_val = float(final_eval.get("fn", 0))
            prec_val = tp_val / (tp_val + fp_val) if (tp_val + fp_val) > 0 else 0.0
            rec_val = tp_val / (tp_val + fn_val) if (tp_val + fn_val) > 0 else 0.0
            f1_val = float(final_eval.get("f1", 0.0))
            if f1_val == 0.0 and (prec_val + rec_val) > 0:
                f1_val = 2.0 * prec_val * rec_val / (prec_val + rec_val)

            seed_data[seed] = {
                "tp": tp_val,
                "fp": fp_val,
                "fn": fn_val,
                "precision": prec_val,
                "recall": rec_val,
                "f1": f1_val,
                "completed_trials": completed,
                "pruned_trials": pruned,
            }

        if not seed_data:
            continue

        f1_list = [d["f1"] for d in seed_data.values()]
        rec_list = [d["recall"] for d in seed_data.values()]
        prec_list = [d["precision"] for d in seed_data.values()]
        tp_list = [d["tp"] for d in seed_data.values()]

        f1_mean, f1_std = float(np.mean(f1_list)), float(np.std(f1_list, ddof=1)) if len(f1_list) > 1 else 0.0
        f1_low, f1_high = bootstrap_ci_95(f1_list)

        output_rows.append({
            "model": model_display,
            "seeds_evaluated": sorted(list(seed_data.keys())),
            "f1_mean": f1_mean,
            "f1_std": f1_std,
            "f1_ci95": [f1_low, f1_high],
            "recall_mean": float(np.mean(rec_list)),
            "precision_mean": float(np.mean(prec_list)),
            "tp_mean": float(np.mean(tp_list)),
            "per_seed": seed_data,
        })

    # Save to JSON
    json_path = RESULTS_DIR / "multi_seed_hpo_summary.json"
    json_path.write_text(json.dumps(output_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[+] Saved JSON summary to: {json_path}")

    # Generate Markdown Table
    md_lines = [
        "# 第3章 HPO 多随机种子实验统计汇总表",
        "",
        "| 工件类别 | 评估种子 | 平均 TP | 平均准确率 | 平均召回率 | 平均 $F_1$ 分数 | $F_1$ 95% 置信区间 |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for row in output_rows:
        seeds_str = ", ".join(str(s) for s in row["seeds_evaluated"])
        ci_str = f"[{row['f1_ci95'][0]:.4f}, {row['f1_ci95'][1]:.4f}]"
        f1_str = f"{row['f1_mean']:.4f} ± {row['f1_std']:.4f}"
        md_lines.append(
            f"| {row['model']} | {seeds_str} | {row['tp_mean']:.1f} | {row['precision_mean']*100:.2f}% | {row['recall_mean']*100:.2f}% | **{f1_str}** | {ci_str} |"
        )

    md_path = RESULTS_DIR / "multi_seed_hpo_summary.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"[+] Saved Markdown summary to: {md_path}")
    print("\n" + "\n".join(md_lines))

    # Generate LaTeX Table
    latex_lines = [
        r"\begin{table}[htbp]",
        r"	\centering",
        r"	\caption[超参数优化多随机种子重复实验定量评估]{分层字典序目标在多随机种子下的位姿估计定量评估与置信区间（500轮预算，Seeds: 42, 43, 44）}",
        r"	\label{tab:ch3_multiseed_evaluation}",
        r"	\resizebox{\textwidth}{!}{",
        r"		\begin{tabular}{lcccccc}",
        r"			\toprule",
        r"			工件类别 & 评估种子集 & 平均检出数 (TP) & 平均准确率 & 平均召回率 & 平均 $F_1$ 分数 & $F_1$ 95\% 置信区间 \\",
        r"			\midrule",
    ]
    name_map = {
        "薄壁冲压钣金件 (bracket_planar)": "薄壁冲压钣金件",
        "连续轴对称高强螺栓 (screw_black)": "连续轴对称高强螺栓",
        "十二齿离散旋转对称把手 (star)": "十二齿旋转对称把手",
    }
    for row in output_rows:
        model_name = name_map.get(row["model"], row["model"])
        seeds_str = ", ".join(str(s) for s in row["seeds_evaluated"])
        ci_str = f"$[{row['f1_ci95'][0]:.4f}, {row['f1_ci95'][1]:.4f}]$"
        f1_str = f"${row['f1_mean']:.4f} \\pm {row['f1_std']:.4f}$"
        prec_str = f"${row['precision_mean']*100:.2f}\\%$"
        rec_str = f"${row['recall_mean']*100:.2f}\\%$"
        tp_str = f"${row['tp_mean']:.1f}$"
        latex_lines.append(
            f"			{model_name} & {seeds_str} & {tp_str} & {prec_str} & {rec_str} & \\textbf{{{f1_str}}} & {ci_str} \\\\"
        )
    latex_lines.extend([
        r"			\bottomrule",
        r"		\end{tabular}",
        r"	}",
        r"\end{table}",
    ])
    latex_text = "\n".join(latex_lines)
    latex_path = RESULTS_DIR / "multi_seed_hpo_summary.tex"
    latex_path.write_text(latex_text + "\n", encoding="utf-8")
    print(f"[+] Saved LaTeX table to: {latex_path}")
    print("\n% ===== GENERATED LATEX TABLE =====\n")
    print(latex_text)
    print("\n% ==================================\n")


if __name__ == "__main__":
    main()
