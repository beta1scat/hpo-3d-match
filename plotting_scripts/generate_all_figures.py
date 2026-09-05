"""Generate all Chapter 3 figures for the PhD thesis.

Imports and executes each plotting script sequentially:
  - Figure 3-3: Optimization convergence curves (plot_fig3_convergence.py)
  - Figure 3-4: Objective formulation ablation (plot_fig4_objective_ablation.py)
  - Figure 3-5: Median Pruner acceleration efficiency (plot_fig5_pruning_analysis.py)
  - Figure 3-6: Hyperparameter importance via fANOVA (plot_fig6_param_importance.py)
  - Figure 3-7: Sim2Real PBR vs Real Oracle comparison (plot_fig7_sim_vs_oracle.py)
"""

import sys
from pathlib import Path

# Add current folder to path
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

import plot_fig3_convergence
import plot_fig4_objective_ablation
import plot_fig5_pruning_analysis
import plot_fig6_param_importance
import plot_fig7_sim_vs_oracle


def main():
    print("==================================================")
    print("Generating Chapter 3 Figures for PhD Thesis...")
    print("==================================================")

    print("\n[1/5] Plotting Figure 3-3: Convergence Curves...")
    plot_fig3_convergence.main()

    print("\n[2/5] Plotting Figure 3-4: Objective Formulation Ablation...")
    plot_fig4_objective_ablation.main()

    print("\n[3/5] Plotting Figure 3-5: Pruning Acceleration Efficiency...")
    plot_fig5_pruning_analysis.main()

    print("\n[4/5] Plotting Figure 3-6: Hyperparameter Importance (fANOVA)...")
    plot_fig6_param_importance.main()

    print("\n[5/5] Plotting Figure 3-7: Sim2Real PBR vs Real Oracle...")
    plot_fig7_sim_vs_oracle.main()

    print("\n==================================================")
    print("All Chapter 3 figures successfully generated!")
    print("==================================================")


if __name__ == "__main__":
    main()
