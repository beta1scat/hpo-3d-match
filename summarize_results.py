import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Summarize experiment results")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results"),
        help="Root directory containing experiment folders (default: results)",
    )
    args = parser.parse_args()

    results_dir = args.results_root
    summary_files = sorted(results_dir.glob("*/experiment_summary.json"))
    if not summary_files:
        print(f"No experiment_summary.json files found under {results_dir}/")
        return

    print("=" * 155)
    print(f"{'Experiment Directory / Model':<36} | {'Sampler':<8} | {'Pruner':<8} | {'Objective':<10} | {'Budget':<6} | {'Base F1':<8} | {'Final F1':<8} | {'TP/FP/FN':<10} | {'Final Loss':<10} | {'Peak Dev Result'}")
    print("=" * 155)
    for p in summary_files:
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            base_f1 = d.get("default_baseline", {}).get("f1", 0.0)
            checkpoints = d.get("checkpoints", [])
            final_ckpt = checkpoints[-1] if checkpoints else {}
            peak_ckpt = max(checkpoints, key=lambda c: c.get("f1", 0.0)) if checkpoints else {}
            
            final_f1 = final_ckpt.get("f1", 0.0)
            final_loss = final_ckpt.get("objective", 0.0)
            final_counts = f"{final_ckpt.get('tp', 0)}/{final_ckpt.get('fp', 0)}/{final_ckpt.get('fn', 0)}"
            
            peak_f1 = peak_ckpt.get("f1", 0.0)
            peak_tp = peak_ckpt.get("tp", 0)
            peak_loss = peak_ckpt.get("objective", 0.0)
            peak_info = f"TP={peak_tp} (Loss={peak_loss:.0f})"

            exp_name = d.get("experiment_name", p.parent.name)
            sampler = d.get("sampler", "")
            pruner = d.get("pruner", "")
            obj_ver = d.get("objective_version", "")
            obj_tag = "lexrecall" if ("recall" in obj_ver or "lex" in obj_ver) else "fixedpen"
            budget = d.get("budget", len(checkpoints))
            print(f"{exp_name:<36} | {sampler:<8} | {pruner:<8} | {obj_tag:<10} | {budget:<6} | {base_f1:<8.4f} | {final_f1:<8.4f} | {final_counts:<10} | {final_loss:<10.2f} | {peak_info}")
        except Exception as e:
            print(f"Error reading {p}: {e}")
    print("=" * 145)


if __name__ == "__main__":
    main()
