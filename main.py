"""Main entry point for the parameter optimization framework.

Usage:
    # Run optimization for a single model with all sampler x pruner combinations
    python main.py --model star --n-trials 5000 --n-jobs 32

    # Run with specific sampler and pruner
    python main.py --model screw_black --sampler TPE --pruner Nop --n-trials 5000

    # Print results only
    python main.py --model star --print-results

    # Evaluate specific parameters on scenes
    python main.py --model star --evaluate default
    python main.py --model star --evaluate best

    # Custom dataset path
    python main.py --model star --dataset-path ./data --n-trials 5000
"""

import argparse
import sys
from pathlib import Path

from config import (
    MODEL_CONFIGS,
    SAMPLER_NAMES,
    PRUNER_NAMES,
    DatasetPaths,
    ROIConfig,
    DEFAULT_PARAMS,
)
from optimizer import (
    create_and_run_study,
    run_all_combinations,
    print_study_results,
)


def parse_args():
    default_data_dir = Path(__file__).parent / "data"

    parser = argparse.ArgumentParser(
        description="Parameter Optimization Framework for Pose Estimation (HALCON find_surface_model)"
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=list(MODEL_CONFIGS.keys()),
        help="Object model name: star, screw_black, bracket_planar",
    )
    parser.add_argument(
        "--dataset-path",
        default=str(default_data_dir),
        help="Dataset root. Supports local layout ./data and original ITODD layout.",
    )
    parser.add_argument(
        "--gt-dir",
        default=None,
        help="Directory containing scene_gt_*.json files (default: ./data/)",
    )
    parser.add_argument(
        "--storage-dir",
        default="./results",
        help="Directory for SQLite result databases (default: ./results)",
    )

    # Optimization parameters
    parser.add_argument(
        "--n-trials",
        type=int,
        default=5000,
        help="Number of optimization trials (default: 5000)",
    )
    parser.add_argument(
        "--n-jobs", type=int, default=32, help="Number of parallel jobs (default: 32)"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="Timeout in seconds for each find_surface_model call (default: 5.0)",
    )

    # Sampler/pruner selection
    parser.add_argument(
        "--sampler",
        default=None,
        choices=SAMPLER_NAMES,
        help="Specific sampler to use (default: run all samplers)",
    )
    parser.add_argument(
        "--pruner",
        default=None,
        choices=PRUNER_NAMES,
        help="Specific pruner to use (default: run all pruners)",
    )

    # Actions
    parser.add_argument(
        "--print-results",
        action="store_true",
        help="Print results from stored studies instead of running optimization",
    )
    parser.add_argument(
        "--evaluate",
        default=None,
        choices=["default", "best"],
        help="Evaluate default or best parameters on scenes",
    )

    return parser.parse_args()


def run_evaluate(args, model_cfg, dataset_paths, roi):
    """Evaluate specific parameters (default or best) on all scenes."""
    from objective import (
        load_ground_truth,
        run_surface_matching,
        compute_objective_for_scene,
        compute_roi_transform,
    )
    from scene_loader import load_scene, filter_scene_roi, load_model
    import optuna

    gt_dir = args.gt_dir or str(Path(__file__).parent / "data")
    gt_path = Path(gt_dir) / f"scene_gt_{args.model}.json"
    scene_list, pose_gt_dict = load_ground_truth(str(gt_path), model_cfg)

    model_3d = load_model(dataset_paths.model_path(args.model))
    roi_mat, roi_pose_inv = compute_roi_transform(roi)

    if args.evaluate == "default":
        params = DEFAULT_PARAMS.copy()
        print(f"Evaluating DEFAULT parameters: {params}")
    else:
        # Load best from stored study
        db_path = Path(args.storage_dir) / f"{args.model}.db"
        storage_url = f"sqlite:///{db_path}"
        # Find the study with the best value across all combinations
        best_value = float("inf")
        best_params = None
        best_study_name = None
        sampler_names = [args.sampler] if args.sampler else SAMPLER_NAMES
        pruner_names = [args.pruner] if args.pruner else PRUNER_NAMES
        for s in sampler_names:
            for p in pruner_names:
                study_name = f"{args.model}_{s}_{p}"
                try:
                    study = optuna.load_study(
                        storage=storage_url, study_name=study_name
                    )
                    if study.best_value < best_value:
                        best_value = study.best_value
                        best_params = study.best_params
                        best_study_name = study_name
                except Exception:
                    continue
        if best_params is None:
            print("No stored results found. Run optimization first.")
            sys.exit(1)
        params = best_params
        print(
            f"Evaluating BEST parameters from study '{best_study_name}' (value={best_value}):"
        )
        print(f"  {params}")

    # Run matching with these params
    model_surface = None
    import halcon as ha

    model_surface = ha.create_surface_model(
        model_3d, params["RelSamplingDistance"], [], []
    )

    total_obj = 0.0
    for scene_id in scene_list:
        scene_prefix = dataset_paths.scene_image_prefix(scene_id)
        scene_3d = load_scene(scene_prefix)
        scene_roi = filter_scene_roi(scene_3d, roi_pose_inv, roi)

        try:
            Pose, Score = run_surface_matching(
                model_surface, scene_roi, params, args.timeout
            )
            n_matches = len(Score) if Score else 0
            scene_obj = compute_objective_for_scene(
                Pose, Score, pose_gt_dict[scene_id], roi_mat, model_cfg
            )
            total_obj += scene_obj
            print(
                f"  Scene {scene_id}: matches={n_matches}, "
                f"gt_count={len(pose_gt_dict[scene_id])}, obj={scene_obj:.4f}"
            )
        except Exception as e:
            print(f"  Scene {scene_id}: FAILED ({e})")
            total_obj += 2000

    print(f"\nTotal objective: {total_obj:.4f}")


def main():
    from scene_loader import load_model
    from objective import load_ground_truth, create_objective

    args = parse_args()

    model_cfg = MODEL_CONFIGS[args.model]
    dataset_paths = DatasetPaths(base_path=args.dataset_path)
    roi = ROIConfig()

    # Print results mode
    if args.print_results:
        sampler_names = [args.sampler] if args.sampler else SAMPLER_NAMES
        pruner_names = [args.pruner] if args.pruner else PRUNER_NAMES
        print_study_results(args.storage_dir, args.model, sampler_names, pruner_names)
        return

    # Evaluate mode
    if args.evaluate:
        run_evaluate(args, model_cfg, dataset_paths, roi)
        return

    # Optimization mode
    gt_dir = args.gt_dir or str(Path(__file__).parent / "data")
    gt_path = Path(gt_dir) / f"scene_gt_{args.model}.json"

    if not gt_path.exists():
        print(f"Ground truth file not found: {gt_path}")
        sys.exit(1)

    print(f"Model: {args.model}")
    print(
        f"  position_bound={model_cfg.position_bound}, "
        f"rotation_bound={model_cfg.rotation_bound}, "
        f"check_axis={model_cfg.check_axis}, "
        f"symmetry_angle={model_cfg.symmetry_angle}"
    )
    print(f"Dataset: {args.dataset_path}")
    print(f"GT file: {gt_path}")

    scene_list, pose_gt_dict = load_ground_truth(str(gt_path), model_cfg)
    print(f"Loaded {len(scene_list)} scenes with ground truth")

    model_3d = load_model(dataset_paths.model_path(args.model))
    print("Model loaded with surface normals")

    objective = create_objective(
        model_3d=model_3d,
        model_cfg=model_cfg,
        scene_list=scene_list,
        pose_gt_dict=pose_gt_dict,
        dataset_paths=dataset_paths,
        roi=roi,
        timeout_sec=args.timeout,
    )

    sampler_names = (
        [args.sampler] if args.sampler else ["TPE", "CmaEs", "NSGAII", "QMC"]
    )
    pruner_names = [args.pruner] if args.pruner else ["Median", "Nop", "Hyperband"]

    if args.sampler and args.pruner:
        # Single combination
        study, elapsed = create_and_run_study(
            objective=objective,
            model_name=args.model,
            sampler_name=args.sampler,
            pruner_name=args.pruner,
            n_trials=args.n_trials,
            n_jobs=args.n_jobs,
            storage_dir=args.storage_dir,
        )
    else:
        # All combinations (also includes Random+Nop)
        results = run_all_combinations(
            objective=objective,
            model_name=args.model,
            sampler_names=sampler_names,
            pruner_names=pruner_names,
            n_trials=args.n_trials,
            n_jobs=args.n_jobs,
            storage_dir=args.storage_dir,
        )

        # Also run Random+Nop if not already included
        if "Random" not in sampler_names:
            print(f"\n{'=' * 60}")
            print("Running Random+Nop baseline...")
            study, elapsed = create_and_run_study(
                objective=objective,
                model_name=args.model,
                sampler_name="Random",
                pruner_name="Nop",
                n_trials=args.n_trials,
                n_jobs=args.n_jobs,
                storage_dir=args.storage_dir,
            )

    # Print all results
    print(f"\n{'=' * 60}")
    print("FINAL RESULTS")
    print_study_results(
        args.storage_dir, args.model, sampler_names + ["Random"], pruner_names + ["Nop"]
    )


if __name__ == "__main__":
    main()
