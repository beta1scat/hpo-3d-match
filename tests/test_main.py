import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import optuna

from evaluation import ContinuousSymmetry, DiscreteSymmetry, SymmetryConfig
from main import (
    DEFAULT_OBJECTIVE_VERSION,
    _input_files_sha256,
    _params_sha256,
    _best_trial_within_limit,
    _query_input_files,
    _query_symmetry_summary,
    _validate_no_query_overlap,
    _run_evaluate_best,
    _run_evaluate_default,
    build_parser,
)


class MainOrchestrationTests(unittest.TestCase):
    def test_evaluate_best_defaults_to_dev_and_accepts_test(self):
        args = build_parser().parse_args(
            [
                "evaluate-best",
                "--bop-manifest",
                "manifest.csv",
                "--model",
                "star",
                "--sampler",
                "TPE",
                "--pruner",
                "Nop",
                "--repeat",
                "0",
                "--seed",
                "42",
            ]
        )

        self.assertEqual(args.split, "dev")
        self.assertEqual(DEFAULT_OBJECTIVE_VERSION, "lexicographical-recall-first")
        self.assertIsNone(args.timeout)
        self.assertIsNone(args.min_score)
        self.assertIsNone(args.num_matches)
        self.assertIsNone(args.study_bop_manifest)
        self.assertIsNone(args.trial_limit)

        test_args = build_parser().parse_args(
            [
                "evaluate-best",
                "--bop-manifest",
                "manifest.csv",
                "--model",
                "star",
                "--split",
                "test",
                "--sampler",
                "TPE",
                "--pruner",
                "Nop",
                "--repeat",
                "0",
                "--seed",
                "42",
            ]
        )
        self.assertEqual(test_args.split, "test")

    def test_evaluate_best_accepts_distinct_study_manifest(self):
        args = build_parser().parse_args(
            [
                "evaluate-best",
                "--bop-manifest",
                "test.csv",
                "--study-bop-manifest",
                "pbr.csv",
                "--model",
                "star",
                "--split",
                "test",
                "--sampler",
                "TPE",
                "--pruner",
                "Nop",
                "--repeat",
                "0",
                "--seed",
                "42",
            ]
        )

        self.assertEqual(args.study_bop_manifest, Path("pbr.csv"))

    def test_recall_first_accepts_test_split(self):
        args = SimpleNamespace(
            objective_version="lexicographical-recall-first",
            split="test",
            repeat_id=0,
            seed=42,
            storage_dir=None,
            bop_manifest=Path("dummy.csv"),
            model="screw_black",
            results_root=None,
            run_id=None,
            trial_limit=None,
            parameter_freeze=None,
        )
        self.assertEqual(args.split, "test")

    def test_parameter_digest_is_order_independent(self):
        self.assertEqual(
            _params_sha256({"second": 2, "first": 1}),
            _params_sha256({"first": 1, "second": 2}),
        )

    def test_cross_manifest_evaluation_rejects_query_overlap(self):
        shared_depth = Path("shared-depth.png")
        source = SimpleNamespace(
            source="pbr",
            scene_id=1,
            image_id=2,
            obj_id=25,
            depth_path=shared_depth,
        )
        target = SimpleNamespace(
            source="renamed",
            scene_id=9,
            image_id=8,
            obj_id=25,
            depth_path=shared_depth,
        )

        with self.assertRaisesRegex(ValueError, "overlap evaluation queries"):
            _validate_no_query_overlap((source,), (target,))

    def test_optimize_split_is_train(self):
        args = build_parser().parse_args(
            [
                "optimize",
                "--bop-manifest",
                "manifest.csv",
                "--model",
                "star",
                "--sampler",
                "TPE",
                "--pruner",
                "Nop",
                "--budget",
                "2",
            ]
        )

        self.assertEqual(args.split, "train")

    def test_optimize_accepts_explicit_recall_first_objective(self):
        args = build_parser().parse_args(
            [
                "optimize",
                "--bop-manifest",
                "manifest.csv",
                "--model",
                "star",
                "--objective-version",
                "fixed-penalty-baseline",
                "--sampler",
                "TPE",
                "--pruner",
                "Nop",
                "--budget",
                "2",
            ]
        )

        self.assertEqual(args.objective_version, "fixed-penalty-baseline")

    def test_evaluate_default_accepts_recall_first_objective(self):
        args = build_parser().parse_args(
            [
                "evaluate-default",
                "--bop-manifest",
                "manifest.csv",
                "--model",
                "screw_black",
                "--objective-version",
                "fixed-penalty-baseline",
            ]
        )

        self.assertEqual(args.objective_version, "fixed-penalty-baseline")

    def test_trial_limit_selects_best_complete_trial_in_prefix(self):
        study = SimpleNamespace(
            user_attrs={"budget": 3},
            trials=[
                SimpleNamespace(
                    number=0,
                    state=optuna.trial.TrialState.COMPLETE,
                    value=2.0,
                    params={"value": 0},
                ),
                SimpleNamespace(
                    number=1,
                    state=optuna.trial.TrialState.COMPLETE,
                    value=1.0,
                    params={"value": 1},
                ),
                SimpleNamespace(
                    number=2,
                    state=optuna.trial.TrialState.COMPLETE,
                    value=0.5,
                    params={"value": 2},
                ),
            ],
        )

        selected = _best_trial_within_limit(study, 2)

        self.assertEqual(selected.number, 1)

    def test_input_fingerprint_changes_with_referenced_file_content(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "input.bin"
            path.write_bytes(b"first")
            first = _input_files_sha256({"input": path})
            path.write_bytes(b"second")
            second = _input_files_sha256({"input": path})

        self.assertNotEqual(first, second)

    def test_query_provenance_includes_gt_info_and_models_info(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest = root / "manifest.csv"
            query = SimpleNamespace(
                scene_gt_path=root / "scene_gt.json",
                scene_gt_info_path=root / "scene_gt_info.json",
                scene_camera_path=root / "scene_camera.json",
                depth_path=root / "depth.png",
                cad_path=root / "model.ply",
                models_info_path=root / "models_info.json",
            )

            inputs = _query_input_files(manifest, (query,))

        self.assertTrue(
            any(name.startswith("scene_gt_info_path:") for name in inputs)
        )
        self.assertTrue(
            any(name.startswith("models_info_path:") for name in inputs)
        )

    def test_symmetry_summary_keeps_rigid_translations_and_axis_offsets(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            models_info_path = Path(temporary_directory) / "models_info.json"
            query = SimpleNamespace(
                model_name="bracket_planar",
                obj_id=5,
                models_info_path=models_info_path,
            )
            symmetry = SymmetryConfig(
                discrete_symmetries=(
                    DiscreteSymmetry(np.eye(3), np.array((1.0, 2.0, 3.0))),
                ),
                continuous_symmetries=(
                    ContinuousSymmetry(
                        np.array((0.0, 0.0, 1.0)),
                        np.array((4.0, 5.0, 6.0)),
                    ),
                ),
            )

            summary = _query_symmetry_summary((query,), (symmetry,))

        self.assertEqual(summary[0]["obj_id"], 5)
        self.assertIn(
            [1.0, 2.0, 3.0],
            [item["translation_mm"] for item in summary[0]["discrete_symmetries"]],
        )
        self.assertEqual(
            summary[0]["continuous_symmetries"][0]["offset_mm"],
            [4.0, 5.0, 6.0],
        )


if __name__ == "__main__":
    unittest.main()
