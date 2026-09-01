import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from main import _load_parameter_freeze, build_parser
from selection import (
    REVISED_OBJECTIVE,
    SCHEMA,
    VERSION,
    build_parameter_freeze,
    build_revised_parameter_freeze,
    canonical_params_sha256,
    select_parameter_freeze,
)


SEEDS = [42, 3407, 8128, 19121, 65537]


class ParameterSelectionTests(unittest.TestCase):
    def _write_candidate(
        self,
        root: Path,
        sampler: str,
        repeat: int,
        seed: int,
        objective: float,
        recall: float = 0.5,
        runtime_ms: float = 10.0,
        objective_version: str = "fixed-penalty-baseline",
    ) -> Path:
        run_id = f"dev-{sampler}-{repeat}"
        run_dir = root / run_id
        run_dir.mkdir()
        summaries = run_dir / "scene_summaries.jsonl"
        tp = int(recall * 10)
        records = []
        for query_index in range(10):
            records.append(
                json.dumps({
                    "run_id": run_id,
                    "dataset": "bop_itodd",
                    "split": "dev",
                    "object_name": "star",
                    "method": (
                        f"best:star_{objective_version}_{sampler}_Nop_"
                        f"repeat{repeat}_seed{seed}"
                    ),
                    "repeat_id": repeat,
                    "seed": seed,
                    "scene_id": f"1:{query_index}",
                    "tp": tp,
                    "fp": 0,
                    "fn": 10 - tp,
                    "runtime_ms": runtime_ms,
                    "status": "COMPLETE",
                })
            )
        summaries.write_text(
            "\n".join(records) + "\n",
            encoding="utf-8",
        )
        params = {"num_samples": seed, "refine_pose": True}
        params_sha256 = canonical_params_sha256(params)
        dev_manifest = root / "dev_manifest.csv"
        if not dev_manifest.exists():
            dev_manifest.write_text("fixed-dev-manifest\n", encoding="utf-8")
        source_study = (
            f"star_{objective_version}_{sampler}_Nop_repeat{repeat}_seed{seed}"
        )
        manifest = {
            "schema_version": 1,
            "run_id": run_id,
            "command": "evaluate-best",
            "status": "COMPLETE",
            "cli_config": {
                "command": "evaluate-best",
                "parameter_freeze": None,
                "model": "star",
                "split": "dev",
                "objective_version": objective_version,
                "sampler": sampler,
                "pruner": "Nop",
                "repeat": repeat,
                "seed": seed,
            },
            "source_study": source_study,
            "source_best_value": objective - 0.1,
            "source_study_split": "train",
            "source_study_budget": 5,
            "source_study_terminal_trials": 5,
            "fixed_params": params,
            "fixed_params_sha256": params_sha256,
            "input_files": [
                {
                    "name": "bop_manifest",
                    "path": str(dev_manifest),
                    "size_bytes": dev_manifest.stat().st_size,
                    "sha256": hashlib.sha256(dev_manifest.read_bytes()).hexdigest(),
                }
            ],
            "result": {
                "run_id": run_id,
                "method": f"best:{source_study}",
                "repeat_id": repeat,
                "seed": seed,
                "scene_count": 10,
                "tp": tp * 10,
                "fp": 0,
                "fn": (10 - tp) * 10,
                "objective": objective,
            },
            "outputs": {"scene_summaries_jsonl": str(summaries)},
        }
        manifest_path = run_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return manifest_path

    def _matrix(self, root: Path) -> list[Path]:
        paths = []
        offsets = {"TPE": 0.0, "NSGAII": 10.0, "Random": 20.0}
        for sampler, offset in offsets.items():
            for repeat, seed in enumerate(SEEDS):
                paths.append(
                    self._write_candidate(
                        root,
                        sampler,
                        repeat,
                        seed,
                        objective=offset + repeat,
                        recall=0.5 + repeat / 10.0,
                        runtime_ms=20.0 - repeat,
                    )
                )
        return paths

    def test_selects_sampler_median_and_representative_candidate(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifests = self._matrix(root)
            output = root / "parameter_freeze.json"

            freeze = select_parameter_freeze(manifests, output)

            self.assertEqual(freeze["schema"], SCHEMA)
            self.assertEqual(freeze["version"], VERSION)
            self.assertEqual(freeze["winning_sampler"], "TPE")
            self.assertEqual(freeze["selected_study_identity"]["seed"], 8128)
            self.assertEqual(len(freeze["candidates"]), 15)
            self.assertEqual(len(freeze["input_manifests"]), 15)
            self.assertEqual(
                freeze["params_sha256"], canonical_params_sha256(freeze["params"])
            )
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), freeze)
            identity = freeze["selected_study_identity"]
            args = SimpleNamespace(
                model="star",
                objective_version="fixed-penalty-baseline",
                sampler=identity["sampler"],
                pruner=identity["pruner"],
                repeat=identity["repeat"],
                seed=identity["seed"],
            )
            study = SimpleNamespace(
                study_name=identity["source_study"],
                best_value=identity["source_best_value"],
                best_params=freeze["params"],
            )
            self.assertEqual(_load_parameter_freeze(output, args, study), freeze)

    def test_rejects_duplicate_source_study(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifests = self._matrix(root)
            first = json.loads(manifests[0].read_text(encoding="utf-8"))
            second = json.loads(manifests[1].read_text(encoding="utf-8"))
            second["source_study"] = first["source_study"]
            manifests[1].write_text(json.dumps(second), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "source_study"):
                build_parameter_freeze(manifests)

    def test_freeze_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifests = self._matrix(root)
            output = root / "parameter_freeze.json"
            output.write_text("existing", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "already exists"):
                select_parameter_freeze(manifests, output)

            self.assertEqual(output.read_text(encoding="utf-8"), "existing")

    def test_exact_ties_use_sampler_name_then_seed(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifests = []
            for sampler in ("TPE", "NSGAII", "Random"):
                for repeat, seed in enumerate(SEEDS):
                    manifests.append(
                        self._write_candidate(
                            root,
                            sampler,
                            repeat,
                            seed,
                            objective=1.0,
                            recall=0.5,
                            runtime_ms=10.0,
                        )
                    )

            freeze = build_parameter_freeze(manifests)

            self.assertEqual(freeze["winning_sampler"], "NSGAII")
            self.assertEqual(freeze["selected_study_identity"]["seed"], 42)

    def test_revised_selection_rejects_all_zero_recall_candidates(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifests = []
            for sampler in ("TPE", "NSGAII", "Random"):
                for repeat, seed in enumerate(SEEDS):
                    manifests.append(
                        self._write_candidate(
                            root,
                            sampler,
                            repeat,
                            seed,
                            objective=1.0,
                            recall=0.0,
                            objective_version=REVISED_OBJECTIVE,
                        )
                    )

            with self.assertRaisesRegex(
                ValueError, "NO_ELIGIBLE_POSITIVE_RECALL_CANDIDATE"
            ):
                build_revised_parameter_freeze(manifests)

    def test_revised_selection_rejects_truncated_scene_summaries(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifests = []
            for sampler in ("TPE", "NSGAII", "Random"):
                for repeat, seed in enumerate(SEEDS):
                    manifests.append(
                        self._write_candidate(
                            root,
                            sampler,
                            repeat,
                            seed,
                            objective=1.0,
                            objective_version=REVISED_OBJECTIVE,
                        )
                    )
            first = json.loads(manifests[0].read_text(encoding="utf-8"))
            summaries = Path(first["outputs"]["scene_summaries_jsonl"])
            lines = summaries.read_text(encoding="utf-8").splitlines()
            summaries.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "must contain 10 queries"):
                build_revised_parameter_freeze(manifests)

    def test_revised_selection_rejects_aggregate_metric_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifests = []
            for sampler in ("TPE", "NSGAII", "Random"):
                for repeat, seed in enumerate(SEEDS):
                    manifests.append(
                        self._write_candidate(
                            root,
                            sampler,
                            repeat,
                            seed,
                            objective=1.0,
                            objective_version=REVISED_OBJECTIVE,
                        )
                    )
            first = json.loads(manifests[0].read_text(encoding="utf-8"))
            first["result"]["tp"] += 1
            manifests[0].write_text(json.dumps(first), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "result.tp"):
                build_revised_parameter_freeze(manifests)

    def test_test_split_requires_parameter_freeze_argument(self):
        args = build_parser().parse_args(
            [
                "evaluate-best",
                "--bop-manifest",
                "test.csv",
                "--model",
                "star",
                "--split",
                "test",
                "--sampler",
                "TPE",
                "--pruner",
                "Nop",
                "--repeat",
                "2",
                "--seed",
                "8128",
                "--parameter-freeze",
                "parameter_freeze.json",
            ]
        )

        self.assertEqual(args.parameter_freeze, Path("parameter_freeze.json"))

    def test_freeze_validation_rejects_unverifiable_minimal_file(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            params = {"first": 1, "second": True}
            freeze = {
                "schema": SCHEMA,
                "version": VERSION,
                "model": "star",
                "winning_sampler": "TPE",
                "selected_study_identity": {
                    "source_study": "selected-study",
                    "source_best_value": 1.5,
                    "objective_version": "fixed-penalty-baseline",
                    "sampler": "TPE",
                    "pruner": "Nop",
                    "repeat": 2,
                    "seed": 8128,
                },
                "params": params,
                "params_sha256": canonical_params_sha256(params),
            }
            path = root / "parameter_freeze.json"
            path.write_text(json.dumps(freeze), encoding="utf-8")
            args = SimpleNamespace(
                model="star",
                objective_version="fixed-penalty-baseline",
                sampler="TPE",
                pruner="Nop",
                repeat=2,
                seed=8128,
            )
            study = SimpleNamespace(
                study_name="selected-study", best_value=1.5, best_params=params
            )

            with self.assertRaisesRegex(ValueError, "15 input manifests"):
                _load_parameter_freeze(path, args, study)


if __name__ == "__main__":
    unittest.main()
