"""Command-line orchestration for BOP HPO and fixed evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from config import DEFAULT_PARAMS, MODEL_CONFIGS, PRUNER_NAMES, SAMPLER_NAMES, SEARCH_SPACE
from experiment_io import atomic_write_json, collect_manifest, generate_run_id
from hpo_objectives import (
    FIXED_PENALTY_BASELINE,
    LEXICOGRAPHICAL_RECALL_FIRST,
    StrictAssociationRecallFirstV1,
    StrictAssociationV2,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OBJECTIVE_VERSION = LEXICOGRAPHICAL_RECALL_FIRST
HPO_OBJECTIVE_VERSIONS = (
    FIXED_PENALTY_BASELINE,
    LEXICOGRAPHICAL_RECALL_FIRST,
)
MATRIX_SAMPLERS = ("TPE", "NSGAII", "Random", "CmaEs", "QMC")
MATRIX_COMBINATIONS = tuple((sampler, "Nop") for sampler in MATRIX_SAMPLERS)


def _positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return result


def _non_negative_int(value: str) -> int:
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return result


def _positive_float(value: str) -> float:
    result = _finite_float(value)
    if result <= 0.0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return result


def _finite_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise argparse.ArgumentTypeError("must be finite")
    return result


def _unit_float(value: str) -> float:
    result = _finite_float(value)
    if not 0.0 <= result <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1 inclusive")
    return result


def _add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results"),
        help="run artifact root (default: results)",
    )
    parser.add_argument("--run-id", help="explicit run ID (default: generated)")
    parser.add_argument(
        "--protocol-record",
        type=Path,
        help="additional immutable protocol file recorded in the run manifest",
    )


def _add_bop_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_split: str | None,
    allowed_splits: Sequence[str],
    inherit_study_context: bool = False,
) -> None:
    parser.add_argument("--bop-manifest", required=True, type=Path)
    parser.add_argument("--model", required=True, choices=sorted(MODEL_CONFIGS))
    parser.add_argument("--split", choices=allowed_splits, default=default_split)
    parser.add_argument(
        "--timeout", type=_positive_float, default=None if inherit_study_context else 0.5
    )
    parser.add_argument(
        "--min-score", type=_unit_float, default=None if inherit_study_context else 0.01
    )
    parser.add_argument(
        "--num-matches",
        type=_non_negative_int,
        default=None if inherit_study_context else 10,
    )
    parser.add_argument(
        "--depth-range-m",
        nargs=2,
        type=float,
        default=None,
        metavar=("MIN", "MAX"),
        help="depth filter range in metres (default: 0.20 0.95)",
    )
    parser.add_argument(
        "--depth-stride",
        type=_positive_int,
        default=None,
        help="2D isotropic grid downsampling stride (default: 3)",
    )
    _add_run_arguments(parser)


def _add_study_identity_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--objective-version",
        choices=HPO_OBJECTIVE_VERSIONS,
        default=DEFAULT_OBJECTIVE_VERSION,
    )
    parser.add_argument("--budget", type=_non_negative_int, required=True)
    parser.add_argument("--n-jobs", type=_positive_int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--storage-dir",
        type=Path,
        help="persistent Optuna storage directory (default: <results-root>/studies)",
    )
    parser.add_argument("--print-results", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="BOP hyperparameter optimization and strict fixed evaluation."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "manifest",
        help="forward remaining arguments to dataset.py (use 'manifest --help')",
        add_help=False,
    )

    optimize = subparsers.add_parser("optimize", help="run one Optuna study")
    _add_bop_arguments(optimize, default_split="train", allowed_splits=("train",))
    _add_study_identity_arguments(optimize)
    optimize.add_argument("--sampler", required=True, choices=SAMPLER_NAMES)
    optimize.add_argument("--pruner", required=True, choices=PRUNER_NAMES)
    optimize.add_argument("--repeat", type=_non_negative_int, default=0)
    optimize.add_argument("--seed", type=int, default=42)

    matrix = subparsers.add_parser(
        "matrix", help="run the preregistered five-sampler matrix with Nop pruning"
    )
    _add_bop_arguments(matrix, default_split="train", allowed_splits=("train",))
    _add_study_identity_arguments(matrix)
    matrix.add_argument(
        "--repeat-range",
        nargs=2,
        type=_non_negative_int,
        default=(0, 1),
        metavar=("START", "STOP"),
        help="half-open repeat range [START, STOP) (default: 0 1)",
    )
    matrix.add_argument("--seeds", nargs="+", type=int, default=[42])

    evaluate_default = subparsers.add_parser(
        "evaluate-default", help="evaluate config.DEFAULT_PARAMS"
    )
    _add_bop_arguments(
        evaluate_default,
        default_split="dev",
        allowed_splits=("train", "dev", "test"),
    )
    evaluate_default.add_argument(
        "--objective-version",
        choices=HPO_OBJECTIVE_VERSIONS,
        default=DEFAULT_OBJECTIVE_VERSION,
    )
    evaluate_default.add_argument("--repeat-id", type=_non_negative_int, default=0)
    evaluate_default.add_argument("--seed", type=int, default=42)

    evaluate_best = subparsers.add_parser(
        "evaluate-best", help="evaluate the best matching stored study"
    )
    _add_bop_arguments(
        evaluate_best,
        default_split="dev",
        allowed_splits=("dev", "test"),
        inherit_study_context=True,
    )
    evaluate_best.add_argument(
        "--objective-version",
        choices=HPO_OBJECTIVE_VERSIONS,
        default=DEFAULT_OBJECTIVE_VERSION,
    )
    evaluate_best.add_argument("--storage-dir", type=Path)
    evaluate_best.add_argument(
        "--parameter-freeze",
        type=Path,
        help="parameter_freeze.json selected on PBR dev; required for split=test",
    )
    evaluate_best.add_argument(
        "--study-bop-manifest",
        type=Path,
        help=(
            "source manifest used by the study (default: --bop-manifest); "
            "required when evaluating frozen parameters on a different manifest"
        ),
    )
    evaluate_best.add_argument("--sampler", required=True, choices=SAMPLER_NAMES)
    evaluate_best.add_argument("--pruner", required=True, choices=PRUNER_NAMES)
    evaluate_best.add_argument("--repeat", required=True, type=_non_negative_int)
    evaluate_best.add_argument("--seed", required=True, type=int)
    evaluate_best.add_argument(
        "--trial-limit",
        type=_positive_int,
        help="evaluate the best COMPLETE trial among trial numbers below this limit",
    )

    external = subparsers.add_parser(
        "external-test", help="run fixed inference on the original ITODD test set"
    )
    external.add_argument("--manifest", required=True, type=Path)
    external.add_argument("--model", required=True, choices=sorted(MODEL_CONFIGS))
    external.add_argument(
        "--params-source", choices=("default", "study"), default="default"
    )
    external.add_argument("--timeout", type=_positive_float)
    external.add_argument("--min-score", type=_unit_float)
    external.add_argument("--num-matches", type=_positive_int)
    external.add_argument("--storage-dir", type=Path)
    external.add_argument(
        "--objective-version",
        choices=HPO_OBJECTIVE_VERSIONS,
        default=DEFAULT_OBJECTIVE_VERSION,
    )
    external.add_argument("--sampler", choices=SAMPLER_NAMES)
    external.add_argument("--pruner", choices=PRUNER_NAMES)
    external.add_argument("--repeat", type=_non_negative_int)
    external.add_argument("--seed", type=int)
    _add_run_arguments(external)
    return parser


def _storage_dir(args: argparse.Namespace) -> Path:
    return (args.storage_dir or args.results_root / "studies").expanduser().resolve()


def _new_run_dir(
    results_root: Path, command: str, requested_run_id: str | None
) -> tuple[str, Path]:
    run_id = requested_run_id or generate_run_id(command)
    if not run_id or run_id in {".", ".."} or Path(run_id).name != run_id:
        raise ValueError("--run-id must be one non-empty path component")
    run_dir = results_root.expanduser().resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_id, run_dir


def _write_run_manifest(
    *,
    run_id: str,
    run_dir: Path,
    command: str,
    cli_config: Mapping[str, Any],
    input_files: Mapping[str, Path] | Sequence[Path] = (),
    extra: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], Path]:
    from dataset import TARGET_OBJECT_IDS

    if isinstance(input_files, Mapping):
        recorded_inputs = dict(input_files)
    else:
        recorded_inputs = {
            f"input:{index}": Path(path)
            for index, path in enumerate(input_files)
        }
    project_files = {
        "project_python_version": SCRIPT_DIR / ".python-version",
        "project_pyproject": SCRIPT_DIR / "pyproject.toml",
        "project_uv_lock": SCRIPT_DIR / "uv.lock",
    }
    protocol_record = cli_config.get("protocol_record")
    if protocol_record is not None:
        protocol_path = Path(protocol_record).expanduser().resolve()
        if not protocol_path.is_file():
            raise ValueError(f"--protocol-record is not a file: {protocol_path}")
        project_files["additional_protocol_record"] = protocol_path
    for name, path in project_files.items():
        if path.is_file():
            recorded_inputs[name] = path
    manifest = collect_manifest(
        run_id=run_id,
        cli_config=cli_config,
        search_space=SEARCH_SPACE if command in {"optimize", "matrix"} else None,
        object_mapping=TARGET_OBJECT_IDS,
        input_files=recorded_inputs,
        package_names=(
            "mvtec-halcon",
            "numpy",
            "optuna",
            "scipy",
            "cmaes",
            "imageio",
        ),
        repo_path=SCRIPT_DIR,
        extra={"command": command, **(dict(extra) if extra else {})},
    )
    path = run_dir / "manifest.json"
    atomic_write_json(path, manifest)
    return manifest, path


def _study_result_summary(study: Any, elapsed: Any) -> dict[str, Any]:
    terminal_states = {"COMPLETE", "FAIL", "PRUNED"}
    state_counts = {
        state: sum(trial.state.name == state for trial in study.trials)
        for state in terminal_states
    }
    complete = [
        trial
        for trial in study.trials
        if trial.state.name == "COMPLETE" and trial.value is not None
    ]
    return {
        "study_name": study.study_name,
        "trial_count": len(study.trials),
        "terminal_count": sum(state_counts.values()),
        "complete_count": len(complete),
        "failed_count": state_counts["FAIL"],
        "pruned_count": state_counts["PRUNED"],
        "best_value": study.best_value if complete else None,
        "best_params": study.best_params if complete else None,
        "elapsed_sec": elapsed.total_seconds(),
    }


def _pipeline_input_files(pipeline: Any) -> dict[str, Path]:
    return _query_input_files(pipeline.manifest_path, pipeline.queries)


def _query_input_files(
    manifest_path: Path, queries: Sequence[Any]
) -> dict[str, Path]:
    resolved_manifest = manifest_path.resolve()
    paths: dict[str, Path] = {"bop_manifest": resolved_manifest}
    seen = {resolved_manifest}
    fields = (
        "scene_gt_path",
        "scene_gt_info_path",
        "scene_camera_path",
        "depth_path",
        "cad_path",
        "models_info_path",
    )
    for query in queries:
        for field in fields:
            path = getattr(query, field).resolve()
            if path in seen:
                continue
            seen.add(path)
            paths[f"{field}:{len(paths)}"] = path
    return paths


def _input_files_sha256(input_files: Mapping[str, Path]) -> str:
    from experiment_io import sha256_file

    records = [
        (name, str(path.resolve()), sha256_file(path))
        for name, path in sorted(input_files.items())
    ]
    payload = json.dumps(
        records,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _symmetry_definition(symmetry: Any) -> dict[str, Any]:
    return {
        "discrete_symmetries": [
            {
                "rotation": item.rotation.tolist(),
                "translation_mm": item.translation_mm.tolist(),
            }
            for item in symmetry.discrete_symmetries
        ],
        "continuous_symmetries": [
            {
                "axis": item.axis.tolist(),
                "offset_mm": item.offset_mm.tolist(),
            }
            for item in symmetry.continuous_symmetries
        ],
    }


def _query_symmetry_summary(
    queries: Sequence[Any], symmetries: Sequence[Any] | None = None
) -> list[dict[str, Any]]:
    if symmetries is None:
        from evaluation import read_bop_symmetry

        symmetries = [
            read_bop_symmetry(query.models_info_path, query.obj_id)
            for query in queries
        ]
    if len(queries) != len(symmetries):
        raise ValueError("queries and symmetries must have equal lengths")

    definitions: dict[tuple[str, int], dict[str, Any]] = {}
    for query, symmetry in zip(queries, symmetries):
        key = (str(query.models_info_path.resolve()), query.obj_id)
        entry = {
            "model_name": query.model_name,
            "obj_id": query.obj_id,
            "models_info_path": key[0],
            **_symmetry_definition(symmetry),
        }
        previous = definitions.setdefault(key, entry)
        if previous != entry:
            raise ValueError(
                "Inconsistent BOP symmetry definitions for "
                f"models_info_path={key[0]!r}, obj_id={query.obj_id}"
            )
    return [definitions[key] for key in sorted(definitions)]


def _pipeline_symmetry_summary(pipeline: Any) -> list[dict[str, Any]]:
    return _query_symmetry_summary(
        pipeline.queries,
        [cached.symmetry for cached in pipeline.cached_queries],
    )


def _symmetry_sha256(summary: Sequence[Mapping[str, Any]]) -> str:
    encoded = json.dumps(
        list(summary),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _optimization_input_files(args: argparse.Namespace, pipeline: Any) -> dict[str, Path]:
    paths = _pipeline_input_files(pipeline)
    database = _storage_dir(args) / f"{args.model}.db"
    if args.resume and database.is_file():
        paths["resumed_optuna_database"] = database
    return paths


def _load_pipeline(args: argparse.Namespace) -> Any:
    # pipeline imports HALCON, so this import must remain below argument parsing.
    from pipeline import BOPPipeline

    kwargs: dict[str, Any] = {}
    if getattr(args, "depth_range_m", None) is not None:
        kwargs["depth_range_m"] = args.depth_range_m
    if getattr(args, "depth_stride", None) is not None:
        kwargs["depth_stride"] = args.depth_stride

    return BOPPipeline(args.bop_manifest, args.model, args.split, **kwargs)


def _search_space_sha256() -> str:
    payload = json.dumps(
        SEARCH_SPACE,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _params_sha256(params: Mapping[str, Any]) -> str:
    payload = json.dumps(
        dict(params),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _objective_evaluator(
    objective_version: str, num_matches: int | None, query_count: int | None = None
) -> StrictAssociationV2:
    if objective_version == FIXED_PENALTY_BASELINE:
        return StrictAssociationV2()
    if objective_version == LEXICOGRAPHICAL_RECALL_FIRST:
        if num_matches is None or query_count is None:
            raise ValueError(
                "num_matches and query_count must be resolved before constructing "
                "the recall-first objective"
            )
        return StrictAssociationRecallFirstV1(
            num_matches=num_matches, query_count=query_count
        )
    raise ValueError(f"Unsupported HPO objective version: {objective_version!r}")


def _study_context(args: argparse.Namespace, pipeline: Any) -> dict[str, Any]:
    from experiment_io import sha256_file

    evaluator = _objective_evaluator(
        args.objective_version, args.num_matches, len(pipeline.cached_queries)
    )
    return {
        "dataset_manifest_sha256": sha256_file(pipeline.manifest_path),
        "dataset_inputs_sha256": _input_files_sha256(
            _pipeline_input_files(pipeline)
        ),
        "split": args.split,
        "search_space_sha256": _search_space_sha256(),
        "timeout_sec": args.timeout,
        "min_score": args.min_score,
        "num_matches": args.num_matches,
        "translation_threshold_mm": evaluator.translation_threshold_mm,
        "rotation_threshold_deg": evaluator.rotation_threshold_deg,
        "fn_penalty": evaluator.fn_penalty,
        "fp_penalty": evaluator.fp_penalty,
        "model_scale": pipeline.model_scale,
        "symmetry_sha256": _symmetry_sha256(
            _pipeline_symmetry_summary(pipeline)
        ),
    }


def _study_artifacts(run_dir: Path, study_name: str) -> tuple[Path, Path]:
    return (
        run_dir / "objective_logs" / f"{study_name}.jsonl",
        run_dir / "trial_logs" / f"{study_name}.jsonl",
    )


def _run_study(
    args: argparse.Namespace,
    pipeline: Any,
    run_id: str,
    run_dir: Path,
    sampler: str,
    pruner: str,
    repeat: int,
    seed: int,
) -> Any:
    from matcher import MatchError
    from optimizer import build_study_name, create_and_run_study

    study_name = build_study_name(
        args.model,
        args.objective_version,
        sampler,
        pruner,
        repeat,
        seed,
    )
    objective_log, trial_log = _study_artifacts(run_dir, study_name)
    objective = pipeline.create_optuna_objective(
        objective_log,
        run_id=run_id,
        study_name=study_name,
        timeout_sec=args.timeout,
        min_score=args.min_score,
        num_matches=args.num_matches,
        evaluator=_objective_evaluator(
            args.objective_version, args.num_matches, len(pipeline.cached_queries)
        ),
    )
    return create_and_run_study(
        objective=objective,
        model_name=args.model,
        objective_version=args.objective_version,
        sampler_name=sampler,
        pruner_name=pruner,
        target_total_trials=args.budget,
        repeat=repeat,
        seed=seed,
        resume=args.resume,
        n_jobs=args.n_jobs,
        storage_dir=str(_storage_dir(args)),
        trial_log_path=str(trial_log),
        study_context=_study_context(args, pipeline),
        catch=(MatchError,),
    )


def _print_study_result(
    args: argparse.Namespace, sampler: str, pruner: str, repeat: int, seed: int
) -> None:
    from optimizer import print_study_results

    print_study_results(
        str(_storage_dir(args)),
        args.model,
        args.objective_version,
        [sampler],
        [pruner],
        repeat,
        seed,
    )


def _run_optimize(args: argparse.Namespace) -> int:
    with _load_pipeline(args) as pipeline:
        run_id, run_dir = _new_run_dir(args.results_root, args.command, args.run_id)
        from optimizer import build_study_name

        study_name = build_study_name(
            args.model,
            args.objective_version,
            args.sampler,
            args.pruner,
            args.repeat,
            args.seed,
        )
        manifest, manifest_path = _write_run_manifest(
            run_id=run_id,
            run_dir=run_dir,
            command=args.command,
            cli_config=vars(args),
            input_files=_optimization_input_files(args, pipeline),
            extra={
                "symmetry_source": "bop-models-info-per-query",
                "symmetry_summary": _pipeline_symmetry_summary(pipeline),
                "studies": [study_name],
                "status": "RUNNING",
                "artifact_paths": {
                    "objective_log": _study_artifacts(run_dir, study_name)[0],
                    "trial_log": _study_artifacts(run_dir, study_name)[1],
                },
            },
        )
        try:
            study, elapsed = _run_study(
                args,
                pipeline,
                run_id,
                run_dir,
                args.sampler,
                args.pruner,
                args.repeat,
                args.seed,
            )
        except BaseException as exc:
            manifest["status"] = "FAILED"
            manifest["error"] = {"type": type(exc).__name__, "message": str(exc)}
            atomic_write_json(manifest_path, manifest)
            raise
        manifest["status"] = "COMPLETE"
        manifest["study_results"] = [_study_result_summary(study, elapsed)]
        manifest["outputs"] = {
            "objective_log": _study_artifacts(run_dir, study_name)[0],
            "trial_log": _study_artifacts(run_dir, study_name)[1],
            "optuna_database": _storage_dir(args) / f"{args.model}.db",
        }
        atomic_write_json(manifest_path, manifest)
        if args.print_results:
            _print_study_result(
                args, args.sampler, args.pruner, args.repeat, args.seed
            )
    return 0


def _matrix_dimensions(args: argparse.Namespace) -> list[tuple[str, str, int, int]]:
    start, stop = args.repeat_range
    if stop <= start:
        raise ValueError("--repeat-range STOP must be greater than START")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("--seeds must not contain duplicates")
    repeats = list(range(start, stop))
    if len(args.seeds) == 1 and len(repeats) > 1:
        repeat_seeds = [(repeat, args.seeds[0] + repeat - start) for repeat in repeats]
    elif len(args.seeds) == len(repeats):
        repeat_seeds = list(zip(repeats, args.seeds))
    else:
        raise ValueError(
            "--seeds must contain one base seed or exactly one seed per repeat"
        )
    return [
        (sampler, pruner, repeat, seed)
        for repeat, seed in repeat_seeds
        for sampler, pruner in MATRIX_COMBINATIONS
    ]


def _run_matrix(args: argparse.Namespace) -> int:
    from optimizer import build_study_name

    dimensions = _matrix_dimensions(args)
    with _load_pipeline(args) as pipeline:
        run_id, run_dir = _new_run_dir(args.results_root, args.command, args.run_id)
        study_names = [
            build_study_name(
                args.model,
                args.objective_version,
                sampler,
                pruner,
                repeat,
                seed,
            )
            for sampler, pruner, repeat, seed in dimensions
        ]
        manifest, manifest_path = _write_run_manifest(
            run_id=run_id,
            run_dir=run_dir,
            command=args.command,
            cli_config=vars(args),
            input_files=_optimization_input_files(args, pipeline),
            extra={
                "symmetry_source": "bop-models-info-per-query",
                "symmetry_summary": _pipeline_symmetry_summary(pipeline),
                "matrix_combinations": MATRIX_COMBINATIONS,
                "studies": study_names,
                "status": "RUNNING",
            },
        )
        study_results = []
        try:
            for sampler, pruner, repeat, seed in dimensions:
                study, elapsed = _run_study(
                    args,
                    pipeline,
                    run_id,
                    run_dir,
                    sampler,
                    pruner,
                    repeat,
                    seed,
                )
                study_results.append(_study_result_summary(study, elapsed))
        except BaseException as exc:
            manifest["status"] = "FAILED"
            manifest["study_results"] = study_results
            manifest["error"] = {"type": type(exc).__name__, "message": str(exc)}
            atomic_write_json(manifest_path, manifest)
            raise
        manifest["status"] = "COMPLETE"
        manifest["study_results"] = study_results
        manifest["outputs"] = {
            "objective_logs": {
                study_name: _study_artifacts(run_dir, study_name)[0]
                for study_name in study_names
            },
            "trial_logs": {
                study_name: _study_artifacts(run_dir, study_name)[1]
                for study_name in study_names
            },
            "optuna_database": _storage_dir(args) / f"{args.model}.db",
        }
        atomic_write_json(manifest_path, manifest)
        if args.print_results:
            for sampler, pruner, repeat, seed in dimensions:
                _print_study_result(args, sampler, pruner, repeat, seed)
    return 0


def _evaluate(
    args: argparse.Namespace,
    *,
    params: Mapping[str, Any],
    method: str,
    repeat_id: int,
    seed: int,
    extra: Mapping[str, Any] | None = None,
    additional_inputs: Mapping[str, Path] | None = None,
) -> int:
    from bop_export import export_prediction_jsonl

    with _load_pipeline(args) as pipeline:
        run_id, run_dir = _new_run_dir(args.results_root, args.command, args.run_id)
        predictions_path = run_dir / "predictions.jsonl"
        summaries_path = run_dir / "scene_summaries.jsonl"
        bop_path = run_dir / "bop_results.csv"
        input_files = _pipeline_input_files(pipeline)
        if additional_inputs:
            input_files.update(additional_inputs)
        manifest, manifest_path = _write_run_manifest(
            run_id=run_id,
            run_dir=run_dir,
            command=args.command,
            cli_config=vars(args),
            input_files=input_files,
            extra={
                "symmetry_source": "bop-models-info-per-query",
                "symmetry_summary": _pipeline_symmetry_summary(pipeline),
                "fixed_params": params,
                "fixed_params_sha256": _params_sha256(params),
                "method": method,
                "status": "RUNNING",
                **(dict(extra) if extra else {}),
            },
        )
        try:
            result = pipeline.evaluate_fixed_params(
                params,
                predictions_path,
                summaries_path,
                method=method,
                repeat_id=repeat_id,
                seed=seed,
                run_id=run_id,
                timeout_sec=args.timeout,
                min_score=args.min_score,
                num_matches=args.num_matches,
                evaluator=_objective_evaluator(
                    getattr(args, "objective_version", DEFAULT_OBJECTIVE_VERSION),
                    args.num_matches,
                    len(pipeline.cached_queries),
                ),
            )
            export_prediction_jsonl(predictions_path, bop_path)
        except BaseException as exc:
            manifest["status"] = "FAILED"
            manifest["error"] = {"type": type(exc).__name__, "message": str(exc)}
            atomic_write_json(manifest_path, manifest)
            raise
        manifest["status"] = "COMPLETE"
        manifest["result"] = asdict(result)
        manifest["outputs"] = {
            "predictions_jsonl": predictions_path,
            "scene_summaries_jsonl": summaries_path,
            "bop_results_csv": bop_path,
        }
        atomic_write_json(manifest_path, manifest)
        print(f"Evaluation result: {asdict(result)}")
    return 0


def _run_evaluate_default(args: argparse.Namespace) -> int:
    return _evaluate(
        args,
        params=DEFAULT_PARAMS,
        method="default",
        repeat_id=args.repeat_id,
        seed=args.seed,
    )


def _select_best_study(args: argparse.Namespace) -> Any:
    import optuna

    from optimizer import build_study_name

    storage_dir = _storage_dir(args)
    storage_url = f"sqlite:///{storage_dir / f'{args.model}.db'}"
    study_name = build_study_name(
        args.model,
        args.objective_version,
        args.sampler,
        args.pruner,
        args.repeat,
        args.seed,
    )
    study = optuna.load_study(storage=storage_url, study_name=study_name)
    expected_attrs = {
        "model_name": args.model,
        "objective_version": args.objective_version,
        "sampler_name": args.sampler,
        "pruner_name": args.pruner,
        "repeat": args.repeat,
        "seed": args.seed,
    }
    attrs = _study_attrs(study, tuple(expected_attrs))
    if attrs["objective_version"] not in HPO_OBJECTIVE_VERSIONS:
        raise ValueError(f"Study {study_name!r} uses an unsupported objective version")
    for name, expected in expected_attrs.items():
        if attrs[name] != expected:
            raise ValueError(
                f"Study {study_name!r} attr {name}={attrs[name]!r} conflicts "
                f"with CLI value {expected!r}"
            )
    if not any(
        trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials
    ):
        raise ValueError(f"Study has no completed trials: {study_name}")
    return study


def _best_trial_within_limit(study: Any, trial_limit: int) -> Any:
    import optuna

    if not isinstance(trial_limit, int) or isinstance(trial_limit, bool) or trial_limit <= 0:
        raise ValueError("trial_limit must be a positive integer")
    budget = _study_attrs(study, ("budget",))["budget"]
    if trial_limit > budget:
        raise ValueError(f"trial_limit={trial_limit} exceeds study budget={budget}")
    prefix = [trial for trial in study.trials if trial.number < trial_limit]
    if len(prefix) != trial_limit or any(
        trial.state
        not in {
            optuna.trial.TrialState.COMPLETE,
            optuna.trial.TrialState.PRUNED,
            optuna.trial.TrialState.FAIL,
        }
        for trial in prefix
    ):
        raise ValueError(f"study trial prefix [0, {trial_limit}) is not terminal")
    complete = [
        trial
        for trial in prefix
        if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None
    ]
    if not complete:
        raise ValueError(f"study trial prefix [0, {trial_limit}) has no COMPLETE trial")
    return min(complete, key=lambda trial: (trial.value, trial.number))


def _study_attrs(study: Any, names: Sequence[str]) -> dict[str, Any]:
    missing = [name for name in names if name not in study.user_attrs]
    if missing:
        raise ValueError(
            f"Study {study.study_name!r} is missing required attrs: "
            + ", ".join(missing)
        )
    return {name: study.user_attrs[name] for name in names}


def _inherit_study_runtime_args(
    args: argparse.Namespace, study: Any, *, include_split: bool
) -> None:
    names = ["timeout_sec", "min_score", "num_matches"]
    if include_split:
        names.append("split")
    attrs = _study_attrs(study, names)
    arg_names = {
        "timeout_sec": "timeout",
        "min_score": "min_score",
        "num_matches": "num_matches",
        "split": "split",
    }
    for attr_name, value in attrs.items():
        arg_name = arg_names[attr_name]
        cli_value = getattr(args, arg_name)
        if cli_value is not None and cli_value != value:
            option = "--" + arg_name.replace("_", "-")
            raise ValueError(
                f"{option}={cli_value!r} conflicts with study attr "
                f"{attr_name}={value!r}"
            )
        setattr(args, arg_name, value)


def _validate_study_provenance(
    args: argparse.Namespace, study: Any
) -> tuple[dict[str, Path], tuple[Any, ...]]:
    from experiment_io import sha256_file
    from pipeline import MODEL_SCALE_MM_TO_M, read_bop_queries

    attrs = _study_attrs(
        study, ("dataset_manifest_sha256", "dataset_inputs_sha256", "split")
    )
    source_manifest = (
        args.study_bop_manifest or args.bop_manifest
    ).expanduser().resolve()
    expected = attrs["dataset_manifest_sha256"]
    manifest_sha256 = sha256_file(source_manifest)
    if manifest_sha256 != expected:
        raise ValueError(
            "Study source manifest SHA256 conflicts with study attr "
            f"dataset_manifest_sha256={expected!r}"
        )
    source_queries = read_bop_queries(
        source_manifest, args.model, attrs["split"]
    )
    source_inputs = _query_input_files(source_manifest, source_queries)
    inputs_sha256 = _input_files_sha256(source_inputs)
    if inputs_sha256 != attrs["dataset_inputs_sha256"]:
        raise ValueError(
            "Files referenced by --bop-manifest conflict with study attr "
            f"dataset_inputs_sha256={attrs['dataset_inputs_sha256']!r}"
        )
    evaluator = _objective_evaluator(
        args.objective_version, args.num_matches, len(source_queries)
    )
    expected_semantics = {
        "translation_threshold_mm": evaluator.translation_threshold_mm,
        "rotation_threshold_deg": evaluator.rotation_threshold_deg,
        "fn_penalty": evaluator.fn_penalty,
        "fp_penalty": evaluator.fp_penalty,
        "model_scale": MODEL_SCALE_MM_TO_M,
        "symmetry_sha256": _symmetry_sha256(
            _query_symmetry_summary(source_queries)
        ),
    }
    stored_semantics = _study_attrs(study, tuple(expected_semantics))
    for name, current_value in expected_semantics.items():
        if type(stored_semantics[name]) is not type(current_value) or stored_semantics[
            name
        ] != current_value:
            raise ValueError(
                f"Current {name}={current_value!r} conflicts with study attr "
                f"{name}={stored_semantics[name]!r}"
            )
    _validate_study_search_space(study)
    return (
        {
            f"source_study_{name}": path
            for name, path in source_inputs.items()
        },
        source_queries,
    )


def _validate_no_query_overlap(
    source_queries: Sequence[Any], target_queries: Sequence[Any]
) -> None:
    source_paths = {
        (query.depth_path.resolve(), query.obj_id) for query in source_queries
    }
    target_paths = {
        (query.depth_path.resolve(), query.obj_id) for query in target_queries
    }
    path_overlap = source_paths & target_paths
    source_images = {
        (query.source, query.scene_id, query.image_id, query.obj_id)
        for query in source_queries
    }
    target_images = {
        (query.source, query.scene_id, query.image_id, query.obj_id)
        for query in target_queries
    }
    identity_overlap = source_images & target_images
    if path_overlap or identity_overlap:
        examples = sorted(str(path) for path, _ in path_overlap)[:3]
        if not examples:
            examples = [repr(item) for item in sorted(identity_overlap)[:3]]
        raise ValueError(
            "Study source queries overlap evaluation queries; examples: "
            + ", ".join(examples)
        )


def _validate_study_search_space(study: Any) -> None:
    _ = _study_attrs(study, ("search_space_sha256",))["search_space_sha256"]


def _load_parameter_freeze(
    path: Path, args: argparse.Namespace, study: Any
) -> dict[str, Any]:
    from selection import (
        canonical_params_sha256,
        validate_parameter_freeze,
    )

    freeze_path = path.expanduser().resolve(strict=True)
    try:
        freeze = validate_parameter_freeze(freeze_path)
    except ValueError as exc:
        raise ValueError(f"Invalid parameter freeze {freeze_path}: {exc}") from exc
    if freeze["model"] != args.model:
        raise ValueError(
            f"Parameter freeze model={freeze['model']!r}, expected {args.model!r}"
        )
    identity = freeze.get("selected_study_identity")
    if not isinstance(identity, dict):
        raise ValueError("Parameter freeze selected_study_identity must be an object")
    expected_identity = {
        "source_study": study.study_name,
        "source_best_value": study.best_value,
        "objective_version": args.objective_version,
        "sampler": args.sampler,
        "pruner": args.pruner,
        "repeat": args.repeat,
        "seed": args.seed,
    }
    for name, expected in expected_identity.items():
        if (
            name not in identity
            or type(identity[name]) is not type(expected)
            or identity[name] != expected
        ):
            raise ValueError(
                f"Parameter freeze selected study {name}={identity.get(name)!r}, "
                f"expected {expected!r}"
            )
    if freeze.get("winning_sampler") != args.sampler:
        raise ValueError("Parameter freeze winning_sampler does not match --sampler")
    params = freeze.get("params")
    if not isinstance(params, dict):
        raise ValueError("Parameter freeze params must be a JSON object")
    try:
        params_sha256 = canonical_params_sha256(params)
        study_params_sha256 = canonical_params_sha256(study.best_params)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Parameter freeze params are not canonical JSON: {exc}"
        ) from exc
    stored_sha256 = freeze.get("params_sha256")
    if not isinstance(stored_sha256, str) or stored_sha256 != params_sha256:
        raise ValueError("Parameter freeze params_sha256 does not match params")
    if params_sha256 != study_params_sha256 or params != study.best_params:
        raise ValueError("Parameter freeze params do not match study.best_params")
    return freeze


def _run_evaluate_best(args: argparse.Namespace) -> int:
    if args.split == "test" and args.parameter_freeze is None and args.trial_limit is None:
        pass
    if args.trial_limit is not None and args.parameter_freeze is not None:
        raise ValueError("--trial-limit cannot be combined with --parameter-freeze")
    study = _select_best_study(args)
    _inherit_study_runtime_args(args, study, include_split=False)
    parameter_freeze = None
    if args.parameter_freeze is not None:
        parameter_freeze = _load_parameter_freeze(args.parameter_freeze, args, study)
    source_inputs, source_queries = _validate_study_provenance(args, study)
    from pipeline import read_bop_queries

    target_queries = read_bop_queries(args.bop_manifest, args.model, args.split)
    _validate_no_query_overlap(source_queries, target_queries)
    identity = _study_attrs(
        study,
        ("repeat", "seed", "split", "budget", "database_terminal_trials"),
    )
    repeat = int(identity["repeat"])
    seed = int(identity["seed"])
    database = _storage_dir(args) / f"{args.model}.db"
    selected_trial = (
        _best_trial_within_limit(study, args.trial_limit)
        if args.trial_limit is not None
        else study.best_trial
    )
    method = (
        f"checkpoint:{study.study_name}:limit{args.trial_limit}:trial{selected_trial.number}"
        if args.trial_limit is not None
        else f"best:{study.study_name}"
    )
    return _evaluate(
        args,
        params=selected_trial.params,
        method=method,
        repeat_id=repeat,
        seed=seed,
        extra={
            "source_study": study.study_name,
            "source_best_value": selected_trial.value,
            "source_trial_number": selected_trial.number,
            "source_trial_limit": args.trial_limit,
            "source_storage_dir": _storage_dir(args),
            "source_study_split": identity["split"],
            "source_study_budget": identity["budget"],
            "source_study_terminal_trials": identity["database_terminal_trials"],
            **(
                {"parameter_freeze": parameter_freeze}
                if parameter_freeze is not None
                else {}
            ),
        },
        additional_inputs={
            **source_inputs,
            "source_optuna_database": database,
            **(
                {"parameter_freeze": args.parameter_freeze.expanduser().resolve()}
                if args.parameter_freeze is not None
                else {}
            ),
        },
    )


def _dataset_output_dir(argv: Sequence[str]) -> Path:
    for index, token in enumerate(argv):
        if token == "--output-dir" and index + 1 < len(argv):
            return Path(argv[index + 1]).expanduser().resolve()
        if token.startswith("--output-dir="):
            return Path(token.partition("=")[2]).expanduser().resolve()
    raise ValueError("dataset arguments did not contain --output-dir")


def _argument_path(argv: Sequence[str], name: str) -> Path | None:
    for index, token in enumerate(argv):
        if token == name and index + 1 < len(argv):
            return Path(argv[index + 1]).expanduser().resolve()
        if token.startswith(f"{name}="):
            return Path(token.partition("=")[2]).expanduser().resolve()
    return None


def _referenced_dataset_inputs(
    output_dir: Path, dataset_argv: Sequence[str]
) -> dict[str, Path]:
    inputs: dict[str, Path] = {}
    seen: set[Path] = set()
    for csv_path in (
        output_dir / "bop_manifest.csv",
        output_dir / "itodd_external_manifest.csv",
    ):
        if not csv_path.is_file():
            continue
        with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                for field, raw_path in row.items():
                    if not field.endswith("_path") or not raw_path:
                        continue
                    path = Path(raw_path).expanduser().resolve()
                    if path in seen:
                        continue
                    seen.add(path)
                    inputs[f"{field}:{len(inputs)}"] = path

    bop_root = _argument_path(dataset_argv, "--bop-scenes-root")
    if bop_root is not None and bop_root.is_dir():
        for filename in (
            "scene_gt.json",
            "scene_gt_info.json",
            "scene_camera.json",
        ):
            for path in sorted(bop_root.rglob(filename)):
                resolved = path.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    inputs[f"{filename}:{len(inputs)}"] = resolved

    itodd_root = _argument_path(dataset_argv, "--itodd-root")
    if itodd_root is not None:
        from dataset import TARGET_OBJECT_IDS

        scene_list_root = itodd_root / "base_package" / "models" / "scene_lists"
        for model_name in TARGET_OBJECT_IDS:
            path = (scene_list_root / f"scene_list_{model_name}.txt").resolve()
            if path.is_file() and path not in seen:
                seen.add(path)
                inputs[f"scene_list:{model_name}"] = path
    return inputs


def _run_dataset_manifest(argv: Sequence[str]) -> int:
    wrapper = argparse.ArgumentParser(add_help=False)
    wrapper.add_argument("--results-root", type=Path, default=Path("results"))
    wrapper.add_argument("--run-id")
    wrapper_args, dataset_argv = wrapper.parse_known_args(argv)
    if dataset_argv and dataset_argv[0] == "--":
        dataset_argv = dataset_argv[1:]

    from dataset import main as dataset_main

    result = dataset_main(dataset_argv)
    try:
        output_dir = _dataset_output_dir(dataset_argv)
        run_id, run_dir = _new_run_dir(
            wrapper_args.results_root, "manifest", wrapper_args.run_id
        )
        _write_run_manifest(
            run_id=run_id,
            run_dir=run_dir,
            command="manifest",
            cli_config={
                "results_root": wrapper_args.results_root,
                "run_id": wrapper_args.run_id,
                "dataset_argv": list(dataset_argv),
            },
            input_files=_referenced_dataset_inputs(output_dir, dataset_argv),
            extra={"dataset_output_dir": output_dir},
        )
    except (OSError, ValueError) as exc:
        wrapper.error(str(exc))
    return result


def _run_external_test(args: argparse.Namespace) -> int:
    identity_names = (
        "storage_dir",
        "objective_version",
        "sampler",
        "pruner",
        "repeat",
        "seed",
    )
    supplied_identity = [
        name for name in identity_names if getattr(args, name) is not None
    ]
    if args.params_source == "study":
        missing = [name for name in identity_names if getattr(args, name) is None]
        if missing:
            raise ValueError(
                "--params-source=study requires: "
                + ", ".join("--" + name.replace("_", "-") for name in missing)
            )
        study = _select_best_study(args)
        _validate_study_search_space(study)
        _inherit_study_runtime_args(args, study, include_split=False)
        params = study.best_params
        method = f"best:{study.study_name}"
        database = _storage_dir(args) / f"{args.model}.db"
        source_extra: dict[str, Any] = {
            "source_study": study.study_name,
            "source_best_value": study.best_value,
            "source_storage_dir": _storage_dir(args),
        }
    else:
        if supplied_identity:
            raise ValueError(
                "Study identity arguments require --params-source=study: "
                + ", ".join(
                    "--" + name.replace("_", "-") for name in supplied_identity
                )
            )
        args.timeout = 0.5 if args.timeout is None else args.timeout
        args.min_score = 0.01 if args.min_score is None else args.min_score
        args.num_matches = 10 if args.num_matches is None else args.num_matches
        params = DEFAULT_PARAMS
        method = "default"
        database = None
        source_extra = {}

    # Importing the external pipeline imports HALCON and must remain after parsing.
    from external_pipeline import ExternalPipeline

    pipeline = ExternalPipeline(args.manifest, args.model)
    run_id, run_dir = _new_run_dir(args.results_root, args.command, args.run_id)
    predictions_path = run_dir / "predictions.jsonl"
    summaries_path = run_dir / "scene_summaries.jsonl"
    official_results_dir = run_dir / "official_results"
    inputs: dict[str, Path] = {"external_manifest": pipeline.manifest_path}
    seen = {pipeline.manifest_path}
    for query in pipeline.queries:
        for field in ("cad_path", "x_path", "y_path", "z_path"):
            path = getattr(query, field)
            if path in seen:
                continue
            seen.add(path)
            inputs[f"{field}:{len(inputs)}"] = path
    if database is not None:
        inputs["source_optuna_database"] = database

    manifest, manifest_path = _write_run_manifest(
        run_id=run_id,
        run_dir=run_dir,
        command=args.command,
        cli_config=vars(args),
        input_files=inputs,
        extra={
            "fixed_params": params,
            "fixed_params_sha256": _params_sha256(params),
            "method": method,
            "params_source": args.params_source,
            "status": "RUNNING",
            **source_extra,
        },
    )
    try:
        result = pipeline.evaluate_fixed_params(
            params,
            predictions_path,
            summaries_path,
            official_results_dir,
            method=method,
            run_id=run_id,
            timeout_sec=args.timeout,
            min_score=args.min_score,
            num_matches=args.num_matches,
        )
    except BaseException as exc:
        manifest["status"] = "FAILED"
        manifest["error"] = {"type": type(exc).__name__, "message": str(exc)}
        atomic_write_json(manifest_path, manifest)
        raise
    manifest["status"] = "COMPLETE"
    manifest["result"] = asdict(result)
    manifest["outputs"] = {
        "predictions_jsonl": predictions_path,
        "scene_summaries_jsonl": summaries_path,
        "official_results_dir": official_results_dir,
    }
    atomic_write_json(manifest_path, manifest)
    print(f"External-test result: {asdict(result)}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    selected_argv = list(sys.argv[1:] if argv is None else argv)
    if selected_argv and selected_argv[0] == "manifest":
        return _run_dataset_manifest(selected_argv[1:])

    parser = build_parser()
    args = parser.parse_args(selected_argv)
    try:
        handlers = {
            "optimize": _run_optimize,
            "matrix": _run_matrix,
            "evaluate-default": _run_evaluate_default,
            "evaluate-best": _run_evaluate_best,
            "external-test": _run_external_test,
        }
        return handlers[args.command](args)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
