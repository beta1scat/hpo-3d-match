"""Preregistered dev-set sampler selection and parameter freezing."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import secrets
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiment_io import atomic_write_json, sha256_file


SCHEMA = "hpo-3d-match/parameter-freeze"
VERSION = 1
OBJECTIVE = "fixed-penalty-baseline"
REVISED_SCHEMA = "hpo-3d-match/pbr-revised-parameter-freeze"
REVISED_VERSION = 1
REVISED_OBJECTIVE = "lexicographical-recall-first"
SAMPLERS = ("TPE", "NSGAII", "Random")
SEEDS = (42, 3407, 8128, 19121, 65537)
SEED_TO_REPEAT = {seed: repeat for repeat, seed in enumerate(SEEDS)}
TOLERANCE = 1e-12


def canonical_params_sha256(params: Mapping[str, Any]) -> str:
    payload = json.dumps(
        dict(params),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"non-standard JSON constant {value!r}")


def load_strict_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as stream:
            return json.load(
                stream,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_json_constant,
            )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"Cannot read strict JSON {path}: {exc}") from exc


def _mapping(value: Any, name: str, location: Path) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{location}: {name} must be a JSON object")
    return value


def _required(mapping: Mapping[str, Any], name: str, location: Path) -> Any:
    if name not in mapping:
        raise ValueError(f"{location}: missing required field {name!r}")
    return mapping[name]


def _string(value: Any, name: str, location: Path) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{location}: {name} must be a non-empty string")
    return value


def _integer(value: Any, name: str, location: Path) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{location}: {name} must be an integer")
    return value


def _finite_number(value: Any, name: str, location: Path) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{location}: {name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{location}: {name} must be finite")
    return result


def _non_negative_number(value: Any, name: str, location: Path) -> float:
    result = _finite_number(value, name, location)
    if result < 0.0:
        raise ValueError(f"{location}: {name} must be non-negative")
    return result


def _validate_json_value(value: Any, name: str, location: Path) -> None:
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, int) and not isinstance(value, bool):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{location}: {name} contains a non-finite float")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{name}[{index}]", location)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{location}: {name} contains a non-string key")
            _validate_json_value(item, f"{name}.{key}", location)
        return
    raise ValueError(f"{location}: {name} contains unsupported JSON data")


def _summary_metrics(
    path: Path,
    *,
    model: str,
    run_id: str,
    source_study: str,
    repeat: int,
    seed: int,
) -> tuple[float, float, int, str, int, int, int]:
    recalls: list[float] = []
    runtimes: list[float] = []
    query_ids: set[str] = set()
    total_tp = total_fp = total_fn = 0
    try:
        stream = path.open("r", encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"Cannot read scene summaries {path}: {exc}") from exc
    with stream:
        for line_number, line in enumerate(stream, 1):
            location = Path(f"{path}:{line_number}")
            if not line.strip():
                raise ValueError(f"{location}: blank JSONL record")
            try:
                record = json.loads(
                    line,
                    object_pairs_hook=_reject_duplicate_keys,
                    parse_constant=_reject_json_constant,
                )
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"{location}: invalid strict JSON: {exc}") from exc
            record = _mapping(record, "record", location)
            expected = {
                "run_id": run_id,
                "dataset": "bop_itodd",
                "split": "dev",
                "object_name": model,
                "method": f"best:{source_study}",
                "repeat_id": repeat,
                "seed": seed,
            }
            for name, expected_value in expected.items():
                value = _required(record, name, location)
                if type(value) is not type(expected_value) or value != expected_value:
                    raise ValueError(
                        f"{location}: {name}={value!r}, expected {expected_value!r}"
                    )
            query_id = _string(
                _required(record, "scene_id", location), "scene_id", location
            )
            if query_id in query_ids:
                raise ValueError(f"{location}: duplicate query scene_id {query_id!r}")
            query_ids.add(query_id)
            tp = _integer(_required(record, "tp", location), "tp", location)
            fp = _integer(_required(record, "fp", location), "fp", location)
            fn = _integer(_required(record, "fn", location), "fn", location)
            if tp < 0 or fp < 0 or fn < 0:
                raise ValueError(f"{location}: tp, fp, and fn must be non-negative")
            total_tp += tp
            total_fp += fp
            total_fn += fn
            status = _string(_required(record, "status", location), "status", location)
            if status not in {"COMPLETE", "TIMEOUT"}:
                raise ValueError(f"{location}: status must be COMPLETE or TIMEOUT")
            denominator = tp + fn
            recalls.append(tp / denominator if denominator else 0.0)
            runtimes.append(
                _non_negative_number(
                    _required(record, "runtime_ms", location), "runtime_ms", location
                )
            )
    if not recalls:
        raise ValueError(f"{path}: scene summaries must contain at least one query")
    query_payload = json.dumps(
        sorted(query_ids), separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")
    return (
        statistics.fmean(recalls),
        statistics.median(runtimes),
        len(recalls),
        hashlib.sha256(query_payload).hexdigest(),
        total_tp,
        total_fp,
        total_fn,
    )


def _load_candidate(
    manifest_path: Path, *, objective_version: str = OBJECTIVE
) -> dict[str, Any]:
    path = manifest_path.expanduser().resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"Input manifest is not a file: {path}")
    manifest = _mapping(load_strict_json(path), "manifest", path)
    schema_version = _required(manifest, "schema_version", path)
    if type(schema_version) is not int or schema_version != 1:
        raise ValueError(f"{path}: schema_version={schema_version!r}, expected 1")
    if _required(manifest, "status", path) != "COMPLETE":
        raise ValueError(f"{path}: evaluation run status must be COMPLETE")
    cli = _mapping(_required(manifest, "cli_config", path), "cli_config", path)
    expected_fields = {
        "command": "evaluate-best",
        "split": "dev",
        "objective_version": objective_version,
        "pruner": "Nop",
    }
    for name, expected in expected_fields.items():
        source = manifest if name == "command" else cli
        value = _required(source, name, path)
        if type(value) is not type(expected) or value != expected:
            raise ValueError(f"{path}: {name}={value!r}, expected {expected!r}")
    if _required(cli, "command", path) != "evaluate-best":
        raise ValueError(f"{path}: cli_config.command must be 'evaluate-best'")
    if cli.get("parameter_freeze") is not None:
        raise ValueError(f"{path}: dev selection input must not use parameter_freeze")

    run_id = _string(_required(manifest, "run_id", path), "run_id", path)
    model = _string(_required(cli, "model", path), "model", path)
    sampler = _string(_required(cli, "sampler", path), "sampler", path)
    if sampler not in SAMPLERS:
        raise ValueError(f"{path}: sampler must be one of {SAMPLERS}, got {sampler!r}")
    seed = _integer(_required(cli, "seed", path), "seed", path)
    repeat = _integer(_required(cli, "repeat", path), "repeat", path)
    if seed not in SEED_TO_REPEAT or repeat != SEED_TO_REPEAT[seed]:
        raise ValueError(f"{path}: seed/repeat must follow {SEED_TO_REPEAT}")

    source_study = _string(
        _required(manifest, "source_study", path), "source_study", path
    )
    expected_study = (
        f"{model}_{objective_version}_{sampler}_Nop_repeat{repeat}_seed{seed}"
    )
    if source_study != expected_study:
        raise ValueError(
            f"{path}: source_study={source_study!r}, expected {expected_study!r}"
        )
    source_best_value = _finite_number(
        _required(manifest, "source_best_value", path), "source_best_value", path
    )
    fixed_params = _mapping(
        _required(manifest, "fixed_params", path), "fixed_params", path
    )
    _validate_json_value(fixed_params, "fixed_params", path)
    params_sha256 = canonical_params_sha256(fixed_params)
    if _required(manifest, "fixed_params_sha256", path) != params_sha256:
        raise ValueError(f"{path}: fixed_params_sha256 does not match fixed_params")
    if _required(manifest, "source_study_split", path) != "train":
        raise ValueError(f"{path}: source_study_split must be 'train'")
    source_study_budget = _integer(
        _required(manifest, "source_study_budget", path),
        "source_study_budget",
        path,
    )
    source_study_terminal_trials = _integer(
        _required(manifest, "source_study_terminal_trials", path),
        "source_study_terminal_trials",
        path,
    )
    if source_study_budget <= 0:
        raise ValueError(f"{path}: source_study_budget must be positive")
    if source_study_terminal_trials != source_study_budget:
        raise ValueError(
            f"{path}: source study has {source_study_terminal_trials} terminal "
            f"trials, expected frozen budget {source_study_budget}"
        )
    result = _mapping(_required(manifest, "result", path), "result", path)
    objective = _finite_number(
        _required(result, "objective", path), "result.objective", path
    )
    result_expected = {
        "run_id": run_id,
        "method": f"best:{source_study}",
        "repeat_id": repeat,
        "seed": seed,
    }
    for name, expected in result_expected.items():
        value = _required(result, name, path)
        if type(value) is not type(expected) or value != expected:
            raise ValueError(
                f"{path}: result.{name}={value!r}, expected {expected!r}"
            )
    outputs = _mapping(_required(manifest, "outputs", path), "outputs", path)
    raw_input_files = _required(manifest, "input_files", path)
    if not isinstance(raw_input_files, list):
        raise ValueError(f"{path}: input_files must be a list")
    dev_manifest_inputs = [
        item
        for item in raw_input_files
        if isinstance(item, dict) and item.get("name") == "bop_manifest"
    ]
    if len(dev_manifest_inputs) != 1:
        raise ValueError(f"{path}: input_files must contain one bop_manifest")
    dev_manifest_input = dev_manifest_inputs[0]
    dev_manifest_sha256 = _string(
        _required(dev_manifest_input, "sha256", path),
        "input_files.bop_manifest.sha256",
        path,
    )
    raw_summaries = _string(
        _required(outputs, "scene_summaries_jsonl", path),
        "outputs.scene_summaries_jsonl",
        path,
    )
    summaries_path = Path(raw_summaries).expanduser()
    if not summaries_path.is_absolute():
        summaries_path = path.parent / summaries_path
    summaries_path = summaries_path.resolve(strict=True)
    if not summaries_path.is_file():
        raise ValueError(f"Scene summaries path is not a file: {summaries_path}")
    (
        macro_recall,
        runtime_median_ms,
        query_count,
        query_set_sha256,
        summary_tp,
        summary_fp,
        summary_fn,
    ) = _summary_metrics(
        summaries_path,
        model=model,
        run_id=run_id,
        source_study=source_study,
        repeat=repeat,
        seed=seed,
    )
    if objective_version == REVISED_OBJECTIVE:
        if query_count != 10:
            raise ValueError(f"{path}: revised dev evaluation must contain 10 queries")
        aggregate_expected = {
            "scene_count": query_count,
            "tp": summary_tp,
            "fp": summary_fp,
            "fn": summary_fn,
        }
        for name, expected in aggregate_expected.items():
            value = _required(result, name, path)
            if type(value) is not type(expected) or value != expected:
                raise ValueError(
                    f"{path}: result.{name}={value!r}, expected {expected!r}"
                )
    manifest_sha256 = sha256_file(path)
    return {
        "manifest_path": str(path),
        "manifest_sha256": manifest_sha256,
        "dev_manifest_sha256": dev_manifest_sha256,
        "scene_summaries_jsonl": str(summaries_path),
        "scene_summaries_sha256": sha256_file(summaries_path),
        "run_id": run_id,
        "model": model,
        "objective_version": objective_version,
        "sampler": sampler,
        "pruner": "Nop",
        "repeat": repeat,
        "seed": seed,
        "source_study": source_study,
        "source_best_value": source_best_value,
        "source_study_budget": source_study_budget,
        "source_study_terminal_trials": source_study_terminal_trials,
        "params": dict(fixed_params),
        "params_sha256": params_sha256,
        "dev_objective": objective,
        "dev_macro_recall": macro_recall,
        "dev_runtime_median_ms": runtime_median_ms,
        "query_count": query_count,
        "query_set_sha256": query_set_sha256,
    }


def build_parameter_freeze(manifest_paths: Sequence[Path]) -> dict[str, Any]:
    if len(manifest_paths) != len(SAMPLERS) * len(SEEDS):
        raise ValueError("Exactly 15 evaluate-best dev manifests are required")
    candidates = [_load_candidate(path) for path in manifest_paths]
    models = {candidate["model"] for candidate in candidates}
    if len(models) != 1:
        raise ValueError(f"All manifests must use the same model, got {sorted(models)}")
    query_sets = {candidate["query_set_sha256"] for candidate in candidates}
    if len(query_sets) != 1:
        raise ValueError("All manifests must evaluate the same dev query set")
    dev_manifests = {candidate["dev_manifest_sha256"] for candidate in candidates}
    if len(dev_manifests) != 1:
        raise ValueError("All runs must use the same frozen dev manifest")
    budgets = {candidate["source_study_budget"] for candidate in candidates}
    if len(budgets) != 1:
        raise ValueError("All source studies must use the same frozen budget")
    studies = [candidate["source_study"] for candidate in candidates]
    if len(set(studies)) != len(studies):
        raise ValueError("Every source_study must be unique")

    sampler_audit = []
    for sampler in SAMPLERS:
        group = [
            candidate for candidate in candidates if candidate["sampler"] == sampler
        ]
        if len(group) != len(SEEDS):
            raise ValueError(f"Sampler {sampler} must have exactly five manifests")
        seeds = {candidate["seed"] for candidate in group}
        if seeds != set(SEEDS):
            raise ValueError(f"Sampler {sampler} seeds must be exactly {list(SEEDS)}")
        sampler_audit.append(
            {
                "sampler": sampler,
                "candidate_count": len(group),
                "dev_objective_median": statistics.median(
                    candidate["dev_objective"] for candidate in group
                ),
                "dev_macro_recall_median": statistics.median(
                    candidate["dev_macro_recall"] for candidate in group
                ),
                "dev_runtime_median_ms": statistics.median(
                    candidate["dev_runtime_median_ms"] for candidate in group
                ),
            }
        )
    best_sampler_objective = min(
        item["dev_objective_median"] for item in sampler_audit
    )
    tied_samplers = [
        item
        for item in sampler_audit
        if item["dev_objective_median"] - best_sampler_objective <= TOLERANCE
    ]
    winning_sampler_audit = min(
        tied_samplers,
        key=lambda item: (
            -item["dev_macro_recall_median"],
            item["dev_runtime_median_ms"],
            item["sampler"],
        ),
    )
    winning_sampler = winning_sampler_audit["sampler"]
    objective_median = winning_sampler_audit["dev_objective_median"]
    finalists = [
        candidate for candidate in candidates if candidate["sampler"] == winning_sampler
    ]
    for candidate in candidates:
        sampler_median = next(
            item["dev_objective_median"]
            for item in sampler_audit
            if item["sampler"] == candidate["sampler"]
        )
        candidate["distance_to_sampler_objective_median"] = abs(
            candidate["dev_objective"] - sampler_median
        )
    best_distance = min(
        candidate["distance_to_sampler_objective_median"] for candidate in finalists
    )
    tied_finalists = [
        candidate
        for candidate in finalists
        if (
            candidate["distance_to_sampler_objective_median"] - best_distance
            <= TOLERANCE
        )
    ]
    selected = min(
        tied_finalists,
        key=lambda item: (
            -item["dev_macro_recall"],
            item["dev_runtime_median_ms"],
            item["seed"],
        ),
    )
    selected_identity = {
        name: selected[name]
        for name in (
            "source_study",
            "source_best_value",
            "objective_version",
            "sampler",
            "pruner",
            "repeat",
            "seed",
        )
    }
    ordered_candidates = sorted(
        candidates, key=lambda item: (item["sampler"], item["seed"])
    )
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "model": next(iter(models)),
        "selection_rules": {
            "eligible_samplers": list(SAMPLERS),
            "pruner": "Nop",
            "seeds": list(SEEDS),
            "repeat_by_seed": {
                str(seed): repeat for seed, repeat in SEED_TO_REPEAT.items()
            },
            "objective": OBJECTIVE,
            "tolerance": TOLERANCE,
            "tolerance_application": (
                "include values no more than tolerance above the minimum primary value"
            ),
            "query_macro_recall": (
                "mean(tp / (tp + fn), with zero denominator defined as 0)"
            ),
            "candidate_runtime": "median query runtime_ms",
            "sampler_order": [
                "minimum median dev objective",
                "maximum median dev macro recall",
                "minimum median dev runtime_ms",
                "sampler lexicographic order",
            ],
            "candidate_order": [
                "minimum absolute distance from winning sampler median dev objective",
                "maximum dev macro recall",
                "minimum median dev runtime_ms",
                "minimum seed",
            ],
        },
        "input_manifests": [
            {"path": item["manifest_path"], "sha256": item["manifest_sha256"]}
            for item in ordered_candidates
        ],
        "candidates": ordered_candidates,
        "sampler_audit": sorted(sampler_audit, key=lambda item: item["sampler"]),
        "winning_sampler": winning_sampler,
        "selected_study_identity": selected_identity,
        "params": selected["params"],
        "params_sha256": selected["params_sha256"],
    }


def build_revised_parameter_freeze(
    manifest_paths: Sequence[Path],
) -> dict[str, Any]:
    """Build a PBR-only freeze while rejecting zero-recall selections."""

    if len(manifest_paths) != len(SAMPLERS) * len(SEEDS):
        raise ValueError("Exactly 15 evaluate-best dev manifests are required")
    candidates = [
        _load_candidate(path, objective_version=REVISED_OBJECTIVE)
        for path in manifest_paths
    ]
    models = {candidate["model"] for candidate in candidates}
    if len(models) != 1:
        raise ValueError(f"All manifests must use the same model, got {sorted(models)}")
    if len({candidate["query_set_sha256"] for candidate in candidates}) != 1:
        raise ValueError("All manifests must evaluate the same dev query set")
    if len({candidate["dev_manifest_sha256"] for candidate in candidates}) != 1:
        raise ValueError("All runs must use the same frozen dev manifest")
    if len({candidate["source_study_budget"] for candidate in candidates}) != 1:
        raise ValueError("All source studies must use the same frozen budget")
    studies = [candidate["source_study"] for candidate in candidates]
    if len(set(studies)) != len(studies):
        raise ValueError("Every source_study must be unique")

    sampler_audit = []
    for sampler in SAMPLERS:
        group = [
            candidate for candidate in candidates if candidate["sampler"] == sampler
        ]
        if len(group) != len(SEEDS):
            raise ValueError(f"Sampler {sampler} must have exactly five manifests")
        if {candidate["seed"] for candidate in group} != set(SEEDS):
            raise ValueError(f"Sampler {sampler} seeds must be exactly {list(SEEDS)}")
        positive_count = sum(
            candidate["dev_macro_recall"] > 0.0 for candidate in group
        )
        sampler_audit.append(
            {
                "sampler": sampler,
                "candidate_count": len(group),
                "positive_recall_candidate_count": positive_count,
                "eligible": positive_count > 0,
                "dev_objective_median": statistics.median(
                    candidate["dev_objective"] for candidate in group
                ),
                "dev_macro_recall_median": statistics.median(
                    candidate["dev_macro_recall"] for candidate in group
                ),
                "dev_runtime_median_ms": statistics.median(
                    candidate["dev_runtime_median_ms"] for candidate in group
                ),
            }
        )

    eligible = [item for item in sampler_audit if item["eligible"]]
    if not eligible:
        raise ValueError("NO_ELIGIBLE_POSITIVE_RECALL_CANDIDATE")
    best_sampler_objective = min(item["dev_objective_median"] for item in eligible)
    tied_samplers = [
        item
        for item in eligible
        if item["dev_objective_median"] - best_sampler_objective <= TOLERANCE
    ]
    winning_sampler_audit = min(
        tied_samplers,
        key=lambda item: (
            -item["dev_macro_recall_median"],
            item["dev_runtime_median_ms"],
            item["sampler"],
        ),
    )
    winning_sampler = winning_sampler_audit["sampler"]
    objective_median = winning_sampler_audit["dev_objective_median"]
    finalists = [
        candidate
        for candidate in candidates
        if candidate["sampler"] == winning_sampler
        and candidate["dev_macro_recall"] > 0.0
    ]
    for candidate in candidates:
        sampler_median = next(
            item["dev_objective_median"]
            for item in sampler_audit
            if item["sampler"] == candidate["sampler"]
        )
        candidate["distance_to_sampler_objective_median"] = abs(
            candidate["dev_objective"] - sampler_median
        )
        candidate["positive_recall_eligible"] = (
            candidate["dev_macro_recall"] > 0.0
        )
    best_distance = min(
        candidate["distance_to_sampler_objective_median"] for candidate in finalists
    )
    tied_finalists = [
        candidate
        for candidate in finalists
        if candidate["distance_to_sampler_objective_median"] - best_distance
        <= TOLERANCE
    ]
    selected = min(
        tied_finalists,
        key=lambda item: (
            -item["dev_macro_recall"],
            item["dev_runtime_median_ms"],
            item["seed"],
        ),
    )
    if selected["dev_macro_recall"] <= 0.0:
        raise AssertionError("revised selection produced a zero-recall candidate")
    selected_identity = {
        name: selected[name]
        for name in (
            "source_study",
            "source_best_value",
            "objective_version",
            "sampler",
            "pruner",
            "repeat",
            "seed",
        )
    }
    ordered_candidates = sorted(
        candidates, key=lambda item: (item["sampler"], item["seed"])
    )
    return {
        "schema": REVISED_SCHEMA,
        "version": REVISED_VERSION,
        "model": next(iter(models)),
        "selection_status": "POSITIVE_RECALL_CANDIDATE_SELECTED",
        "selection_rules": {
            "eligible_samplers": list(SAMPLERS),
            "sampler_eligibility": "at least one candidate with dev macro recall > 0",
            "candidate_eligibility": "dev macro recall > 0",
            "pruner": "Nop",
            "seeds": list(SEEDS),
            "repeat_by_seed": {
                str(seed): repeat for seed, repeat in SEED_TO_REPEAT.items()
            },
            "objective": REVISED_OBJECTIVE,
            "tolerance": TOLERANCE,
            "query_macro_recall": (
                "mean(tp / (tp + fn), with zero denominator defined as 0)"
            ),
            "sampler_order": [
                "minimum median dev objective among eligible samplers",
                "maximum median dev macro recall",
                "minimum median dev runtime_ms",
                "sampler lexicographic order",
            ],
            "candidate_order": [
                "positive dev macro recall only",
                "minimum absolute distance from winning sampler median dev objective",
                "maximum dev macro recall",
                "minimum median dev runtime_ms",
                "minimum seed",
            ],
        },
        "input_manifests": [
            {"path": item["manifest_path"], "sha256": item["manifest_sha256"]}
            for item in ordered_candidates
        ],
        "candidates": ordered_candidates,
        "sampler_audit": sorted(sampler_audit, key=lambda item: item["sampler"]),
        "winning_sampler": winning_sampler,
        "winning_sampler_objective_median": objective_median,
        "selected_study_identity": selected_identity,
        "selected_dev_macro_recall": selected["dev_macro_recall"],
        "params": selected["params"],
        "params_sha256": selected["params_sha256"],
    }


def _write_freeze_once(freeze: Mapping[str, Any], output_path: Path) -> None:
    destination = output_path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / (
        f".{destination.name}.{secrets.token_hex(8)}.tmp"
    )
    try:
        atomic_write_json(temporary, freeze)
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise ValueError(
                f"Parameter freeze output already exists: {destination}"
            ) from exc
    finally:
        if temporary.exists():
            temporary.unlink()


def select_parameter_freeze(
    manifest_paths: Sequence[Path], output_path: Path
) -> dict[str, Any]:
    freeze = build_parameter_freeze(manifest_paths)
    _write_freeze_once(freeze, output_path)
    return freeze


def select_revised_parameter_freeze(
    manifest_paths: Sequence[Path], output_path: Path
) -> dict[str, Any]:
    freeze = build_revised_parameter_freeze(manifest_paths)
    _write_freeze_once(freeze, output_path)
    return freeze


def validate_parameter_freeze(path: Path) -> dict[str, Any]:
    freeze_path = path.expanduser().resolve(strict=True)
    freeze = load_strict_json(freeze_path)
    if not isinstance(freeze, dict):
        raise ValueError("Parameter freeze must be a JSON object")
    if freeze.get("schema") != SCHEMA or freeze.get("version") != VERSION:
        raise ValueError("Parameter freeze schema or version is invalid")
    raw_inputs = freeze.get("input_manifests")
    if not isinstance(raw_inputs, list) or len(raw_inputs) != len(SAMPLERS) * len(SEEDS):
        raise ValueError("Parameter freeze must contain exactly 15 input manifests")
    manifest_paths = []
    for index, item in enumerate(raw_inputs):
        if not isinstance(item, dict):
            raise ValueError(f"input_manifests[{index}] must be an object")
        raw_path = item.get("path")
        expected_sha256 = item.get("sha256")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError(f"input_manifests[{index}].path must be non-empty")
        if not isinstance(expected_sha256, str):
            raise ValueError(f"input_manifests[{index}].sha256 must be a string")
        manifest_path = Path(raw_path).expanduser().resolve(strict=True)
        if sha256_file(manifest_path) != expected_sha256:
            raise ValueError(f"Input manifest SHA256 changed: {manifest_path}")
        manifest_paths.append(manifest_path)
    rebuilt = build_parameter_freeze(manifest_paths)
    if freeze != rebuilt:
        raise ValueError("Parameter freeze does not match recomputed dev selection")
    return freeze


def validate_revised_parameter_freeze(path: Path) -> dict[str, Any]:
    freeze_path = path.expanduser().resolve(strict=True)
    freeze = load_strict_json(freeze_path)
    if not isinstance(freeze, dict):
        raise ValueError("Revised parameter freeze must be a JSON object")
    if (
        freeze.get("schema") != REVISED_SCHEMA
        or freeze.get("version") != REVISED_VERSION
    ):
        raise ValueError("Revised parameter freeze schema or version is invalid")
    raw_inputs = freeze.get("input_manifests")
    if not isinstance(raw_inputs, list) or len(raw_inputs) != len(SAMPLERS) * len(SEEDS):
        raise ValueError("Revised parameter freeze must contain exactly 15 inputs")
    manifest_paths = []
    for index, item in enumerate(raw_inputs):
        if not isinstance(item, dict):
            raise ValueError(f"input_manifests[{index}] must be an object")
        raw_path = item.get("path")
        expected_sha256 = item.get("sha256")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError(f"input_manifests[{index}].path must be non-empty")
        path_value = Path(raw_path).expanduser().resolve(strict=True)
        if sha256_file(path_value) != expected_sha256:
            raise ValueError(f"Input manifest SHA256 changed: {path_value}")
        manifest_paths.append(path_value)
    rebuilt = build_revised_parameter_freeze(manifest_paths)
    if freeze != rebuilt:
        raise ValueError("Revised freeze does not match recomputed dev selection")
    return freeze


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select a preregistered sampler on PBR dev and freeze parameters."
    )
    parser.add_argument("manifests", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--protocol", choices=("original", "pbr-revised"), default="original"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.protocol == "pbr-revised":
        freeze = select_revised_parameter_freeze(args.manifests, args.output)
    else:
        freeze = select_parameter_freeze(args.manifests, args.output)
    print(
        f"Selected {freeze['winning_sampler']} / "
        f"{freeze['selected_study_identity']['source_study']} -> "
        f"{args.output.expanduser().resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
