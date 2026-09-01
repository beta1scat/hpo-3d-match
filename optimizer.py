"""Optuna study creation and execution (Algorithm 1 in the paper).

Creates studies with various sampler x pruner combinations, runs optimization,
and stores results in SQLite.
"""

import datetime as dt
import math
import sys
from typing import Callable, List, Mapping, Tuple
from pathlib import Path

import optuna

from experiment_io import append_trial_record
from hpo_objectives import (
    FIXED_PENALTY_BASELINE,
    LEXICOGRAPHICAL_RECALL_FIRST,
)


JSONScalar = str | int | float | bool | None
CURRENT_OBJECTIVE_VERSION = LEXICOGRAPHICAL_RECALL_FIRST
RESUMABLE_OBJECTIVE_VERSIONS = frozenset(
    {
        FIXED_PENALTY_BASELINE,
        LEXICOGRAPHICAL_RECALL_FIRST,
    }
)

REQUIRED_STUDY_CONTEXT_KEYS = frozenset(
    {
        "dataset_manifest_sha256",
        "dataset_inputs_sha256",
        "split",
        "search_space_sha256",
        "timeout_sec",
        "min_score",
        "num_matches",
        "translation_threshold_mm",
        "rotation_threshold_deg",
        "fn_penalty",
        "fp_penalty",
        "model_scale",
        "symmetry_sha256",
    }
)
RESERVED_STUDY_ATTR_KEYS = frozenset(
    {
        "model_name",
        "objective_version",
        "sampler_name",
        "pruner_name",
        "repeat",
        "seed",
        "direction",
        "budget",
        "resumed_from_trials",
        "resumed_from_terminal_trials",
        "database_total_trials",
        "database_terminal_trials",
        "resume_not_trajectory_equivalent",
        "recovered_stale_running_trials",
    }
)


def _validate_study_context(
    study_context: Mapping[str, JSONScalar] | None,
) -> dict[str, JSONScalar]:
    if study_context is None:
        return {}

    context = dict(study_context)
    missing = REQUIRED_STUDY_CONTEXT_KEYS - context.keys()
    if missing:
        raise ValueError(
            "study_context is missing required keys: " + ", ".join(sorted(missing))
        )
    reserved = context.keys() & RESERVED_STUDY_ATTR_KEYS
    if reserved:
        raise ValueError(
            "study_context uses reserved keys: " + ", ".join(sorted(reserved))
        )

    for key, value in context.items():
        if not isinstance(key, str):
            raise TypeError("study_context keys must be strings")
        if value is not None and type(value) not in (str, int, float, bool):
            raise TypeError(f"study_context[{key!r}] must be a JSON scalar")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"study_context[{key!r}] must be finite")
    for key in (
        "dataset_manifest_sha256",
        "dataset_inputs_sha256",
        "search_space_sha256",
        "symmetry_sha256",
    ):
        value = context[key]
        if not isinstance(value, str) or len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise ValueError(f"study_context[{key!r}] must be a SHA256 hex digest")
    if not isinstance(context["split"], str) or not context["split"].strip():
        raise ValueError("study_context['split'] must be a non-empty string")
    for key in ("timeout_sec", "translation_threshold_mm", "rotation_threshold_deg"):
        value = context[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"study_context[{key!r}] must be greater than zero")
    min_score = context["min_score"]
    if (
        isinstance(min_score, bool)
        or not isinstance(min_score, (int, float))
        or not 0 <= min_score <= 1
    ):
        raise ValueError("study_context['min_score'] must be between zero and one")
    num_matches = context["num_matches"]
    if isinstance(num_matches, bool) or not isinstance(num_matches, int) or num_matches <= 0:
        raise ValueError("study_context['num_matches'] must be a positive integer")
    for key in ("fn_penalty", "fp_penalty"):
        value = context[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise ValueError(f"study_context[{key!r}] must be non-negative")
    model_scale = context["model_scale"]
    if not (
        model_scale == "mm"
        or (
            not isinstance(model_scale, bool)
            and isinstance(model_scale, (int, float))
            and model_scale > 0
        )
    ):
        raise ValueError("study_context['model_scale'] must be 'mm' or positive")
    return context


def _strictly_equal(left: JSONScalar, right: JSONScalar) -> bool:
    return type(left) is type(right) and left == right


def _terminal_trial_count(study: optuna.Study) -> int:
    terminal_states = {
        optuna.trial.TrialState.COMPLETE,
        optuna.trial.TrialState.PRUNED,
        optuna.trial.TrialState.FAIL,
    }
    return sum(trial.state in terminal_states for trial in study.trials)


def _dispose_rdb_storage(storage: optuna.storages.RDBStorage) -> None:
    storage.remove_session()
    storage.engine.dispose()


def get_sampler(name: str, seed: int) -> optuna.samplers.BaseSampler:
    """Create an Optuna sampler by name.

    Supported: TPE, CmaEs, NSGAII, QMC, Random
    """
    samplers = {
        "TPE": optuna.samplers.TPESampler,
        "CmaEs": optuna.samplers.CmaEsSampler,
        "NSGAII": optuna.samplers.NSGAIISampler,
        "QMC": optuna.samplers.QMCSampler,
        "Random": optuna.samplers.RandomSampler,
    }
    if name not in samplers:
        raise ValueError(
            f"Unknown sampler: {name}. Choose from {list(samplers.keys())}"
        )
    return samplers[name](seed=seed)


def get_pruner(name: str) -> optuna.pruners.BasePruner:
    """Create an Optuna pruner by name.

    Supported: Median, Nop, Hyperband
    """
    if name == "Median":
        return optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=15)
    elif name == "Hyperband":
        return optuna.pruners.HyperbandPruner(min_resource=5, max_resource=177, reduction_factor=3)
    elif name == "Nop":
        return optuna.pruners.NopPruner()
    else:
        raise ValueError(f"Unknown pruner: {name}. Choose from ['Median', 'Nop', 'Hyperband']")


def build_study_name(
    model_name: str,
    objective_version: str,
    sampler_name: str,
    pruner_name: str,
    repeat: int,
    seed: int,
) -> str:
    """Build a stable study name from experimental identity dimensions."""
    return (
        f"{model_name}_{objective_version}_{sampler_name}_{pruner_name}"
        f"_repeat{repeat}_seed{seed}"
    )


def create_trial_jsonl_callback(
    trial_log_path: str,
    *,
    model_name: str,
    objective_version: str,
    sampler_name: str,
    pruner_name: str,
    budget: int,
    repeat: int,
    seed: int,
    direction: str,
    resumed_from_trials: int,
    resumed_from_terminal_trials: int,
    database_total_trials: int,
    database_terminal_trials: int,
    study_context: Mapping[str, JSONScalar] | None = None,
) -> Callable[[optuna.Study, optuna.trial.FrozenTrial], None]:
    """Create an Optuna callback that appends complete trial records to JSONL."""
    context = dict(study_context or {})

    def log_trial(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        append_trial_record(
            trial_log_path,
            {
                "study_name": study.study_name,
                "trial_number": trial.number,
                "state": trial.state.name,
                "model_name": model_name,
                "objective_version": objective_version,
                "sampler_name": sampler_name,
                "pruner_name": pruner_name,
                "budget": budget,
                "repeat": repeat,
                "seed": seed,
                "direction": direction,
                "resumed_from_trials": resumed_from_trials,
                "resumed_from_terminal_trials": resumed_from_terminal_trials,
                "database_total_trials": database_total_trials,
                "database_terminal_trials": database_terminal_trials,
                "study_context": context,
                "params": trial.params,
                "objective_value": trial.value,
                "duration_sec": (
                    trial.duration.total_seconds() if trial.duration else None
                ),
                "datetime_start": trial.datetime_start,
                "datetime_complete": trial.datetime_complete,
                "intermediate_values": trial.intermediate_values,
                "user_attrs": trial.user_attrs,
                "system_attrs": trial.system_attrs,
            },
            format="jsonl",
        )

    return log_trial


def create_and_run_study(
    objective: Callable,
    model_name: str,
    objective_version: str,
    sampler_name: str,
    pruner_name: str,
    target_total_trials: int = 5000,
    repeat: int = 0,
    seed: int = 42,
    resume: bool = False,
    n_jobs: int = 1,
    storage_dir: str = "./results",
    trial_log_path: str | None = None,
    study_context: Mapping[str, JSONScalar] | None = None,
    catch: tuple[type[Exception], ...] = (),
) -> Tuple[optuna.Study, dt.timedelta]:
    """Create an Optuna study, run optimization, and return results.

    Implements Algorithm 1 from the paper:
      1. Select Sampler and Pruner, create Optuna study
      2. Define objective function
      3. Run study with N trials for minimizing L
      4. Return L_min and best parameters

    Args:
        objective:    Callable objective function (from objective.create_objective)
        model_name:   Model name (used in study naming)
        objective_version: Objective function version used in study naming
        sampler_name:      Sampler name (TPE, CmaEs, NSGAII, QMC, Random)
        pruner_name:       Pruner name (Median, Nop, Hyperband)
        target_total_trials: Desired total number of trials in the study
        repeat:            Repeat index used in study naming
        seed:              Sampler seed used in study naming and trial records
        resume:            Load an existing study instead of creating a new one
        n_jobs:            Number of parallel jobs
        storage_dir:       Directory for SQLite database files
        trial_log_path:    Optional JSONL path for completed trial records
        study_context:     Optional immutable experimental context
        catch:             Trial-local exception types Optuna may record and continue

    Returns:
        Tuple of (study, elapsed_time)
    """
    if target_total_trials < 0:
        raise ValueError("target_total_trials must be non-negative")
    if resume and objective_version not in RESUMABLE_OBJECTIVE_VERSIONS:
        raise ValueError(
            "Cannot resume an existing study with objective_version "
            f"{objective_version!r}; expected one of "
            f"{sorted(RESUMABLE_OBJECTIVE_VERSIONS)!r}. "
            "Use an explicit new study name."
        )
    context = _validate_study_context(study_context)

    Path(storage_dir).mkdir(parents=True, exist_ok=True)
    db_path = Path(storage_dir) / f"{model_name}.db"
    storage_url = f"sqlite:///{db_path}"
    storage = optuna.storages.RDBStorage(storage_url)

    study_name = build_study_name(
        model_name,
        objective_version,
        sampler_name,
        pruner_name,
        repeat,
        seed,
    )

    sampler = get_sampler(sampler_name, seed)
    pruner = get_pruner(pruner_name)

    existing_study_names = optuna.get_all_study_names(storage=storage)
    study_existed = study_name in existing_study_names
    if resume and not study_existed:
        _dispose_rdb_storage(storage)
        raise ValueError(
            f"Cannot resume study {study_name!r}: no study with that exact name "
            "exists. Create it explicitly without --resume; existing studies are "
            "never renamed or reused across objective versions."
        )
    try:
        study = optuna.create_study(
            storage=storage,
            study_name=study_name,
            sampler=sampler,
            pruner=pruner,
            direction="minimize",
            load_if_exists=resume,
        )
    except BaseException:
        _dispose_rdb_storage(storage)
        raise

    direction = study.direction.name.lower()
    immutable_attrs: dict[str, JSONScalar] = {
        "model_name": model_name,
        "objective_version": objective_version,
        "sampler_name": sampler_name,
        "pruner_name": pruner_name,
        "repeat": repeat,
        "seed": seed,
        "direction": direction,
        "budget": target_total_trials,
    }
    immutable_attrs.update(context)

    try:
        if resume:
            existing_objective_version = study.user_attrs.get("objective_version")
            if not _strictly_equal(existing_objective_version, objective_version):
                raise ValueError(
                    f"Cannot resume existing study {study_name!r}: objective_version is "
                    f"{existing_objective_version!r}, expected {objective_version!r}. "
                    "Use an explicit new study name."
                )
            existing_context_keys = study.user_attrs.keys() - RESERVED_STUDY_ATTR_KEYS
            if existing_context_keys != context.keys():
                missing = existing_context_keys - context.keys()
                added = context.keys() - existing_context_keys
                differences = []
                if missing:
                    differences.append("missing: " + ", ".join(sorted(missing)))
                if added:
                    differences.append("added: " + ", ".join(sorted(added)))
                raise ValueError(
                    f"Existing study {study_name!r} has conflicting study_context keys "
                    f"({'; '.join(differences)})"
                )
            for key, expected in immutable_attrs.items():
                if key == "search_space_sha256":
                    continue
                if key not in study.user_attrs:
                    raise ValueError(
                        f"Existing study {study_name!r} is missing immutable attr {key!r}"
                    )
                actual = study.user_attrs[key]
                if not _strictly_equal(actual, expected):
                    raise ValueError(
                        f"Existing study {study_name!r} has conflicting immutable attr "
                        f"{key!r}: expected {expected!r}, found {actual!r}"
                    )
        else:
            for key, value in immutable_attrs.items():
                study.set_user_attr(key, value)
            from config import DEFAULT_PARAMS

            study.enqueue_trial(dict(DEFAULT_PARAMS))
    except BaseException:
        _dispose_rdb_storage(storage)
        raise

    recovered_stale_running_trials = 0
    if resume and objective_version in RESUMABLE_OBJECTIVE_VERSIONS:
        running_trials = study.get_trials(
            deepcopy=False, states=(optuna.trial.TrialState.RUNNING,)
        )
        for trial in running_trials:
            if not storage.set_trial_state_values(
                trial._trial_id, optuna.trial.TrialState.FAIL
            ):
                _dispose_rdb_storage(storage)
                raise RuntimeError(
                    f"Cannot recover stale RUNNING trial {trial.number} in {study_name!r}"
                )
        recovered_stale_running_trials = len(running_trials)

    resumed_from_trials = len(study.trials)
    resumed_from_terminal_trials = _terminal_trial_count(study)
    study_metadata = {
        **immutable_attrs,
        "budget": target_total_trials,
        "resumed_from_trials": resumed_from_trials,
        "resumed_from_terminal_trials": resumed_from_terminal_trials,
        "database_total_trials": resumed_from_trials,
        "database_terminal_trials": resumed_from_terminal_trials,
    }
    mutable_metadata = {
        key: study_metadata[key]
        for key in (
            "resumed_from_trials",
            "resumed_from_terminal_trials",
            "database_total_trials",
            "database_terminal_trials",
        )
    }
    if resume:
        mutable_metadata["resume_not_trajectory_equivalent"] = True
        mutable_metadata["recovered_stale_running_trials"] = (
            recovered_stale_running_trials
        )
    for key, value in mutable_metadata.items():
        study.set_user_attr(key, value)

    remaining_trials = max(0, target_total_trials - resumed_from_terminal_trials)
    callbacks = (
        [
            create_trial_jsonl_callback(
                trial_log_path,
                model_name=model_name,
                objective_version=objective_version,
                sampler_name=sampler_name,
                pruner_name=pruner_name,
                budget=target_total_trials,
                repeat=repeat,
                seed=seed,
                direction=direction,
                resumed_from_trials=resumed_from_trials,
                resumed_from_terminal_trials=resumed_from_terminal_trials,
                database_total_trials=resumed_from_trials,
                database_terminal_trials=resumed_from_terminal_trials,
                study_context=context,
            )
        ]
        if trial_log_path is not None
        else None
    )

    print(f"Study: {study_name}")
    print(f"  Sampler: {study.sampler.__class__.__name__}")
    print(f"  Pruner:  {study.pruner.__class__.__name__}")
    print(
        f"  Trials before run: {resumed_from_trials} total, "
        f"{resumed_from_terminal_trials}/{target_total_trials} terminal, "
        f"Remaining: {remaining_trials}, Jobs: {n_jobs}"
    )

    start = dt.datetime.now()
    try:
        if remaining_trials:
            study.optimize(
                objective,
                n_trials=remaining_trials,
                n_jobs=n_jobs,
                callbacks=callbacks,
                catch=catch,
            )
    finally:
        elapsed = dt.datetime.now() - start
        active_exception = sys.exc_info()[0] is not None
        metadata_error = None
        for name, value in (
            ("database_total_trials", len(study.trials)),
            ("database_terminal_trials", _terminal_trial_count(study)),
        ):
            try:
                study.set_user_attr(name, value)
            except Exception as exc:
                if metadata_error is None:
                    metadata_error = exc
        if metadata_error is not None and not active_exception:
            _dispose_rdb_storage(storage)
            raise metadata_error
        if active_exception:
            _dispose_rdb_storage(storage)

    final_total_trials = len(study.trials)
    final_terminal_trials = _terminal_trial_count(study)
    print(f"  Completed in {elapsed}")
    print(
        f"  Trials after run:  {final_total_trials} total, "
        f"{final_terminal_trials}/{target_total_trials} terminal"
    )
    completed_trials = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    if completed_trials:
        print(f"  Best value:  {study.best_value}")
        print(f"  Best params: {study.best_params}")
    else:
        print("  Best value:  unavailable (no completed trials)")
        print("  Best params: unavailable (no completed trials)")

    _dispose_rdb_storage(storage)
    return study, elapsed


def run_all_combinations(
    objective: Callable,
    model_name: str,
    objective_version: str,
    sampler_names: List[str],
    pruner_names: List[str],
    target_total_trials: int = 5000,
    repeat: int = 0,
    seed: int = 42,
    resume: bool = False,
    n_jobs: int = 1,
    storage_dir: str = "./results",
    trial_log_path: str | None = None,
    study_context: Mapping[str, JSONScalar] | None = None,
) -> List[Tuple[str, str, optuna.Study, dt.timedelta]]:
    """Run optimization for all sampler x pruner combinations.

    This is the full experimental setup from Section 3.1 of the paper.

    Args:
        objective:     Callable objective function
        model_name:    Model name
        objective_version: Objective function version
        sampler_names:     List of sampler names to test
        pruner_names:      List of pruner names to test
        target_total_trials: Desired total trials per combination
        repeat:            Repeat index
        seed:              Sampler seed
        resume:            Resume existing studies
        n_jobs:            Parallel jobs
        storage_dir:       SQLite storage directory
        trial_log_path:    Optional shared JSONL trial log path
        study_context:     Optional immutable experimental context

    Returns:
        List of (sampler_name, pruner_name, study, elapsed_time) tuples
    """
    results = []
    total_start = dt.datetime.now()

    for sampler_name in sampler_names:
        for pruner_name in pruner_names:
            print(f"\n{'=' * 60}")
            study, elapsed = create_and_run_study(
                objective=objective,
                model_name=model_name,
                objective_version=objective_version,
                sampler_name=sampler_name,
                pruner_name=pruner_name,
                target_total_trials=target_total_trials,
                repeat=repeat,
                seed=seed,
                resume=resume,
                n_jobs=n_jobs,
                storage_dir=storage_dir,
                trial_log_path=trial_log_path,
                study_context=study_context,
            )
            results.append((sampler_name, pruner_name, study, elapsed))

    total_elapsed = dt.datetime.now() - total_start
    print(f"\n{'=' * 60}")
    print(f"All combinations completed in {total_elapsed}")

    return results


def print_study_results(
    storage_dir: str,
    model_name: str,
    objective_version: str,
    sampler_names: List[str],
    pruner_names: List[str],
    repeat: int,
    seed: int,
):
    """Load and print results for all sampler x pruner combinations.

    Args:
        storage_dir:   SQLite storage directory
        model_name:    Model name
        objective_version: Objective function version
        sampler_names:     List of sampler names
        pruner_names:      List of pruner names
        repeat:            Repeat index
        seed:              Sampler seed
    """
    db_path = Path(storage_dir) / f"{model_name}.db"
    storage_url = f"sqlite:///{db_path}"

    for sampler_name in sampler_names:
        for pruner_name in pruner_names:
            study_name = build_study_name(
                model_name,
                objective_version,
                sampler_name,
                pruner_name,
                repeat,
                seed,
            )
            try:
                study = optuna.load_study(
                    storage=storage_url,
                    study_name=study_name,
                )
                print(f"{'=' * 60}")
                print(f"Study: {study_name}")
                completed_trials = [
                    trial
                    for trial in study.trials
                    if trial.state == optuna.trial.TrialState.COMPLETE
                ]
                if completed_trials:
                    print(f"  Best value:  {study.best_value}")
                    print(f"  Best params: {study.best_params}")
                else:
                    print("  Best value:  unavailable (no completed trials)")
                    print("  Best params: unavailable (no completed trials)")
                print(f"  N trials:    {len(study.trials)}")
            except Exception as e:
                print(f"Study {study_name}: not found ({e})")
