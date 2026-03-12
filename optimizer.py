"""Optuna study creation and execution (Algorithm 1 in the paper).

Creates studies with various sampler x pruner combinations, runs optimization,
and stores results in SQLite.
"""

import optuna
import datetime as dt
from typing import Callable, List, Optional, Tuple
from pathlib import Path


def get_sampler(name: str) -> optuna.samplers.BaseSampler:
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
    return samplers[name]()


def get_pruner(name: str) -> optuna.pruners.BasePruner:
    """Create an Optuna pruner by name.

    Supported: Median, Nop, Hyperband
    """
    pruners = {
        "Median": optuna.pruners.MedianPruner,
        "Nop": optuna.pruners.NopPruner,
        "Hyperband": optuna.pruners.HyperbandPruner,
    }
    if name not in pruners:
        raise ValueError(f"Unknown pruner: {name}. Choose from {list(pruners.keys())}")
    return pruners[name]()


def create_and_run_study(
    objective: Callable,
    model_name: str,
    sampler_name: str,
    pruner_name: str,
    n_trials: int = 5000,
    n_jobs: int = 32,
    storage_dir: str = "./results",
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
        sampler_name: Sampler name (TPE, CmaEs, NSGAII, QMC, Random)
        pruner_name:  Pruner name (Median, Nop, Hyperband)
        n_trials:     Number of optimization trials
        n_jobs:       Number of parallel jobs
        storage_dir:  Directory for SQLite database files

    Returns:
        Tuple of (study, elapsed_time)
    """
    Path(storage_dir).mkdir(parents=True, exist_ok=True)
    db_path = Path(storage_dir) / f"{model_name}.db"
    storage_url = f"sqlite:///{db_path}"

    study_name = f"{model_name}_{sampler_name}_{pruner_name}"

    sampler = get_sampler(sampler_name)
    pruner = get_pruner(pruner_name)

    study = optuna.create_study(
        storage=storage_url,
        study_name=study_name,
        sampler=sampler,
        pruner=pruner,
        direction="minimize",
        load_if_exists=True,
    )

    print(f"Study: {study_name}")
    print(f"  Sampler: {study.sampler.__class__.__name__}")
    print(f"  Pruner:  {study.pruner.__class__.__name__}")
    print(f"  Trials:  {n_trials}, Jobs: {n_jobs}")

    start = dt.datetime.now()
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)
    elapsed = dt.datetime.now() - start

    print(f"  Completed in {elapsed}")
    print(f"  Best value:  {study.best_value}")
    print(f"  Best params: {study.best_params}")

    return study, elapsed


def run_all_combinations(
    objective: Callable,
    model_name: str,
    sampler_names: List[str],
    pruner_names: List[str],
    n_trials: int = 5000,
    n_jobs: int = 32,
    storage_dir: str = "./results",
) -> List[Tuple[str, str, optuna.Study, dt.timedelta]]:
    """Run optimization for all sampler x pruner combinations.

    This is the full experimental setup from Section 3.1 of the paper.

    Args:
        objective:     Callable objective function
        model_name:    Model name
        sampler_names: List of sampler names to test
        pruner_names:  List of pruner names to test
        n_trials:      Number of trials per combination
        n_jobs:        Parallel jobs
        storage_dir:   SQLite storage directory

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
                sampler_name=sampler_name,
                pruner_name=pruner_name,
                n_trials=n_trials,
                n_jobs=n_jobs,
                storage_dir=storage_dir,
            )
            results.append((sampler_name, pruner_name, study, elapsed))

    total_elapsed = dt.datetime.now() - total_start
    print(f"\n{'=' * 60}")
    print(f"All combinations completed in {total_elapsed}")

    return results


def print_study_results(
    storage_dir: str, model_name: str, sampler_names: List[str], pruner_names: List[str]
):
    """Load and print results for all sampler x pruner combinations.

    Args:
        storage_dir:   SQLite storage directory
        model_name:    Model name
        sampler_names: List of sampler names
        pruner_names:  List of pruner names
    """
    db_path = Path(storage_dir) / f"{model_name}.db"
    storage_url = f"sqlite:///{db_path}"

    for sampler_name in sampler_names:
        for pruner_name in pruner_names:
            study_name = f"{model_name}_{sampler_name}_{pruner_name}"
            try:
                study = optuna.load_study(
                    storage=storage_url,
                    study_name=study_name,
                )
                print(f"{'=' * 60}")
                print(f"Study: {study_name}")
                print(f"  Best value:  {study.best_value}")
                print(f"  Best params: {study.best_params}")
                print(f"  N trials:    {len(study.trials)}")
            except Exception as e:
                print(f"Study {study_name}: not found ({e})")
