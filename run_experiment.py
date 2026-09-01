"""Unified experiment runner for BOP 3D surface matching hyperparameter optimization.

Executes the complete experimental lifecycle:
1. Default baseline evaluation on dev split
2. Optuna hyperparameter optimization (with initial seed injection & live progress)
3. Checkpoint evaluations across the optimization trajectory
4. Decision criteria check & final report generation
"""

from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from config import DEFAULT_PARAMS, PRUNER_NAMES, SAMPLER_NAMES, TARGET_OBJECT_IDS
from experiment_io import atomic_write_json, sha256_file, utc_now
from hpo_objectives import (
    FIXED_PENALTY_BASELINE,
    LEXICOGRAPHICAL_RECALL_FIRST,
)


def read_json_dict(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

ROOT = Path(__file__).resolve().parent
_GCCI = ctypes.windll.psapi.GetProcessMemoryInfo if hasattr(ctypes.windll, "psapi") else None


class _ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def _read_process_memory_bytes(pid: int) -> int:
    if _GCCI is None:
        return 0
    handle = ctypes.windll.kernel32.OpenProcess(0x0400 | 0x0010, False, pid)
    if not handle:
        return 0
    try:
        counters = _ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(_ProcessMemoryCounters)
        if _GCCI(handle, ctypes.byref(counters), counters.cb):
            return int(counters.PagefileUsage)
        return 0
    finally:
        ctypes.windll.kernel32.CloseHandle(handle)


def _python() -> str:
    return str(Path(sys.executable).resolve())


def local_now() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def run_phase_subprocess(
    command: Sequence[str],
    log_file: Path,
    phase_name: str,
    wall_limit_sec: float = 86400.0,
    memory_limit_bytes: int | None = None,
) -> float:
    """Execute a CLI subcommand in a monitored subprocess with memory & wall clock guards."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    start_time = time.monotonic()
    peak_memory = 0

    print(f"[{local_now()}] {phase_name}: starting (max wall time: {wall_limit_sec:.0f}s)...")
    with open(log_file, "w", encoding="utf-8") as out:
        proc = subprocess.Popen(
            command,
            cwd=str(ROOT),
            stdout=out,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            while True:
                ret = proc.poll()
                if ret is not None:
                    if ret != 0:
                        raise RuntimeError(
                            f"{phase_name} failed with exit code {ret}. See log: {log_file}"
                        )
                    break
                elapsed = time.monotonic() - start_time
                if elapsed > wall_limit_sec:
                    proc.kill()
                    raise TimeoutError(
                        f"{phase_name} exceeded wall clock limit ({wall_limit_sec}s)"
                    )
                current_mem = _read_process_memory_bytes(proc.pid)
                peak_memory = max(peak_memory, current_mem)
                if memory_limit_bytes and current_mem > memory_limit_bytes:
                    proc.kill()
                    raise MemoryError(
                        f"{phase_name} exceeded memory limit: {current_mem / (1024**3):.2f} GB"
                    )
                time.sleep(1.0)
        finally:
            if proc.poll() is None:
                proc.kill()

    duration = time.monotonic() - start_time
    print(f"[{local_now()}] {phase_name}: completed in {duration:.1f}s (peak memory: {peak_memory / (1024**2):.1f} MB)")
    return duration


def run_single_experiment(
    config_path: Path | None = None,
    model: str = "screw_black",
    sampler: str = "TPE",
    pruner: str = "Nop",
    budget: int = 100,
    checkpoints: Sequence[int] | None = None,
    seed: int = 42,
    repeat: int = 0,
    objective_version: str = LEXICOGRAPHICAL_RECALL_FIRST,
    results_root: Path | None = None,
    manifest_path: Path | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Run full HPO optimization lifecycle for a specified configuration."""
    manifest = manifest_path or (ROOT / "data" / "manifests" / "itodd_scene0_v1" / "bop_manifest.csv")
    obj_tag = "lexrecall" if ("recall" in objective_version or "lex" in objective_version) else "fixedpen"
    exp_dir_name = f"{model}_{sampler.lower()}_{pruner.lower()}_{obj_tag}_b{budget}_s{seed}"
    out_root = results_root or (ROOT / "results" / exp_dir_name)
    step = 25 if budget <= 100 else 50
    ckpt_list = list(checkpoints or range(step, budget + 1, step))
    if not ckpt_list or ckpt_list[-1] != budget:
        if budget not in ckpt_list:
            ckpt_list.append(budget)

    studies_dir = out_root / "studies"
    evals_dir = out_root / "evaluations"
    logs_dir = out_root / "logs"
    studies_dir.mkdir(parents=True, exist_ok=True)
    evals_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    common_opts = [
        "--bop-manifest", str(manifest),
        "--model", model,
        "--objective-version", objective_version,
    ]
    if config_path and config_path.exists():
        common_opts += ["--protocol-record", str(config_path)]

    # 1. Default Baseline Evaluation
    default_run_id = f"eval-default-dev"
    default_manifest = evals_dir / default_run_id / "manifest.json"
    if not default_manifest.exists():
        cmd = [
            _python(), str(ROOT / "main.py"),
            "evaluate-default",
            *common_opts,
            "--split", "dev",
            "--results-root", str(evals_dir),
            "--run-id", default_run_id,
            "--seed", str(seed),
        ]
        run_phase_subprocess(cmd, logs_dir / "evaluate_default.log", "Baseline Evaluation")

    default_result = read_json_dict(default_manifest)["result"]
    print(f"[*] Baseline Dev Result: TP={default_result['tp']}, FP={default_result['fp']}, FN={default_result['fn']}, F1={default_result['f1']:.4f}")

    # 2. HPO Optimization
    opts_dir = out_root / "optimizations"
    opts_dir.mkdir(parents=True, exist_ok=True)
    opt_cmd = [
        _python(), str(ROOT / "main.py"),
        "optimize",
        *common_opts,
        "--split", "train",
        "--budget", str(budget),
        "--sampler", sampler,
        "--pruner", pruner,
        "--repeat", str(repeat),
        "--seed", str(seed),
        "--storage-dir", str(studies_dir),
        "--results-root", str(opts_dir),
    ]
    if resume:
        opt_cmd.append("--resume")

    run_phase_subprocess(opt_cmd, logs_dir / "optimization.log", f"Optimization ({sampler}+{pruner}, budget={budget})")

    # 3. Checkpoint Evaluations
    checkpoint_results = []
    for ckpt in ckpt_list:
        ckpt_run_id = f"eval-ckpt-{ckpt}-dev"
        ckpt_manifest = evals_dir / ckpt_run_id / "manifest.json"
        if not ckpt_manifest.exists():
            eval_cmd = [
                _python(), str(ROOT / "main.py"),
                "evaluate-best",
                *common_opts,
                "--split", "dev",
                "--results-root", str(evals_dir),
                "--run-id", ckpt_run_id,
                "--trial-limit", str(ckpt),
                "--storage-dir", str(studies_dir),
                "--sampler", sampler,
                "--pruner", pruner,
                "--repeat", str(repeat),
                "--seed", str(seed),
            ]
            run_phase_subprocess(eval_cmd, logs_dir / f"eval_ckpt_{ckpt}.log", f"Checkpoint {ckpt} Evaluation")
        
        res = read_json_dict(ckpt_manifest)["result"]
        checkpoint_results.append({
            "checkpoint": ckpt,
            "tp": res["tp"],
            "fp": res["fp"],
            "fn": res["fn"],
            "f1": res["f1"],
            "objective": res["objective"],
            "method": res["method"],
        })
        print(f"[*] Checkpoint {ckpt:4d}: TP={res['tp']:2d}, FP={res['fp']:3d}, FN={res['fn']:2d}, F1={res['f1']:.4f}, Loss={res['objective']:.2f}")

    summary = {
        "experiment_name": exp_dir_name,
        "model": model,
        "sampler": sampler,
        "pruner": pruner,
        "objective_version": objective_version,
        "budget": budget,
        "seed": seed,
        "default_baseline": default_result,
        "checkpoints": checkpoint_results,
    }
    atomic_write_json(out_root / "experiment_summary.json", summary)
    print(f"\n[+] Experiment completed. Summary written to {out_root / 'experiment_summary.json'}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Unified HPO Experiment Runner")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "hpo_protocol.json",
        help="Path to experiment protocol JSON",
    )
    parser.add_argument("--model", default="screw_black", choices=list(TARGET_OBJECT_IDS.keys()))
    parser.add_argument("--sampler", default="TPE", choices=SAMPLER_NAMES)
    parser.add_argument("--pruner", default="Nop", choices=PRUNER_NAMES)
    parser.add_argument("--budget", type=int, default=100)
    parser.add_argument(
        "--objective-version",
        default=LEXICOGRAPHICAL_RECALL_FIRST,
        choices=[
            LEXICOGRAPHICAL_RECALL_FIRST,
            FIXED_PENALTY_BASELINE,
        ],
        help="HPO objective version (default: lexicographical-recall-first)",
    )
    parser.add_argument("--checkpoints", type=str, help="Comma-separated checkpoints, e.g. '25,50,75,100'")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--matrix",
        action="store_true",
        help="Run full comparison matrix across all 3 core samplers (TPE, CmaEs, Random)",
    )

    args = parser.parse_args()
    ckpts = [int(c.strip()) for c in args.checkpoints.split(",")] if args.checkpoints else None

    if args.matrix:
        print(f"[*] Starting 3x3 Comparison Matrix for {args.model} across 3 Samplers ({', '.join(SAMPLER_NAMES)}) x {args.pruner}...")
        for s in SAMPLER_NAMES:
            print(f"\n{'='*80}\nRunning Sampler: {s} | Pruner: {args.pruner}\n{'='*80}")
            run_single_experiment(
                config_path=args.config,
                model=args.model,
                sampler=s,
                pruner=args.pruner,
                budget=args.budget,
                checkpoints=ckpts,
                seed=args.seed,
                repeat=args.repeat,
                objective_version=args.objective_version,
                manifest_path=args.manifest,
                resume=args.resume,
            )
        return 0

    run_single_experiment(
        config_path=args.config,
        model=args.model,
        sampler=args.sampler,
        pruner=args.pruner,
        budget=args.budget,
        checkpoints=ckpts,
        seed=args.seed,
        repeat=args.repeat,
        objective_version=args.objective_version,
        results_root=args.results_root,
        manifest_path=args.manifest,
        resume=args.resume,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
