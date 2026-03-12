# hpo-3d-match

Hyperparameter optimization for 3D object pose estimation with HALCON `find_surface_model`.

This project implements the method described in the paper `A Parameters Optimization Framework for Pose Estimation Algorithm Based on Point Cloud`. It uses Optuna to optimize the key parameters of HALCON surface matching and evaluates them with a pose-error-based objective function.

## What this project does

- optimizes 8 parameters of `find_surface_model`
- uses the pose-error objective proposed in the paper
- supports three object types with different symmetry rules
- runs Optuna studies with multiple sampler and pruner combinations
- evaluates default parameters and optimized parameters on stored scenes

## Features

- local dataset layout support out of the box
- compatible with the original ITODD-style folder layout
- `uv`-based dependency management
- SQLite-backed Optuna studies
- paper-aligned ROI filtering and objective calculation

## Repository layout

```text
hpo-3d-match/
  config.py          Model settings, search space, ROI, dataset paths
  kinematics.py      Homogeneous transform helpers
  scene_loader.py    Scene loading and ROI filtering
  objective.py       Objective function and HALCON matching wrapper
  optimizer.py       Optuna study creation and reporting
  main.py            CLI entry point
  data/
    cad_models/
      star.ply
      screw.ply
      bracket_planar.ply
    scene_gt_star.json
    scene_gt_screw_black.json
    scene_gt_bracket_planar.json
    scenes/
      scene_0000/
      scene_0003/
      scene_0293/
      scene_0296/
      scene_0450/
```

## Requirements

- Python 3.10+
- `uv`
- HALCON Python environment available on your machine

Note: this repository does not include HALCON environment or license setup. Configure HALCON locally before running the scripts.

## Install

```bash
uv sync
```

## Data layout

By default the code reads data from the local `data/` directory in this project.

Before running the project, download the required ITODD dataset files manually and place them into `data/` using the layout shown below.

Expected local layout:

```text
data/
  cad_models/
    star.ply
    screw.ply
    bracket_planar.ply
  scene_gt_*.json
  scenes/
    scene_0000/
      3d_long_baseline_x.tif
      3d_long_baseline_y.tif
      3d_long_baseline_z.tif
      3d_long_baseline_l.tif
```

The code also supports an ITODD-style layout when `--dataset-path` points to the corresponding dataset root.

In other words, this repository does not ship the ITODD data itself. You need to prepare the model `.ply` files, the `scene_gt_*.json` files, and the scene folders manually, then organize them under `data/` in the same format.

## Supported models

| Model | Symmetry handling | Checked axes | GT z-offset |
|---|---|---:|---:|
| `star` | discrete rotational symmetry, 30 deg | 3 | -5.68374 mm |
| `screw_black` | axial symmetry | 2 | 0 mm |
| `bracket_planar` | no symmetry | 3 | -5.68374 mm |

## Search space

The implementation optimizes the following HALCON parameters:

- `RelSamplingDistance`
- `KeyPointFraction`
- `max_overlap_dist_rel`
- `pose_ref_num_steps`
- `pose_ref_sub_sampling`
- `pose_ref_dist_threshold_rel`
- `pose_ref_scoring_dist_rel`
- `pose_ref_use_scene_normals`

The search space follows Table 2 of the paper and is defined in `config.py`.

## Usage

### 1. Evaluate default parameters

```bash
uv run python main.py --model star --evaluate default
uv run python main.py --model screw_black --evaluate default
uv run python main.py --model bracket_planar --evaluate default
```

### 2. Run optimization

Run one sampler and pruner combination:

```bash
uv run python main.py --model star --sampler TPE --pruner Nop --n-trials 500
```

Run all configured combinations:

```bash
uv run python main.py --model star --n-trials 500 --n-jobs 8
```

### 3. Print stored optimization results

```bash
uv run python main.py --model star --print-results
```

### 4. Evaluate best stored parameters

```bash
uv run python main.py --model star --evaluate best
```

### 5. Use a custom dataset path

```bash
uv run python main.py --model star --dataset-path ./data --n-trials 500
```

## Main CLI arguments

- `--model`: `star`, `screw_black`, or `bracket_planar`
- `--dataset-path`: dataset root, default is local `./data`
- `--gt-dir`: directory containing `scene_gt_*.json`
- `--storage-dir`: directory for Optuna SQLite databases
- `--n-trials`: number of optimization trials
- `--n-jobs`: number of parallel Optuna jobs
- `--timeout`: timeout for each `find_surface_model` call
- `--sampler`: choose one sampler
- `--pruner`: choose one pruner
- `--evaluate`: evaluate `default` or `best`
- `--print-results`: print saved study results

## Output

- optimization results are stored in SQLite databases under `results/`
- one database file is created per model
- each study is named as `<model>_<sampler>_<pruner>`

## Method overview

The pipeline follows the paper:

1. load the target model and scene point clouds
2. crop the scene with a fixed ROI
3. run HALCON `find_surface_model` with sampled parameters
4. compare matched poses with ground-truth poses
5. compute the objective from pose error and unmatched-pose penalty
6. minimize the total objective with Optuna

## Notes

- the current default dataset layout expects model files in `data/cad_models/`
- `screw_black` uses `data/cad_models/screw.ply`
- if `uv run` can import `halcon`, the project should be ready to execute

## Citation

If you use this repository in research, please cite the associated paper.

```bibtex
@article{niu2024parameters,
  title={A Parameters optimization framework for pose estimation algorithm based on point cloud},
  author={Niu, Qun and Wang, Ziru and Li, Hongkun and Zhao, Jieliang},
  journal={Journal of Physics: Conference Series},
  volume={2746},
  number={1},
  pages={012039},
  year={2024},
  organization={IOP Publishing}
}

```
