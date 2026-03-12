"""Objective function for hyperparameter optimization (Section 2.4.2 of the paper).

Implements:
  - Cost Function:   C(Pose, Pose_gt) = sum|p_i - p_i^gt| + sum|r_j - r_j^gt|
  - Loss Function:   L = sum of cost over all matched poses below threshold
  - Invalid Poses Penalty:  P(N_I) = N_I * (THOLD_p + checkAxis * THOLD_r)
  - Objective Function:     O = L + P

The objective supports three types of objects with different symmetry handling:
  - bracket_planar (no symmetry):    check all 3 rotation axes
  - screw_black (axial symmetry):    check 2 rotation axes (X, Y only)
  - star (discrete rotational sym.): check 3 axes with modular Z angle
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path

import halcon as ha
from spatialmath import SE3

from kinematics import transXYZ, rotX, rotY, rotZ
from config import ModelConfig, ROIConfig, DatasetPaths, SEARCH_SPACE
from scene_loader import load_scene, filter_scene_roi, compute_roi_transform, load_model


def load_ground_truth(
    gt_json_path: str, model_cfg: ModelConfig
) -> Tuple[List[str], Dict[str, List[SE3]]]:
    """Load ground truth poses from JSON file.

    Args:
        gt_json_path: Path to scene_gt_{model}.json
        model_cfg:    ModelConfig with gt_z_offset

    Returns:
        scene_list:   List of scene IDs (strings)
        pose_gt_dict: Dict mapping scene_id -> list of SE3 ground truth poses
    """
    scene_list = []
    pose_gt_dict = {}

    with open(gt_json_path, "r") as f:
        data = json.load(f)

    for scene_id in data:
        scene_list.append(scene_id)
        pose_list = []
        for pose_dict in data[scene_id]:
            R = pose_dict["cam_R_m2c"]
            t = pose_dict["cam_t_m2c"]
            pose = SE3(
                [
                    [R[0], R[1], R[2], t[0]],
                    [R[3], R[4], R[5], t[1]],
                    [R[6], R[7], R[8], t[2] + model_cfg.gt_z_offset],
                    [0, 0, 0, 1],
                ]
            )
            pose_list.append(pose)
        pose_gt_dict[scene_id] = pose_list

    return scene_list, pose_gt_dict


def suggest_params(trial) -> Dict:
    """Sample hyperparameters from the search space using an Optuna trial.

    Args:
        trial: Optuna trial object

    Returns:
        Dictionary of parameter name -> sampled value
    """
    params = {}
    for name, spec in SEARCH_SPACE.items():
        if spec["type"] == "float":
            params[name] = trial.suggest_float(
                name, spec["low"], spec["high"], step=spec["step"]
            )
        elif spec["type"] == "int":
            params[name] = trial.suggest_int(
                name, spec["low"], spec["high"], step=spec["step"]
            )
        elif spec["type"] == "categorical":
            params[name] = trial.suggest_categorical(name, spec["choices"])
    return params


def run_surface_matching(
    model_surface, scene_roi, params: Dict, timeout_sec: float = 5.0
):
    """Run HALCON find_surface_model with given parameters.

    Args:
        model_surface: HALCON surface model handle
        scene_roi:     HALCON scene point cloud (already ROI-filtered)
        params:        Dict of hyperparameters
        timeout_sec:   Timeout in seconds for the matching operation

    Returns:
        Tuple of (Pose, Score) or (None, None) if matching fails/times out
    """
    ha.set_operator_timeout("find_surface_model", timeout_sec, "cancel")
    Pose, Score, _ = ha.find_surface_model(
        model_surface,
        scene_roi,
        params["RelSamplingDistance"],
        params["KeyPointFraction"],
        0.01,  # MinScore
        "false",  # ReturnResultHandle
        [
            "num_matches",
            "scene_invert_normals",
            "max_overlap_dist_rel",
            "pose_ref_num_steps",
            "pose_ref_sub_sampling",
            "pose_ref_dist_threshold_rel",
            "pose_ref_scoring_dist_rel",
            "pose_ref_use_scene_normals",
        ],
        [
            10,
            "true",
            params["max_overlap_dist_rel"],
            round(params["pose_ref_num_steps"]),
            round(params["pose_ref_sub_sampling"]),
            params["pose_ref_dist_threshold_rel"],
            params["pose_ref_scoring_dist_rel"],
            params["pose_ref_use_scene_normals"],
        ],
    )
    return Pose, Score


def compute_rotation_error(match: SE3, gt: SE3, model_cfg: ModelConfig) -> float:
    """Compute rotation error between a matched pose and ground truth.

    Handles symmetry differently per model type:
      - bracket_planar: sum of |rx| + |ry| + |rz|
      - screw_black:    sum of |rx| + |ry|  (Z is symmetry axis)
      - star:           sum of |rx| + |ry| + |rz % symmetryAngle|

    Args:
        match:     SE3 of the matched pose
        gt:        SE3 of the ground truth pose
        model_cfg: ModelConfig with symmetry info

    Returns:
        Rotation error in degrees
    """
    err = gt.inv() * match
    rpy = err.rpy(unit="deg", order="xyz")  # returns [rx, ry, rz]

    if model_cfg.check_axis == 2:
        # Axially symmetric (e.g. screw_black): ignore Z rotation
        # Note: spatialmath rpy order='xyz' -> index 0=rx, 1=ry, 2=rz
        # But original code used order='zyx' for index 0 and order='xyz' for index 1
        rpy_zyx = err.rpy(unit="deg", order="zyx")
        return abs(rpy_zyx[0]) + abs(rpy[1])

    elif model_cfg.symmetry_angle is not None:
        # Discrete rotational symmetry (e.g. star with 30-degree symmetry)
        rpy_zyx = err.rpy(unit="deg", order="zyx")
        diff_z = abs(rpy[2]) % model_cfg.symmetry_angle
        if diff_z > model_cfg.symmetry_angle / 2:
            diff_z = model_cfg.symmetry_angle - diff_z
        return abs(rpy_zyx[0]) + abs(rpy[1]) + diff_z

    else:
        # No symmetry (e.g. bracket_planar): check all 3 axes
        rpy_zyx = err.rpy(unit="deg", order="zyx")
        return abs(rpy_zyx[0]) + abs(rpy[1]) + abs(rpy[2])


def compute_objective_for_scene(
    Pose, Score, gt_poses: List[SE3], roi_mat: np.ndarray, model_cfg: ModelConfig
) -> float:
    """Compute objective function value for a single scene (Eq. 5 in paper).

    For each matched pose:
      1. Transform from ROI frame back to camera frame
      2. Compute position error (mm) and rotation error (deg) against all GT poses
      3. Assign match to GT with lowest combined error if below thresholds
      4. Apply penalty for unmatched GT poses

    Args:
        Pose:      Flat list of pose values from HALCON (7 values per pose: tx,ty,tz,rx,ry,rz,type)
        Score:     List of matching scores
        gt_poses:  List of SE3 ground truth poses for this scene
        roi_mat:   4x4 ROI transform matrix
        model_cfg: Model configuration

    Returns:
        Objective function value for this scene
    """
    n_gt = len(gt_poses)

    if len(Score) == 0:
        # No matches: fixed penalty per scene (matches original code)
        return 200

    # Track best loss for each GT pose (-1 means unmatched)
    loss_list = -1 * np.ones(n_gt)

    for idx in range(n_gt):
        # HALCON packs 7 values per pose (tx, ty, tz, rx, ry, rz, type).
        # Original indexing: Pose[idx+6*idx : idx+6*idx+6] == Pose[7*idx : 7*idx+6]
        pose_vals = Pose[7 * idx : 7 * idx + 6]

        # Build 4x4 transform in ROI frame
        translation = transXYZ(pose_vals[0], pose_vals[1], pose_vals[2])
        rotation = rotX(pose_vals[3]).dot(rotY(pose_vals[4])).dot(rotZ(pose_vals[5]))
        pose_in_roi = translation.dot(rotation)

        # Transform to camera frame
        pose_in_cam = roi_mat.dot(pose_in_roi)
        match = SE3(pose_in_cam)

        # Compute position errors against all GT (in mm)
        pos_errors = []
        for gt in gt_poses:
            err = match.t * 1000 - gt.t  # match is in meters, GT in mm
            pos_errors.append(np.linalg.norm(err))

        # Compute rotation errors against all GT
        rot_errors = []
        for gt in gt_poses:
            rot_errors.append(compute_rotation_error(match, gt, model_cfg))

        # Assign to best matching GT if within thresholds
        for gt_idx in range(n_gt):
            if pos_errors[gt_idx] > model_cfg.position_bound:
                continue
            if rot_errors[gt_idx] > model_cfg.check_axis * model_cfg.rotation_bound:
                continue
            combined = pos_errors[gt_idx] + rot_errors[gt_idx]
            if loss_list[gt_idx] == -1 or loss_list[gt_idx] > combined:
                loss_list[gt_idx] = combined

    # Compute final objective: average matched loss + penalty for misses
    matched_losses = [l for l in loss_list if l != -1]
    n_matched = len(matched_losses)

    if n_matched > 0:
        obj_value = sum(matched_losses) / n_matched + model_cfg.penalty_per_miss * (
            n_gt - n_matched
        )
    else:
        obj_value = model_cfg.penalty_per_miss * n_gt

    return obj_value


def create_objective(
    model_3d,
    model_cfg: ModelConfig,
    scene_list: List[str],
    pose_gt_dict: Dict[str, List[SE3]],
    dataset_paths: DatasetPaths,
    roi: ROIConfig,
    timeout_sec: float = 5.0,
    timeout_penalty: float = 2000.0,
):
    """Create an Optuna objective function for a given model and scenes.

    This is the main factory function implementing Algorithm 1 / Eq. 6 from the paper:
      V = CalculateObjective(M_i, S_j, [P_min, P_max], Poses_gt, [theta_1, ..., theta_k])

    Args:
        model_3d:       HALCON model with normals
        model_cfg:      ModelConfig
        scene_list:     List of scene IDs
        pose_gt_dict:   Ground truth poses per scene
        dataset_paths:  Dataset path configuration
        roi:            ROI configuration
        timeout_sec:    Per-scene matching timeout
        timeout_penalty: Penalty value returned on timeout

    Returns:
        Callable objective function for Optuna study.optimize()
    """
    roi_mat, roi_pose_inv = compute_roi_transform(roi)

    def objective(trial):
        params = suggest_params(trial)
        model_surface = ha.create_surface_model(
            model_3d, params["RelSamplingDistance"], [], []
        )

        total_objective = 0.0
        for scene_id in scene_list:
            # Load and filter scene
            scene_prefix = dataset_paths.scene_image_prefix(scene_id)
            scene_3d = load_scene(scene_prefix)
            scene_roi = filter_scene_roi(scene_3d, roi_pose_inv, roi)

            try:
                Pose, Score = run_surface_matching(
                    model_surface, scene_roi, params, timeout_sec
                )

                scene_obj = compute_objective_for_scene(
                    Pose, Score, pose_gt_dict[scene_id], roi_mat, model_cfg
                )
                total_objective += scene_obj

            except Exception:
                return timeout_penalty

        return total_objective

    return objective
