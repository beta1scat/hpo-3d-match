"""Unified Visualization Tool for 3D Surface Matching on BOP and Native ITODD Datasets.

Given optimized parameters and a scene specification:
1. Performs surface matching using HALCON SurfaceMatcher.
2. Projects and renders matched 3D CAD models onto 2D sensor images (RGB / Grayscale).
3. Renders 3D scene point clouds overlaid with the detected CAD models in 3D.
4. Saves all prediction metadata (poses R/t, scores, time, params) to JSON/TXT.

Supports both dataset types:
- BOP format (e.g. data/itoddmv_val)
- ITODD Native format (e.g. data/3d_long_baseline)
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import cv2
import halcon as ha
import imageio.v3 as iio
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from bop_scene_loader import (
    BOPCamera,
    backproject_depth,
    create_halcon_point_cloud,
    read_bop_camera,
    read_depth_image,
)
from config import DEFAULT_PARAMS, RANSAC_TABLETOP_THRESHOLDS, TARGET_OBJECT_IDS
from dataset import ITODD_EXTERNAL_MANIFEST_FIELDS, TARGET_CAD_FILES
from external_pipeline import (
    load_original_cad,
    load_xyz_point_cloud,
    read_external_queries,
)
from matcher import MatchError, SurfaceMatcher
from pipeline import _load_model, surface_matching_config


def get_best_params_from_study(storage_dir: Path, model_name: str) -> dict[str, Any]:
    """Read the best parameter set from an Optuna study DB."""
    import optuna

    resolved_dir = storage_dir.resolve()
    if not resolved_dir.is_dir():
        alt = (Path(__file__).parent / storage_dir).resolve()
        if alt.is_dir():
            resolved_dir = alt
        else:
            raise FileNotFoundError(
                f"Specified --storage-dir does not exist: {storage_dir} (tried {resolved_dir} and {alt})"
            )

    db_path = resolved_dir / f"{model_name}.db"
    if not db_path.is_file():
        studies_sub = resolved_dir / "studies" / f"{model_name}.db"
        if studies_sub.is_file():
            db_path = studies_sub
        else:
            raise FileNotFoundError(
                f"Optuna database '{model_name}.db' not found in {resolved_dir} or {resolved_dir / 'studies'}"
            )

    storage_url = f"sqlite:///{db_path.resolve()}"
    studies = optuna.get_all_study_summaries(storage=storage_url)
    if not studies:
        raise ValueError(f"No studies found in Optuna database: {db_path}")

    study = optuna.load_study(storage=storage_url, study_name=studies[0].study_name)
    print(f"[+] Loaded best params from Optuna study: {study.study_name} (Value: {study.best_value:.2f})")
    print(f"    Best params: {study.best_params}")
    return dict(study.best_params)


def project_points(pts_3d_m: np.ndarray, cam_k: np.ndarray) -> np.ndarray:
    """Project 3D camera coordinates in metres to 2D image pixels."""
    pts_cam = pts_3d_m.T  # 3 x N
    z = pts_cam[2, :]
    valid = z > 1e-4
    pts_2d = np.zeros((pts_3d_m.shape[0], 2), dtype=np.float32)
    
    fx, fy = cam_k[0, 0], cam_k[1, 1]
    cx, cy = cam_k[0, 2], cam_k[1, 2]
    
    pts_2d[valid, 0] = fx * (pts_cam[0, valid] / z[valid]) + cx
    pts_2d[valid, 1] = fy * (pts_cam[1, valid] / z[valid]) + cy
    return pts_2d


def render_2d_overlay(
    image_2d: np.ndarray,
    cad_mesh: o3d.geometry.TriangleMesh,
    poses: list[np.ndarray],
    cam_k: np.ndarray,
    out_path: Path,
    scores: list[float] | None = None,
) -> None:
    """Render 2D image with 3D CAD outlines/vertices projected onto it."""
    if image_2d.ndim == 2:
        img_vis = cv2.cvtColor(image_2d, cv2.COLOR_GRAY2BGR)
    else:
        img_vis = image_2d.copy()

    # Normalize image to uint8 if needed
    if img_vis.dtype != np.uint8:
        img_min, img_max = img_vis.min(), img_vis.max()
        if img_max > img_min:
            img_vis = ((img_vis - img_min) / (img_max - img_min) * 255.0).astype(np.uint8)
        else:
            img_vis = img_vis.astype(np.uint8)

    cad_vertices = np.asarray(cad_mesh.vertices)  # N x 3 in metres

    colors = [(0, 255, 0), (0, 215, 255), (255, 100, 0), (0, 128, 255)]

    for idx, pose in enumerate(poses):
        R = pose[:3, :3]
        t = pose[:3, 3]
        trans_verts = (R @ cad_vertices.T).T + t
        pts_2d = project_points(trans_verts, cam_k)
        
        color = colors[idx % len(colors)]
        # Draw convex hull of projected model
        pts_int = pts_2d.astype(np.int32)
        hull = cv2.convexHull(pts_int)
        cv2.polylines(img_vis, [hull], isClosed=True, color=color, thickness=2)
        
        # Add label
        cx, cy = np.mean(pts_2d[:, 0]), np.mean(pts_2d[:, 1])
        if np.isfinite(cx) and np.isfinite(cy):
            score_text = f" #{idx+1}" + (f": {scores[idx]:.2f}" if scores else "")
            cv2.putText(img_vis, score_text, (int(cx) - 20, int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img_vis)
    print(f"[+] Saved 2D overlay image to {out_path}")


def render_3d_pointcloud(
    scene_points_m: np.ndarray,
    cad_mesh: o3d.geometry.TriangleMesh,
    poses: list[np.ndarray],
    out_path: Path,
    scene_colors: np.ndarray | None = None,
    peeled_points_list: list[np.ndarray] | None = None,
    camera_json_path: Path | None = None,
) -> None:
    """Render 3D point cloud in millimeter space matching visualize_annotation.py standard."""
    pts_arr = np.asarray(scene_points_m, dtype=np.float64)
    # Convert to millimeters if scene points are in meters (max coordinate < 20.0)
    pts_mm = pts_arr * 1000.0 if pts_arr.size > 0 and np.max(np.abs(pts_arr)) < 20.0 else pts_arr

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_mm)
    if len(pts_mm) > 0:
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=10.0, max_nn=30))

    if scene_colors is not None and len(scene_colors) == len(pts_mm):
        pcd.colors = o3d.utility.Vector3dVector(scene_colors)
    else:
        pcd.paint_uniform_color([0.65, 0.70, 0.75])

    vis_geoms = [pcd]
    colors = [
        [0.0, 0.90, 0.35],   # Neon Green
        [1.0, 0.65, 0.0],    # Amber Orange
        [0.1, 0.65, 1.0],    # Electric Blue
        [0.95, 0.2, 0.85],   # Vivid Magenta
        [0.2, 0.95, 0.9],    # Bright Cyan
    ]

    # Ensure cad_mesh template is in millimeter scale
    mesh_template = copy.deepcopy(cad_mesh)
    verts = np.asarray(mesh_template.vertices)
    if len(verts) > 0 and np.max(np.abs(verts)) < 5.0:
        mesh_template.scale(1000.0, center=(0, 0, 0))

    for idx, pose in enumerate(poses):
        inst_mesh = copy.deepcopy(mesh_template)
        T = np.array(pose, dtype=np.float64, copy=True)
        # If translation in meters, scale to mm
        if np.max(np.abs(T[:3, 3])) < 20.0:
            T[:3, 3] *= 1000.0
        inst_mesh.transform(T)
        inst_mesh.compute_vertex_normals()
        inst_mesh.paint_uniform_color(colors[idx % len(colors)])
        vis_geoms.append(inst_mesh)

        # Local coordinate axis at object origin (size: 35.0 mm, matching visualize_annotation.py)
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=35.0, origin=[0, 0, 0])
        frame.transform(T)
        vis_geoms.append(frame)

    # If peeling progression layers are provided, render them with distinct progression colors
    if peeled_points_list:
        peel_palette = [
            [0.95, 0.25, 0.25],  # Layer 1: Coral red
            [1.00, 0.55, 0.10],  # Layer 2: Vivid orange
            [0.85, 0.20, 0.80],  # Layer 3: Purple magenta
            [0.20, 0.80, 0.85],  # Layer 4: Cyan teal
            [0.90, 0.80, 0.20],  # Layer 5: Gold
        ]
        for idx, peeled_pts in enumerate(peeled_points_list):
            if len(peeled_pts) > 0:
                p_arr = np.asarray(peeled_pts, dtype=np.float64)
                p_mm = p_arr * 1000.0 if np.max(np.abs(p_arr)) < 20.0 else p_arr
                p_pcd = o3d.geometry.PointCloud()
                p_pcd.points = o3d.utility.Vector3dVector(p_mm)
                p_pcd.paint_uniform_color(peel_palette[idx % len(peel_palette)])
                vis_geoms.append(p_pcd)

    vis = o3d.visualization.Visualizer()
    win_w, win_h = 1600, 950
    vis.create_window(visible=False, width=win_w, height=win_h)
    for g in vis_geoms:
        vis.add_geometry(g)

    # Render options for publication quality
    opt = vis.get_render_option()
    opt.background_color = np.asarray([1.0, 1.0, 1.0])  # Pure white background
    opt.point_size = 3.0
    opt.light_on = True

    ctr = vis.get_view_control()
    has_custom_camera = False
    if camera_json_path is not None and Path(camera_json_path).exists():
        try:
            param = o3d.io.read_pinhole_camera_parameters(str(camera_json_path))
            ctr.convert_from_pinhole_camera_parameters(param, allow_arbitrary=True)
            has_custom_camera = True
        except Exception as e:
            print(f"[!] Warning: Failed to apply camera JSON {camera_json_path}: {e}")

    if not has_custom_camera:
        # Fallback to elevated 3D oblique perspective: global bounding box focus
        min_b = pcd.get_min_bound()
        max_b = pcd.get_max_bound()
        for g in vis_geoms[1:]:
            min_b = np.minimum(min_b, g.get_min_bound())
            max_b = np.maximum(max_b, g.get_max_bound())
        global_center = (min_b + max_b) / 2.0

        ctr.set_lookat(global_center)
        ctr.set_front([0.32, -0.42, -0.85])
        ctr.set_up([0.12, -0.90, 0.42])
        ctr.set_zoom(0.58)

    vis.poll_events()
    vis.update_renderer()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vis.capture_screen_image(str(out_path))
    vis.destroy_window()

    # Option A: When using fixed camera JSON, do NOT auto-crop so framing is strictly preserved
    if not has_custom_camera:
        img_bgr = cv2.imread(str(out_path))
        if img_bgr is not None:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            mask = gray < 250
            coords = cv2.findNonZero(mask.astype(np.uint8))
            if coords is not None:
                x, y, w, h = cv2.boundingRect(coords)
                pad = 50
                x1 = max(0, x - pad)
                y1 = max(0, y - pad)
                x2 = min(img_bgr.shape[1], x + w + pad)
                y2 = min(img_bgr.shape[0], y + h + pad)
                cropped = img_bgr[y1:y2, x1:x2]
                cv2.imwrite(str(out_path), cropped)

    print(f"[+] Saved 3D point cloud visualization to {out_path}")


def sequential_subtraction_matching(
    model_point_cloud_halcon: Any,
    cad_mesh_o3d: o3d.geometry.TriangleMesh,
    initial_scene_points_xyz_m: np.ndarray,
    matching_params: dict[str, Any],
    min_score: float = 0.50,
    subtraction_radius: float = 0.005,
    max_iterations: int = 10,
    interactive: bool = False,
) -> tuple[list[np.ndarray], list[float], list[dict[str, Any]], list[np.ndarray]]:
    """Execute recursive single-instance matching and spatial point subtraction (Route A).

    Args:
        model_point_cloud_halcon: HALCON surface model handle or 3D object model
        cad_mesh_o3d: Open3D CAD mesh in metres
        initial_scene_points_xyz_m: Initial camera-space scene point cloud in metres
        matching_params: PPF/ICP hyperparameters
        min_score: Minimum matching and convergence threshold; stops when best match score < min_score
        subtraction_radius: Distance threshold (metres) to peel away points around matched CAD
        max_iterations: Safety ceiling on iterations
        interactive: If True, opens Open3D GUI window at each step (press 'q' to proceed)

    Returns:
        poses_4x4: List of detected 4x4 pose matrices in camera frame
        scores: List of matching confidence scores
        history: Step-by-step diagnostic records
        peeled_layers: List of peeled point clouds per iteration for visual inspection
    """
    current_points = np.array(initial_scene_points_xyz_m, dtype=np.float64, copy=True)
    cad_surface_pts = np.asarray(cad_mesh_o3d.sample_points_poisson_disk(number_of_points=12000).points)

    detected_poses: list[np.ndarray] = []
    detected_scores: list[float] = []
    history: list[dict[str, Any]] = []
    peeled_layers: list[np.ndarray] = []

    instance_colors = [
        [0.0, 0.85, 0.20],  # Instance 1: Green
        [1.0, 0.80, 0.00],  # Instance 2: Yellow
        [0.0, 0.75, 1.00],  # Instance 3: Cyan
        [1.0, 0.20, 0.70],  # Instance 4: Magenta
        [1.0, 0.50, 0.00],  # Instance 5: Orange
        [0.5, 0.30, 0.90],  # Instance 6: Purple
    ]

    print(f"\n[*] Starting Sequential Subtraction Matching (Min Score: {min_score:.4f}, Subtraction Radius: {subtraction_radius*1000:.1f}mm, Max Iterations: {max_iterations})")
    print(f"    Initial point count: {len(current_points):,}")

    for iteration in range(1, max_iterations + 1):
        if len(current_points) < 500:
            print(f"[*] Stopping: remaining point count ({len(current_points)}) too small for matching.")
            break

        halcon_scene = create_halcon_point_cloud(current_points)
        # Search for multiple candidates to ensure HALCON explores the global voting space without premature early-exit
        config = surface_matching_config(matching_params, timeout_sec=5.0, min_score=min_score, num_matches=10)

        t_start = time.perf_counter()
        with SurfaceMatcher(model_point_cloud_halcon, config) as matcher:
            match_res = matcher.match(halcon_scene)
        iter_time_ms = (time.perf_counter() - t_start) * 1000.0

        if not match_res.predictions:
            print(f"[*] Iteration {iteration}: No hypotheses generated by HALCON.")
            history.append({
                "iteration": iteration,
                "status": "NO_PREDICTIONS",
                "runtime_ms": iter_time_ms,
                "remaining_points": len(current_points),
            })
            break

        # Strictly select the global highest-scoring candidate among all returned hypotheses
        best_idx = int(np.argmax(match_res.scores))
        best_pred = match_res.predictions[best_idx]
        best_score = float(match_res.scores[best_idx])

        if best_score < min_score:
            print(f"[*] Stopping at Iteration {iteration}: Highest score {best_score:.4f} < min_score {min_score:.4f}")
            history.append({
                "iteration": iteration,
                "status": "BELOW_THRESHOLD",
                "score": float(best_score),
                "runtime_ms": iter_time_ms,
                "remaining_points": len(current_points),
            })
            break

        # Convert to 4x4 matrix in camera frame
        mat = np.eye(4, dtype=np.float64)
        mat[:3, :3] = best_pred.rotation
        mat[:3, 3] = best_pred.translation_mm / 1000.0

        detected_poses.append(mat)
        detected_scores.append(best_score)

        # Spatial Point Subtraction using cKDTree
        cad_trans = (mat[:3, :3] @ cad_surface_pts.T + mat[:3, 3:4]).T
        cad_tree = cKDTree(cad_trans)

        dists, _ = cad_tree.query(current_points, k=1, distance_upper_bound=subtraction_radius)
        is_occupied = dists <= subtraction_radius
        peeled_count = int(np.count_nonzero(is_occupied))

        peeled_pts = current_points[is_occupied]
        peeled_layers.append(peeled_pts)

        # -------------------------------------------------------------
        # STEP 1 VISUALIZATION: Matched instance + points to be removed (RED)
        # -------------------------------------------------------------
        if interactive:
            pcd_scene_iter = o3d.geometry.PointCloud()
            pcd_scene_iter.points = o3d.utility.Vector3dVector(current_points[~is_occupied])
            pcd_scene_iter.paint_uniform_color([0.65, 0.65, 0.70])

            pcd_peeled_iter = o3d.geometry.PointCloud()
            pcd_peeled_iter.points = o3d.utility.Vector3dVector(peeled_pts)
            pcd_peeled_iter.paint_uniform_color([1.0, 0.20, 0.20])

            curr_mesh = copy.deepcopy(cad_mesh_o3d)
            curr_mesh.transform(mat)
            curr_mesh.paint_uniform_color([0.1, 0.9, 0.2])

            prev_meshes = []
            for p_idx, p_mat in enumerate(detected_poses[:-1]):
                m = copy.deepcopy(cad_mesh_o3d)
                m.transform(p_mat)
                m.paint_uniform_color(instance_colors[p_idx % len(instance_colors)])
                prev_meshes.append(m)

            print(f"\n[>>> Open3D Window: Step {iteration} - Instance #{len(detected_poses)} Detected <<<]")
            print(f"    - Score: {best_score:.4f} | Subtraction target: {peeled_count:,} points highlighted in RED")
            print(f"    --> Rotate/zoom in 3D, then PRESS 'q' on keyboard to subtract points & proceed...")
            win_title = f"Step {iteration}: Detected #{len(detected_poses)} (Score: {best_score:.3f}) | RED={peeled_count:,} pts to peel | Press 'q' to subtract"
            o3d.visualization.draw_geometries([pcd_scene_iter, pcd_peeled_iter, curr_mesh, *prev_meshes], window_name=win_title, width=1440, height=900)

        # Perform the subtraction
        current_points = current_points[~is_occupied]

        # -------------------------------------------------------------
        # STEP 2 VISUALIZATION: Cleaned point cloud AFTER subtraction
        # -------------------------------------------------------------
        if interactive:
            pcd_remaining = o3d.geometry.PointCloud()
            pcd_remaining.points = o3d.utility.Vector3dVector(current_points)
            pcd_remaining.paint_uniform_color([0.70, 0.75, 0.80])

            accum_meshes = []
            for p_idx, p_mat in enumerate(detected_poses):
                m = copy.deepcopy(cad_mesh_o3d)
                m.transform(p_mat)
                m.paint_uniform_color(instance_colors[p_idx % len(instance_colors)])
                accum_meshes.append(m)

            print(f"[>>> Open3D Window: Step {iteration} - Point Cloud AFTER Subtraction <<<]")
            print(f"    - Remaining points: {len(current_points):,}")
            print(f"    --> Rotate/zoom to inspect cleaned scene, then PRESS 'q' to continue to next match...\n")
            win_title = f"Step {iteration}: Point Cloud After Peeling (Remaining: {len(current_points):,} pts) | Press 'q' to continue"
            o3d.visualization.draw_geometries([pcd_remaining, *accum_meshes], window_name=win_title, width=1440, height=900)

        print(f"  [+] Iteration {iteration:2d}: Detected Instance #{len(detected_poses)} | Score: {best_score:.4f} | Peeled: {peeled_count:,} pts | Remaining: {len(current_points):,} pts ({iter_time_ms:.1f}ms)")

        history.append({
            "iteration": iteration,
            "status": "ACCEPTED",
            "score": float(best_score),
            "pose": mat.tolist(),
            "peeled_points": peeled_count,
            "remaining_points": len(current_points),
            "runtime_ms": iter_time_ms,
        })

    total_time_ms = sum(h.get("runtime_ms", 0.0) for h in history)
    print(f"[*] Sequential Matching Finished: Detected {len(detected_poses)} instances across {len(history)} iterations (Total Time: {total_time_ms:.1f}ms).\n")

    # -------------------------------------------------------------
    # FINAL VISUALIZATION: All detected instances together
    # -------------------------------------------------------------
    if interactive and detected_poses:
        print(f"[>>> Open3D Window: Final Multi-Instance Assembly <<<]")
        print(f"    - Total instances detected: {len(detected_poses)}")
        print(f"    --> Rotate/zoom, then PRESS 'q' to finish & save artifacts...\n")
        pcd_final = o3d.geometry.PointCloud()
        pcd_final.points = o3d.utility.Vector3dVector(current_points)
        pcd_final.paint_uniform_color([0.65, 0.65, 0.70])
        final_meshes = []
        for p_idx, p_mat in enumerate(detected_poses):
            m = copy.deepcopy(cad_mesh_o3d)
            m.transform(p_mat)
            m.paint_uniform_color(instance_colors[p_idx % len(instance_colors)])
            final_meshes.append(m)
        win_title = f"Finished: {len(detected_poses)} Instances Matched | Press 'q' to exit"
        o3d.visualization.draw_geometries([pcd_final, *final_meshes], window_name=win_title, width=1440, height=900)

    return detected_poses, detected_scores, history, peeled_layers


def visualize_bop_scene(
    model_name: str,
    manifest_csv: Path,
    scene_id: int,
    image_id: int,
    params: dict[str, Any],
    out_dir: Path,
    show_full_scene: bool = False,
    sequential_subtraction: bool = False,
    interactive: bool = False,
    min_score: float = 0.50,
    subtraction_radius: float = 0.005,
    max_iterations: int = 10,
    camera_json_path: Path | None = None,
) -> None:
    """Run matching and visualization on a BOP dataset scene (e.g. itoddmv_val)."""
    print(f"\n==================== [BOP Scene] Model: {model_name} | Scene: {scene_id} | Image: {image_id} ====================")
    obj_id = TARGET_OBJECT_IDS[model_name]
    
    # Locate files in itoddmv_val
    scene_dir = Path("data/itoddmv_val/val") / f"{scene_id:06d}"
    depth_path = scene_dir / "depth_3dlong" / f"{image_id:06d}.tif"
    gray_path = scene_dir / "gray_3dlong" / f"{image_id:06d}.tif"
    camera_path = scene_dir / "scene_camera_3dlong.json"
    cad_path = Path("data/itoddmv_models/models") / f"obj_{obj_id:06d}.ply"

    if not depth_path.exists():
        raise FileNotFoundError(f"Depth image not found: {depth_path}")

    depth = read_depth_image(depth_path)
    camera = read_bop_camera(camera_path, image_id)
    stride = 2
    points_xyz_m = backproject_depth(depth, camera, stride=stride)

    # 2D Image & Real Sensor Texture mapping
    img_2d = iio.imread(gray_path) if gray_path.exists() else np.zeros(depth.shape, dtype=np.uint8)
    depth_array = np.asarray(depth)[::stride, ::stride]
    depth_m = depth_array.astype(np.float64) * camera.depth_scale / 1000.0
    valid = np.isfinite(depth_m) & (depth_m > 0.0)
    rows, columns = np.nonzero(valid)
    
    img_norm = np.asarray(img_2d)[::stride, ::stride].astype(np.float64)
    img_norm = (img_norm - img_norm.min()) / (img_norm.max() - img_norm.min() + 1e-6)
    pt_gray = img_norm[rows, columns]
    # Realistic industrial sensor monochrome point tint
    point_colors = np.column_stack([pt_gray, pt_gray, pt_gray])

    # CAD mesh for Open3D
    cad_mesh = o3d.io.read_triangle_mesh(str(cad_path))
    cad_mesh.compute_vertex_normals()
    if np.asarray(cad_mesh.vertices).max() > 10.0:  # If in mm, convert to metres
        cad_mesh.scale(0.001, center=(0, 0, 0))

    # Tabletop RANSAC plane removal
    if not show_full_scene:
        from bop_scene_loader import filter_points_ransac_tabletop

        thresh = RANSAC_TABLETOP_THRESHOLDS.get(model_name, 0.0020)
        match_pts, non_table_mask = filter_points_ransac_tabletop(
            points_xyz_m, distance_threshold_m=thresh, return_mask=True
        )
        render_pts = match_pts
        render_colors = point_colors[non_table_mask]
    else:
        match_pts = points_xyz_m
        render_pts = points_xyz_m
        render_colors = point_colors
    halcon_model = _load_model(cad_path, "mm")

    if sequential_subtraction:
        poses_4x4, scores, history, peeled_layers = sequential_subtraction_matching(
            model_point_cloud_halcon=halcon_model,
            cad_mesh_o3d=cad_mesh,
            initial_scene_points_xyz_m=match_pts,
            matching_params=params,
            min_score=min_score,
            subtraction_radius=subtraction_radius,
            max_iterations=max_iterations,
            interactive=interactive,
        )

        prefix = f"seq_sub_bop_{model_name}_s{scene_id}_im{image_id}"
        render_2d_overlay(img_2d, cad_mesh, poses_4x4, camera.intrinsic_matrix, out_dir / f"{prefix}_2d_overlay.png", scores)
        render_3d_pointcloud(render_pts, cad_mesh, poses_4x4, out_dir / f"{prefix}_3d_pointcloud.png", scene_colors=render_colors, camera_json_path=camera_json_path)
        render_3d_pointcloud(points_xyz_m, cad_mesh, poses_4x4, out_dir / f"{prefix}_3d_raw_pointcloud.png", scene_colors=point_colors, camera_json_path=camera_json_path)
        render_3d_pointcloud(render_pts, cad_mesh, poses_4x4, out_dir / f"{prefix}_3d_peeling_progression.png", peeled_points_list=peeled_layers, camera_json_path=camera_json_path)

        meta = {
            "pipeline": "sequential_subtraction",
            "dataset": "bop_itoddmv_val",
            "model": model_name,
            "obj_id": obj_id,
            "scene_id": scene_id,
            "image_id": image_id,
            "detected_count": len(poses_4x4),
            "min_score": min_score,
            "subtraction_radius_m": subtraction_radius,
            "scores": scores,
            "poses": [p.tolist() for p in poses_4x4],
            "history": history,
            "params": params,
        }
        (out_dir / f"{prefix}_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"[+] Saved sequential subtraction metadata to {out_dir / f'{prefix}_meta.json'}")
        return

    # Standard one-shot matching
    halcon_scene = create_halcon_point_cloud(match_pts)
    config = surface_matching_config(params, timeout_sec=5.0, min_score=min_score, num_matches=10)

    with SurfaceMatcher(halcon_model, config) as matcher:
        match_res = matcher.match(halcon_scene)

    poses_4x4 = []
    for pred in match_res.predictions:
        mat = np.eye(4)
        mat[:3, :3] = pred.rotation
        mat[:3, 3] = pred.translation_mm / 1000.0
        poses_4x4.append(mat)

    print(f"[*] Found {len(poses_4x4)} matches in {match_res.runtime_ms:.1f}ms")

    # Render 2D & 3D
    prefix = f"bop_{model_name}_s{scene_id}_im{image_id}"
    render_2d_overlay(img_2d, cad_mesh, poses_4x4, camera.intrinsic_matrix, out_dir / f"{prefix}_2d_overlay.png", match_res.scores)
    render_3d_pointcloud(render_pts, cad_mesh, poses_4x4, out_dir / f"{prefix}_3d_pointcloud.png", scene_colors=render_colors, camera_json_path=camera_json_path)
    render_3d_pointcloud(points_xyz_m, cad_mesh, poses_4x4, out_dir / f"{prefix}_3d_raw_pointcloud.png", scene_colors=point_colors, camera_json_path=camera_json_path)

    # Save metadata JSON
    meta = {
        "dataset": "bop_itoddmv_val",
        "model": model_name,
        "obj_id": obj_id,
        "scene_id": scene_id,
        "image_id": image_id,
        "runtime_ms": match_res.runtime_ms,
        "match_count": len(poses_4x4),
        "scores": list(match_res.scores),
        "poses": [p.tolist() for p in poses_4x4],
        "params": params,
    }
    (out_dir / f"{prefix}_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[+] Saved metadata to {out_dir / f'{prefix}_meta.json'}")


def visualize_native_scene(
    model_name: str,
    scene_id: int,
    params: dict[str, Any],
    out_dir: Path,
    show_full_scene: bool = False,
    sequential_subtraction: bool = False,
    interactive: bool = False,
    min_score: float = 0.50,
    subtraction_radius: float = 0.005,
    max_iterations: int = 10,
    camera_json_path: Path | None = None,
) -> None:
    """Run matching and visualization on an ITODD Native scene (data/3d_long_baseline)."""
    print(f"\n==================== [ITODD Native Scene] Model: {model_name} | Scene: {scene_id} ====================")
    scene_dir = Path("data/3d_long_baseline/scenes") / f"scene_{scene_id:04d}"
    cad_path = Path("data/base_package/models/cad_models") / TARGET_CAD_FILES[model_name]
    x_path = scene_dir / "3d_long_baseline_x.tif"
    y_path = scene_dir / "3d_long_baseline_y.tif"
    z_path = scene_dir / "3d_long_baseline_z.tif"
    img_l_path = scene_dir / "3d_long_baseline_l.tif"

    if not x_path.exists():
        raise FileNotFoundError(f"Native scene not found: {scene_dir}")

    # Load 2D image
    img_2d = iio.imread(img_l_path) if img_l_path.exists() else np.zeros((1000, 1000), dtype=np.uint8)

    # Load XYZ for Open3D and map real sensor luminance
    x_img = np.asarray(iio.imread(x_path))
    y_img = np.asarray(iio.imread(y_path))
    z_img = np.asarray(iio.imread(z_path))
    valid = np.isfinite(z_img) & (z_img > 0.05)
    stride = 2
    points_xyz_m = np.stack([x_img[valid], y_img[valid], z_img[valid]], axis=-1)[::stride]
    
    l_img = np.asarray(img_2d)[valid][::stride].astype(np.float64)
    l_norm = (l_img - l_img.min()) / (l_img.max() - l_img.min() + 1e-6)
    point_colors = np.column_stack([l_norm, l_norm, l_norm])

    # Tabletop RANSAC plane removal
    if not show_full_scene:
        from bop_scene_loader import filter_points_ransac_tabletop

        thresh = RANSAC_TABLETOP_THRESHOLDS.get(model_name, 0.0020)
        filtered_scene_pts, non_table_mask = filter_points_ransac_tabletop(
            points_xyz_m, distance_threshold_m=thresh, return_mask=True
        )
        render_pts = filtered_scene_pts
        render_colors = point_colors[non_table_mask]
    else:
        filtered_scene_pts = points_xyz_m
        render_pts = points_xyz_m
        render_colors = point_colors

    # CAD mesh for Open3D
    cad_mesh = o3d.io.read_triangle_mesh(str(cad_path))
    cad_mesh.compute_vertex_normals()
    if np.asarray(cad_mesh.vertices).max() > 10.0:
        cad_mesh.scale(0.001, center=(0, 0, 0))

    # Approximate default camera K for ITODD sensor
    cam_k = np.array([[2900.0, 0.0, img_2d.shape[1] / 2.0], [0.0, 2900.0, img_2d.shape[0] / 2.0], [0.0, 0.0, 1.0]])

    model_point_cloud = load_original_cad(cad_path)

    if sequential_subtraction:
        init_pts = filtered_scene_pts
        poses_4x4, scores, history, peeled_layers = sequential_subtraction_matching(
            model_point_cloud_halcon=model_point_cloud,
            cad_mesh_o3d=cad_mesh,
            initial_scene_points_xyz_m=init_pts,
            matching_params=params,
            min_score=min_score,
            subtraction_radius=subtraction_radius,
            max_iterations=max_iterations,
            interactive=interactive,
        )

        prefix = f"seq_sub_native_{model_name}_scene{scene_id:04d}"
        render_2d_overlay(img_2d, cad_mesh, poses_4x4, cam_k, out_dir / f"{prefix}_2d_overlay.png", scores)
        render_3d_pointcloud(render_pts, cad_mesh, poses_4x4, out_dir / f"{prefix}_3d_pointcloud.png", scene_colors=render_colors, camera_json_path=camera_json_path)
        render_3d_pointcloud(points_xyz_m, cad_mesh, poses_4x4, out_dir / f"{prefix}_3d_raw_pointcloud.png", scene_colors=point_colors, camera_json_path=camera_json_path)
        render_3d_pointcloud(render_pts, cad_mesh, poses_4x4, out_dir / f"{prefix}_3d_peeling_progression.png", peeled_points_list=peeled_layers, camera_json_path=camera_json_path)

        meta = {
            "pipeline": "sequential_subtraction",
            "dataset": "itodd_native_3d_long_baseline",
            "model": model_name,
            "scene_id": scene_id,
            "detected_count": len(poses_4x4),
            "min_score": min_score,
            "subtraction_radius_m": subtraction_radius,
            "scores": scores,
            "poses": [p.tolist() for p in poses_4x4],
            "history": history,
            "params": params,
        }
        (out_dir / f"{prefix}_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"[+] Saved sequential subtraction metadata to {out_dir / f'{prefix}_meta.json'}")
        return

    # Standard one-shot HALCON matching with RANSAC tabletop removal
    from bop_scene_loader import create_halcon_point_cloud

    scene_3d = create_halcon_point_cloud(filtered_scene_pts)

    config = surface_matching_config(
        params, timeout_sec=5.0, min_score=min_score, num_matches=10
    )

    with SurfaceMatcher(model_point_cloud, config) as matcher:
        match_res = matcher.match(scene_3d)

    predictions = match_res.predictions

    poses_4x4 = []
    for pred in predictions:
        mat = np.eye(4)
        mat[:3, :3] = pred.rotation
        mat[:3, 3] = pred.translation_mm / 1000.0
        poses_4x4.append(mat)

    print(f"[*] Found {len(poses_4x4)} matches in {match_res.runtime_ms:.1f}ms")

    prefix = f"native_{model_name}_scene{scene_id:04d}"
    render_2d_overlay(
        img_2d,
        cad_mesh,
        poses_4x4,
        cam_k,
        out_dir / f"{prefix}_2d_overlay.png",
        match_res.scores,
    )
    render_3d_pointcloud(
        render_pts,
        cad_mesh,
        poses_4x4,
        out_dir / f"{prefix}_3d_pointcloud.png",
        scene_colors=render_colors,
        camera_json_path=camera_json_path,
    )
    render_3d_pointcloud(
        points_xyz_m,
        cad_mesh,
        poses_4x4,
        out_dir / f"{prefix}_3d_raw_pointcloud.png",
        scene_colors=point_colors,
        camera_json_path=camera_json_path,
    )

    meta = {
        "dataset": "itodd_native_3d_long_baseline",
        "model": model_name,
        "scene_id": scene_id,
        "runtime_ms": match_res.runtime_ms,
        "match_count": len(poses_4x4),
        "scores": list(match_res.scores),
        "poses": [p.tolist() for p in poses_4x4],
        "params": params,
    }
    (out_dir / f"{prefix}_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[+] Saved visualization metadata to {out_dir / f'{prefix}_meta.json'}")


def load_scene_ids_from_file(file_path: Path) -> list[int]:
    """Parse scene IDs from a scene list file, ignoring comments and empty lines."""
    scene_ids = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            match = re.search(r"scene_(\d+)", line)
            if match:
                scene_ids.append(int(match.group(1)))
            else:
                try:
                    scene_ids.append(int(line))
                except ValueError:
                    continue
    return scene_ids


def resolve_camera_json(camera_path: Path | None) -> Path | None:
    """Resolve Open3D camera parameter JSON file path across common working directories."""
    if camera_path is not None:
        if camera_path.exists():
            return camera_path
        alt = Path(__file__).parent / camera_path
        if alt.exists():
            return alt
        return camera_path

    candidates = [
        Path.cwd() / "ScreenCamera_2026-09-05-17-46-04.json",
        Path(__file__).parent / "ScreenCamera_2026-09-05-17-46-04.json",
        Path("code/hpo-3d-match/ScreenCamera_2026-09-05-17-46-04.json"),
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize surface matching on BOP or ITODD Native scenes.")
    parser.add_argument("--model", required=True, choices=["bracket_planar", "screw_black", "star"])
    parser.add_argument("--dataset-type", choices=["bop", "native"], default="native", help="Dataset format: bop or native")
    parser.add_argument("--storage-dir", type=Path, help="Directory containing study .db to extract optimal parameters")
    parser.add_argument("--scene-list", type=Path, help="Path to scene list text file (default: data/base_package/models/scene_lists/scene_list_<model>.txt)")
    parser.add_argument("--scenes", type=str, help="Explicit comma-separated scene IDs to visualize (overrides --scene-list)")
    parser.add_argument("--max-scenes", type=int, default=0, help="Max scenes to process from list (default: 0 for all scenes, set >0 to limit)")
    parser.add_argument("--show-full-scene", action="store_true", help="Render full scene point cloud instead of tabletop-filtered scene")
    parser.add_argument("--out-dir", type=Path, default=Path("visualizations/results"))
    parser.add_argument("--sequential-subtraction", "--peeling", dest="sequential_subtraction", action="store_true", default=False, help="Enable recursive point cloud subtraction matching")
    parser.add_argument("--interactive", dest="interactive", action="store_true", default=False, help="Enable step-by-step Open3D interactive visualization (press 'q' to proceed)")
    parser.add_argument("--min-score", type=float, default=None, help="Minimum matching score threshold (default: study params or 0.50 for sequential subtraction, 0.01 for standard)")
    parser.add_argument("--subtraction-radius", type=float, default=0.005, help="Point cloud peeling radius in metres (default: 0.005 = 5mm)")
    parser.add_argument("--max-iterations", type=int, default=10, help="Maximum iterations per scene for sequential subtraction (default: 10)")
    parser.add_argument(
        "--camera-json",
        type=Path,
        default=None,
        help="Path to Open3D camera parameter JSON (default: ScreenCamera_2026-09-05-17-46-04.json if found)",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    params = get_best_params_from_study(args.storage_dir, args.model) if args.storage_dir else dict(DEFAULT_PARAMS)
    if args.min_score is not None:
        min_score = float(args.min_score)
        params["min_score"] = min_score
        print(f"[+] Setting min_score from CLI: {min_score:.4f}")
    else:
        default_val = 0.50 if args.sequential_subtraction else 0.01
        min_score = float(params.get("min_score", default_val))
        params["min_score"] = min_score

    camera_json_path = resolve_camera_json(args.camera_json)
    if camera_json_path and camera_json_path.exists():
        print(f"[+] Using camera viewpoint parameters from: {camera_json_path}")
    else:
        if args.camera_json is not None:
            print(f"[!] Warning: Specified camera JSON not found: {args.camera_json}")

    if args.dataset_type == "native":
        if args.scenes:
            scene_ids = [int(s.strip()) for s in args.scenes.split(",") if s.strip()]
        else:
            list_file = args.scene_list or (Path("data/base_package/models/scene_lists") / f"scene_list_{args.model}.txt")
            if not list_file.exists():
                raise FileNotFoundError(f"Scene list file not found: {list_file}")
            print(f"[+] Loading native scenes from: {list_file}")
            scene_ids = load_scene_ids_from_file(list_file)
            if args.max_scenes > 0:
                scene_ids = scene_ids[:args.max_scenes]

        print(f"[*] Processing {len(scene_ids)} native scenes: {scene_ids}")
        for scene_id in scene_ids:
            try:
                visualize_native_scene(
                    args.model,
                    scene_id,
                    params,
                    args.out_dir,
                    show_full_scene=args.show_full_scene,
                    sequential_subtraction=args.sequential_subtraction,
                    interactive=args.interactive,
                    min_score=min_score,
                    subtraction_radius=args.subtraction_radius,
                    max_iterations=args.max_iterations,
                    camera_json_path=camera_json_path,
                )
            except Exception as e:
                print(f"[!] Error processing native scene {scene_id}: {e}")
    else:
        # For BOP itoddmv_val
        manifest = Path("data/manifests/itoddmv_val/bop_manifest.csv")
        # Sample images from the manifest
        im_ids = {"bracket_planar": [450, 468], "screw_black": [293, 296], "star": [0, 3]}
        selected_ims = im_ids.get(args.model, [0])
        for im_id in selected_ims:
            try:
                visualize_bop_scene(
                    args.model,
                    manifest,
                    scene_id=1,
                    image_id=im_id,
                    params=params,
                    out_dir=args.out_dir,
                    show_full_scene=args.show_full_scene,
                    sequential_subtraction=args.sequential_subtraction,
                    interactive=args.interactive,
                    min_score=min_score,
                    subtraction_radius=args.subtraction_radius,
                    max_iterations=args.max_iterations,
                    camera_json_path=camera_json_path,
                )
            except Exception as e:
                print(f"[!] Error processing BOP scene: {e}")

    print(f"\n[+] Visualization complete! All artifacts saved to {args.out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
