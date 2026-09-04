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
import sys
from pathlib import Path
from typing import Any, Mapping

import cv2
import halcon as ha
import imageio.v3 as iio
import numpy as np
import open3d as o3d

from bop_scene_loader import (
    BOPCamera,
    backproject_depth,
    create_halcon_point_cloud,
    read_bop_camera,
    read_depth_image,
)
from config import DEFAULT_PARAMS, ROIConfig, TARGET_OBJECT_IDS
from dataset import ITODD_EXTERNAL_MANIFEST_FIELDS, TARGET_CAD_FILES
from external_pipeline import (
    load_original_cad,
    load_xyz_point_cloud,
    read_external_queries,
    transform_roi_pose_to_scene,
)
from matcher import MatchError, SurfaceMatcher
from pipeline import _load_model, surface_matching_config
from scene_loader import compute_roi_transform, filter_scene_roi


def get_best_params_from_study(storage_dir: Path, model_name: str) -> dict[str, Any]:
    """Read the best parameter set from an Optuna study DB or JSONL."""
    import optuna

    db_path = storage_dir / f"{model_name}.db"
    if db_path.exists():
        try:
            storage_url = f"sqlite:///{db_path.resolve()}"
            studies = optuna.get_all_study_summaries(storage=storage_url)
            if studies:
                study = optuna.load_study(storage=storage_url, study_name=studies[0].study_name)
                print(f"[+] Loaded best params from Optuna study: {study.study_name} (Value: {study.best_value})")
                return study.best_params
        except Exception as exc:
            print(f"[!] Warning reading Optuna DB: {exc}")

    # Fallback: check jsonl studies
    jsonl_files = list(storage_dir.glob("*.jsonl"))
    if jsonl_files:
        best_file = max(jsonl_files, key=os.path.getctime)
        print(f"[+] Reading params from journal: {best_file.name}")
        # Return DEFAULT_PARAMS if parsing jsonl is not needed
    print("[*] Using default parameters.")
    return dict(DEFAULT_PARAMS)


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
) -> None:
    """Render 3D point cloud with textured sensor points and CAD models in 3D perspective."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(scene_points_m)
    
    if scene_colors is not None and len(scene_colors) == len(scene_points_m):
        pcd.colors = o3d.utility.Vector3dVector(scene_colors)
    else:
        pcd.paint_uniform_color([0.35, 0.38, 0.45])

    vis_geoms = [pcd]
    colors = [
        [0.0, 0.90, 0.35],   # Neon Green
        [1.0, 0.65, 0.0],    # Amber Orange
        [0.1, 0.65, 1.0],    # Electric Blue
        [0.95, 0.2, 0.85],   # Vivid Magenta
        [0.2, 0.95, 0.9],    # Bright Cyan
    ]

    for idx, pose in enumerate(poses):
        inst_mesh = copy.deepcopy(cad_mesh)
        inst_mesh.transform(pose)
        inst_mesh.compute_vertex_normals()  # 3D shading highlights
        inst_mesh.paint_uniform_color(colors[idx % len(colors)])
        vis_geoms.append(inst_mesh)

    vis = o3d.visualization.Visualizer()
    # 4K Ultra-HD resolution buffer
    vis.create_window(visible=False, width=3840, height=2160)
    for g in vis_geoms:
        vis.add_geometry(g)

    # Render options for publication quality
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.99, 0.99, 0.99])  # Clean white background
    opt.point_size = 2.8
    opt.light_on = True

    # Elevated 3D oblique perspective: global bounding box focus
    ctr = vis.get_view_control()
    min_b = pcd.get_min_bound()
    max_b = pcd.get_max_bound()
    for g in vis_geoms[1:]:
        min_b = np.minimum(min_b, g.get_min_bound())
        max_b = np.maximum(max_b, g.get_max_bound())
    global_center = (min_b + max_b) / 2.0

    ctr.set_lookat(global_center)
    ctr.set_front([0.32, -0.42, -0.85])
    ctr.set_up([0.12, -0.90, 0.42])
    ctr.set_zoom(0.58)  # Optimal zoom ensuring full containment without clipping
    
    vis.poll_events()
    vis.update_renderer()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vis.capture_screen_image(str(out_path))
    vis.destroy_window()

    # Auto-crop surrounding blank white margins with safe padding
    img_bgr = cv2.imread(str(out_path))
    if img_bgr is not None:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        mask = gray < 248
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

    print(f"[+] Saved 4K Ultra-HD 3D point cloud visualization to {out_path}")


def visualize_bop_scene(
    model_name: str,
    manifest_csv: Path,
    scene_id: int,
    image_id: int,
    params: dict[str, Any],
    out_dir: Path,
    show_full_scene: bool = False,
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
    point_colors = np.column_stack([pt_gray * 0.70, pt_gray * 0.75, pt_gray * 0.82])

    # CAD mesh for Open3D
    cad_mesh = o3d.io.read_triangle_mesh(str(cad_path))
    cad_mesh.compute_vertex_normals()
    if np.asarray(cad_mesh.vertices).max() > 10.0:  # If in mm, convert to metres
        cad_mesh.scale(0.001, center=(0, 0, 0))

    # 3D ROI Box Filter (strip tabletop and background points in camera frame)
    roi = ROIConfig()
    roi_mat = np.array(roi.matrix, dtype=np.float64)
    inv_roi_mat = np.linalg.inv(roi_mat)
    pts_h = np.column_stack([points_xyz_m, np.ones(len(points_xyz_m))])
    pts_roi = (inv_roi_mat @ pts_h.T).T[:, :3]
    z_min, z_max = roi.get_z_range(is_bop=True)
    in_roi = (
        (pts_roi[:, 0] >= roi.x_range[0])
        & (pts_roi[:, 0] <= roi.x_range[1])
        & (pts_roi[:, 1] >= roi.y_range[0])
        & (pts_roi[:, 1] <= roi.y_range[1])
        & (pts_roi[:, 2] >= z_min)
        & (pts_roi[:, 2] <= z_max)
    )

    # Match using HALCON on ROI-filtered points (or full scene if requested)
    match_pts = points_xyz_m if show_full_scene else points_xyz_m[in_roi]
    halcon_scene = create_halcon_point_cloud(match_pts)
    halcon_model = _load_model(cad_path, "mm")
    config = surface_matching_config(params, timeout_sec=5.0, min_score=0.01, num_matches=10)

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
    render_pts = points_xyz_m if show_full_scene else points_xyz_m[in_roi]
    render_colors = point_colors if show_full_scene else point_colors[in_roi]
    render_3d_pointcloud(render_pts, cad_mesh, poses_4x4, out_dir / f"{prefix}_3d_pointcloud.png", scene_colors=render_colors)

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
    point_colors = np.column_stack([l_norm * 0.70, l_norm * 0.75, l_norm * 0.82])

    # HALCON matching with ROI
    roi = ROIConfig()
    roi_mat, roi_pose_inv = compute_roi_transform(roi)
    images = [ha.read_image(str(x_path)), ha.read_image(str(y_path)), ha.read_image(str(z_path))]
    scene_3d = ha.xyz_to_object_model_3d(*images)
    scene_roi = filter_scene_roi(scene_3d, roi_pose_inv, roi)

    model_point_cloud = load_original_cad(cad_path)
    config = surface_matching_config(params, timeout_sec=5.0, min_score=0.01, num_matches=10)

    with SurfaceMatcher(model_point_cloud, config) as matcher:
        match_res = matcher.match(scene_roi)

    predictions = tuple(
        transform_roi_pose_to_scene(pose, roi_mat) for pose in match_res.predictions
    )
    
    poses_4x4 = []
    for pred in predictions:
        mat = np.eye(4)
        mat[:3, :3] = pred.rotation
        mat[:3, 3] = pred.translation_mm / 1000.0
        poses_4x4.append(mat)

    print(f"[*] Found {len(poses_4x4)} matches in {match_res.runtime_ms:.1f}ms")

    # CAD mesh for Open3D
    cad_mesh = o3d.io.read_triangle_mesh(str(cad_path))
    cad_mesh.compute_vertex_normals()

    # Approximate default camera K for ITODD sensor
    cam_k = np.array([[2900.0, 0.0, img_2d.shape[1] / 2.0], [0.0, 2900.0, img_2d.shape[0] / 2.0], [0.0, 0.0, 1.0]])

    # Filter 3D points by ROI for 3D visualization
    inv_roi_mat = np.linalg.inv(roi_mat)
    pts_h = np.column_stack([points_xyz_m, np.ones(len(points_xyz_m))])
    pts_roi = (inv_roi_mat @ pts_h.T).T[:, :3]
    in_roi = (
        (pts_roi[:, 0] >= roi.x_range[0]) & (pts_roi[:, 0] <= roi.x_range[1]) &
        (pts_roi[:, 1] >= roi.y_range[0]) & (pts_roi[:, 1] <= roi.y_range[1]) &
        (pts_roi[:, 2] >= roi.z_range[0]) & (pts_roi[:, 2] <= roi.z_range[1])
    )
    render_pts = points_xyz_m if show_full_scene else points_xyz_m[in_roi]
    render_colors = point_colors if show_full_scene else point_colors[in_roi]

    prefix = f"native_{model_name}_scene{scene_id:04d}"
    render_2d_overlay(img_2d, cad_mesh, poses_4x4, cam_k, out_dir / f"{prefix}_2d_overlay.png", match_res.scores)
    render_3d_pointcloud(render_pts, cad_mesh, poses_4x4, out_dir / f"{prefix}_3d_pointcloud.png", scene_colors=render_colors)

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
    print(f"[+] Saved metadata to {out_dir / f'{prefix}_meta.json'}")


def load_scene_ids_from_file(file_path: Path) -> list[int]:
    """Parse scene IDs from a scene list file, ignoring comments and empty lines."""
    scene_ids = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                scene_ids.append(int(line))
            except ValueError:
                continue
    return scene_ids


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize surface matching on BOP or ITODD Native scenes.")
    parser.add_argument("--model", required=True, choices=["bracket_planar", "screw_black", "star"])
    parser.add_argument("--dataset-type", choices=["bop", "native"], default="native", help="Dataset format: bop or native")
    parser.add_argument("--storage-dir", type=Path, help="Directory containing study .db to extract optimal parameters")
    parser.add_argument("--scene-list", type=Path, help="Path to scene list text file (default: data/base_package/models/scene_lists/scene_list_<model>.txt)")
    parser.add_argument("--scenes", type=str, help="Explicit comma-separated scene IDs to visualize (overrides --scene-list)")
    parser.add_argument("--max-scenes", type=int, default=5, help="Max scenes to process from list (default: 5, set <=0 for all)")
    parser.add_argument("--show-full-scene", action="store_true", help="Render uncropped full scene point cloud instead of cropped ROI")
    parser.add_argument("--out-dir", type=Path, default=Path("visualizations/results"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    params = get_best_params_from_study(args.storage_dir, args.model) if args.storage_dir else dict(DEFAULT_PARAMS)

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
                visualize_native_scene(args.model, scene_id, params, args.out_dir, show_full_scene=args.show_full_scene)
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
                )
            except Exception as e:
                print(f"[!] Error processing BOP scene: {e}")

    print(f"\n[+] Visualization complete! All artifacts saved to {args.out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
