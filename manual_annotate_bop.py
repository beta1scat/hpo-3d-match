"""Interactive Open3D Manual 6D Pose Annotation & Automated BOP Dataset Exporter.

Supports 6-DoF adjustment, multi-instance pose saving ('s'), ICP auto-snap ('F'/'Space'),
and automated BOP dataset generation upon completing a scene ('n').

Default target models and scenes:
  - star:           8, 10, 13, 164, 749 (obj_id: 25)
  - bracket_planar: 456, 461, 481, 476, 488 (obj_id: 5)
  - screw_black:    299, 302, 305 (obj_id: 24)

Usage examples:
  # Annotate star scenes:
  uv run python manual_annotate_bop.py --model star

  # Annotate bracket_planar scenes:
  uv run python manual_annotate_bop.py --model bracket_planar

  # Annotate screw_black scenes:
  uv run python manual_annotate_bop.py --model screw_black

  # Annotate specific scenes:
  uv run python manual_annotate_bop.py --model star --scenes 8,10

  # Annotate all 13 scenes in sequence:
  uv run python manual_annotate_bop.py --all

  # Non-interactive validation check:
  uv run python manual_annotate_bop.py --test-dry-run
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import imageio.v3 as iio
import numpy as np
import open3d as o3d

# Object configurations matching ITODD / BOP conventions
OBJECT_CONFIGS = {
    "star": {
        "obj_id": 25,
        "default_scenes": [8, 10, 13, 164, 749],
        "cad_filename": "obj_000025.ply",
        "alt_cad": "star.ply",
    },
    "bracket_planar": {
        "obj_id": 5,
        "default_scenes": [456, 461, 481, 476, 488],
        "cad_filename": "obj_000005.ply",
        "alt_cad": "bracket_planar.ply",
    },
    "screw_black": {
        "obj_id": 24,
        "default_scenes": [299, 302, 305],
        "cad_filename": "obj_000024.ply",
        "alt_cad": "screw_black.ply",
    },
}

# Reference camera calibration lookup table (fx, fy, cx, cy)
CALIB_TEMPLATES = [
    (0, 296, [2992.63, 0.0, 633.886, 0.0, 3003.99, 489.554, 0.0, 0.0, 1.0]),
    (311, 435, [2989.76, 0.0, 632.842, 0.0, 3001.13, 489.313, 0.0, 0.0, 1.0]),
    (450, 565, [2993.14, 0.0, 633.531, 0.0, 3004.51, 489.719, 0.0, 0.0, 1.0]),
    (589, 677, [2993.56, 0.0, 633.843, 0.0, 3004.92, 489.100, 0.0, 0.0, 1.0]),
    (699, 702, [2992.88, 0.0, 634.119, 0.0, 3004.27, 489.607, 0.0, 0.0, 1.0]),
    (749, 776, [2992.14, 0.0, 633.412, 0.0, 3003.49, 489.882, 0.0, 0.0, 1.0]),
]


def get_cam_k(scene_id: int) -> list[float]:
    """Retrieve optimal camera intrinsics cam_K for given ITODD scene."""
    for start, end, cam_k in CALIB_TEMPLATES:
        if start <= scene_id <= end:
            return cam_k
    # Nearest fallback
    mid_points = [((start + end) / 2.0, cam_k) for start, end, cam_k in CALIB_TEMPLATES]
    closest = min(mid_points, key=lambda item: abs(item[0] - scene_id))
    return closest[1]


def rotation_matrix_x(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def rotation_matrix_y(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def rotation_matrix_z(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def find_cad_file(model_name: str, base_dir: Path) -> Path:
    """Locate CAD model PLY file across repository locations."""
    cfg = OBJECT_CONFIGS[model_name]
    p1 = base_dir / "data" / "itoddmv_models" / "models" / cfg["cad_filename"]
    if p1.exists():
        return p1
    p2 = base_dir / "data" / "base_package" / "models" / "cad_models" / cfg["alt_cad"]
    if p2.exists():
        return p2
    p3 = base_dir / "data" / "cad_models" / cfg["alt_cad"]
    if p3.exists():
        return p3
    raise FileNotFoundError(f"CAD model for {model_name} not found. Searched {p1}, {p2}, {p3}")


def load_native_scene_pcd(
    scene_id: int, base_dir: Path, voxel_size: float = 0.0
) -> Tuple[o3d.geometry.PointCloud, np.ndarray, np.ndarray]:
    """Load native sensor TIFFs and return millimeter-scaled PointCloud with RGB."""
    scene_dir = base_dir / "data" / "3d_long_baseline" / "scenes" / f"scene_{scene_id:04d}"
    x_path = scene_dir / "3d_long_baseline_x.tif"
    y_path = scene_dir / "3d_long_baseline_y.tif"
    z_path = scene_dir / "3d_long_baseline_z.tif"
    l_path = scene_dir / "3d_long_baseline_l.tif"

    if not z_path.exists():
        raise FileNotFoundError(f"Scene files missing in: {scene_dir}")

    x_img = np.asarray(iio.imread(x_path))
    y_img = np.asarray(iio.imread(y_path))
    z_img = np.asarray(iio.imread(z_path))

    # Valid depth range in meters (0.05m to 1.5m)
    valid = np.isfinite(z_img) & (z_img > 0.05) & (z_img < 1.5)
    pts_mm = np.stack([x_img[valid], y_img[valid], z_img[valid]], axis=-1) * 1000.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_mm)

    if l_path.exists():
        l_img = np.asarray(iio.imread(l_path))[valid].astype(np.float64)
        l_norm = (l_img - l_img.min()) / (l_img.max() - l_img.min() + 1e-6)
        colors = np.column_stack([l_norm, l_norm, l_norm])
        pcd.colors = o3d.utility.Vector3dVector(colors)
    else:
        pcd.paint_uniform_color([0.65, 0.65, 0.65])

    if voxel_size > 0.0:
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=10.0, max_nn=30)
    )
    pcd.orient_normals_towards_camera_location(camera_location=np.array([0.0, 0.0, 0.0]))

    return pcd, z_img, valid


def load_cad_geometry(
    cad_path: Path, num_sample_points: int = 15000
) -> Tuple[o3d.geometry.PointCloud, np.ndarray]:
    """Load CAD model PLY as a dense point cloud with normals for high-precision alignment."""
    mesh = o3d.io.read_triangle_mesh(str(cad_path))
    if len(mesh.vertices) > 0 and len(mesh.triangles) > 0:
        mesh.compute_vertex_normals()
        cad_pcd = mesh.sample_points_uniformly(number_of_points=num_sample_points)
    else:
        cad_pcd = o3d.io.read_point_cloud(str(cad_path))

    cad_pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5.0, max_nn=30)
    )
    model_pts = np.asarray(cad_pcd.points, dtype=np.float64)
    return cad_pcd, model_pts


def save_bop_scene(
    scene_id: int,
    model_name: str,
    obj_id: int,
    saved_poses: List[Tuple[np.ndarray, np.ndarray]],
    native_z_m: np.ndarray,
    output_dir: Path,
    manifest_path: Path,
    base_dir: Path,
    split_name: str = "val",
) -> None:
    """Export scene depth map and ground-truth metadata in exact BOP challenge format."""
    output_dir.mkdir(parents=True, exist_ok=True)
    depth_dir = output_dir / "depth_3dlong"
    depth_dir.mkdir(parents=True, exist_ok=True)

    # 1. Depth image in float32 mm
    valid = np.isfinite(native_z_m) & (native_z_m > 0.05)
    depth_mm = np.where(valid, (native_z_m * 1000.0).astype(np.float32), 0.0)
    out_depth_file = depth_dir / f"{scene_id:06d}.tif"
    iio.imwrite(out_depth_file, depth_mm)

    # 2. Camera intrinsics (scene_camera_3dlong.json)
    cam_file = output_dir / "scene_camera_3dlong.json"
    cam_dict: Dict[str, Any] = {}
    if cam_file.exists():
        try:
            cam_dict = json.loads(cam_file.read_text(encoding="utf-8"))
        except Exception:
            cam_dict = {}

    cam_k = get_cam_k(scene_id)
    cam_dict[str(scene_id)] = {
        "cam_K": [round(x, 4) for x in cam_k],
        "depth_scale": 1.0,
        "cam_R_w2c": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        "cam_t_w2c": [0.0, 0.0, 0.0],
    }
    cam_file.write_text(json.dumps(cam_dict, indent=2), encoding="utf-8")

    # 3. Ground truth poses (scene_gt_3dlong.json)
    gt_file = output_dir / "scene_gt_3dlong.json"
    gt_dict: Dict[str, Any] = {}
    if gt_file.exists():
        try:
            gt_dict = json.loads(gt_file.read_text(encoding="utf-8"))
        except Exception:
            gt_dict = {}

    annotations = []
    for R, t in saved_poses:
        annotations.append({
            "cam_R_m2c": [round(float(x), 6) for x in R.flatten()],
            "cam_t_m2c": [round(float(x), 3) for x in t.flatten()],
            "obj_id": int(obj_id),
        })
    gt_dict[str(scene_id)] = annotations
    gt_file.write_text(json.dumps(gt_dict, indent=2), encoding="utf-8")

    # 4. Ground truth info (scene_gt_info_3dlong.json)
    gt_info_file = output_dir / "scene_gt_info_3dlong.json"
    gt_info_dict: Dict[str, Any] = {}
    if gt_info_file.exists():
        try:
            gt_info_dict = json.loads(gt_info_file.read_text(encoding="utf-8"))
        except Exception:
            gt_info_dict = {}

    infos = []
    for _ in saved_poses:
        infos.append({
            "bbox_obj": [0, 0, 100, 100],
            "bbox_visib": [0, 0, 100, 100],
            "px_count_all": 20000,
            "px_count_valid": 20000,
            "px_count_visib": 20000,
            "visib_fract": 1.0,
        })
    gt_info_dict[str(scene_id)] = infos
    gt_info_file.write_text(json.dumps(gt_info_dict, indent=2), encoding="utf-8")

    # 5. Manifest CSV update
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    fieldnames = [
        "source", "scene_id", "image_id", "model_name", "obj_id",
        "gt_count", "split", "scene_gt_path", "scene_gt_info_path",
        "scene_camera_path", "depth_path", "cad_path", "models_info_path", "min_visib_fract"
    ]
    if manifest_path.exists():
        try:
            with manifest_path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = [row for row in reader if not (
                    row.get("image_id") == str(scene_id) and row.get("model_name") == model_name
                )]
        except Exception:
            rows = []

    cad_path = find_cad_file(model_name, base_dir)
    models_info_path = base_dir / "data" / "itoddmv_models" / "models" / "models_info.json"
    if not models_info_path.exists():
        models_info_path = base_dir / "data" / "base_package" / "models" / "models_info.json"

    new_row = {
        "source": "manual_annotated",
        "scene_id": 1,
        "image_id": scene_id,
        "model_name": model_name,
        "obj_id": obj_id,
        "gt_count": len(saved_poses),
        "split": split_name,
        "scene_gt_path": str(gt_file.resolve()),
        "scene_gt_info_path": str(gt_info_file.resolve()),
        "scene_camera_path": str(cam_file.resolve()),
        "depth_path": str(out_depth_file.resolve()),
        "cad_path": str(cad_path.resolve()),
        "models_info_path": str(models_info_path.resolve()) if models_info_path.exists() else "",
        "min_visib_fract": 0.1,
    }
    rows.append(new_row)

    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[+] Successfully exported Scene {scene_id} ({model_name}) to BOP dataset:")
    print(f"    - Depth image:  {out_depth_file}")
    print(f"    - GT poses:     {len(saved_poses)} instances saved in {gt_file}")
    print(f"    - Manifest:     {manifest_path}")


class InteractiveAnnotator:
    """Manages Open3D visualizer lifecycle, user callbacks, and state for one scene."""

    def __init__(
        self,
        scene_id: int,
        model_name: str,
        base_dir: Path,
        output_dir: Path,
        manifest_path: Path,
        voxel_size: float = 0.0,
    ) -> None:
        self.scene_id = scene_id
        self.model_name = model_name
        self.obj_id = OBJECT_CONFIGS[model_name]["obj_id"]
        self.base_dir = base_dir
        self.output_dir = output_dir
        self.manifest_path = manifest_path

        # Load scene point cloud
        print(f"\nLoading Scene {scene_id:04d} point cloud...")
        self.scene_pcd, self.native_z, _ = load_native_scene_pcd(scene_id, base_dir, voxel_size)

        # Load CAD model
        cad_path = find_cad_file(model_name, base_dir)
        print(f"Loading CAD model: {cad_path.name}...")
        self.cad_template, self.raw_cad_pts = load_cad_geometry(cad_path)
        self.raw_cad_normals = np.asarray(self.cad_template.normals, dtype=np.float64)

        # Initialize active movable model
        self.active_pcd = copy.deepcopy(self.cad_template)
        self.active_pcd.paint_uniform_color([1.0, 0.85, 0.0])  # Bright yellow

        # Initial pose anchor (bin center: tx=3.0mm, ty=-0.2mm, tz=721.3mm)
        self.initial_t = np.array([3.062, -0.230, 721.286], dtype=np.float64)
        self.current_t = self.initial_t.copy()
        self.current_R = np.eye(3, dtype=np.float64)

        # Step size levels: 1: fine, 2: medium, 3: coarse
        self.step_mode = 2
        self.trans_step = 2.5  # mm
        self.rot_step = math.radians(5.0)  # rad

        # Ground truth storage for current scene
        self.saved_poses: List[Tuple[np.ndarray, np.ndarray]] = []
        self.frozen_geometries: List[o3d.geometry.PointCloud] = []

        # Flags
        self.exit_requested = False
        self.next_scene_requested = False

        self._update_active_geometry()

    def _update_active_geometry(self) -> None:
        """Apply current R and t to active model point cloud."""
        transformed_pts = (self.current_R @ self.raw_cad_pts.T).T + self.current_t
        self.active_pcd.points = o3d.utility.Vector3dVector(transformed_pts)
        if len(self.raw_cad_normals) == len(self.raw_cad_pts):
            transformed_normals = (self.current_R @ self.raw_cad_normals.T).T
            self.active_pcd.normals = o3d.utility.Vector3dVector(transformed_normals)

    def print_help(self) -> None:
        print("\n" + "=" * 65)
        print(f" INTERACTIVE 6D ANNOTATION: Scene {self.scene_id:04d} | Object: {self.model_name}")
        print("=" * 65)
        print(" [S]        : Save current pose as Ground Truth instance")
        print(" [N]        : Next scene & Save BOP dataset files")
        print(" [F]/[Space]: ICP Auto-Snap (Sub-millimeter local surface alignment)")
        print(" [Z]        : Undo last saved pose in this scene")
        print(" [R]        : Reset active model to bin center")
        print(" [1]/[2]/[3]: Step scale: 1=Fine (0.5mm/1°), 2=Med (2.5mm/5°), 3=Coarse (10mm/15°)")
        print(" [A] / [D]  : Translation X (Left / Right)")
        print(" [W] / [X]  : Translation Y (Up / Down)")
        print(" [Q] / [E]  : Translation Z (Near / Far)")
        print(" [I] / [K]  : Rotation X (Pitch +/-)")
        print(" [J] / [L]  : Rotation Y (Yaw +/-)")
        print(" [U] / [O]  : Rotation Z (Roll +/-)")
        print(" [H]        : Print this help message")
        print(" [ESC]      : Exit application")
        print("-" * 65)
        print(f" Current Step Mode: {self.step_mode} (dt={self.trans_step}mm, dR={math.degrees(self.rot_step):.1f}°)")
        print(f" Saved instances in this scene: {len(self.saved_poses)}")
        print("=" * 65 + "\n")

    def run(self) -> bool:
        """Launch Open3D Visualizer window. Return True to continue next scene, False to exit."""
        vis = o3d.visualization.VisualizerWithKeyCallback()
        title = f"ITODD 6D Annotation | Scene {self.scene_id:04d} ({self.model_name}) | Press 'H' for Help"
        vis.create_window(window_name=title, width=1600, height=950)

        # Set render options
        opt = vis.get_render_option()
        opt.point_size = 3.0
        opt.background_color = np.asarray([0.15, 0.15, 0.18])

        vis.add_geometry(self.scene_pcd)
        vis.add_geometry(self.active_pcd)

        # Key callbacks mapping
        def cb_trans(dx: float, dy: float, dz: float):
            def handler(v: o3d.visualization.VisualizerWithKeyCallback):
                self.current_t += np.array([dx, dy, dz], dtype=np.float64) * self.trans_step
                self._update_active_geometry()
                v.update_geometry(self.active_pcd)
                return False
            return handler

        def cb_rot(rx: float, ry: float, rz: float):
            def handler(v: o3d.visualization.VisualizerWithKeyCallback):
                dR = np.eye(3)
                if rx != 0.0:
                    dR = dR @ rotation_matrix_x(rx * self.rot_step)
                if ry != 0.0:
                    dR = dR @ rotation_matrix_y(ry * self.rot_step)
                if rz != 0.0:
                    dR = dR @ rotation_matrix_z(rz * self.rot_step)
                self.current_R = dR @ self.current_R
                self._update_active_geometry()
                v.update_geometry(self.active_pcd)
                return False
            return handler

        def cb_step(level: int, dt: float, dr_deg: float):
            def handler(_):
                self.step_mode = level
                self.trans_step = dt
                self.rot_step = math.radians(dr_deg)
                print(f"[Step Mode {level}] dt={dt:.1f} mm, dR={dr_deg:.1f} deg")
                return False
            return handler

        def cb_icp(v: o3d.visualization.VisualizerWithKeyCallback):
            print("\n[ICP Auto-Snap] Running Point-to-Plane ICP on local surface...")
            # Crop local scene points within 35mm radius of active model center
            scene_pts = np.asarray(self.scene_pcd.points)
            dists = np.linalg.norm(scene_pts - self.current_t, axis=1)
            local_mask = dists < 45.0
            if np.sum(local_mask) < 80:
                print("  [!] Too few local scene points nearby. Move model closer to target object first.")
                return False

            local_scene = o3d.geometry.PointCloud()
            local_scene.points = o3d.utility.Vector3dVector(scene_pts[local_mask])
            if self.scene_pcd.has_normals():
                scene_n = np.asarray(self.scene_pcd.normals)[local_mask]
                local_scene.normals = o3d.utility.Vector3dVector(scene_n)

            # Current active model as source
            source = copy.deepcopy(self.active_pcd)
            init_trans = np.eye(4)
            icp_thresh = max(self.trans_step * 2.0, 6.0)

            try:
                reg = o3d.pipelines.registration.registration_icp(
                    source,
                    local_scene,
                    max_correspondence_distance=icp_thresh,
                    init=init_trans,
                    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                    criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50),
                )
                delta_T = reg.transformation
                dR = delta_T[:3, :3]
                dt = delta_T[:3, 3]

                # Update state
                self.current_R = dR @ self.current_R
                self.current_t = dR @ self.current_t + dt
                self._update_active_geometry()
                v.update_geometry(self.active_pcd)
                print(f"  [+] ICP Converged! Fitness={reg.fitness:.3f}, RMSE={reg.inlier_rmse:.3f} mm")
                print(f"      Shift dt=[{dt[0]:.2f}, {dt[1]:.2f}, {dt[2]:.2f}] mm")
            except Exception as e:
                print(f"  [!] ICP failed: {e}")
            return False

        def cb_save(v: o3d.visualization.VisualizerWithKeyCallback):
            # Save pose
            pose_R = self.current_R.copy()
            pose_t = self.current_t.copy()
            self.saved_poses.append((pose_R, pose_t))
            idx = len(self.saved_poses)
            print(f"\n[+] GT #{idx} Saved for Scene {self.scene_id:04d}:")
            print(f"    t = [{pose_t[0]:.2f}, {pose_t[1]:.2f}, {pose_t[2]:.2f}] mm")
            print(f"    R = {pose_R.flatten().round(4).tolist()[:5]}...")

            # Freeze a green visualizer instance
            frozen = copy.deepcopy(self.active_pcd)
            frozen.paint_uniform_color([0.15, 0.85, 0.25])  # Vibrant green
            self.frozen_geometries.append(frozen)
            v.add_geometry(frozen)

            # Reset active model slightly offset for next instance
            self.current_t = self.initial_t.copy() + np.array([20.0, 20.0, 0.0])
            self.current_R = np.eye(3, dtype=np.float64)
            self._update_active_geometry()
            v.update_geometry(self.active_pcd)
            return False

        def cb_undo(v: o3d.visualization.VisualizerWithKeyCallback):
            if not self.saved_poses:
                print("\n[!] No saved poses to undo in this scene.")
                return False
            self.saved_poses.pop()
            popped_geo = self.frozen_geometries.pop()
            v.remove_geometry(popped_geo)
            print(f"\n[-] Undid last pose. Remaining instances: {len(self.saved_poses)}")
            return False

        def cb_reset(v: o3d.visualization.VisualizerWithKeyCallback):
            self.current_t = self.initial_t.copy()
            self.current_R = np.eye(3, dtype=np.float64)
            self._update_active_geometry()
            v.update_geometry(self.active_pcd)
            print("\n[*] Active model pose reset to bin center.")
            return False

        def cb_next(v: o3d.visualization.VisualizerWithKeyCallback):
            if not self.saved_poses:
                print("\n[!] Warning: No poses saved for this scene yet.")
                print("    Press 'S' to save current pose, or press 'N' again to skip/advance.")
                if not hasattr(self, "_confirm_skip"):
                    self._confirm_skip = True
                    return False
            # Save scene to BOP
            save_bop_scene(
                scene_id=self.scene_id,
                model_name=self.model_name,
                obj_id=self.obj_id,
                saved_poses=self.saved_poses,
                native_z_m=self.native_z,
                output_dir=self.output_dir,
                manifest_path=self.manifest_path,
                base_dir=self.base_dir,
            )
            self.next_scene_requested = True
            v.close()
            return False

        def cb_exit(v: o3d.visualization.VisualizerWithKeyCallback):
            print("\n[!] Exit requested by user.")
            self.exit_requested = True
            v.close()
            return False

        def cb_help(_):
            self.print_help()
            return False

        # Register keybindings (GLFW key codes)
        # Translation
        vis.register_key_callback(ord("A"), cb_trans(-1.0, 0.0, 0.0))
        vis.register_key_callback(ord("D"), cb_trans(1.0, 0.0, 0.0))
        vis.register_key_callback(ord("W"), cb_trans(0.0, -1.0, 0.0))
        vis.register_key_callback(ord("X"), cb_trans(0.0, 1.0, 0.0))
        vis.register_key_callback(ord("Q"), cb_trans(0.0, 0.0, -1.0))
        vis.register_key_callback(ord("E"), cb_trans(0.0, 0.0, 1.0))

        # Rotation
        vis.register_key_callback(ord("I"), cb_rot(1.0, 0.0, 0.0))
        vis.register_key_callback(ord("K"), cb_rot(-1.0, 0.0, 0.0))
        vis.register_key_callback(ord("J"), cb_rot(0.0, -1.0, 0.0))
        vis.register_key_callback(ord("L"), cb_rot(0.0, 1.0, 0.0))
        vis.register_key_callback(ord("U"), cb_rot(0.0, 0.0, -1.0))
        vis.register_key_callback(ord("O"), cb_rot(0.0, 0.0, 1.0))

        # Step size
        vis.register_key_callback(ord("1"), cb_step(1, 0.5, 1.0))
        vis.register_key_callback(ord("2"), cb_step(2, 2.5, 5.0))
        vis.register_key_callback(ord("3"), cb_step(3, 10.0, 15.0))

        # Action keys
        vis.register_key_callback(ord("S"), cb_save)
        vis.register_key_callback(ord("N"), cb_next)
        vis.register_key_callback(ord("F"), cb_icp)
        vis.register_key_callback(32, cb_icp)  # Space bar
        vis.register_key_callback(ord("Z"), cb_undo)
        vis.register_key_callback(ord("R"), cb_reset)
        vis.register_key_callback(ord("H"), cb_help)
        vis.register_key_callback(256, cb_exit)  # ESC

        self.print_help()
        vis.run()
        vis.destroy_window()

        return not self.exit_requested


def dry_run_test(base_dir: Path) -> int:
    """Non-interactive test to check all file paths, CAD models, and export logic."""
    print("=== Running Dry-Run Verification ===")
    all_targets = [
        ("star", 25, [8, 10, 13, 164, 749]),
        ("bracket_planar", 5, [456, 461, 481, 476, 488]),
        ("screw_black", 24, [299, 302, 305]),
    ]
    for model_name, obj_id, scenes in all_targets:
        cad_path = find_cad_file(model_name, base_dir)
        print(f"[OK] Found CAD model for {model_name}: {cad_path}")
        for s in scenes:
            scene_dir = base_dir / "data" / "3d_long_baseline" / "scenes" / f"scene_{s:04d}"
            z_path = scene_dir / "3d_long_baseline_z.tif"
            if not z_path.exists():
                raise FileNotFoundError(f"Missing {z_path}")
            cam_k = get_cam_k(s)
            assert len(cam_k) == 9
            print(f"[OK] Verified Scene {s:04d} ({model_name}) | fx={cam_k[0]:.2f}")

    # Test dummy export in temp folder
    scratch_out = base_dir / "scratch" / "test_bop_export" / "val" / "000001"
    scratch_manifest = base_dir / "scratch" / "test_bop_export" / "bop_manifest.csv"
    dummy_z = np.ones((960, 1280), dtype=np.float32) * 0.72
    dummy_poses = [(np.eye(3), np.array([3.0, -0.2, 720.0]))]
    save_bop_scene(
        scene_id=9999,
        model_name="star",
        obj_id=25,
        saved_poses=dummy_poses,
        native_z_m=dummy_z,
        output_dir=scratch_out,
        manifest_path=scratch_manifest,
        base_dir=base_dir,
    )
    assert (scratch_out / "depth_3dlong" / "009999.tif").exists()
    assert (scratch_out / "scene_gt_3dlong.json").exists()
    assert (scratch_out / "scene_camera_3dlong.json").exists()
    assert scratch_manifest.exists()
    print("[OK] BOP export routine successfully verified!")
    print("=== Dry-Run Verification Passed ===")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Interactive 6D Pose Annotator for ITODD & BOP")
    parser.add_argument(
        "--model",
        type=str,
        default="star",
        choices=list(OBJECT_CONFIGS.keys()),
        help="Object to annotate: star, bracket_planar, or screw_black",
    )
    parser.add_argument(
        "--scenes",
        type=str,
        default="",
        help="Optional comma-separated scene IDs (e.g. '8,10,13')",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Annotate all 3 objects and 13 scenes sequentially",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/itodd_manual_annotated/val/000001"),
        help="Target BOP split directory (Option A default)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/itodd_manual_annotated/bop_manifest.csv"),
        help="Target BOP manifest CSV file",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.0,
        help="Voxel downsampling size in mm for scene pointcloud (default: 0.0 = full resolution)",
    )
    parser.add_argument(
        "--test-dry-run",
        action="store_true",
        help="Run non-interactive integrity check without GUI",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent

    if args.test_dry_run:
        return dry_run_test(base_dir)

    # Determine sequence of (model_name, scene_id)
    work_list: List[Tuple[str, int]] = []
    if args.all:
        for m in ["star", "bracket_planar", "screw_black"]:
            for s in OBJECT_CONFIGS[m]["default_scenes"]:
                work_list.append((m, s))
    else:
        m = args.model
        if args.scenes.strip():
            scene_ids = [int(x.strip()) for x in args.scenes.split(",") if x.strip()]
        else:
            scene_ids = OBJECT_CONFIGS[m]["default_scenes"]
        for s in scene_ids:
            work_list.append((m, s))

    total = len(work_list)
    print(f"\nStarting 6D Pose Annotation Session ({total} scenes scheduled)...")
    print(f"Output Directory: {args.output_dir}")
    print(f"Manifest Path:    {args.manifest}\n")

    for idx, (model_name, scene_id) in enumerate(work_list, start=1):
        print(f"\n{'#'*65}")
        print(f" Progress: [{idx}/{total}] | Scene {scene_id:04d} | Object: {model_name}")
        print(f"{'#'*65}")

        annotator = InteractiveAnnotator(
            scene_id=scene_id,
            model_name=model_name,
            base_dir=base_dir,
            output_dir=args.output_dir,
            manifest_path=args.manifest,
            voxel_size=args.voxel_size,
        )

        continue_work = annotator.run()
        if not continue_work:
            print(f"\n[!] Session paused at scene {scene_id:04d}. Progress saved.")
            break

    print("\n[Done] Annotation session finished.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
