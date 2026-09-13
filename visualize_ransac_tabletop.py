"""Dynamic RANSAC Tabletop Plane Removal and Interactive 3D Visualizer.

Segments and removes the tabletop/bin floor point cloud in real time using RANSAC.
Allows dynamically tuning the distance threshold with hotkeys (+/- or Arrow Keys)
and toggling tabletop visibility (Show in Red / Completely Removed / Muted Gray).

Key controls:
  - [+] / [=] / [Up]    : Increase RANSAC threshold by +0.2 mm
  - [-] / [_] / [Down]  : Decrease RANSAC threshold by -0.2 mm
  - []] / [Right]       : Coarse increase threshold by +1.0 mm
  - [[] / [Left]        : Coarse decrease threshold by -1.0 mm
  - [1] ~ [9]           : Quickly set threshold to 1.0, 2.0, ..., 9.0 mm
  - [T]                 : Toggle Tabletop display (Red -> Completely Removed -> Muted Gray)
  - [G]                 : Toggle Ground Truth CAD models overlay
  - [C]                 : Capture high-resolution screenshot (auto-cropped white margin)
  - [V]                 : Reset camera perspective to canonical oblique / Camera JSON view
  - [S]                 : Save current filtered object point cloud to PLY
  - [R]                 : Reset threshold to default (2.0 mm)
  - [N] / [Space]       : Advance to next scene
  - [P]                 : Go to previous scene
  - [H]                 : Print detailed terminal help
  - [ESC]               : Exit

Usage examples:
  # Inspect star scene 8 with dynamic RANSAC tabletop removal:
  uv run python visualize_ransac_tabletop.py --scene-id 8 --model star

  # Inspect scene 299 (screw_black) with GT models:
  uv run python visualize_ransac_tabletop.py --scene-id 299 --model screw_black --show-gt

  # Sequentially browse all star scenes:
  uv run python visualize_ransac_tabletop.py --model star
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import imageio.v3 as iio
import numpy as np
import open3d as o3d

INSTANCE_COLORS = [
    [0.0, 0.90, 0.35],   # Neon Green
    [1.0, 0.65, 0.0],    # Amber Orange
    [0.1, 0.65, 1.0],    # Electric Blue
    [0.95, 0.2, 0.85],   # Vivid Magenta
    [0.2, 0.95, 0.9],    # Bright Cyan
]


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

OBJECT_CONFIGS = {
    "star": {
        "obj_id": 25,
        "cad_filename": "obj_000025.ply",
        "alt_cad": "star.ply",
        "default_scenes": [0, 3, 8, 10, 13, 164, 749],
        "default_thresh": 1.0,
    },
    "bracket_planar": {
        "obj_id": 5,
        "cad_filename": "obj_000005.ply",
        "alt_cad": "bracket_planar.ply",
        "default_scenes": [450, 456, 461, 468, 476, 481, 488],
        "default_thresh": 0.8,
    },
    "screw_black": {
        "obj_id": 24,
        "cad_filename": "obj_000024.ply",
        "alt_cad": "screw_black.ply",
        "default_scenes": [293, 296, 299, 302, 305],
        "default_thresh": 4.0,
    },
}

OBJ_ID_TO_MODEL = {25: "star", 5: "bracket_planar", 24: "screw_black"}


def find_cad_file(model_name: str, base_dir: Path) -> Path:
    cfg = OBJECT_CONFIGS.get(model_name)
    if not cfg:
        for name, c in OBJECT_CONFIGS.items():
            if name == model_name:
                cfg = c
                break
    if not cfg:
        raise ValueError(f"Unknown model name: {model_name}")

    candidates = [
        base_dir / "data" / "itoddmv_models" / "models" / cfg["cad_filename"],
        base_dir / "data" / "base_package" / "models" / "cad_models" / cfg["alt_cad"],
        base_dir / "data" / "cad_models" / cfg["alt_cad"],
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"CAD model for {model_name} not found.")


def load_scene_points(
    scene_id: int, base_dir: Path, source: str = "auto"
) -> Tuple[np.ndarray, np.ndarray]:
    """Load scene point cloud (XYZ in mm) and normalized RGB colors."""
    manual_depth = base_dir / "data" / "itodd_manual_annotated" / "val" / "000001" / "depth_3dlong" / f"{scene_id:06d}.tif"
    manual_cam = base_dir / "data" / "itodd_manual_annotated" / "val" / "000001" / "scene_camera_3dlong.json"
    native_scene_dir = base_dir / "data" / "3d_long_baseline" / "scenes" / f"scene_{scene_id:04d}"

    if (source in ("auto", "bop_manual")) and manual_depth.exists():
        depth = np.asarray(iio.imread(manual_depth))
        cams = json.loads(manual_cam.read_text(encoding="utf-8"))
        cam_k = np.array(cams[str(scene_id)]["cam_K"]).reshape(3, 3)
        valid = depth > 0.0
        r, c = np.where(valid)
        z_mm = depth[r, c]
        x_mm = (c - cam_k[0, 2]) * z_mm / cam_k[0, 0]
        y_mm = (r - cam_k[1, 2]) * z_mm / cam_k[1, 1]
        pts_mm = np.column_stack([x_mm, y_mm, z_mm])

        l_path = native_scene_dir / "3d_long_baseline_l.tif"
        if l_path.exists():
            l_img = np.asarray(iio.imread(l_path))[valid].astype(np.float64)
            l_norm = (l_img - l_img.min()) / (l_img.max() - l_img.min() + 1e-6)
            colors = np.column_stack([l_norm, l_norm, l_norm])
        else:
            colors = np.tile(np.array([0.65, 0.70, 0.75]), (len(pts_mm), 1))
    else:
        x_path = native_scene_dir / "3d_long_baseline_x.tif"
        y_path = native_scene_dir / "3d_long_baseline_y.tif"
        z_path = native_scene_dir / "3d_long_baseline_z.tif"
        l_path = native_scene_dir / "3d_long_baseline_l.tif"
        if not z_path.exists():
            raise FileNotFoundError(f"Native scene files missing: {native_scene_dir}")

        x_img = np.asarray(iio.imread(x_path))
        y_img = np.asarray(iio.imread(y_path))
        z_img = np.asarray(iio.imread(z_path))
        valid = np.isfinite(z_img) & (z_img > 0.05) & (z_img < 1.5)
        pts_mm = np.stack([x_img[valid], y_img[valid], z_img[valid]], axis=-1) * 1000.0

        if l_path.exists():
            l_img = np.asarray(iio.imread(l_path))[valid].astype(np.float64)
            l_norm = (l_img - l_img.min()) / (l_img.max() - l_img.min() + 1e-6)
            colors = np.column_stack([l_norm, l_norm, l_norm])
        else:
            colors = np.tile(np.array([0.65, 0.70, 0.75]), (len(pts_mm), 1))

    return pts_mm, colors


def find_ground_truths(scene_id: int, base_dir: Path) -> List[Tuple[np.ndarray, np.ndarray, int]]:
    gt_sources = [
        base_dir / "data" / "itodd_manual_annotated" / "val" / "000001" / "scene_gt_3dlong.json",
        base_dir / "data" / "itoddmv_val" / "val" / "000001" / "scene_gt_3dlong.json",
    ]
    for gt_path in gt_sources:
        if not gt_path.exists():
            continue
        try:
            doc = json.loads(gt_path.read_text(encoding="utf-8"))
            key = str(scene_id)
            if key in doc and len(doc[key]) > 0:
                poses = []
                for entry in doc[key]:
                    R = np.array(entry["cam_R_m2c"], dtype=np.float64).reshape(3, 3)
                    t = np.array(entry["cam_t_m2c"], dtype=np.float64)
                    obj_id = entry.get("obj_id", -1)
                    poses.append((R, t, obj_id))
                return poses
        except Exception:
            pass
    return []


class RansacTabletopViewer:
    """Interactive visualizer for RANSAC tabletop extraction with real-time threshold tuning."""

    def __init__(
        self,
        scene_id: int,
        model_name: str,
        base_dir: Path,
        source: str = "auto",
        show_gt: bool = True,
        initial_thresh: float = 2.0,
        camera_json_path: Optional[Path] = None,
    ) -> None:
        self.scene_id = scene_id
        self.model_name = model_name
        self.base_dir = base_dir
        self.thresh = initial_thresh
        self.show_gt = show_gt
        self.camera_json_path = camera_json_path

        # Display mode for tabletop:
        # 0: Highlighted in Red
        # 1: Completely Removed (Hidden)
        # 2: Muted Soft Blue-Gray
        self.tabletop_mode = 0

        # Load raw points
        raw_pts, raw_colors = load_scene_points(scene_id, base_dir, source)
        self.raw_pts = raw_pts
        self.raw_colors = raw_colors

        # Base PointCloud object used for RANSAC and normal estimation
        self.base_pcd = o3d.geometry.PointCloud()
        self.base_pcd.points = o3d.utility.Vector3dVector(self.raw_pts)
        if len(self.raw_pts) > 0:
            self.base_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=10.0, max_nn=30))
            self.raw_normals = np.asarray(self.base_pcd.normals)
        else:
            self.raw_normals = np.empty((0, 3))

        # Geometries for rendering
        self.pcd_objects = o3d.geometry.PointCloud()
        self.pcd_tabletop = o3d.geometry.PointCloud()

        # Cache for GT CAD models
        self.gt_models: List[o3d.geometry.Geometry] = []
        gt_list = find_ground_truths(scene_id, base_dir)
        if gt_list:
            for idx, (R, t, obj_id) in enumerate(gt_list):
                m_name = OBJ_ID_TO_MODEL.get(obj_id, model_name)
                try:
                    cad_file = find_cad_file(m_name, base_dir)
                    mesh = o3d.io.read_triangle_mesh(str(cad_file))
                    if len(mesh.triangles) > 0:
                        mesh.compute_vertex_normals()
                    else:
                        mesh = o3d.io.read_point_cloud(str(cad_file))
                    T = np.eye(4)
                    T[:3, :3] = R
                    T[:3, 3] = t
                    mesh.transform(T)
                    mesh.paint_uniform_color(INSTANCE_COLORS[idx % len(INSTANCE_COLORS)])
                    self.gt_models.append(mesh)

                    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=35.0, origin=[0, 0, 0])
                    frame.transform(T)
                    self.gt_models.append(frame)
                except Exception:
                    pass

        self.plane_model = [0.0, 0.0, 1.0, 0.0]
        self.last_inliers: np.ndarray = np.array([])
        self.last_outliers: np.ndarray = np.array([])

        # Perform initial segmentation
        self._segment_plane()

    def _segment_plane(self) -> None:
        """Run Open3D C++ RANSAC plane segmentation."""
        if len(self.base_pcd.points) < 100:
            return

        plane_model, inliers = self.base_pcd.segment_plane(
            distance_threshold=self.thresh,
            ransac_n=3,
            num_iterations=1000,
        )
        self.plane_model = plane_model
        inlier_set = set(inliers)
        all_indices = np.arange(len(self.raw_pts))
        outlier_indices = np.array([i for i in all_indices if i not in inlier_set], dtype=int)

        self.last_inliers = inliers
        self.last_outliers = outlier_indices

        # 1. Update Object Points
        self.pcd_objects.points = o3d.utility.Vector3dVector(self.raw_pts[outlier_indices])
        self.pcd_objects.colors = o3d.utility.Vector3dVector(self.raw_colors[outlier_indices])
        if len(self.raw_normals) == len(self.raw_pts):
            self.pcd_objects.normals = o3d.utility.Vector3dVector(self.raw_normals[outlier_indices])

        # 2. Update Tabletop Points
        if self.tabletop_mode == 0:
            # Highlight in coral red matching sequential subtraction peel palette
            self.pcd_tabletop.points = o3d.utility.Vector3dVector(self.raw_pts[inliers])
            self.pcd_tabletop.paint_uniform_color([0.95, 0.25, 0.25])
            if len(self.raw_normals) == len(self.raw_pts):
                self.pcd_tabletop.normals = o3d.utility.Vector3dVector(self.raw_normals[inliers])
        elif self.tabletop_mode == 1:
            # Completely removed / hidden
            self.pcd_tabletop.points = o3d.utility.Vector3dVector([])
            self.pcd_tabletop.normals = o3d.utility.Vector3dVector([])
        elif self.tabletop_mode == 2:
            # Muted soft blue-gray matching visualize_results default point tone
            self.pcd_tabletop.points = o3d.utility.Vector3dVector(self.raw_pts[inliers])
            self.pcd_tabletop.paint_uniform_color([0.65, 0.70, 0.75])
            if len(self.raw_normals) == len(self.raw_pts):
                self.pcd_tabletop.normals = o3d.utility.Vector3dVector(self.raw_normals[inliers])

    def print_status(self) -> None:
        total = len(self.raw_pts)
        n_in = len(self.last_inliers)
        n_out = len(self.last_outliers)
        a, b, c, d = self.plane_model
        mode_names = ["Tabletop in RED", "Tabletop REMOVED", "Tabletop in GRAY"]
        mode_str = mode_names[self.tabletop_mode]

        print(
            f"\r[Threshold: {self.thresh:4.1f} mm | Mode: {mode_str:<16}] "
            f"Tabletop: {n_in:,} ({n_in/total*100:4.1f}%) | "
            f"Objects: {n_out:,} ({n_out/total*100:4.1f}%) | "
            f"Plane: {a:+.2f}x + {b:+.2f}y + {c:+.2f}z {d:+.1f}=0",
            end="",
        )
        sys.stdout.flush()

    def print_help(self) -> None:
        print("\n" + "=" * 70)
        print(f" DYNAMIC RANSAC TABLETOP REMOVER: Scene {self.scene_id:04d} ({self.model_name})")
        print("=" * 70)
        print(" [Up] / [+]     : Fine increase threshold (+0.2 mm)")
        print(" [Down] / [-]   : Fine decrease threshold (-0.2 mm)")
        print(" [Right] / []]  : Coarse increase threshold (+1.0 mm)")
        print(" [Left] / [[]   : Coarse decrease threshold (-1.0 mm)")
        print(" [1] ~ [9]      : Quick set threshold to 1.0, 2.0, ..., 9.0 mm")
        print(" [T]            : Toggle Tabletop display (Red -> Completely Removed -> Gray)")
        print(" [G]            : Toggle Ground Truth CAD models visibility")
        print(" [C]            : Capture high-resolution screenshot (auto-cropped white margin)")
        print(" [V]            : Reset camera perspective to canonical oblique / Camera JSON view")
        print(" [S]            : Save current clean object point cloud to PLY")
        print(" [R]            : Reset threshold to 2.0 mm")
        print(" [N] / [Space]  : Next scene")
        print(" [P]            : Previous scene")
        print(" [H]            : Print this help message")
        print(" [ESC]          : Exit visualizer")
        print("=" * 70 + "\n")

    def save_filtered_pcd(self) -> Path:
        out_dir = self.base_dir / "data" / "exported_ply" / "ransac_filtered"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"scene_{self.scene_id:04d}_thresh_{self.thresh:.1f}mm_objects.ply"
        o3d.io.write_point_cloud(str(out_file), self.pcd_objects)
        print(f"\n[+] Saved clean object point cloud ({len(self.pcd_objects.points):,} pts) -> {out_file}")
        return out_file

    def run(self) -> str:
        """Launch Open3D Visualizer. Return 'next', 'prev', or 'exit'."""
        vis = o3d.visualization.VisualizerWithKeyCallback()
        title = f"RANSAC Tabletop Remover | Scene {self.scene_id:04d} | Press 'H' for Help"
        vis.create_window(window_name=title, width=1600, height=950)

        opt = vis.get_render_option()
        opt.point_size = 3.0
        opt.background_color = np.asarray([1.0, 1.0, 1.0])
        opt.light_on = True

        vis.add_geometry(self.pcd_objects)
        vis.add_geometry(self.pcd_tabletop)

        if self.show_gt:
            for geom in self.gt_models:
                vis.add_geometry(geom)

        def _apply_view_perspective(ctr):
            has_custom = False
            if self.camera_json_path is not None and Path(self.camera_json_path).exists():
                try:
                    param = o3d.io.read_pinhole_camera_parameters(str(self.camera_json_path))
                    ctr.convert_from_pinhole_camera_parameters(param, allow_arbitrary=True)
                    has_custom = True
                    print(f"\n[+] Applied camera viewpoint parameters from: {self.camera_json_path}")
                except Exception as e:
                    print(f"\n[!] Warning: Failed to apply camera JSON {self.camera_json_path}: {e}")

            if not has_custom:
                vis_geoms = [self.pcd_objects]
                if len(self.pcd_tabletop.points) > 0:
                    vis_geoms.append(self.pcd_tabletop)
                if self.show_gt:
                    vis_geoms.extend(self.gt_models)

                min_b = self.pcd_objects.get_min_bound()
                max_b = self.pcd_objects.get_max_bound()
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
        ctr = vis.get_view_control()
        _apply_view_perspective(ctr)

        nav_state = {"action": "exit"}

        def cb_adjust_thresh(delta: float):
            def handler(v):
                self.thresh = max(0.2, round(self.thresh + delta, 2))
                self._segment_plane()
                v.update_geometry(self.pcd_objects)
                v.update_geometry(self.pcd_tabletop)
                self.print_status()
                return False
            return handler

        def cb_set_thresh(val: float):
            def handler(v):
                self.thresh = val
                self._segment_plane()
                v.update_geometry(self.pcd_objects)
                v.update_geometry(self.pcd_tabletop)
                self.print_status()
                return False
            return handler

        def cb_toggle_tabletop(v):
            self.tabletop_mode = (self.tabletop_mode + 1) % 3
            self._segment_plane()
            v.update_geometry(self.pcd_tabletop)
            self.print_status()
            return False

        def cb_toggle_gt(v):
            self.show_gt = not self.show_gt
            for geom in self.gt_models:
                if self.show_gt:
                    v.add_geometry(geom)
                else:
                    v.remove_geometry(geom)
            print(f"\n[*] GT CAD Models: {'Visible' if self.show_gt else 'Hidden'}")
            return False

        def cb_save(_):
            self.save_filtered_pcd()
            return False

        def cb_capture(v):
            out_dir = self.base_dir / "data" / "exported_ply" / "ransac_filtered"
            out_dir.mkdir(parents=True, exist_ok=True)
            mode_tag = ["table_red", "table_removed", "table_gray"][self.tabletop_mode]
            out_file = out_dir / f"scene_{self.scene_id:04d}_{self.model_name}_thresh_{self.thresh:.1f}mm_{mode_tag}.png"
            v.poll_events()
            v.update_renderer()
            v.capture_screen_image(str(out_file))

            # Auto crop white margins for publication figure readiness
            img_bgr = cv2.imread(str(out_file))
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
                    cv2.imwrite(str(out_file), cropped)

            print(f"\n[+] Saved high-resolution cropped screenshot -> {out_file}")
            self.print_status()
            return False

        def cb_reset_camera(v):
            _apply_view_perspective(v.get_view_control())
            v.update_renderer()
            print("\n[*] Camera perspective reset to canonical view.")
            self.print_status()
            return False

        def cb_reset(v):
            self.thresh = 2.0
            self.tabletop_mode = 0
            self._segment_plane()
            v.update_geometry(self.pcd_objects)
            v.update_geometry(self.pcd_tabletop)
            print("\n[*] Reset threshold to 2.0 mm.")
            self.print_status()
            return False

        def cb_next(v):
            nav_state["action"] = "next"
            v.close()
            return False

        def cb_prev(v):
            nav_state["action"] = "prev"
            v.close()
            return False

        def cb_exit(v):
            nav_state["action"] = "exit"
            v.close()
            return False

        def cb_help(_):
            self.print_help()
            self.print_status()
            return False

        # Register GLFW key callbacks
        # Threshold fine tuning
        vis.register_key_callback(265, cb_adjust_thresh(+0.2))  # Up Arrow
        vis.register_key_callback(264, cb_adjust_thresh(-0.2))  # Down Arrow
        vis.register_key_callback(ord("+"), cb_adjust_thresh(+0.2))
        vis.register_key_callback(ord("="), cb_adjust_thresh(+0.2))
        vis.register_key_callback(ord("-"), cb_adjust_thresh(-0.2))

        # Threshold coarse tuning
        vis.register_key_callback(262, cb_adjust_thresh(+1.0))  # Right Arrow
        vis.register_key_callback(263, cb_adjust_thresh(-1.0))  # Left Arrow
        vis.register_key_callback(ord("]"), cb_adjust_thresh(+1.0))
        vis.register_key_callback(ord("["), cb_adjust_thresh(-1.0))

        # Direct number keys 1 to 9
        for num in range(1, 10):
            vis.register_key_callback(ord(str(num)), cb_set_thresh(float(num)))

        # Feature toggles
        vis.register_key_callback(ord("T"), cb_toggle_tabletop)
        vis.register_key_callback(ord("t"), cb_toggle_tabletop)
        vis.register_key_callback(ord("G"), cb_toggle_gt)
        vis.register_key_callback(ord("g"), cb_toggle_gt)
        vis.register_key_callback(ord("C"), cb_capture)
        vis.register_key_callback(ord("c"), cb_capture)
        vis.register_key_callback(ord("V"), cb_reset_camera)
        vis.register_key_callback(ord("v"), cb_reset_camera)
        vis.register_key_callback(ord("S"), cb_save)
        vis.register_key_callback(ord("s"), cb_save)
        vis.register_key_callback(ord("R"), cb_reset)
        vis.register_key_callback(ord("r"), cb_reset)
        vis.register_key_callback(ord("H"), cb_help)
        vis.register_key_callback(ord("h"), cb_help)

        # Navigation
        vis.register_key_callback(ord("N"), cb_next)
        vis.register_key_callback(ord("n"), cb_next)
        vis.register_key_callback(32, cb_next)  # Space bar
        vis.register_key_callback(ord("P"), cb_prev)
        vis.register_key_callback(ord("p"), cb_prev)
        vis.register_key_callback(256, cb_exit)  # ESC

        self.print_help()
        self.print_status()

        vis.run()
        vis.destroy_window()
        print()  # Newline after status line

        return nav_state["action"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Dynamic RANSAC Tabletop Removal Visualizer")
    parser.add_argument("--scene-id", type=int, help="Single scene ID to inspect (e.g. 8)")
    parser.add_argument(
        "--model",
        type=str,
        choices=list(OBJECT_CONFIGS.keys()),
        help="Object category: star, bracket_planar, or screw_black",
    )
    parser.add_argument(
        "--scenes",
        type=str,
        default="",
        help="Comma-separated scene IDs (e.g. '8,10,13')",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Inspect all scenes in sequence",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="auto",
        choices=["auto", "bop_manual", "native"],
        help="Scene point cloud source",
    )
    parser.add_argument(
        "--thresh",
        type=float,
        default=None,
        help="Initial RANSAC distance threshold in mm (defaults to model-specific threshold)",
    )
    parser.add_argument("--no-gt", dest="show_gt", action="store_false", default=True, help="Hide Ground Truth CAD models")
    parser.add_argument(
        "--camera-json",
        type=Path,
        default=None,
        help="Path to Open3D camera parameter JSON (default: ScreenCamera_2026-09-05-17-46-04.json if found)",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent

    camera_json_path = resolve_camera_json(args.camera_json)
    if camera_json_path and camera_json_path.exists():
        print(f"[+] Using camera viewpoint parameters from: {camera_json_path}")
    else:
        if args.camera_json is not None:
            print(f"[!] Warning: Specified camera JSON not found: {args.camera_json}")

    work_list: List[Tuple[int, str]] = []
    if args.scene_id is not None:
        m = args.model or "star"
        work_list.append((args.scene_id, m))
    elif args.all:
        for m in ["star", "bracket_planar", "screw_black"]:
            for s in OBJECT_CONFIGS[m]["default_scenes"]:
                work_list.append((s, m))
    elif args.scenes.strip():
        s_ids = [int(x.strip()) for x in args.scenes.split(",") if x.strip()]
        for s in s_ids:
            m = args.model
            if not m:
                for cand_m, cfg in OBJECT_CONFIGS.items():
                    if s in cfg["default_scenes"]:
                        m = cand_m
                        break
            work_list.append((s, m or "star"))
    elif args.model:
        for s in OBJECT_CONFIGS[args.model]["default_scenes"]:
            work_list.append((s, args.model))
    else:
        # Default: star scenes
        for s in OBJECT_CONFIGS["star"]["default_scenes"]:
            work_list.append((s, "star"))

    current_idx = 0
    total = len(work_list)
    print(f"\n[+] Scheduled {total} scene(s) for dynamic RANSAC tabletop removal analysis.")

    while 0 <= current_idx < total:
        s_id, m_name = work_list[current_idx]
        viewer = RansacTabletopViewer(
            scene_id=s_id,
            model_name=m_name,
            base_dir=base_dir,
            source=args.source,
            show_gt=args.show_gt,
            initial_thresh=(
                args.thresh
                if args.thresh is not None
                else OBJECT_CONFIGS.get(m_name, {}).get("default_thresh", 2.0)
            ),
            camera_json_path=camera_json_path,
        )
        action = viewer.run()
        if action == "next":
            current_idx += 1
        elif action == "prev":
            current_idx = max(0, current_idx - 1)
        else:  # exit
            break

    print("\n[+] Dynamic RANSAC tabletop inspection session finished.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
