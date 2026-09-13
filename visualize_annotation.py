"""3D Scene & CAD Model Pose Alignment Visualizer.

Supports sequentially inspecting manual annotations across all scenes or by model.
Press 'N' or Space to advance to the next scene; press 'ESC' to exit.

Usage examples:
  # Sequentially inspect all 7 star scenes:
  uv run python visualize_annotation.py --model star

  # Sequentially inspect all 7 bracket_planar scenes:
  uv run python visualize_annotation.py --model bracket_planar

  # Sequentially inspect all 5 screw_black scenes:
  uv run python visualize_annotation.py --model screw_black

  # Sequentially inspect ALL 19 scenes in order:
  uv run python visualize_annotation.py --all

  # Inspect specific scenes:
  uv run python visualize_annotation.py --scenes 8,10,13

  # Inspect a single scene:
  uv run python visualize_annotation.py --scene-id 8 --model star
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import imageio.v3 as iio
import numpy as np
import open3d as o3d

# Distinct colors for multiple GT instances
INSTANCE_COLORS = [
    [0.15, 0.85, 0.25],  # Vibrant Green
    [0.00, 0.75, 1.00],  # Cyan
    [1.00, 0.55, 0.00],  # Orange
    [0.90, 0.20, 0.80],  # Magenta
    [1.00, 0.85, 0.10],  # Yellow-Gold
    [0.30, 0.50, 1.00],  # Royal Blue
]

OBJECT_CONFIGS = {
    "star": {
        "obj_id": 25,
        "cad_filename": "obj_000025.ply",
        "alt_cad": "star.ply",
        "default_scenes": [0, 3, 8, 10, 13, 164, 749],
    },
    "bracket_planar": {
        "obj_id": 5,
        "cad_filename": "obj_000005.ply",
        "alt_cad": "bracket_planar.ply",
        "default_scenes": [450, 456, 461, 468, 476, 481, 488],
    },
    "screw_black": {
        "obj_id": 24,
        "cad_filename": "obj_000024.ply",
        "alt_cad": "screw_black.ply",
        "default_scenes": [293, 296, 299, 302, 305],
    },
}

OBJ_ID_TO_MODEL = {
    25: "star",
    5: "bracket_planar",
    24: "screw_black",
}


def find_cad_file(model_name: str, base_dir: Path) -> Path:
    cfg = OBJECT_CONFIGS.get(model_name)
    if not cfg:
        # Fallback by obj_id search
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
    raise FileNotFoundError(f"CAD model for {model_name} not found. Searched: {candidates}")


def load_native_scene(scene_id: int, base_dir: Path) -> o3d.geometry.PointCloud:
    scene_dir = base_dir / "data" / "3d_long_baseline" / "scenes" / f"scene_{scene_id:04d}"
    x_path = scene_dir / "3d_long_baseline_x.tif"
    y_path = scene_dir / "3d_long_baseline_y.tif"
    z_path = scene_dir / "3d_long_baseline_z.tif"
    l_path = scene_dir / "3d_long_baseline_l.tif"

    if not z_path.exists():
        raise FileNotFoundError(f"Native scene files missing: {scene_dir}")

    x_img = np.asarray(iio.imread(x_path))
    y_img = np.asarray(iio.imread(y_path))
    z_img = np.asarray(iio.imread(z_path))

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
        pcd.paint_uniform_color([0.7, 0.7, 0.7])

    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=10.0, max_nn=30))
    return pcd


def load_bop_scene(depth_path: Path, cam_path: Path, scene_id: int, base_dir: Path) -> o3d.geometry.PointCloud:
    if not depth_path.exists():
        raise FileNotFoundError(f"BOP depth missing: {depth_path}")
    if not cam_path.exists():
        raise FileNotFoundError(f"BOP camera json missing: {cam_path}")

    depth = np.asarray(iio.imread(depth_path))
    cams = json.loads(cam_path.read_text(encoding="utf-8"))
    key = str(scene_id)
    if key not in cams:
        raise KeyError(f"Scene {scene_id} missing from {cam_path}")

    cam_k = np.array(cams[key]["cam_K"]).reshape(3, 3)
    valid = depth > 0.0
    r, c = np.where(valid)
    z_mm = depth[r, c]
    x_mm = (c - cam_k[0, 2]) * z_mm / cam_k[0, 0]
    y_mm = (r - cam_k[1, 2]) * z_mm / cam_k[1, 1]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.column_stack([x_mm, y_mm, z_mm]))

    # Attempt to load native grayscale texture
    l_path = base_dir / "data" / "3d_long_baseline" / "scenes" / f"scene_{scene_id:04d}" / "3d_long_baseline_l.tif"
    if l_path.exists():
        l_img = np.asarray(iio.imread(l_path))[valid].astype(np.float64)
        l_norm = (l_img - l_img.min()) / (l_img.max() - l_img.min() + 1e-6)
        pcd.colors = o3d.utility.Vector3dVector(np.column_stack([l_norm, l_norm, l_norm]))
    else:
        pcd.paint_uniform_color([0.65, 0.70, 0.75])

    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=10.0, max_nn=30))
    return pcd


def find_ground_truths(scene_id: int, base_dir: Path) -> List[Tuple[np.ndarray, np.ndarray, int]]:
    """Search for ground truth poses across manual annotations and validation sets."""
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


def visualize_single_scene(
    scene_id: int,
    model_name: str,
    base_dir: Path,
    source: str = "auto",
    voxel_size: float = 0.0,
    progress_str: str = "",
) -> bool:
    """Render scene and GT models. Return True to advance to next scene, False to exit."""
    print("\n" + "=" * 65)
    print(f" {progress_str} Scene {scene_id:04d} | Target Object: {model_name}")
    print("=" * 65)

    # 1. Load Scene Point Cloud
    manual_depth = base_dir / "data" / "itodd_manual_annotated" / "val" / "000001" / "depth_3dlong" / f"{scene_id:06d}.tif"
    manual_cam = base_dir / "data" / "itodd_manual_annotated" / "val" / "000001" / "scene_camera_3dlong.json"
    val_depth = base_dir / "data" / "itoddmv_val" / "val" / "000001" / "depth_3dlong" / f"{scene_id:06d}.tif"
    val_cam = base_dir / "data" / "itoddmv_val" / "val" / "000001" / "scene_camera_3dlong.json"

    if source == "bop_manual" or (source == "auto" and manual_depth.exists()):
        print(f"[+] Loaded from manual BOP depth: {manual_depth.name}")
        scene_pcd = load_bop_scene(manual_depth, manual_cam, scene_id, base_dir)
    elif source == "bop_val" or (source == "auto" and val_depth.exists()):
        print(f"[+] Loaded from official BOP depth: {val_depth.name}")
        scene_pcd = load_bop_scene(val_depth, val_cam, scene_id, base_dir)
    else:
        print(f"[+] Loaded from native sensor TIFFs (scene_{scene_id:04d})...")
        scene_pcd = load_native_scene(scene_id, base_dir)

    if voxel_size > 0.0:
        scene_pcd = scene_pcd.voxel_down_sample(voxel_size=voxel_size)

    pts = np.asarray(scene_pcd.points)
    print(f"    - Scene Points: {len(pts):,}, Z range: [{pts[:, 2].min():.1f}, {pts[:, 2].max():.1f}] mm")

    # 2. Check for GT Poses
    gt_list = find_ground_truths(scene_id, base_dir)
    geometries_to_render: List[o3d.geometry.Geometry] = [scene_pcd]

    # Resolve active model if GT specifies another obj_id
    if gt_list and gt_list[0][2] in OBJ_ID_TO_MODEL:
        model_name = OBJ_ID_TO_MODEL[gt_list[0][2]]

    cad_path = find_cad_file(model_name, base_dir)
    mesh_template = o3d.io.read_triangle_mesh(str(cad_path))
    if len(mesh_template.triangles) > 0:
        mesh_template.compute_vertex_normals()
    else:
        pcd_cad = o3d.io.read_point_cloud(str(cad_path))
        pcd_cad.estimate_normals()
        mesh_template = pcd_cad

    if gt_list:
        print(f"[+] Displaying {len(gt_list)} Ground-Truth Instances:")
        for idx, (R, t, _) in enumerate(gt_list):
            color = INSTANCE_COLORS[idx % len(INSTANCE_COLORS)]
            print(f"    - Instance #{idx + 1}: t = [{t[0]:.2f}, {t[1]:.2f}, {t[2]:.2f}] mm, distance={np.linalg.norm(t):.1f} mm")

            instance_geom = copy.deepcopy(mesh_template)
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = R
            T[:3, 3] = t
            instance_geom.transform(T)
            instance_geom.paint_uniform_color(color)
            geometries_to_render.append(instance_geom)

            # Local coordinate axis at object origin (size: 35mm)
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=35.0, origin=[0, 0, 0])
            frame.transform(T)
            geometries_to_render.append(frame)
    else:
        print(f"[*] Notice: No Ground-Truth annotations found yet for Scene {scene_id:04d}.")
        print("    Displaying CAD model at default bin center for reference inspection:")
        init_t = np.array([3.062, -0.230, 721.286], dtype=np.float64)
        print(f"    t = [{init_t[0]:.2f}, {init_t[1]:.2f}, {init_t[2]:.2f}] mm (Yellow Model)")

        ref_geom = copy.deepcopy(mesh_template)
        T = np.eye(4, dtype=np.float64)
        T[:3, 3] = init_t
        ref_geom.transform(T)
        ref_geom.paint_uniform_color([1.0, 0.85, 0.0])  # Yellow
        geometries_to_render.append(ref_geom)

        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=40.0, origin=init_t)
        geometries_to_render.append(frame)

    # Camera World Origin Coordinate Axis (size: 60mm)
    cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=60.0, origin=[0, 0, 0])
    geometries_to_render.append(cam_frame)

    print("-" * 65)
    print(" Controls:")
    print("   - [N] / [Space]    : Next scene")
    print("   - [ESC]            : Stop / Exit inspection")
    print("   - Left Mouse Drag  : Rotate viewpoint")
    print("   - Right Mouse Drag : Pan camera")
    print("   - Mouse Scroll     : Zoom in / out")
    print("=" * 65)

    # 4. Launch Open3D Visualizer with Key Callbacks
    vis = o3d.visualization.VisualizerWithKeyCallback()
    window_title = f"ITODD Inspection | {progress_str} Scene {scene_id:04d} ({model_name}) | Press 'N' for Next, 'ESC' to Exit"
    vis.create_window(window_name=window_title, width=1600, height=950)

    for geom in geometries_to_render:
        vis.add_geometry(geom)

    opt = vis.get_render_option()
    opt.point_size = 3.0
    opt.background_color = np.asarray([0.15, 0.15, 0.18])
    opt.show_coordinate_frame = False

    state = {"continue": True}

    def cb_next(v):
        v.close()
        return False

    def cb_exit(v):
        state["continue"] = False
        v.close()
        return False

    vis.register_key_callback(ord("N"), cb_next)
    vis.register_key_callback(32, cb_next)  # Space bar
    vis.register_key_callback(256, cb_exit)  # ESC

    vis.run()
    vis.destroy_window()

    return state["continue"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize Scene Point Cloud with Overlaid CAD Model")
    parser.add_argument("--scene-id", type=int, help="Single Scene ID to visualize (e.g. 8)")
    parser.add_argument(
        "--model",
        type=str,
        choices=list(OBJECT_CONFIGS.keys()),
        help="Target object model name: star, bracket_planar, or screw_black",
    )
    parser.add_argument(
        "--scenes",
        type=str,
        default="",
        help="Comma-separated scene IDs to inspect sequentially (e.g. '8,10,13')",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Sequentially inspect ALL 19 annotated scenes in order",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="auto",
        choices=["auto", "native", "bop_manual", "bop_val"],
        help="Point cloud source (default: auto)",
    )
    parser.add_argument("--voxel-size", type=float, default=0.0, help="Downsampling voxel size (mm)")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent

    # Build work sequence: list of (scene_id, model_name)
    work_list: List[Tuple[int, str]] = []

    if args.scene_id is not None:
        # Single scene
        m = args.model or "star"
        work_list.append((args.scene_id, m))
    elif args.all:
        # All scenes across 3 objects
        for m in ["star", "bracket_planar", "screw_black"]:
            for s in OBJECT_CONFIGS[m]["default_scenes"]:
                work_list.append((s, m))
    elif args.scenes.strip():
        # Custom scenes list
        scene_ids = [int(x.strip()) for x in args.scenes.split(",") if x.strip()]
        for s in scene_ids:
            # Auto-detect model if not specified
            m = args.model
            if not m:
                for candidate_m, cfg in OBJECT_CONFIGS.items():
                    if s in cfg["default_scenes"]:
                        m = candidate_m
                        break
            work_list.append((s, m or "star"))
    elif args.model:
        # All scenes for specified model
        for s in OBJECT_CONFIGS[args.model]["default_scenes"]:
            work_list.append((s, args.model))
    else:
        # Default: all star scenes
        print("[*] No target specified. Defaulting to inspecting 'star' scenes.")
        for s in OBJECT_CONFIGS["star"]["default_scenes"]:
            work_list.append((s, "star"))

    total = len(work_list)
    print(f"\n[+] Scheduled {total} scene(s) for visual inspection.")

    for idx, (scene_id, model_name) in enumerate(work_list, start=1):
        progress_str = f"[{idx}/{total}]"
        keep_going = visualize_single_scene(
            scene_id=scene_id,
            model_name=model_name,
            base_dir=base_dir,
            source=args.source,
            voxel_size=args.voxel_size,
            progress_str=progress_str,
        )
        if not keep_going:
            print("\n[!] Inspection session terminated by user (ESC pressed).")
            break

    print("\n[+] Visual inspection completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
