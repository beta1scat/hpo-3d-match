"""Convert official ITODD BOP validation scenes into native-consistent manual dataset format.

Compensates for the ~3.41mm physical sensor vs. BOP virtual pinhole camera coordinate offset,
ensuring all scenes in data/itodd_manual_annotated share the exact same physical sensor coordinate frame.

Targets official BOP validation scenes:
  - star:           0, 3
  - screw_black:    293, 296
  - bracket_planar: 450, 468

Usage:
  # Convert all 6 target scenes with offset compensation:
  uv run python convert_itoddval_to_manual_format.py

  # Convert specific scene (e.g. scene 0):
  uv run python convert_itoddval_to_manual_format.py --scenes 0
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import imageio.v3 as iio
import numpy as np
import open3d as o3d

TARGET_SCENES = {
    0: ("star", 25),
    3: ("star", 25),
    293: ("screw_black", 24),
    296: ("screw_black", 24),
    450: ("bracket_planar", 5),
    468: ("bracket_planar", 5),
}


def compute_bop_to_native_transform(
    scene_id: int, base_dir: Path, cam_k: np.ndarray
) -> Tuple[np.ndarray, float]:
    """Compute exact rigid transformation T_bop_to_native via Point-to-Point ICP."""
    # 1. Native point cloud in mm
    scene_dir = base_dir / "data" / "3d_long_baseline" / "scenes" / f"scene_{scene_id:04d}"
    x_m = iio.imread(scene_dir / "3d_long_baseline_x.tif")
    y_m = iio.imread(scene_dir / "3d_long_baseline_y.tif")
    z_m = iio.imread(scene_dir / "3d_long_baseline_z.tif")
    valid_nat = np.isfinite(z_m) & (z_m > 0.05) & (z_m < 1.0)
    pts_nat_mm = np.stack([x_m[valid_nat], y_m[valid_nat], z_m[valid_nat]], axis=-1)[::4] * 1000.0

    # 2. BOP point cloud in mm
    bop_depth_file = base_dir / "data" / "itoddmv_val" / "val" / "000001" / "depth_3dlong" / f"{scene_id:06d}.tif"
    depth_bop = iio.imread(bop_depth_file)
    valid_bop = depth_bop > 0
    r, c = np.where(valid_bop)
    z_bop = depth_bop[r, c]
    x_bop = (c - cam_k[0, 2]) * z_bop / cam_k[0, 0]
    y_bop = (r - cam_k[1, 2]) * z_bop / cam_k[1, 1]
    pts_bop_mm = np.column_stack([x_bop, y_bop, z_bop])[::4]

    pcd_nat = o3d.geometry.PointCloud()
    pcd_nat.points = o3d.utility.Vector3dVector(pts_nat_mm)
    pcd_bop = o3d.geometry.PointCloud()
    pcd_bop.points = o3d.utility.Vector3dVector(pts_bop_mm)

    reg = o3d.pipelines.registration.registration_icp(
        pcd_nat,
        pcd_bop,
        max_correspondence_distance=10.0,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )
    T_nat2bop = reg.transformation
    T_bop2nat = np.linalg.inv(T_nat2bop)
    return T_bop2nat, reg.inlier_rmse


def convert_scene(
    scene_id: int,
    model_name: str,
    obj_id: int,
    base_dir: Path,
    dst_dir: Path,
    dst_manifest: Path,
) -> None:
    """Convert one official BOP scene to the native-consistent manual dataset format."""
    src_val_dir = base_dir / "data" / "itoddmv_val" / "val" / "000001"
    src_cam_file = src_val_dir / "scene_camera_3dlong.json"
    src_gt_file = src_val_dir / "scene_gt_3dlong.json"
    src_info_file = src_val_dir / "scene_gt_info_3dlong.json"

    native_scene_dir = base_dir / "data" / "3d_long_baseline" / "scenes" / f"scene_{scene_id:04d}"
    native_z_file = native_scene_dir / "3d_long_baseline_z.tif"

    if not native_z_file.exists():
        raise FileNotFoundError(f"Native scene files missing: {native_scene_dir}")

    # 1. Load Camera parameters
    cams = json.loads(src_cam_file.read_text(encoding="utf-8"))
    cam_k = np.array(cams[str(scene_id)]["cam_K"]).reshape(3, 3)

    # 2. Compute exact rigid compensation transform T_bop_to_native
    T_bop2nat, rmse = compute_bop_to_native_transform(scene_id, base_dir, cam_k)
    dt = T_bop2nat[:3, 3]

    print(f"\n[+] Scene {scene_id:04d} ({model_name}):")
    print(f"    - Offset Compensation (BOP -> Native): dt = [{dt[0]:+.3f}, {dt[1]:+.3f}, {dt[2]:+.3f}] mm (RMSE={rmse:.3f} mm)")

    # 3. Export Native Depth in float32 mm
    dst_dir.mkdir(parents=True, exist_ok=True)
    depth_dir = dst_dir / "depth_3dlong"
    depth_dir.mkdir(parents=True, exist_ok=True)

    native_z = np.asarray(iio.imread(native_z_file))
    valid = np.isfinite(native_z) & (native_z > 0.05)
    depth_mm = np.where(valid, (native_z * 1000.0).astype(np.float32), 0.0)
    out_depth_file = depth_dir / f"{scene_id:06d}.tif"
    iio.imwrite(out_depth_file, depth_mm)

    # 4. Update scene_camera_3dlong.json
    dst_cam_file = dst_dir / "scene_camera_3dlong.json"
    dst_cams = {}
    if dst_cam_file.exists():
        try:
            dst_cams = json.loads(dst_cam_file.read_text(encoding="utf-8"))
        except Exception:
            dst_cams = {}
    dst_cams[str(scene_id)] = cams[str(scene_id)]
    dst_cam_file.write_text(json.dumps(dst_cams, indent=2), encoding="utf-8")

    # 5. Transform and update Ground Truth Poses (scene_gt_3dlong.json)
    src_gts = json.loads(src_gt_file.read_text(encoding="utf-8"))
    dst_gt_file = dst_dir / "scene_gt_3dlong.json"
    dst_gts = {}
    if dst_gt_file.exists():
        try:
            dst_gts = json.loads(dst_gt_file.read_text(encoding="utf-8"))
        except Exception:
            dst_gts = {}

    compensated_poses = []
    if str(scene_id) in src_gts:
        for entry in src_gts[str(scene_id)]:
            if entry.get("obj_id") != obj_id:
                continue
            R_bop = np.array(entry["cam_R_m2c"], dtype=np.float64).reshape(3, 3)
            t_bop = np.array(entry["cam_t_m2c"], dtype=np.float64)

            T_m2bop = np.eye(4)
            T_m2bop[:3, :3] = R_bop
            T_m2bop[:3, 3] = t_bop

            # Transform into native sensor coordinates
            T_m2nat = T_bop2nat @ T_m2bop
            R_nat = T_m2nat[:3, :3]
            t_nat = T_m2nat[:3, 3]

            print(f"    - GT Pose compensated: Z_bop={t_bop[2]:.2f} mm -> Z_nat={t_nat[2]:.2f} mm (dZ={t_nat[2]-t_bop[2]:+.2f} mm)")

            compensated_poses.append({
                "cam_R_m2c": [round(float(x), 6) for x in R_nat.flatten()],
                "cam_t_m2c": [round(float(x), 3) for x in t_nat.flatten()],
                "obj_id": int(obj_id),
            })
    dst_gts[str(scene_id)] = compensated_poses
    dst_gt_file.write_text(json.dumps(dst_gts, indent=2), encoding="utf-8")

    # 6. Update Ground Truth Info (scene_gt_info_3dlong.json)
    src_infos = json.loads(src_info_file.read_text(encoding="utf-8"))
    dst_info_file = dst_dir / "scene_gt_info_3dlong.json"
    dst_infos = {}
    if dst_info_file.exists():
        try:
            dst_infos = json.loads(dst_info_file.read_text(encoding="utf-8"))
        except Exception:
            dst_infos = {}

    if str(scene_id) in src_infos:
        dst_infos[str(scene_id)] = src_infos[str(scene_id)][:len(compensated_poses)]
    dst_info_file.write_text(json.dumps(dst_infos, indent=2), encoding="utf-8")

    # 7. Update target manifest CSV
    dst_manifest.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source", "scene_id", "image_id", "model_name", "obj_id",
        "gt_count", "split", "scene_gt_path", "scene_gt_info_path",
        "scene_camera_path", "depth_path", "cad_path", "models_info_path", "min_visib_fract"
    ]
    rows: List[Dict[str, Any]] = []
    if dst_manifest.exists():
        try:
            with dst_manifest.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = [row for row in reader if not (
                    row.get("image_id") == str(scene_id) and row.get("model_name") == model_name
                )]
        except Exception:
            rows = []

    cad_path = base_dir / "data" / "itoddmv_models" / "models" / f"obj_{obj_id:06d}.ply"
    models_info_path = base_dir / "data" / "itoddmv_models" / "models" / "models_info.json"

    new_row = {
        "source": "itoddmv_val_converted",
        "scene_id": 1,
        "image_id": scene_id,
        "model_name": model_name,
        "obj_id": obj_id,
        "gt_count": len(compensated_poses),
        "split": "train",
        "scene_gt_path": str(dst_gt_file.resolve()),
        "scene_gt_info_path": str(dst_info_file.resolve()),
        "scene_camera_path": str(dst_cam_file.resolve()),
        "depth_path": str(out_depth_file.resolve()),
        "cad_path": str(cad_path.resolve()),
        "models_info_path": str(models_info_path.resolve()) if models_info_path.exists() else "",
        "min_visib_fract": 0.1,
    }
    rows.append(new_row)

    with dst_manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"    - Updated manifest row: Scene {scene_id}, instances={len(compensated_poses)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert ITODD BOP validation scenes to native-consistent manual dataset format.")
    parser.add_argument("--scenes", type=str, help="Optional comma-separated scene IDs (default: 0,3,293,296,450,468)")
    parser.add_argument(
        "--dst-dir",
        type=Path,
        default=Path("data/itodd_manual_annotated/val/000001"),
        help="Destination manual dataset directory",
    )
    parser.add_argument(
        "--dst-manifest",
        type=Path,
        default=Path("data/manifests/itodd_manual_annotated/bop_manifest.csv"),
        help="Destination manual manifest CSV",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent

    if args.scenes:
        scene_ids = [int(s.strip()) for s in args.scenes.split(",") if s.strip()]
    else:
        scene_ids = list(TARGET_SCENES.keys())

    print("=" * 65)
    print(" CONVERTING ITODDMV_VAL SCENES -> NATIVE-CONSISTENT MANUAL FORMAT")
    print(f" Destination Directory: {args.dst_dir}")
    print(f" Destination Manifest:  {args.dst_manifest}")
    print(f" Target Scenes:         {scene_ids}")
    print("=" * 65)

    for s_id in scene_ids:
        if s_id not in TARGET_SCENES:
            print(f"[!] Warning: Scene {s_id} is not one of target objects (star/screw/bracket). Skipping.")
            continue
        model_name, obj_id = TARGET_SCENES[s_id]
        convert_scene(
            scene_id=s_id,
            model_name=model_name,
            obj_id=obj_id,
            base_dir=base_dir,
            dst_dir=args.dst_dir,
            dst_manifest=args.dst_manifest,
        )

    print("\n" + "=" * 65)
    print("[+] All requested scenes successfully converted and integrated into:")
    print(f"    {args.dst_dir.resolve()}")
    print(f"    {args.dst_manifest.resolve()}")
    print("=" * 65)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
