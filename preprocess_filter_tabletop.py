"""Offline deterministic tabletop plane removal and point cloud caching for BOP datasets.

Processes real sensor depth images once, fits tabletop plane with fixed random seed
and model-specific physical thresholds, and saves pure workpiece point clouds as
standard .ply files for 100% deterministic loading across optimization and evaluation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import open3d as o3d

# Ensure project root is in sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from bop_scene_loader import (
    backproject_depth,
    read_bop_camera,
    read_depth_image,
)
from config import RANSAC_TABLETOP_THRESHOLDS

DEFAULT_MANIFEST = SCRIPT_DIR / "data" / "manifests" / "itodd_manual_annotated" / "bop_manifest.csv"
DEFAULT_DEPTH_RANGE_M = (0.20, 0.95)
DEFAULT_DEPTH_STRIDE = 3


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def filter_tabletop_deterministic(
    points_xyz_m: np.ndarray,
    distance_threshold_m: float,
    seed: int = 42,
    ransac_n: int = 3,
    num_iterations: int = 1000,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float, float]]:
    """Segment and remove tabletop plane deterministically with locked seed.

    Returns:
        (filtered_points, inlier_indices, plane_model)
    """
    if len(points_xyz_m) < 100:
        return points_xyz_m, np.array([], dtype=int), (0.0, 0.0, 0.0, 0.0)

    # Lock Open3D and NumPy random state
    o3d.utility.random.seed(seed)
    np.random.seed(seed)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(points_xyz_m, dtype=np.float64))

    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold_m,
        ransac_n=ransac_n,
        num_iterations=num_iterations,
    )

    if len(inliers) == 0:
        return points_xyz_m, np.array([], dtype=int), tuple(plane_model)

    inliers_set = set(inliers)
    mask = np.ones(len(points_xyz_m), dtype=bool)
    mask[list(inliers_set)] = False
    filtered_points = points_xyz_m[mask]

    return filtered_points, np.array(inliers), tuple(plane_model)


def process_manifest(
    manifest_path: Path,
    out_dir: Path | None = None,
    seed: int = 42,
    depth_range_m: Sequence[float] = DEFAULT_DEPTH_RANGE_M,
    stride: int = DEFAULT_DEPTH_STRIDE,
    force: bool = False,
) -> Path:
    """Preprocess and cache tabletop-filtered point clouds for all unique scenes in manifest."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    # Read unique image queries
    unique_queries: dict[int, dict[str, Any]] = {}
    with open(manifest_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            im_id = int(row["image_id"])
            if im_id not in unique_queries:
                unique_queries[im_id] = {
                    "image_id": im_id,
                    "scene_id": int(row["scene_id"]),
                    "model_name": row["model_name"],
                    "obj_id": int(row["obj_id"]),
                    "depth_path": Path(row["depth_path"]),
                    "scene_camera_path": Path(row["scene_camera_path"]),
                }

    print(f"[*] Found {len(unique_queries)} unique scene images in {manifest_path.name}")

    first_depth = next(iter(unique_queries.values()))["depth_path"]
    target_dir = out_dir or (first_depth.parent.parent / "points_tabletop_filtered")
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"[*] Caching filtered .ply point clouds to: {target_dir.resolve()}")

    report_entries = []

    for im_id in sorted(unique_queries.keys()):
        q = unique_queries[im_id]
        model_name = q["model_name"]
        thresh = RANSAC_TABLETOP_THRESHOLDS.get(model_name, 0.0020)
        ply_name = f"{im_id:06d}_filtered.ply"
        ply_path = target_dir / ply_name

        if ply_path.exists() and not force:
            print(f"    [SKIP] Image {im_id:06d} ({model_name}): already exists -> {ply_name}")
            sha = sha256_file(ply_path)
            report_entries.append({
                "image_id": im_id,
                "model_name": model_name,
                "obj_id": q["obj_id"],
                "threshold_m": thresh,
                "ply_file": ply_name,
                "ply_sha256": sha,
                "status": "CACHED",
            })
            continue

        print(f"    [PROC] Image {im_id:06d} ({model_name}, threshold={thresh*1000:.1f}mm)...")
        camera = read_bop_camera(q["scene_camera_path"], im_id)
        depth_img = read_depth_image(q["depth_path"])
        raw_pts = backproject_depth(depth_img, camera, depth_range_m=depth_range_m, stride=stride)

        filtered_pts, inliers, plane = filter_tabletop_deterministic(
            raw_pts, distance_threshold_m=thresh, seed=seed
        )

        pcd_out = o3d.geometry.PointCloud()
        pcd_out.points = o3d.utility.Vector3dVector(np.ascontiguousarray(filtered_pts, dtype=np.float64))
        o3d.io.write_point_cloud(str(ply_path), pcd_out, write_ascii=False)

        sha = sha256_file(ply_path)
        removed_pct = len(inliers) / len(raw_pts) * 100.0 if len(raw_pts) > 0 else 0.0

        print(f"           Raw: {len(raw_pts)} pts | Removed: {len(inliers)} pts ({removed_pct:.1f}%) | Workpiece: {len(filtered_pts)} pts")

        report_entries.append({
            "image_id": im_id,
            "scene_id": q["scene_id"],
            "model_name": model_name,
            "obj_id": q["obj_id"],
            "threshold_m": thresh,
            "raw_point_count": len(raw_pts),
            "tabletop_point_count": len(inliers),
            "filtered_point_count": len(filtered_pts),
            "tabletop_fraction": removed_pct / 100.0,
            "plane_equation": list(plane),
            "ply_file": ply_name,
            "ply_sha256": sha,
            "status": "CREATED",
        })

    report_path = target_dir / "tabletop_filtering_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "manifest_source": str(manifest_path.resolve()),
                "total_scenes": len(unique_queries),
                "seed": seed,
                "depth_range_m": list(depth_range_m),
                "stride": stride,
                "scenes": report_entries,
            },
            f,
            indent=2,
        )
    print(f"\n[+] Preprocessing complete! Quality report saved to: {report_path.resolve()}")
    return report_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Preprocess and cache tabletop-removed point clouds deterministically."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"Path to bop_manifest.csv (default: {DEFAULT_MANIFEST})",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Target directory for .ply point clouds (default: <dataset>/points_tabletop_filtered/)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global random seed for deterministic RANSAC (default: 42)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=DEFAULT_DEPTH_STRIDE,
        help=f"Depth sampling stride (default: {DEFAULT_DEPTH_STRIDE})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing .ply files",
    )
    args = parser.parse_args()

    process_manifest(
        manifest_path=args.manifest,
        out_dir=args.out_dir,
        seed=args.seed,
        stride=args.stride,
        force=args.force,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
