# /// script
# dependencies = [
#     "imageio",
#     "numpy",
#     "open3d",
#     "tifffile",
# ]
# ///
"""
Export 3D Point Clouds to PLY files for CloudCompare inspection and geometric calibration.
Supports both BOP format (data/itoddmv_val) and ITODD Native format (data/3d_long_baseline).
"""

import argparse
import json
from pathlib import Path
import imageio.v3 as iio
import numpy as np
import open3d as o3d
from bop_scene_loader import backproject_depth, read_bop_camera, read_depth_image


def export_bop_scene_to_ply(
    scene_id: int,
    image_id: int,
    out_path: Path,
    stride: int = 1,
    scale: str = "m",
) -> None:
    """Export a BOP format scene image (from data/itoddmv_val) to a PLY point cloud."""
    scene_dir = Path("data/itoddmv_val/val") / f"{scene_id:06d}"
    depth_path = scene_dir / "depth_3dlong" / f"{image_id:06d}.tif"
    gray_path = scene_dir / "gray_3dlong" / f"{image_id:06d}.tif"
    camera_path = scene_dir / "scene_camera_3dlong.json"

    if not depth_path.exists():
        raise FileNotFoundError(f"BOP Depth image not found: {depth_path}")

    depth = read_depth_image(depth_path)
    camera = read_bop_camera(camera_path, image_id)
    points_xyz_m = backproject_depth(depth, camera, stride=stride)
    pts = points_xyz_m * 1000.0 if scale == "mm" else points_xyz_m

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    # If 2D image exists, add RGB colors
    if gray_path.exists():
        img_2d = iio.imread(gray_path)
        depth_array = np.asarray(depth)[::stride, ::stride]
        depth_m = depth_array.astype(np.float64) * camera.depth_scale / 1000.0
        valid = np.isfinite(depth_m) & (depth_m > 0.0)
        rows, columns = np.nonzero(valid)
        
        img_norm = np.asarray(img_2d)[::stride, ::stride].astype(np.float64)
        img_norm = (img_norm - img_norm.min()) / (img_norm.max() - img_norm.min() + 1e-6)
        pt_gray = img_norm[rows, columns]
        colors = np.column_stack([pt_gray, pt_gray, pt_gray])
        pcd.colors = o3d.utility.Vector3dVector(colors)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(out_path), pcd)
    print(f"[+] [BOP] Exported scene_{scene_id:06d}_im_{image_id:06d} ({len(pts):,} points, scale={scale}) -> {out_path}")


def export_native_scene_to_ply(
    scene_id: int,
    out_path: Path,
    stride: int = 1,
    scale: str = "m",
) -> None:
    """Export an ITODD native scene (from data/3d_long_baseline) to a PLY point cloud."""
    scene_dir = Path("data/3d_long_baseline/scenes") / f"scene_{scene_id:04d}"
    x_path = scene_dir / "3d_long_baseline_x.tif"
    y_path = scene_dir / "3d_long_baseline_y.tif"
    z_path = scene_dir / "3d_long_baseline_z.tif"
    img_l_path = scene_dir / "3d_long_baseline_l.tif"

    if not x_path.exists():
        raise FileNotFoundError(f"Native scene XYZ files not found in: {scene_dir}")

    x_img = np.asarray(iio.imread(x_path))
    y_img = np.asarray(iio.imread(y_path))
    z_img = np.asarray(iio.imread(z_path))
    valid = np.isfinite(z_img) & (z_img > 0.05)

    points_xyz_m = np.stack([x_img[valid], y_img[valid], z_img[valid]], axis=-1)[::stride]
    pts = points_xyz_m * 1000.0 if scale == "mm" else points_xyz_m

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    if img_l_path.exists():
        img_2d = iio.imread(img_l_path)
        l_img = np.asarray(img_2d)[valid][::stride].astype(np.float64)
        l_norm = (l_img - l_img.min()) / (l_img.max() - l_img.min() + 1e-6)
        colors = np.column_stack([l_norm, l_norm, l_norm])
        pcd.colors = o3d.utility.Vector3dVector(colors)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(out_path), pcd)
    print(f"[+] [Native] Exported scene_{scene_id:04d} ({len(pts):,} points, scale={scale}) -> {out_path}")


def export_manual_annotated_scene_to_ply(
    image_id: int,
    out_path: Path,
    scene_id: int = 1,
    stride: int = 1,
    scale: str = "m",
    export_gt: bool = True,
) -> None:
    """Export a manually annotated scene (from data/itodd_manual_annotated) to PLY.

    Reads depth from depth_3dlong/{image_id:06d}.tif, back-projects to 3D point cloud,
    attaches grayscale RGB texture from native sensors, and optionally exports
    CAD models transformed by Ground Truth poses to companion PLYs.
    """
    manual_dir = Path("data/itodd_manual_annotated/val") / f"{scene_id:06d}"
    depth_path = manual_dir / "depth_3dlong" / f"{image_id:06d}.tif"
    camera_path = manual_dir / "scene_camera_3dlong.json"
    gt_path = manual_dir / "scene_gt_3dlong.json"

    # Native grayscale image path for realistic point cloud coloring
    native_scene_dir = Path("data/3d_long_baseline/scenes") / f"scene_{image_id:04d}"
    img_l_path = native_scene_dir / "3d_long_baseline_l.tif"

    if not depth_path.exists():
        raise FileNotFoundError(f"Manual annotated depth image not found: {depth_path}")
    if not camera_path.exists():
        raise FileNotFoundError(f"Manual annotated camera json not found: {camera_path}")

    depth = read_depth_image(depth_path)
    camera = read_bop_camera(camera_path, image_id)
    points_xyz_m = backproject_depth(depth, camera, stride=stride)
    pts = points_xyz_m * 1000.0 if scale == "mm" else points_xyz_m

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    if img_l_path.exists():
        img_2d = iio.imread(img_l_path)
        depth_array = np.asarray(depth)[::stride, ::stride]
        depth_m = depth_array.astype(np.float64) * camera.depth_scale / 1000.0
        valid = np.isfinite(depth_m) & (depth_m > 0.0)
        rows, columns = np.nonzero(valid)

        img_norm = np.asarray(img_2d)[::stride, ::stride].astype(np.float64)
        img_norm = (img_norm - img_norm.min()) / (img_norm.max() - img_norm.min() + 1e-6)
        pt_gray = img_norm[rows, columns]
        colors = np.column_stack([pt_gray, pt_gray, pt_gray])
        pcd.colors = o3d.utility.Vector3dVector(colors)
    else:
        pcd.paint_uniform_color([0.65, 0.70, 0.75])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(out_path), pcd)
    print(f"[+] [Manual] Exported scene_{scene_id:06d}_im_{image_id:06d} ({len(pts):,} points, scale={scale}) -> {out_path}")

    # Export Ground Truth CAD models if requested
    if export_gt and gt_path.exists():
        try:
            gt_doc = json.loads(gt_path.read_text(encoding="utf-8"))
            key = str(image_id)
            if key in gt_doc:
                for idx, entry in enumerate(gt_doc[key]):
                    obj_id = entry.get("obj_id")
                    R = np.array(entry["cam_R_m2c"], dtype=np.float64).reshape(3, 3)
                    t_mm = np.array(entry["cam_t_m2c"], dtype=np.float64)
                    cad_file = Path("data/itoddmv_models/models") / f"obj_{obj_id:06d}.ply"
                    if not cad_file.exists():
                        cad_file = Path("data/base_package/models/cad_models") / f"obj_{obj_id:06d}.ply"
                    if cad_file.exists():
                        cad_mesh = o3d.io.read_triangle_mesh(str(cad_file))
                        if len(cad_mesh.triangles) > 0:
                            cad_mesh.compute_vertex_normals()
                            T = np.eye(4)
                            T[:3, :3] = R
                            if scale == "mm":
                                T[:3, 3] = t_mm
                            else:
                                cad_mesh.scale(1.0 / 1000.0, center=np.array([0.0, 0.0, 0.0]))
                                T[:3, 3] = t_mm / 1000.0
                            cad_mesh.transform(T)
                            cad_mesh.paint_uniform_color([0.15, 0.85, 0.25])
                            gt_out = out_path.parent / f"{out_path.stem}_gt_{idx:02d}_obj{obj_id:02d}.ply"
                            o3d.io.write_triangle_mesh(str(gt_out), cad_mesh)
                            print(f"    - [GT CAD] Exported GT #{idx} (obj_{obj_id:02d}) -> {gt_out}")
        except Exception as err:
            print(f"[!] Warning exporting GT CAD models for image {image_id}: {err}")


def load_scene_ids_from_file(file_path: Path) -> list[int]:
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
    parser = argparse.ArgumentParser(description="Export BOP itoddmv_val, ITODD Native, and Manually Annotated scenes to PLY point clouds.")
    parser.add_argument("--dataset-type", choices=["all", "bop", "native", "manual"], default="all", help="Dataset to export")
    parser.add_argument("--stride", type=int, default=1, help="Point cloud downsampling stride (1=full resolution, 2=half resolution)")
    parser.add_argument("--scale", choices=["m", "mm"], default="m", help="Point cloud scale: 'm' (meters) or 'mm' (millimeters)")
    parser.add_argument("--scenes", type=str, help="Comma-separated scene IDs for native dataset (e.g. '0,3,293,450')")
    parser.add_argument("--scene-list", type=Path, help="Path to scene list text file for native dataset")
    parser.add_argument("--bop-images", type=str, default="0,3,293,296,450,468", help="Comma-separated BOP image IDs to export")
    parser.add_argument("--manual-images", type=str, default="", help="Comma-separated manual annotated image IDs (default: auto-detect)")
    parser.add_argument("--export-gt", action="store_true", default=True, help="Export transformed CAD model PLYs at Ground Truth poses")
    parser.add_argument("--no-gt", dest="export_gt", action="store_false", help="Do not export CAD models at GT poses")
    parser.add_argument("--out-dir", type=Path, default=Path("data/exported_ply"), help="Output directory for PLY files")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Export BOP itoddmv_val scenes
    if args.dataset_type in ("all", "bop"):
        bop_dir = args.out_dir / "itoddmv_val"
        im_ids = [int(i.strip()) for i in args.bop_images.split(",") if i.strip()]
        print(f"\n=== Exporting {len(im_ids)} BOP itoddmv_val Scenes to PLY ===")
        for im_id in im_ids:
            try:
                export_bop_scene_to_ply(
                    scene_id=1,
                    image_id=im_id,
                    out_path=bop_dir / f"itoddmv_val_scene1_im{im_id:06d}.ply",
                    stride=args.stride,
                    scale=args.scale,
                )
            except Exception as e:
                print(f"[!] Error exporting BOP image {im_id}: {e}")

    # 2. Export ITODD Native scenes
    if args.dataset_type in ("all", "native"):
        native_dir = args.out_dir / "native_scenes"
        if args.scenes:
            scene_ids = [int(s.strip()) for s in args.scenes.split(",") if s.strip()]
        elif args.scene_list:
            scene_ids = load_scene_ids_from_file(args.scene_list)
        else:
            # Default representative scenes
            scene_ids = [0, 3, 293, 296, 450, 468]

        print(f"\n=== Exporting {len(scene_ids)} Native ITODD Scenes to PLY ===")
        for s_id in scene_ids:
            try:
                export_native_scene_to_ply(
                    scene_id=s_id,
                    out_path=native_dir / f"native_scene_{s_id:04d}.ply",
                    stride=args.stride,
                    scale=args.scale,
                )
            except Exception as e:
                print(f"[!] Error exporting Native scene {s_id}: {e}")

    # 3. Export Manually Annotated scenes
    if args.dataset_type in ("all", "manual"):
        manual_dir = args.out_dir / "manual_annotated"
        manual_src_depth = Path("data/itodd_manual_annotated/val/000001/depth_3dlong")

        if args.manual_images:
            man_ids = [int(i.strip()) for i in args.manual_images.split(",") if i.strip()]
        elif manual_src_depth.exists():
            man_ids = sorted([
                int(f.stem) for f in manual_src_depth.glob("*.tif") if f.stem.isdigit()
            ])
        else:
            man_ids = [8, 10, 13, 164, 749, 456, 461, 481, 476, 488, 299, 302, 305]

        if man_ids:
            print(f"\n=== Exporting {len(man_ids)} Manually Annotated Scenes to PLY ===")
            for im_id in man_ids:
                try:
                    export_manual_annotated_scene_to_ply(
                        image_id=im_id,
                        out_path=manual_dir / f"manual_scene1_im{im_id:06d}.ply",
                        scene_id=1,
                        stride=args.stride,
                        scale=args.scale,
                        export_gt=args.export_gt,
                    )
                except Exception as e:
                    print(f"[!] Error exporting Manual Annotated image {im_id}: {e}")

    print(f"\n[+] All requested point clouds exported successfully to: {args.out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
