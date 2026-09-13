"""Scene loading and tabletop filtering pipeline.

Loads scene point clouds from ITODD TIF depth images (X, Y, Z channels),
and applies RANSAC tabletop plane extraction to isolate foreground objects.
"""

import sys
import halcon as ha
import numpy as np


def load_scene(scene_image_prefix: str):
    """Load a scene from ITODD TIF depth images.

    Args:
        scene_image_prefix: Path prefix, e.g. ".../scene_0014/3d_long_baseline"
                            Files: {prefix}_x.tif, {prefix}_y.tif, {prefix}_z.tif

    Returns:
        HALCON HObjectModel3D of the scene
    """
    x = ha.read_image(scene_image_prefix + "_x.tif")
    y = ha.read_image(scene_image_prefix + "_y.tif")
    z = ha.read_image(scene_image_prefix + "_z.tif")
    return ha.xyz_to_object_model_3d(x, y, z)


def load_scene_ransac(scene_image_prefix: str, distance_threshold_m: float = 0.002):
    """Load an ITODD scene and filter out the tabletop plane using RANSAC.

    Args:
        scene_image_prefix: Path prefix to TIF files.
        distance_threshold_m: RANSAC distance threshold in meters.

    Returns:
        HALCON HObjectModel3D of the filtered scene.
    """
    import imageio.v3 as iio
    from bop_scene_loader import create_halcon_point_cloud, filter_points_ransac_tabletop

    x_img = np.asarray(iio.imread(scene_image_prefix + "_x.tif"))
    y_img = np.asarray(iio.imread(scene_image_prefix + "_y.tif"))
    z_img = np.asarray(iio.imread(scene_image_prefix + "_z.tif"))
    valid = np.isfinite(z_img) & (z_img > 0.05) & (z_img < 1.5)
    points = np.stack(
        [x_img[valid], y_img[valid], z_img[valid]], axis=-1
    ).astype(np.float64)

    filtered_pts = filter_points_ransac_tabletop(
        points, distance_threshold_m=distance_threshold_m
    )
    return create_halcon_point_cloud(filtered_pts)


def load_model(model_path: str):
    """Load a model PLY file and compute surface normals.

    Args:
        model_path: Path to the model PLY file

    Returns:
        model_3d: HALCON HObjectModel3D with surface normals
    """
    model_3d_raw = model_3d = None
    try:
        model_3d_raw, _ = ha.read_object_model_3d(model_path, "m", [], [])
        model_3d = ha.surface_normals_object_model_3d(
            model_3d_raw, "mls", [], []
        )
        return model_3d
    finally:
        if model_3d_raw is not None:
            active_exception = sys.exc_info()[0] is not None
            try:
                ha.clear_object_model_3d(model_3d_raw)
            except ha.HOperatorError:
                if not active_exception:
                    if model_3d is not None:
                        try:
                            ha.clear_object_model_3d(model_3d)
                        except ha.HOperatorError:
                            pass
                    raise
