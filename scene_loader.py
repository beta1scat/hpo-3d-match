"""Scene loading and ROI filtering pipeline.

Loads scene point clouds from ITODD TIF depth images (X, Y, Z channels),
applies ROI pose transform and bounding box crop per Section 2.3 of the paper.
"""

import sys

import halcon as ha
import numpy as np
from kinematics import transXYZ, rotX, rotY, rotZ
from config import ROIConfig


def compute_roi_transform(roi: ROIConfig):
    """Compute the ROI 4x4 homogeneous transform and its HALCON inverse pose.

    Returns:
        roi_mat:     4x4 numpy array, ROI pose in camera frame
        roi_pose_inv: HALCON pose (inverted), used to transform scene into ROI frame
    """
    translation = transXYZ(roi.tx, roi.ty, roi.tz)
    rotation = rotX(roi.rx).dot(rotY(roi.ry)).dot(rotZ(roi.rz))
    roi_mat = translation.dot(rotation)

    halcon_pose = ha.create_pose(
        roi.tx,
        roi.ty,
        roi.tz,
        roi.rx,
        roi.ry,
        roi.rz,
        "Rp+T",
        "gba",
        "point",
    )
    hom_mat = ha.pose_to_hom_mat3d(halcon_pose)
    hom_mat_inv = ha.hom_mat3d_invert(hom_mat)
    roi_pose_inv = ha.hom_mat3d_to_pose(hom_mat_inv)

    return roi_mat, roi_pose_inv


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


def filter_scene_roi(scene_3d, roi_pose_inv, roi: ROIConfig):
    """Apply ROI transform and bounding box crop to a scene point cloud.

    Steps:
      1. Transform scene to ROI coordinate frame (inverse of ROI pose)
      2. Crop X, Y, Z within the bounding box

    Args:
        scene_3d:     HALCON HObjectModel3D
        roi_pose_inv: HALCON pose (inverted ROI transform)
        roi:          ROIConfig with bounding box ranges

    Returns:
        HALCON HObjectModel3D of the cropped scene
    """
    scene_transformed = scene_x = scene_y = scene_roi = result = None
    try:
        scene_transformed = ha.rigid_trans_object_model_3d(scene_3d, roi_pose_inv)
        scene_x = ha.select_points_object_model_3d(
            scene_transformed, ["point_coord_x"], roi.x_range[0], roi.x_range[1]
        )
        scene_y = ha.select_points_object_model_3d(
            scene_x, ["point_coord_y"], roi.y_range[0], roi.y_range[1]
        )
        scene_roi = ha.select_points_object_model_3d(
            scene_y, ["point_coord_z"], roi.z_range[0], roi.z_range[1]
        )
        result = scene_roi
        scene_roi = None
        return result
    finally:
        active_exception = sys.exc_info()[0] is not None
        cleanup_error = None
        for handle in (scene_roi, scene_y, scene_x, scene_transformed):
            if handle is not None:
                try:
                    ha.clear_object_model_3d(handle)
                except ha.HOperatorError as exc:
                    if cleanup_error is None:
                        cleanup_error = exc
        if cleanup_error is not None and not active_exception:
            if result is not None:
                try:
                    ha.clear_object_model_3d(result)
                except ha.HOperatorError:
                    pass
            raise cleanup_error


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
