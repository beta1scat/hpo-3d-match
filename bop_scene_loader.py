"""Load BOP depth scenes and ground-truth poses without GT-based cropping.

BOP translations and ``depth_scale``-scaled depths are expressed in
millimetres. Scene points returned by this module are expressed in metres,
while ground-truth translations retain the millimetre convention required by
``evaluation.PoseRecord``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import imageio.v3 as iio
import numpy as np

from evaluation import PoseRecord


DEFAULT_MIN_VISIB_FRACT = 0.1


@dataclass(frozen=True)
class BOPCamera:
    """Camera intrinsics and BOP depth scale for one image."""

    intrinsic_matrix: np.ndarray
    depth_scale: float

    def __post_init__(self) -> None:
        intrinsic_matrix = np.array(
            self.intrinsic_matrix, dtype=np.float64, copy=True
        )
        if intrinsic_matrix.shape != (3, 3):
            raise ValueError(
                "intrinsic_matrix must have shape (3, 3), got "
                f"{intrinsic_matrix.shape}"
            )
        if not np.all(np.isfinite(intrinsic_matrix)):
            raise ValueError("intrinsic_matrix must contain only finite values")
        if intrinsic_matrix[0, 0] == 0.0 or intrinsic_matrix[1, 1] == 0.0:
            raise ValueError("camera focal lengths must be non-zero")
        if not np.isfinite(self.depth_scale) or self.depth_scale <= 0.0:
            raise ValueError("depth_scale must be finite and greater than zero")
        intrinsic_matrix.setflags(write=False)
        object.__setattr__(self, "intrinsic_matrix", intrinsic_matrix)
        object.__setattr__(self, "depth_scale", float(self.depth_scale))


@dataclass(frozen=True)
class BOPSceneData:
    """Pure-data representation of one object query in a BOP depth image."""

    scene_id: int
    image_id: int
    obj_id: int
    points_xyz_m: np.ndarray
    camera: BOPCamera
    ground_truths: tuple[PoseRecord, ...]

    def __post_init__(self) -> None:
        points = np.array(self.points_xyz_m, dtype=np.float64, copy=True)
        if points.ndim != 2 or points.shape[1:] != (3,):
            raise ValueError(f"points_xyz_m must have shape (N, 3), got {points.shape}")
        if not np.all(np.isfinite(points)):
            raise ValueError("points_xyz_m must contain only finite values")
        points.setflags(write=False)
        object.__setattr__(self, "points_xyz_m", points)
        object.__setattr__(self, "ground_truths", tuple(self.ground_truths))


def _read_json_object(path: str | Path, label: str) -> dict[str, Any]:
    path = Path(path)
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in {label} {path}: line {exc.lineno}, "
            f"column {exc.colno}: {exc.msg}"
        ) from exc
    except OSError as exc:
        raise ValueError(f"Cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {label}: {path}")
    return value


def _integer_keyed_object(
    value: dict[str, Any], path: str | Path, label: str
) -> dict[int, Any]:
    result = {}
    original_keys = {}
    for key, item in value.items():
        try:
            integer_key = int(key)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Non-integer key {key!r} in {label}: {Path(path)}"
            ) from exc
        if integer_key < 0:
            raise ValueError(f"Negative key {integer_key} in {label}: {Path(path)}")
        if integer_key in result:
            raise ValueError(
                f"Duplicate integer key {integer_key} from original keys "
                f"{original_keys[integer_key]!r} and {key!r} in {label}: {Path(path)}"
            )
        result[integer_key] = item
        original_keys[integer_key] = key
    return result


def _image_entry(
    document: dict[int, Any], image_id: int, path: str | Path, label: str
) -> Any:
    if image_id not in document:
        raise ValueError(f"Image {image_id} is missing from {label}: {Path(path)}")
    return document[image_id]


def read_bop_ground_truths(
    scene_gt_path: str | Path,
    scene_gt_info_path: str | Path,
    scene_id: int,
    image_id: int,
    obj_id: int,
    min_visib_fract: float = DEFAULT_MIN_VISIB_FRACT,
) -> list[PoseRecord]:
    """Read sufficiently visible poses for ``obj_id`` in one BOP scene image.

    Each pose uses its zero-based index in the image annotation list as its
    ``PoseRecord.record_id``. ``scene_id`` identifies the query because the BOP
    JSON document itself is scoped to a scene and does not store that ID.
    """

    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in (scene_id, image_id, obj_id)
    ):
        raise ValueError(
            "scene_id, image_id, and obj_id must be non-negative integers"
        )
    if (
        isinstance(min_visib_fract, bool)
        or not isinstance(min_visib_fract, (int, float))
        or not np.isfinite(min_visib_fract)
        or not 0.0 <= min_visib_fract <= 1.0
    ):
        raise ValueError("min_visib_fract must be finite and between 0 and 1")

    document = _integer_keyed_object(
        _read_json_object(scene_gt_path, "BOP scene_gt.json"),
        scene_gt_path,
        "BOP scene_gt.json",
    )
    info_document = _integer_keyed_object(
        _read_json_object(scene_gt_info_path, "BOP scene_gt_info.json"),
        scene_gt_info_path,
        "BOP scene_gt_info.json",
    )
    if set(document) != set(info_document):
        gt_only = sorted(set(document) - set(info_document))
        info_only = sorted(set(info_document) - set(document))
        raise ValueError(
            "Image keys are not aligned between BOP scene_gt.json and "
            f"scene_gt_info.json; GT-only={gt_only}, GT-info-only={info_only}"
        )
    for aligned_image_id in sorted(document):
        aligned_annotations = document[aligned_image_id]
        aligned_infos = info_document[aligned_image_id]
        if not isinstance(aligned_annotations, list):
            raise ValueError(
                f"Expected an annotation list for image {aligned_image_id} in "
                f"{scene_gt_path}"
            )
        if not isinstance(aligned_infos, list):
            raise ValueError(
                f"Expected a GT-info list for image {aligned_image_id} in "
                f"{scene_gt_info_path}"
            )
        if len(aligned_annotations) != len(aligned_infos):
            raise ValueError(
                f"GT/GT-info index alignment mismatch at image {aligned_image_id}: "
                f"{len(aligned_annotations)} annotations in {scene_gt_path} but "
                f"{len(aligned_infos)} entries in {scene_gt_info_path}"
            )
    annotations = _image_entry(
        document, image_id, scene_gt_path, "BOP scene_gt.json"
    )
    infos = _image_entry(
        info_document, image_id, scene_gt_info_path, "BOP scene_gt_info.json"
    )

    poses = []
    for gt_index, (annotation, info) in enumerate(zip(annotations, infos)):
        if not isinstance(annotation, dict):
            raise ValueError(
                f"Expected an object at GT index {gt_index} for image {image_id} "
                f"in {scene_gt_path}"
            )
        if not isinstance(info, dict):
            raise ValueError(
                f"Expected an object at GT-info index {gt_index} for image "
                f"{image_id} in {scene_gt_info_path}"
            )
        visib_fract = info.get("visib_fract")
        if (
            isinstance(visib_fract, bool)
            or not isinstance(visib_fract, (int, float))
            or not np.isfinite(visib_fract)
            or not 0.0 <= visib_fract <= 1.0
        ):
            raise ValueError(
                f"visib_fract must be finite and between 0 and 1 at image "
                f"{image_id}, GT index {gt_index} in {scene_gt_info_path}"
            )
        if annotation.get("obj_id") != obj_id:
            continue
        if visib_fract < min_visib_fract:
            continue
        rotation = annotation.get("cam_R_m2c")
        translation = annotation.get("cam_t_m2c")
        if not isinstance(rotation, list) or len(rotation) != 9:
            raise ValueError(
                f"cam_R_m2c must contain 9 items at image {image_id}, "
                f"GT index {gt_index} in {scene_gt_path}"
            )
        if not isinstance(translation, list) or len(translation) != 3:
            raise ValueError(
                f"cam_t_m2c must contain 3 millimetre values at image {image_id}, "
                f"GT index {gt_index} in {scene_gt_path}"
            )
        try:
            poses.append(
                PoseRecord(
                    translation_mm=np.asarray(translation, dtype=np.float64),
                    rotation=np.asarray(rotation, dtype=np.float64).reshape(3, 3),
                    record_id=gt_index,
                )
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid pose at image {image_id}, GT index {gt_index} in "
                f"{scene_gt_path}: {exc}"
            ) from exc
    return poses


def read_bop_camera(
    scene_camera_path: str | Path, image_id: int
) -> BOPCamera:
    """Read ``cam_K`` and ``depth_scale`` for one BOP image."""

    if isinstance(image_id, bool) or not isinstance(image_id, int) or image_id < 0:
        raise ValueError("image_id must be a non-negative integer")
    document = _integer_keyed_object(
        _read_json_object(scene_camera_path, "BOP scene_camera.json"),
        scene_camera_path,
        "BOP scene_camera.json",
    )
    camera = _image_entry(
        document, image_id, scene_camera_path, "BOP scene_camera.json"
    )
    if not isinstance(camera, dict):
        raise ValueError(
            f"Expected a camera object for image {image_id} in {scene_camera_path}"
        )
    cam_k = camera.get("cam_K")
    if not isinstance(cam_k, list) or len(cam_k) != 9:
        raise ValueError(
            f"cam_K must contain 9 items for image {image_id} in {scene_camera_path}"
        )
    if "depth_scale" not in camera:
        raise ValueError(
            f"depth_scale is missing for image {image_id} in {scene_camera_path}"
        )
    try:
        intrinsic_matrix = np.asarray(cam_k, dtype=np.float64).reshape(3, 3)
        depth_scale = float(camera["depth_scale"])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid camera values for image {image_id} in {scene_camera_path}"
        ) from exc
    return BOPCamera(intrinsic_matrix, depth_scale)


def read_depth_image(depth_path: str | Path) -> np.ndarray:
    """Read a BOP PNG or TIF depth image as a two-dimensional NumPy array."""

    path = Path(depth_path)
    if path.suffix.lower() not in {".png", ".tif", ".tiff"}:
        raise ValueError(f"Depth image must be PNG or TIF: {path}")
    try:
        depth = np.asarray(iio.imread(path))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read depth image {path}: {exc}") from exc
    if depth.ndim != 2:
        raise ValueError(
            f"Depth image must be two-dimensional, got {depth.shape}: {path}"
        )
    return depth


def backproject_depth(
    depth: np.ndarray,
    camera: BOPCamera,
    depth_range_m: Sequence[float] | None = None,
    stride: int = 1,
) -> np.ndarray:
    """Back-project valid depth pixels to camera-frame XYZ points in metres.

    ``depth_range_m`` is an optional inclusive global range applied to scaled
    camera-frame Z depth. It is independent of ground-truth object poses.
    ``stride`` is an optional positive integer for 2D isotropic grid downsampling.
    """

    if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
        raise ValueError(f"stride must be a positive integer >= 1, got {stride!r}")

    depth_array = np.asarray(depth)
    if depth_array.ndim != 2:
        raise ValueError(
            f"depth must be a two-dimensional array, got {depth_array.shape}"
        )
    if stride > 1:
        depth_array = depth_array[::stride, ::stride]
        intrinsics_scaled = camera.intrinsic_matrix.copy()
        intrinsics_scaled[0, 0] /= stride
        intrinsics_scaled[1, 1] /= stride
        intrinsics_scaled[0, 2] /= stride
        intrinsics_scaled[1, 2] /= stride
        camera = BOPCamera(intrinsics_scaled, camera.depth_scale)

    minimum_depth = maximum_depth = None
    if depth_range_m is not None:
        if len(depth_range_m) != 2:
            raise ValueError("depth_range_m must contain minimum and maximum depth")
        minimum_depth, maximum_depth = map(float, depth_range_m)
        if (
            not np.isfinite(minimum_depth)
            or not np.isfinite(maximum_depth)
            or minimum_depth < 0.0
            or minimum_depth > maximum_depth
        ):
            raise ValueError(
                "depth_range_m must be a finite non-negative (minimum, maximum) pair"
            )

    depth_m = (
        np.asarray(depth_array, dtype=np.float64) * camera.depth_scale / 1000.0
    )
    valid = np.isfinite(depth_m) & (depth_m > 0.0)
    if minimum_depth is not None:
        valid &= depth_m >= minimum_depth
        valid &= depth_m <= maximum_depth

    rows, columns = np.nonzero(valid)
    z = depth_m[rows, columns]
    intrinsics = camera.intrinsic_matrix
    x = (
        (columns.astype(np.float64) - intrinsics[0, 2])
        * z
        / intrinsics[0, 0]
    )
    y = (rows.astype(np.float64) - intrinsics[1, 2]) * z / intrinsics[1, 1]
    points = np.column_stack((x, y, z))
    return points[np.all(np.isfinite(points), axis=1)]


def filter_points_roi(
    points_xyz_m: np.ndarray,
    roi: Any = None,
    *,
    is_bop: bool = True,
) -> np.ndarray:
    """Filter camera-frame 3D points by ROI bounding box, retaining camera coordinates."""
    if len(points_xyz_m) == 0:
        return points_xyz_m

    if roi is None:
        from config import ROIConfig

        roi = ROIConfig()

    roi_mat = (
        np.array(roi.matrix, dtype=np.float64)
        if hasattr(roi, "matrix") and roi.matrix is not None
        else None
    )
    if roi_mat is None:
        return points_xyz_m

    inv_roi_mat = np.linalg.inv(roi_mat)
    pts_h = np.column_stack([points_xyz_m, np.ones(len(points_xyz_m))])
    pts_roi = (inv_roi_mat @ pts_h.T).T[:, :3]

    if hasattr(roi, "get_z_range"):
        z_min, z_max = roi.get_z_range(is_bop=is_bop)
    else:
        z_min, z_max = roi.z_range

    in_roi = (
        (pts_roi[:, 0] >= roi.x_range[0])
        & (pts_roi[:, 0] <= roi.x_range[1])
        & (pts_roi[:, 1] >= roi.y_range[0])
        & (pts_roi[:, 1] <= roi.y_range[1])
        & (pts_roi[:, 2] >= z_min)
        & (pts_roi[:, 2] <= z_max)
    )
    return points_xyz_m[in_roi]


def load_bop_scene_data(
    scene_gt_path: str | Path,
    scene_gt_info_path: str | Path,
    scene_camera_path: str | Path,
    depth_path: str | Path,
    scene_id: int,
    image_id: int,
    obj_id: int,
    min_visib_fract: float = DEFAULT_MIN_VISIB_FRACT,
    depth_range_m: Sequence[float] | None = None,
    stride: int = 1,
    use_roi: bool = False,
    roi: Any = None,
) -> BOPSceneData:
    """Load a BOP scene query without importing or requiring HALCON."""

    camera = read_bop_camera(scene_camera_path, image_id)
    ground_truths = read_bop_ground_truths(
        scene_gt_path,
        scene_gt_info_path,
        scene_id,
        image_id,
        obj_id,
        min_visib_fract=min_visib_fract,
    )
    points = backproject_depth(
        read_depth_image(depth_path),
        camera,
        depth_range_m=depth_range_m,
        stride=stride,
    )
    if use_roi:
        points = filter_points_roi(points, roi=roi, is_bop=True)
    return BOPSceneData(
        scene_id=scene_id,
        image_id=image_id,
        obj_id=obj_id,
        points_xyz_m=points,
        camera=camera,
        ground_truths=tuple(ground_truths),
    )


def create_halcon_point_cloud(points_xyz_m: np.ndarray):
    """Create a HALCON object model, importing HALCON only when requested."""

    points = np.asarray(points_xyz_m, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (3,):
        raise ValueError(f"points_xyz_m must have shape (N, 3), got {points.shape}")
    if not np.all(np.isfinite(points)):
        raise ValueError("points_xyz_m must contain only finite values")

    import halcon as ha

    return ha.gen_object_model_3d_from_points(
        points[:, 0].tolist(), points[:, 1].tolist(), points[:, 2].tolist()
    )


def load_bop_scene(
    scene_gt_path: str | Path,
    scene_gt_info_path: str | Path,
    scene_camera_path: str | Path,
    depth_path: str | Path,
    scene_id: int,
    image_id: int,
    obj_id: int,
    min_visib_fract: float = DEFAULT_MIN_VISIB_FRACT,
    depth_range_m: Sequence[float] | None = None,
    stride: int = 1,
    use_roi: bool = False,
    roi: Any = None,
) -> tuple[object, list[PoseRecord]]:
    """Return a HALCON scene point cloud and matching BOP GT pose list."""

    data = load_bop_scene_data(
        scene_gt_path=scene_gt_path,
        scene_gt_info_path=scene_gt_info_path,
        scene_camera_path=scene_camera_path,
        depth_path=depth_path,
        scene_id=scene_id,
        image_id=image_id,
        obj_id=obj_id,
        min_visib_fract=min_visib_fract,
        depth_range_m=depth_range_m,
        stride=stride,
        use_roi=use_roi,
        roi=roi,
    )
    return create_halcon_point_cloud(data.points_xyz_m), list(data.ground_truths)

