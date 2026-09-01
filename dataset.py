"""Build independent manifests for BOP ITODD and original MVTec ITODD."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


TARGET_OBJECT_IDS = {
    "bracket_planar": 5,
    "screw_black": 24,
    "star": 25,
}
OBJECT_NAMES_BY_ID = {obj_id: name for name, obj_id in TARGET_OBJECT_IDS.items()}
TARGET_CAD_FILES = {
    "bracket_planar": "bracket_planar.ply",
    "screw_black": "screw_black.ply",
    "star": "star.ply",
}

SPLIT_NAMES = ("train", "dev", "test")
DEFAULT_TRAIN_RATIO = 0.8
MIN_VISIB_FRACT = 0.1
BOP_MANIFEST_FIELDS = (
    "source",
    "scene_id",
    "image_id",
    "model_name",
    "obj_id",
    "gt_count",
    "split",
    "scene_gt_path",
    "scene_gt_info_path",
    "scene_camera_path",
    "depth_path",
    "cad_path",
    "models_info_path",
    "min_visib_fract",
)
ITODD_EXTERNAL_MANIFEST_FIELDS = (
    "scene_id",
    "model_name",
    "obj_id",
    "split",
    "cad_path",
    "x_path",
    "y_path",
    "z_path",
)


class DatasetError(ValueError):
    """Raised when an input dataset is missing, malformed, or inconsistent."""


@dataclass(frozen=True)
class BOPScene:
    """The annotations and camera metadata for one standard BOP scene."""

    scene_id: int
    scene_dir: Path
    scene_gt_path: Path
    scene_gt_info_path: Path
    scene_camera_path: Path
    scene_gt: dict[int, list[dict[str, Any]]]
    scene_gt_info: dict[int, list[dict[str, Any]]]
    scene_camera: dict[int, dict[str, Any]]


def _require_file(path: Path, label: str) -> Path:
    path = path.resolve()
    if not path.is_file():
        raise DatasetError(f"Missing {label}: {path}")
    return path


def _require_directory(path: Path, label: str) -> Path:
    path = path.resolve()
    if not path.is_dir():
        raise DatasetError(f"Missing {label} directory: {path}")
    return path


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    path = _require_file(path, label)
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except json.JSONDecodeError as exc:
        raise DatasetError(
            f"Invalid JSON in {label} {path}: line {exc.lineno}, column {exc.colno}: "
            f"{exc.msg}"
        ) from exc
    except OSError as exc:
        raise DatasetError(f"Cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise DatasetError(f"Expected a JSON object in {label}: {path}")
    return value


def _integer_keyed_object(
    value: Mapping[str, Any], path: Path, label: str
) -> dict[int, Any]:
    result = {}
    for key, item in value.items():
        try:
            integer_key = int(key)
        except (TypeError, ValueError) as exc:
            raise DatasetError(f"Non-integer key {key!r} in {label}: {path}") from exc
        if integer_key < 0:
            raise DatasetError(f"Negative key {integer_key} in {label}: {path}")
        if integer_key in result:
            raise DatasetError(f"Duplicate integer key {integer_key} in {label}: {path}")
        result[integer_key] = item
    return result


def _validate_annotation_suffix(annotation_suffix: str) -> None:
    if (
        annotation_suffix
        and re.fullmatch(r"_[A-Za-z0-9_-]+", annotation_suffix) is None
    ):
        raise DatasetError(
            "BOP annotation suffix must be empty or match _[A-Za-z0-9_-]+"
        )


def load_bop_scene(
    scene_dir: str | Path, annotation_suffix: str = ""
) -> BOPScene:
    """Read aligned BOP GT, GT-info, and camera JSON files for one suffix."""

    _validate_annotation_suffix(annotation_suffix)
    scene_dir = _require_directory(Path(scene_dir), "BOP scene")
    try:
        scene_id = int(scene_dir.name)
    except ValueError as exc:
        raise DatasetError(
            f"BOP scene directory name must be an integer, got: {scene_dir}"
        ) from exc

    gt_filename = f"scene_gt{annotation_suffix}.json"
    gt_info_filename = f"scene_gt_info{annotation_suffix}.json"
    camera_filename = f"scene_camera{annotation_suffix}.json"
    gt_path = scene_dir / gt_filename
    gt_info_path = scene_dir / gt_info_filename
    camera_path = scene_dir / camera_filename
    raw_gt = _integer_keyed_object(
        _read_json_object(gt_path, f"BOP {gt_filename}"), gt_path, gt_filename
    )
    raw_camera = _integer_keyed_object(
        _read_json_object(camera_path, f"BOP {camera_filename}"),
        camera_path,
        camera_filename,
    )
    raw_gt_info = _integer_keyed_object(
        _read_json_object(gt_info_path, f"BOP {gt_info_filename}"),
        gt_info_path,
        gt_info_filename,
    )

    if set(raw_gt) != set(raw_gt_info):
        gt_only = sorted(set(raw_gt) - set(raw_gt_info))
        info_only = sorted(set(raw_gt_info) - set(raw_gt))
        raise DatasetError(
            f"Image keys are not aligned between {gt_path.resolve()} and "
            f"{gt_info_path.resolve()}; GT-only={gt_only}, GT-info-only={info_only}"
        )

    scene_gt: dict[int, list[dict[str, Any]]] = {}
    scene_gt_info: dict[int, list[dict[str, Any]]] = {}
    scene_camera: dict[int, dict[str, Any]] = {}
    for image_id, annotations in raw_gt.items():
        if not isinstance(annotations, list) or not all(
            isinstance(annotation, dict) for annotation in annotations
        ):
            raise DatasetError(
                f"Expected an annotation list at image {image_id} in {gt_path.resolve()}"
            )
        if image_id not in raw_camera:
            raise DatasetError(
                f"Image {image_id} exists in {gt_path.resolve()} but not in "
                f"{camera_path.resolve()}"
            )
        infos = raw_gt_info[image_id]
        if not isinstance(infos, list) or not all(
            isinstance(info, dict) for info in infos
        ):
            raise DatasetError(
                f"Expected a GT-info list at image {image_id} in "
                f"{gt_info_path.resolve()}"
            )
        if len(annotations) != len(infos):
            raise DatasetError(
                f"GT/GT-info index alignment mismatch at image {image_id}: "
                f"{len(annotations)} annotations in {gt_path.resolve()} but "
                f"{len(infos)} entries in {gt_info_path.resolve()}"
            )
        scene_gt[image_id] = annotations
        scene_gt_info[image_id] = infos

    for image_id, camera in raw_camera.items():
        if not isinstance(camera, dict):
            raise DatasetError(
                f"Expected a camera object at image {image_id} in {camera_path.resolve()}"
            )
        scene_camera[image_id] = camera

    return BOPScene(
        scene_id=scene_id,
        scene_dir=scene_dir,
        scene_gt_path=gt_path.resolve(),
        scene_gt_info_path=gt_info_path.resolve(),
        scene_camera_path=camera_path.resolve(),
        scene_gt=scene_gt,
        scene_gt_info=scene_gt_info,
        scene_camera=scene_camera,
    )


def load_models_info(models_info_path: str | Path) -> dict[int, dict[str, Any]]:
    """Read a standard BOP ``models_info.json`` keyed by integer object ID."""

    path = Path(models_info_path)
    raw = _integer_keyed_object(
        _read_json_object(path, "BOP models_info.json"), path, "models_info.json"
    )
    models_info = {}
    for obj_id, info in raw.items():
        if not isinstance(info, dict):
            raise DatasetError(
                f"Expected a model object for ID {obj_id} in {path.resolve()}"
            )
        models_info[obj_id] = info
    return models_info


def read_scene_list(path: str | Path) -> list[int]:
    """Read one scene ID per line, ignoring blank text and ``#`` comments."""

    path = _require_file(Path(path), "ITODD scene list")
    scene_ids = []
    seen = set()
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                token = line.partition("#")[0].strip()
                if not token:
                    continue
                try:
                    scene_id = int(token)
                except ValueError as exc:
                    raise DatasetError(
                        f"Invalid scene ID {token!r} at {path}:{line_number}"
                    ) from exc
                if scene_id < 0:
                    raise DatasetError(
                        f"Negative scene ID {scene_id} at {path}:{line_number}"
                    )
                if scene_id in seen:
                    raise DatasetError(
                        f"Duplicate scene ID {scene_id} at {path}:{line_number}"
                    )
                seen.add(scene_id)
                scene_ids.append(scene_id)
    except OSError as exc:
        raise DatasetError(f"Cannot read ITODD scene list {path}: {exc}") from exc
    if not scene_ids:
        raise DatasetError(f"ITODD scene list contains no scene IDs: {path}")
    return scene_ids


def _validate_ratios(ratios: Sequence[float]) -> None:
    if len(ratios) != len(SPLIT_NAMES):
        raise DatasetError("Split ratios must contain train, dev, and test values")
    if any(not math.isfinite(ratio) or ratio < 0 for ratio in ratios):
        raise DatasetError(f"Split ratios must be finite and non-negative: {ratios}")
    if not math.isclose(sum(ratios), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise DatasetError(f"Split ratios must sum to 1.0, got {sum(ratios):.12g}")


def split_bop_groups(
    groups: Iterable[tuple[int, int]],
    ratios: Sequence[float] = (0.6, 0.2, 0.2),
    seed: int = 42,
) -> dict[str, list[tuple[int, int]]]:
    """Split unique ``(scene_id, image_id)`` groups deterministically."""

    _validate_ratios(ratios)
    shuffled = sorted(set(groups))
    random.Random(seed).shuffle(shuffled)
    exact_counts = [len(shuffled) * ratio for ratio in ratios]
    counts = [math.floor(count) for count in exact_counts]
    remainder = len(shuffled) - sum(counts)
    order = sorted(
        range(len(ratios)),
        key=lambda index: (-(exact_counts[index] - counts[index]), index),
    )
    for index in order[:remainder]:
        counts[index] += 1

    splits = {}
    offset = 0
    for name, count in zip(SPLIT_NAMES, counts):
        splits[name] = sorted(shuffled[offset : offset + count])
        offset += count
    return splits


def split_bop_scenes(
    scene_ids: Iterable[int], train_ratio: float = DEFAULT_TRAIN_RATIO, seed: int = 42
) -> dict[str, list[int]]:
    """Split complete scenes into deterministic train and development sets."""

    if not math.isfinite(train_ratio) or not 0.0 <= train_ratio <= 1.0:
        raise DatasetError("BOP train ratio must be finite and between 0 and 1")
    shuffled = sorted(set(scene_ids))
    random.Random(seed).shuffle(shuffled)
    train_count = math.floor(len(shuffled) * train_ratio + 0.5)
    return {
        "train": sorted(shuffled[:train_count]),
        "dev": sorted(shuffled[train_count:]),
        "test": [],
    }


def _sampling_seed(seed: int, split: str, obj_id: int, purpose: str) -> int:
    digest = hashlib.sha256(
        f"{seed}:{split}:{obj_id}:{purpose}".encode("ascii")
    ).digest()
    return int.from_bytes(digest[:8], "big")


def _sample_queries_proportionally_by_scene(
    rows: Sequence[dict[str, str | int | float]], limit: int, seed: int
) -> list[dict[str, str | int | float]]:
    """Select a deterministic scene-proportional subset of object queries."""

    if limit >= len(rows):
        return list(rows)
    split = str(rows[0]["split"])
    obj_id = int(rows[0]["obj_id"])
    by_scene: dict[int, list[dict[str, str | int | float]]] = {}
    for row in rows:
        by_scene.setdefault(int(row["scene_id"]), []).append(row)

    total = len(rows)
    allocations = {
        scene_id: math.floor(limit * len(scene_rows) / total)
        for scene_id, scene_rows in by_scene.items()
    }
    remainder = limit - sum(allocations.values())
    tie_order = sorted(by_scene)
    random.Random(_sampling_seed(seed, split, obj_id, "allocation")).shuffle(
        tie_order
    )
    tie_rank = {scene_id: rank for rank, scene_id in enumerate(tie_order)}
    allocation_order = sorted(
        by_scene,
        key=lambda scene_id: (
            -(limit * len(by_scene[scene_id]) / total - allocations[scene_id]),
            tie_rank[scene_id],
        ),
    )
    for scene_id in allocation_order[:remainder]:
        allocations[scene_id] += 1

    selected = []
    for scene_id in sorted(by_scene):
        scene_rows = sorted(by_scene[scene_id], key=lambda row: int(row["image_id"]))
        random.Random(
            _sampling_seed(seed, split, obj_id, f"scene:{scene_id}")
        ).shuffle(scene_rows)
        selected.extend(scene_rows[: allocations[scene_id]])
    return selected


def _discover_bop_scenes(
    bop_scenes_root: Path, annotation_suffix: str
) -> list[BOPScene]:
    _validate_annotation_suffix(annotation_suffix)
    root = _require_directory(bop_scenes_root, "BOP scenes root")
    gt_filename = f"scene_gt{annotation_suffix}.json"
    gt_paths = sorted(root.rglob(gt_filename))
    if not gt_paths:
        raise DatasetError(f"No BOP {gt_filename} files found under: {root}")
    scenes = [
        load_bop_scene(path.parent, annotation_suffix=annotation_suffix)
        for path in gt_paths
    ]
    scene_ids = [scene.scene_id for scene in scenes]
    if len(scene_ids) != len(set(scene_ids)):
        raise DatasetError(
            f"Duplicate BOP scene directory IDs found under {root}; pass one BOP split "
            "as --bop-scenes-root"
        )
    return scenes


def _resolve_bop_depth(scene: BOPScene, image_id: int, depth_dir: str) -> Path:
    depth_root = scene.scene_dir / depth_dir
    stem = f"{image_id:06d}"
    candidates = [depth_root / f"{stem}.png", depth_root / f"{stem}.tif"]
    existing = [path.resolve() for path in candidates if path.is_file()]
    if not existing:
        expected = ", ".join(str(path.resolve()) for path in candidates)
        raise DatasetError(
            f"Missing BOP depth for scene {scene.scene_id}, image {image_id}; "
            f"expected one of: {expected}"
        )
    if len(existing) > 1:
        raise DatasetError(
            f"Ambiguous BOP depth for scene {scene.scene_id}, image {image_id}; "
            f"both files exist: {existing[0]}, {existing[1]}"
        )
    return existing[0]


def _validate_bop_models(bop_models_dir: Path) -> tuple[dict[str, Path], Path]:
    bop_models_dir = _require_directory(bop_models_dir, "BOP models")
    models_info_path = bop_models_dir / "models_info.json"
    models_info = load_models_info(models_info_path)
    missing_ids = sorted(set(TARGET_OBJECT_IDS.values()) - set(models_info))
    if missing_ids:
        raise DatasetError(
            f"Target object IDs missing from {models_info_path.resolve()}: {missing_ids}"
        )
    cad_paths = {
        name: _require_file(
            bop_models_dir / f"obj_{obj_id:06d}.ply",
            f"BOP CAD for {name} (object {obj_id})",
        )
        for name, obj_id in TARGET_OBJECT_IDS.items()
    }
    return cad_paths, models_info_path.resolve()


def _validate_query_limit(limit: int | None, split: str) -> None:
    if limit is not None and (
        isinstance(limit, bool) or not isinstance(limit, int) or limit < 0
    ):
        raise DatasetError(
            f"{split} query limit per object must be a non-negative integer"
        )


def _query_counts(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, dict[str, int]]:
    counts = {
        split: {model_name: 0 for model_name in TARGET_OBJECT_IDS}
        for split in SPLIT_NAMES
    }
    for row in rows:
        counts[str(row["split"])][str(row["model_name"])] += 1
    return counts


def build_bop_manifest(
    bop_scenes_root: str | Path,
    bop_models_dir: str | Path,
    source: str,
    depth_dir: str = "depth",
    annotation_suffix: str = "",
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    seed: int = 42,
    fixed_split: str | None = None,
    train_query_limit_per_object: int | None = None,
    dev_query_limit_per_object: int | None = None,
    test_query_limit_per_object: int | None = None,
    min_visib_fract: float = MIN_VISIB_FRACT,
    scene_ids: Sequence[int] | None = None,
) -> tuple[list[dict[str, str | int | float]], dict[str, Any]]:
    """Build a GT-bearing BOP manifest without using original ITODD IDs."""

    depth_path = Path(depth_dir)
    if (
        not depth_dir
        or depth_dir in (".", "..")
        or depth_path.is_absolute()
        or len(depth_path.parts) != 1
    ):
        raise DatasetError("BOP depth directory must be one relative directory name")
    if fixed_split is not None and fixed_split not in SPLIT_NAMES:
        raise DatasetError(f"Fixed BOP split must be one of {SPLIT_NAMES}: {fixed_split}")
    if not isinstance(source, str) or not source.strip():
        raise DatasetError("BOP source must be a non-empty string")
    if not math.isfinite(train_ratio) or not 0.0 <= train_ratio <= 1.0:
        raise DatasetError("BOP train ratio must be finite and between 0 and 1")
    if not math.isfinite(min_visib_fract) or not 0.0 <= min_visib_fract <= 1.0:
        raise DatasetError("Minimum visibility fraction must be between 0 and 1")
    limits = {
        "train": train_query_limit_per_object,
        "dev": dev_query_limit_per_object,
        "test": test_query_limit_per_object,
    }
    for split, limit in limits.items():
        _validate_query_limit(limit, split)

    _validate_annotation_suffix(annotation_suffix)
    cad_paths, models_info_path = _validate_bop_models(Path(bop_models_dir))
    scenes = _discover_bop_scenes(Path(bop_scenes_root), annotation_suffix)
    if scene_ids is not None:
        scene_set = {int(s) for s in scene_ids}
        scenes = [s for s in scenes if int(s.scene_id) in scene_set]
        if not scenes:
            raise DatasetError(f"No matching scenes found for scene IDs: {sorted(scene_set)}")
    pending_rows = []
    visibility_counts = {
        "annotations_total": 0,
        "target_annotations_total": 0,
        "target_annotations_included": 0,
        "target_annotations_excluded": 0,
    }
    for scene in scenes:
        for image_id, annotations in scene.scene_gt.items():
            target_counts: dict[int, int] = {}
            infos = scene.scene_gt_info[image_id]
            for annotation_index, (annotation, info) in enumerate(
                zip(annotations, infos)
            ):
                visibility_counts["annotations_total"] += 1
                obj_id = annotation.get("obj_id")
                if not isinstance(obj_id, int):
                    raise DatasetError(
                        f"Missing or non-integer obj_id at image {image_id}, GT index "
                        f"{annotation_index} in {scene.scene_gt_path}"
                    )
                visib_fract = info.get("visib_fract")
                if isinstance(visib_fract, bool) or not isinstance(
                    visib_fract, (int, float)
                ):
                    raise DatasetError(
                        f"Missing or non-numeric visib_fract at image {image_id}, "
                        f"GT index {annotation_index} in {scene.scene_gt_info_path}"
                    )
                visib_fract = float(visib_fract)
                if not math.isfinite(visib_fract) or not 0.0 <= visib_fract <= 1.0:
                    raise DatasetError(
                        f"visib_fract must be finite and between 0 and 1 at image "
                        f"{image_id}, GT index {annotation_index} in "
                        f"{scene.scene_gt_info_path}"
                    )
                if obj_id in OBJECT_NAMES_BY_ID:
                    visibility_counts["target_annotations_total"] += 1
                    if visib_fract < min_visib_fract:
                        visibility_counts["target_annotations_excluded"] += 1
                        continue
                    visibility_counts["target_annotations_included"] += 1
                    target_counts[obj_id] = target_counts.get(obj_id, 0) + 1
            if not target_counts:
                continue

            depth_path = _resolve_bop_depth(scene, image_id, depth_dir)
            for obj_id, gt_count in sorted(target_counts.items()):
                model_name = OBJECT_NAMES_BY_ID[obj_id]
                pending_rows.append(
                    {
                        "source": source.strip(),
                        "scene_id": scene.scene_id,
                        "image_id": image_id,
                        "model_name": model_name,
                        "obj_id": obj_id,
                        "gt_count": gt_count,
                        "scene_gt_path": str(scene.scene_gt_path),
                        "scene_gt_info_path": str(scene.scene_gt_info_path),
                        "scene_camera_path": str(scene.scene_camera_path),
                        "depth_path": str(depth_path),
                        "cad_path": str(cad_paths[model_name]),
                        "models_info_path": str(models_info_path),
                        "min_visib_fract": min_visib_fract,
                    }
                )
    if not pending_rows:
        raise DatasetError(
            f"No GT annotations for target object IDs "
            f"{sorted(TARGET_OBJECT_IDS.values())} were found under "
            f"{Path(bop_scenes_root).resolve()}"
        )

    eligible_scene_ids = sorted({int(row["scene_id"]) for row in pending_rows})
    if fixed_split is None:
        if len(eligible_scene_ids) > 1:
            splits = split_bop_scenes(
                eligible_scene_ids, train_ratio=train_ratio, seed=seed
            )
            mode = "generated_scene"
            split_by_scene = {
                scene_id: split_name
                for split_name, split_scenes in splits.items()
                for scene_id in split_scenes
            }
            unsampled_rows = [
                {**row, "split": split_by_scene[int(row["scene_id"])]}
                for row in pending_rows
            ]
        else:
            image_groups = sorted({(int(row["scene_id"]), int(row["image_id"])) for row in pending_rows})
            image_splits = split_bop_groups(
                image_groups,
                ratios=(train_ratio, 1.0 - train_ratio, 0.0),
                seed=seed,
            )
            split_by_image = {
                group: split_name
                for split_name, split_groups in image_splits.items()
                for group in split_groups
            }
            splits = image_splits
            mode = "generated_image"
            unsampled_rows = [
                {**row, "split": split_by_image[(int(row["scene_id"]), int(row["image_id"]))]}
                for row in pending_rows
            ]
        split_seed: int | None = seed
        split_train_ratio: float | None = train_ratio
    else:
        splits = {name: [] for name in SPLIT_NAMES}
        splits[fixed_split] = eligible_scene_ids
        mode = "fixed"
        split_seed = None
        split_train_ratio = None
        unsampled_rows = [{**row, "split": fixed_split} for row in pending_rows]
    before_counts = _query_counts(unsampled_rows)
    rows = []
    for split in SPLIT_NAMES:
        for obj_id in sorted(OBJECT_NAMES_BY_ID):
            object_rows = [
                row
                for row in unsampled_rows
                if row["split"] == split and int(row["obj_id"]) == obj_id
            ]
            limit = limits[split]
            if limit is not None and object_rows:
                object_rows = _sample_queries_proportionally_by_scene(
                    object_rows, limit, seed
                )
            rows.extend(object_rows)
    rows.sort(
        key=lambda row: (
            int(row["scene_id"]),
            int(row["image_id"]),
            int(row["obj_id"]),
        )
    )
    split_data = {
        "schema_version": 2,
        "source": source.strip(),
        "group_by": ["scene_id"],
        "annotation_suffix": annotation_suffix,
        "mode": mode,
        "seed": split_seed,
        "train_ratio": split_train_ratio,
        "min_visib_fract": min_visib_fract,
        "splits": splits,
        "visibility_counts": visibility_counts,
        "sampling_counts": {
            "limits_per_object": limits,
            "before": before_counts,
            "after": _query_counts(rows),
        },
    }
    return rows, split_data


def build_itodd_external_manifest(
    itodd_root: str | Path,
) -> list[dict[str, str | int]]:
    """Build the no-GT external-test manifest for original MVTec ITODD."""

    itodd_root = _require_directory(Path(itodd_root), "original ITODD root")
    rows = []
    for model_name, obj_id in TARGET_OBJECT_IDS.items():
        cad_path = _require_file(
            itodd_root
            / "base_package"
            / "models"
            / "cad_models"
            / TARGET_CAD_FILES[model_name],
            f"original ITODD CAD for {model_name}",
        )
        scene_list_path = (
            itodd_root
            / "base_package"
            / "models"
            / "scene_lists"
            / f"scene_list_{model_name}.txt"
        )
        for scene_id in read_scene_list(scene_list_path):
            scene_dir = (
                itodd_root
                / "3d_long_baseline"
                / "scenes"
                / f"scene_{scene_id:04d}"
            )
            xyz_paths = {
                axis: _require_file(
                    scene_dir / f"3d_long_baseline_{axis}.tif",
                    f"original ITODD {axis.upper()} file for scene {scene_id}",
                )
                for axis in ("x", "y", "z")
            }
            rows.append(
                {
                    "scene_id": scene_id,
                    "model_name": model_name,
                    "obj_id": obj_id,
                    "split": "external_test",
                    "cad_path": str(cad_path),
                    "x_path": str(xyz_paths["x"]),
                    "y_path": str(xyz_paths["y"]),
                    "z_path": str(xyz_paths["z"]),
                }
            )
    rows.sort(key=lambda row: (int(row["scene_id"]), str(row["model_name"])))
    return rows


def write_csv_manifest(
    rows: Iterable[Mapping[str, Any]],
    fields: Sequence[str],
    output_path: str | Path,
) -> None:
    """Write manifest rows using the requested stable CSV schema."""

    output_path = Path(output_path)
    try:
        with output_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    except OSError as exc:
        raise DatasetError(
            f"Cannot write CSV manifest {output_path.resolve()}: {exc}"
        ) from exc


def write_split(split_data: Mapping[str, Any], output_path: str | Path) -> None:
    """Write BOP split metadata as JSON."""

    output_path = Path(output_path)
    try:
        with output_path.open("w", encoding="utf-8") as stream:
            json.dump(split_data, stream, indent=2, sort_keys=True)
            stream.write("\n")
    except OSError as exc:
        raise DatasetError(
            f"Cannot write JSON split {output_path.resolve()}: {exc}"
        ) from exc


def _parse_train_ratio(value: str) -> float:
    try:
        ratio = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("train ratio must be a number") from exc
    if not math.isfinite(ratio) or not 0.0 <= ratio <= 1.0:
        raise argparse.ArgumentTypeError("train ratio must be between 0 and 1")
    return ratio


def _parse_query_limit(value: str) -> int:
    try:
        limit = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "query limit per object must be an integer"
        ) from exc
    if limit < 0:
        raise argparse.ArgumentTypeError(
            "query limit per object must be non-negative"
        )
    return limit


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build independent BOP and original ITODD manifests."
    )
    parser.add_argument(
        "--bop-scenes-root",
        type=Path,
        help="One BOP split directory containing numeric scene directories",
    )
    parser.add_argument(
        "--bop-models-dir",
        type=Path,
        help="BOP models directory containing models_info.json and obj_*.ply",
    )
    parser.add_argument(
        "--source", help="Dataset source label written to BOP manifest rows"
    )
    parser.add_argument("--bop-depth-dir", default="depth")
    parser.add_argument(
        "--bop-annotation-suffix",
        default="",
        help="Annotation filename suffix, for example _3dlong",
    )
    parser.add_argument("--bop-fixed-split", choices=SPLIT_NAMES)
    parser.add_argument("--itodd-root", type=Path, help="Original MVTec ITODD root")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--train-ratio",
        type=_parse_train_ratio,
        default=DEFAULT_TRAIN_RATIO,
    )
    parser.add_argument("--train-query-limit-per-object", type=_parse_query_limit)
    parser.add_argument("--dev-query-limit-per-object", type=_parse_query_limit)
    parser.add_argument("--test-query-limit-per-object", type=_parse_query_limit)
    parser.add_argument("--scenes", type=str, help="Comma-separated scene IDs to filter, e.g. '0' or '0,1,2'")
    args = parser.parse_args(argv)

    if args.bop_scenes_root is None and args.itodd_root is None:
        parser.error("provide --bop-scenes-root, --itodd-root, or both")
    if (args.bop_scenes_root is None) != (args.bop_models_dir is None):
        parser.error("--bop-scenes-root and --bop-models-dir must be provided together")
    if args.bop_scenes_root is not None and args.source is None:
        parser.error("--source is required with --bop-scenes-root")
    if args.bop_scenes_root is None and args.source is not None:
        parser.error("--source requires --bop-scenes-root")
    if args.bop_scenes_root is None and args.bop_fixed_split is not None:
        parser.error("--bop-fixed-split requires --bop-scenes-root")
    if args.bop_scenes_root is None and args.bop_annotation_suffix:
        parser.error("--bop-annotation-suffix requires --bop-scenes-root")

    try:
        bop_result = None
        external_rows = None
        if args.bop_scenes_root is not None:
            scene_ids = [int(s.strip()) for s in args.scenes.split(",")] if args.scenes else None
            bop_result = build_bop_manifest(
                bop_scenes_root=args.bop_scenes_root,
                bop_models_dir=args.bop_models_dir,
                source=args.source,
                depth_dir=args.bop_depth_dir,
                annotation_suffix=args.bop_annotation_suffix,
                train_ratio=args.train_ratio,
                seed=args.seed,
                fixed_split=args.bop_fixed_split,
                train_query_limit_per_object=args.train_query_limit_per_object,
                dev_query_limit_per_object=args.dev_query_limit_per_object,
                test_query_limit_per_object=args.test_query_limit_per_object,
                scene_ids=scene_ids,
            )
        if args.itodd_root is not None:
            external_rows = build_itodd_external_manifest(args.itodd_root)

        args.output_dir.mkdir(parents=True, exist_ok=True)
        if bop_result is not None:
            bop_rows, split_data = bop_result
            write_csv_manifest(
                bop_rows,
                BOP_MANIFEST_FIELDS,
                args.output_dir / "bop_manifest.csv",
            )
            write_split(split_data, args.output_dir / "bop_split.json")
        if external_rows is not None:
            write_csv_manifest(
                external_rows,
                ITODD_EXTERNAL_MANIFEST_FIELDS,
                args.output_dir / "itodd_external_manifest.csv",
            )
    except DatasetError as exc:
        parser.error(str(exc))
    except OSError as exc:
        parser.error(f"Cannot create output directory {args.output_dir.resolve()}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
