"""Fixed-parameter inference for the original no-GT MVTec ITODD test set."""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

import halcon as ha
import numpy as np

from bop_scene_loader import (
    create_halcon_point_cloud,
    filter_points_ransac_tabletop,
)
from config import DEFAULT_PARAMS, RANSAC_TABLETOP_THRESHOLDS
from dataset import ITODD_EXTERNAL_MANIFEST_FIELDS, TARGET_OBJECT_IDS
from evaluation import PoseRecord
from experiment_io import append_jsonl, generate_run_id
from matcher import MatchError, MatchTimeout, SurfaceMatcher
from pipeline import surface_matching_config


DATASET_NAME = "itodd_original"
EXTERNAL_SPLIT = "external_test"
OFFICIAL_SENSOR = "3d_large_baseline"
# Original MVTec CAD PLY coordinates and XYZ TIF coordinates are both metres.
ORIGINAL_CAD_UNIT = "m"


@dataclass(frozen=True)
class ExternalQuery:
    """One model/scene query from ``itodd_external_manifest.csv``."""

    scene_id: int
    model_name: str
    obj_id: int
    split: str
    cad_path: Path
    x_path: Path
    y_path: Path
    z_path: Path


@dataclass(frozen=True)
class ExternalRunResult:
    """Run-level counts for an external inference pass."""

    run_id: str
    method: str
    scene_count: int
    match_count: int
    complete_count: int
    timeout_count: int
    fail_count: int
    runtime_sec: float


def _manifest_path(value: str, manifest_dir: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = manifest_dir / path
    return path.resolve()


def read_external_queries(
    manifest_path: str | Path, model_name: str
) -> tuple[ExternalQuery, ...]:
    """Read and validate external-test rows selected by ``model_name``."""

    if model_name not in TARGET_OBJECT_IDS:
        raise ValueError(
            f"Unsupported model_name {model_name!r}; expected one of "
            f"{sorted(TARGET_OBJECT_IDS)}"
        )
    manifest = Path(manifest_path).expanduser().resolve()
    queries: dict[int, ExternalQuery] = {}
    try:
        with manifest.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            missing = set(ITODD_EXTERNAL_MANIFEST_FIELDS).difference(
                reader.fieldnames or ()
            )
            if missing:
                raise ValueError(
                    f"External manifest is missing required fields: {sorted(missing)}"
                )
            for line_number, row in enumerate(reader, start=2):
                if row["model_name"] != model_name:
                    continue
                try:
                    query = ExternalQuery(
                        scene_id=int(row["scene_id"]),
                        model_name=row["model_name"],
                        obj_id=int(row["obj_id"]),
                        split=row["split"],
                        cad_path=_manifest_path(row["cad_path"], manifest.parent),
                        x_path=_manifest_path(row["x_path"], manifest.parent),
                        y_path=_manifest_path(row["y_path"], manifest.parent),
                        z_path=_manifest_path(row["z_path"], manifest.parent),
                    )
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid external manifest row at {manifest}:{line_number}: "
                        f"{exc}"
                    ) from exc
                if query.scene_id < 0 or query.obj_id < 0:
                    raise ValueError(
                        f"Negative ID in external manifest row at "
                        f"{manifest}:{line_number}"
                    )
                if query.obj_id != TARGET_OBJECT_IDS[model_name]:
                    raise ValueError(
                        f"obj_id {query.obj_id} does not match model "
                        f"{model_name!r} at {manifest}:{line_number}"
                    )
                if query.split != EXTERNAL_SPLIT:
                    raise ValueError(
                        f"Expected split={EXTERNAL_SPLIT!r} at "
                        f"{manifest}:{line_number}, got {query.split!r}"
                    )
                if query.cad_path.suffix.lower() != ".ply":
                    raise ValueError(
                        f"Original CAD must be a PLY file at "
                        f"{manifest}:{line_number}: {query.cad_path}"
                    )
                for label, path in (
                    ("CAD", query.cad_path),
                    ("X TIF", query.x_path),
                    ("Y TIF", query.y_path),
                    ("Z TIF", query.z_path),
                ):
                    if not path.is_file():
                        raise ValueError(
                            f"Missing {label} at {manifest}:{line_number}: {path}"
                        )
                previous = queries.setdefault(query.scene_id, query)
                if previous != query:
                    raise ValueError(
                        f"Inconsistent duplicate external scene {query.scene_id} "
                        f"at {manifest}:{line_number}"
                    )
    except OSError as exc:
        raise ValueError(f"Cannot read external manifest {manifest}: {exc}") from exc

    if not queries:
        raise ValueError(
            f"No external queries for model_name={model_name!r} in {manifest}"
        )
    cad_paths = {query.cad_path for query in queries.values()}
    if len(cad_paths) != 1:
        raise ValueError(
            f"Expected one original CAD path for {model_name!r}, got: "
            f"{sorted(str(path) for path in cad_paths)}"
        )
    return tuple(queries[scene_id] for scene_id in sorted(queries))


def load_xyz_point_cloud(
    query: ExternalQuery,
    distance_threshold_m: float | None = None,
) -> Any:
    """Create a HALCON object model from manifest X/Y/Z TIF files, removing tabletop with RANSAC."""

    import imageio.v3 as iio

    x_img = np.asarray(iio.imread(query.x_path))
    y_img = np.asarray(iio.imread(query.y_path))
    z_img = np.asarray(iio.imread(query.z_path))
    valid = np.isfinite(z_img) & (z_img > 0.05) & (z_img < 1.5)
    points = np.stack(
        [x_img[valid], y_img[valid], z_img[valid]], axis=-1
    ).astype(np.float64)

    if distance_threshold_m is not None:
        points = filter_points_ransac_tabletop(
            points, distance_threshold_m=distance_threshold_m
        )
    return create_halcon_point_cloud(points)


def load_original_cad(cad_path: str | Path) -> Any:
    """Load an original MVTec PLY whose coordinates are explicitly metres."""

    path = Path(cad_path)
    raw_model = model_with_normals = None
    try:
        raw_model, _ = ha.read_object_model_3d(
            str(path), ORIGINAL_CAD_UNIT, (), ()
        )
        model_with_normals = ha.surface_normals_object_model_3d(
            raw_model, "mls", (), ()
        )
        return model_with_normals
    finally:
        active_exception = sys.exc_info()[0] is not None
        cleanup_error = _clear_object_models((raw_model,))
        if cleanup_error is not None and not active_exception:
            _clear_object_models((model_with_normals,))
            raise cleanup_error


def _pose_json(pose: PoseRecord) -> dict[str, list[float]]:
    return {
        "R": pose.rotation.reshape(9).tolist(),
        "t": pose.translation_mm.tolist(),
    }


def _official_result_text(
    predictions: tuple[PoseRecord, ...],
    scores: tuple[float, ...],
    runtime_sec: float,
) -> str:
    lines = [f"Time: {runtime_sec:.9g}", f"Sensor: {OFFICIAL_SENSOR}"]
    for prediction, score in zip(predictions, scores):
        transform_m = np.eye(4, dtype=np.float64)
        transform_m[:3, :3] = prediction.rotation
        transform_m[:3, 3] = prediction.translation_mm / 1000.0
        values = " ".join(f"{value:.12g}" for value in transform_m.reshape(16))
        lines.append(f"Result: {values} {score:.12g}")
    return "\n".join(lines) + "\n"


def _clear_object_models(handles: tuple[Any | None, ...]) -> MatchError | None:
    first_error = None
    for handle in handles:
        if handle is None:
            continue
        try:
            ha.clear_object_model_3d(handle)
        except ha.HOperatorError as exc:
            if first_error is None:
                first_error = MatchError(
                    f"HALCON clear_object_model_3d failed: {exc}"
                )
    return first_error


class ExternalPipeline:
    """Run fixed matching on original ITODD scenes with RANSAC tabletop removal."""

    def __init__(
        self,
        manifest_path: str | Path,
        model_name: str,
        ransac_threshold_m: float | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.model_name = model_name
        self.queries = read_external_queries(self.manifest_path, model_name)
        if ransac_threshold_m is not None:
            self.ransac_threshold_m = float(ransac_threshold_m)
        else:
            self.ransac_threshold_m = RANSAC_TABLETOP_THRESHOLDS.get(
                model_name, 0.0020
            )

    def evaluate_fixed_params(
        self,
        params: Mapping[str, Any] | None,
        predictions_jsonl_path: str | Path,
        scene_summaries_jsonl_path: str | Path,
        official_results_dir: str | Path,
        *,
        method: str,
        run_id: str | None = None,
        timeout_sec: float = 5.0,
        min_score: float = 0.01,
        num_matches: int = 10,
    ) -> ExternalRunResult:
        """Match all selected scenes and write no-GT JSONL and official files."""

        if not isinstance(method, str) or not method.strip():
            raise ValueError("method must be a non-empty string")
        selected_run_id = run_id or generate_run_id("external-test")
        config = surface_matching_config(
            dict(DEFAULT_PARAMS if params is None else params),
            timeout_sec=timeout_sec,
            min_score=min_score,
            num_matches=num_matches,
        )
        predictions_path = Path(predictions_jsonl_path)
        summaries_path = Path(scene_summaries_jsonl_path)
        official_dir = Path(official_results_dir)
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        summaries_path.parent.mkdir(parents=True, exist_ok=True)
        official_dir.mkdir(parents=True, exist_ok=False)
        predictions_path.touch(exist_ok=False)
        summaries_path.touch(exist_ok=False)

        total_matches = complete_count = timeout_count = fail_count = 0
        run_start = perf_counter()
        model_point_cloud = None
        matcher = None
        try:
            model_point_cloud = load_original_cad(self.queries[0].cad_path)
            matcher = SurfaceMatcher(model_point_cloud, config)
            for query in self.queries:
                predictions: tuple[PoseRecord, ...] = ()
                scores: tuple[float, ...] = ()
                status = "COMPLETE"
                error: dict[str, str] | None = None
                scene_3d = None
                scene_start = perf_counter()
                try:
                    scene_3d = load_xyz_point_cloud(
                        query, distance_threshold_m=self.ransac_threshold_m
                    )
                    match_result = matcher.match(scene_3d)
                    predictions = match_result.predictions
                    scores = match_result.scores
                except MatchTimeout as exc:
                    status = "TIMEOUT"
                    error = {"type": type(exc).__name__, "message": str(exc)}
                except (MatchError, ha.HOperatorError) as exc:
                    status = "FAIL"
                    error = {"type": type(exc).__name__, "message": str(exc)}
                finally:
                    active_exception = sys.exc_info()[0] is not None
                    cleanup_error = _clear_object_models((scene_3d,))
                    if cleanup_error is not None:
                        if error is None and not active_exception:
                            raise cleanup_error
                        if error is not None:
                            error["cleanup_error"] = str(cleanup_error)

                runtime_sec = perf_counter() - scene_start
                match_count = len(predictions)
                total_matches += match_count
                complete_count += status == "COMPLETE"
                timeout_count += status == "TIMEOUT"
                fail_count += status == "FAIL"

                for prediction, score in zip(predictions, scores):
                    append_jsonl(
                        predictions_path,
                        {
                            "run_id": selected_run_id,
                            "dataset": DATASET_NAME,
                            "split": EXTERNAL_SPLIT,
                            "object_name": query.model_name,
                            "obj_id": query.obj_id,
                            "method": method,
                            "scene_id": query.scene_id,
                            "score": score,
                            "pose": _pose_json(prediction),
                            "time": runtime_sec,
                            "status": status,
                        },
                    )
                summary = {
                    "run_id": selected_run_id,
                    "dataset": DATASET_NAME,
                    "split": EXTERNAL_SPLIT,
                    "object_name": query.model_name,
                    "obj_id": query.obj_id,
                    "method": method,
                    "scene_id": query.scene_id,
                    "match_count": match_count,
                    "runtime_sec": runtime_sec,
                    "timeout": status == "TIMEOUT",
                    "fail": status == "FAIL",
                    "status": status,
                }
                if error is not None:
                    summary["error"] = error
                append_jsonl(summaries_path, summary)
                result_path = (
                    official_dir
                    / f"result_{query.model_name}_{query.scene_id}.txt"
                )
                result_path.write_text(
                    _official_result_text(predictions, scores, runtime_sec),
                    encoding="utf-8",
                    newline="\n",
                )
        finally:
            active_exception = sys.exc_info()[0] is not None
            cleanup_error = None
            if matcher is not None:
                try:
                    matcher.close()
                except MatchError as exc:
                    cleanup_error = exc
            model_cleanup_error = _clear_object_models((model_point_cloud,))
            if cleanup_error is None:
                cleanup_error = model_cleanup_error
            if cleanup_error is not None and not active_exception:
                raise cleanup_error

        return ExternalRunResult(
            run_id=selected_run_id,
            method=method,
            scene_count=len(self.queries),
            match_count=total_matches,
            complete_count=complete_count,
            timeout_count=timeout_count,
            fail_count=fail_count,
            runtime_sec=perf_counter() - run_start,
        )


def run_external_pipeline(
    manifest_path: str | Path,
    model_name: str,
    params: Mapping[str, Any] | None,
    predictions_jsonl_path: str | Path,
    scene_summaries_jsonl_path: str | Path,
    official_results_dir: str | Path,
    *,
    method: str,
    run_id: str | None = None,
    timeout_sec: float = 5.0,
    min_score: float = 0.01,
    num_matches: int = 10,
) -> ExternalRunResult:
    """Convenience function for callers that do not need a pipeline instance."""

    return ExternalPipeline(manifest_path, model_name).evaluate_fixed_params(
        params,
        predictions_jsonl_path,
        scene_summaries_jsonl_path,
        official_results_dir,
        method=method,
        run_id=run_id,
        timeout_sec=timeout_sec,
        min_score=min_score,
        num_matches=num_matches,
    )


__all__ = [
    "DATASET_NAME",
    "EXTERNAL_SPLIT",
    "ExternalPipeline",
    "ExternalQuery",
    "ExternalRunResult",
    "OFFICIAL_SENSOR",
    "ORIGINAL_CAD_UNIT",
    "load_original_cad",
    "load_xyz_point_cloud",
    "read_external_queries",
    "run_external_pipeline",
]
