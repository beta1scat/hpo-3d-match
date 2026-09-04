"""Cached BOP matching pipeline for strict HPO and fixed evaluation."""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

import halcon as ha
import numpy as np
import optuna

from bop_scene_loader import (
    backproject_depth,
    create_halcon_point_cloud,
    filter_points_roi,
    read_bop_camera,
    read_bop_ground_truths,
    read_depth_image,
)
from config import DEFAULT_PARAMS, ROIConfig, SEARCH_SPACE
from dataset import SPLIT_NAMES, TARGET_OBJECT_IDS
from evaluation import PoseRecord, SymmetryConfig, read_bop_symmetry
from experiment_io import append_jsonl, generate_run_id
from hpo_objectives import (
    StrictAssociationRecallFirstV1,
    StrictAssociationScene,
    StrictAssociationV2,
)
from matcher import MatchError, MatchTimeout, SurfaceMatcher, SurfaceMatchingConfig


# HALCON documents "mm" as multiplication by 0.001 into its internal metres.
MODEL_SCALE_MM_TO_M = "mm"

DEFAULT_DEPTH_RANGE_M: tuple[float, float] = (0.20, 0.95)
DEFAULT_DEPTH_STRIDE: int = 3

SYMMETRY_SOURCE = "bop-models-info-per-query"


@dataclass(frozen=True)
class BOPQuery:
    """One unique object query from a BOP manifest."""

    source: str
    model_name: str
    split: str
    scene_id: int
    image_id: int
    obj_id: int
    gt_count: int
    scene_gt_path: Path
    scene_gt_info_path: Path
    scene_camera_path: Path
    depth_path: Path
    cad_path: Path
    models_info_path: Path
    min_visib_fract: float

    @property
    def key(self) -> tuple[str, str, str, int, int, int]:
        return (
            self.source,
            self.model_name,
            self.split,
            self.scene_id,
            self.image_id,
            self.obj_id,
        )


@dataclass(frozen=True)
class CachedBOPQuery:
    query: BOPQuery
    ground_truths: tuple[PoseRecord, ...]
    symmetry: SymmetryConfig
    scene_point_cloud: Any
    preprocessing_runtime_sec: float


@dataclass(frozen=True)
class FixedEvaluationResult:
    """Aggregate strict metrics returned by fixed-parameter evaluation."""

    run_id: str
    method: str
    repeat_id: int
    seed: int
    scene_count: int
    tp: int
    fp: int
    fn: int
    f1: float
    objective: float


def _manifest_path(value: str, manifest_dir: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = manifest_dir / path
    return path.resolve()


def read_bop_queries(
    manifest_path: str | Path,
    model_name: str,
    split: str,
) -> tuple[BOPQuery, ...]:
    """Aggregate manifest GT rows into unique object/image queries."""

    if model_name not in TARGET_OBJECT_IDS:
        raise ValueError(
            f"Unsupported model_name {model_name!r}; expected one of "
            f"{sorted(TARGET_OBJECT_IDS)}"
        )
    if split not in SPLIT_NAMES:
        raise ValueError(f"split must be one of {SPLIT_NAMES}, got {split!r}")
    manifest = Path(manifest_path).expanduser().resolve()
    required_fields = {
        "source",
        "scene_id",
        "image_id",
        "model_name",
        "obj_id",
        "split",
        "gt_count",
        "scene_gt_path",
        "scene_gt_info_path",
        "scene_camera_path",
        "depth_path",
        "cad_path",
        "models_info_path",
        "min_visib_fract",
    }
    grouped: dict[tuple[str, str, str, int, int, int], BOPQuery] = {}
    try:
        with manifest.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            missing = required_fields.difference(reader.fieldnames or ())
            if missing:
                raise ValueError(
                    f"BOP manifest is missing required fields: {sorted(missing)}"
                )
            for line_number, row in enumerate(reader, start=2):
                if row["model_name"] != model_name or row["split"] != split:
                    continue
                try:
                    query = BOPQuery(
                        source=row["source"].strip(),
                        model_name=row["model_name"],
                        split=row["split"],
                        scene_id=int(row["scene_id"]),
                        image_id=int(row["image_id"]),
                        obj_id=int(row["obj_id"]),
                        gt_count=int(row["gt_count"]),
                        scene_gt_path=_manifest_path(
                            row["scene_gt_path"], manifest.parent
                        ),
                        scene_gt_info_path=_manifest_path(
                            row["scene_gt_info_path"], manifest.parent
                        ),
                        scene_camera_path=_manifest_path(
                            row["scene_camera_path"], manifest.parent
                        ),
                        depth_path=_manifest_path(row["depth_path"], manifest.parent),
                        cad_path=_manifest_path(row["cad_path"], manifest.parent),
                        models_info_path=_manifest_path(
                            row["models_info_path"], manifest.parent
                        ),
                        min_visib_fract=float(row["min_visib_fract"]),
                    )
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid BOP manifest row at {manifest}:{line_number}: {exc}"
                    ) from exc
                if min(query.scene_id, query.image_id, query.obj_id) < 0:
                    raise ValueError(
                        f"Negative ID in BOP manifest row at {manifest}:{line_number}"
                    )
                if not query.source:
                    raise ValueError(
                        f"Empty source in BOP manifest row at {manifest}:{line_number}"
                    )
                if query.gt_count <= 0:
                    raise ValueError(
                        f"gt_count must be positive at {manifest}:{line_number}"
                    )
                if not np.isfinite(query.min_visib_fract) or not (
                    0.0 <= query.min_visib_fract <= 1.0
                ):
                    raise ValueError(
                        "min_visib_fract must be finite and between 0 and 1 at "
                        f"{manifest}:{line_number}"
                    )
                expected_obj_id = TARGET_OBJECT_IDS[query.model_name]
                if query.obj_id != expected_obj_id:
                    raise ValueError(
                        f"obj_id {query.obj_id} does not match model_name "
                        f"{query.model_name!r} at {manifest}:{line_number}; "
                        f"expected {expected_obj_id}"
                    )
                previous = grouped.setdefault(query.key, query)
                if previous != query:
                    raise ValueError(
                        f"Inconsistent duplicate BOP query at {manifest}:{line_number}: "
                        f"{query.key}"
                    )
    except OSError as exc:
        raise ValueError(f"Cannot read BOP manifest {manifest}: {exc}") from exc

    if not grouped:
        raise ValueError(
            f"No BOP queries for model_name={model_name!r}, split={split!r} in "
            f"{manifest}"
        )
    return tuple(grouped[key] for key in sorted(grouped))


def suggest_search_params(trial: Any) -> dict[str, Any]:
    """Sample every parameter directly from ``config.SEARCH_SPACE``."""

    params: dict[str, Any] = {}
    for name, spec in SEARCH_SPACE.items():
        parameter_type = spec["type"]
        if parameter_type == "float":
            params[name] = trial.suggest_float(
                name, spec["low"], spec["high"], step=spec.get("step")
            )
        elif parameter_type == "int":
            params[name] = trial.suggest_int(
                name, spec["low"], spec["high"], step=spec.get("step", 1)
            )
        elif parameter_type == "categorical":
            params[name] = trial.suggest_categorical(name, spec["choices"])
        else:
            raise ValueError(f"Unsupported search-space type for {name}: {parameter_type}")
    return params


def surface_matching_config(
    params: Mapping[str, Any],
    *,
    timeout_sec: float = 0.5,
    min_score: float = 0.01,
    num_matches: int = 10,
) -> SurfaceMatchingConfig:
    """Construct ``SurfaceMatchingConfig`` from the search parameters."""

    missing = set(SEARCH_SPACE).difference(params)
    if missing:
        raise ValueError(f"Missing surface-matching parameters: {sorted(missing)}")
    actual_min_score = float(params.get("min_score", min_score))
    find_names = tuple(
        name
        for name in SEARCH_SPACE
        if name not in {"RelSamplingDistance", "KeyPointFraction", "min_score"}
    )
    fixed_names = (
        "dense_pose_refinement",
        "sparse_pose_refinement",
        "score_type",
        "scene_normal_computation",
    )
    fixed_values = ("true", "true", "model_point_fraction", "fast")
    return SurfaceMatchingConfig(
        rel_sampling_distance=float(params["RelSamplingDistance"]),
        key_point_fraction=float(params["KeyPointFraction"]),
        min_score=actual_min_score,
        num_matches=num_matches,
        timeout_sec=timeout_sec,
        find_gen_param_names=("scene_invert_normals", *find_names, *fixed_names),
        find_gen_param_values=(
            "true",
            *(params[name] for name in find_names),
            *fixed_values,
        ),
    )


def _load_model(cad_path: Path, model_scale: str | float) -> Any:
    if isinstance(model_scale, str):
        if model_scale != "mm":
            raise ValueError("String model_scale must be 'mm' for BOP CAD models")
    elif not np.isfinite(model_scale) or model_scale <= 0.0:
        raise ValueError("Numeric model_scale must be finite and greater than zero")
    if not cad_path.is_file():
        raise ValueError(f"Missing BOP CAD model: {cad_path}")
    # BOP CAD coordinates are millimetres; HALCON's Scale converts them to the
    # metre convention used by the cached scene point clouds.
    raw_model, _ = ha.read_object_model_3d(str(cad_path), model_scale, (), ())
    model_with_normals = None
    try:
        model_with_normals = ha.surface_normals_object_model_3d(
            raw_model, "mls", (), ()
        )
    except BaseException:
        _clear_object_models([raw_model], suppress_errors=True)
        raise
    cleanup_error = _clear_object_models([raw_model])
    if cleanup_error is not None:
        _clear_object_models([model_with_normals], suppress_errors=True)
        raise cleanup_error
    return model_with_normals


def _clear_object_models(
    handles: Sequence[Any], *, suppress_errors: bool = False
) -> MatchError | None:
    """Try every HALCON handle and return the first cleanup error."""

    first_error = None
    for handle in handles:
        if handle is None:
            continue
        try:
            ha.clear_object_model_3d(handle)
        except Exception as exc:
            if first_error is None:
                first_error = MatchError(
                    f"HALCON clear_object_model_3d failed: {exc}"
                )
    return None if suppress_errors else first_error


def _association_errors(detail: Any) -> list[dict[str, float | int | str | None]]:
    evaluation = detail.evaluation
    if evaluation is None:
        return []
    return [
        {
            "prediction": item.prediction_id,
            "ground_truth": item.ground_truth_id,
            "translation_mm": item.translation_error_mm,
            "rotation_deg": item.rotation_error_deg,
            "normalized": item.normalized_cost,
        }
        for item in evaluation.associations
    ]


def _pose_json(pose: PoseRecord) -> dict[str, Any]:
    return {
        "R": pose.rotation.reshape(9).tolist(),
        "t": pose.translation_mm.tolist(),
    }


class BOPPipeline:
    """Preload one model/split and reuse each HALCON scene point cloud."""

    def __init__(
        self,
        manifest_path: str | Path,
        model_name: str,
        split: str,
        *,
        model_scale: str | float = MODEL_SCALE_MM_TO_M,
        depth_range_m: Sequence[float] | None = DEFAULT_DEPTH_RANGE_M,
        depth_stride: int = DEFAULT_DEPTH_STRIDE,
        use_roi: bool = False,
        roi: ROIConfig | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.model_name = model_name
        self.split = split
        self.model_scale = model_scale
        self.depth_range_m = depth_range_m
        self.depth_stride = depth_stride
        self.use_roi = bool(use_roi)
        self.roi = roi if roi is not None else ROIConfig()
        self.queries = read_bop_queries(self.manifest_path, model_name, split)
        self.model_point_cloud = None
        self.cached_queries: tuple[CachedBOPQuery, ...] = ()
        self._closed = False

        cad_paths = {query.cad_path for query in self.queries}
        if len(cad_paths) != 1:
            raise ValueError(
                f"Expected one CAD path for {model_name!r}, got: "
                f"{sorted(str(path) for path in cad_paths)}"
            )
        cached = []
        scene_handles = []
        pending_scene_handle = None
        try:
            self.model_point_cloud = _load_model(
                next(iter(cad_paths)), self.model_scale
            )
            for query in self.queries:
                ground_truths = read_bop_ground_truths(
                    query.scene_gt_path,
                    query.scene_gt_info_path,
                    query.scene_id,
                    query.image_id,
                    query.obj_id,
                    min_visib_fract=query.min_visib_fract,
                )
                if len(ground_truths) != query.gt_count:
                    raise ValueError(
                        "Filtered BOP GT count does not match manifest gt_count for "
                        f"scene={query.scene_id}, image={query.image_id}, "
                        f"obj_id={query.obj_id}: expected {query.gt_count}, "
                        f"found {len(ground_truths)}"
                    )
                symmetry = read_bop_symmetry(query.models_info_path, query.obj_id)
                preprocessing_start = perf_counter()
                camera = read_bop_camera(query.scene_camera_path, query.image_id)
                points_xyz_m = backproject_depth(
                    read_depth_image(query.depth_path),
                    camera,
                    depth_range_m=self.depth_range_m,
                    stride=self.depth_stride,
                )
                if self.use_roi:
                    points_xyz_m = filter_points_roi(
                        points_xyz_m, self.roi, is_bop=True
                    )
                pending_scene_handle = create_halcon_point_cloud(points_xyz_m)
                scene_handles.append(pending_scene_handle)
                scene_point_cloud = pending_scene_handle
                pending_scene_handle = None
                cached.append(
                    CachedBOPQuery(
                        query=query,
                        ground_truths=tuple(ground_truths),
                        symmetry=symmetry,
                        scene_point_cloud=scene_point_cloud,
                        preprocessing_runtime_sec=(
                            perf_counter() - preprocessing_start
                        ),
                    )
                )
            self.cached_queries = tuple(cached)
        except BaseException:
            _clear_object_models(
                scene_handles + [pending_scene_handle, self.model_point_cloud],
                suppress_errors=True,
            )
            self.model_point_cloud = None
            self.cached_queries = ()
            self._closed = True
            raise

    def __enter__(self) -> BOPPipeline:
        self._require_open()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        try:
            self.close()
        except MatchError:
            if exc_type is None:
                raise

    def close(self) -> None:
        """Release every cached HALCON object model handle exactly once."""

        if self._closed:
            return
        handles = [item.scene_point_cloud for item in self.cached_queries]
        handles.append(self.model_point_cloud)
        self.cached_queries = ()
        self.model_point_cloud = None
        self._closed = True
        error = _clear_object_models(handles)
        if error is not None:
            raise error

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("BOPPipeline is closed")

    def create_optuna_objective(
        self,
        objective_jsonl_path: str | Path,
        *,
        run_id: str | None = None,
        study_name: str | None = None,
        timeout_sec: float = 0.5,
        min_score: float = 0.01,
        num_matches: int = 10,
        evaluator: StrictAssociationV2 | None = None,
    ):
        """Return an Optuna objective reporting the running mean per query."""

        self._require_open()
        selected_run_id = run_id or generate_run_id("hpo")
        log_path = Path(objective_jsonl_path)
        strict = evaluator or StrictAssociationV2()

        def objective(trial: Any) -> float:
            self._require_open()
            params = suggest_search_params(trial)
            config = surface_matching_config(
                params,
                timeout_sec=timeout_sec,
                min_score=min_score,
                num_matches=num_matches,
            )
            selected_study_name = (
                study_name
                or getattr(getattr(trial, "study", None), "study_name", None)
                or "unknown-study"
            )
            trial.set_user_attr("model_scale", self.model_scale)
            trial.set_user_attr("model_scale_units", "BOP mm to scene m")
            trial.set_user_attr("symmetry_source", SYMMETRY_SOURCE)
            total = 0.0

            try:
                matcher_context = SurfaceMatcher(self.model_point_cloud, config)
            except MatchError as exc:
                trial.set_user_attr("match_error_type", type(exc).__name__)
                trial.set_user_attr("match_error", str(exc))
                trial.set_user_attr("match_error_stage", "create_surface_model")
                raise

            matcher = matcher_context
            try:
                for step, cached in enumerate(self.cached_queries):
                    query = cached.query
                    start = perf_counter()
                    status = "COMPLETE"
                    try:
                        result = matcher.match(cached.scene_point_cloud)
                        predictions = result.predictions
                        runtime = result.runtime_ms / 1000.0
                    except MatchTimeout:
                        predictions = ()
                        runtime = perf_counter() - start
                        status = "TIMEOUT"
                    except MatchError as exc:
                        runtime = perf_counter() - start
                        record = self._hpo_scene_record(
                            selected_run_id,
                            selected_study_name,
                            trial.number,
                            query,
                            "FAIL",
                            runtime * 1000.0,
                            None,
                        )
                        record["errors"] = [
                            {"type": type(exc).__name__, "message": str(exc)}
                        ]
                        append_jsonl(log_path, record)
                        trial.set_user_attr("match_error_type", type(exc).__name__)
                        trial.set_user_attr("match_error", str(exc))
                        trial.set_user_attr("match_error_scene", query.scene_id)
                        trial.set_user_attr("match_error_image", query.image_id)
                        raise

                    detail = strict.evaluate_scene(
                        StrictAssociationScene(
                            scene_id=f"{query.scene_id}:{query.image_id}",
                            predictions=tuple(predictions),
                            ground_truths=cached.ground_truths,
                            symmetry=cached.symmetry,
                        )
                    )
                    total += detail.objective
                    running_mean = total / (step + 1)
                    append_jsonl(
                        log_path,
                        self._hpo_scene_record(
                            selected_run_id,
                            selected_study_name,
                            trial.number,
                            query,
                            status,
                            runtime * 1000.0,
                            detail,
                        ),
                    )
                    trial.report(running_mean, step)
                    if trial.should_prune():
                        raise optuna.TrialPruned(
                            f"Pruned after {step + 1} BOP queries; mean={running_mean}"
                        )
            finally:
                active_exception = sys.exc_info()[0] is not None
                try:
                    matcher.close()
                except MatchError as exc:
                    try:
                        trial.set_user_attr("cleanup_error_type", type(exc).__name__)
                        trial.set_user_attr("cleanup_error", str(exc))
                        trial.set_user_attr(
                            "cleanup_error_stage", "clear_surface_model"
                        )
                    except Exception:
                        pass
                    if not active_exception:
                        raise
            return total / len(self.cached_queries)

        return objective

    def evaluate_fixed_params(
        self,
        params: Mapping[str, Any] | None,
        predictions_jsonl_path: str | Path,
        scene_summaries_jsonl_path: str | Path,
        *,
        method: str,
        repeat_id: int,
        seed: int,
        run_id: str | None = None,
        timeout_sec: float = 5.0,
        min_score: float = 0.01,
        num_matches: int = 10,
        evaluator: StrictAssociationV2 | None = None,
    ) -> FixedEvaluationResult:
        """Evaluate fixed parameters and write predictions and scene summaries."""

        self._require_open()
        selected_params = dict(DEFAULT_PARAMS if params is None else params)
        config = surface_matching_config(
            selected_params,
            timeout_sec=timeout_sec,
            min_score=min_score,
            num_matches=num_matches,
        )
        strict = evaluator or StrictAssociationV2()
        if not isinstance(method, str) or not method.strip():
            raise ValueError("method must be a non-empty string")
        if (
            isinstance(repeat_id, bool)
            or not isinstance(repeat_id, int)
            or repeat_id < 0
        ):
            raise ValueError("repeat_id must be a non-negative integer")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        selected_run_id = run_id or generate_run_id("evaluation")
        total_objective = 0.0
        total_tp = total_fp = total_fn = 0
        for output_path in (predictions_jsonl_path, scene_summaries_jsonl_path):
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch(exist_ok=False)

        with SurfaceMatcher(self.model_point_cloud, config) as matcher:
            for cached in self.cached_queries:
                query = cached.query
                start = perf_counter()
                status = "COMPLETE"
                try:
                    result = matcher.match(cached.scene_point_cloud)
                    predictions = result.predictions
                    scores = result.scores
                    runtime = cached.preprocessing_runtime_sec + perf_counter() - start
                except MatchTimeout:
                    predictions = ()
                    scores = ()
                    runtime = cached.preprocessing_runtime_sec + perf_counter() - start
                    status = "TIMEOUT"
                except MatchError:
                    runtime = cached.preprocessing_runtime_sec + perf_counter() - start
                    append_jsonl(
                        scene_summaries_jsonl_path,
                        self._fixed_scene_summary(
                            selected_run_id,
                            method,
                            repeat_id,
                            seed,
                            query,
                            len(cached.ground_truths),
                            "FAIL",
                            runtime * 1000.0,
                            None,
                        ),
                    )
                    raise

                detail = strict.evaluate_scene(
                    StrictAssociationScene(
                        scene_id=f"{query.scene_id}:{query.image_id}",
                        predictions=tuple(predictions),
                        ground_truths=cached.ground_truths,
                        symmetry=cached.symmetry,
                    )
                )
                for prediction, score in zip(predictions, scores):
                    append_jsonl(
                        predictions_jsonl_path,
                        {
                            "run_id": selected_run_id,
                            "record_type": "prediction",
                            "dataset": "bop_itodd",
                            "split": query.split,
                            "object_name": query.model_name,
                            "obj_id": query.obj_id,
                            "method": method,
                            "repeat_id": repeat_id,
                            "seed": seed,
                            "scene_id": query.scene_id,
                            "im_id": query.image_id,
                            "score": score,
                            "pose": _pose_json(prediction),
                            "time": runtime,
                        },
                    )
                append_jsonl(
                    scene_summaries_jsonl_path,
                    self._fixed_scene_summary(
                        selected_run_id,
                        method,
                        repeat_id,
                        seed,
                        query,
                        len(cached.ground_truths),
                        status,
                        runtime * 1000.0,
                        detail,
                    ),
                )
                evaluation = detail.evaluation
                if evaluation is None:
                    raise RuntimeError("Strict evaluation did not return metrics")
                total_objective += detail.objective
                total_tp += evaluation.tp
                total_fp += evaluation.fp
                total_fn += evaluation.fn

        precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
        recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        return FixedEvaluationResult(
            run_id=selected_run_id,
            method=method,
            repeat_id=repeat_id,
            seed=seed,
            scene_count=len(self.cached_queries),
            tp=total_tp,
            fp=total_fp,
            fn=total_fn,
            f1=f1,
            objective=total_objective / len(self.cached_queries),
        )

    def _hpo_scene_record(
        self,
        run_id: str,
        study_name: str,
        trial_number: int,
        query: BOPQuery,
        status: str,
        runtime_ms: float,
        detail: Any | None,
    ) -> dict[str, Any]:
        evaluation = detail.evaluation if detail is not None else None
        return {
            "run_id": run_id,
            "study_name": study_name,
            "trial_number": trial_number,
            "object_name": query.model_name,
            "scene_id": query.scene_id,
            "im_id": query.image_id,
            "status": status,
            "runtime_ms": runtime_ms,
            "tp": evaluation.tp if evaluation is not None else None,
            "fp": evaluation.fp if evaluation is not None else None,
            "fn": evaluation.fn if evaluation is not None else None,
            "errors": _association_errors(detail) if detail is not None else [],
            "objective": detail.objective if detail is not None else None,
        }

    def _fixed_scene_summary(
        self,
        run_id: str,
        method: str,
        repeat_id: int,
        seed: int,
        query: BOPQuery,
        ground_truth_count: int,
        status: str,
        runtime_ms: float,
        detail: Any | None,
    ) -> dict[str, Any]:
        evaluation = detail.evaluation if detail is not None else None
        associations = evaluation.associations if evaluation is not None else ()
        return {
            "run_id": run_id,
            "dataset": "bop_itodd",
            "split": query.split,
            "object_name": query.model_name,
            "method": method,
            "repeat_id": repeat_id,
            "seed": seed,
            "scene_id": f"{query.scene_id}:{query.image_id}",
            "status": status,
            "tp": evaluation.tp if evaluation is not None else 0,
            "fp": evaluation.fp if evaluation is not None else 0,
            "fn": evaluation.fn if evaluation is not None else ground_truth_count,
            "translation_errors_mm": [
                association.translation_error_mm for association in associations
            ],
            "rotation_errors_deg": [
                association.rotation_error_deg for association in associations
            ],
            "runtime_ms": runtime_ms,
        }


__all__ = [
    "BOPPipeline",
    "BOPQuery",
    "FixedEvaluationResult",
    "MODEL_SCALE_MM_TO_M",
    "SYMMETRY_SOURCE",
    "read_bop_queries",
    "suggest_search_params",
    "surface_matching_config",
]
