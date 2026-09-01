"""HALCON surface matching with evaluator-ready pose output."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import halcon as ha
import numpy as np

from evaluation import PoseRecord


HalconParameter = str | int | float


class MatchError(RuntimeError):
    """A HALCON surface matching operation failed."""


class MatchTimeout(MatchError):
    """HALCON cancelled a surface matching operation after a timeout."""


@dataclass(frozen=True)
class SurfaceMatchingConfig:
    """Complete configuration used to create and find a surface model."""

    rel_sampling_distance: float = 0.05
    key_point_fraction: float = 0.2
    min_score: float = 0.01
    num_matches: int = 10
    timeout_sec: float = 0.5
    create_gen_param_names: tuple[str, ...] = ()
    create_gen_param_values: tuple[HalconParameter, ...] = ()
    find_gen_param_names: tuple[str, ...] = (
        "scene_invert_normals",
        "max_overlap_dist_rel",
        "dense_pose_refinement",
        "pose_ref_num_steps",
        "pose_ref_sub_sampling",
        "pose_ref_dist_threshold_rel",
        "pose_ref_scoring_dist_rel",
        "pose_ref_use_scene_normals",
        "sparse_pose_refinement",
        "score_type",
        "scene_normal_computation",
    )
    find_gen_param_values: tuple[HalconParameter, ...] = (
        "true",
        0.5,
        "true",
        5,
        2,
        0.1,
        0.005,
        "false",
        "true",
        "model_point_fraction",
        "fast",
    )
    return_result_handle: str = field(default="false", init=False)

    def __post_init__(self) -> None:
        if not np.isfinite(self.rel_sampling_distance) or self.rel_sampling_distance <= 0:
            raise ValueError("rel_sampling_distance must be finite and greater than zero")
        if not np.isfinite(self.key_point_fraction) or self.key_point_fraction <= 0:
            raise ValueError("key_point_fraction must be finite and greater than zero")
        if not np.isfinite(self.min_score) or self.min_score < 0:
            raise ValueError("min_score must be finite and non-negative")
        if (
            isinstance(self.num_matches, bool)
            or not isinstance(self.num_matches, int)
            or self.num_matches < 0
        ):
            raise ValueError("num_matches must be a non-negative integer")
        if not np.isfinite(self.timeout_sec) or self.timeout_sec <= 0:
            raise ValueError("timeout_sec must be finite and greater than zero")
        if len(self.create_gen_param_names) != len(self.create_gen_param_values):
            raise ValueError(
                "create_gen_param_names and create_gen_param_values must have equal length"
            )
        if len(self.find_gen_param_names) != len(self.find_gen_param_values):
            raise ValueError(
                "find_gen_param_names and find_gen_param_values must have equal length"
            )
        if "num_matches" in self.find_gen_param_names:
            raise ValueError("set num_matches through the dedicated config field")

    @property
    def effective_find_gen_param_names(self) -> tuple[str, ...]:
        return ("num_matches", *self.find_gen_param_names)

    @property
    def effective_find_gen_param_values(self) -> tuple[HalconParameter, ...]:
        return (self.num_matches, *self.find_gen_param_values)


@dataclass(frozen=True)
class MatchResult:
    """All estimates and timing produced for one scene point cloud."""

    predictions: tuple[PoseRecord, ...]
    scores: tuple[float, ...]
    runtime_ms: float
    config: SurfaceMatchingConfig


class SurfaceMatcher:
    """Own a HALCON surface model and match target-frame scene point clouds."""

    def __init__(self, model_point_cloud: Any, config: SurfaceMatchingConfig) -> None:
        if not isinstance(config, SurfaceMatchingConfig):
            raise TypeError("config must be a SurfaceMatchingConfig")

        self.config = config
        self._surface_model = None
        try:
            self._surface_model = ha.create_surface_model(
                model_point_cloud,
                config.rel_sampling_distance,
                config.create_gen_param_names,
                config.create_gen_param_values,
            )
        except ha.HOperatorError as error:
            raise _match_exception("create_surface_model", error) from error

    def __enter__(self) -> SurfaceMatcher:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        try:
            self.close()
        except MatchError:
            if exc_type is None:
                raise

    def close(self) -> None:
        """Release the internally created HALCON surface model handle."""

        if self._surface_model is None:
            return
        try:
            ha.clear_surface_model(self._surface_model)
        except ha.HOperatorError as error:
            raise _match_exception("clear_surface_model", error) from error
        self._surface_model = None

    def match(self, scene_point_cloud: Any) -> MatchResult:
        """Match one scene already expressed in the desired output frame."""

        if self._surface_model is None:
            raise MatchError("surface matcher is closed")

        try:
            start = perf_counter()
            ha.set_operator_timeout(
                "find_surface_model", self.config.timeout_sec, "cancel"
            )
            poses, scores, _ = ha.find_surface_model(
                self._surface_model,
                scene_point_cloud,
                self.config.rel_sampling_distance,
                self.config.key_point_fraction,
                self.config.min_score,
                self.config.return_result_handle,
                self.config.effective_find_gen_param_names,
                self.config.effective_find_gen_param_values,
            )
            runtime_ms = (perf_counter() - start) * 1000.0
        except ha.HOperatorError as error:
            raise _match_exception("find_surface_model", error) from error

        try:
            predictions, converted_scores = _convert_matches(poses, scores)
        except ha.HOperatorError as error:
            raise _match_exception("pose_to_hom_mat3d", error) from error

        return MatchResult(
            predictions=predictions,
            scores=converted_scores,
            runtime_ms=runtime_ms,
            config=self.config,
        )


def _convert_matches(
    poses: Any, scores: Any
) -> tuple[tuple[PoseRecord, ...], tuple[float, ...]]:
    pose_values = tuple(poses)
    score_values = tuple(float(score) for score in scores)
    if len(pose_values) % 7 != 0:
        raise MatchError(
            f"HALCON returned {len(pose_values)} pose values; expected a multiple of 7"
        )

    pose_count = len(pose_values) // 7
    if pose_count != len(score_values):
        raise MatchError(
            f"HALCON returned {pose_count} poses but {len(score_values)} scores"
        )

    predictions = []
    for index in range(pose_count):
        # Passing all seven values preserves HALCON's pose type code.
        pose = pose_values[7 * index : 7 * (index + 1)]
        hom_mat = np.asarray(ha.pose_to_hom_mat3d(pose), dtype=np.float64)
        if hom_mat.shape != (12,):
            raise MatchError(
                f"pose_to_hom_mat3d returned shape {hom_mat.shape}; expected (12,)"
            )
        transform = hom_mat.reshape(3, 4)
        predictions.append(
            PoseRecord(
                translation_mm=transform[:, 3] * 1000.0,
                rotation=transform[:, :3],
                record_id=index,
            )
        )

    return tuple(predictions), score_values


def _match_exception(operation: str, error: Exception) -> MatchError:
    message = str(error)
    error_type = MatchTimeout if any(
        marker in message.lower() for marker in ("timeout", "cancel")
    ) else MatchError
    return error_type(f"HALCON {operation} failed: {message}")
