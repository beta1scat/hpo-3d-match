"""Strict, HALCON-independent evaluation for 6D object poses.

Translations are expressed in millimetres and rotations are 3x3 matrices.
Threshold comparisons are inclusive, so a pose exactly on both thresholds is
considered feasible.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment


DEFAULT_TRANSLATION_THRESHOLD_MM = 10.0
DEFAULT_ROTATION_THRESHOLD_DEG = 10.0
# BOP JSON rotations are commonly rounded to six decimal places.
_ROTATION_ATOL = 1e-5
_THRESHOLD_ATOL = 1e-9


def _validated_rotation(value: np.ndarray, name: str) -> np.ndarray:
    rotation = np.array(value, dtype=np.float64, copy=True)
    if rotation.shape != (3, 3):
        raise ValueError(f"{name} must have shape (3, 3), got {rotation.shape}")
    if not np.all(np.isfinite(rotation)):
        raise ValueError(f"{name} must contain only finite values")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=_ROTATION_ATOL, rtol=0.0):
        raise ValueError(f"{name} must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=_ROTATION_ATOL, rtol=0.0):
        raise ValueError(f"{name} must have determinant +1")
    rotation.setflags(write=False)
    return rotation


@dataclass(frozen=True)
class PoseRecord:
    """A pose supplied explicitly to the evaluator.

    ``record_id`` is copied to association output and has no effect on matching.
    """

    translation_mm: np.ndarray
    rotation: np.ndarray
    record_id: str | int | None = None

    def __post_init__(self) -> None:
        translation = np.array(self.translation_mm, dtype=np.float64, copy=True)
        if translation.shape != (3,):
            raise ValueError(
                f"translation_mm must have shape (3,), got {translation.shape}"
            )
        if not np.all(np.isfinite(translation)):
            raise ValueError("translation_mm must contain only finite values")
        translation.setflags(write=False)
        object.__setattr__(self, "translation_mm", translation)
        object.__setattr__(
            self, "rotation", _validated_rotation(self.rotation, "rotation")
        )


@dataclass(frozen=True)
class DiscreteSymmetry:
    """One BOP object-frame rigid symmetry transform."""

    rotation: np.ndarray
    translation_mm: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "rotation", _validated_rotation(self.rotation, "rotation")
        )
        translation = np.array(self.translation_mm, dtype=np.float64, copy=True)
        if translation.shape != (3,):
            raise ValueError(
                f"translation_mm must have shape (3,), got {translation.shape}"
            )
        if not np.all(np.isfinite(translation)):
            raise ValueError("translation_mm must contain only finite values")
        translation.setflags(write=False)
        object.__setattr__(self, "translation_mm", translation)


@dataclass(frozen=True)
class ContinuousSymmetry:
    """One BOP continuous rotation axis and a point on that axis."""

    axis: np.ndarray
    offset_mm: np.ndarray

    def __post_init__(self) -> None:
        axis = np.array(self.axis, dtype=np.float64, copy=True)
        if axis.shape != (3,):
            raise ValueError(f"axis must have shape (3,), got {axis.shape}")
        if not np.all(np.isfinite(axis)):
            raise ValueError("axis must contain only finite values")
        if not np.isclose(
            np.linalg.norm(axis), 1.0, atol=_ROTATION_ATOL, rtol=0.0
        ):
            raise ValueError("axis must be a unit vector")
        axis.setflags(write=False)
        object.__setattr__(self, "axis", axis)

        offset = np.array(self.offset_mm, dtype=np.float64, copy=True)
        if offset.shape != (3,):
            raise ValueError(f"offset_mm must have shape (3,), got {offset.shape}")
        if not np.all(np.isfinite(offset)):
            raise ValueError("offset_mm must contain only finite values")
        offset.setflags(write=False)
        object.__setattr__(self, "offset_mm", offset)


@dataclass(frozen=True, init=False)
class SymmetryConfig:
    """BOP discrete rigid and continuous axial object-frame symmetries.

    Identity is always present in ``discrete_symmetries``. The legacy
    ``discrete_rotations`` and ``continuous_axis`` constructor arguments remain
    available for concise rotation-only configurations.
    """

    discrete_symmetries: tuple[DiscreteSymmetry, ...]
    continuous_symmetries: tuple[ContinuousSymmetry, ...]

    def __init__(
        self,
        discrete_symmetries: Sequence[DiscreteSymmetry] = (),
        continuous_symmetries: Sequence[ContinuousSymmetry] = (),
        *,
        discrete_rotations: Sequence[np.ndarray] | None = None,
        continuous_axis: np.ndarray | None = None,
    ) -> None:
        if discrete_rotations is not None:
            if discrete_symmetries:
                raise ValueError(
                    "discrete_symmetries and discrete_rotations cannot both be set"
                )
            discrete_symmetries = tuple(
                DiscreteSymmetry(rotation, np.zeros(3))
                for rotation in discrete_rotations
            )
        if continuous_axis is not None:
            if continuous_symmetries:
                raise ValueError(
                    "continuous_symmetries and continuous_axis cannot both be set"
                )
            continuous_symmetries = (
                ContinuousSymmetry(continuous_axis, np.zeros(3)),
            )

        discrete = tuple(discrete_symmetries)
        continuous = tuple(continuous_symmetries)
        for index, item in enumerate(discrete):
            if not isinstance(item, DiscreteSymmetry):
                raise TypeError(
                    f"discrete_symmetries[{index}] must be a DiscreteSymmetry"
                )
        for index, item in enumerate(continuous):
            if not isinstance(item, ContinuousSymmetry):
                raise TypeError(
                    f"continuous_symmetries[{index}] must be a ContinuousSymmetry"
                )

        if not any(
            np.allclose(item.rotation, np.eye(3), atol=_ROTATION_ATOL, rtol=0.0)
            and np.allclose(
                item.translation_mm, np.zeros(3), atol=_ROTATION_ATOL, rtol=0.0
            )
            for item in discrete
        ):
            discrete = (DiscreteSymmetry(np.eye(3), np.zeros(3)),) + discrete
        object.__setattr__(self, "discrete_symmetries", discrete)
        object.__setattr__(self, "continuous_symmetries", continuous)

    @property
    def discrete_rotations(self) -> tuple[np.ndarray, ...]:
        """Rotation-only compatibility view used by existing callers."""

        return tuple(item.rotation for item in self.discrete_symmetries)

    @property
    def continuous_axis(self) -> np.ndarray | None:
        """Single-axis compatibility view used by existing callers."""

        if len(self.continuous_symmetries) == 1:
            return self.continuous_symmetries[0].axis
        return None


def read_bop_symmetry(
    models_info_path: str | Path, obj_id: int
) -> SymmetryConfig:
    """Read one object's symmetry definitions from BOP ``models_info.json``."""

    if isinstance(obj_id, bool) or not isinstance(obj_id, int) or obj_id <= 0:
        raise ValueError("obj_id must be a positive integer")
    try:
        path = Path(models_info_path)
    except TypeError as error:
        raise ValueError("models_info_path must be path-like") from error
    try:
        with path.open("r", encoding="utf-8") as stream:
            models_info = json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read BOP models info from {path}: {error}") from error
    if not isinstance(models_info, dict):
        raise ValueError("BOP models info must be a JSON object")
    model_info = models_info.get(str(obj_id))
    if not isinstance(model_info, dict):
        raise ValueError(f"obj_id {obj_id} is missing from BOP models info")

    raw_discrete = model_info.get("symmetries_discrete", [])
    if not isinstance(raw_discrete, list):
        raise ValueError("symmetries_discrete must be a list")
    discrete = []
    for index, values in enumerate(raw_discrete):
        try:
            transform = np.asarray(values, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"symmetries_discrete[{index}] must contain numeric values"
            ) from error
        if transform.shape != (16,):
            raise ValueError(
                f"symmetries_discrete[{index}] must contain 16 values"
            )
        if not np.all(np.isfinite(transform)):
            raise ValueError(
                f"symmetries_discrete[{index}] must contain only finite values"
            )
        transform = transform.reshape(4, 4)
        if not np.allclose(
            transform[3], (0.0, 0.0, 0.0, 1.0), atol=_ROTATION_ATOL, rtol=0.0
        ):
            raise ValueError(
                f"symmetries_discrete[{index}] must be a homogeneous transform"
            )
        try:
            discrete.append(DiscreteSymmetry(transform[:3, :3], transform[:3, 3]))
        except ValueError as error:
            raise ValueError(f"Invalid symmetries_discrete[{index}]: {error}") from error

    raw_continuous = model_info.get("symmetries_continuous", [])
    if not isinstance(raw_continuous, list):
        raise ValueError("symmetries_continuous must be a list")
    continuous = []
    for index, definition in enumerate(raw_continuous):
        if not isinstance(definition, dict):
            raise ValueError(f"symmetries_continuous[{index}] must be an object")
        if "axis" not in definition or "offset" not in definition:
            raise ValueError(
                f"symmetries_continuous[{index}] must contain axis and offset"
            )
        try:
            continuous.append(
                ContinuousSymmetry(definition["axis"], definition["offset"])
            )
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid symmetries_continuous[{index}]: {error}") from error

    return SymmetryConfig(discrete, continuous)


@dataclass(frozen=True)
class CostMatrix:
    """Pairwise errors, feasibility mask, and normalized assignment costs."""

    costs: np.ndarray
    feasible: np.ndarray
    translation_errors_mm: np.ndarray
    rotation_errors_deg: np.ndarray


@dataclass(frozen=True)
class AssociationRecord:
    prediction_index: int
    ground_truth_index: int
    prediction_id: str | int | None
    ground_truth_id: str | int | None
    translation_error_mm: float
    rotation_error_deg: float
    normalized_cost: float


@dataclass(frozen=True)
class ErrorStatistics:
    count: int
    mean: float | None
    median: float | None
    std: float | None
    minimum: float | None
    maximum: float | None
    rmse: float | None


@dataclass(frozen=True)
class EvaluationResult:
    associations: tuple[AssociationRecord, ...]
    unmatched_prediction_indices: tuple[int, ...]
    unmatched_ground_truth_indices: tuple[int, ...]
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    translation_error_mm: ErrorStatistics
    rotation_error_deg: ErrorStatistics


def translation_error_mm(prediction: PoseRecord, ground_truth: PoseRecord) -> float:
    """Return Euclidean translation error in millimetres."""

    return float(np.linalg.norm(prediction.translation_mm - ground_truth.translation_mm))


def so3_geodesic_angle_deg(rotation_a: np.ndarray, rotation_b: np.ndarray) -> float:
    """Return the SO(3) geodesic angle between two validated rotations."""

    relative = rotation_a.T @ rotation_b
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def discrete_symmetry_rotation_error_deg(
    prediction_rotation: np.ndarray,
    ground_truth_rotation: np.ndarray,
    symmetry_rotations: Sequence[np.ndarray],
) -> float:
    """Return minimum geodesic error over a non-empty finite symmetry group."""

    if not symmetry_rotations:
        raise ValueError("symmetry_rotations must not be empty")
    return min(
        so3_geodesic_angle_deg(
            prediction_rotation, ground_truth_rotation @ symmetry_rotation
        )
        for symmetry_rotation in symmetry_rotations
    )


def continuous_axis_rotation_error_deg(
    prediction_rotation: np.ndarray,
    ground_truth_rotation: np.ndarray,
    object_axis: np.ndarray,
) -> float:
    """Return angular error between the two transformed directed axes."""

    prediction_axis = prediction_rotation @ object_axis
    ground_truth_axis = ground_truth_rotation @ object_axis
    cosine = np.clip(np.dot(prediction_axis, ground_truth_axis), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def rotation_error_deg(
    prediction: PoseRecord,
    ground_truth: PoseRecord,
    symmetry: SymmetryConfig | None = None,
) -> float:
    """Return symmetry-aware rotation error in degrees."""

    if symmetry is None:
        return so3_geodesic_angle_deg(prediction.rotation, ground_truth.rotation)
    if not isinstance(symmetry, SymmetryConfig):
        raise TypeError("symmetry must be a SymmetryConfig or None")
    _validate_supported_continuous_symmetries(symmetry)
    errors = []
    for discrete in symmetry.discrete_symmetries:
        equivalent_rotation = ground_truth.rotation @ discrete.rotation
        if symmetry.continuous_symmetries:
            errors.extend(
                continuous_axis_rotation_error_deg(
                    prediction.rotation, equivalent_rotation, continuous.axis
                )
                for continuous in symmetry.continuous_symmetries
            )
        else:
            errors.append(
                so3_geodesic_angle_deg(prediction.rotation, equivalent_rotation)
            )
    return min(errors)


def _validate_supported_continuous_symmetries(symmetry: SymmetryConfig) -> None:
    for index, continuous in enumerate(symmetry.continuous_symmetries):
        if np.any(continuous.offset_mm != 0.0):
            raise NotImplementedError(
                "Evaluation of nonzero continuous symmetry offsets is not supported "
                f"(continuous_symmetries[{index}].offset_mm)"
            )


def _joint_symmetry_errors(
    prediction: PoseRecord,
    ground_truth: PoseRecord,
    symmetry: SymmetryConfig | None,
    translation_threshold_mm: float,
    rotation_threshold_deg: float,
) -> tuple[float, float]:
    if symmetry is None:
        return (
            translation_error_mm(prediction, ground_truth),
            so3_geodesic_angle_deg(prediction.rotation, ground_truth.rotation),
        )

    candidates = []
    for discrete in symmetry.discrete_symmetries:
        equivalent_translation = (
            ground_truth.translation_mm
            + ground_truth.rotation @ discrete.translation_mm
        )
        translation_error = float(
            np.linalg.norm(prediction.translation_mm - equivalent_translation)
        )
        equivalent_rotation = ground_truth.rotation @ discrete.rotation
        if symmetry.continuous_symmetries:
            rotation_error = min(
                continuous_axis_rotation_error_deg(
                    prediction.rotation, equivalent_rotation, continuous.axis
                )
                for continuous in symmetry.continuous_symmetries
            )
        else:
            rotation_error = so3_geodesic_angle_deg(
                prediction.rotation, equivalent_rotation
            )
        normalized_cost = (
            translation_error / translation_threshold_mm
            + rotation_error / rotation_threshold_deg
        )
        candidates.append((normalized_cost, translation_error, rotation_error))

    feasible_candidates = [
        item
        for item in candidates
        if item[1] <= translation_threshold_mm + _THRESHOLD_ATOL
        and item[2] <= rotation_threshold_deg + _THRESHOLD_ATOL
    ]
    selected_candidates = feasible_candidates or candidates
    _, translation_error, rotation_error = min(
        selected_candidates, key=lambda item: item[0]
    )
    return translation_error, rotation_error


def build_feasible_cost_matrix(
    predictions: Sequence[PoseRecord],
    ground_truths: Sequence[PoseRecord],
    symmetry: SymmetryConfig | None = None,
    translation_threshold_mm: float = DEFAULT_TRANSLATION_THRESHOLD_MM,
    rotation_threshold_deg: float = DEFAULT_ROTATION_THRESHOLD_DEG,
) -> CostMatrix:
    """Build pairwise errors and a threshold-gated normalized cost matrix.

    Rows correspond to predictions and columns to ground truths. Infeasible
    entries have infinite cost. Both thresholds are inclusive.
    """

    _validate_threshold(translation_threshold_mm, "translation_threshold_mm")
    _validate_threshold(rotation_threshold_deg, "rotation_threshold_deg")
    if symmetry is not None and not isinstance(symmetry, SymmetryConfig):
        raise TypeError("symmetry must be a SymmetryConfig or None")
    if symmetry is not None:
        _validate_supported_continuous_symmetries(symmetry)
    for index, prediction in enumerate(predictions):
        if not isinstance(prediction, PoseRecord):
            raise TypeError(f"predictions[{index}] must be a PoseRecord")
    for index, ground_truth in enumerate(ground_truths):
        if not isinstance(ground_truth, PoseRecord):
            raise TypeError(f"ground_truths[{index}] must be a PoseRecord")

    shape = (len(predictions), len(ground_truths))
    translation_errors = np.empty(shape, dtype=np.float64)
    rotation_errors = np.empty(shape, dtype=np.float64)

    for prediction_index, prediction in enumerate(predictions):
        for ground_truth_index, ground_truth in enumerate(ground_truths):
            translation_error, rotation_error = _joint_symmetry_errors(
                prediction,
                ground_truth,
                symmetry,
                translation_threshold_mm,
                rotation_threshold_deg,
            )
            translation_errors[prediction_index, ground_truth_index] = translation_error
            rotation_errors[prediction_index, ground_truth_index] = rotation_error

    translation_feasible = (translation_errors <= translation_threshold_mm) | np.isclose(
        translation_errors,
        translation_threshold_mm,
        atol=_THRESHOLD_ATOL,
        rtol=0.0,
    )
    rotation_feasible = (rotation_errors <= rotation_threshold_deg) | np.isclose(
        rotation_errors,
        rotation_threshold_deg,
        atol=_THRESHOLD_ATOL,
        rtol=0.0,
    )
    feasible = translation_feasible & rotation_feasible
    normalized = (
        translation_errors / translation_threshold_mm
        + rotation_errors / rotation_threshold_deg
    )
    costs = np.where(feasible, normalized, np.inf)
    return CostMatrix(costs, feasible, translation_errors, rotation_errors)


def evaluate_poses(
    predictions: Sequence[PoseRecord],
    ground_truths: Sequence[PoseRecord],
    symmetry: SymmetryConfig | None = None,
    translation_threshold_mm: float = DEFAULT_TRANSLATION_THRESHOLD_MM,
    rotation_threshold_deg: float = DEFAULT_ROTATION_THRESHOLD_DEG,
) -> EvaluationResult:
    """Strictly evaluate poses using maximum-cardinality one-to-one matching.

    The assignment objective first maximizes the number of feasible matches,
    then minimizes their total normalized translation-plus-rotation error.
    Duplicate detections can therefore match at most one ground-truth pose.
    Metrics with a zero denominator are reported as 0.0.
    """

    matrix = build_feasible_cost_matrix(
        predictions,
        ground_truths,
        symmetry,
        translation_threshold_mm,
        rotation_threshold_deg,
    )
    associations: list[AssociationRecord] = []

    if len(predictions) > 0 and len(ground_truths) > 0:
        assignment_size = min(len(predictions), len(ground_truths))
        finite_feasible_costs = matrix.costs[
            matrix.feasible & np.isfinite(matrix.costs)
        ]
        max_feasible_cost = (
            float(np.max(finite_feasible_costs))
            if finite_feasible_costs.size
            else 0.0
        )
        max_float = np.finfo(np.float64).max
        if max_feasible_cost <= max_float / assignment_size:
            max_feasible_assignment_cost = max_feasible_cost * assignment_size
            infeasible_penalty = np.nextafter(
                max_feasible_assignment_cost, np.inf
            )
        else:
            infeasible_penalty = np.inf

        if np.isfinite(infeasible_penalty):
            assignment_costs = np.where(
                matrix.feasible, matrix.costs, infeasible_penalty
            )
        else:
            # Uniform scaling preserves feasible costs while avoiding overflow.
            assignment_costs = np.where(
                matrix.feasible,
                matrix.costs / max_feasible_cost,
                np.nextafter(float(assignment_size), np.inf),
            )
        prediction_indices, ground_truth_indices = linear_sum_assignment(
            assignment_costs
        )
        for prediction_index, ground_truth_index in zip(
            prediction_indices.tolist(), ground_truth_indices.tolist()
        ):
            if not matrix.feasible[prediction_index, ground_truth_index]:
                continue
            prediction = predictions[prediction_index]
            ground_truth = ground_truths[ground_truth_index]
            associations.append(
                AssociationRecord(
                    prediction_index=prediction_index,
                    ground_truth_index=ground_truth_index,
                    prediction_id=prediction.record_id,
                    ground_truth_id=ground_truth.record_id,
                    translation_error_mm=float(
                        matrix.translation_errors_mm[
                            prediction_index, ground_truth_index
                        ]
                    ),
                    rotation_error_deg=float(
                        matrix.rotation_errors_deg[
                            prediction_index, ground_truth_index
                        ]
                    ),
                    normalized_cost=float(
                        matrix.costs[prediction_index, ground_truth_index]
                    ),
                )
            )

    matched_predictions = {item.prediction_index for item in associations}
    matched_ground_truths = {item.ground_truth_index for item in associations}
    unmatched_predictions = tuple(
        index for index in range(len(predictions)) if index not in matched_predictions
    )
    unmatched_ground_truths = tuple(
        index
        for index in range(len(ground_truths))
        if index not in matched_ground_truths
    )
    tp = len(associations)
    fp = len(predictions) - tp
    fn = len(ground_truths) - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0

    return EvaluationResult(
        associations=tuple(associations),
        unmatched_prediction_indices=unmatched_predictions,
        unmatched_ground_truth_indices=unmatched_ground_truths,
        tp=tp,
        fp=fp,
        fn=fn,
        precision=precision,
        recall=recall,
        f1=f1,
        translation_error_mm=_error_statistics(
            [item.translation_error_mm for item in associations]
        ),
        rotation_error_deg=_error_statistics(
            [item.rotation_error_deg for item in associations]
        ),
    )


def _validate_threshold(value: float, name: str) -> None:
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and greater than zero")


def _error_statistics(values: Sequence[float]) -> ErrorStatistics:
    if not values:
        return ErrorStatistics(0, None, None, None, None, None, None)
    array = np.asarray(values, dtype=np.float64)
    return ErrorStatistics(
        count=len(values),
        mean=float(np.mean(array)),
        median=float(np.median(array)),
        std=float(np.std(array)),
        minimum=float(np.min(array)),
        maximum=float(np.max(array)),
        rmse=float(np.sqrt(np.mean(np.square(array)))),
    )
