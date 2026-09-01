"""Versioned, HALCON-independent objectives for HPO pose evaluation.

Retains exclusively the two primary objectives evaluated in the thesis ablation study:
1. StrictAssociationRecallFirstV1 ("strict-association-recall-first-v1"): Lexicographical Recall-First
2. StrictAssociationV2 ("strict-association-v2"): Fixed penalty baseline
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite
from typing import Protocol, Sequence

from evaluation import (
    DEFAULT_ROTATION_THRESHOLD_DEG,
    DEFAULT_TRANSLATION_THRESHOLD_MM,
    EvaluationResult,
    PoseRecord,
    SymmetryConfig,
    evaluate_poses,
)


LEXICOGRAPHICAL_RECALL_FIRST = "lexicographical-recall-first"
FIXED_PENALTY_BASELINE = "fixed-penalty-baseline"

OBJECTIVE_VERSIONS = (
    LEXICOGRAPHICAL_RECALL_FIRST,
    FIXED_PENALTY_BASELINE,
)


class TrialReporter(Protocol):
    """Minimal interface required from an Optuna-like trial."""

    def report(self, value: float, step: int) -> None: ...


@dataclass(frozen=True)
class SceneObjectiveDetail:
    """Objective components for one evaluated scene."""

    scene_id: str | int
    objective: float
    matched_mean_error: float
    matched_count: int
    false_negatives: int
    false_positives: int
    cumulative_mean: float | None = None
    evaluation: EvaluationResult | None = None


@dataclass(frozen=True)
class ObjectiveResult:
    """Mean objective and ordered per-scene details."""

    version: str
    objective: float
    scenes: tuple[SceneObjectiveDetail, ...]

    def report(self, trial: TrialReporter, step_offset: int = 0) -> None:
        """Report each running scene mean in evaluation order."""

        if step_offset < 0:
            raise ValueError("step_offset must be non-negative")
        for step, scene in enumerate(self.scenes, start=step_offset):
            if scene.cumulative_mean is None:
                raise ValueError("scene cumulative_mean has not been computed")
            trial.report(scene.cumulative_mean, step)


@dataclass(frozen=True)
class StrictAssociationScene:
    """A scene representation for one-to-one association evaluation."""

    scene_id: str | int
    predictions: tuple[PoseRecord, ...]
    ground_truths: tuple[PoseRecord, ...]
    symmetry: SymmetryConfig | None = None


class StrictAssociationV2:
    """Fixed-penalty baseline objective based on strict one-to-one pose association.

    Default: heavy penalty for missed targets (FN=60.0) and zero penalty for FP.
    """

    OBJECTIVE_VERSION = FIXED_PENALTY_BASELINE

    def __init__(
        self,
        translation_threshold_mm: float = DEFAULT_TRANSLATION_THRESHOLD_MM,
        rotation_threshold_deg: float = DEFAULT_ROTATION_THRESHOLD_DEG,
        fn_penalty: float = 60.0,
        fp_penalty: float = 0.0,
    ) -> None:
        _validate_positive_finite(
            translation_threshold_mm, "translation_threshold_mm"
        )
        _validate_positive_finite(rotation_threshold_deg, "rotation_threshold_deg")
        _validate_non_negative_finite(fn_penalty, "fn_penalty")
        _validate_non_negative_finite(fp_penalty, "fp_penalty")
        self.translation_threshold_mm = float(translation_threshold_mm)
        self.rotation_threshold_deg = float(rotation_threshold_deg)
        self.fn_penalty = float(fn_penalty)
        self.fp_penalty = float(fp_penalty)

    def evaluate_scene(self, scene: StrictAssociationScene) -> SceneObjectiveDetail:
        if not isinstance(scene, StrictAssociationScene):
            raise TypeError("scene must be a StrictAssociationScene")
        evaluation = evaluate_poses(
            scene.predictions,
            scene.ground_truths,
            symmetry=scene.symmetry,
            translation_threshold_mm=self.translation_threshold_mm,
            rotation_threshold_deg=self.rotation_threshold_deg,
        )
        matched_mean = (
            sum(item.normalized_cost for item in evaluation.associations) / evaluation.tp
            if evaluation.tp
            else 0.0
        )
        objective = (
            matched_mean
            + self.fn_penalty * evaluation.fn
            + self.fp_penalty * evaluation.fp
        )
        return SceneObjectiveDetail(
            scene_id=scene.scene_id,
            objective=objective,
            matched_mean_error=matched_mean,
            matched_count=evaluation.tp,
            false_negatives=evaluation.fn,
            false_positives=evaluation.fp,
            evaluation=evaluation,
        )

    def evaluate(self, scenes: Sequence[StrictAssociationScene]) -> ObjectiveResult:
        return _aggregate(self.OBJECTIVE_VERSION, [self.evaluate_scene(s) for s in scenes])


class StrictAssociationRecallFirstV1(StrictAssociationV2):
    """Lexicographical Recall-First objective with dynamic W_FN weighting.

    Rank order: Maximize Recall (Minimize FN) -> Minimize FP -> Minimize Pose Error.
    """

    OBJECTIVE_VERSION = LEXICOGRAPHICAL_RECALL_FIRST

    def __init__(
        self,
        num_matches: int,
        query_count: int,
        translation_threshold_mm: float = DEFAULT_TRANSLATION_THRESHOLD_MM,
        rotation_threshold_deg: float = DEFAULT_ROTATION_THRESHOLD_DEG,
    ) -> None:
        if (
            not isinstance(num_matches, int)
            or isinstance(num_matches, bool)
            or num_matches < 0
        ):
            raise ValueError("num_matches must be a non-negative integer")
        if (
            not isinstance(query_count, int)
            or isinstance(query_count, bool)
            or query_count <= 0
        ):
            raise ValueError("query_count must be a positive integer")
        effective_bound = 20 if num_matches == 0 else num_matches
        fn_weight = query_count * effective_bound + 1
        super().__init__(
            translation_threshold_mm=translation_threshold_mm,
            rotation_threshold_deg=rotation_threshold_deg,
            fn_penalty=float(fn_weight),
            fp_penalty=1.0,
        )
        self.num_matches = num_matches
        self.query_count = query_count
        self.fn_weight = fn_weight
        self.rank_scale = 2.0 * query_count + 1.0

    def evaluate_scene(self, scene: StrictAssociationScene) -> SceneObjectiveDetail:
        if not isinstance(scene, StrictAssociationScene):
            raise TypeError("scene must be a StrictAssociationScene")
        if len(scene.ground_truths) == 0:
            raise ValueError("recall-first objective requires at least one ground truth")
        if self.num_matches > 0 and len(scene.predictions) > self.num_matches:
            raise ValueError(
                "prediction count exceeds the objective's frozen num_matches"
            )

        detail = super().evaluate_scene(scene)
        rank = (
            detail.false_negatives * self.fn_weight
            + detail.false_positives
        )
        objective = rank * self.rank_scale + detail.matched_mean_error
        return replace(detail, objective=objective)


def _aggregate(
    version: str, details: Sequence[SceneObjectiveDetail]
) -> ObjectiveResult:
    if not details:
        raise ValueError("scenes must not be empty")
    total = 0.0
    accumulated = []
    for count, detail in enumerate(details, start=1):
        total += detail.objective
        accumulated.append(replace(detail, cumulative_mean=total / count))
    return ObjectiveResult(version, total / len(accumulated), tuple(accumulated))


def _validate_positive_finite(value: float, name: str) -> None:
    if isinstance(value, bool) or not isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and greater than zero")


def _validate_non_negative_finite(value: float, name: str) -> None:
    if isinstance(value, bool) or not isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")


# Self-explanatory class aliases
LexicographicalRecallFirstObjective = StrictAssociationRecallFirstV1
FixedPenaltyLinearObjective = StrictAssociationV2

