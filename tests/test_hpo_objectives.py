import unittest

import numpy as np

from evaluation import PoseRecord
from hpo_objectives import (
    StrictAssociationRecallFirstV1,
    StrictAssociationScene,
    StrictAssociationV2,
)


class StrictAssociationObjectiveTests(unittest.TestCase):
    @staticmethod
    def _pose(x_mm, record_id):
        return PoseRecord(
            translation_mm=np.array((x_mm, 0.0, 0.0)),
            rotation=np.eye(3),
            record_id=record_id,
        )

    def test_current_objective_version_is_fixed_penalty(self):
        self.assertEqual(StrictAssociationV2.OBJECTIVE_VERSION, "fixed-penalty-baseline")

    def test_v2_numerical_semantics_are_unchanged(self):
        scene = StrictAssociationScene(
            scene_id="v2-golden",
            predictions=(self._pose(0.0, "tp"), self._pose(1000.0, "fp")),
            ground_truths=(self._pose(0.0, "gt-1"), self._pose(100.0, "gt-2")),
        )

        detail = StrictAssociationV2().evaluate_scene(scene)

        self.assertEqual((detail.matched_count, detail.false_positives, detail.false_negatives), (1, 1, 1))
        self.assertAlmostEqual(detail.objective, 60.0)

    def test_recall_first_prefers_tp_despite_maximum_fp_disadvantage(self):
        evaluator = StrictAssociationRecallFirstV1(num_matches=3, query_count=1)
        ground_truth = (self._pose(0.0, "gt"),)
        with_tp = StrictAssociationScene(
            scene_id="with-tp",
            predictions=(
                self._pose(0.0, "tp"),
                self._pose(100.0, "fp-1"),
                self._pose(200.0, "fp-2"),
            ),
            ground_truths=ground_truth,
        )
        empty = StrictAssociationScene(
            scene_id="empty", predictions=(), ground_truths=ground_truth
        )

        self.assertLess(
            evaluator.evaluate_scene(with_tp).objective,
            evaluator.evaluate_scene(empty).objective,
        )

    def test_recall_first_uses_fp_then_pose_error_as_tie_breakers(self):
        evaluator = StrictAssociationRecallFirstV1(num_matches=3, query_count=1)
        ground_truth = (self._pose(0.0, "gt"),)
        exact = StrictAssociationScene(
            scene_id="exact",
            predictions=(self._pose(0.0, "tp"),),
            ground_truths=ground_truth,
        )
        less_accurate = StrictAssociationScene(
            scene_id="less-accurate",
            predictions=(self._pose(5.0, "tp"),),
            ground_truths=ground_truth,
        )
        extra_fp = StrictAssociationScene(
            scene_id="extra-fp",
            predictions=(self._pose(0.0, "tp"), self._pose(100.0, "fp")),
            ground_truths=ground_truth,
        )

        exact_loss = evaluator.evaluate_scene(exact).objective
        less_accurate_loss = evaluator.evaluate_scene(less_accurate).objective
        extra_fp_loss = evaluator.evaluate_scene(extra_fp).objective
        self.assertLess(exact_loss, less_accurate_loss)
        self.assertLess(less_accurate_loss, extra_fp_loss)

    def test_recall_first_rejects_empty_gt_and_prediction_overflow(self):
        evaluator = StrictAssociationRecallFirstV1(num_matches=1, query_count=1)
        with self.assertRaisesRegex(ValueError, "at least one ground truth"):
            evaluator.evaluate_scene(
                StrictAssociationScene(scene_id="empty-gt", predictions=(), ground_truths=())
            )
        with self.assertRaisesRegex(ValueError, "exceeds"):
            evaluator.evaluate_scene(
                StrictAssociationScene(
                    scene_id="overflow",
                    predictions=(self._pose(0.0, "one"), self._pose(1.0, "two")),
                    ground_truths=(self._pose(0.0, "gt"),),
                )
            )

    def test_recall_first_dominance_holds_across_query_aggregation(self):
        evaluator = StrictAssociationRecallFirstV1(num_matches=10, query_count=3)
        ground_truth = (self._pose(0.0, "gt"),)
        empty_scenes = tuple(
            StrictAssociationScene(
                scene_id=f"empty-{index}", predictions=(), ground_truths=ground_truth
            )
            for index in range(3)
        )
        false_predictions = tuple(
            self._pose(100.0 + index * 20.0, f"fp-{index}")
            for index in range(10)
        )
        one_tp_scenes = (
            StrictAssociationScene(
                scene_id="tp",
                predictions=(self._pose(0.0, "tp"), *false_predictions[:9]),
                ground_truths=ground_truth,
            ),
            StrictAssociationScene(
                scene_id="miss-1",
                predictions=false_predictions,
                ground_truths=ground_truth,
            ),
            StrictAssociationScene(
                scene_id="miss-2",
                predictions=false_predictions,
                ground_truths=ground_truth,
            ),
        )

        self.assertLess(
            evaluator.evaluate(one_tp_scenes).objective,
            evaluator.evaluate(empty_scenes).objective,
        )


if __name__ == "__main__":
    unittest.main()
