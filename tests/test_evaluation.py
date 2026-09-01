import math
import unittest

import numpy as np

from evaluation import PoseRecord, SymmetryConfig, evaluate_poses, rotation_error_deg


def rotation_z(degrees):
    radians = math.radians(degrees)
    cosine = math.cos(radians)
    sine = math.sin(radians)
    return np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]
    )


def pose(translation=(0.0, 0.0, 0.0), rotation=None, record_id=None):
    if rotation is None:
        rotation = np.eye(3)
    return PoseRecord(np.asarray(translation), rotation, record_id=record_id)


class RotationErrorTests(unittest.TestCase):
    def test_six_decimal_bop_rotation_is_accepted(self):
        rotation = np.array(
            [
                [-0.671004, 0.630740, 0.389770],
                [0.136158, -0.411922, 0.900989],
                [0.728844, 0.657638, 0.190521],
            ]
        )

        record = pose(rotation=rotation)

        np.testing.assert_array_equal(record.rotation, rotation)

    def test_identical_pose_has_zero_error_and_is_true_positive(self):
        prediction = pose(record_id="prediction")
        ground_truth = pose(record_id="ground-truth")

        result = evaluate_poses((prediction,), (ground_truth,))

        self.assertEqual((result.tp, result.fp, result.fn), (1, 0, 0))
        self.assertAlmostEqual(result.associations[0].translation_error_mm, 0.0)
        self.assertAlmostEqual(result.associations[0].rotation_error_deg, 0.0)
        self.assertEqual((result.precision, result.recall, result.f1), (1.0, 1.0, 1.0))

    def test_rotations_at_359_and_1_degrees_are_two_degrees_apart(self):
        error = rotation_error_deg(
            pose(rotation=rotation_z(359.0)),
            pose(rotation=rotation_z(1.0)),
        )

        self.assertAlmostEqual(error, 2.0, places=10)

    def test_star_thirty_degree_discrete_symmetry_removes_rotation_error(self):
        star_symmetry = SymmetryConfig(
            discrete_rotations=tuple(rotation_z(30.0 * index) for index in range(12))
        )

        error = rotation_error_deg(
            pose(rotation=rotation_z(30.0)),
            pose(),
            symmetry=star_symmetry,
        )

        self.assertAlmostEqual(error, 0.0, places=10)

    def test_continuous_z_axis_symmetry_ignores_rotation_about_z(self):
        z_axis_symmetry = SymmetryConfig(
            continuous_axis=np.array([0.0, 0.0, 1.0])
        )

        error = rotation_error_deg(
            pose(rotation=rotation_z(137.0)),
            pose(),
            symmetry=z_axis_symmetry,
        )

        self.assertAlmostEqual(error, 0.0, places=10)


class PoseAssociationTests(unittest.TestCase):
    def test_duplicate_estimates_produce_one_true_and_one_false_positive(self):
        ground_truth = pose(record_id="ground-truth")
        predictions = (
            pose(record_id="first"),
            pose(record_id="duplicate"),
        )

        result = evaluate_poses(predictions, (ground_truth,))

        self.assertEqual((result.tp, result.fp, result.fn), (1, 1, 0))
        self.assertEqual(len(result.associations), 1)
        self.assertEqual(len(result.unmatched_prediction_indices), 1)

    def test_metrics_and_identity_pairs_are_invariant_to_input_order(self):
        predictions = (
            pose((1.0, 0.0, 0.0), record_id="prediction-a"),
            pose((101.0, 0.0, 0.0), record_id="prediction-b"),
        )
        ground_truths = (
            pose((0.0, 0.0, 0.0), record_id="ground-truth-a"),
            pose((100.0, 0.0, 0.0), record_id="ground-truth-b"),
        )

        forward = evaluate_poses(predictions, ground_truths)
        reversed_inputs = evaluate_poses(
            tuple(reversed(predictions)), tuple(reversed(ground_truths))
        )

        forward_pairs = {
            (item.prediction_id, item.ground_truth_id) for item in forward.associations
        }
        reversed_pairs = {
            (item.prediction_id, item.ground_truth_id)
            for item in reversed_inputs.associations
        }
        self.assertEqual(forward_pairs, reversed_pairs)
        self.assertEqual(forward_pairs, {
            ("prediction-a", "ground-truth-a"),
            ("prediction-b", "ground-truth-b"),
        })
        self.assertEqual(
            (forward.tp, forward.fp, forward.fn, forward.precision, forward.recall),
            (
                reversed_inputs.tp,
                reversed_inputs.fp,
                reversed_inputs.fn,
                reversed_inputs.precision,
                reversed_inputs.recall,
            ),
        )
        self.assertAlmostEqual(
            forward.translation_error_mm.mean,
            reversed_inputs.translation_error_mm.mean,
        )

    def test_empty_prediction_and_ground_truth_collections(self):
        one_pose = pose()
        cases = (
            ((), (), (0, 0, 0)),
            ((one_pose,), (), (0, 1, 0)),
            ((), (one_pose,), (0, 0, 1)),
        )

        for predictions, ground_truths, expected_counts in cases:
            with self.subTest(expected_counts=expected_counts):
                result = evaluate_poses(predictions, ground_truths)
                self.assertEqual((result.tp, result.fp, result.fn), expected_counts)
                self.assertEqual((result.precision, result.recall, result.f1), (0.0, 0.0, 0.0))
                self.assertEqual(result.translation_error_mm.count, 0)
                self.assertIsNone(result.translation_error_mm.mean)
                self.assertEqual(result.rotation_error_deg.count, 0)
                self.assertIsNone(result.rotation_error_deg.mean)

    def test_pose_exactly_on_both_thresholds_is_feasible(self):
        result = evaluate_poses(
            (pose((10.0, 0.0, 0.0), rotation_z(10.0)),),
            (pose(),),
            translation_threshold_mm=10.0,
            rotation_threshold_deg=10.0,
        )

        self.assertEqual((result.tp, result.fp, result.fn), (1, 0, 0))
        association = result.associations[0]
        self.assertAlmostEqual(association.translation_error_mm, 10.0, places=10)
        self.assertAlmostEqual(association.rotation_error_deg, 10.0, places=10)
        self.assertAlmostEqual(association.normalized_cost, 2.0, places=10)

    def test_matching_maximizes_cardinality_before_minimizing_cost(self):
        predictions = (
            pose((4.0, 0.0, 0.0), record_id="shared"),
            pose((-9.0, 0.0, 0.0), record_id="only-ground-truth-1"),
        )
        ground_truths = (
            pose((0.0, 0.0, 0.0), record_id="ground-truth-1"),
            pose((9.0, 0.0, 0.0), record_id="ground-truth-2"),
        )

        result = evaluate_poses(
            predictions,
            ground_truths,
            translation_threshold_mm=10.0,
        )

        self.assertEqual((result.tp, result.fp, result.fn), (2, 0, 0))
        self.assertEqual(
            {(item.prediction_id, item.ground_truth_id) for item in result.associations},
            {
                ("shared", "ground-truth-2"),
                ("only-ground-truth-1", "ground-truth-1"),
            },
        )


if __name__ == "__main__":
    unittest.main()
