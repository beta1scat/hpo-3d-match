import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluation import (
    ContinuousSymmetry,
    DiscreteSymmetry,
    PoseRecord,
    SymmetryConfig,
    evaluate_poses,
    read_bop_symmetry,
)


def rotation_z(degrees):
    radians = math.radians(degrees)
    cosine = math.cos(radians)
    sine = math.sin(radians)
    return np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]
    )


def bop_transform(rotation, translation):
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform.reshape(-1).tolist()


def pose(translation=(0.0, 0.0, 0.0), rotation=None):
    return PoseRecord(
        np.asarray(translation, dtype=np.float64),
        np.eye(3) if rotation is None else rotation,
    )


class BopSymmetryTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.models_info_path = Path(self.temporary_directory.name) / "models_info.json"

    def tearDown(self):
        self.temporary_directory.cleanup()

    def write_models_info(self, value):
        self.models_info_path.write_text(json.dumps(value), encoding="utf-8")

    def test_obj5_discrete_rotation_and_translation_are_evaluated_jointly(self):
        self.write_models_info(
            {
                "5": {
                    "symmetries_discrete": [
                        bop_transform(rotation_z(180.0), [100.0, 0.0, 0.0])
                    ]
                }
            }
        )
        symmetry = read_bop_symmetry(self.models_info_path, 5)
        ground_truth = pose(rotation=rotation_z(90.0))
        prediction = pose((0.0, 100.0, 0.0), rotation_z(270.0))

        result = evaluate_poses((prediction,), (ground_truth,), symmetry=symmetry)

        self.assertEqual(len(symmetry.discrete_symmetries), 2)
        self.assertEqual(result.tp, 1)
        self.assertAlmostEqual(result.associations[0].translation_error_mm, 0.0)
        self.assertAlmostEqual(result.associations[0].rotation_error_deg, 0.0)

    def test_obj24_continuous_zero_offset_axis_is_analytic(self):
        self.write_models_info(
            {
                "24": {
                    "symmetries_continuous": [
                        {"axis": [0.0, 0.0, 1.0], "offset": [0.0, 0.0, 0.0]}
                    ]
                }
            }
        )
        symmetry = read_bop_symmetry(self.models_info_path, 24)

        result = evaluate_poses(
            (pose(rotation=rotation_z(137.0)),),
            (pose(),),
            symmetry=symmetry,
        )

        self.assertEqual(len(symmetry.continuous_symmetries), 1)
        self.assertAlmostEqual(result.associations[0].rotation_error_deg, 0.0)

    def test_obj25_has_twelve_discrete_symmetries_including_identity(self):
        self.write_models_info(
            {
                "25": {
                    "symmetries_discrete": [
                        bop_transform(rotation_z(30.0 * index), [0.0, 0.0, 0.0])
                        for index in range(1, 12)
                    ]
                }
            }
        )

        symmetry = read_bop_symmetry(self.models_info_path, 25)

        self.assertEqual(len(symmetry.discrete_symmetries), 12)
        identity = symmetry.discrete_symmetries[0]
        np.testing.assert_allclose(identity.rotation, np.eye(3))
        np.testing.assert_allclose(identity.translation_mm, np.zeros(3))

    def test_discrete_and_continuous_symmetries_can_be_combined(self):
        self.write_models_info(
            {
                "5": {
                    "symmetries_discrete": [
                        bop_transform(rotation_z(180.0), [20.0, 0.0, 0.0])
                    ],
                    "symmetries_continuous": [
                        {"axis": [1.0, 0.0, 0.0], "offset": [0.0, 0.0, 0.0]}
                    ],
                }
            }
        )
        symmetry = read_bop_symmetry(self.models_info_path, 5)
        prediction_rotation = rotation_z(180.0) @ np.array(
            [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
        )

        result = evaluate_poses(
            (pose((20.0, 0.0, 0.0), prediction_rotation),),
            (pose(),),
            symmetry=symmetry,
        )

        self.assertEqual(result.tp, 1)
        self.assertAlmostEqual(result.associations[0].translation_error_mm, 0.0)
        self.assertAlmostEqual(result.associations[0].rotation_error_deg, 0.0)

    def test_pair_uses_one_symmetry_for_translation_and_rotation(self):
        symmetry = SymmetryConfig(
            discrete_symmetries=(
                DiscreteSymmetry(rotation_z(90.0), np.array([10.0, 0.0, 0.0])),
            )
        )

        result = evaluate_poses(
            (pose(rotation=rotation_z(90.0)),),
            (pose(),),
            symmetry=symmetry,
            translation_threshold_mm=20.0,
            rotation_threshold_deg=10.0,
        )

        association = result.associations[0]
        self.assertAlmostEqual(association.translation_error_mm, 10.0)
        self.assertAlmostEqual(association.rotation_error_deg, 0.0)
        self.assertAlmostEqual(association.normalized_cost, 0.5)

    def test_feasible_symmetry_is_preferred_over_lower_infeasible_cost(self):
        symmetry = SymmetryConfig(
            discrete_symmetries=(
                DiscreteSymmetry(rotation_z(6.0), np.array([5.0, 0.0, 0.0])),
            )
        )

        result = evaluate_poses(
            (pose((11.0, 0.0, 0.0)),),
            (pose(),),
            symmetry=symmetry,
            translation_threshold_mm=10.0,
            rotation_threshold_deg=10.0,
        )

        self.assertEqual(result.tp, 1)
        self.assertAlmostEqual(result.associations[0].translation_error_mm, 6.0)
        self.assertAlmostEqual(result.associations[0].rotation_error_deg, 6.0)
        self.assertAlmostEqual(result.associations[0].normalized_cost, 1.2)

    def test_nonzero_continuous_offset_is_rejected_during_evaluation(self):
        symmetry = SymmetryConfig(
            continuous_symmetries=(
                ContinuousSymmetry(
                    np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0])
                ),
            )
        )

        with self.assertRaisesRegex(NotImplementedError, "nonzero continuous"):
            evaluate_poses((pose(),), (pose(),), symmetry=symmetry)

    def test_malformed_models_info_is_rejected(self):
        malformed_values = (
            [],
            {"5": {"symmetries_discrete": [[1.0] * 15]}},
            {
                "5": {
                    "symmetries_continuous": [
                        {"axis": [0.0, 0.0, 2.0], "offset": [0.0, 0.0, 0.0]}
                    ]
                }
            },
            {"5": {"symmetries_continuous": [{"axis": [0.0, 0.0, 1.0]}]}},
        )
        for value in malformed_values:
            with self.subTest(value=value):
                self.write_models_info(value)
                with self.assertRaises(ValueError):
                    read_bop_symmetry(self.models_info_path, 5)

    def test_missing_object_is_rejected(self):
        self.write_models_info({"24": {}})

        with self.assertRaisesRegex(ValueError, "obj_id 5 is missing"):
            read_bop_symmetry(self.models_info_path, 5)


if __name__ == "__main__":
    unittest.main()
