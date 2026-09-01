import json
import tempfile
import unittest
from pathlib import Path

from bop_scene_loader import read_bop_ground_truths
from dataset import (
    BOP_MANIFEST_FIELDS,
    DatasetError,
    ITODD_EXTERNAL_MANIFEST_FIELDS,
    build_bop_manifest,
    load_bop_scene,
)


IDENTITY_ROTATION = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]


def _write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


class DatasetProtocolTests(unittest.TestCase):
    def test_manifest_field_contracts_are_explicit_and_external_is_unchanged(self):
        self.assertEqual(
            BOP_MANIFEST_FIELDS,
            (
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
            ),
        )
        self.assertEqual(
            ITODD_EXTERNAL_MANIFEST_FIELDS,
            (
                "scene_id",
                "model_name",
                "obj_id",
                "split",
                "cad_path",
                "x_path",
                "y_path",
                "z_path",
            ),
        )

    def test_scene_loader_rejects_gt_info_index_misalignment(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            scene_dir = Path(temporary_directory) / "000001"
            scene_dir.mkdir()
            _write_json(scene_dir / "scene_gt.json", {"0": [{"obj_id": 5}]})
            _write_json(scene_dir / "scene_gt_info.json", {"0": []})
            _write_json(scene_dir / "scene_camera.json", {"0": {}})

            with self.assertRaisesRegex(DatasetError, "index alignment mismatch"):
                load_bop_scene(scene_dir)

    def test_ground_truth_visibility_is_inclusive_and_preserves_gt_index(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            gt_path = root / "scene_gt.json"
            info_path = root / "scene_gt_info.json"
            annotations = [
                self._pose(5, 0.0),
                self._pose(24, 1.0),
                self._pose(5, 2.0),
                self._pose(5, 3.0),
            ]
            _write_json(gt_path, {"0": annotations})
            _write_json(
                info_path,
                {"0": [{"visib_fract": value} for value in (0.09, 1.0, 0.1, 0.8)]},
            )

            poses = read_bop_ground_truths(
                gt_path,
                info_path,
                scene_id=1,
                image_id=0,
                obj_id=5,
                min_visib_fract=0.1,
            )

            self.assertEqual([pose.record_id for pose in poses], [2, 3])

    def test_manifest_uses_scene_splits_and_independent_proportional_limits(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            scenes_root = root / "scenes"
            models_dir = root / "models"
            scenes_root.mkdir()
            models_dir.mkdir()
            self._create_models(models_dir)
            for scene_id in range(1, 5):
                self._create_scene(scenes_root, scene_id)

            rows, metadata = build_bop_manifest(
                scenes_root,
                models_dir,
                source="itodd-bop",
                train_ratio=0.5,
                seed=17,
                train_query_limit_per_object=2,
                dev_query_limit_per_object=4,
                test_query_limit_per_object=0,
            )

            self.assertEqual(metadata["schema_version"], 2)
            self.assertEqual(metadata["group_by"], ["scene_id"])
            self.assertEqual(metadata["min_visib_fract"], 0.1)
            self.assertEqual(metadata["visibility_counts"]["target_annotations_total"], 24)
            self.assertEqual(metadata["visibility_counts"]["target_annotations_included"], 12)
            self.assertEqual(metadata["visibility_counts"]["target_annotations_excluded"], 12)
            self.assertEqual(
                metadata["sampling_counts"]["before"]["train"]["bracket_planar"],
                6,
            )
            self.assertEqual(
                metadata["sampling_counts"]["after"]["train"]["bracket_planar"],
                2,
            )
            self.assertEqual(
                metadata["sampling_counts"]["after"]["dev"]["bracket_planar"],
                4,
            )

            train_scenes = set(metadata["splits"]["train"])
            dev_scenes = set(metadata["splits"]["dev"])
            self.assertEqual(len(train_scenes), 2)
            self.assertEqual(len(dev_scenes), 2)
            self.assertTrue(train_scenes.isdisjoint(dev_scenes))
            self.assertEqual(
                {int(row["scene_id"]) for row in rows if row["split"] == "train"},
                train_scenes,
            )
            self.assertEqual(
                {int(row["scene_id"]) for row in rows if row["split"] == "dev"},
                dev_scenes,
            )
            self.assertTrue(all(row["source"] == "itodd-bop" for row in rows))
            self.assertTrue(all(row["min_visib_fract"] == 0.1 for row in rows))
            self.assertEqual(
                set(BOP_MANIFEST_FIELDS),
                set(rows[0]),
            )

            repeated_rows, repeated_metadata = build_bop_manifest(
                scenes_root,
                models_dir,
                source="itodd-bop",
                train_ratio=0.5,
                seed=17,
                train_query_limit_per_object=2,
                dev_query_limit_per_object=4,
                test_query_limit_per_object=0,
            )
            self.assertEqual(rows, repeated_rows)
            self.assertEqual(metadata, repeated_metadata)

    def test_fixed_split_accepts_train_dev_and_test(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            scenes_root = root / "scenes"
            models_dir = root / "models"
            scenes_root.mkdir()
            models_dir.mkdir()
            self._create_models(models_dir)
            self._create_scene(scenes_root, 1)

            for split in ("train", "dev", "test"):
                rows, metadata = build_bop_manifest(
                    scenes_root,
                    models_dir,
                    source="itodd-bop",
                    fixed_split=split,
                )
                self.assertEqual({row["split"] for row in rows}, {split})
                self.assertEqual(metadata["splits"][split], [1])

    @staticmethod
    def _pose(obj_id, translation):
        return {
            "obj_id": obj_id,
            "cam_R_m2c": IDENTITY_ROTATION,
            "cam_t_m2c": [translation, 0.0, 100.0],
        }

    @staticmethod
    def _create_models(models_dir):
        _write_json(
            models_dir / "models_info.json",
            {"5": {}, "24": {}, "25": {}},
        )
        for obj_id in (5, 24, 25):
            (models_dir / f"obj_{obj_id:06d}.ply").touch()

    def _create_scene(self, scenes_root, scene_id):
        scene_dir = scenes_root / f"{scene_id:06d}"
        depth_dir = scene_dir / "depth"
        depth_dir.mkdir(parents=True)
        scene_gt = {}
        scene_gt_info = {}
        scene_camera = {}
        for image_id in range(3):
            scene_gt[str(image_id)] = [self._pose(5, image_id), self._pose(5, image_id)]
            scene_gt_info[str(image_id)] = [
                {"visib_fract": 0.1},
                {"visib_fract": 0.09},
            ]
            scene_camera[str(image_id)] = {}
            (depth_dir / f"{image_id:06d}.png").touch()
        _write_json(scene_dir / "scene_gt.json", scene_gt)
        _write_json(scene_dir / "scene_gt_info.json", scene_gt_info)
        _write_json(scene_dir / "scene_camera.json", scene_camera)


if __name__ == "__main__":
    unittest.main()
