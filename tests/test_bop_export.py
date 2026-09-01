import csv
import tempfile
import unittest
from pathlib import Path

from bop_export import (
    BOP19_FIELDS,
    BOPExportError,
    IMAGE_TIME_TOLERANCE_SEC,
    write_bop19_csv,
)


def prediction(object_name, score, runtime):
    return {
        "record_type": "prediction",
        "scene_id": 3,
        "im_id": 7,
        "object_name": object_name,
        "score": score,
        "pose": {
            "R": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            "t": [10.0, 20.0, 30.0],
        },
        "time": runtime,
    }


class BOPExportTests(unittest.TestCase):
    def test_csv_has_exact_bop19_fields_and_accepts_runtime_within_tolerance(self):
        first_runtime = 0.25
        predictions = (
            prediction("star", 0.9, first_runtime),
            prediction(
                "screw_black",
                0.8,
                first_runtime + IMAGE_TIME_TOLERANCE_SEC / 2.0,
            ),
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "results.csv"
            write_bop19_csv(predictions, output_path)
            with output_path.open("r", encoding="utf-8", newline="") as stream:
                reader = csv.DictReader(stream)
                rows = list(reader)
                fieldnames = reader.fieldnames

        self.assertEqual(tuple(fieldnames), BOP19_FIELDS)
        self.assertEqual(len(rows), 2)
        self.assertEqual(set(rows[0]), set(BOP19_FIELDS))
        self.assertEqual({int(row["obj_id"]) for row in rows}, {24, 25})
        self.assertTrue(all(float(row["time"]) == first_runtime for row in rows))
        self.assertTrue(all(len(row["R"].split()) == 9 for row in rows))
        self.assertTrue(all(len(row["t"].split()) == 3 for row in rows))

    def test_runtime_difference_above_tolerance_is_rejected(self):
        predictions = (
            prediction("star", 0.9, 0.25),
            prediction("screw_black", 0.8, 0.25 + 2.0 * IMAGE_TIME_TOLERANCE_SEC),
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "results.csv"
            with self.assertRaisesRegex(BOPExportError, "time must agree within"):
                write_bop19_csv(predictions, output_path)


if __name__ == "__main__":
    unittest.main()
