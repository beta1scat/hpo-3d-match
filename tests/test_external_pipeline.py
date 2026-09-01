import unittest

import numpy as np

from evaluation import PoseRecord
from external_pipeline import _official_result_text


class ExternalResultTests(unittest.TestCase):
    def test_official_result_translation_is_written_in_metres(self):
        pose = PoseRecord(
            translation_mm=np.array((100.0, 200.0, 300.0)),
            rotation=np.eye(3),
            record_id=0,
        )

        result_line = _official_result_text((pose,), (0.75,), 1.25).splitlines()[2]
        values = [float(value) for value in result_line.removeprefix("Result: ").split()]

        self.assertAlmostEqual(values[3], 0.1)
        self.assertAlmostEqual(values[7], 0.2)
        self.assertAlmostEqual(values[11], 0.3)
        self.assertAlmostEqual(values[16], 0.75)


if __name__ == "__main__":
    unittest.main()
