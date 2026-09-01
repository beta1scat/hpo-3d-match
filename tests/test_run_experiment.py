import unittest
from pathlib import Path
from config import SAMPLER_NAMES, PRUNER_NAMES, SEARCH_SPACE, DEFAULT_PARAMS
from run_experiment import ROOT

class RunExperimentModuleTests(unittest.TestCase):
    def test_search_space_continuous(self):
        for name in ["RelSamplingDistance", "KeyPointFraction", "min_score", "max_overlap_dist_rel", "pose_ref_dist_threshold_rel"]:
            self.assertIn(name, SEARCH_SPACE)
            self.assertEqual(SEARCH_SPACE[name]["type"], "float")
            self.assertNotIn("step", SEARCH_SPACE[name])

    def test_samplers_and_pruners(self):
        self.assertEqual(SAMPLER_NAMES, ("TPE", "CmaEs", "Random"))
        self.assertEqual(PRUNER_NAMES, ("Nop", "Median", "Hyperband"))

    def test_default_params_match_search_space(self):
        for k, v in DEFAULT_PARAMS.items():
            self.assertIn(k, SEARCH_SPACE)
            spec = SEARCH_SPACE[k]
            if spec["type"] == "float":
                self.assertGreaterEqual(v, spec["low"])
                self.assertLessEqual(v, spec["high"])
            elif spec["type"] == "int":
                self.assertGreaterEqual(v, spec["low"])
                self.assertLessEqual(v, spec["high"])
            elif spec["type"] == "categorical":
                self.assertIn(v, spec["choices"])

if __name__ == "__main__":
    unittest.main()
