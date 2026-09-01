import tempfile
import unittest

from optimizer import create_and_run_study


class OptimizerResumeTests(unittest.TestCase):
    def test_old_objective_version_cannot_resume(self):
        with tempfile.TemporaryDirectory() as storage_dir:
            with self.assertRaisesRegex(
                ValueError, "expected one of.*new study name"
            ):
                create_and_run_study(
                    objective=lambda trial: 0.0,
                    model_name="star",
                    objective_version="strict-association-v1",
                    sampler_name="TPE",
                    pruner_name="Nop",
                    target_total_trials=0,
                    resume=True,
                    storage_dir=storage_dir,
                )

    def test_resume_never_creates_a_missing_study(self):
        with tempfile.TemporaryDirectory() as storage_dir:
            with self.assertRaisesRegex(ValueError, "no study with that exact name"):
                create_and_run_study(
                    objective=lambda trial: 0.0,
                    model_name="star",
                    objective_version="fixed-penalty-baseline",
                    sampler_name="TPE",
                    pruner_name="Nop",
                    target_total_trials=0,
                    resume=True,
                    storage_dir=storage_dir,
                )

    def test_resume_cannot_change_frozen_budget(self):
        with tempfile.TemporaryDirectory() as storage_dir:
            create_and_run_study(
                objective=lambda trial: 0.0,
                model_name="star",
                objective_version="fixed-penalty-baseline",
                sampler_name="TPE",
                pruner_name="Nop",
                target_total_trials=0,
                storage_dir=storage_dir,
            )

            with self.assertRaisesRegex(ValueError, "immutable attr 'budget'"):
                create_and_run_study(
                    objective=lambda trial: 0.0,
                    model_name="star",
                    objective_version="fixed-penalty-baseline",
                    sampler_name="TPE",
                    pruner_name="Nop",
                    target_total_trials=1,
                    resume=True,
                    storage_dir=storage_dir,
                )


if __name__ == "__main__":
    unittest.main()
