import unittest

from dataset import split_bop_groups


class DatasetSplitTests(unittest.TestCase):
    def test_split_is_deterministic_across_repeated_and_reordered_input(self):
        groups = [(scene_id, image_id) for scene_id in range(4) for image_id in range(5)]

        first = split_bop_groups(groups, seed=1729)
        second = split_bop_groups(list(reversed(groups)) + groups[:4], seed=1729)

        self.assertEqual(first, second)
        assigned_groups = [group for split in first.values() for group in split]
        self.assertEqual(len(assigned_groups), len(groups))
        self.assertEqual(set(assigned_groups), set(groups))
        self.assertEqual(len(assigned_groups), len(set(assigned_groups)))


if __name__ == "__main__":
    unittest.main()
