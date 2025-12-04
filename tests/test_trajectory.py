import unittest

import torch

from safemeta_rl.utils.trajectory import sliding_trajectories


class SlidingTrajectoryTest(unittest.TestCase):
    def test_labels_respect_feasible_flags(self) -> None:
        states = [torch.ones(3) * i for i in range(5)]
        flags = [1.0, 1.0, 0.0, 1.0, 1.0]
        traj, labels, indices = sliding_trajectories(states, flags, horizon=2, limit=3)
        self.assertEqual(traj.shape, (3, 6))
        self.assertListEqual(indices, [0, 1, 2])
        self.assertListEqual(labels.tolist(), [1.0, 0.0, 0.0])

    def test_error_when_not_enough_states(self) -> None:
        states = [torch.zeros(3)]
        flags = [1.0]
        with self.assertRaises(ValueError):
            sliding_trajectories(states, flags, horizon=2, limit=1)


if __name__ == "__main__":
    unittest.main()
