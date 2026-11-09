"""Small CPU-only checks for the matched-coordinate control."""
import unittest

import numpy as np

from fed_learning_cifar_experiment.study.counterfactual import coordinate_matched_projection
from fed_learning_cifar_experiment.study.spectral import BANDS, channel_band_projection, frequency_masks


class CounterfactualTests(unittest.TestCase):
    def test_control_energy_and_support(self):
        rng = np.random.default_rng(8)
        delta = rng.normal(size=(3, 2, 3, 3)).astype(np.float32)
        for band in BANDS:
            spec = channel_band_projection(delta, 1, band)
            coord = coordinate_matched_projection(delta, 1, band)
            self.assertAlmostEqual(
                float(np.linalg.norm(spec)), float(np.linalg.norm(coord)), places=5
            )
            self.assertEqual(int(np.count_nonzero(coord[0])), 0)
            self.assertEqual(int(np.count_nonzero(coord[2])), 0)
            self.assertLessEqual(
                int(np.count_nonzero(coord[1])),
                int(frequency_masks(3, 3)[band].sum() * delta.shape[1]),
            )


if __name__ == "__main__":
    unittest.main()
