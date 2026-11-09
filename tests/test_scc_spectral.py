"""Unit tests for spatial-frequency decomposition. Run: python -m unittest discover -s tests"""
import unittest

import numpy as np

from fed_learning_cifar_experiment.study.spectral import (
    BANDS,
    channel_band_energy,
    channel_band_projection,
    frequency_masks,
    rank_channels_by_energy,
    summarize_spatial_bands,
)


class SpectralTests(unittest.TestCase):
    def test_bands_partition_3x3(self):
        masks = frequency_masks(3, 3)
        self.assertEqual([int(masks[k].sum()) for k in BANDS], [3, 3, 3])
        np.testing.assert_array_equal(sum(masks.values()), np.ones((3, 3)))

    def test_projection_reconstructs_original(self):
        rng = np.random.default_rng(17)
        delta = rng.normal(size=(4, 3, 3, 3)).astype(np.float32)
        pieces = [channel_band_projection(delta, 2, b) for b in BANDS]
        reconstructed = np.sum(pieces, axis=0)
        np.testing.assert_allclose(reconstructed[2], delta[2], atol=1e-6)
        np.testing.assert_allclose(reconstructed[[0, 1, 3]], 0, atol=1e-7)

    def test_parseval_energy(self):
        rng = np.random.default_rng(19)
        delta = rng.normal(size=(2, 3, 3, 3)).astype(np.float32)
        energies = channel_band_energy(delta, 0)
        self.assertAlmostEqual(sum(energies.values()), float(np.sum(delta[0] ** 2)), places=5)

    def test_skip_one_by_one_and_dense(self):
        rows = summarize_spatial_bands({
            "conv.weight": np.ones((2, 2, 3, 3)),
            "skip.weight": np.ones((2, 2, 1, 1)),
            "fc.weight": np.ones((2, 2)),
        })
        self.assertEqual(len(rows), 3)
        self.assertEqual({r["layer"] for r in rows}, {"conv.weight"})

    def test_channel_order_ranking(self):
        x = np.zeros((3, 1, 3, 3), dtype=np.float32)
        x[2] = 3
        x[1] = 2
        self.assertEqual(rank_channels_by_energy(x, 2), [2, 1])


if __name__ == "__main__":
    unittest.main()
