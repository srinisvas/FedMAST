"""An additive matrix has zero two-way interaction residual."""
import unittest

from scripts.scc_interactions import interaction_residuals


class InteractionTests(unittest.TestCase):
    def test_additive_matrix_zero(self):
        rows = [
            {"channel": str(c), "band": band, "delta_asr": str(c * 2 + j)}
            for c in (0, 1, 2) for j, band in enumerate(("low", "mid", "high"))
        ]
        residuals = interaction_residuals(rows)
        self.assertEqual(len(residuals), 9)
        self.assertTrue(all(abs(r["interaction"]) < 1e-10 for r in residuals))

    def test_incomplete_matrix_rejected(self):
        self.assertEqual(interaction_residuals([
            {"channel": "0", "band": "low", "delta_asr": "0.1"}
        ]), [])


if __name__ == "__main__":
    unittest.main()
