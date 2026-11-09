"""Integration test: observational wrappers must not alter aggregation output."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

from fed_learning_cifar_experiment.study.capture import attach_study_capture
from fed_learning_cifar_experiment.task import get_resnet_cnn_model


class FakeStrategy:
    _cid_to_partition = {"client-1": 5}
    _round_malicious_ids_set = {"client-1"}

    def configure_fit(self, server_round, parameters, client_manager):
        return []

    def aggregate_fit(self, server_round, results, failures):
        return results[0][1].parameters, {"original": 1.0}


class CaptureTests(unittest.TestCase):
    def test_snapshot_and_aggregation_identity(self):
        model = get_resnet_cnn_model()
        initial = [v.detach().cpu().numpy().copy() for v in model.state_dict().values()]
        updated = [v.copy() for v in initial]
        updated[0] = updated[0] + np.float32(0.001)
        initial_params = ndarrays_to_parameters(initial)
        updated_params = ndarrays_to_parameters(updated)
        proxy = SimpleNamespace(cid="client-1")
        fit_res = SimpleNamespace(parameters=updated_params, num_examples=17,
                                  metrics={"train_loss": 0.3})
        with tempfile.TemporaryDirectory() as tmp:
            strategy = FakeStrategy()
            attach_study_capture(strategy, tmp, capture_rounds={2})
            strategy.configure_fit(2, initial_params, None)
            out, metrics = strategy.aggregate_fit(2, [(proxy, fit_res)], [])
            self.assertEqual(metrics, {"original": 1.0})
            np.testing.assert_array_equal(parameters_to_ndarrays(out)[0], updated[0])
            root = Path(tmp)
            self.assertTrue((root / "snapshots/round_0002.npz").exists())
            self.assertTrue((root / "snapshots/round_0002.json").exists())
            rows = (root / "per_client_updates.jsonl").read_text().splitlines()
            self.assertEqual(len(rows), 1)
            self.assertEqual(json.loads(rows[0])["malicious_flag"], 1)
            self.assertGreater(json.loads(rows[0])["update_l2"], 0)


if __name__ == "__main__":
    unittest.main()
