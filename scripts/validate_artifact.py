#!/usr/bin/env python3
import json
from pathlib import Path

from fed_learning_cifar_experiment.models.resnet_cnn_model import tiny_resnet18
from fed_learning_cifar_experiment.state.fedmast_metrics_strategy import FEATURE_BANK, SPECTRAL_DRIFT_ALL


ROOT = Path(__file__).resolve().parents[1]
HISTORY = ROOT / "fedmast_history_clean.json"


def main() -> None:
    model = tiny_resnet18(num_classes=10, base_width=8)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_params == 176_402, f"Unexpected model parameter count: {n_params}"
    assert len(FEATURE_BANK) == 22, f"Expected 22 structural features, found {len(FEATURE_BANK)}"
    assert len(SPECTRAL_DRIFT_ALL) == 6, f"Expected 6 spectral features, found {len(SPECTRAL_DRIFT_ALL)}"

    with HISTORY.open("r", encoding="utf-8") as f:
        history = json.load(f)

    assert history.get("version") == "fedmast-history-v1"
    assert history.get("total_rounds") == 50
    assert len(history.get("raw_baseline", {})) == 22
    assert len(history.get("z_baseline", {})) == 22

    meta = history.get("metadata", {})
    assert meta.get("dataset") == "CIFAR-10"
    assert meta.get("model_id") == "TinyResNet18-base_width_8"
    assert str(meta.get("dirichlet_alpha")) == "0.3"
    assert str(meta.get("partition_seed")) == "42"

    print("FedMAST artifact validation passed")
    print(f"  trainable parameters: {n_params}")
    print(f"  structural features: {len(FEATURE_BANK)}")
    print(f"  spectral features: {len(SPECTRAL_DRIFT_ALL)}")
    print(f"  history rounds: {history['total_rounds']}")
    print(f"  history alpha: {meta['dirichlet_alpha']}")


if __name__ == "__main__":
    main()
