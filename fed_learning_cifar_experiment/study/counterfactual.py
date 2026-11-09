"""Offline, paired counterfactual diagnostics of recorded FL model updates.

These are *single-update model interventions*. They do not reconstruct the
counterfactual aggregation decision of a non-linear FL defense.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from fed_learning_cifar_experiment.study.spectral import (
    BANDS, channel_band_projection, frequency_masks, is_spatial_kernel,
    rank_channels_by_energy,
)
from fed_learning_cifar_experiment.task import (
    get_resnet_cnn_model, load_test_data_for_eval, set_weights,
)
from fed_learning_cifar_experiment.utils.backdoor_attack import add_trigger


def load_snapshot(path: str | Path):
    path = Path(path)
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    with np.load(path, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    return meta, data


def get_state(data: dict, prefix: str, n: int) -> list[np.ndarray]:
    return [data[f"{prefix}_{i}"] for i in range(n)]


def coordinate_matched_projection(
    update: np.ndarray, channel: int, band: str
) -> np.ndarray:
    """Top-k raw-coordinate control with matched projected L2 energy.

    Matches the number of DCT degrees of freedom and intervention L2 norm.
    It is not guaranteed to match the clean-accuracy impact.
    """
    arr = np.asarray(update)
    if not is_spatial_kernel(arr):
        raise ValueError("Expected a spatial convolution update")
    reference = channel_band_projection(arr, channel, band)
    energy = float(np.linalg.norm(reference.astype(np.float64)))
    out = np.zeros_like(arr, dtype=np.float64)
    if energy <= 0:
        return out.astype(arr.dtype)
    k = int(frequency_masks(*arr.shape[-2:])[band].sum() * arr.shape[1])
    vec = arr[channel].astype(np.float64).ravel()
    ids = np.argsort(-np.abs(vec), kind="stable")[:k]
    chosen = vec[ids]
    norm = float(np.linalg.norm(chosen))
    if norm > 0:
        target = out[channel].ravel()
        target[ids] = chosen * (energy / norm)
    return out.astype(arr.dtype, copy=False)


def cache_eval_batches(max_samples: int, batch_size: int = 64):
    """Cache the same fixed, clean CIFAR-10 examples for every intervention."""
    if max_samples <= 0:
        raise ValueError("max_samples must be positive")
    loader = load_test_data_for_eval(batch_size=batch_size)
    batches = []
    remaining = max_samples
    for batch in loader:
        imgs, labels = (batch["img"], batch["label"]) if isinstance(batch, dict) else batch
        take = min(remaining, len(labels))
        batches.append((imgs[:take].clone(), labels[:take].clone()))
        remaining -= take
        if remaining <= 0:
            break
    return batches


@torch.no_grad()
def evaluate_paired(model, batches, device, target_label=2):
    """Clean accuracy and targeted ASR on exactly the same fixed subset."""
    model.to(device)
    model.eval()
    clean_correct = clean_total = triggered_success = triggered_total = 0
    for imgs, labels in batches:
        imgs, labels = imgs.to(device), labels.to(device)
        pred = model(imgs).argmax(dim=1)
        clean_correct += int((pred == labels).sum().item())
        clean_total += int(labels.numel())
        mask = labels != target_label
        if mask.any():
            triggered = torch.stack([add_trigger(x) for x in imgs[mask]])
            pred_bd = model(triggered).argmax(dim=1)
            triggered_success += int((pred_bd == target_label).sum().item())
            triggered_total += int(mask.sum().item())
    return {
        "clean_accuracy": clean_correct / clean_total if clean_total else 0.0,
        "asr": triggered_success / triggered_total if triggered_total else 0.0,
        "clean_count": clean_total,
        "triggered_count": triggered_total,
    }


def run_counterfactuals(
    snapshot: str | Path,
    output_csv: str | Path,
    *,
    client_index: int | None = None,
    layer_names: list[str] | None = None,
    max_layers: int = 2,
    max_channels: int = 4,
    selection: str = "energy",
    seed: int = 13,
    max_samples: int = 512,
    device: str | None = None,
):
    """Compare channel-band deletion with a matched coordinate deletion.

    The reference is the received client model or returned aggregate model.
    No gradient steps, new attacks, or modifications to the FL run occur.
    """
    meta, data = load_snapshot(snapshot)
    state_keys = meta["state_keys"]
    n = len(state_keys)
    global_state = get_state(data, "global", n)
    if client_index is None:
        if not meta["aggregate_present"]:
            raise ValueError("Snapshot does not contain an aggregated model")
        source_state = get_state(data, "aggregate", n)
        source_label = "aggregate"
        malicious_flag = None
    else:
        if not 0 <= client_index < len(meta["clients"]):
            raise IndexError("client_index out of range")
        source_state = get_state(data, f"client_{client_index}", n)
        source_label = f"client_{client_index}"
        malicious_flag = meta["clients"][client_index]["malicious_flag"]

    candidate_layers = [
        key for key, raw in zip(state_keys, source_state)
        if key.endswith(".weight") and is_spatial_kernel(raw)
    ]
    if layer_names:
        missing = set(layer_names) - set(candidate_layers)
        if missing:
            raise ValueError(f"Unknown or non-spatial convolution layers: {sorted(missing)}")
        chosen_layers = layer_names
    else:
        if max_layers <= 0:
            raise ValueError("max_layers must be positive")
        ids = np.linspace(0, len(candidate_layers) - 1,
                          min(max_layers, len(candidate_layers)), dtype=int)
        chosen_layers = [candidate_layers[int(i)] for i in ids]

    device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    batches = cache_eval_batches(max_samples)
    model = get_resnet_cnn_model()
    set_weights(model, source_state)
    baseline = evaluate_paired(model, batches, device)
    rng = np.random.default_rng(seed)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "snapshot", "round", "source", "malicious_flag", "layer", "channel",
        "band", "control", "intervention_l2", "clean_accuracy", "asr",
        "delta_clean_accuracy", "delta_asr", "baseline_clean_accuracy",
        "baseline_asr", "clean_count", "triggered_count",
    ]
    count = 0
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for layer in chosen_layers:
            i = state_keys.index(layer)
            delta = (np.asarray(source_state[i], dtype=np.float64) -
                     np.asarray(global_state[i], dtype=np.float64))
            if selection == "energy":
                channels = rank_channels_by_energy(delta, max_channels)
            elif selection == "random":
                channels = rng.choice(delta.shape[0], size=min(max_channels, delta.shape[0]),
                                      replace=False).tolist()
            elif selection == "all":
                channels = list(range(delta.shape[0]))
            else:
                raise ValueError("selection must be energy, random, or all")
            for channel in channels:
                for band in BANDS:
                    for control, projector in (
                        ("spectral", channel_band_projection),
                        ("coordinate_matched", coordinate_matched_projection),
                    ):
                        removal = projector(delta, channel, band)
                        modified = list(source_state)
                        modified[i] = (np.asarray(source_state[i]) - removal).astype(
                            np.asarray(source_state[i]).dtype, copy=False
                        )
                        set_weights(model, modified)
                        metrics = evaluate_paired(model, batches, device)
                        writer.writerow({
                            "snapshot": str(snapshot),
                            "round": meta["round"],
                            "source": source_label,
                            "malicious_flag": malicious_flag,
                            "layer": layer,
                            "channel": channel,
                            "band": band,
                            "control": control,
                            "intervention_l2": float(np.linalg.norm(removal.astype(np.float64))),
                            "clean_accuracy": metrics["clean_accuracy"],
                            "asr": metrics["asr"],
                            "delta_clean_accuracy": baseline["clean_accuracy"] - metrics["clean_accuracy"],
                            "delta_asr": baseline["asr"] - metrics["asr"],
                            "baseline_clean_accuracy": baseline["clean_accuracy"],
                            "baseline_asr": baseline["asr"],
                            "clean_count": metrics["clean_count"],
                            "triggered_count": metrics["triggered_count"],
                        })
                        count += 1
    return {"rows": count, "baseline": baseline, "output": str(output_csv),
            "layers": chosen_layers, "source": source_label}
