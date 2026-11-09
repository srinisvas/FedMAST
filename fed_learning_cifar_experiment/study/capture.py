"""Opt-in, read-only instrumentation of the existing Flower strategies.

No changes to client updates, attacker selection, defense decisions, or
aggregation. All snapshots are saved on the server for offline analysis.
"""
from __future__ import annotations

import json
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from flwr.common import parameters_to_ndarrays

from fed_learning_cifar_experiment.task import get_resnet_cnn_model
from fed_learning_cifar_experiment.study.spectral import summarize_spatial_bands


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return str(value)


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_jsonable(record), sort_keys=True, allow_nan=False) + "\n")


def write_run_metadata(output_dir: str, run_config: dict) -> None:
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    # Never persist possible credentials from run_config.
    cfg = {
        str(k): ("[REDACTED]" if any(s in str(k).lower() for s in
                 ("secret", "password", "token", "credential", "api-key"))
                 else _jsonable(v))
        for k, v in dict(run_config).items()
    }
    (dest / "study_run.json").write_text(
        json.dumps({
            "schema_version": 1,
            "started_at_utc": _now(),
            "run_config": cfg,
            "snapshot_format": "npz + json",
            "note": "Observational snapshots, not proof of causal mechanism",
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def attach_study_capture(
    strategy,
    output_dir: str,
    *,
    capture_rounds: set[int],
    spectral_summary: bool = True,
) -> None:
    """Wrap the two Flower methods on a strategy instance.

    Captures *received* client models before defense filtering and the
    *returned* aggregated model afterwards. For non-linear aggregation,
    individual client deltas are not additive contributions to the server
    model. Do not interpret the saved client deltas that way.
    """
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    model = get_resnet_cnn_model()
    state_keys = list(model.state_dict().keys())
    trainable_keys = set(dict(model.named_parameters()).keys())
    original_configure_fit = strategy.configure_fit
    original_aggregate_fit = strategy.aggregate_fit
    incoming_global = {"parameters": None}

    def observed_configure_fit(server_round, parameters, client_manager):
        incoming_global["parameters"] = parameters
        return original_configure_fit(server_round, parameters, client_manager)

    def observed_aggregate_fit(server_round, results, failures):
        # Snapshot inputs before the defense processes them.
        current = incoming_global["parameters"]
        pre = None
        clients = []
        try:
            if current is not None:
                pre = parameters_to_ndarrays(current)
                if len(pre) != len(state_keys):
                    raise ValueError("Global state_dict length does not match model")
            for proxy, fit_res in results:
                nds = parameters_to_ndarrays(fit_res.parameters)
                if len(nds) != len(state_keys):
                    raise ValueError("Client state_dict length does not match model")
                clients.append((str(proxy.cid), fit_res, nds))
        except Exception as exc:
            print(f"[SCC][round {server_round}] input capture failed: {exc}")
            traceback.print_exc()
            pre, clients = None, []

        # The original aggregation must run exactly once and unmodified.
        outcome = original_aggregate_fit(server_round, results, failures)

        try:
            if pre is None:
                return outcome
            aggregated = outcome[0] if outcome else None
            post = parameters_to_ndarrays(aggregated) if aggregated is not None else None
            if post is not None and len(post) != len(state_keys):
                raise ValueError("Aggregated state_dict length does not match model")

            client_metadata = []
            for j, (cid, fit_res, nds) in enumerate(clients):
                pids = getattr(strategy, "_cid_to_partition", {})
                malicious = getattr(strategy, "_round_malicious_ids_set", set())
                deltas = {
                    k: (np.asarray(v, dtype=np.float64) - np.asarray(g, dtype=np.float64))
                    for k, v, g in zip(state_keys, nds, pre)
                    if k in trainable_keys
                }
                total_sq = sum(float(np.sum(d * d)) for d in deltas.values())
                changed = sum(int(np.count_nonzero(d)) for d in deltas.values())
                metadata = {
                    "round": int(server_round),
                    "client_index": j,
                    "cid": cid,
                    "partition_id": int(pids.get(cid, -1)),
                    "malicious_flag": int(cid in malicious),
                    "num_examples": int(fit_res.num_examples),
                    "update_l2": float(np.sqrt(total_sq)),
                    "changed_parameters": changed,
                    "client_metrics": dict(getattr(fit_res, "metrics", {}) or {}),
                }
                client_metadata.append(metadata)
                append_jsonl(root / "per_client_updates.jsonl", metadata)
                if spectral_summary:
                    for row in summarize_spatial_bands(deltas):
                        append_jsonl(root / "per_layer_spectral.jsonl", {
                            "round": int(server_round),
                            "client_index": j,
                            "cid": cid,
                            "malicious_flag": metadata["malicious_flag"],
                            **row,
                        })

            aggregated_sq = None
            if post is not None:
                aggregated_sq = sum(
                    float(np.sum((np.asarray(a, dtype=np.float64) -
                                  np.asarray(g, dtype=np.float64)) ** 2))
                    for k, a, g in zip(state_keys, post, pre)
                    if k in trainable_keys
                )
            append_jsonl(root / "per_round_study.jsonl", {
                "round": int(server_round),
                "received_clients": len(clients),
                "failures": len(failures),
                "aggregate_update_l2": (
                    float(np.sqrt(aggregated_sq)) if aggregated_sq is not None else None
                ),
                "aggregation_returned_parameters": post is not None,
            })

            if int(server_round) in capture_rounds:
                snap = root / "snapshots"
                snap.mkdir(exist_ok=True)
                prefix = f"round_{int(server_round):04d}"
                arrays = {f"global_{i}": v for i, v in enumerate(pre)}
                for j, (_, _, nds) in enumerate(clients):
                    arrays.update({f"client_{j}_{i}": v for i, v in enumerate(nds)})
                if post is not None:
                    arrays.update({f"aggregate_{i}": v for i, v in enumerate(post)})
                np.savez_compressed(snap / f"{prefix}.npz", **arrays)
                (snap / f"{prefix}.json").write_text(
                    json.dumps(_jsonable({
                        "schema_version": 1,
                        "round": int(server_round),
                        "state_keys": state_keys,
                        "trainable_keys": sorted(trainable_keys),
                        "clients": client_metadata,
                        "aggregate_present": post is not None,
                        "warning": (
                            "Client snapshots are pre-defense. For non-linear "
                            "aggregators their deltas are not server contributions."
                        ),
                    }), indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8",
                )
                print(f"[SCC] Captured round {server_round}: {len(clients)} clients")
        except Exception as exc:
            # Instrumentation must not change an experiment's behavior.
            print(f"[SCC][round {server_round}] capture failed: {exc}")
            traceback.print_exc()
        return outcome

    strategy.configure_fit = observed_configure_fit
    strategy.aggregate_fit = observed_aggregate_fit
