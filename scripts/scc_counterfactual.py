#!/usr/bin/env python3
"""Evaluate paired spectral vs coordinate controls on a saved study snapshot.

Example:
  python scripts/scc_counterfactual.py \
    study_runs/scc_pilot_lga_fedavg_s11/snapshots/round_0010.npz \
    --client-index 0 --max-channels 4 --max-samples 512
"""
import argparse
import json
from pathlib import Path

from fed_learning_cifar_experiment.study.counterfactual import run_counterfactuals


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("snapshot", help="Path to round_NNNN.npz")
    p.add_argument("--client-index", type=int, help="Client index from paired JSON. Omit for aggregate.")
    p.add_argument("--layers", help="Comma-separated exact layer names")
    p.add_argument("--max-layers", type=int, default=2)
    p.add_argument("--max-channels", type=int, default=4)
    p.add_argument("--selection", choices=("energy", "random", "all"), default="energy")
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--max-samples", type=int, default=512)
    p.add_argument("--device")
    p.add_argument("--trigger-type", choices=("auto", "standard", "dba"), default="auto")
    p.add_argument("--output", help="CSV path (default: beside snapshot)")
    args = p.parse_args()
    src = Path(args.snapshot)
    name = f"counterfactual_{src.stem}_{'aggregate' if args.client_index is None else 'client_'+str(args.client_index)}.csv"
    output = args.output or str(src.with_name(name))
    result = run_counterfactuals(
        src, output, client_index=args.client_index,
        layer_names=args.layers.split(",") if args.layers else None,
        max_layers=args.max_layers, max_channels=args.max_channels,
        selection=args.selection, seed=args.seed, max_samples=args.max_samples,
        device=args.device, trigger_type=args.trigger_type,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
