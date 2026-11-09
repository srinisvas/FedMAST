#!/usr/bin/env python3
"""Plan or execute an isolated FedMAST/SCC observational benchmark matrix.

Dry-run by default. No novel attack is implemented here.
Examples:
  python scripts/run_scc_matrix.py --profile smoke
  python scripts/run_scc_matrix.py --profile smoke --execute
  python scripts/run_scc_matrix.py --profile pilot --aggregations fedavg,fedmast --execute
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ATTACKS = (
    "none", "lga", "neurotoxin", "constrain-and-scale", "dba",
    "bc-layers", "covertlayers", "3dfed",
)
AGGREGATIONS = ("fedavg", "fedmast", "multikrum", "trimmed_mean", "alignins", "flame")
PROFILES = {
    "smoke": {"rounds": 4, "attack_start": 2, "attack_end": 3, "seeds": (11,),
              "attacks": ("none", "lga"), "aggregations": ("fedavg", "fedmast"),
              "capture_rounds": (2, 3, 4)},
    "pilot": {"rounds": 25, "attack_start": 5, "attack_end": 15, "seeds": (11, 29),
              "attacks": ATTACKS, "aggregations": ("fedavg", "fedmast"),
              "capture_rounds": (5, 10, 15, 20, 25)},
    "full": {"rounds": 100, "attack_start": 20, "attack_end": 60, "seeds": (11, 29, 47),
             "attacks": ATTACKS, "aggregations": AGGREGATIONS,
             "capture_rounds": (20, 40, 60, 80, 100)},
}


def _parse_choices(raw: str | None, available: tuple[str, ...], default: tuple[str, ...]):
    if raw is None:
        return default
    values = tuple(x.strip().lower() for x in raw.split(",") if x.strip())
    unknown = set(values) - set(available)
    if unknown:
        raise ValueError(f"Unknown choices: {sorted(unknown)}")
    return values


def make_run_config(attack: str, aggregation: str, seed: int, profile: dict,
                    run_id: str, run_dir: Path) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("Unsafe simulation ID")
    effective_attack = (
        "bc-layers-lp-full" if attack in ("bc-layers", "covertlayers") else attack
    )
    enabled = attack != "none"
    cfg = {
        "simulation-id": run_id,
        "aggregation-method": aggregation,
        "num-server-rounds": profile["rounds"],
        "num-clients": 100,
        "dirichlet-alpha": 0.3,
        "backdoor-attack-mode": "per-round-attack" if enabled else "none",
        "backdoor-attack-type": effective_attack,
        # All attacked runs use 4 malicious clients for DBA comparability.
        "num-malicious-clients": 4 if enabled else 0,
        "num-malicious-clients-per-round": 4 if enabled else 0,
        "attack-start-round": profile["attack_start"],
        "attack-end-round": profile["attack_end"],
        "bc-defense-sim": "fedmast" if attack == "covertlayers" else "multikrum",
        "study-enabled": True,
        "study-run-dir": str(run_dir),
        "study-seed": int(seed),
        "study-capture-rounds": json.dumps(list(profile["capture_rounds"])),
        "study-spectral-summary": True,
    }
    return cfg


def encode_flower_config(config: dict) -> str:
    """Encode Flower's --run-config key=value syntax without shell execution."""
    tokens = []
    for key, value in config.items():
        if isinstance(value, bool):
            encoded = "true" if value else "false"
        elif isinstance(value, (int, float)):
            encoded = str(value)
        else:
            value = str(value)
            if "'" in value or "\n" in value or "\r" in value:
                raise ValueError("Unsupported quote/newline in Flower run-config value")
            encoded = f"'{value}'"
        tokens.append(f"{key}={encoded}")
    return " ".join(tokens)


def build_runs(args):
    profile = dict(PROFILES[args.profile])
    attacks = _parse_choices(args.attacks, ATTACKS, profile["attacks"])
    aggregations = _parse_choices(args.aggregations, AGGREGATIONS, profile["aggregations"])
    seeds = tuple(int(s) for s in args.seeds.split(",")) if args.seeds else profile["seeds"]
    if not seeds:
        raise ValueError("At least one seed required")
    runs = []
    for attack, aggregation, seed in itertools.product(attacks, aggregations, seeds):
        run_id = f"scc_{args.profile}_{attack.replace('-', '_')}_{aggregation}_s{seed}"
        run_dir = (ROOT / args.output / run_id).resolve()
        cfg = make_run_config(attack, aggregation, seed, profile, run_id, run_dir)
        cmd = ["flwr", "run", ".", "--run-config", encode_flower_config(cfg)]
        runs.append({"id": run_id, "attack": attack, "aggregation": aggregation,
                     "seed": seed, "run_dir": str(run_dir), "config": cfg,
                     "command": cmd})
    return runs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", choices=PROFILES, default="smoke")
    p.add_argument("--attacks", help="Comma-separated attack subset")
    p.add_argument("--aggregations", help="Comma-separated aggregator subset")
    p.add_argument("--seeds", help="Comma-separated seed overrides")
    p.add_argument("--output", default="study_runs")
    p.add_argument("--execute", action="store_true", help="Actually run Flower")
    p.add_argument("--retry", action="store_true", help="Rerun previously completed jobs")
    args = p.parse_args()
    runs = build_runs(args)
    print(f"SCC {args.profile}: {len(runs)} runs; execute={args.execute}")
    for job in runs:
        print(f"  {job['id']}")
    if not args.execute:
        print("Dry run only. Add --execute to run experiments.")
        return 0

    failed = 0
    for job in runs:
        run_dir = Path(job["run_dir"])
        run_dir.mkdir(parents=True, exist_ok=True)
        status_path = run_dir / "driver_status.json"
        if status_path.exists() and not args.retry:
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if status.get("returncode") == 0:
                print(f"[skip] {job['id']} (already completed)")
                continue
        status = {**job, "started_at_utc": datetime.now(timezone.utc).isoformat(),
                  "returncode": None}
        status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
        print(f"[run] {job['id']}", flush=True)
        try:
            with (run_dir / "flower.log").open("w", encoding="utf-8") as log:
                result = subprocess.run(
                    job["command"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                    check=False, env={**os.environ, "RAY_DEDUP_LOGS": "0"},
                )
            status["returncode"] = result.returncode
        except OSError as exc:
            status["returncode"] = -1
            status["error"] = str(exc)
        status["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
        if status["returncode"] != 0:
            failed += 1
            print(f"[failed] {job['id']}: see {run_dir / 'flower.log'}")
    print(f"Finished {len(runs)} planned jobs; failures={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
