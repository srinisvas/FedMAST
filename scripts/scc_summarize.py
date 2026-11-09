#!/usr/bin/env python3
"""Summarize completed SCC Flower runs and paired benign baselines.

No statistical significance is claimed from a single seed. Output contains
per-run metrics and across-seed mean/SD; use held-out runs for conclusions.
"""
import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def read_csv(path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def mean(values):
    finite = [v for v in values if math.isfinite(v)]
    return statistics.mean(finite) if finite else math.nan


def run_summary(folder):
    status_path = folder / "driver_status.json"
    if not status_path.exists():
        return None
    status = json.loads(status_path.read_text(encoding="utf-8"))
    history = read_csv(folder / "per_round_centralized.csv")
    history = [r for r in history if r.get("round", "").isdigit()]
    history.sort(key=lambda r: int(r["round"]))
    if not history:
        return {"run_id": status["id"], "attack": status["attack"],
                "aggregation": status["aggregation"], "seed": status["seed"],
                "status": status.get("returncode"), "rounds_recorded": 0}
    asrs = [safe_float(r["centralized_asr"]) for r in history]
    accs = [safe_float(r["centralized_mta"]) for r in history]
    end = int(status["config"]["attack-end-round"])
    post = [safe_float(r["centralized_asr"]) for r in history if int(r["round"]) > end]
    attack_window = [
        safe_float(r["centralized_asr"]) for r in history
        if int(status["config"]["attack-start-round"]) <= int(r["round"]) <= end
    ]
    return {
        "run_id": status["id"], "attack": status["attack"],
        "aggregation": status["aggregation"], "seed": status["seed"],
        "status": status.get("returncode"),
        "rounds_recorded": len(history),
        "last_round": int(history[-1]["round"]),
        "final_clean_accuracy": accs[-1],
        "final_asr": asrs[-1],
        "mean_clean_accuracy": mean(accs),
        "mean_asr": mean(asrs),
        "attack_window_mean_asr": mean(attack_window),
        "post_attack_mean_asr": mean(post),
        "post_attack_rounds": len(post),
        "max_asr": max(asrs),
    }


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="study_runs")
    args = p.parse_args()
    root = Path(args.root)
    runs = [r for folder in sorted(root.glob("scc_*")) if folder.is_dir()
            if (r := run_summary(folder)) is not None]
    baselines = {
        (r["aggregation"], r["seed"]): r
        for r in runs if r["attack"] == "none" and r["rounds_recorded"] > 0
    }
    for r in runs:
        base = baselines.get((r["aggregation"], r["seed"]))
        r["final_clean_drop_vs_benign"] = (
            base["final_clean_accuracy"] - r["final_clean_accuracy"]
            if base and r["rounds_recorded"] else math.nan
        )
        r["final_asr_excess_vs_benign"] = (
            r["final_asr"] - base["final_asr"]
            if base and r["rounds_recorded"] else math.nan
        )
    write_csv(root / "scc_run_summary.csv", runs)
    groups = defaultdict(list)
    for r in runs:
        if r["rounds_recorded"] > 0 and r["status"] == 0:
            groups[(r["attack"], r["aggregation"])].append(r)
    rows = []
    fields = ("final_clean_accuracy", "final_asr", "attack_window_mean_asr",
              "post_attack_mean_asr", "final_clean_drop_vs_benign",
              "final_asr_excess_vs_benign")
    for (attack, agg), group in sorted(groups.items()):
        row = {"attack": attack, "aggregation": agg, "completed_seeds": len(group)}
        for key in fields:
            values = [safe_float(r.get(key)) for r in group]
            finite = [v for v in values if math.isfinite(v)]
            row[key + "_mean"] = mean(values)
            row[key + "_sd"] = statistics.stdev(finite) if len(finite) >= 2 else math.nan
        rows.append(row)
    write_csv(root / "scc_group_summary.csv", rows)
    print(f"Summarized {len(runs)} runs, {len(groups)} groups")
    print(f"  {root / 'scc_run_summary.csv'}")
    print(f"  {root / 'scc_group_summary.csv'}")


if __name__ == "__main__":
    main()
