#!/usr/bin/env python3
"""Compute descriptive channel x band interaction residuals from counterfactual CSVs.

This is a diagnostic, NOT an attack success comparison and NOT a causal proof.
Use discovery checkpoints for ranking and held-out checkpoints for validation.
"""
import argparse
import csv
import glob
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def interaction_residuals(rows, value_key="delta_asr"):
    """Two-way centered I_cb - mean_c - mean_b + grand_mean.

    The matrix must be complete for the selected channels and all 3 bands.
    """
    cells = {(int(r["channel"]), r["band"]): float(r[value_key]) for r in rows}
    channels = sorted({c for c, _ in cells})
    bands = ("low", "mid", "high")
    if len(channels) < 2 or any((c, b) not in cells for c in channels for b in bands):
        return []
    grand = statistics.mean(cells.values())
    row_mean = {c: statistics.mean(cells[c, b] for b in bands) for c in channels}
    col_mean = {b: statistics.mean(cells[c, b] for c in channels) for b in bands}
    return [
        {"channel": c, "band": b,
         "interaction": cells[c, b] - row_mean[c] - col_mean[b] + grand,
         "effect": cells[c, b]}
        for c in channels for b in bands
    ]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("files", nargs="+", help="Counterfactual CSV paths or glob patterns")
    p.add_argument("--output", default="study_runs/scc_interactions.csv")
    args = p.parse_args()
    paths = sorted(set(str(p) for pattern in args.files for p in glob.glob(pattern)))
    if not paths:
        p.error("No matching counterfactual CSV files")
    groups = defaultdict(list)
    for path in paths:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                key = (path, row["snapshot"], row["source"], row["round"],
                       row["layer"], row["control"])
                groups[key].append(row)
    results = []
    for key, rows in groups.items():
        for metric in ("delta_asr", "delta_clean_accuracy"):
            residuals = interaction_residuals(rows, metric)
            for r in residuals:
                results.append({
                    "csv": key[0], "snapshot": key[1], "source": key[2],
                    "round": key[3], "layer": key[4], "control": key[5],
                    "metric": metric, **r,
                })
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if results:
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)
    else:
        print("No complete channel-by-band matrices; no output written")
        return
    print(f"Wrote {len(results)} descriptive residuals to {out}")
    print("Compare spectral vs coordinate controls on HELD-OUT runs.")
    print("Do not treat channel cells within one run as independent replicates.")


if __name__ == "__main__":
    main()
