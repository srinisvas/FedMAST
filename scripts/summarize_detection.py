#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd


def safe_rate(num: int, den: int) -> float:
    return float(num / den) if den else float("nan")


def summarize(df: pd.DataFrame) -> dict:
    required = {"malicious_flag", "selected_by_aggregator"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    malicious = pd.to_numeric(df["malicious_flag"], errors="raise").astype(int).eq(1)
    accepted = pd.to_numeric(df["selected_by_aggregator"], errors="raise").astype(int).eq(1)
    rejected = ~accepted

    tp = int((malicious & rejected).sum())
    fn = int((malicious & accepted).sum())
    fp = int((~malicious & rejected).sum())
    tn = int((~malicious & accepted).sum())

    return {
        "updates": int(len(df)),
        "TP": tp,
        "FN": fn,
        "FP": fp,
        "TN": tn,
        "precision": safe_rate(tp, tp + fp),
        "recall": safe_rate(tp, tp + fn),
        "FPR": safe_rate(fp, fp + tn),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize FedMAST client-level detection metrics from per-update CSV logs."
    )
    parser.add_argument("csv", nargs="+", type=Path)
    parser.add_argument("--start-round", type=int, default=None)
    parser.add_argument("--end-round", type=int, default=None)
    parser.add_argument("--per-round", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    overall_rows = []
    for path in args.csv:
        df = pd.read_csv(path)
        if (args.start_round is not None or args.end_round is not None) and "round" not in df.columns:
            raise ValueError(f"{path}: round filtering requested but no 'round' column exists")
        if args.start_round is not None:
            df = df[pd.to_numeric(df["round"], errors="raise") >= args.start_round]
        if args.end_round is not None:
            df = df[pd.to_numeric(df["round"], errors="raise") <= args.end_round]

        row = {"file": str(path), **summarize(df)}
        overall_rows.append(row)
        print(f"\n{path}")
        print(pd.DataFrame([row]).to_string(index=False))

        if args.per_round:
            if "round" not in df.columns:
                raise ValueError(f"{path}: --per-round requires a 'round' column")
            rows = [{"round": rnd, **summarize(group)} for rnd, group in df.groupby("round", sort=True)]
            print("\nPer-round")
            print(pd.DataFrame(rows).to_string(index=False))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(overall_rows).to_csv(args.output, index=False)
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
