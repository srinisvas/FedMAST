import sys
import os
import glob
import math
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

META_COLS = {
    "timestamp", "simulation_id", "round", "cid", "partition_id",
    "malicious_flag", "attack_mode", "attack_type", "experiment_attack_type",
    "local_data_size", "local_epochs", "local_lr", "scale_factor",
    "selected_by_aggregator", "krum_score", "krum_rank", "dirichlet_alpha",
    "target_label", "aggregation_method", "num_clients",
    "fedmast_score", "fedmast_d_round", "fedmast_d_squeeze", "fedmast_d_hist",
    "fedmast_d_traj", "fedmast_threshold", "fedmast_dominant_axis",
}

FEDMAST_SCORE_COLS = [
    "fedmast_score", "fedmast_d_round", "fedmast_d_squeeze",
    "fedmast_d_hist", "fedmast_d_traj",
]

FEDMAST_AXES = ["fedmast_d_round", "fedmast_d_squeeze", "fedmast_d_hist", "fedmast_d_traj"]
AXIS_LABELS = {
    "fedmast_d_round": "D_round",
    "fedmast_d_squeeze": "D_squeeze",
    "fedmast_d_hist": "D_hist",
    "fedmast_d_traj": "D_traj",
}


def fmt_pct(n, d):
    return f"{100*n/d:.1f}%" if d > 0 else "N/A"


def fmt_mean_sd(values):
    if len(values) == 0:
        return "N/A"
    return f"{np.mean(values):.4f} ± {np.std(values):.4f}"


def fmt_mean_sd_pct(values):
    if len(values) == 0:
        return "N/A"
    return f"{100*np.mean(values):.1f}% ± {100*np.std(values):.1f}%"


def load_features(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    has_fedmast = "fedmast_score" in df.columns
    if not has_fedmast:
        print(f"  WARNING: {path} missing FedMAST columns — using krum_score as fallback")
        df["fedmast_score"] = df.get("krum_score", 0.0)
        df["fedmast_d_round"] = 0.0
        df["fedmast_d_squeeze"] = 0.0
        df["fedmast_d_hist"] = 0.0
        df["fedmast_d_traj"] = 0.0
        df["fedmast_threshold"] = 0.0
        df["fedmast_dominant_axis"] = "unknown"
    return df


def load_centralized(features_path: str) -> pd.DataFrame:
    d = os.path.dirname(features_path) or "."
    central_path = os.path.join(d, "per_round_centralized.csv")
    if os.path.exists(central_path):
        return pd.read_csv(central_path)
    central_path = os.path.join(os.path.dirname(d) or ".", "per_round_centralized.csv")
    if os.path.exists(central_path):
        return pd.read_csv(central_path)
    return None


def section_a_overview(df: pd.DataFrame, csv_path: str):
    print("\n" + "=" * 80)
    print("A. DATA OVERVIEW")
    print("=" * 80)

    attack_type = df["experiment_attack_type"].iloc[0] if "experiment_attack_type" in df.columns else "unknown"
    agg = df["aggregation_method"].iloc[0] if "aggregation_method" in df.columns else "unknown"
    n_rounds = df["round"].nunique()
    n_updates = len(df)
    n_mal = int(df["malicious_flag"].sum())
    n_hon = n_updates - n_mal
    mal_pids = df[df["malicious_flag"] == 1]["partition_id"].nunique()
    hon_pids = df[df["malicious_flag"] == 0]["partition_id"].nunique()
    clients_per_round = df.groupby("round").size()

    print(f"\n  File:           {os.path.basename(csv_path)}")
    print(f"  Attack:         {attack_type}")
    print(f"  Aggregation:    {agg}")
    print(f"  Rounds:         {n_rounds}")
    print(f"  Updates:        {n_updates} ({n_hon} honest, {n_mal} malicious)")
    print(f"  Mal ratio:      {fmt_pct(n_mal, n_updates)}")
    print(f"  Partitions:     {hon_pids} honest, {mal_pids} malicious")
    print(f"  Clients/round:  {clients_per_round.mean():.1f} ± {clients_per_round.std():.1f}")

    mal_per_round = df[df["malicious_flag"] == 1].groupby("round").size()
    if len(mal_per_round) > 0:
        print(f"  Mal/round:      {mal_per_round.mean():.1f} ± {mal_per_round.std():.1f} "
              f"(min={mal_per_round.min()}, max={mal_per_round.max()})")

    return attack_type


def section_b_mta_asr(cdf: pd.DataFrame, n_rounds: int):
    print("\n" + "=" * 80)
    print("B. MTA / ASR TRAJECTORY")
    print("=" * 80)

    if cdf is None:
        print("\n  per_round_centralized.csv not found — skipping MTA/ASR analysis.")
        return None

    mta_col = "centralized_mta" if "centralized_mta" in cdf.columns else None
    asr_col = "centralized_asr" if "centralized_asr" in cdf.columns else None
    bsr_col = "centralized_bsr" if "centralized_bsr" in cdf.columns else None

    if mta_col is None:
        print("\n  No MTA column found in centralized CSV.")
        return None

    max_rnd = int(cdf["round"].max())
    quarter = max(max_rnd // 4, 1)
    phases = [
        ("Early",  1, quarter),
        ("Mid-1",  quarter + 1, 2 * quarter),
        ("Mid-2",  2 * quarter + 1, 3 * quarter),
        ("Late",   3 * quarter + 1, max_rnd),
    ]

    cols = ["Phase", "Rounds"]
    if mta_col:
        cols += ["MTA (mean±SD)"]
    if asr_col:
        cols += ["ASR (mean±SD)"]
    if bsr_col:
        cols += ["BSR (mean±SD)"]

    print(f"\n  {'Phase':<8s} {'Rounds':<12s}", end="")
    if mta_col:
        print(f" {'MTA (mean±SD)':<22s}", end="")
    if asr_col:
        print(f" {'ASR (mean±SD)':<22s}", end="")
    if bsr_col:
        print(f" {'BSR (mean±SD)':<22s}", end="")
    print()
    print(f"  {'-'*8:<8s} {'-'*12:<12s}", end="")
    for _ in [c for c in [mta_col, asr_col, bsr_col] if c]:
        print(f" {'-'*22:<22s}", end="")
    print()

    results = {}
    for pname, r_start, r_end in phases:
        phase_df = cdf[(cdf["round"] >= r_start) & (cdf["round"] <= r_end)]
        if phase_df.empty:
            continue
        line = f"  {pname:<8s} {r_start:>3d}-{r_end:<6d}  "
        if mta_col:
            vals = phase_df[mta_col].dropna()
            line += f" {fmt_mean_sd(vals):<22s}"
            results[f"mta_{pname.lower()}"] = vals
        if asr_col:
            vals = phase_df[asr_col].dropna()
            line += f" {fmt_mean_sd(vals):<22s}"
            results[f"asr_{pname.lower()}"] = vals
        if bsr_col:
            vals = phase_df[bsr_col].dropna()
            line += f" {fmt_mean_sd(vals):<22s}"
        print(line)

    final = cdf.tail(10)
    print(f"\n  Final 10 rounds:")
    if mta_col:
        print(f"    MTA: {fmt_mean_sd(final[mta_col].dropna())}")
    if asr_col:
        print(f"    ASR: {fmt_mean_sd(final[asr_col].dropna())}")
    if bsr_col:
        print(f"    BSR: {fmt_mean_sd(final[bsr_col].dropna())}")

    print(f"\n  Final round ({int(cdf['round'].max())}):")
    if mta_col:
        print(f"    MTA = {cdf[mta_col].iloc[-1]:.4f}")
    if asr_col:
        print(f"    ASR = {cdf[asr_col].iloc[-1]:.4f}")

    return results


def section_c_defense(df: pd.DataFrame):
    print("\n" + "=" * 80)
    print("C. DEFENSE EFFECTIVENESS")
    print("=" * 80)

    rounds = sorted(df["round"].unique())
    round_metrics = []

    for rnd in rounds:
        rdf = df[df["round"] == rnd]
        mal = rdf["malicious_flag"] == 1
        rejected = rdf["selected_by_aggregator"] == 0

        tp = int((mal & rejected).sum())
        fn = int((mal & ~rejected).sum())
        fp = int((~mal & rejected).sum())
        tn = int((~mal & ~rejected).sum())
        n_mal = int(mal.sum())

        prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        round_metrics.append({
            "round": rnd, "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "n_mal": n_mal, "precision": prec, "recall": rec, "f1": f1,
        })

    rdf_all = pd.DataFrame(round_metrics)

    rdf_active = rdf_all[rdf_all["n_mal"] > 0]

    if rdf_active.empty:
        print("\n  No rounds with malicious clients found.")
        return rdf_all

    total_tp = rdf_active["tp"].sum()
    total_fn = rdf_active["fn"].sum()
    total_fp = rdf_active["fp"].sum()
    total_tn = rdf_active["tn"].sum()
    total_mal = rdf_active["n_mal"].sum()
    total_decisions = total_tp + total_fn + total_fp + total_tn

    agg_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 1.0
    agg_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 1.0
    agg_f1 = 2 * agg_prec * agg_rec / (agg_prec + agg_rec) if (agg_prec + agg_rec) > 0 else 0.0

    print(f"\n  Aggregate (rounds with attackers: {len(rdf_active)}/{len(rdf_all)}):")
    print(f"    TP={total_tp}  FN={total_fn}  FP={total_fp}  TN={total_tn}")
    print(f"    Precision:  {agg_prec:.4f}")
    print(f"    Recall:     {agg_rec:.4f}")
    print(f"    F1:         {agg_f1:.4f}")
    print(f"    Mal caught: {total_tp}/{total_mal} ({fmt_pct(total_tp, total_mal)})")
    print(f"    FP rate:    {fmt_pct(total_fp, total_fp + total_tn)}")

    print(f"\n  Per-round statistics (mean ± SD across {len(rdf_active)} active rounds):")
    print(f"    Precision:  {fmt_mean_sd(rdf_active['precision'])}")
    print(f"    Recall:     {fmt_mean_sd(rdf_active['recall'])}")
    print(f"    F1:         {fmt_mean_sd(rdf_active['f1'])}")

    perfect = rdf_active[(rdf_active["tp"] == rdf_active["n_mal"]) & (rdf_active["fp"] == 0)]
    print(f"\n    Perfect rounds (all mal caught, 0 FP): {len(perfect)}/{len(rdf_active)} "
          f"({fmt_pct(len(perfect), len(rdf_active))})")

    zero_catch = rdf_active[rdf_active["tp"] == 0]
    print(f"    Zero-catch rounds (all mal accepted): {len(zero_catch)}/{len(rdf_active)} "
          f"({fmt_pct(len(zero_catch), len(rdf_active))})")

    max_rnd = int(rdf_all["round"].max())
    half = max_rnd // 2
    early = rdf_active[rdf_active["round"] <= half]
    late = rdf_active[rdf_active["round"] > half]

    if len(early) > 0 and len(late) > 0:
        print(f"\n  Phase comparison:")
        print(f"    Early (1-{half})   Prec={fmt_mean_sd(early['precision'])}  "
              f"Rec={fmt_mean_sd(early['recall'])}")
        print(f"    Late  ({half+1}-{max_rnd})  Prec={fmt_mean_sd(late['precision'])}  "
              f"Rec={fmt_mean_sd(late['recall'])}")

    return rdf_all


def section_d_axis_analysis(df: pd.DataFrame):
    print("\n" + "=" * 80)
    print("D. FedMAST SCORING AXIS ANALYSIS")
    print("=" * 80)

    if "fedmast_dominant_axis" not in df.columns:
        print("\n  No FedMAST axis data available.")
        return

    rejected_mal = df[(df["malicious_flag"] == 1) & (df["selected_by_aggregator"] == 0)]
    accepted_mal = df[(df["malicious_flag"] == 1) & (df["selected_by_aggregator"] == 1)]

    if len(rejected_mal) > 0:
        print(f"\n  Dominant axis for REJECTED malicious clients ({len(rejected_mal)} updates):")
        axis_counts = rejected_mal["fedmast_dominant_axis"].value_counts()
        for axis, count in axis_counts.items():
            print(f"    {axis:<15s}: {count:>5d} ({fmt_pct(count, len(rejected_mal))})")

    mal = df[df["malicious_flag"] == 1]
    hon = df[df["malicious_flag"] == 0]

    print(f"\n  Mean FedMAST scores by class:")
    print(f"    {'Axis':<15s} {'Honest':>12s} {'Malicious':>12s} {'Separation':>12s}")
    print(f"    {'-'*15:<15s} {'-'*12:>12s} {'-'*12:>12s} {'-'*12:>12s}")
    for col in FEDMAST_AXES:
        if col not in df.columns:
            continue
        h_mean = hon[col].mean()
        m_mean = mal[col].mean()
        sep = m_mean - h_mean
        label = AXIS_LABELS.get(col, col)
        marker = " <<<" if sep > 2.0 else (" <<" if sep > 1.0 else (" <" if sep > 0.5 else ""))
        print(f"    {label:<15s} {h_mean:>12.3f} {m_mean:>12.3f} {sep:>+12.3f}{marker}")

    if "fedmast_score" in df.columns:
        h_score = hon["fedmast_score"].mean()
        m_score = mal["fedmast_score"].mean()
        print(f"    {'Combined':.<15s} {h_score:>12.3f} {m_score:>12.3f} {m_score-h_score:>+12.3f}")

    if "fedmast_threshold" in df.columns:
        thresholds = df.groupby("round")["fedmast_threshold"].first()
        print(f"\n  Threshold statistics across rounds:")
        print(f"    Mean ± SD:  {fmt_mean_sd(thresholds)}")
        print(f"    Range:      [{thresholds.min():.3f}, {thresholds.max():.3f}]")

    if "fedmast_score" in df.columns and len(rejected_mal) > 0:
        print(f"\n  Per-axis contribution to detection (mean score, rejected mal vs accepted hon):")
        for col in FEDMAST_AXES:
            if col not in df.columns:
                continue
            rej_m = rejected_mal[col].mean()
            acc_h = hon[hon["selected_by_aggregator"] == 1][col].mean() if "selected_by_aggregator" in df.columns else hon[col].mean()
            ratio = rej_m / max(acc_h, 1e-6)
            label = AXIS_LABELS.get(col, col)
            print(f"    {label:<15s}: rej_mal={rej_m:.3f}  acc_hon={acc_h:.3f}  ratio={ratio:.1f}x")


def section_e_score_distributions(df: pd.DataFrame):
    print("\n" + "=" * 80)
    print("E. SCORE DISTRIBUTIONS")
    print("=" * 80)

    if "fedmast_score" not in df.columns:
        print("\n  No FedMAST score data available.")
        return

    mal = df[df["malicious_flag"] == 1]
    hon = df[df["malicious_flag"] == 0]

    print(f"\n  Combined FedMAST score distribution:")
    print(f"    Honest:    {fmt_mean_sd(hon['fedmast_score'])}  "
          f"[med={hon['fedmast_score'].median():.3f}]")
    print(f"    Malicious: {fmt_mean_sd(mal['fedmast_score'])}  "
          f"[med={mal['fedmast_score'].median():.3f}]")

    h_95 = hon["fedmast_score"].quantile(0.95)
    mal_below_h95 = (mal["fedmast_score"] <= h_95).sum()
    print(f"\n    Honest 95th percentile: {h_95:.3f}")
    print(f"    Malicious below honest 95th: {mal_below_h95}/{len(mal)} "
          f"({fmt_pct(mal_below_h95, len(mal))}) — these are hard to catch")

    print(f"\n  Score percentiles:")
    print(f"    {'Pctile':<8s} {'Honest':>10s} {'Malicious':>10s}")
    for pct in [25, 50, 75, 90, 95, 99]:
        h_p = hon["fedmast_score"].quantile(pct / 100)
        m_p = mal["fedmast_score"].quantile(pct / 100)
        print(f"    {pct:>4d}th   {h_p:>10.3f} {m_p:>10.3f}")


DEFENSE_SQUEEZE_PAIRS = [
    ("displacement × late-layer energy",
     "l2_distance_to_reference", "stage_layer4_l2_norm"),
    ("alignment × spectral concentration",
     "cosine_to_round_mean_update", "stage_head_spectral_entropy"),
    ("head energy × head spectral rank",
     "stage_head_norm_ratio_to_total", "stage_head_top_sv_ratio"),
    ("cross-layer × distributional shape",
     "classifier_to_backbone_norm_ratio", "backbone_kurtosis_max"),
]


def section_f_squeeze_pairs(df: pd.DataFrame):
    print("\n" + "=" * 80)
    print("F. SQUEEZE PAIR EFFECTIVENESS")
    print("=" * 80)

    if "fedmast_d_squeeze" not in df.columns:
        print("\n  No squeeze data available.")
        return

    mal = df[df["malicious_flag"] == 1]
    hon = df[df["malicious_flag"] == 0]

    print(f"\n  D_squeeze separation:")
    print(f"    Honest:    {fmt_mean_sd(hon['fedmast_d_squeeze'])}")
    print(f"    Malicious: {fmt_mean_sd(mal['fedmast_d_squeeze'])}")

    feat_cols = [c for c in df.columns if c not in META_COLS and df[c].dtype in ('float64', 'float32')]

    for pair_name, fa, fb in DEFENSE_SQUEEZE_PAIRS:
        if fa not in feat_cols or fb not in feat_cols:
            continue

        print(f"\n  {pair_name}:")
        print(f"    {fa} × {fb}")

        h_xy = hon[[fa, fb]].dropna()
        m_xy = mal[[fa, fb]].dropna()

        if len(h_xy) < 10 or len(m_xy) < 5:
            print(f"    Insufficient data")
            continue

        def _cd(h, m):
            ps = np.sqrt((h.var() + m.var()) / 2 + 1e-12)
            return abs(h.mean() - m.mean()) / (ps + 1e-12)

        d_a = _cd(h_xy[fa], m_xy[fa])
        d_b = _cd(h_xy[fb], m_xy[fb])

        diff = h_xy.mean().values - m_xy.mean().values
        n_h, n_m = len(h_xy), len(m_xy)
        pooled = ((n_h - 1) * h_xy.cov().values + (n_m - 1) * m_xy.cov().values) / (n_h + n_m - 2)
        try:
            inv_c = np.linalg.pinv(pooled + 1e-12 * np.eye(2))
            d_2d = float(np.sqrt(diff @ inv_c @ diff))
        except Exception:
            d_2d = float("nan")

        best_1d = max(d_a, d_b)
        gain = d_2d / (best_1d + 1e-12) if not math.isnan(d_2d) else 0
        verdict = "VALIDATED" if gain > 1.2 else ("weak" if gain > 0.9 else "no signal")

        rho_h = h_xy[fa].corr(h_xy[fb])
        rho_m = m_xy[fa].corr(m_xy[fb])

        print(f"    d({fa[:30]})={d_a:.3f}  d({fb[:30]})={d_b:.3f}")
        print(f"    d_2D={d_2d:.3f}  gain={gain:.2f}x  → {verdict}")
        print(f"    ρ_honest={rho_h:+.3f}  ρ_malicious={rho_m:+.3f}")


def section_g_temporal_evolution(df: pd.DataFrame):
    print("\n" + "=" * 80)
    print("G. TEMPORAL COMPONENT EVOLUTION")
    print("=" * 80)

    if "fedmast_d_hist" not in df.columns or "fedmast_d_traj" not in df.columns:
        print("\n  No temporal data available.")
        return

    rounds = sorted(df["round"].unique())
    max_rnd = max(rounds)

    print(f"\n  When do temporal axes activate? (mean score for malicious clients)")
    print(f"\n  {'Round range':<15s} {'D_hist (mal)':>14s} {'D_traj (mal)':>14s} {'D_round (mal)':>14s}")
    print(f"  {'-'*15:<15s} {'-'*14:>14s} {'-'*14:>14s} {'-'*14:>14s}")

    chunk_size = max(max_rnd // 8, 1)
    for start in range(1, max_rnd + 1, chunk_size):
        end = min(start + chunk_size - 1, max_rnd)
        chunk = df[(df["round"] >= start) & (df["round"] <= end) & (df["malicious_flag"] == 1)]
        if chunk.empty:
            continue
        dh = chunk["fedmast_d_hist"].mean()
        dt = chunk["fedmast_d_traj"].mean()
        dr = chunk["fedmast_d_round"].mean()
        marker = ""
        if dh > 1.0 or dt > 1.0:
            marker = " ← temporal active"
        print(f"  {start:>4d}-{end:<8d}  {dh:>14.3f} {dt:>14.3f} {dr:>14.3f}{marker}")

    mal_rounds = df[df["malicious_flag"] == 1].groupby("round").agg({
        "fedmast_d_hist": "mean", "fedmast_d_round": "mean", "fedmast_d_traj": "mean"
    })
    crossover = mal_rounds[mal_rounds["fedmast_d_hist"] > mal_rounds["fedmast_d_round"]]
    if len(crossover) > 0:
        first_cross = crossover.index.min()
        print(f"\n  D_hist first exceeds D_round at round {first_cross}")
    else:
        print(f"\n  D_hist never exceeds D_round (structural axis dominates)")


def section_h_partition_analysis(df: pd.DataFrame):
    print("\n" + "=" * 80)
    print("H. PER-PARTITION ANALYSIS")
    print("=" * 80)

    mal_pids = sorted(df[df["malicious_flag"] == 1]["partition_id"].unique())
    hon_pids = sorted(df[df["malicious_flag"] == 0]["partition_id"].unique())

    if not mal_pids:
        print("\n  No malicious partitions found.")
        return

    print(f"\n  Malicious partitions: {mal_pids}")
    print(f"\n  {'PID':>5s} {'Apps':>5s} {'Rejected':>9s} {'Rec%':>6s} "
          f"{'Score':>12s} {'Dominant':>12s}")
    print(f"  {'-'*5:>5s} {'-'*5:>5s} {'-'*9:>9s} {'-'*6:>6s} "
          f"{'-'*12:>12s} {'-'*12:>12s}")

    for pid in mal_pids:
        pdata = df[(df["partition_id"] == pid) & (df["malicious_flag"] == 1)]
        n_apps = len(pdata)
        n_rej = int((pdata["selected_by_aggregator"] == 0).sum())
        rec = n_rej / n_apps if n_apps > 0 else 0
        mean_score = pdata["fedmast_score"].mean() if "fedmast_score" in pdata.columns else 0

        dom = "N/A"
        if "fedmast_dominant_axis" in pdata.columns:
            dom = pdata["fedmast_dominant_axis"].mode().iloc[0] if len(pdata) > 0 else "N/A"

        print(f"  {pid:>5d} {n_apps:>5d} {n_rej:>5d}/{n_apps:<3d} "
              f"{100*rec:>5.1f}% {mean_score:>12.3f} {dom:>12s}")

    hon_rejected = df[(df["malicious_flag"] == 0) & (df["selected_by_aggregator"] == 0)]
    if len(hon_rejected) > 0:
        print(f"\n  Benign false positives: {len(hon_rejected)} updates from "
              f"{hon_rejected['partition_id'].nunique()} partitions")
        fp_pids = hon_rejected["partition_id"].value_counts().head(5)
        print(f"  Top FP partitions:")
        for pid, count in fp_pids.items():
            total = len(df[(df["partition_id"] == pid) & (df["malicious_flag"] == 0)])
            print(f"    PID {pid}: {count}/{total} rejected ({fmt_pct(count, total)})")
    else:
        print(f"\n  Benign false positives: 0 (perfect specificity)")


def section_z_paper_summary(df: pd.DataFrame, cdf: pd.DataFrame, attack_type: str, rdf_all: pd.DataFrame):
    print("\n" + "=" * 80)
    print("Z. PAPER SUMMARY TABLE")
    print("=" * 80)
    print("  (Copy-paste ready for LaTeX / results section)\n")

    n_rounds = df["round"].nunique()

    final_mta, final_asr = "N/A", "N/A"
    mta_late, asr_late = "N/A", "N/A"
    if cdf is not None:
        max_rnd = int(cdf["round"].max())
        late_start = max(1, max_rnd - max_rnd // 4)
        late = cdf[cdf["round"] >= late_start]
        if "centralized_mta" in cdf.columns:
            final_mta = f"{cdf['centralized_mta'].iloc[-1]:.4f}"
            mta_late = fmt_mean_sd(late["centralized_mta"].dropna())
        if "centralized_asr" in cdf.columns:
            final_asr = f"{cdf['centralized_asr'].iloc[-1]:.4f}"
            asr_late = fmt_mean_sd(late["centralized_asr"].dropna())

    rdf_active = rdf_all[rdf_all["n_mal"] > 0] if rdf_all is not None else pd.DataFrame()

    total_tp = int(rdf_active["tp"].sum()) if len(rdf_active) > 0 else 0
    total_fn = int(rdf_active["fn"].sum()) if len(rdf_active) > 0 else 0
    total_fp = int(rdf_active["fp"].sum()) if len(rdf_active) > 0 else 0
    total_tn = int(rdf_active["tn"].sum()) if len(rdf_active) > 0 else 0

    prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    fpr = total_fp / (total_fp + total_tn) if (total_fp + total_tn) > 0 else 0

    prec_str = fmt_mean_sd(rdf_active["precision"]) if len(rdf_active) > 0 else "N/A"
    rec_str = fmt_mean_sd(rdf_active["recall"]) if len(rdf_active) > 0 else "N/A"
    f1_str = fmt_mean_sd(rdf_active["f1"]) if len(rdf_active) > 0 else "N/A"

    rej_mal = df[(df["malicious_flag"] == 1) & (df["selected_by_aggregator"] == 0)]
    dominant = "N/A"
    if "fedmast_dominant_axis" in rej_mal.columns and len(rej_mal) > 0:
        dominant = rej_mal["fedmast_dominant_axis"].mode().iloc[0]

    print(f"  Attack:              {attack_type}")
    print(f"  Defense:             FedMAST")
    print(f"  Rounds:              {n_rounds}")
    print(f"  ─────────────────────────────────────────")
    print(f"  MTA (final):         {final_mta}")
    print(f"  MTA (late, mean±SD): {mta_late}")
    print(f"  ASR (final):         {final_asr}")
    print(f"  ASR (late, mean±SD): {asr_late}")
    print(f"  ─────────────────────────────────────────")
    print(f"  Precision (mean±SD): {prec_str}")
    print(f"  Recall (mean±SD):    {rec_str}")
    print(f"  F1 (mean±SD):        {f1_str}")
    print(f"  FPR:                 {100*fpr:.2f}%")
    print(f"  TP / (TP+FN):        {total_tp}/{total_tp+total_fn}")
    print(f"  ─────────────────────────────────────────")
    print(f"  Dominant axis:       {dominant}")

    print(f"\n  LaTeX table row:")
    mta_f = f"{cdf['centralized_mta'].iloc[-1]*100:.1f}" if cdf is not None and "centralized_mta" in cdf.columns else "-"
    asr_f = f"{cdf['centralized_asr'].iloc[-1]*100:.1f}" if cdf is not None and "centralized_asr" in cdf.columns else "-"
    print(f"  {attack_type} & {mta_f}\\% & {asr_f}\\% & "
          f"{prec:.2f} & {rec:.2f} & {f1:.2f} & {100*fpr:.1f}\\% & {dominant} \\\\")


def main(csv_path: str):
    df = load_features(csv_path)
    cdf = load_centralized(csv_path)

    attack_type = section_a_overview(df, csv_path)
    mta_asr = section_b_mta_asr(cdf, df["round"].nunique())
    rdf_all = section_c_defense(df)
    section_d_axis_analysis(df)
    section_e_score_distributions(df)
    section_f_squeeze_pairs(df)
    section_g_temporal_evolution(df)
    section_h_partition_analysis(df)
    section_z_paper_summary(df, cdf, attack_type, rdf_all)

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if os.path.isdir(arg):
            candidates = sorted(glob.glob(os.path.join(arg, "per_update_features_fedmast_*.csv")))
            if not candidates:
                candidates = sorted(glob.glob(os.path.join(arg, "per_update_features_*.csv")))
            if len(candidates) == 1:
                main(candidates[0])
            elif len(candidates) > 1:
                print(f"Found {len(candidates)} CSVs — running all:\n")
                for c in candidates:
                    print(f"\n{'#'*80}")
                    print(f"# {os.path.basename(c)}")
                    print(f"{'#'*80}")
                    main(c)
            else:
                print(f"No per_update_features*.csv in {arg}")
                sys.exit(1)
        else:
            main(arg)
    else:
        candidates = sorted(glob.glob("per_update_features_fedmast_*.csv"))
        if not candidates:
            candidates = sorted(glob.glob("per_update_features_*.csv"))
        if len(candidates) == 1:
            main(candidates[0])
        elif len(candidates) > 1:
            print("Multiple CSVs found:")
            for i, c in enumerate(candidates):
                print(f"  [{i}] {c}")
            print(f"\nUsage: python {sys.argv[0]} <path_or_directory>")
            sys.exit(1)
        else:
            print(f"No per_update_features*.csv found.")
            print(f"Usage: python {sys.argv[0]} <path_or_directory>")
            sys.exit(1)
