import sys
import math
import numpy as np
import pandas as pd

EXPECTED_NUM_CLASSES = 10
ENTROPY_MAX = math.log(EXPECTED_NUM_CLASSES)


def check_id_consistency(df: pd.DataFrame) -> bool:
    print("\n[1] Client ID stability across rounds")
    bad = []
    for cid, sub in df.groupby("cid"):
        unique_pids = sub["partition_id"].unique()
        if len(unique_pids) > 1:
            bad.append((cid, list(unique_pids)))
    if bad:
        print(f"  FAIL: {len(bad)} cids map to multiple partition_ids:")
        for cid, pids in bad[:5]:
            print(f"    cid={cid} -> partition_ids={pids}")
        print("  TEMPORAL ANALYSIS WILL BE GARBAGE. Fix client_fn before proceeding.")
        return False
    bad_rev = []
    for pid, sub in df.groupby("partition_id"):
        unique_cids = sub["cid"].unique()
        if len(unique_cids) > 1:
            bad_rev.append((pid, list(unique_cids)))
    if bad_rev:
        print(f"  WARN: {len(bad_rev)} partition_ids appear under multiple cids "
              f"(less critical; partition_id is still a stable join key).")
    print(f"  OK: {df['cid'].nunique()} unique cids, "
          f"{df['partition_id'].nunique()} unique partition_ids, 1:1 mapping.")
    return True


def check_entropy_signal(df: pd.DataFrame) -> bool:
    print("\n[2] per_class_delta_entropy signal")
    e = df["per_class_delta_entropy"].dropna()
    if e.empty:
        print("  FAIL: column is all NaN.")
        return False
    print(f"  range: [{e.min():.4f}, {e.max():.4f}], "
          f"mean: {e.mean():.4f}, std: {e.std():.4f}, "
          f"max possible: {ENTROPY_MAX:.4f}")
    if e.std() < 0.05:
        print(f"  FAIL: std too low ({e.std():.4f}). Feature is degenerate.")
        return False
    if e.max() > ENTROPY_MAX * 0.999 and e.std() < 0.1:
        print(f"  FAIL: pegged near max entropy with low spread.")
        return False
    print(f"  OK: feature has signal (spread > 0.05).")
    return True


def check_cosine_sanity(df: pd.DataFrame) -> bool:
    print("\n[3] Cosine features sanity")
    cos_cols = [c for c in df.columns if "cos" in c.lower()]
    failed = False
    suspicious = []
    for c in cos_cols:
        vals = df[c].dropna()
        if vals.empty:
            continue
        if vals.min() < -1.001 or vals.max() > 1.001:
            print(f"  FAIL: {c} out of [-1, 1]: min={vals.min():.4f}, max={vals.max():.4f}")
            failed = True
        if vals.mean() > 0.95 and vals.std() < 0.05:
            suspicious.append((c, vals.mean(), vals.std()))
    if suspicious:
        print(f"  WARN: {len(suspicious)} cosine features pegged near 1 with low spread:")
        for c, m, s in suspicious[:5]:
            print(f"    {c}: mean={m:.4f}, std={s:.4f}")
        print("  These features may be redundant/uninformative; consider dropping.")
    if failed:
        return False
    print(f"  OK: {len(cos_cols)} cosine features all in [-1, 1].")
    return True


def check_three_population_separation(df: pd.DataFrame) -> bool:
    print("\n[4] Three-population separation (proxy: honest entropy binning)")
    honest = df[df["malicious_flag"] == 0].copy()
    malicious = df[df["malicious_flag"] == 1].copy()
    if honest.empty or malicious.empty:
        print("  SKIP: need both honest and malicious clients in CSV.")
        return True
    median_e = honest["per_class_delta_entropy"].median()
    near_iid = honest[honest["per_class_delta_entropy"] >= median_e]
    extreme_niid = honest[honest["per_class_delta_entropy"] < median_e]

    print(f"  populations: near-IID honest n={len(near_iid)}, "
          f"non-IID honest n={len(extreme_niid)}, malicious n={len(malicious)}")

    candidate_features = [
        "per_class_delta_entropy",
        "classifier_to_backbone_norm_ratio",
        "stage_head_top_sv_ratio",
        "stage_head_kurtosis",
        "stage_head_l2_norm",
        "stage_head_spectral_entropy",
        "total_update_l2_norm",
        "stage_layer4_kurtosis",
        "stage_energy_concentration_top2",
    ]

    print(f"\n  {'feature':<42s} {'near_IID':>10s} {'non_IID':>10s} {'malicious':>10s} {'sep':>6s}")
    print(f"  {'-'*42:<42s} {'-'*10:>10s} {'-'*10:>10s} {'-'*10:>10s} {'-'*6:>6s}")
    discriminative = 0
    for f in candidate_features:
        if f not in df.columns:
            continue
        m1 = near_iid[f].mean()
        m2 = extreme_niid[f].mean()
        m3 = malicious[f].mean()
        all_means = [m1, m2, m3]
        spread = max(all_means) - min(all_means)
        v1 = float(near_iid[f].var(ddof=0)) if len(near_iid) > 0 else 0.0
        v2 = float(extreme_niid[f].var(ddof=0)) if len(extreme_niid) > 0 else 0.0
        v3 = float(malicious[f].var(ddof=0)) if len(malicious) > 0 else 0.0
        pooled_std = np.sqrt((v1 + v2 + v3) / 3 + 1e-12)
        sep = spread / (pooled_std + 1e-12)
        marker = " *" if sep > 2.0 else ""
        print(f"  {f:<42s} {m1:>10.4f} {m2:>10.4f} {m3:>10.4f} {sep:>6.2f}{marker}")
        if sep > 2.0:
            discriminative += 1

    print(f"\n  Features with separation > 2.0 (Cohen's-d-like): {discriminative}")
    if discriminative >= 3:
        print("  OK: enough discriminative features for downstream modeling.")
        return True
    else:
        print("  WARN: few discriminative features. Inspect feature distributions.")
        return False


def main(csv_path: str) -> int:
    print(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path)
    print(f"  {len(df)} rows, {len(df.columns)} columns, "
          f"{df['round'].nunique()} rounds, {df['cid'].nunique()} unique cids")

    results = [
        check_id_consistency(df),
        check_entropy_signal(df),
        check_cosine_sanity(df),
        check_three_population_separation(df),
    ]
    n_pass = sum(1 for r in results if r)
    print(f"\n{'='*60}")
    print(f"Passed {n_pass}/{len(results)} checks.")
    print(f"{'='*60}")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        import glob
        candidates = sorted(glob.glob("per_update_features_*.csv"))
        if len(candidates) == 1:
            path = candidates[0]
        elif len(candidates) > 1:
            print("Multiple feature CSVs found:")
            for i, c in enumerate(candidates):
                print(f"  [{i}] {c}")
            print(f"\nUsage: python {sys.argv[0]} <path>")
            sys.exit(1)
        else:
            print("No per_update_features_*.csv found in current directory.")
            print(f"Usage: python {sys.argv[0]} <path>")
            sys.exit(1)
    sys.exit(main(path))
