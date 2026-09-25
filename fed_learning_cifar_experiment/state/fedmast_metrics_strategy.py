import json
import math
import os
import traceback
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import flwr as fl
from flwr.common import Parameters, parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server.client_proxy import ClientProxy

import torch

from fed_learning_cifar_experiment.utils.logger import append_per_update_features
from fed_learning_cifar_experiment.utils.metrics_extractor import (
    ParamRegistry,
    nds_to_trainable_state_dict,
    state_dict_delta,
    flat_param_vec_to_per_key_dict,
    mean_delta_per_key,
    extract_per_update_features,
    build_leave_one_out_means,
    build_stage_projection_bases,
    ref_deltas_flat_to_per_key_list,
)
from fed_learning_cifar_experiment.task import get_resnet_cnn_model, set_weights

from fed_learning_cifar_experiment.state.server_strategy import SaveFedAvgMetricsStrategy


FEATURE_BANK = OrderedDict([
    ("total_update_l2_norm",        "magnitude"),
    ("l2_distance_to_round_mean",   "magnitude"),
    ("l2_distance_to_reference",    "magnitude"),
    ("cosine_to_round_mean_update", "magnitude"),

    ("cosine_to_mean_leave_one_out",        "alignment"),
    ("cosine_to_reference_clean_delta",     "alignment"),
    ("stage_norm_signature_cos_to_round_mean", "alignment"),
    ("head_sign_agreement_with_ref",        "alignment"),

    ("stage_head_l2_norm",                  "layer_energy"),
    ("stage_layer4_l2_norm",                "layer_energy"),
    ("stage_head_norm_ratio_to_total",      "layer_energy"),
    ("classifier_to_backbone_norm_ratio",   "layer_energy"),
    ("head_backbone_conservation_product",  "layer_energy"),

    ("stage_head_spectral_entropy",     "spectral"),
    ("stage_head_top_sv_ratio",         "spectral"),
    ("stage_layer4_skewness",           "spectral"),
    ("stage_layer3_kurtosis",           "spectral"),
    ("backbone_kurtosis_max",           "spectral"),

    ("stage_layer4_cos_to_round_mean",  "cross_layer"),
    ("stage_head_cos_to_mean_loo",      "cross_layer"),
    ("per_class_delta_entropy",         "cross_layer"),
    ("stage_layer4_linf_to_l2_ratio",   "cross_layer"),
])

FAMILIES = sorted(set(FEATURE_BANK.values()))
BANK_FEATURES = list(FEATURE_BANK.keys())
N_BANK = len(BANK_FEATURES)

PAPER_RENAME = {
    "l2_distance_to_reference":         "distance_to_baseline_update",
    "cosine_to_reference_clean_delta":  "cosine_to_historical_baseline",
    "head_sign_agreement_with_ref":     "head_sign_agreement_with_baseline",
    "stage_head_cos_to_reference":      "head_cos_to_baseline",
}


SQUEEZE_PAIRS = [
    {
        "name": "displacement × late-layer energy",
        "feat_a": "l2_distance_to_reference",
        "feat_b": "stage_layer4_l2_norm",
    },
    {
        "name": "alignment × spectral concentration",
        "feat_a": "cosine_to_round_mean_update",
        "feat_b": "stage_head_spectral_entropy",
    },
    {
        "name": "head energy ratio × head spectral rank",
        "feat_a": "stage_head_norm_ratio_to_total",
        "feat_b": "stage_head_top_sv_ratio",
    },
    {
        "name": "cross-layer ratio × distributional shape",
        "feat_a": "classifier_to_backbone_norm_ratio",
        "feat_b": "backbone_kurtosis_max",
    },
]


TEMPORAL_TRACKED_FEATURES = [
    "classifier_to_backbone_norm_ratio",
    "stage_head_norm_ratio_to_total",
    "stage_head_top_sv_ratio",
    "stage_layer4_linf_to_l2_ratio",
    "stage_layer4_skewness",
]


SPECTRAL_DRIFT_TSV_FEATURES = [
    "stage_layer2_top_sv_ratio",
    "stage_layer3_top_sv_ratio",
    "stage_layer4_top_sv_ratio",
]

SPECTRAL_DRIFT_ENT_FEATURES = [
    "stage_layer2_spectral_entropy",
    "stage_layer3_spectral_entropy",
    "stage_layer4_spectral_entropy",
]

SPECTRAL_DRIFT_ALL = SPECTRAL_DRIFT_TSV_FEATURES + SPECTRAL_DRIFT_ENT_FEATURES


def _robust_zscore(values: np.ndarray) -> Tuple[np.ndarray, float, float]:
    med = float(np.median(values))
    mad_raw = float(np.median(np.abs(values - med)))
    scaled_mad = 1.4826 * mad_raw
    if scaled_mad < 1e-12:
        return np.zeros_like(values), med, 0.0
    return (values - med) / scaled_mad, med, scaled_mad


def _top2_mean(arr: np.ndarray) -> float:
    if len(arr) == 0:
        return 0.0
    if len(arr) == 1:
        return float(np.abs(arr[0]))
    top2 = np.sort(np.abs(arr))[-2:]
    return float(np.mean(top2))


class RollingAcceptedBuffer:

    def __init__(self, max_rounds: int = 20):
        self.max_rounds = max_rounds
        self._buffer: List[Dict] = []

    def add_round(self, rnd: int, accepted_z_vectors: List[Dict[str, float]]):
        for zv in accepted_z_vectors:
            entry = {"_round": rnd}
            entry.update(zv)
            self._buffer.append(entry)
        cutoff = rnd - self.max_rounds
        self._buffer = [b for b in self._buffer if b["_round"] > cutoff]

    def compute_baseline_stats(self) -> Optional[Dict[str, Tuple[float, float]]]:
        if len(self._buffer) < 10:
            return None
        stats = {}
        for feat in BANK_FEATURES:
            vals = [b.get(feat, np.nan) for b in self._buffer]
            vals = np.array([v for v in vals if np.isfinite(v)])
            if len(vals) < 5:
                continue
            med = float(np.median(vals))
            mad = 1.4826 * float(np.median(np.abs(vals - med)))
            if mad >= 0.1:
                stats[feat] = (med, mad)
        return stats if len(stats) >= N_BANK // 2 else None

    @property
    def size(self) -> int:
        return len(self._buffer)


class RawAcceptedBuffer:

    def __init__(self, max_rounds: int = 20):
        self.max_rounds = max_rounds
        self._buffer: List[Dict] = []

    def add_round(self, rnd: int, raw_vectors: List[Dict[str, float]]):
        for rv in raw_vectors:
            entry = {"_round": rnd}
            entry.update(rv)
            self._buffer.append(entry)
        cutoff = rnd - self.max_rounds
        self._buffer = [b for b in self._buffer if b["_round"] > cutoff]

    def compute_baseline_stats(self) -> Optional[Dict[str, Tuple[float, float]]]:
        if len(self._buffer) < 10:
            return None
        stats = {}
        for feat in BANK_FEATURES:
            vals = [b.get(feat, np.nan) for b in self._buffer]
            vals = np.array([v for v in vals if np.isfinite(v)])
            if len(vals) < 5:
                continue
            med = float(np.median(vals))
            mad = 1.4826 * float(np.median(np.abs(vals - med)))
            if mad >= 1e-9:
                stats[feat] = (med, mad)
        return stats if len(stats) >= N_BANK // 3 else None

    @property
    def size(self) -> int:
        return len(self._buffer)


class ClientTrajectoryTracker:

    def __init__(self, max_window: int = 15):
        self.max_window = max_window
        self._trajectories: Dict[int, List[Dict]] = {}

    def update(self, partition_id: int, rnd: int, z_vector: Dict[str, float]):
        if partition_id not in self._trajectories:
            self._trajectories[partition_id] = []
        entry = {"_round": rnd}
        entry.update(z_vector)
        self._trajectories[partition_id].append(entry)
        self._trajectories[partition_id] = (
            self._trajectories[partition_id][-self.max_window:]
        )

    def compute_trajectory_features(
        self,
        partition_id: int,
        current_z: Dict[str, float],
    ) -> Dict[str, float]:
        history = self._trajectories.get(partition_id, [])
        if len(history) < 3:
            return {}

        temporal = {}
        for feat in TEMPORAL_TRACKED_FEATURES:
            vals = [h.get(feat, np.nan) for h in history]
            vals = [v for v in vals if np.isfinite(v)]
            if len(vals) < 2:
                continue
            arr = np.array(vals, dtype=np.float64)
            cur = current_z.get(feat, np.nan)
            if not np.isfinite(cur):
                continue

            alpha = 0.3
            ema = arr[0]
            for v in arr[1:]:
                ema = alpha * v + (1.0 - alpha) * ema
            temporal[f"{feat}__ema_drift"] = abs(cur - ema)
            temporal[f"{feat}__volatility"] = float(np.std(np.append(arr, cur)))

            if len(arr) >= 3:
                x = np.arange(len(arr) + 1, dtype=float)
                full = np.append(arr, cur)
                slope = float(np.polyfit(x, full, 1)[0])
                temporal[f"{feat}__slope"] = abs(slope)

        if len(history) >= 2:
            deltas = []
            for j in range(1, len(history)):
                d = np.array([
                    history[j].get(f, 0) - history[j - 1].get(f, 0)
                    for f in TEMPORAL_TRACKED_FEATURES
                ])
                deltas.append(d)
            last_to_current = np.array([
                current_z.get(f, 0) - history[-1].get(f, 0)
                for f in TEMPORAL_TRACKED_FEATURES
            ])
            deltas.append(last_to_current)

            if len(deltas) >= 2:
                cosines = []
                for j in range(1, len(deltas)):
                    n1 = np.linalg.norm(deltas[j - 1])
                    n2 = np.linalg.norm(deltas[j])
                    if n1 > 1e-12 and n2 > 1e-12:
                        cosines.append(
                            float(np.dot(deltas[j - 1], deltas[j]) / (n1 * n2))
                        )
                if cosines:
                    temporal["direction_consistency"] = float(np.mean(cosines))

        return temporal

    def n_appearances(self, partition_id: int) -> int:
        return len(self._trajectories.get(partition_id, []))


class ScoreMomentumTracker:

    def __init__(self, decay: float = 0.7, min_appearances: int = 5):
        self.decay = decay
        self.min_appearances = min_appearances
        self._momentum: Dict[int, float] = {}
        self._appearances: Dict[int, int] = {}

    def update(self, partition_id: int, score_percentile: float):
        if partition_id not in self._momentum:
            self._momentum[partition_id] = score_percentile
            self._appearances[partition_id] = 1
        else:
            self._momentum[partition_id] = (
                self.decay * self._momentum[partition_id]
                + (1.0 - self.decay) * score_percentile
            )
            self._appearances[partition_id] += 1

    def get_momentum(self, partition_id: int) -> float:
        if self._appearances.get(partition_id, 0) < self.min_appearances:
            return 0.0
        return self._momentum.get(partition_id, 0.0)

    def is_active(self, partition_id: int) -> bool:
        return self._appearances.get(partition_id, 0) >= self.min_appearances


class HistoricalAggregateTracker:

    def __init__(self, decay: float = 0.8, min_rounds: int = 5):
        self.decay = decay
        self.min_rounds = min_rounds
        self._ema_signature: Optional[np.ndarray] = None
        self._n_rounds: int = 0

    def update(self, stage_norms: np.ndarray):
        if self._ema_signature is None:
            self._ema_signature = stage_norms.copy()
        else:
            self._ema_signature = (
                self.decay * self._ema_signature
                + (1.0 - self.decay) * stage_norms
            )
        self._n_rounds += 1

    def is_active(self) -> bool:
        return self._n_rounds >= self.min_rounds

    def deviation(self, stage_norms: np.ndarray) -> float:
        if not self.is_active() or self._ema_signature is None:
            return 0.0
        n1 = np.linalg.norm(self._ema_signature)
        n2 = np.linalg.norm(stage_norms)
        if n1 < 1e-12 or n2 < 1e-12:
            return 0.0
        cos = float(np.dot(self._ema_signature, stage_norms) / (n1 * n2))
        return max(0.0, 1.0 - cos)


class SpectralDriftTracker:

    def __init__(
        self,
        min_appearances: int = 5,
        decay: float = 0.7,
        threshold_percentile: float = 90.0,
        min_threshold: float = 1.5,
    ):
        self.min_appearances = min_appearances
        self.decay = decay
        self.threshold_percentile = threshold_percentile
        self.min_threshold = min_threshold

        self._scores: Dict[int, float] = {}
        self._appearances: Dict[int, int] = {}

    def update(self, partition_id: int, round_directional_z: float):
        if partition_id not in self._scores:
            self._scores[partition_id] = round_directional_z
            self._appearances[partition_id] = 1
        else:
            self._scores[partition_id] = (
                self.decay * self._scores[partition_id]
                + (1.0 - self.decay) * round_directional_z
            )
            self._appearances[partition_id] += 1

    def get_score(self, partition_id: int) -> float:
        return self._scores.get(partition_id, 0.0)

    def is_active(self, partition_id: int) -> bool:
        return self._appearances.get(partition_id, 0) >= self.min_appearances

    def n_appearances(self, partition_id: int) -> int:
        return self._appearances.get(partition_id, 0)

    def compute_threshold(self) -> float:
        active_scores = [
            s for pid, s in self._scores.items()
            if self._appearances.get(pid, 0) >= self.min_appearances
        ]
        if len(active_scores) < 5:
            return float("inf")
        pct_thresh = float(np.percentile(active_scores, self.threshold_percentile))
        return max(pct_thresh, self.min_threshold)

    def n_active(self) -> int:
        return sum(
            1 for pid in self._appearances
            if self._appearances[pid] >= self.min_appearances
        )


def _iterative_threshold(values: np.ndarray, k: float, passes: int = 2) -> float:
    if len(values) < 3:
        return float("inf")

    thresholds = []
    mask = np.ones(len(values), dtype=bool)

    for p in range(passes):
        subset = values[mask]
        if len(subset) < 3:
            break

        med = float(np.median(subset))
        mad_raw = float(np.median(np.abs(subset - med)))
        scaled_mad = 1.4826 * mad_raw

        if scaled_mad > 1e-12:
            thresh = med + k * scaled_mad
        else:
            std = float(np.std(subset))
            if std > 1e-12:
                thresh = med + k * std
            else:
                thresh = float("inf")

        thresholds.append(thresh)
        mask = values <= thresh

    if not thresholds:
        return float("inf")
    if len(thresholds) == 1:
        return thresholds[0]

    initial = thresholds[0]
    refined = min(thresholds)
    floor = 0.5 * initial

    return max(refined, floor)


class FedMASTHistory:

    def __init__(self):
        self.raw_baseline: Optional[Dict[str, Tuple[float, float]]] = None
        self.z_baseline: Optional[Dict[str, Tuple[float, float]]] = None
        self.aggregate_ema: Optional[List[float]] = None
        self.aggregate_n_rounds: int = 0
        self.total_rounds: int = 0
        self.metadata: Dict[str, str] = {}

    def save(self, path: str, raw_buffer, z_buffer, agg_tracker, rnd: int,
             model_id: str = "", dataset: str = ""):
        data = {
            "version": "fedmast-history-v1",
            "total_rounds": rnd,
            "metadata": {
                "model_id": model_id,
                "dataset": dataset,
            },
            "raw_baseline": {},
            "z_baseline": {},
            "aggregate_ema": None,
            "aggregate_n_rounds": 0,
        }

        raw_stats = raw_buffer.compute_baseline_stats()
        if raw_stats:
            data["raw_baseline"] = {
                k: {"median": v[0], "mad": v[1]} for k, v in raw_stats.items()
            }

        z_stats = z_buffer.compute_baseline_stats()
        if z_stats:
            data["z_baseline"] = {
                k: {"median": v[0], "mad": v[1]} for k, v in z_stats.items()
            }

        if agg_tracker.is_active():
            data["aggregate_ema"] = agg_tracker._ema_signature.tolist()
            data["aggregate_n_rounds"] = agg_tracker._n_rounds

        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[FedMAST] History saved: {path} ({rnd} rounds, "
              f"{len(data.get('raw_baseline', {}))} raw features, "
              f"{len(data.get('z_baseline', {}))} z features)")

    @classmethod
    def load(cls, path: str) -> Optional["FedMASTHistory"]:
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r") as f:
                data = json.load(f)
            if data.get("version") != "fedmast-history-v1":
                print(f"[FedMAST] History version mismatch: {data.get('version')}")
                return None

            h = cls()
            h.total_rounds = data.get("total_rounds", 0)
            h.metadata = data.get("metadata", {})

            raw = data.get("raw_baseline", {})
            if raw:
                h.raw_baseline = {
                    k: (v["median"], v["mad"]) for k, v in raw.items()
                }
            z = data.get("z_baseline", {})
            if z:
                h.z_baseline = {
                    k: (v["median"], v["mad"]) for k, v in z.items()
                }
            ema = data.get("aggregate_ema")
            if ema:
                h.aggregate_ema = ema
                h.aggregate_n_rounds = data.get("aggregate_n_rounds", 0)

            print(f"[FedMAST] History loaded: {path} ({h.total_rounds} rounds, "
                  f"{len(h.raw_baseline or {})} raw features)")
            return h
        except Exception as e:
            print(f"[FedMAST] History load failed: {e}")
            return None


def _coordinate_median(nds_list: List[List[np.ndarray]], global_nds) -> List[np.ndarray]:
    agg = []
    for li in range(len(global_nds)):
        base = np.asarray(global_nds[li])
        if not np.issubdtype(base.dtype, np.floating):
            agg.append(base.copy())
            continue
        stacked = np.stack([np.asarray(nds[li]).astype(np.float64) for nds in nds_list])
        agg.append(np.median(stacked, axis=0).astype(base.dtype))
    return agg


def _trimmed_mean(nds_list: List[List[np.ndarray]], global_nds,
                  trim_frac: float = 0.2) -> List[np.ndarray]:
    agg = []
    n = len(nds_list)
    k = max(1, int(n * trim_frac))
    for li in range(len(global_nds)):
        base = np.asarray(global_nds[li])
        if not np.issubdtype(base.dtype, np.floating):
            agg.append(base.copy())
            continue
        stacked = np.stack([np.asarray(nds[li]).astype(np.float64) for nds in nds_list])
        sorted_arr = np.sort(stacked, axis=0)
        if n > 2 * k:
            trimmed = sorted_arr[k:n-k]
        else:
            trimmed = sorted_arr
        agg.append(trimmed.mean(axis=0).astype(base.dtype))
    return agg


class SaveFedMASTMetricsStrategy(SaveFedAvgMetricsStrategy):

    def __init__(
        self,
        *,
        fedmast_threshold_k: float = 3.0,
        fedmast_min_accept: int = 5,
        fedmast_warmup_rounds: int = 2,
        fedmast_buffer_rounds: int = 20,
        fedmast_trajectory_window: int = 15,
        fedmast_verbose: bool = True,
        fedmast_suspicious_policy: str = "coord-median",
        fedmast_history_mode: str = "none",
        fedmast_history_path: str = "fedmast_history.json",
        fedmast_spectral_min_appearances: int = 5,
        fedmast_spectral_decay: float = 0.7,
        fedmast_spectral_percentile: float = 85.0,
        fedmast_spectral_min_threshold: float = 0.5,
        fedmast_ablate_squeeze: bool = False,
        fedmast_ablate_spectral: bool = False,
        fedmast_ablate_history: bool = False,
        fedmast_ablate_round: bool = False,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        self._ablate_squeeze = bool(fedmast_ablate_squeeze)
        self._ablate_spectral = bool(fedmast_ablate_spectral)
        self._ablate_history = bool(fedmast_ablate_history)
        self._ablate_round = bool(fedmast_ablate_round)

        self.fedmast_history_mode = str(fedmast_history_mode)
        self.fedmast_history_path = str(fedmast_history_path)

        if self._ablate_squeeze:
            print("[FedMAST] *** ABLATION: D_squeeze DISABLED ***")
        if self._ablate_spectral:
            print("[FedMAST] *** ABLATION: D_spectral DISABLED ***")
        if self._ablate_history:
            print("[FedMAST] *** ABLATION: D_hist + D_anchor DISABLED "
                  "(history_mode forced to 'none') ***")
            self.fedmast_history_mode = "none"
        if self._ablate_round:
            print("[FedMAST] *** ABLATION: D_round DISABLED ***")

        self.fedmast_threshold_k = float(fedmast_threshold_k)
        self.fedmast_min_accept = int(fedmast_min_accept)
        self.fedmast_warmup_rounds = int(fedmast_warmup_rounds)
        self.fedmast_verbose = bool(fedmast_verbose)

        valid_policies = {"fedavg", "noop", "coord-median", "trimmed-mean"}
        self.fedmast_suspicious_policy = str(fedmast_suspicious_policy)
        if self.fedmast_suspicious_policy not in valid_policies:
            raise ValueError(f"fedmast_suspicious_policy must be one of {valid_policies}")

        self._accepted_buffer = RollingAcceptedBuffer(max_rounds=int(fedmast_buffer_rounds))
        self._raw_buffer = RawAcceptedBuffer(max_rounds=int(fedmast_buffer_rounds))
        self._trajectory_tracker = ClientTrajectoryTracker(max_window=int(fedmast_trajectory_window))
        self._momentum_tracker = ScoreMomentumTracker(decay=0.7, min_appearances=5)
        self._aggregate_tracker = HistoricalAggregateTracker(decay=0.8, min_rounds=5)

        self._spectral_drift_tracker = SpectralDriftTracker(
            min_appearances=int(fedmast_spectral_min_appearances),
            decay=float(fedmast_spectral_decay),
            threshold_percentile=float(fedmast_spectral_percentile),
            min_threshold=float(fedmast_spectral_min_threshold),
        )

        self._frozen_baseline: Optional[FedMASTHistory] = None
        self._history_loaded = False

        if self.fedmast_history_mode in ("load-readonly", "load-update-safe"):
            self._frozen_baseline = FedMASTHistory.load(self.fedmast_history_path)
            if self._frozen_baseline is None:
                raise FileNotFoundError(
                    "FedMAST history was requested but could not be loaded: "
                    f"{self.fedmast_history_path}"
                )
            self._history_loaded = True
            if self._frozen_baseline.aggregate_ema is not None:
                self._aggregate_tracker._ema_signature = np.array(
                    self._frozen_baseline.aggregate_ema
                )
                self._aggregate_tracker._n_rounds = (
                    self._frozen_baseline.aggregate_n_rounds
                )


    def _compute_d_round(
        self,
        z_matrix: Dict[str, Dict[str, float]],
    ) -> Dict[str, Tuple[float, Dict[str, float]]]:
        cids = list(z_matrix.keys())
        results = {}

        for cid in cids:
            group_scores = {}
            for fam in FAMILIES:
                fam_z = [
                    z_matrix[cid].get(feat, 0.0)
                    for feat, f in FEATURE_BANK.items()
                    if f == fam
                ]
                group_scores[fam] = _top2_mean(np.array(fam_z))
            d_round = max(group_scores.values()) if group_scores else 0.0
            results[cid] = (d_round, group_scores)

        return results


    def _compute_d_squeeze(
        self,
        z_matrix: Dict[str, Dict[str, float]],
    ) -> Dict[str, Tuple[float, Dict[str, float]]]:
        cids = list(z_matrix.keys())
        results = {}

        for cid in cids:
            pair_scores = {}
            for sp in SQUEEZE_PAIRS:
                z_a = z_matrix[cid].get(sp["feat_a"], 0.0)
                z_b = z_matrix[cid].get(sp["feat_b"], 0.0)
                pair_scores[sp["name"]] = math.sqrt(z_a ** 2 + z_b ** 2)
            d_squeeze = max(pair_scores.values()) if pair_scores else 0.0
            results[cid] = (d_squeeze, pair_scores)

        return results


    def _compute_d_hist(
        self,
        z_matrix: Dict[str, Dict[str, float]],
    ) -> Dict[str, float]:
        baseline_stats = None
        if self._frozen_baseline is not None and self._frozen_baseline.z_baseline:
            baseline_stats = self._frozen_baseline.z_baseline
        else:
            baseline_stats = self._accepted_buffer.compute_baseline_stats()
        if baseline_stats is None:
            return {cid: 0.0 for cid in z_matrix}

        cids = list(z_matrix.keys())
        results = {}

        for cid in cids:
            hist_z = {}
            for feat, (bl_med, bl_mad) in baseline_stats.items():
                val = z_matrix[cid].get(feat, np.nan)
                if np.isfinite(val) and bl_mad > 1e-12:
                    hist_z[feat] = (val - bl_med) / bl_mad
                else:
                    hist_z[feat] = 0.0

            group_scores = {}
            for fam in FAMILIES:
                fam_z = [
                    hist_z.get(feat, 0.0)
                    for feat, f in FEATURE_BANK.items()
                    if f == fam
                ]
                group_scores[fam] = _top2_mean(np.array(fam_z))
            results[cid] = max(group_scores.values()) if group_scores else 0.0

        return results


    def _compute_d_anchor(
        self,
        raw_features: Dict[str, Dict[str, float]],
    ) -> Dict[str, float]:
        raw_stats = None
        if self._frozen_baseline is not None and self._frozen_baseline.raw_baseline:
            raw_stats = self._frozen_baseline.raw_baseline
        else:
            raw_stats = self._raw_buffer.compute_baseline_stats()
        if raw_stats is None:
            return {cid: 0.0 for cid in raw_features}

        cids = list(raw_features.keys())
        results = {}

        for cid in cids:
            anchor_z: Dict[str, float] = {}
            for feat, (hist_med, hist_mad) in raw_stats.items():
                val = raw_features[cid].get(feat, np.nan)
                if np.isfinite(val) and hist_mad > 1e-12:
                    anchor_z[feat] = (val - hist_med) / hist_mad
                else:
                    anchor_z[feat] = 0.0

            group_scores = {}
            for fam in FAMILIES:
                fam_z = [
                    anchor_z.get(feat, 0.0)
                    for feat, f in FEATURE_BANK.items()
                    if f == fam and feat in anchor_z
                ]
                group_scores[fam] = _top2_mean(np.array(fam_z)) if fam_z else 0.0
            results[cid] = max(group_scores.values()) if group_scores else 0.0

        return results


    def _compute_d_traj(
        self,
        z_matrix: Dict[str, Dict[str, float]],
        partition_ids: Dict[str, int],
    ) -> Dict[str, float]:
        cids = list(z_matrix.keys())

        traj_features: Dict[str, Dict[str, float]] = {}
        for cid in cids:
            pid = partition_ids.get(cid, -1)
            if pid < 0:
                continue
            tf = self._trajectory_tracker.compute_trajectory_features(
                pid, z_matrix[cid]
            )
            if tf:
                traj_features[cid] = tf

        if len(traj_features) < 3:
            return {cid: 0.0 for cid in cids}

        all_tf_names = set()
        for tf in traj_features.values():
            all_tf_names.update(tf.keys())

        if not all_tf_names:
            return {cid: 0.0 for cid in cids}

        traj_cids = list(traj_features.keys())
        per_feat_z: Dict[str, Dict[str, float]] = {cid: {} for cid in traj_cids}

        for tf_name in all_tf_names:
            vals = []
            valid_cids = []
            for cid in traj_cids:
                v = traj_features[cid].get(tf_name, np.nan)
                if np.isfinite(v):
                    vals.append(v)
                    valid_cids.append(cid)

            if len(vals) < 3:
                continue

            z_arr, _, _ = _robust_zscore(np.array(vals))
            for j, cid in enumerate(valid_cids):
                per_feat_z[cid][tf_name] = float(z_arr[j])

        results = {}
        for cid in cids:
            if cid in per_feat_z and per_feat_z[cid]:
                z_vals = np.array(list(per_feat_z[cid].values()))
                results[cid] = _top2_mean(z_vals)
            else:
                results[cid] = 0.0

        return results


    def _compute_d_momentum(
        self,
        partition_ids: Dict[str, int],
    ) -> Dict[str, float]:
        cids = list(partition_ids.keys())

        raw_momentum: Dict[str, float] = {}
        for cid in cids:
            pid = partition_ids.get(cid, -1)
            if pid < 0:
                continue
            if self._momentum_tracker.is_active(pid):
                raw_momentum[cid] = self._momentum_tracker.get_momentum(pid)

        if len(raw_momentum) < 3:
            return {cid: 0.0 for cid in cids}

        mom_cids = list(raw_momentum.keys())
        mom_vals = np.array([raw_momentum[c] for c in mom_cids])
        z_arr, _, _ = _robust_zscore(mom_vals)

        results = {}
        for cid in cids:
            if cid in raw_momentum:
                idx = mom_cids.index(cid)
                results[cid] = max(0.0, float(z_arr[idx]))
            else:
                results[cid] = 0.0

        return results


    def _compute_d_spectral(
        self,
        z_matrix: Dict[str, Dict[str, float]],
        raw_features: Dict[str, Dict[str, float]],
        partition_ids: Dict[str, int],
    ) -> Dict[str, float]:
        cids = list(z_matrix.keys())

        spectral_z: Dict[str, Dict[str, float]] = {cid: {} for cid in cids}

        for feat in SPECTRAL_DRIFT_ALL:
            vals = np.array([
                raw_features[cid].get(feat, np.nan) for cid in cids
            ], dtype=np.float64)
            finite_mask = np.isfinite(vals)
            if finite_mask.sum() < 3:
                continue
            if not finite_mask.all():
                vals[~finite_mask] = np.median(vals[finite_mask])
            z_arr, _, scaled_mad = _robust_zscore(vals)
            if scaled_mad < 1e-12:
                continue
            for i, cid in enumerate(cids):
                spectral_z[cid][feat] = float(z_arr[i])

        for cid in cids:
            pid = partition_ids.get(cid, -1)
            if pid < 0:
                continue

            sz = spectral_z.get(cid, {})
            tsv_z = [sz[f] for f in SPECTRAL_DRIFT_TSV_FEATURES if f in sz]
            ent_z = [sz[f] for f in SPECTRAL_DRIFT_ENT_FEATURES if f in sz]

            if len(tsv_z) < 2 or len(ent_z) < 2:
                continue

            round_dir_z = -np.mean(tsv_z) + np.mean(ent_z)
            self._spectral_drift_tracker.update(pid, round_dir_z)

        results = {}
        for cid in cids:
            pid = partition_ids.get(cid, -1)
            if pid >= 0 and self._spectral_drift_tracker.is_active(pid):
                results[cid] = self._spectral_drift_tracker.get_score(pid)
            else:
                results[cid] = 0.0

        return results


    def _cluster_bisection(
        self,
        raw_features: Dict[str, Dict[str, float]],
    ) -> Optional[Tuple[set, set, float]]:
        raw_stats = None
        if self._frozen_baseline is not None and self._frozen_baseline.raw_baseline:
            raw_stats = self._frozen_baseline.raw_baseline
        else:
            raw_stats = self._raw_buffer.compute_baseline_stats()
        if raw_stats is None:
            return None

        cids = list(raw_features.keys())
        n = len(cids)
        if n < 5:
            return None

        feat_names = sorted(raw_stats.keys())
        if len(feat_names) < 5:
            return None

        X = np.zeros((n, len(feat_names)))
        for j, feat in enumerate(feat_names):
            hist_med, hist_mad = raw_stats[feat]
            safe_mad = max(hist_mad, 1e-12)
            for i, cid in enumerate(cids):
                val = raw_features[cid].get(feat, hist_med)
                X[i, j] = (val - hist_med) / safe_mad

        pairwise_sq = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2)
        i0, i1 = np.unravel_index(np.argmax(pairwise_sq), pairwise_sq.shape)
        centroids = X[[i0, i1]].copy()

        labels = np.zeros(n, dtype=int)
        for _ in range(30):
            d0 = np.sum((X - centroids[0]) ** 2, axis=1)
            d1 = np.sum((X - centroids[1]) ** 2, axis=1)
            new_labels = (d1 < d0).astype(int)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for k_idx in range(2):
                mask = labels == k_idx
                if mask.any():
                    centroids[k_idx] = X[mask].mean(axis=0)

        cluster_0 = {cids[i] for i in range(n) if labels[i] == 0}
        cluster_1 = {cids[i] for i in range(n) if labels[i] == 1}

        if len(cluster_0) < 2 or len(cluster_1) < 2:
            return None

        dist_0 = float(np.linalg.norm(centroids[0]))
        dist_1 = float(np.linalg.norm(centroids[1]))

        min_dist = min(dist_0, dist_1)
        max_dist = max(dist_0, dist_1)
        ratio = max_dist / max(min_dist, 1e-12)

        if ratio < 1.5:
            return None

        if dist_0 <= dist_1:
            return cluster_0, cluster_1, ratio
        else:
            return cluster_1, cluster_0, ratio


    def _score_clients(
        self,
        raw_features: Dict[str, Dict[str, float]],
        partition_ids: Dict[str, int],
    ) -> Tuple[Dict[str, Dict], List[Dict[str, float]], List[Dict[str, float]], bool]:
        cids = list(raw_features.keys())
        n = len(cids)

        _empty = {
            "score": 0.0, "d_round": 0.0, "d_squeeze": 0.0,
            "d_hist": 0.0, "d_anchor": 0.0, "d_traj": 0.0,
            "d_momentum": 0.0, "d_spectral": 0.0,
            "dominant_axis": "n/a", "flagging_axes": [],
            "group_scores": {}, "pair_scores": {},
            "accepted": True, "reason": "too_few_clients",
            "threshold": float("inf"), "per_axis_thresholds": {},
            "spectral_threshold": float("inf"),
            "score_median": 0.0, "score_mad": 0.0, "z_vector": {},
        }
        if n < 3:
            results = {cid: dict(_empty) for cid in cids}
            return results, [{} for _ in cids], [], False

        z_matrix: Dict[str, Dict[str, float]] = {cid: {} for cid in cids}
        for feat in BANK_FEATURES:
            vals = np.array([
                raw_features[cid].get(feat, np.nan) for cid in cids
            ], dtype=np.float64)

            finite_mask = np.isfinite(vals)
            if not finite_mask.any():
                continue
            if not finite_mask.all():
                vals[~finite_mask] = np.median(vals[finite_mask])

            z_arr, _, _ = _robust_zscore(vals)
            for i, cid in enumerate(cids):
                z_matrix[cid][feat] = float(z_arr[i])

        d_round_results = self._compute_d_round(z_matrix)
        if self._ablate_round:
            d_round_results = {cid: (0.0, {}) for cid in cids}
        d_squeeze_results = self._compute_d_squeeze(z_matrix)
        if self._ablate_squeeze:
            d_squeeze_results = {cid: (0.0, {}) for cid in cids}
        d_hist_scores = self._compute_d_hist(z_matrix)
        if self._ablate_history:
            d_hist_scores = {cid: 0.0 for cid in cids}
        d_anchor_scores = self._compute_d_anchor(raw_features)
        if self._ablate_history:
            d_anchor_scores = {cid: 0.0 for cid in cids}

        ANCHOR_CATASTROPHE_THRESHOLD = 100.0
        d_anchor_values = np.array([d_anchor_scores.get(c, 0.0) for c in cids])
        median_d_anchor = float(np.median(d_anchor_values))
        anchor_disabled = median_d_anchor > ANCHOR_CATASTROPHE_THRESHOLD
        if anchor_disabled:
            d_anchor_scores = {cid: 0.0 for cid in cids}
            print(f"[FedMAST] D_anchor DISABLED: round median={median_d_anchor:.1f} "
                  f"> {ANCHOR_CATASTROPHE_THRESHOLD} (catastrophic inflation)")

        d_traj_scores = self._compute_d_traj(z_matrix, partition_ids)
        d_momentum_scores = self._compute_d_momentum(partition_ids)
        if self._ablate_spectral:
            d_spectral_scores = {cid: 0.0 for cid in cids}
        else:
            d_spectral_scores = self._compute_d_spectral(z_matrix, raw_features, partition_ids)

        k = self.fedmast_threshold_k

        hard_axis_values = {
            "D_round":   np.array([d_round_results[c][0] for c in cids]),
            "D_squeeze": np.array([d_squeeze_results[c][0] for c in cids]),
            "D_hist":    np.array([d_hist_scores.get(c, 0.0) for c in cids]),
        }

        ITERATIVE_AXES = {"D_hist"}

        per_axis_thresholds = {}
        per_axis_flagged: Dict[str, set] = {}

        for axis_name, vals in hard_axis_values.items():
            if np.max(np.abs(vals)) < 1e-12:
                per_axis_thresholds[axis_name] = float("inf")
                per_axis_flagged[axis_name] = set()
                continue

            passes = 2 if axis_name in ITERATIVE_AXES else 1
            thresh = _iterative_threshold(vals, k, passes=passes)
            per_axis_thresholds[axis_name] = thresh
            per_axis_flagged[axis_name] = {
                cids[i] for i in range(n) if vals[i] > thresh
            }

        hard_flag_counts = {cid: 0 for cid in cids}
        for flagged in per_axis_flagged.values():
            for cid in flagged:
                hard_flag_counts[cid] += 1

        HARD_CONSENSUS = 2
        hard_flagged = {cid for cid, cnt in hard_flag_counts.items() if cnt >= HARD_CONSENSUS}

        hard_strong = set()
        for cid, cnt in hard_flag_counts.items():
            if cnt < 3:
                continue
            d_r_thresh = per_axis_thresholds.get("D_round", float("inf"))
            d_s_thresh = per_axis_thresholds.get("D_squeeze", float("inf"))
            d_h_thresh = per_axis_thresholds.get("D_hist", float("inf"))
            if (d_round_results[cid][0] > 2.0 * d_r_thresh
                    or d_squeeze_results[cid][0] > 2.0 * d_s_thresh
                    or d_hist_scores.get(cid, 0.0) > 2.0 * d_h_thresh):
                hard_strong.add(cid)

        spectral_threshold = self._spectral_drift_tracker.compute_threshold()
        spectral_flagged = set()

        for cid in cids:
            pid = partition_ids.get(cid, -1)
            if pid < 0:
                continue
            if (self._spectral_drift_tracker.is_active(pid)
                    and d_spectral_scores.get(cid, 0.0) > spectral_threshold):
                spectral_flagged.add(cid)

        per_axis_thresholds["D_spectral"] = spectral_threshold

        soft_axis_values = {
            "D_anchor":   np.array([d_anchor_scores.get(c, 0.0) for c in cids]),
            "D_traj":     np.array([d_traj_scores.get(c, 0.0) for c in cids]),
            "D_momentum": np.array([d_momentum_scores.get(c, 0.0) for c in cids]),
        }
        for axis_name, vals in soft_axis_values.items():
            if np.max(np.abs(vals)) < 1e-12:
                per_axis_thresholds[axis_name] = float("inf")
            else:
                per_axis_thresholds[axis_name] = _iterative_threshold(vals, k, passes=1)

        all_flagged = hard_flagged | spectral_flagged

        combined = {}
        for cid in cids:
            d_r = d_round_results[cid][0]
            d_s = d_squeeze_results[cid][0]
            d_h = d_hist_scores.get(cid, 0.0)
            d_a = d_anchor_scores.get(cid, 0.0)
            d_t = d_traj_scores.get(cid, 0.0)
            d_m = d_momentum_scores.get(cid, 0.0)
            d_sp = d_spectral_scores.get(cid, 0.0)
            combined[cid] = max(d_r, d_s, d_h, d_a, d_t, d_m, d_sp)

        score_arr = np.array([combined[cid] for cid in cids])
        score_med = float(np.median(score_arr))
        score_mad = 1.4826 * float(np.median(np.abs(score_arr - score_med)))

        HARD_MIN = 3

        ranked = sorted(cids, key=lambda c: combined[c])
        accepted = set()
        rejected = set()
        round_suspicious = False
        cluster_mode = False
        spectral_risk = len(spectral_flagged) > 0

        cluster_result = None if anchor_disabled else self._cluster_bisection(raw_features)

        if cluster_result is not None:
            trusted_cluster, suspicious_cluster, cluster_ratio = cluster_result
            cluster_mode = True
            round_suspicious = True

            print(f"[FedMAST] Cluster mode: trusted={len(trusted_cluster)}, "
                  f"suspicious={len(suspicious_cluster)}, ratio={cluster_ratio:.2f}")

            accepted = set(trusted_cluster)
            rejected = set(suspicious_cluster)

            for cid in list(accepted):
                if cid in spectral_flagged or cid in hard_strong:
                    accepted.discard(cid)
                    rejected.add(cid)

            to_reject = [
                cid for cid in list(accepted)
                if cid in hard_flagged and cid not in hard_strong
            ]
            to_reject.sort(key=lambda c: combined[c], reverse=True)
            for cid in to_reject:
                if len(accepted) - 1 >= HARD_MIN:
                    accepted.discard(cid)
                    rejected.add(cid)
        else:
            for cid in ranked:
                if cid not in all_flagged:
                    accepted.add(cid)
                else:
                    rejected.add(cid)

            if len(accepted) < self.fedmast_min_accept:
                round_suspicious = True
                for cid in ranked:
                    if (cid in rejected
                            and cid not in spectral_flagged
                            and cid not in hard_strong
                            and len(accepted) < self.fedmast_min_accept):
                        accepted.add(cid)
                        rejected.discard(cid)

            if len(accepted) < HARD_MIN:
                for cid in ranked:
                    if (cid in rejected
                            and cid not in spectral_flagged
                            and cid not in hard_strong
                            and len(accepted) < HARD_MIN):
                        accepted.add(cid)
                        rejected.discard(cid)

        if not anchor_disabled and len(accepted) >= 3 and len(rejected) >= 2:
            acc_anchor = np.mean([d_anchor_scores.get(c, 0.0) for c in accepted])
            rej_anchor = np.mean([d_anchor_scores.get(c, 0.0) for c in rejected])
            if acc_anchor > rej_anchor * 1.5 and acc_anchor > 10.0:
                round_suspicious = True
                print(f"[FedMAST] INVERSION detected: accepted mean D_anchor "
                      f"({acc_anchor:.1f}) > rejected ({rej_anchor:.1f}). "
                      f"Re-sorting by D_anchor.")
                anchor_ranked = sorted(cids, key=lambda c: d_anchor_scores.get(c, 0.0))
                accepted = set()
                rejected = set()
                for cid in anchor_ranked:
                    anchor_thresh = per_axis_thresholds.get("D_anchor", float("inf"))
                    if (len(accepted) < HARD_MIN
                            or d_anchor_scores.get(cid, 0.0) <= anchor_thresh):
                        if cid in spectral_flagged:
                            rejected.add(cid)
                        else:
                            accepted.add(cid)
                    else:
                        rejected.add(cid)

        STAGE_NORM_KEYS = [
            "stage_stem_l2_norm", "stage_layer1_l2_norm",
            "stage_layer2_l2_norm", "stage_layer3_l2_norm",
            "stage_layer4_l2_norm", "stage_head_l2_norm",
        ]

        if (not cluster_mode
                and not anchor_disabled
                and self._aggregate_tracker.is_active()
                and len(accepted) >= 3):
            acc_sigs = []
            for cid in accepted:
                sig = [raw_features.get(cid, {}).get(k, 0.0) for k in STAGE_NORM_KEYS]
                if all(np.isfinite(v) for v in sig):
                    acc_sigs.append(np.array(sig))

            if acc_sigs:
                agg_sig = np.mean(acc_sigs, axis=0)
                deviation = self._aggregate_tracker.deviation(agg_sig)

                if deviation > 0.3:
                    round_suspicious = True
                    print(f"[FedMAST] Coalition check: accepted aggregate deviates "
                          f"from history (cos_dist={deviation:.3f}). "
                          f"Re-filtering by D_anchor.")

                    anchor_ranked = sorted(cids, key=lambda c: d_anchor_scores.get(c, 0.0))
                    accepted = set()
                    rejected = set()
                    anchor_thresh = per_axis_thresholds.get("D_anchor", float("inf"))
                    for cid in anchor_ranked:
                        if cid in spectral_flagged:
                            rejected.add(cid)
                        elif (len(accepted) < HARD_MIN
                                or d_anchor_scores.get(cid, 0.0) <= anchor_thresh):
                            accepted.add(cid)
                        else:
                            rejected.add(cid)

        if spectral_risk:
            round_suspicious = True

        accepted_z_vectors = []
        accepted_raw_vectors = []
        for cid in cids:
            if cid in accepted and combined[cid] <= score_med:
                accepted_z_vectors.append(z_matrix[cid])
                accepted_raw_vectors.append(raw_features[cid])

        results = {}
        for cid in cids:
            d_r, gs = d_round_results[cid]
            d_s, ps = d_squeeze_results[cid]
            d_h = d_hist_scores.get(cid, 0.0)
            d_a = d_anchor_scores.get(cid, 0.0)
            d_t = d_traj_scores.get(cid, 0.0)
            d_m = d_momentum_scores.get(cid, 0.0)
            d_sp = d_spectral_scores.get(cid, 0.0)

            flagging_axes = []
            if cid in hard_flagged:
                flagging_axes.extend(
                    ax for ax, flagged in per_axis_flagged.items()
                    if cid in flagged
                )
            if cid in spectral_flagged:
                flagging_axes.append("D_spectral")

            axis_scores = {
                "D_round": d_r, "D_squeeze": d_s,
                "D_hist": d_h, "D_anchor": d_a,
                "D_traj": d_t, "D_momentum": d_m,
                "D_spectral": d_sp,
            }
            dominant = max(axis_scores, key=axis_scores.get)

            results[cid] = {
                "score": combined[cid],
                "d_round": d_r, "d_squeeze": d_s,
                "d_hist": d_h, "d_anchor": d_a,
                "d_traj": d_t, "d_momentum": d_m,
                "d_spectral": d_sp,
                "dominant_axis": dominant,
                "flagging_axes": flagging_axes,
                "group_scores": gs, "pair_scores": ps,
                "accepted": cid in accepted,
                "reason": (
                    "accepted" if cid in accepted
                    else (
                        "spectral_flagged" if cid in spectral_flagged
                        else f"hard-consensus ({','.join(flagging_axes)})"
                    )
                ),
                "threshold": min(t for t in per_axis_thresholds.values() if t < 1e6) if per_axis_thresholds else float("inf"),
                "per_axis_thresholds": per_axis_thresholds,
                "spectral_threshold": spectral_threshold,
                "score_median": score_med, "score_mad": score_mad,
                "z_vector": z_matrix.get(cid, {}),
                "round_suspicious": round_suspicious,
                "spectral_risk": spectral_risk,
            }

        return results, accepted_z_vectors, accepted_raw_vectors, round_suspicious


    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, Any]],
        failures,
    ):
        if not results:
            return super().aggregate_fit(rnd, results, failures)

        global_params = self._round_global_parameters
        if global_params is None:
            global_params = self.prev_global_parameters
        if global_params is None:
            print(f"[FedMAST][Round {rnd}] No global params — FedAvg fallback")
            return super().aggregate_fit(rnd, results, failures)

        global_nds = parameters_to_ndarrays(global_params)

        if self._param_registry is None:
            tmp_model = get_resnet_cnn_model()
            self._param_registry = ParamRegistry(tmp_model)
            print(f"[FedMAST] ParamRegistry: {self._param_registry.total_trainable_params} params")
        registry = self._param_registry
        global_sd_train = nds_to_trainable_state_dict(global_nds, registry.state_dict_keys)

        client_ids, client_proxies, client_fit_res, client_nds_list = [], [], [], []
        for cp, fr in results:
            client_ids.append(str(cp.cid))
            client_proxies.append(cp)
            client_fit_res.append(fr)
            client_nds_list.append(parameters_to_ndarrays(fr.parameters))
        n = len(client_ids)

        per_client_deltas: List[Optional[Dict]] = []
        for i in range(n):
            try:
                csd = nds_to_trainable_state_dict(client_nds_list[i], registry.state_dict_keys)
                per_client_deltas.append(state_dict_delta(csd, global_sd_train))
            except Exception as e:
                print(f"[FedMAST][Round {rnd}] Delta failed CID={client_ids[i]}: {e}")
                per_client_deltas.append(None)

        valid_deltas = [d for d in per_client_deltas if d is not None]
        if not valid_deltas:
            return super().aggregate_fit(rnd, results, failures)

        round_mean_sd = mean_delta_per_key(d for d in valid_deltas)
        valid_idx = [i for i, d in enumerate(per_client_deltas) if d is not None]
        loo_means = build_leave_one_out_means([per_client_deltas[i] for i in valid_idx])
        loo_map = {valid_idx[j]: loo_means[j] for j in range(len(valid_idx))}

        ref_delta_pk, ref_deltas_pk_list = None, None
        if self._round_ref_deltas_flat is not None and len(self._round_ref_deltas_flat) > 0:
            try:
                rmf = torch.from_numpy(self._round_ref_deltas_flat.mean(axis=0)).float()
                if rmf.numel() == registry.total_trainable_params:
                    ref_delta_pk = flat_param_vec_to_per_key_dict(rmf, registry)
                    ref_deltas_pk_list = ref_deltas_flat_to_per_key_list(
                        self._round_ref_deltas_flat, registry
                    )
            except Exception:
                pass

        stage_proj_bases, stage_proj_ref_means = None, None
        if ref_deltas_pk_list is not None:
            try:
                stage_proj_bases, stage_proj_ref_means = build_stage_projection_bases(
                    ref_deltas_pk_list, registry, k=3
                )
            except Exception:
                pass

        raw_features: Dict[str, Dict[str, float]] = {}
        partition_ids: Dict[str, int] = {}
        for i in range(n):
            cid = client_ids[i]
            if per_client_deltas[i] is None:
                continue
            pid = self._get_partition_id(client_proxies[i])
            partition_ids[cid] = pid
            try:
                raw_features[cid] = extract_per_update_features(
                    client_delta_per_key=per_client_deltas[i],
                    round_mean_delta_per_key=round_mean_sd,
                    registry=registry,
                    reference_delta_per_key=ref_delta_pk,
                    leave_one_out_mean_per_key=loo_map.get(i),
                    stage_projection_bases=stage_proj_bases,
                    stage_projection_ref_means=stage_proj_ref_means,
                )
            except Exception as e:
                print(f"[FedMAST][Round {rnd}] Feature failed CID={cid}: {e}")

        if rnd == 1 and raw_features:
            sample = next(iter(raw_features.values()))
            missing = [f for f in SPECTRAL_DRIFT_ALL if f not in sample]
            if missing:
                print(f"[FedMAST][WARN] Missing spectral drift features in extractor "
                      f"output — D_spectral will be degraded: {missing}")

        in_warmup = rnd <= self.fedmast_warmup_rounds and not self._history_loaded

        if in_warmup or len(raw_features) < 3:
            defense_results = {
                cid: {
                    "score": 0.0, "d_round": 0.0, "d_squeeze": 0.0,
                    "d_hist": 0.0, "d_anchor": 0.0, "d_traj": 0.0,
                    "d_momentum": 0.0, "d_spectral": 0.0,
                    "dominant_axis": "warmup",
                    "flagging_axes": [],
                    "group_scores": {}, "pair_scores": {},
                    "accepted": True,
                    "reason": "warmup" if in_warmup else "insufficient_data",
                    "threshold": float("inf"),
                    "per_axis_thresholds": {},
                    "spectral_threshold": float("inf"),
                    "score_median": 0.0,
                    "score_mad": 0.0, "z_vector": {},
                    "round_suspicious": False,
                    "spectral_risk": False,
                }
                for cid in client_ids
            }
            accepted_cids = set(client_ids)
            rejected_cids = set()

            if raw_features:
                z_mat = {}
                for feat in BANK_FEATURES:
                    vals = np.array([
                        raw_features[c].get(feat, 0.0) for c in raw_features
                    ], dtype=np.float64)
                    finite = np.isfinite(vals)
                    if not finite.any():
                        continue
                    if not finite.all():
                        vals[~finite] = np.median(vals[finite])
                    z_arr, _, _ = _robust_zscore(vals)
                    for j, c in enumerate(raw_features):
                        z_mat.setdefault(c, {})[feat] = float(z_arr[j])

                for cid in raw_features:
                    pid = partition_ids.get(cid, -1)
                    if pid >= 0:
                        self._trajectory_tracker.update(pid, rnd, z_mat.get(cid, {}))

                self._compute_d_spectral(z_mat, raw_features, partition_ids)
        else:
            defense_results, accepted_z, accepted_raw, round_suspicious = \
                self._score_clients(raw_features, partition_ids)
            accepted_cids = {c for c, r in defense_results.items() if r["accepted"]}
            rejected_cids = {c for c, r in defense_results.items() if not r["accepted"]}

            spectral_risk = any(
                r.get("spectral_risk", False) for r in defense_results.values()
            )

            if not round_suspicious and not spectral_risk:
                self._accepted_buffer.add_round(rnd, accepted_z)
                self._raw_buffer.add_round(rnd, accepted_raw)

                if accepted_raw:
                    stage_norm_keys = [
                        "stage_stem_l2_norm", "stage_layer1_l2_norm",
                        "stage_layer2_l2_norm", "stage_layer3_l2_norm",
                        "stage_layer4_l2_norm", "stage_head_l2_norm",
                    ]
                    sig_vecs = []
                    for rv in accepted_raw:
                        sig = [rv.get(k, 0.0) for k in stage_norm_keys]
                        if all(np.isfinite(v) for v in sig):
                            sig_vecs.append(np.array(sig))
                    if sig_vecs:
                        agg_sig = np.mean(sig_vecs, axis=0)
                        self._aggregate_tracker.update(agg_sig)
            else:
                reason = []
                if round_suspicious:
                    reason.append("suspicious")
                if spectral_risk:
                    reason.append("spectral_risk")
                print(f"[FedMAST][Round {rnd}] Buffer QUARANTINE — {'+'.join(reason)}")

            for cid in raw_features:
                pid = partition_ids.get(cid, -1)
                if pid >= 0 and cid in defense_results:
                    zv = defense_results[cid].get("z_vector", {})
                    self._trajectory_tracker.update(pid, rnd, zv)

            scored_cids = [c for c in raw_features if c in defense_results]
            if len(scored_cids) >= 3:
                scores = np.array([defense_results[c]["score"] for c in scored_cids])
                for c in scored_cids:
                    pid = partition_ids.get(c, -1)
                    if pid >= 0:
                        pct = float(np.mean(scores <= defense_results[c]["score"]))
                        self._momentum_tracker.update(pid, pct)

            for cid in client_ids:
                if cid not in defense_results:
                    defense_results[cid] = {
                        "score": float("inf"), "d_round": 0.0, "d_squeeze": 0.0,
                        "d_hist": 0.0, "d_anchor": 0.0,
                        "d_traj": 0.0, "d_momentum": 0.0, "d_spectral": 0.0,
                        "dominant_axis": "no_features",
                        "flagging_axes": ["no_features"],
                        "group_scores": {}, "pair_scores": {},
                        "accepted": False, "reason": "feature_extraction_failed",
                        "threshold": 0.0, "per_axis_thresholds": {},
                        "spectral_threshold": 0.0,
                        "score_median": 0.0,
                        "score_mad": 0.0, "z_vector": {},
                        "round_suspicious": round_suspicious,
                        "spectral_risk": spectral_risk,
                    }
                    rejected_cids.add(cid)
                    accepted_cids.discard(cid)

        round_suspicious = any(
            defense_results.get(c, {}).get("round_suspicious", False)
            for c in client_ids
        ) if defense_results else False

        accepted_idx = [i for i, c in enumerate(client_ids) if c in accepted_cids]
        below_safety_floor = len(accepted_idx) < 3
        if below_safety_floor:
            round_suspicious = True
            print(
                f"[FedMAST][Round {rnd}] Accepted set below safety floor "
                f"({len(accepted_idx)} < 3) — retaining global model"
            )

        use_policy = (
            "noop"
            if below_safety_floor
            else (self.fedmast_suspicious_policy if round_suspicious else "fedavg")
        )

        if use_policy == "noop" and round_suspicious:
            print(f"[FedMAST][Round {rnd}] Suspicious round — NOOP (no aggregation)")
            aggregated_params = self._round_global_parameters
            aggregated_metrics = {}

        elif use_policy == "coord-median" and round_suspicious:
            print(f"[FedMAST][Round {rnd}] Suspicious round — coordinate MEDIAN aggregation")
            acc_nds = [client_nds_list[i] for i in accepted_idx]
            if len(acc_nds) >= 3:
                med_nds = _coordinate_median(acc_nds, global_nds)
                aggregated_params = ndarrays_to_parameters(med_nds)
            else:
                aggregated_params = self._round_global_parameters
            aggregated_metrics = {}

        elif use_policy == "trimmed-mean" and round_suspicious:
            print(f"[FedMAST][Round {rnd}] Suspicious round — TRIMMED MEAN aggregation")
            acc_nds = [client_nds_list[i] for i in accepted_idx]
            if len(acc_nds) >= 5:
                tm_nds = _trimmed_mean(acc_nds, global_nds, trim_frac=0.2)
                aggregated_params = ndarrays_to_parameters(tm_nds)
            else:
                aggregated_params = self._round_global_parameters
            aggregated_metrics = {}

        else:
            total_ex = sum(getattr(client_fit_res[i], "num_examples", 1) for i in accepted_idx)
            total_ex = max(total_ex, 1)

            agg_nds = []
            for li in range(len(global_nds)):
                base = np.asarray(global_nds[li])
                if not np.issubdtype(base.dtype, np.floating):
                    agg_nds.append(base.copy())
                    continue
                acc = np.zeros(base.shape, dtype=np.float64)
                for i in accepted_idx:
                    w = getattr(client_fit_res[i], "num_examples", 1) / total_ex
                    acc += w * np.asarray(client_nds_list[i][li]).astype(np.float64)
                agg_nds.append(acc.astype(base.dtype))

            aggregated_params = ndarrays_to_parameters(agg_nds)
            aggregated_metrics = {}

        if self._round_attack_type == "dba" and abs(self.dba_server_eta - 1.0) > 1e-9:
            cand_nds = parameters_to_ndarrays(aggregated_params)
            upd = []
            for old, cand in zip(global_nds, cand_nds):
                oa, ca = np.asarray(old), np.asarray(cand)
                if np.issubdtype(oa.dtype, np.floating):
                    diff = ca.astype(np.float64) - oa.astype(np.float64)
                    upd.append((oa.astype(np.float64) + self.dba_server_eta * diff).astype(oa.dtype))
                else:
                    upd.append(ca.copy())
            aggregated_params = ndarrays_to_parameters(upd)

        if hasattr(self, '_3dfed_coordinator') and self._3dfed_coordinator is not None:
            ind_jsons, idx_lists = [], []
            for _, fr in results:
                m = dict(getattr(fr, "metrics", {}) or {})
                raw = m.get("3dfed_indicator_records")
                if raw:
                    ind_jsons.append(raw)
                raw_idx = m.get("3dfed_indicator_indices")
                if raw_idx:
                    try:
                        parsed = json.loads(raw_idx) if isinstance(raw_idx, str) else list(raw_idx)
                        if parsed:
                            idx_lists.append(parsed)
                    except Exception:
                        pass
            if not self._3dfed_coordinator.indicator_indices and idx_lists:
                self._3dfed_coordinator.indicator_indices = idx_lists[0]
            if ind_jsons:
                self._3dfed_coordinator.receive_indicator_records(ind_jsons)
                try:
                    prev_m = get_resnet_cnn_model()
                    set_weights(prev_m, parameters_to_ndarrays(self._round_global_parameters))
                    cur_m = get_resnet_cnn_model()
                    set_weights(cur_m, parameters_to_ndarrays(aggregated_params))
                    t_keys = [n for n, p in cur_m.named_parameters() if p.requires_grad]
                    self._3dfed_coordinator.read_and_adapt(
                        current_global_sd={k: v.clone() for k, v in cur_m.state_dict().items()},
                        previous_global_sd={k: v.clone() for k, v in prev_m.state_dict().items()},
                        trainable_keys=t_keys, server_round=rnd,
                    )
                except Exception as e:
                    print(f"[FedMAST][3DFed] Feedback failed: {e}")

        self._log_defense(rnd, client_ids, client_proxies, defense_results, partition_ids)
        self._log_csv(rnd, results, client_ids, client_proxies, raw_features, defense_results)


        for p in client_proxies:
            self._get_partition_id(p)

        prev_size = len(getattr(self, '_partition_to_proxy', {}))
        if not hasattr(self, '_partition_to_proxy'):
            self._partition_to_proxy = {}
        for cid, proxy in self._all_client_map.items():
            pid = self._cid_to_partition.get(cid)
            if pid is not None:
                self._partition_to_proxy[pid] = proxy
        new_size = len(self._partition_to_proxy)

        if self._partition_cache_built and new_size > prev_size:
            known = set(self._cid_to_partition.values())
            kb = [p for p in self.backdoor_partition_ids if p in known]
            print(
                f"[FedMAST] Partition cache GREW: {prev_size} → {new_size} mapped "
                f"({len(kb)}/{len(self.backdoor_partition_ids)} backdoor)"
            )

        if not self._partition_cache_built:
            known = set(self._cid_to_partition.values())
            kb = [p for p in self.backdoor_partition_ids if p in known]
            kben = [p for p in self.benign_partition_ids if p in known]
            need_mal = len(self.backdoor_partition_ids)
            need_benign = max(
                1,
                int(self.num_clients * self.fraction_fit)
                - int(self.num_of_malicious_clients_per_round),
            )
            dba_ok = True
            if self._round_attack_type == "dba":
                dba_ok = all(pid in known for pid in self.dba_attacker_partition_ids)
            if len(kb) >= need_mal and len(kben) >= need_benign and dba_ok:
                self._partition_cache_built = True
                print(
                    f"[FedMAST] Partition cache READY: {new_size} mapped "
                    f"({len(kb)}/{need_mal} backdoor, {len(kben)}/{need_benign} benign)"
                )
            elif rnd <= 5 or rnd % 20 == 0:
                print(
                    f"[FedMAST] Partition cache building: {len(kb)}/{need_mal} backdoor, "
                    f"{len(kben)}/{need_benign} benign known (round {rnd})"
                )

        if self.fedmast_history_mode == "build":
            if rnd % 10 == 0:
                FedMASTHistory().save(
                    self.fedmast_history_path,
                    self._raw_buffer,
                    self._accepted_buffer,
                    self._aggregate_tracker,
                    rnd,
                    model_id="TinyResNet18-base_width_8",
                    dataset=f"CIFAR-10;dirichlet_alpha={self.dirichlet_alpha}",
                )
                print(f"[FedMAST] History checkpoint saved (round {rnd})")
        elif self.fedmast_history_mode == "load-update-safe":
            if rnd % 50 == 0 or rnd >= 100:
                if not round_suspicious:
                    FedMASTHistory().save(
                        self.fedmast_history_path,
                        self._raw_buffer,
                        self._accepted_buffer,
                        self._aggregate_tracker,
                        rnd,
                        model_id="TinyResNet18-base_width_8",
                        dataset=f"CIFAR-10;dirichlet_alpha={self.dirichlet_alpha}",
                    )

        return aggregated_params, aggregated_metrics


    def _log_defense(self, rnd, client_ids, client_proxies, defense_results, partition_ids):
        n_acc = sum(1 for r in defense_results.values() if r.get("accepted"))
        n_rej = sum(1 for r in defense_results.values() if not r.get("accepted"))
        first = next(iter(defense_results.values()), {})

        spec_tracker = self._spectral_drift_tracker

        print(f"\n{'='*120}")
        print(f"[FedMAST v4] Round {rnd}  |  "
              f"Buffer={self._accepted_buffer.size}/{self._raw_buffer.size}  |  "
              f"Trajectories={len(self._trajectory_tracker._trajectories)}  |  "
              f"SpectralDrift={spec_tracker.n_active()} active  |  "
              f"AggTracker={'active' if self._aggregate_tracker.is_active() else 'warmup'}  |  "
              f"History={self.fedmast_history_mode}"
              f"{'(loaded)' if self._history_loaded else ''}")
        susp = first.get("round_suspicious", False)
        spec_risk = first.get("spectral_risk", False)
        if susp or spec_risk:
            flags = []
            if susp:
                flags.append("SUSPICIOUS")
            if spec_risk:
                flags.append("SPECTRAL_RISK")
            print(f"  *** {' + '.join(flags)} — buffer QUARANTINE ***")
        print(f"{'='*120}")
        print(f"  Feature bank: {N_BANK} features x {len(FAMILIES)} families  |  "
              f"{len(SQUEEZE_PAIRS)} squeeze pairs  |  "
              f"Hard consensus: 2-of-3 (D_round, D_squeeze, D_hist)")

        pat = first.get("per_axis_thresholds", {})
        if pat:
            parts = []
            for ax in ["D_round", "D_squeeze", "D_hist"]:
                t = pat.get(ax, float("inf"))
                parts.append(f"{ax}={t:.1f}" if t < 1e6 else f"{ax}=inf")
            spec_t = first.get("spectral_threshold", float("inf"))
            parts.append(f"D_spectral={spec_t:.3f}" if spec_t < 1e6 else "D_spectral=inf")
            print(f"  Thresholds (k={self.fedmast_threshold_k}): {' | '.join(parts)}")

            soft_parts = []
            for ax in ["D_anchor", "D_traj", "D_momentum"]:
                t = pat.get(ax, float("inf"))
                soft_parts.append(f"{ax}={t:.1f}" if t < 1e6 else f"{ax}=inf")
            print(f"  Soft axes (monitoring only): {' | '.join(soft_parts)}")

        print(f"  Decision: {n_acc} accepted, {n_rej} rejected\n")

        h = (f"  {'CID':>6} {'PID':>4} {'MAL':>3} |"
             f" {'Score':>7} {'D_rnd':>6} {'D_sqz':>6} {'D_hst':>6} "
             f"{'D_spc':>6} |"
             f" {'D_anc':>6} {'D_trj':>6} {'D_mom':>6} |"
             f" {'Axis':>10} | {'Decision':>18}")
        print(h)
        print(f"  {'-'*len(h)}")

        sorted_cids = sorted(client_ids, key=lambda c: defense_results.get(c, {}).get("score", 0), reverse=True)
        for cid in sorted_cids:
            r = defense_results.get(cid, {})
            pid = partition_ids.get(cid, self._get_partition_id(client_proxies[client_ids.index(cid)]))
            mal = str(cid) in self._round_malicious_ids_set

            dec = "REJECTED" if not r.get("accepted") else "accepted"
            if mal and not r.get("accepted"):
                dec += " TP"
            elif mal and r.get("accepted"):
                dec += " FN"
            elif not mal and not r.get("accepted"):
                dec += " FP"
            else:
                dec += " TN"

            flagging = r.get("flagging_axes", [])
            flag_str = ",".join(a.replace("D_", "") for a in flagging) if flagging else ""

            spec_info = ""
            p = partition_ids.get(cid, -1)
            if p >= 0 and spec_tracker.is_active(p):
                spec_info = f" [spc:{spec_tracker.n_appearances(p)}app]"

            print(f"  {cid:>6} {pid:>4} {'YES' if mal else ' no':>3} |"
                  f" {r.get('score',0):>7.2f} {r.get('d_round',0):>6.2f} "
                  f"{r.get('d_squeeze',0):>6.2f} {r.get('d_hist',0):>6.2f} "
                  f"{r.get('d_spectral',0):>6.3f} |"
                  f" {r.get('d_anchor',0):>6.2f} "
                  f"{r.get('d_traj',0):>6.2f} {r.get('d_momentum',0):>6.2f} |"
                  f" {r.get('dominant_axis','n/a'):>10} | {dec:>18}"
                  f"{'  [' + flag_str + ']' if flag_str else ''}"
                  f"{spec_info}")

        tp = sum(1 for c in client_ids if str(c) in self._round_malicious_ids_set and not defense_results.get(c, {}).get("accepted", True))
        fn = sum(1 for c in client_ids if str(c) in self._round_malicious_ids_set and defense_results.get(c, {}).get("accepted", True))
        fp = sum(1 for c in client_ids if str(c) not in self._round_malicious_ids_set and not defense_results.get(c, {}).get("accepted", True))
        tn = sum(1 for c in client_ids if str(c) not in self._round_malicious_ids_set and defense_results.get(c, {}).get("accepted", True))
        prec = tp / (tp + fp) if (tp + fp) else 0
        rec = tp / (tp + fn) if (tp + fn) else 0
        print(f"\n  Detection: TP={tp} FN={fn} FP={fp} TN={tn}  |  "
              f"Prec={prec:.2f} Rec={rec:.2f}  |  Attackers={tp+fn}")

        if n_rej > 0:
            rej_cids = [c for c in sorted_cids if not defense_results.get(c, {}).get("accepted")]
            if rej_cids:
                r0 = defense_results[rej_cids[0]]
                ps = r0.get("pair_scores", {})
                if ps:
                    print(f"\n  Squeeze scores (most anomalous client {rej_cids[0]}):")
                    for pname, pscore in sorted(ps.items(), key=lambda x: x[1], reverse=True):
                        print(f"    {pname:<45s} {pscore:.3f}")

        print(f"{'='*120}\n")


    def _log_csv(self, rnd, results, client_ids, client_proxies, raw_features, defense_results):
        for i, (cp, fr) in enumerate(results):
            cid = str(cp.cid)
            pid = self._get_partition_id(cp)
            mal = cid in self._round_malicious_ids_set
            cm = dict(getattr(fr, "metrics", {}) or {})
            dr = defense_results.get(cid, {})

            metadata = {
                "simulation_id": self.simulation_id, "round": rnd,
                "cid": cid, "partition_id": int(pid) if pid is not None else -1,
                "malicious_flag": int(mal),
                "attack_mode": self._round_attack_mode,
                "attack_type": self._round_attack_type if mal else "none",
                "experiment_attack_type": self._round_attack_type,
                "local_data_size": int(getattr(fr, "num_examples", 0)),
                "local_epochs": cm.get("local_epochs"),
                "local_lr": cm.get("local_lr"),
                "scale_factor": cm.get("scale_factor"),
                "selected_by_aggregator": int(dr.get("accepted", True)),
                "krum_score": float(dr.get("score", 0)),
                "krum_rank": 0,
                "dirichlet_alpha": self.dirichlet_alpha, "target_label": 2,
                "aggregation_method": "fedmast",
                "num_clients": int(self.num_clients),
                "fedmast_score": float(dr.get("score", 0)),
                "fedmast_d_round": float(dr.get("d_round", 0)),
                "fedmast_d_squeeze": float(dr.get("d_squeeze", 0)),
                "fedmast_d_hist": float(dr.get("d_hist", 0)),
                "fedmast_d_anchor": float(dr.get("d_anchor", 0)),
                "fedmast_d_traj": float(dr.get("d_traj", 0)),
                "fedmast_d_momentum": float(dr.get("d_momentum", 0)),
                "fedmast_d_spectral": float(dr.get("d_spectral", 0)),
                "fedmast_threshold": float(dr.get("threshold", 0)),
                "fedmast_spectral_threshold": float(dr.get("spectral_threshold", 0)),
                "fedmast_dominant_axis": str(dr.get("dominant_axis", "n/a")),
            }
            feat = raw_features.get(cid)
            if feat:
                try:
                    append_per_update_features(metadata=metadata, features=feat)
                except Exception as e:
                    print(f"[FedMAST][Round {rnd}] CSV fail CID={cid}: {e}")

    def aggregate_evaluate(self, rnd, results, failures):
        return super().aggregate_evaluate(rnd, results, failures)
