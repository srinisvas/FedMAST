import json
import random
import traceback
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import flwr as fl
from flwr.common import FitIns, Parameters, GetPropertiesIns
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
import torch
from torch.nn.utils import parameters_to_vector

from fed_learning_cifar_experiment.task import (
    get_resnet_cnn_model,
    set_weights,
    load_data,
    train,
)
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

from fed_learning_cifar_experiment.state.krum_metrics_strategy import SaveKrumMetricsStrategy


class SaveTrimmedMeanMetricsStrategy(SaveKrumMetricsStrategy):

    def __init__(
        self,
        *,
        num_to_trim: Optional[int] = None,
        trim_ratio: Optional[float] = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        self.num_to_trim = None if num_to_trim is None else int(num_to_trim)
        self.trim_ratio = None if trim_ratio is None else float(trim_ratio)

        self._last_round_malicious_ids: List[str] = []

        self._param_registry: Optional[ParamRegistry] = None
        self._round_global_parameters: Optional[Parameters] = None
        self._round_ref_deltas_flat: Optional[np.ndarray] = None
        self._round_attack_mode: str = "none"
        self._round_attack_type: str = "none"
        self._round_malicious_ids_set: set = set()

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        num_available = len(client_manager.all())
        sample_size, min_num = self.num_fit_clients(num_available)

        all_clients = list(client_manager.all().values())

        if self.attacker_selection_mode == "persistent":
            try:
                cfg0 = self.on_fit_config_fn(server_round) if self.on_fit_config_fn else {}
                raw = cfg0.get("malicious-client-ids", "[]")
                malicious_pool = json.loads(raw) if isinstance(raw, str) else list(raw)
            except Exception:
                print(
                    f"[Round {server_round}] Failed to parse malicious client IDs "
                    f"from config. Falling back to empty pool."
                )
                malicious_pool = []

            malicious_pool = set(str(cid) for cid in malicious_pool)

            attacker_clients = [
                c for c in all_clients if str(c.cid) in malicious_pool
            ][: int(self.num_of_malicious_clients_per_round)]

            attacker_cids = set(c.cid for c in attacker_clients)

            benign_candidates = [
                c for c in all_clients if c.cid not in attacker_cids
            ]

            remaining = max(0, sample_size - len(attacker_clients))
            sampled_benign = random.sample(
                benign_candidates, min(len(benign_candidates), remaining)
            )

            sampled_clients = attacker_clients + sampled_benign
            malicious_ids = [c.cid for c in attacker_clients]

        else:
            sampled_clients = list(client_manager.sample(sample_size, min_num))
            sampled_ids_tmp = [c.cid for c in sampled_clients]

            num_malicious = min(
                self.num_of_malicious_clients_per_round, len(sampled_ids_tmp)
            )
            malicious_ids = random.sample(sampled_ids_tmp, num_malicious)

        sampled_ids = [c.cid for c in sampled_clients]

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        nds = parameters_to_ndarrays(parameters)

        model_tmp = get_resnet_cnn_model()
        set_weights(model_tmp, nds)
        model_tmp.to(device)
        init_vec = parameters_to_vector(model_tmp.parameters()).detach().cpu()

        ref_partition_ids = random.sample(range(self.num_clients), 6)
        ref_deltas = []

        for pid in ref_partition_ids:
            train_loader, _ = load_data(
                partition_id=pid,
                num_partitions=self.num_clients,
                alpha_val=self.dirichlet_alpha,
                backdoor_enabled=False,
            )

            net_ref = get_resnet_cnn_model()
            set_weights(net_ref, nds)
            net_ref.to(device)

            lr = random.choice([0.003, 0.004, 0.005])
            epochs = random.choice([1, 2])

            _, vec = train(net_ref, train_loader, epochs, device, lr)
            delta = (vec - init_vec).cpu().numpy()
            ref_deltas.append(delta)

        ref_deltas = np.stack(ref_deltas)
        median_norm = float(np.median(np.linalg.norm(ref_deltas, axis=1)))

        self._last_round_malicious_ids = list(map(str, malicious_ids))

        fit_ins_list: List[Tuple[ClientProxy, FitIns]] = []
        print(f"Sampled clients for round {server_round}: {sampled_ids}")
        print(f"Malicious clients for round {server_round}: {malicious_ids}")

        for client in sampled_clients:
            config = self.on_fit_config_fn(server_round) if self.on_fit_config_fn else {}
            config.update(
                {
                    "current-round": server_round,
                    "sampled_client_ids": json.dumps(sampled_ids),
                    "malicious_client_ids": json.dumps(malicious_ids),
                    "is_malicious": str(client.cid in malicious_ids),
                    "shared_ref_deltas": json.dumps(ref_deltas.tolist()),
                    "shared_ref_median_norm": median_norm,
                }
            )

            if self.attacker_selection_mode == "persistent":
                config["krum_selected_cid"] = self.last_krum_selected_cid
                config["krum_ref_delta"] = (
                    json.dumps(self.last_krum_selected_delta.tolist())
                    if self.last_krum_selected_delta is not None
                    else None
                )

            if self.prev_global_parameters is not None:
                config["prev_global_tensors_hex"] = json.dumps(
                    [t.hex() for t in self.prev_global_parameters.tensors]
                )
                config["prev_global_tensor_type"] = self.prev_global_parameters.tensor_type
            else:
                config["prev_global_tensors_hex"] = "[]"
                config["prev_global_tensor_type"] = "numpy.ndarray"

            fit_ins_list.append((client, FitIns(parameters, config)))

        if self._param_registry is None:
            tmp_model = get_resnet_cnn_model()
            self._param_registry = ParamRegistry(tmp_model)
            print(
                f"[Extractor] ParamRegistry built: "
                f"{len(self._param_registry.entries)} trainable params, "
                f"{self._param_registry.total_trainable_params} elements"
            )

        self._round_global_parameters = parameters
        self._round_ref_deltas_flat = ref_deltas
        self._round_malicious_ids_set = set(str(x) for x in malicious_ids)

        cfg_template = self.on_fit_config_fn(server_round) if self.on_fit_config_fn else {}
        self._round_attack_mode = str(cfg_template.get("backdoor-attack-mode", "none"))
        self._round_attack_type = str(cfg_template.get("backdoor-attack-type", "none"))

        self.prev_global_parameters = parameters

        return fit_ins_list

    @staticmethod
    def _is_float_arr(a: np.ndarray) -> bool:
        return np.issubdtype(a.dtype, np.floating)

    @staticmethod
    def _is_int_or_bool_arr(a: np.ndarray) -> bool:
        return np.issubdtype(a.dtype, np.integer) or np.issubdtype(a.dtype, np.bool_)

    def _effective_trim_count(self, n: int) -> int:
        if self.num_to_trim is not None:
            t = int(self.num_to_trim)
        elif self.trim_ratio is not None:
            t = int(np.floor(float(self.trim_ratio) * n))
        else:
            f = getattr(self, "num_byzantine", None)
            if f is None:
                f = getattr(self, "num_malicious_clients", 0)
            t = int(f)

        max_safe = max(0, (n - 1) // 2)
        return max(0, min(t, max_safe))

    def aggregate_fit(self, rnd: int, results, failures):
        if not results:
            return super().aggregate_fit(rnd, results, failures)
        if failures:
            print(
                f"[Round {rnd}][TrimmedMean] Received {len(failures)} failures; "
                f"continuing with {len(results)} successful results."
            )

        client_ids: List[str] = []
        client_params_nds: List[List[np.ndarray]] = []
        client_updates_flat: List[np.ndarray] = []
        client_proxies: List[ClientProxy] = []

        for client_proxy, fit_res in results:
            nds = parameters_to_ndarrays(fit_res.parameters)
            flat = np.concatenate([np.asarray(p).ravel() for p in nds])
            client_updates_flat.append(flat)
            client_params_nds.append([np.asarray(p) for p in nds])
            client_ids.append(str(client_proxy.cid))
            client_proxies.append(client_proxy)

        X = np.stack(client_updates_flat)
        n = len(X)
        trim_count = self._effective_trim_count(n)

        template = client_params_nds[0]
        agg_nds: List[np.ndarray] = []

        for layer_idx, base in enumerate(template):
            base_arr = np.asarray(base)

            if self._is_int_or_bool_arr(base_arr):
                agg_nds.append(base_arr.copy())
                continue

            stacked = np.stack(
                [np.asarray(client_params_nds[i][layer_idx]).astype(np.float64, copy=False)
                 for i in range(n)],
                axis=0,
            )

            if trim_count > 0:
                sorted_vals = np.sort(stacked, axis=0)
                kept = sorted_vals[trim_count : n - trim_count]
            else:
                kept = stacked

            agg = np.mean(kept, axis=0)
            agg_nds.append(agg.astype(base_arr.dtype, copy=False))

        new_parameters = ndarrays_to_parameters(agg_nds)

        agg_flat = np.concatenate([np.asarray(p).ravel() for p in agg_nds])
        dist_to_trimmed_mean = np.linalg.norm(X - agg_flat, axis=1)
        scores: Dict[str, float] = {
            client_ids[i]: float(dist_to_trimmed_mean[i]) for i in range(n)
        }

        canonical_idx = int(np.argmin(dist_to_trimmed_mean))
        canonical_cid = client_ids[canonical_idx]

        print(f"\n[Round {rnd}][Trimmed Mean Dist-to-Aggregate]")
        for cid in sorted(scores, key=scores.get):
            proxy = next(p for p in client_proxies if str(p.cid) == cid)
            pid = self._get_partition_id(proxy)
            is_mal = str(cid) in set(self._last_round_malicious_ids)
            print(
                f"  CID={cid:>6} | Partition={pid:>3} | "
                f"Dist={scores[cid]:.6e} | Malicious={is_mal}"
            )

        print(
            f"[Round {rnd}][Trimmed Mean] "
            f"n={n} | trim_each_tail={trim_count} | "
            f"canonical={canonical_cid}"
        )

        self.last_krum_selected_cid = canonical_cid
        if self.prev_global_parameters is not None:
            prev_nds = parameters_to_ndarrays(self.prev_global_parameters)
            prev_flat = np.concatenate([np.asarray(p).ravel() for p in prev_nds])
            curr_flat = client_updates_flat[canonical_idx]
            self.last_krum_selected_delta = curr_flat - prev_flat
        else:
            self.last_krum_selected_delta = None


        selected_cids = set(client_ids)

        try:
            self._extract_and_log_features(
                rnd=rnd,
                results=results,
                client_ids=client_ids,
                client_proxies=client_proxies,
                scores=scores,
                selected_cids=selected_cids,
            )
        except Exception as e:
            print(
                f"[Extractor][Round {rnd}] Feature extraction failed: "
                f"{type(e).__name__}: {e}"
            )
            traceback.print_exc()

        return new_parameters, {}


    def _extract_and_log_features(
        self,
        rnd: int,
        results,
        client_ids: List[str],
        client_proxies: List,
        scores: Dict[str, float],
        selected_cids: set,
    ) -> None:
        if self._param_registry is None:
            return

        registry = self._param_registry

        global_params = self._round_global_parameters
        if global_params is None:
            global_params = self.prev_global_parameters
        if global_params is None:
            return

        global_nds = parameters_to_ndarrays(global_params)
        global_sd_train = nds_to_trainable_state_dict(global_nds, registry.state_dict_keys)

        per_client = []
        for client_proxy, fit_res in results:
            cid = client_proxy.cid
            try:
                client_nds = parameters_to_ndarrays(fit_res.parameters)
                client_sd = nds_to_trainable_state_dict(client_nds, registry.state_dict_keys)
                delta_sd = state_dict_delta(client_sd, global_sd_train)
            except Exception as e:
                print(f"[Extractor][Round {rnd}] Skipping CID={cid}: {e}")
                continue
            per_client.append({
                "cid": cid,
                "proxy": client_proxy,
                "fit_res": fit_res,
                "delta_sd": delta_sd,
            })

        if not per_client:
            return

        round_mean_sd = mean_delta_per_key(c["delta_sd"] for c in per_client)
        loo_means = build_leave_one_out_means([c["delta_sd"] for c in per_client])
        for i, c in enumerate(per_client):
            c["loo_mean"] = loo_means[i]

        reference_delta_per_key = None
        ref_deltas_per_key_list = None
        if self._round_ref_deltas_flat is not None and len(self._round_ref_deltas_flat) > 0:
            try:
                ref_mean_flat = torch.from_numpy(
                    self._round_ref_deltas_flat.mean(axis=0)
                ).float()
                if ref_mean_flat.numel() == registry.total_trainable_params:
                    reference_delta_per_key = flat_param_vec_to_per_key_dict(
                        ref_mean_flat, registry
                    )
                    ref_deltas_per_key_list = ref_deltas_flat_to_per_key_list(
                        self._round_ref_deltas_flat, registry
                    )
            except Exception as e:
                print(f"[Extractor][Round {rnd}] reference build failed: {e}")

        stage_proj_bases = None
        stage_proj_ref_means = None
        if ref_deltas_per_key_list is not None:
            try:
                stage_proj_bases, stage_proj_ref_means = build_stage_projection_bases(
                    ref_deltas_per_key_list, registry, k=3
                )
            except Exception as e:
                print(f"[Extractor][Round {rnd}] projection basis fit failed: {e}")

        sorted_by_score = sorted(scores.keys(), key=lambda c: scores[c])
        rank_of = {cid: i + 1 for i, cid in enumerate(sorted_by_score)}

        for entry in per_client:
            cid = entry["cid"]
            proxy = entry["proxy"]
            fit_res = entry["fit_res"]
            delta_sd = entry["delta_sd"]
            loo_mean = entry["loo_mean"]

            partition_id = self._get_partition_id(proxy)
            is_malicious = str(cid) in self._round_malicious_ids_set
            client_metrics = dict(getattr(fit_res, "metrics", {}) or {})
            effective_attack_type = self._round_attack_type if is_malicious else "none"

            metadata = {
                "simulation_id": self.simulation_id,
                "round": rnd,
                "cid": str(cid),
                "partition_id": int(partition_id) if partition_id is not None else -1,
                "malicious_flag": int(is_malicious),
                "attack_mode": self._round_attack_mode,
                "attack_type": effective_attack_type,
                "experiment_attack_type": self._round_attack_type,
                "local_data_size": int(getattr(fit_res, "num_examples", 0)),
                "local_epochs": client_metrics.get("local_epochs"),
                "local_lr": client_metrics.get("local_lr"),
                "scale_factor": client_metrics.get("scale_factor"),
                "selected_by_aggregator": int(str(cid) in selected_cids),
                "krum_score": float(scores.get(str(cid), float("nan"))),
                "krum_rank": int(rank_of.get(str(cid), -1)),
                "dirichlet_alpha": self.dirichlet_alpha,
                "target_label": 2,
                "aggregation_method": "trimmed_mean",
                "num_clients": int(self.num_clients),
            }

            try:
                features = extract_per_update_features(
                    client_delta_per_key=delta_sd,
                    round_mean_delta_per_key=round_mean_sd,
                    registry=registry,
                    reference_delta_per_key=reference_delta_per_key,
                    leave_one_out_mean_per_key=loo_mean,
                    stage_projection_bases=stage_proj_bases,
                    stage_projection_ref_means=stage_proj_ref_means,
                )
            except Exception as e:
                print(f"[Extractor][Round {rnd}][CID={cid}] feature compute failed: {e}")
                continue

            try:
                append_per_update_features(metadata=metadata, features=features)
            except Exception as e:
                print(f"[Extractor][Round {rnd}][CID={cid}] CSV write failed: {e}")
