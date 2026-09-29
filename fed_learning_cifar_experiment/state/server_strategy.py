import json
import random
import traceback
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import flwr as fl
from flwr.common import FitIns, GetPropertiesIns, Parameters, parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

import torch
from torch.nn.utils import parameters_to_vector

from fed_learning_cifar_experiment.threeDFed import AttackerCoordinator

from fed_learning_cifar_experiment.utils.logger import (
    append_distributed_round,
    write_experiment_summary,
    append_per_update_features,
)
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
from fed_learning_cifar_experiment.task import (
    get_resnet_cnn_model, set_weights, load_data, train,
)


class SaveFedAvgMetricsStrategy(fl.server.strategy.FedAvg):

    def __init__(
        self,
        simulation_id: str = "",
        num_clients: int = 0,
        num_rounds: int = 0,
        aggregation_method: str = "",
        backdoor_attack_mode: str = "",
        num_of_malicious_clients: int = 0,
        num_of_malicious_clients_per_round: int = 0,
        **kwargs: Any,
    ):
        _backdoor_partition_ids = kwargs.pop("backdoor_partition_ids", None)
        _attack_start_round = kwargs.pop("attack_start_round", 1)
        _attack_end_round = kwargs.pop("attack_end_round", 999999)
        _dba_attacker_partition_ids = kwargs.pop("dba_attacker_partition_ids", [5, 12, 23, 36])
        _dba_server_eta = kwargs.pop("dba_server_eta", 1.0)
        _dirichlet_alpha = kwargs.pop("dirichlet_alpha", 0.3)
        super().__init__(**kwargs)

        self.dirichlet_alpha = float(_dirichlet_alpha)

        self.simulation_id = simulation_id
        self.num_clients = num_clients
        self.num_rounds = num_rounds
        self.aggregation_method = aggregation_method
        self.backdoor_attack_mode = backdoor_attack_mode
        self.num_of_malicious_clients = num_of_malicious_clients
        self.num_of_malicious_clients_per_round = num_of_malicious_clients_per_round

        self.attack_start_round = int(_attack_start_round)
        self.attack_end_round = int(_attack_end_round)

        self._3dfed_coordinator: Optional[AttackerCoordinator] = None

        self.dba_attacker_partition_ids = [int(x) for x in _dba_attacker_partition_ids]
        self.dba_server_eta = float(_dba_server_eta)

        self.history: Dict[str, list] = {"round": [], "mta": [], "asr": []}
        self.central_mta_history: List[float] = []
        self.central_asr_history: List[float] = []
        self.final_centralized_mta: Optional[float] = None
        self.final_centralized_asr: Optional[float] = None

        self._cid_to_partition: Dict[str, int] = {}

        self.prev_global_parameters: Optional[Parameters] = None

        self._param_registry: Optional[ParamRegistry] = None
        self._round_global_parameters: Optional[Parameters] = None
        self._round_ref_deltas_flat: Optional[np.ndarray] = None
        self._round_attack_mode: str = "none"
        self._round_attack_type: str = "none"
        self._round_sampled_ids: List[str] = []
        self._round_malicious_ids_set: set = set()

        if _backdoor_partition_ids is not None:
            self.backdoor_partition_ids: List[int] = list(_backdoor_partition_ids)
        else:
            self.backdoor_partition_ids = random.sample(
                range(int(num_clients)), min(10, int(num_clients))
            )
        self.benign_partition_ids: List[int] = [
            p for p in range(int(num_clients))
            if p not in set(self.backdoor_partition_ids)
        ]

        self._partition_to_proxy: Dict[int, ClientProxy] = {}
        self._partition_cache_built: bool = False

        print(
            f"[Strategy] Backdoor partitions ({len(self.backdoor_partition_ids)}): "
            f"{sorted(self.backdoor_partition_ids)}"
        )


    def _get_partition_id(self, client: ClientProxy) -> int:
        if client.cid in self._cid_to_partition:
            return self._cid_to_partition[client.cid]

        try:
            res = client.get_properties(
                GetPropertiesIns(config={}), timeout=5.0, group_id=None
            )
            if res is None:
                raise RuntimeError("get_properties returned None")
            if not hasattr(res, "properties"):
                raise RuntimeError(f"Invalid GetPropertiesRes: {res}")
            if "partition_id" not in res.properties:
                raise KeyError(f"'partition_id' missing in properties: {res.properties}")
            pid = int(res.properties["partition_id"])
        except Exception as e:
            print("=" * 80)
            print("[ERROR] Failed to fetch partition_id")
            print(f"CID            : {client.cid}")
            print(f"Exception type : {type(e).__name__}")
            print(f"Exception msg  : {e}")
            traceback.print_exc()
            print("=" * 80)
            pid = -1

        self._cid_to_partition[client.cid] = pid
        return pid


    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:

        if not hasattr(self, "_all_client_map") or self._all_client_map is None:
            self._all_client_map = client_manager.all()

        sample_size, _ = self.num_fit_clients(len(self._all_client_map))

        in_attack_window = (
            self.attack_start_round <= server_round <= self.attack_end_round
        )
        effective_malicious_per_round = (
            self.num_of_malicious_clients_per_round if in_attack_window else 0
        )
        print(
            f"[Round {server_round}] Attack window: "
            f"[{self.attack_start_round}, {self.attack_end_round}] | "
            f"in_window={in_attack_window} | "
            f"effective_malicious={effective_malicious_per_round}"
        )

        if not self._partition_cache_built:
            cfg_pre = self.on_fit_config_fn(server_round) if self.on_fit_config_fn else {}
            pre_attack_type = str(cfg_pre.get("backdoor-attack-type", "none")).lower()

            if pre_attack_type == "dba" and in_attack_window:
                known_pids = set(self._cid_to_partition.values()) if hasattr(self, "_cid_to_partition") else set()
                dba_ready = all(
                    pid in known_pids for pid in self.dba_attacker_partition_ids
                )

                if dba_ready:
                    self._partition_to_proxy = {}
                    for cid, proxy in client_manager.all().items():
                        pid = self._cid_to_partition.get(cid)
                        if pid is not None and pid >= 0:
                            self._partition_to_proxy[pid] = proxy

                    self._partition_cache_built = True
                    print(
                        f"[DBA] Partition cache ready: "
                        f"{len(self._partition_to_proxy)} mapped | "
                        f"DBA attackers {self.dba_attacker_partition_ids} all resolved"
                    )
                else:
                    missing = [
                        pid for pid in self.dba_attacker_partition_ids
                        if pid not in known_pids
                    ]
                    print(
                        f"[DBA] Waiting for attacker partitions: "
                        f"missing={missing} | known={len(known_pids)} | "
                        f"Running benign this round"
                    )
                    effective_malicious_per_round = 0

        if self._partition_cache_built:

            cfg_check = self.on_fit_config_fn(server_round) if self.on_fit_config_fn else {}
            current_attack_type = str(cfg_check.get("backdoor-attack-type", "none")).lower()

            if current_attack_type == "dba" and effective_malicious_per_round > 0:
                required = self.dba_attacker_partition_ids
                missing = [
                    pid for pid in required
                    if pid not in self._partition_to_proxy
                ]
                if missing:
                    raise RuntimeError(
                        f"[DBA] Fixed attackers unresolved: {missing}. "
                        f"Cache has {len(self._partition_to_proxy)} partitions."
                    )

                sampled_backdoor = [
                    self._partition_to_proxy[pid] for pid in required
                ]
                benign_candidates = [
                    proxy for pid, proxy in self._partition_to_proxy.items()
                    if pid not in set(required)
                ]
                sampled_benign = random.sample(
                    benign_candidates,
                    min(sample_size - len(sampled_backdoor), len(benign_candidates))
                )

                sampled_clients = sampled_backdoor + sampled_benign
                random.shuffle(sampled_clients)
                sampled_ids = [c.cid for c in sampled_clients]
                malicious_ids = [c.cid for c in sampled_backdoor]

                slot_map = {pid: slot for slot, pid in enumerate(required)}
                mal_parts = [self._cid_to_partition.get(c, -1) for c in malicious_ids]
                mal_slots = [slot_map.get(p, -1) for p in mal_parts]

                print(
                    f"[DBA][Coordinator] round={server_round} "
                    f"partitions={required} slots={mal_slots}"
                )

            else:
                backdoor_proxies = [
                    self._partition_to_proxy[p]
                    for p in self.backdoor_partition_ids
                    if p in self._partition_to_proxy
                ]
                benign_proxies = [
                    self._partition_to_proxy[p]
                    for p in self.benign_partition_ids
                    if p in self._partition_to_proxy
                ]

                num_attackers = int(effective_malicious_per_round)
                if num_attackers > len(backdoor_proxies):
                    mapped_backdoors = sorted([
                        p for p in self.backdoor_partition_ids
                        if p in self._partition_to_proxy
                    ])
                    raise RuntimeError(
                        f"[Round {server_round}] Requested {num_attackers} malicious "
                        f"clients, but only {len(backdoor_proxies)} mapped backdoor "
                        f"proxies available. Mapped: {mapped_backdoors}"
                    )
                sampled_backdoor = random.sample(backdoor_proxies, num_attackers)
                sampled_benign = random.sample(
                    benign_proxies, min(sample_size - num_attackers, len(benign_proxies))
                )

                sampled_clients = sampled_backdoor + sampled_benign
                random.shuffle(sampled_clients)

                sampled_ids = [c.cid for c in sampled_clients]
                malicious_ids = [c.cid for c in sampled_backdoor]
                mal_parts = [self._cid_to_partition.get(c, -1) for c in malicious_ids]

                print(f"[Round {server_round}] STRATIFIED | Malicious partitions: {mal_parts}")

        else:
            sampled_clients = list(client_manager.sample(sample_size, sample_size))
            sampled_ids = [c.cid for c in sampled_clients]

            num_malicious = min(
                effective_malicious_per_round, len(sampled_ids)
            )
            malicious_ids = random.sample(sampled_ids, num_malicious)

            print(
                f"[Round {server_round}] RANDOM (cache: "
                f"{len(self._cid_to_partition)}/{self.num_clients}) | "
                f"Malicious CIDs: {malicious_ids}"
            )

        self._round_malicious_ids_set = set(str(x) for x in malicious_ids)
        self._round_sampled_ids = [str(x) for x in sampled_ids]

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        nds = parameters_to_ndarrays(parameters)

        model_tmp = get_resnet_cnn_model()
        set_weights(model_tmp, nds)
        model_tmp.to(device)
        init_vec = parameters_to_vector(model_tmp.parameters()).detach().cpu()

        ref_partition_ids = random.sample(range(self.num_clients), 6)
        ref_deltas: List[np.ndarray] = []

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
            ref_deltas.append((vec - init_vec).cpu().numpy())

        ref_deltas_np = np.stack(ref_deltas)
        median_norm = float(np.median(np.linalg.norm(ref_deltas_np, axis=1)))

        fit_ins_list: List[Tuple[ClientProxy, FitIns]] = []
        for client in sampled_clients:
            config = self.on_fit_config_fn(server_round) if self.on_fit_config_fn else {}
            config.update({
                "current-round": server_round,
                "sampled_client_ids": json.dumps(sampled_ids),
                "malicious_client_ids": json.dumps(malicious_ids),
                "is_malicious": str(client.cid in malicious_ids),
                "shared_ref_deltas": json.dumps(ref_deltas_np.tolist()),
                "shared_ref_median_norm": median_norm,
            })

            if self.prev_global_parameters is not None:
                config["prev_global_tensors_hex"] = json.dumps(
                    [t.hex() for t in self.prev_global_parameters.tensors]
                )
                config["prev_global_tensor_type"] = self.prev_global_parameters.tensor_type
            else:
                config["prev_global_tensors_hex"] = "[]"
                config["prev_global_tensor_type"] = "numpy.ndarray"

            attack_type = config.get("backdoor-attack-type", "none")
            if attack_type == "dba" and client.cid in malicious_ids:
                raw_dba_parts = config.get("dba-attacker-partition-ids", "[]")
                try:
                    dba_parts = json.loads(raw_dba_parts) if isinstance(raw_dba_parts, str) else list(raw_dba_parts)
                except Exception:
                    dba_parts = []

                if dba_parts:
                    dba_slot_map = {int(pid): slot for slot, pid in enumerate(dba_parts)}
                    client_pid = self._get_partition_id(client)
                    if client_pid is not None and int(client_pid) in dba_slot_map:
                        config["dba-attacker-index"] = dba_slot_map[int(client_pid)]

            attack_type = config.get("backdoor-attack-type", "none")
            if attack_type == "3dfed" and client.cid in malicious_ids:
                if self._3dfed_coordinator is None:
                    self._3dfed_coordinator = AttackerCoordinator(
                        alpha=0.5,
                        alpha_step=float(config.get("3dfed-alpha-step", 0.1)),
                        k_decoy=int(config.get("3dfed-num-decoy-models-init", 0)),
                        max_decoy=int(config.get("3dfed-max-decoy-models", 10)),
                        kappa=float(config.get("3dfed-kappa", 100000.0)),
                        num_backdoor_models=int(config.get("3dfed-num-backdoor-models", 4)),
                        canonical_partition_id=int(config.get("3dfed-canonical-partition-id", 5)),
                    )

                if not hasattr(self, "_3dfed_last_assigned_round") or \
                   self._3dfed_last_assigned_round != server_round:

                    self._3dfed_round_assignments = (
                        self._3dfed_coordinator.assign_roles(
                            malicious_ids, server_round
                        )
                    )
                    self._3dfed_last_assigned_round = server_round

                mal_idx = malicious_ids.index(client.cid)
                if mal_idx < len(self._3dfed_round_assignments):
                    config.update(self._3dfed_round_assignments[mal_idx])

                    if self._3dfed_coordinator.indicator_indices:
                        config["3dfed-indicator-indices"] = json.dumps(
                            self._3dfed_coordinator.indicator_indices
                        )

            fit_ins_list.append((client, FitIns(parameters, config)))

        if self._param_registry is None:
            tmp_model = get_resnet_cnn_model()
            self._param_registry = ParamRegistry(tmp_model)
            print(
                f"[Extractor] ParamRegistry built: "
                f"{len(self._param_registry.entries)} trainable params, "
                f"{self._param_registry.total_trainable_params} elements, "
                f"{len(self._param_registry.state_dict_keys)} state_dict keys"
            )

        self._round_global_parameters = parameters
        self._round_ref_deltas_flat = ref_deltas_np

        cfg_template = self.on_fit_config_fn(server_round) if self.on_fit_config_fn else {}
        self._round_attack_mode = str(cfg_template.get("backdoor-attack-mode", "none"))
        self._round_attack_type = str(cfg_template.get("backdoor-attack-type", "none"))

        self.prev_global_parameters = parameters

        return fit_ins_list


    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, Any]],
        failures,
    ):
        if not results:
            return super().aggregate_fit(rnd, results, failures)

        aggregated_params, aggregated_metrics = super().aggregate_fit(
            rnd, results, failures
        )


        client_ids = [cp.cid for cp, _ in results]
        client_proxies = [cp for cp, _ in results]

        flat_updates: List[np.ndarray] = []
        for _, fit_res in results:
            nds = parameters_to_ndarrays(fit_res.parameters)
            flat_updates.append(np.concatenate([p.flatten() for p in nds]))
        X = np.stack(flat_updates)

        round_mean_flat = np.mean(X, axis=0)
        dist_to_mean = np.linalg.norm(X - round_mean_flat, axis=1)

        rank_order = np.argsort(dist_to_mean)
        ranks = np.empty_like(rank_order)
        ranks[rank_order] = np.arange(1, len(rank_order) + 1)

        scores = {client_ids[i]: float(dist_to_mean[i]) for i in range(len(client_ids))}
        rank_of = {client_ids[i]: int(ranks[i]) for i in range(len(client_ids))}

        print(f"\n[Round {rnd}][FedAvg Dist-to-Mean]")
        for i, cid in enumerate(client_ids):
            pid = self._get_partition_id(client_proxies[i])
            is_mal = str(cid) in self._round_malicious_ids_set
            print(
                f"CID={cid:>6} | Partition={pid:>3} | "
                f"DistToMean={dist_to_mean[i]:.4f} | "
                f"Rank={ranks[i]} | Malicious={is_mal}"
            )

        try:
            self._extract_and_log_features(
                rnd=rnd,
                results=results,
                client_ids=client_ids,
                client_proxies=client_proxies,
                scores=scores,
                rank_of=rank_of,
            )
        except Exception as e:
            print(
                f"[Extractor][Round {rnd}] Feature extraction failed: "
                f"{type(e).__name__}: {e}"
            )
            traceback.print_exc()

        if not self._partition_cache_built:
            for proxy in client_proxies:
                self._get_partition_id(proxy)

            known_pids = set(self._cid_to_partition.values())
            known_backdoor = [
                p for p in self.backdoor_partition_ids if p in known_pids
            ]
            known_benign = [
                p for p in self.benign_partition_ids if p in known_pids
            ]

            required_backdoors = min(
                int(self.num_of_malicious_clients_per_round),
                len(self.backdoor_partition_ids),
            )
            need_benign = max(
                1,
                int(self.num_clients * self.fraction_fit) - required_backdoors,
            )

            dba_check = True
            if self._round_attack_type == "dba":
                dba_check = all(
                    pid in known_pids for pid in self.dba_attacker_partition_ids
                )

            if (
                len(known_backdoor) >= required_backdoors
                and len(known_benign) >= need_benign
                and dba_check
            ):
                self._partition_to_proxy = {}
                for cid, proxy in self._all_client_map.items():
                    pid = self._cid_to_partition.get(cid)
                    if pid is not None:
                        self._partition_to_proxy[pid] = proxy

                self._partition_cache_built = True
                print(
                    f"[Strategy] Partition cache READY after round {rnd}: "
                    f"{len(self._partition_to_proxy)} mapped | "
                    f"backdoor partitions found: {sorted(known_backdoor)} "
                    f"(required {required_backdoors})"
                )
            else:
                print(
                    f"[Strategy] Cache building: {len(known_pids)} mapped, "
                    f"{len(known_backdoor)}/{required_backdoors} backdoor, "
                    f"{len(known_benign)}/{need_benign} benign"
                )

        if self._3dfed_coordinator is not None:
            ind_jsons = []
            idx_lists = []
            for _, fit_res in results:
                m = dict(getattr(fit_res, "metrics", {}) or {})
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
                print(
                    f"[3DFed] Stored {len(idx_lists[0])} indicator indices "
                    f"from client"
                )

            if ind_jsons:
                self._3dfed_coordinator.receive_indicator_records(ind_jsons)

                try:
                    prev_nds = parameters_to_ndarrays(self._round_global_parameters)
                    prev_model = get_resnet_cnn_model()
                    set_weights(prev_model, prev_nds)
                    prev_sd = {k: v.clone() for k, v in prev_model.state_dict().items()}

                    cur_nds = parameters_to_ndarrays(aggregated_params)
                    cur_model = get_resnet_cnn_model()
                    set_weights(cur_model, cur_nds)
                    cur_sd = {k: v.clone() for k, v in cur_model.state_dict().items()}

                    t_keys = [
                        n for n, p in cur_model.named_parameters()
                        if p.requires_grad
                    ]

                    self._3dfed_coordinator.read_and_adapt(
                        current_global_sd=cur_sd,
                        previous_global_sd=prev_sd,
                        trainable_keys=t_keys,
                        server_round=rnd,
                    )
                except Exception as e:
                    print(f"[3DFed] Indicator feedback failed: {e}")
                    traceback.print_exc()

        if (
            self._round_attack_type == "dba"
            and aggregated_params is not None
            and abs(self.dba_server_eta - 1.0) > 1e-9
        ):
            old_nds = parameters_to_ndarrays(self._round_global_parameters)
            candidate_nds = parameters_to_ndarrays(aggregated_params)

            updated_nds = []
            fedavg_step_sq = 0.0
            applied_step_sq = 0.0

            for old, candidate in zip(old_nds, candidate_nds):
                old_arr = np.asarray(old)
                cand_arr = np.asarray(candidate)

                if np.issubdtype(old_arr.dtype, np.floating):
                    diff = cand_arr.astype(np.float64) - old_arr.astype(np.float64)
                    updated = old_arr.astype(np.float64) + self.dba_server_eta * diff
                    fedavg_step_sq += float(np.sum(diff * diff))
                    applied_step_sq += float(np.sum((self.dba_server_eta * diff) ** 2))
                    updated_nds.append(updated.astype(old_arr.dtype))
                else:
                    updated_nds.append(cand_arr.copy())

            aggregated_params = ndarrays_to_parameters(updated_nds)

            print(
                f"[DBA][Server] round={rnd} eta={self.dba_server_eta} "
                f"fedavg_step={fedavg_step_sq**0.5:.6f} "
                f"applied_step={applied_step_sq**0.5:.6f}"
            )

        return aggregated_params, aggregated_metrics


    def aggregate_evaluate(self, rnd, results, failures):
        metrics = super().aggregate_evaluate(rnd, results, failures)

        mta_vals = [res.metrics.get("mta", 0.0) for _, res in results]
        asr_vals = [res.metrics.get("asr", 0.0) for _, res in results]

        avg_mta = sum(mta_vals) / len(mta_vals) if mta_vals else 0.0
        avg_asr = sum(asr_vals) / len(asr_vals) if asr_vals else 0.0

        self.history["round"].append(rnd)
        self.history["mta"].append(avg_mta)
        self.history["asr"].append(avg_asr)

        print(f"[Round {rnd}] MTA={avg_mta:.4f}, ASR={avg_asr:.4f}")

        dist_loss = metrics[0] if metrics else None
        append_distributed_round(
            self.simulation_id, rnd, avg_mta, avg_asr, dist_loss, self.num_clients,
        )

        if rnd >= self.num_rounds:
            write_experiment_summary(
                simulation_id=self.simulation_id,
                meta={
                    "aggregation": str(self.aggregation_method),
                    "num_rounds": str(self.num_rounds),
                    "num_malicious_clients": str(self.num_of_malicious_clients),
                    "backdoor_attack_mode": str(self.backdoor_attack_mode),
                    "alpha": self.dirichlet_alpha,
                },
                final_centralized_mta=self.final_centralized_mta or 0.0,
                final_centralized_asr=self.final_centralized_asr or 0.0,
                dist_mta_history=self.history.get("mta", []),
                dist_asr_history=self.history.get("asr", []),
                central_mta_history=self.central_mta_history,
                central_asr_history=self.central_asr_history,
                notes="",
            )

        return metrics

    def record_centralized_eval(self, rnd: int, loss: float, mta: float, asr: float) -> None:
        self.central_mta_history.append(mta)
        self.central_asr_history.append(asr)
        if rnd == self.num_rounds:
            self.final_centralized_mta = mta
            self.final_centralized_asr = asr


    def _extract_and_log_features(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, Any]],
        client_ids: List[str],
        client_proxies: List[ClientProxy],
        scores: Dict[str, float],
        rank_of: Dict[str, int],
    ) -> None:
        if self._param_registry is None:
            print(f"[Extractor][Round {rnd}] No registry, skipping.")
            return

        registry = self._param_registry

        global_params = self._round_global_parameters
        if global_params is None:
            global_params = self.prev_global_parameters
        if global_params is None:
            print(f"[Extractor][Round {rnd}] No global parameters, skipping.")
            return

        global_nds = parameters_to_ndarrays(global_params)
        global_sd_train = nds_to_trainable_state_dict(global_nds, registry.state_dict_keys)

        per_client: List[Dict[str, Any]] = []
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

        reference_delta_per_key: Optional[Dict] = None
        ref_deltas_per_key_list: Optional[List] = None
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
                else:
                    print(
                        f"[Extractor][Round {rnd}] ref_deltas size mismatch: "
                        f"{ref_mean_flat.numel()} != {registry.total_trainable_params}"
                    )
            except Exception as e:
                print(f"[Extractor][Round {rnd}] reference build failed: {e}")

        stage_proj_bases: Optional[Dict] = None
        stage_proj_ref_means: Optional[Dict] = None
        if ref_deltas_per_key_list is not None:
            try:
                stage_proj_bases, stage_proj_ref_means = build_stage_projection_bases(
                    ref_deltas_per_key_list, registry, k=3
                )
            except Exception as e:
                print(f"[Extractor][Round {rnd}] projection basis fit failed: {e}")

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
                "selected_by_aggregator": 1,
                "krum_score": float(scores.get(cid, float("nan"))),
                "krum_rank": int(rank_of.get(cid, -1)),
                "dirichlet_alpha": self.dirichlet_alpha,
                "target_label": 2,
                "aggregation_method": str(self.aggregation_method),
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
