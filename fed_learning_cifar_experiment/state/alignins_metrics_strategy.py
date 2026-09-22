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
from fed_learning_cifar_experiment.threeDFed import AttackerCoordinator
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


class SaveAlignInsMetricsStrategy(SaveKrumMetricsStrategy):

    def __init__(
        self,
        *,
        alignins_sparsity: float = 0.3,
        alignins_lambda_s: float = 1.0,
        alignins_lambda_c: float = 1.0,
        alignins_eps: float = 1e-12,
        alignins_min_keep: int = 1,
        alignins_weighted_average: bool = True,
        alignins_zero_if_empty: bool = True,
        **kwargs: Any,
    ):
        _dba_attacker_partition_ids = kwargs.pop("dba_attacker_partition_ids", [5, 12, 23, 36])
        _dba_server_eta = kwargs.pop("dba_server_eta", 1.0)
        _backdoor_partition_ids = kwargs.pop("backdoor_partition_ids", [])
        _attack_start_round = kwargs.pop("attack_start_round", 1)
        _attack_end_round = kwargs.pop("attack_end_round", 999999)

        super().__init__(**kwargs)

        self.alignins_sparsity = float(alignins_sparsity)
        self.alignins_lambda_s = float(alignins_lambda_s)
        self.alignins_lambda_c = float(alignins_lambda_c)
        self.alignins_eps = float(alignins_eps)
        self.alignins_min_keep = int(alignins_min_keep)
        self.alignins_weighted_average = bool(alignins_weighted_average)
        self.alignins_zero_if_empty = bool(alignins_zero_if_empty)

        self._last_round_malicious_ids: List[str] = []

        self.backdoor_partition_ids = [int(x) for x in _backdoor_partition_ids]
        self._partition_id_to_proxy: Dict[int, ClientProxy] = {}
        self._stratified_cache_built: bool = False

        self.attack_start_round = int(_attack_start_round)
        self.attack_end_round = int(_attack_end_round)

        self.dba_attacker_partition_ids = [int(x) for x in _dba_attacker_partition_ids]
        self.dba_server_eta = float(_dba_server_eta)
        self._dba_partition_cache_built = False
        self._dba_partition_to_proxy: Dict[int, ClientProxy] = {}

        self._3dfed_coordinator: Optional[AttackerCoordinator] = None

        self._param_registry: Optional[ParamRegistry] = None
        self._round_global_parameters: Optional[Parameters] = None
        self._round_ref_deltas_flat: Optional[np.ndarray] = None
        self._round_attack_mode: str = "none"
        self._round_attack_type: str = "none"
        self._round_malicious_ids_set: set = set()


    def _get_partition_id(self, client: ClientProxy) -> int:
        if not hasattr(self, "_cid_to_partition"):
            self._cid_to_partition = {}
        if client.cid in self._cid_to_partition:
            cached = self._cid_to_partition[client.cid]
            if cached >= 0:
                return cached

        try:
            res = client.get_properties(
                GetPropertiesIns(config={}), timeout=10.0, group_id=None
            )
            if res is None:
                raise RuntimeError("get_properties returned None")
            if not hasattr(res, "properties"):
                raise RuntimeError(f"Invalid GetPropertiesRes: {res}")
            if "partition_id" not in res.properties:
                raise KeyError(f"'partition_id' missing in properties: {res.properties}")
            pid = int(res.properties["partition_id"])
        except Exception as e:
            print(f"[AlignIns] _get_partition_id failed for CID={client.cid}: {type(e).__name__}: {e}")
            pid = -1

        self._cid_to_partition[client.cid] = pid
        return pid


    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        num_available = len(client_manager.all())
        sample_size, min_num = self.num_fit_clients(num_available)
        all_clients = list(client_manager.all().values())

        cfg_pre = self.on_fit_config_fn(server_round) if self.on_fit_config_fn else {}
        current_attack_type = str(cfg_pre.get("backdoor-attack-type", "none")).lower()

        dba_sampling_done = False

        if current_attack_type == "dba":
            for proxy in all_clients:
                pid = self._get_partition_id(proxy)
                if pid >= 0:
                    self._dba_partition_to_proxy[pid] = proxy

            required = self.dba_attacker_partition_ids

            if len(required) != 4:
                raise ValueError(
                    f"DBA requires exactly four attacker partitions, got {required}"
                )

            missing = [
                pid for pid in required
                if pid not in self._dba_partition_to_proxy
            ]

            if missing:
                raise RuntimeError(
                    f"[DBA][AlignIns] Unable to resolve fixed attackers: {missing}"
                )

            num_benign = sample_size - len(required)

            if num_benign < 0:
                raise RuntimeError(
                    f"Sample size {sample_size} is smaller than "
                    f"the four required DBA attackers"
                )

            required_set = set(required)

            benign_candidates = [
                proxy for pid, proxy in self._dba_partition_to_proxy.items()
                if pid not in required_set
            ]

            if len(benign_candidates) < num_benign:
                raise RuntimeError(
                    f"[DBA][AlignIns] Need {num_benign} benign clients, "
                    f"but only {len(benign_candidates)} were resolved"
                )

            sampled_backdoor = [
                self._dba_partition_to_proxy[pid] for pid in required
            ]

            sampled_benign = random.sample(benign_candidates, num_benign)

            sampled_clients = sampled_backdoor + sampled_benign
            random.shuffle(sampled_clients)

            malicious_ids = [client.cid for client in sampled_backdoor]

            slot_map = {pid: slot for slot, pid in enumerate(required)}

            dba_sampling_done = True

            print(
                f"[DBA][AlignIns][Coordinator] round={server_round} "
                f"partitions={required} slots=[0,1,2,3]"
            )

        else:
            in_attack_window = (
                self.attack_start_round <= server_round <= self.attack_end_round
            )
            if in_attack_window:
                effective_malicious_per_round = min(
                    int(self.num_of_malicious_clients_per_round),
                    sample_size,
                )
            else:
                effective_malicious_per_round = 0

            if getattr(self, "attacker_selection_mode", "random") == "persistent":
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
                ][: effective_malicious_per_round]

                attacker_cids = set(c.cid for c in attacker_clients)
                benign_candidates = [c for c in all_clients if c.cid not in attacker_cids]
                remaining = max(0, sample_size - len(attacker_clients))
                sampled_benign = random.sample(
                    benign_candidates, min(len(benign_candidates), remaining)
                )
                sampled_clients = attacker_clients + sampled_benign
                malicious_ids = [c.cid for c in attacker_clients]
            else:
                bdp = list(getattr(self, "backdoor_partition_ids", []) or [])

                if effective_malicious_per_round > 0 and bdp:
                    for proxy in all_clients:
                        pid = self._get_partition_id(proxy)
                        if pid >= 0:
                            self._partition_id_to_proxy[pid] = proxy

                    backdoor_set = set(int(x) for x in bdp)
                    backdoor_proxies = [
                        self._partition_id_to_proxy[p]
                        for p in self.backdoor_partition_ids
                        if p in self._partition_id_to_proxy
                    ]
                    if effective_malicious_per_round > len(backdoor_proxies):
                        mapped = sorted([
                            p for p in self.backdoor_partition_ids
                            if p in self._partition_id_to_proxy
                        ])
                        raise RuntimeError(
                            f"[Round {server_round}][AlignIns] Requested {effective_malicious_per_round} "
                            f"malicious clients, but only {len(backdoor_proxies)} backdoor "
                            f"proxies resolved. Mapped: {mapped}"
                        )
                    sampled_backdoor = random.sample(backdoor_proxies, effective_malicious_per_round)

                    benign_proxies = [
                        proxy for pid, proxy in self._partition_id_to_proxy.items()
                        if pid not in backdoor_set
                    ]
                    num_benign = sample_size - len(sampled_backdoor)
                    sampled_benign = random.sample(
                        benign_proxies, min(len(benign_proxies), num_benign)
                    )

                    sampled_clients = sampled_backdoor + sampled_benign
                    random.shuffle(sampled_clients)
                    malicious_ids = [c.cid for c in sampled_backdoor]

                    mal_parts = [
                        self._cid_to_partition.get(c.cid, -1)
                        for c in sampled_backdoor
                    ]
                    print(
                        f"[Round {server_round}][AlignIns] STRATIFIED | "
                        f"Malicious partitions: {mal_parts}"
                    )

                elif effective_malicious_per_round > 0:
                    sampled_clients = list(client_manager.sample(sample_size, min_num))
                    sampled_ids_tmp = [c.cid for c in sampled_clients]
                    malicious_ids = random.sample(
                        sampled_ids_tmp, effective_malicious_per_round
                    )

                else:
                    sampled_clients = list(
                        client_manager.sample(sample_size, min_num)
                    )
                    malicious_ids = []

            print(
                f"[Round {server_round}][AlignIns] Attack window: "
                f"[{self.attack_start_round}, {self.attack_end_round}] | "
                f"in_window={in_attack_window} | "
                f"malicious_count={len(malicious_ids)}"
            )

        sampled_ids = [c.cid for c in sampled_clients]

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        nds = parameters_to_ndarrays(parameters)

        model_tmp = get_resnet_cnn_model()
        set_weights(model_tmp, nds)
        model_tmp.to(device)
        init_vec = parameters_to_vector(model_tmp.parameters()).detach().cpu()

        ref_partition_ids = random.sample(range(int(self.num_clients)), 6)
        ref_deltas = []
        for pid in ref_partition_ids:
            train_loader, _ = load_data(
                partition_id=pid,
                num_partitions=int(self.num_clients),
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

        self._last_round_malicious_ids = list(map(str, malicious_ids))
        self._round_malicious_ids_set = set(str(x) for x in malicious_ids)

        fit_ins_list: List[Tuple[ClientProxy, FitIns]] = []
        print(f"Sampled clients for round {server_round}: {sampled_ids}")
        print(f"Malicious clients for round {server_round}: {malicious_ids}")

        if hasattr(self, "_3dfed_last_assigned_round") and self._3dfed_last_assigned_round != server_round:
            self._3dfed_round_assignments = []

        for client in sampled_clients:
            config = self.on_fit_config_fn(server_round) if self.on_fit_config_fn else {}
            config.update(
                {
                    "current-round": server_round,
                    "sampled_client_ids": json.dumps(sampled_ids),
                    "malicious_client_ids": json.dumps(malicious_ids),
                    "is_malicious": str(client.cid in malicious_ids),
                    "shared_ref_deltas": json.dumps(ref_deltas_np.tolist()),
                    "shared_ref_median_norm": median_norm,
                }
            )

            if getattr(self, "attacker_selection_mode", "random") == "persistent":
                config["krum_selected_cid"] = getattr(self, "last_krum_selected_cid", None)
                last_delta = getattr(self, "last_krum_selected_delta", None)
                config["krum_ref_delta"] = (
                    json.dumps(last_delta.tolist()) if last_delta is not None else None
                )

            if getattr(self, "prev_global_parameters", None) is not None:
                config["prev_global_tensors_hex"] = json.dumps(
                    [t.hex() for t in self.prev_global_parameters.tensors]
                )
                config["prev_global_tensor_type"] = self.prev_global_parameters.tensor_type
            else:
                config["prev_global_tensors_hex"] = "[]"
                config["prev_global_tensor_type"] = "numpy.ndarray"

            attack_type = str(config.get("backdoor-attack-type", "none")).lower()
            if attack_type == "dba" and client.cid in malicious_ids and dba_sampling_done:
                client_pid = self._get_partition_id(client)
                if client_pid is not None and client_pid in slot_map:
                    config["dba-attacker-index"] = slot_map[int(client_pid)]

            attack_type = str(config.get("backdoor-attack-type", "none")).lower()
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

                if not hasattr(self, "_3dfed_last_assigned_round") or self._3dfed_last_assigned_round != server_round:
                    self._3dfed_round_assignments = self._3dfed_coordinator.assign_roles(
                        malicious_ids, server_round
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
                f"{self._param_registry.total_trainable_params} elements"
            )

        self._round_global_parameters = parameters
        self._round_ref_deltas_flat = ref_deltas_np

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

    @staticmethod
    def _flatten_float_layers(nds: List[np.ndarray]) -> np.ndarray:
        parts = []
        for arr in nds:
            a = np.asarray(arr)
            if np.issubdtype(a.dtype, np.floating):
                parts.append(a.astype(np.float64, copy=False).ravel())
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float64)

    @staticmethod
    def _arrays_subtract(a: List[np.ndarray], b: List[np.ndarray]) -> List[np.ndarray]:
        out = []
        for ai, bi in zip(a, b):
            aa = np.asarray(ai)
            bb = np.asarray(bi)
            if np.issubdtype(aa.dtype, np.floating):
                out.append(aa.astype(np.float64, copy=False) - bb.astype(np.float64, copy=False))
            else:
                out.append(np.zeros_like(aa))
        return out

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom <= eps:
            return 0.0
        return float(np.dot(a, b) / max(denom, eps))

    def _compute_alignins_filter(
        self,
        update_matrix: np.ndarray,
        global_flat: np.ndarray,
    ) -> Tuple[List[int], Dict[str, Any]]:
        n, d = update_matrix.shape
        if n == 0:
            return [], {}
        if d == 0:
            return list(range(n)), {
                "tda": [0.0 for _ in range(n)],
                "mpsa": [0.0 for _ in range(n)],
                "mzscore_tda": [0.0 for _ in range(n)],
                "mzscore_mpsa": [0.0 for _ in range(n)],
            }

        major_sign = np.sign(np.sum(np.sign(update_matrix), axis=0))

        k_sig = max(1, int(d * self.alignins_sparsity))
        k_sig = min(k_sig, d)

        tda_list: List[float] = []
        mpsa_list: List[float] = []
        for i in range(n):
            upd = update_matrix[i]
            if k_sig >= d:
                init_indices = np.arange(d)
            else:
                init_indices = np.argpartition(np.abs(upd), -k_sig)[-k_sig:]

            signs_match = np.sign(upd[init_indices]) == major_sign[init_indices]
            mpsa_list.append(float(np.sum(signs_match) / max(1, len(init_indices))))
            tda_list.append(self._cosine(upd, global_flat, eps=self.alignins_eps))

        mpsa_arr = np.asarray(mpsa_list, dtype=np.float64)
        tda_arr = np.asarray(tda_list, dtype=np.float64)

        mpsa_std = float(np.std(mpsa_arr))
        tda_std = float(np.std(tda_arr))
        mpsa_med = float(np.median(mpsa_arr))
        tda_med = float(np.median(tda_arr))

        mz_mpsa = np.abs(mpsa_arr - mpsa_med) / max(mpsa_std, self.alignins_eps)
        mz_tda = np.abs(tda_arr - tda_med) / max(tda_std, self.alignins_eps)

        selected = [
            i for i in range(n)
            if mz_mpsa[i] < self.alignins_lambda_s and mz_tda[i] < self.alignins_lambda_c
        ]

        if len(selected) == 0 and not self.alignins_zero_if_empty:
            combined = mz_mpsa + mz_tda
            k_keep = max(1, min(self.alignins_min_keep, n))
            selected = np.argsort(combined)[:k_keep].tolist()

        diagnostics = {
            "tda": tda_list,
            "mpsa": mpsa_list,
            "mzscore_tda": mz_tda.tolist(),
            "mzscore_mpsa": mz_mpsa.tolist(),
            "tda_median": tda_med,
            "mpsa_median": mpsa_med,
            "tda_std": tda_std,
            "mpsa_std": mpsa_std,
            "k_significant": k_sig,
        }
        return selected, diagnostics

    def _clip_update_nds(
        self,
        update_nds_list: List[List[np.ndarray]],
        clip_bound: float,
    ) -> Tuple[List[List[np.ndarray]], List[float], List[float]]:
        clipped = []
        norms = []
        scales = []
        for upd in update_nds_list:
            flat = self._flatten_float_layers(upd)
            norm = float(np.linalg.norm(flat))
            scale = 1.0 if norm <= self.alignins_eps else min(1.0, float(clip_bound) / norm)
            norms.append(norm)
            scales.append(scale)
            clipped.append([
                (np.asarray(layer).astype(np.float64, copy=False) * scale)
                if np.issubdtype(np.asarray(layer).dtype, np.floating)
                else np.zeros_like(np.asarray(layer), dtype=np.float64)
                for layer in upd
            ])
        return clipped, norms, scales

    def _aggregate_selected_clipped_updates(
        self,
        global_nds: List[np.ndarray],
        clipped_updates: List[List[np.ndarray]],
        selected_weights: List[int],
    ) -> List[np.ndarray]:
        if not clipped_updates:
            return [np.asarray(arr).copy() for arr in global_nds]

        if self.alignins_weighted_average and selected_weights:
            weights = np.asarray(selected_weights, dtype=np.float64)
            if float(weights.sum()) <= 0:
                weights = np.ones(len(clipped_updates), dtype=np.float64)
        else:
            weights = np.ones(len(clipped_updates), dtype=np.float64)
        weights = weights / max(float(weights.sum()), self.alignins_eps)

        agg_nds: List[np.ndarray] = []
        for layer_idx, base in enumerate(global_nds):
            base_arr = np.asarray(base)
            if self._is_int_or_bool_arr(base_arr):
                agg_nds.append(base_arr.copy())
                continue

            acc = np.zeros(base_arr.shape, dtype=np.float64)
            for w, upd in zip(weights, clipped_updates):
                acc += float(w) * np.asarray(upd[layer_idx]).astype(np.float64, copy=False)

            new_layer = base_arr.astype(np.float64, copy=False) + acc
            agg_nds.append(new_layer.astype(base_arr.dtype, copy=False))

        return agg_nds


    def aggregate_fit(self, rnd: int, results, failures):
        if not results:
            return super().aggregate_fit(rnd, results, failures)
        if failures:
            print(
                f"[Round {rnd}][AlignIns] Received {len(failures)} failures; "
                f"continuing with {len(results)} successful results."
            )

        global_nds = parameters_to_ndarrays(self._round_global_parameters or self.prev_global_parameters)
        global_flat = self._flatten_float_layers(global_nds)

        client_ids: List[str] = []
        client_params_nds: List[List[np.ndarray]] = []
        client_update_nds: List[List[np.ndarray]] = []
        client_updates_flat: List[np.ndarray] = []
        client_proxies: List[ClientProxy] = []
        client_num_examples: List[int] = []

        for client_proxy, fit_res in results:
            nds = [np.asarray(p) for p in parameters_to_ndarrays(fit_res.parameters)]
            upd = self._arrays_subtract(nds, global_nds)
            flat_upd = self._flatten_float_layers(upd)
            client_update_nds.append(upd)
            client_updates_flat.append(flat_upd)
            client_params_nds.append(nds)
            client_ids.append(str(client_proxy.cid))
            client_proxies.append(client_proxy)
            client_num_examples.append(int(getattr(fit_res, "num_examples", 1)))

        X_updates = np.stack(client_updates_flat)
        n = len(client_ids)

        selected_idx, diag = self._compute_alignins_filter(X_updates, global_flat)
        selected_cids = {client_ids[i] for i in selected_idx}

        selected_norms = [float(np.linalg.norm(client_updates_flat[i])) for i in selected_idx]
        if selected_norms:
            norm_clip = float(np.median(selected_norms))
        else:
            norm_clip = 0.0

        all_clipped_updates, raw_norms, clip_scales = self._clip_update_nds(
            client_update_nds, norm_clip
        )
        selected_clipped_updates = [all_clipped_updates[i] for i in selected_idx]
        selected_weights = [client_num_examples[i] for i in selected_idx]

        agg_nds = self._aggregate_selected_clipped_updates(
            global_nds, selected_clipped_updates, selected_weights
        )
        new_parameters = ndarrays_to_parameters(agg_nds)

        mz_tda = np.asarray(diag.get("mzscore_tda", [0.0] * n), dtype=np.float64)
        mz_mpsa = np.asarray(diag.get("mzscore_mpsa", [0.0] * n), dtype=np.float64)
        combined_score = mz_tda + mz_mpsa
        scores: Dict[str, float] = {
            client_ids[i]: float(combined_score[i]) for i in range(n)
        }

        canonical_idx = int(np.argmin(combined_score)) if n > 0 else 0
        canonical_cid = client_ids[canonical_idx] if n > 0 else None

        attacker_selected = any(str(cid) in set(self._last_round_malicious_ids) for cid in selected_cids)
        print(f"\n[Round {rnd}][AlignIns]")
        print(
            f"  selected={sorted(selected_cids)} | kept={len(selected_idx)}/{n} | "
            f"attacker_selected={attacker_selected} | canonical={canonical_cid}"
        )
        print(
            f"  sparsity={self.alignins_sparsity:.4f} | "
            f"lambda_s={self.alignins_lambda_s:.4f} | lambda_c={self.alignins_lambda_c:.4f} | "
            f"norm_clip={norm_clip:.6e} | k_sig={diag.get('k_significant', -1)}"
        )
        for i, cid in enumerate(client_ids):
            proxy = client_proxies[i]
            pid = self._get_partition_id(proxy)
            is_mal = str(cid) in set(self._last_round_malicious_ids)
            print(
                f"  CID={cid:>6} | Partition={pid:>3} | "
                f"Selected={int(cid in selected_cids)} | "
                f"TDA={diag.get('tda', [0]*n)[i]: .6f} | "
                f"MPSA={diag.get('mpsa', [0]*n)[i]: .6f} | "
                f"MZ_TDA={mz_tda[i]:.4f} | MZ_MPSA={mz_mpsa[i]:.4f} | "
                f"Score={scores[cid]:.6e} | Malicious={is_mal}"
            )

        self.last_krum_selected_cid = canonical_cid
        if getattr(self, "prev_global_parameters", None) is not None and canonical_cid is not None:
            prev_nds = parameters_to_ndarrays(self.prev_global_parameters)
            prev_flat = np.concatenate([np.asarray(p).ravel() for p in prev_nds])
            curr_flat = np.concatenate([np.asarray(p).ravel() for p in client_params_nds[canonical_idx]])
            self.last_krum_selected_delta = curr_flat - prev_flat
        else:
            self.last_krum_selected_delta = None

        try:
            self._extract_and_log_features(
                rnd=rnd,
                results=results,
                client_ids=client_ids,
                client_proxies=client_proxies,
                scores=scores,
                selected_cids=selected_cids,
                alignins_diag=diag,
            )
        except Exception as e:
            print(
                f"[Extractor][Round {rnd}] Feature extraction failed: "
                f"{type(e).__name__}: {e}"
            )
            traceback.print_exc()

        self._handle_3dfed_feedback(rnd, results, new_parameters)

        if (
            self._round_attack_type == "dba"
            and new_parameters is not None
            and abs(self.dba_server_eta - 1.0) > 1e-9
        ):
            old_nds = parameters_to_ndarrays(self._round_global_parameters)
            candidate_nds = parameters_to_ndarrays(new_parameters)

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

            new_parameters = ndarrays_to_parameters(updated_nds)

            print(
                f"[DBA][AlignIns][Server] round={rnd} eta={self.dba_server_eta} "
                f"fedavg_step={fedavg_step_sq**0.5:.6f} "
                f"applied_step={applied_step_sq**0.5:.6f}"
            )

        return new_parameters, {}


    def _handle_3dfed_feedback(self, rnd: int, results, aggregated_params: Parameters) -> None:
        if self._3dfed_coordinator is None:
            return

        ind_jsons: List[str] = []
        idx_lists: List[List[int]] = []
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
                f"[3DFed][AlignIns] Stored {len(idx_lists[0])} indicator indices from client"
            )

        if not ind_jsons:
            return

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

            t_keys = [n for n, p in cur_model.named_parameters() if p.requires_grad]
            self._3dfed_coordinator.read_and_adapt(
                current_global_sd=cur_sd,
                previous_global_sd=prev_sd,
                trainable_keys=t_keys,
                server_round=rnd,
            )
        except Exception as e:
            print(f"[3DFed][AlignIns] Indicator feedback failed: {e}")
            traceback.print_exc()


    def _extract_and_log_features(
        self,
        rnd: int,
        results,
        client_ids: List[str],
        client_proxies: List,
        scores: Dict[str, float],
        selected_cids: set,
        alignins_diag: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self._param_registry is None:
            return

        registry = self._param_registry
        global_params = self._round_global_parameters or self.prev_global_parameters
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

        tda_of = {}
        mpsa_of = {}
        mz_tda_of = {}
        mz_mpsa_of = {}
        if alignins_diag:
            for i, cid in enumerate(client_ids):
                tda_of[str(cid)] = float(alignins_diag.get("tda", [0.0] * len(client_ids))[i])
                mpsa_of[str(cid)] = float(alignins_diag.get("mpsa", [0.0] * len(client_ids))[i])
                mz_tda_of[str(cid)] = float(alignins_diag.get("mzscore_tda", [0.0] * len(client_ids))[i])
                mz_mpsa_of[str(cid)] = float(alignins_diag.get("mzscore_mpsa", [0.0] * len(client_ids))[i])

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
                "aggregation_method": "alignins",
                "num_clients": int(self.num_clients),
                "alignins_tda": float(tda_of.get(str(cid), 0.0)),
                "alignins_mpsa": float(mpsa_of.get(str(cid), 0.0)),
                "alignins_mz_tda": float(mz_tda_of.get(str(cid), 0.0)),
                "alignins_mz_mpsa": float(mz_mpsa_of.get(str(cid), 0.0)),
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
