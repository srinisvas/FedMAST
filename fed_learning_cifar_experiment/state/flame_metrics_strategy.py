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


class SaveFLAMEMetricsStrategy(SaveKrumMetricsStrategy):

    def __init__(
        self,
        *,
        flame_min_cluster_size: Optional[int] = None,
        flame_min_samples: Optional[int] = None,
        flame_allow_fallback_dbscan: bool = True,
        flame_dbscan_eps: float = 0.5,
        flame_clip_bound: Optional[float] = None,
        flame_noise_multiplier: float = 0.001,
        flame_noise_mode: str = "coordinate",
        flame_random_seed: int = 42,
        **kwargs: Any,
    ):
        _dba_attacker_partition_ids = kwargs.pop("dba_attacker_partition_ids", [5, 12, 23, 36])
        _dba_server_eta = kwargs.pop("dba_server_eta", 1.0)
        _backdoor_partition_ids = kwargs.pop("backdoor_partition_ids", None)
        _attack_start_round = kwargs.pop("attack_start_round", 1)
        _attack_end_round = kwargs.pop("attack_end_round", 999999)

        super().__init__(**kwargs)

        self.flame_min_cluster_size = flame_min_cluster_size
        self.flame_min_samples = flame_min_samples
        self.flame_allow_fallback_dbscan = bool(flame_allow_fallback_dbscan)
        self.flame_dbscan_eps = float(flame_dbscan_eps)
        self.flame_clip_bound = None if flame_clip_bound is None else float(flame_clip_bound)
        self.flame_noise_multiplier = float(flame_noise_multiplier)
        self.flame_noise_mode = str(flame_noise_mode).lower()
        self.flame_random_seed = int(flame_random_seed)

        self._last_round_malicious_ids: List[str] = []

        self.dba_attacker_partition_ids = [int(x) for x in _dba_attacker_partition_ids]
        self.dba_server_eta = float(_dba_server_eta)
        self._dba_partition_to_proxy: Dict[int, ClientProxy] = {}

        self.backdoor_partition_ids = (
            [int(x) for x in _backdoor_partition_ids] if _backdoor_partition_ids else []
        )
        self.attack_start_round = int(_attack_start_round)
        self.attack_end_round = int(_attack_end_round)

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
            print(f"[FLAME] _get_partition_id failed for CID={client.cid}: {type(e).__name__}: {e}")
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
                raise ValueError(f"DBA requires exactly four attacker partitions, got {required}")

            missing = [pid for pid in required if pid not in self._dba_partition_to_proxy]
            if missing:
                raise RuntimeError(f"[DBA][FLAME] Unable to resolve fixed attackers: {missing}")

            num_benign = sample_size - len(required)
            required_set = set(required)
            benign_candidates = [
                proxy for pid, proxy in self._dba_partition_to_proxy.items()
                if pid not in required_set
            ]

            sampled_backdoor = [self._dba_partition_to_proxy[pid] for pid in required]
            sampled_benign = random.sample(benign_candidates, num_benign)
            sampled_clients = sampled_backdoor + sampled_benign
            random.shuffle(sampled_clients)
            malicious_ids = [client.cid for client in sampled_backdoor]
            slot_map = {pid: slot for slot, pid in enumerate(required)}
            dba_sampling_done = True

            print(
                f"[DBA][FLAME][Coordinator] round={server_round} "
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
                    if not hasattr(self, "_cid_to_partition"):
                        self._cid_to_partition = {}
                    partition_to_proxy = {}
                    for proxy in all_clients:
                        pid = self._get_partition_id(proxy)
                        if pid >= 0:
                            partition_to_proxy[pid] = proxy

                    bdp_set = set(int(x) for x in bdp)

                    available_attackers = [
                        partition_to_proxy[pid]
                        for pid in bdp
                        if pid in partition_to_proxy
                    ]
                    if len(available_attackers) < effective_malicious_per_round:
                        raise RuntimeError(
                            f"[FLAME] Not enough fixed backdoor partitions: "
                            f"need={effective_malicious_per_round}, "
                            f"have={len(available_attackers)}"
                        )
                    sampled_backdoor = random.sample(
                        available_attackers, effective_malicious_per_round
                    )
                    attacker_cids = {c.cid for c in sampled_backdoor}

                    benign_candidates = [
                        proxy for proxy in all_clients
                        if self._get_partition_id(proxy) not in bdp_set
                        and proxy.cid not in attacker_cids
                    ]
                    num_benign = sample_size - len(sampled_backdoor)
                    sampled_benign = random.sample(benign_candidates, num_benign)

                    sampled_clients = sampled_backdoor + sampled_benign
                    random.shuffle(sampled_clients)
                    malicious_ids = [c.cid for c in sampled_backdoor]

                elif effective_malicious_per_round > 0:
                    sampled_clients = list(
                        client_manager.sample(sample_size, min_num)
                    )
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
                f"[Round {server_round}][FLAME] Attack window: "
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

    def _ensure_trainable_indices(self, nds_len: int) -> set:
        if hasattr(self, "_flame_trainable_indices"):
            return self._flame_trainable_indices

        model = get_resnet_cnn_model()
        state_keys = list(model.state_dict().keys())
        trainable_names = {
            name for name, p in model.named_parameters() if p.requires_grad
        }
        param_keys = [
            name for name, p in model.named_parameters() if p.requires_grad
        ]

        if len(state_keys) == nds_len:
            self._flame_trainable_indices = {
                i for i, key in enumerate(state_keys) if key in trainable_names
            }
            protected = [
                key
                for i, key in enumerate(state_keys)
                if i not in self._flame_trainable_indices
            ]
            print(
                f"[FLAME] Paper-faithful parameter map: "
                f"{len(self._flame_trainable_indices)} trainable tensors; "
                f"protected buffers={protected}"
            )
        elif len(param_keys) == nds_len:
            self._flame_trainable_indices = set(range(nds_len))
            print(
                f"[FLAME] Parameter-only ndarray list detected: "
                f"{nds_len} trainable tensors, no buffers present."
            )
        else:
            raise RuntimeError(
                f"[FLAME] Cannot map ndarray list to model parameters safely: "
                f"nds_len={nds_len}, state_dict_len={len(state_keys)}, "
                f"named_param_len={len(param_keys)}"
            )

        return self._flame_trainable_indices

    @staticmethod
    def _flatten_float_layers(nds: List[np.ndarray]) -> np.ndarray:
        parts = []
        for arr in nds:
            a = np.asarray(arr)
            if np.issubdtype(a.dtype, np.floating):
                parts.append(a.astype(np.float64, copy=False).ravel())
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float64)

    def _flatten_trainable_layers(
        self,
        nds: List[np.ndarray],
        trainable_indices: set,
    ) -> np.ndarray:
        parts = []
        for i, arr in enumerate(nds):
            if i not in trainable_indices:
                continue
            a = np.asarray(arr)
            if np.issubdtype(a.dtype, np.floating):
                parts.append(a.astype(np.float64, copy=False).ravel())
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float64)

    @staticmethod
    def _cosine_distance_matrix(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        Xn = X / np.maximum(norms, eps)
        sim = np.clip(Xn @ Xn.T, -1.0, 1.0)
        dist = 1.0 - sim
        np.fill_diagonal(dist, 0.0)
        return dist

    def _cluster_updates(self, update_matrix: np.ndarray) -> Tuple[np.ndarray, str]:
        n = update_matrix.shape[0]
        if n <= 2:
            return np.zeros(n, dtype=int), "single_cluster_n<=2"

        dist = self._cosine_distance_matrix(update_matrix)
        min_cluster_size = self.flame_min_cluster_size
        if min_cluster_size is None:
            min_cluster_size = max(2, n // 2 + 1)
        min_samples = self.flame_min_samples
        if min_samples is None:
            min_samples = 1

        try:
            import hdbscan
            clusterer = hdbscan.HDBSCAN(
                metric="precomputed",
                min_cluster_size=int(min_cluster_size),
                min_samples=int(min_samples),
                allow_single_cluster=True,
            )
            labels = clusterer.fit_predict(dist).astype(int)
            return labels, "hdbscan"
        except Exception as e:
            if not self.flame_allow_fallback_dbscan:
                raise RuntimeError(
                    "FLAME paper-faithful clustering requires hdbscan. "
                    "Install hdbscan or set flame_allow_fallback_dbscan=True for smoke tests."
                ) from e
            try:
                from sklearn.cluster import DBSCAN
                labels = DBSCAN(
                    eps=float(self.flame_dbscan_eps),
                    min_samples=max(1, int(min_samples)),
                    metric="precomputed",
                ).fit_predict(dist).astype(int)
                if np.all(labels == -1):
                    labels = np.zeros(n, dtype=int)
                return labels, "dbscan_fallback"
            except Exception:
                print("[FLAME] Clustering failed; falling back to single cluster")
                traceback.print_exc()
                return np.zeros(n, dtype=int), "single_cluster_fallback"

    @staticmethod
    def _largest_non_noise_cluster(labels: np.ndarray) -> np.ndarray:
        unique = [int(x) for x in sorted(set(labels.tolist())) if int(x) != -1]
        if not unique:
            return np.arange(len(labels))
        counts = {u: int(np.sum(labels == u)) for u in unique}
        best = max(counts, key=counts.get)
        selected = np.where(labels == best)[0]
        return selected if len(selected) > 0 else np.arange(len(labels))

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

    def _clip_selected_updates(
        self,
        selected_updates: List[List[np.ndarray]],
        clip_bound: float,
        trainable_indices: set,
    ) -> Tuple[List[List[np.ndarray]], List[float], List[float]]:
        clipped = []
        norms = []
        scales = []
        for upd in selected_updates:
            flat = self._flatten_trainable_layers(upd, trainable_indices)
            norm = float(np.linalg.norm(flat))
            scale = 1.0 if norm <= 1e-12 else min(1.0, float(clip_bound) / norm)
            norms.append(norm)
            scales.append(scale)
            clipped_upd: List[np.ndarray] = []
            for layer_idx, layer in enumerate(upd):
                arr = np.asarray(layer)
                if self._is_int_or_bool_arr(arr):
                    clipped_upd.append(np.zeros_like(arr, dtype=np.float64))
                elif layer_idx in trainable_indices:
                    clipped_upd.append(arr.astype(np.float64, copy=False) * scale)
                else:
                    clipped_upd.append(arr.astype(np.float64, copy=False))
            clipped.append(clipped_upd)
        return clipped, norms, scales

    def _aggregate_clipped_updates(
        self,
        global_nds: List[np.ndarray],
        clipped_updates: List[List[np.ndarray]],
        clip_bound: float,
        rnd: int,
    ) -> List[np.ndarray]:
        k = max(1, len(clipped_updates))
        rng = np.random.default_rng(self.flame_random_seed + int(rnd))

        total_float_params = int(sum(
            np.asarray(arr).size for arr in global_nds
            if np.issubdtype(np.asarray(arr).dtype, np.floating)
        ))
        if self.flame_noise_mode == "coordinate":
            noise_std = float(self.flame_noise_multiplier) * float(clip_bound)
        else:
            noise_std = (
                float(self.flame_noise_multiplier) * float(clip_bound)
                / np.sqrt(max(1, total_float_params))
            )

        trainable_indices = self._ensure_trainable_indices(len(global_nds))

        agg_nds: List[np.ndarray] = []
        for layer_idx, base in enumerate(global_nds):
            base_arr = np.asarray(base)
            if self._is_int_or_bool_arr(base_arr):
                agg_nds.append(base_arr.copy())
                continue

            acc = np.zeros(base_arr.shape, dtype=np.float64)
            for upd in clipped_updates:
                acc += np.asarray(upd[layer_idx]).astype(np.float64, copy=False)
            acc /= float(k)

            new_layer = base_arr.astype(np.float64, copy=False) + acc

            if noise_std > 0 and layer_idx in trainable_indices:
                new_layer = new_layer + rng.normal(0.0, noise_std, size=new_layer.shape)

            agg_nds.append(new_layer.astype(base_arr.dtype, copy=False))

        return agg_nds


    def aggregate_fit(self, rnd: int, results, failures):
        if not results:
            return super().aggregate_fit(rnd, results, failures)
        if failures:
            print(
                f"[Round {rnd}][FLAME] Received {len(failures)} failures; "
                f"continuing with {len(results)} successful results."
            )

        global_nds = parameters_to_ndarrays(self._round_global_parameters or self.prev_global_parameters)
        trainable_indices = self._ensure_trainable_indices(len(global_nds))

        client_ids: List[str] = []
        client_params_nds: List[List[np.ndarray]] = []
        client_update_nds: List[List[np.ndarray]] = []
        client_updates_flat: List[np.ndarray] = []
        client_proxies: List[ClientProxy] = []

        for client_proxy, fit_res in results:
            nds = [np.asarray(p) for p in parameters_to_ndarrays(fit_res.parameters)]
            upd = self._arrays_subtract(nds, global_nds)
            flat_upd = self._flatten_trainable_layers(upd, trainable_indices)
            client_update_nds.append(upd)
            client_updates_flat.append(flat_upd)
            client_params_nds.append(nds)
            client_ids.append(str(client_proxy.cid))
            client_proxies.append(client_proxy)

        X_updates = np.stack(client_updates_flat)
        n = len(client_ids)

        labels, cluster_method = self._cluster_updates(X_updates)
        selected_idx = self._largest_non_noise_cluster(labels)
        selected_cids = {client_ids[i] for i in selected_idx}

        all_norms = [float(np.linalg.norm(flat)) for flat in client_updates_flat]
        selected_updates = [client_update_nds[i] for i in selected_idx]
        if self.flame_clip_bound is not None:
            clip_bound = float(self.flame_clip_bound)
        else:
            clip_bound = float(np.median(all_norms)) if all_norms else 0.0

        clipped_updates, raw_norms, clip_scales = self._clip_selected_updates(
            selected_updates, clip_bound, trainable_indices
        )

        agg_nds = self._aggregate_clipped_updates(
            global_nds, clipped_updates, clip_bound=clip_bound, rnd=rnd
        )
        new_parameters = ndarrays_to_parameters(agg_nds)

        agg_update_flat = self._flatten_trainable_layers(
            self._arrays_subtract(agg_nds, global_nds), trainable_indices
        )
        scores: Dict[str, float] = {
            client_ids[i]: float(np.linalg.norm(X_updates[i] - agg_update_flat))
            for i in range(n)
        }
        canonical_idx = int(np.argmin([scores[cid] for cid in client_ids]))
        canonical_cid = client_ids[canonical_idx]

        cluster_counts = {int(l): int(np.sum(labels == l)) for l in sorted(set(labels.tolist()))}
        attacker_selected = any(str(cid) in set(self._last_round_malicious_ids) for cid in selected_cids)
        _total_fp = int(sum(
            np.asarray(a).size for a in global_nds
            if np.issubdtype(np.asarray(a).dtype, np.floating)
        ))
        if self.flame_noise_mode == "coordinate":
            _noise_std_log = float(self.flame_noise_multiplier) * float(clip_bound)
        else:
            _noise_std_log = (
                float(self.flame_noise_multiplier) * float(clip_bound)
                / np.sqrt(max(1, _total_fp))
            )
        _eff_mcs = self.flame_min_cluster_size if self.flame_min_cluster_size is not None else max(2, n // 2 + 1)
        print(f"\n[Round {rnd}][FLAME]")
        print(
            f"  clustering={cluster_method} | min_cluster_size={_eff_mcs} | "
            f"labels={cluster_counts} | "
            f"selected={sorted(selected_cids)} | attacker_selected={attacker_selected}"
        )
        print(
            f"  clip_bound(all_median)={clip_bound:.6e} | "
            f"noise_std={_noise_std_log:.6e} | noise_mode={self.flame_noise_mode} | "
            f"canonical={canonical_cid}"
        )
        for i, cid in enumerate(client_ids):
            proxy = client_proxies[i]
            pid = self._get_partition_id(proxy)
            is_mal = str(cid) in set(self._last_round_malicious_ids)
            print(
                f"  CID={cid:>6} | Partition={pid:>3} | "
                f"Cluster={int(labels[i]):>3} | Selected={int(cid in selected_cids)} | "
                f"Norm={all_norms[i]:.6e} | Dist={scores[cid]:.6e} | Malicious={is_mal}"
            )

        self.last_krum_selected_cid = canonical_cid
        if getattr(self, "prev_global_parameters", None) is not None:
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
                cluster_labels=labels,
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
                f"[DBA][FLAME][Server] round={rnd} eta={self.dba_server_eta} "
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
                f"[3DFed][FLAME] Stored {len(idx_lists[0])} indicator indices from client"
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
            print(f"[3DFed][FLAME] Indicator feedback failed: {e}")
            traceback.print_exc()


    def _extract_and_log_features(
        self,
        rnd: int,
        results,
        client_ids: List[str],
        client_proxies: List,
        scores: Dict[str, float],
        selected_cids: set,
        cluster_labels: Optional[np.ndarray] = None,
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
        label_of = {client_ids[i]: int(cluster_labels[i]) for i in range(len(client_ids))} if cluster_labels is not None else {}

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
                "aggregation_method": "flame",
                "num_clients": int(self.num_clients),
                "flame_cluster": int(label_of.get(str(cid), -999)),
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
