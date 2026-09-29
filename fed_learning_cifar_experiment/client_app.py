import json
import random

import torch

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from flwr.common import Parameters, parameters_to_ndarrays

from fed_learning_cifar_experiment.task import (
    get_weights, load_data, set_weights, test, train, get_resnet_cnn_model,
    train_constrain_and_scale_for_fedavg_tm,
    train_neurotoxin,
    train_lga,
    train_bc_layers_lp_full,
    train_bc_layers_lp_full_fedmast,
    train_dba,
    evaluate_dba_asr,
)
from fed_learning_cifar_experiment.threeDFed import train_3dfed
from fed_learning_cifar_experiment.utils.evaluate_attack import evaluate_asr


class FlowerClient(NumPyClient):
    def __init__(self, net, local_epochs, context: Context):
        self.test_set = None
        self.training_set = None
        self.net = net
        self.context = context
        self.local_epochs = local_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
        self.partition_id = str(self.context.node_config.get("partition-id"))
        self.prev_global_vec = None
        self.alpha = float(self.context.run_config.get("dirichlet-alpha", 0.3))

    def get_properties(self, config):
        return {"partition_id": str(self.context.node_config["partition-id"])}

    def fit(self, parameters, config):
        set_weights(self.net, parameters)
        init_state = {k: v.cpu().clone() for k, v in self.net.state_dict().items()}
        init_vec = parameters_to_vector(self.net.parameters()).detach().cpu().clone()

        prev_global_vec = None
        tensors_hex = json.loads(config.get("prev_global_tensors_hex", "[]"))
        if tensors_hex:
            prev_params = Parameters(
                tensors=[bytes.fromhex(h) for h in tensors_hex],
                tensor_type=config.get("prev_global_tensor_type", "numpy.ndarray"),
            )
            prev_nds = parameters_to_ndarrays(prev_params)
            tmp = get_resnet_cnn_model()
            set_weights(tmp, prev_nds)
            prev_global_vec = parameters_to_vector(tmp.parameters()).detach().cpu().clone()
            prev_global_sd = {k: v.cpu().clone() for k, v in tmp.state_dict().items()}
        else:
            prev_global_vec = None
            prev_global_sd = None

        attack_mode = config.get("backdoor-attack-mode", "none").lower()
        attack_type = config.get("backdoor-attack-type", "none").lower()
        partition_id = self.context.node_config["partition-id"]
        num_partitions = self.context.node_config["num-partitions"]
        alpha = self.alpha

        is_malicious = str(config.get("is_malicious", "False")).lower() == "true"
        current_round = config.get("current-round", "N/A")

        is_attacking_round = False
        attack_epochs = self.local_epochs
        learning_rate = 0.005

        if attack_mode == "per-round-attack" and is_malicious:
            print(f"[Round {current_round}] Per-Round Attack Injected #Client ID: {partition_id}")
            is_attacking_round = True
            self.training_set, _ = load_data(
                partition_id, num_partitions, alpha_val=alpha, backdoor_enabled=True
            )
            attack_epochs = 5
            learning_rate = 0.005
        else:
            self.training_set, _ = load_data(partition_id, num_partitions, alpha_val=alpha)

        if is_attacking_round:
            if attack_type == "constrain-and-scale":
                cas_scale = float(config.get("cas-scale-factor", 1.0))
                cas_epochs = int(config.get("cas-epochs", 3))
                cas_lr = float(config.get("cas-lr", 0.01))
                cas_lambda_prox = float(config.get("cas-lambda-prox", 0.5))
                cas_poison_ratio = float(config.get("cas-poison-ratio", 0.5))

                clean_set, _ = load_data(
                    partition_id, num_partitions, alpha_val=alpha, backdoor_enabled=False
                )

                prev_global_vec_cas = None
                if prev_global_sd is not None:
                    from torch.nn.utils import parameters_to_vector as p2v
                    prev_model = get_resnet_cnn_model()
                    prev_model.load_state_dict(prev_global_sd)
                    prev_global_vec_cas = p2v(prev_model.parameters()).detach().cpu()

                final_vec = train_constrain_and_scale_for_fedavg_tm(
                    net=self.net,
                    training_data=self.training_set,
                    clean_data=clean_set,
                    device=self.device,
                    init_vec=init_vec,
                    prev_global_vec=prev_global_vec_cas,
                    epochs=cas_epochs,
                    lr=cas_lr,
                    scale_factor=cas_scale,
                    lambda_prox=cas_lambda_prox,
                    poison_ratio=cas_poison_ratio,
                )

                delta = final_vec.cpu() - init_vec.cpu()
                print(
                    f"[Round {current_round}] CaS Client {partition_id} | "
                    f"epochs={cas_epochs} | lr={cas_lr} | scale={cas_scale} | "
                    f"prox={cas_lambda_prox} | poison={cas_poison_ratio} | "
                    f"delta_norm={delta.norm().item():.6f}"
                )

                vector_to_parameters(final_vec.to(self.device), self.net.parameters())
                self.prev_global_vec = init_vec.clone()
                return get_weights(self.net), len(self.training_set.dataset), {
                    "attack": "constrain-and-scale",
                    "train_loss": 0.0,
                    "local_epochs": int(cas_epochs),
                    "local_lr": float(cas_lr),
                    "scale_factor": float(cas_scale),
                }

            elif attack_type == "neurotoxin":
                attack_epochs = 40
                mask_ratio = float(config.get("neurotoxin-mask-ratio", 0.01))
                neurotoxin_scale = float(config.get("neurotoxin-scale-factor", 1.0))

                benign_grad_approx = None
                if prev_global_vec is not None:
                    benign_grad_approx = (init_vec - prev_global_vec).cpu()

                train_loss, final_vec = train_neurotoxin(
                    net=self.net,
                    training_data=self.training_set,
                    device=self.device,
                    init_vec=init_vec.cpu(),
                    benign_grad_approx=benign_grad_approx,
                    epochs=attack_epochs,
                    lr=learning_rate,
                    mask_ratio=mask_ratio,
                    scale_factor=neurotoxin_scale,
                )

                delta = final_vec - init_vec.cpu()
                print(
                    f"[Round {current_round}] Neurotoxin Client {partition_id} | "
                    f"mask_ratio={mask_ratio:.4f} | "
                    f"delta_norm={delta.norm().item():.4f} | "
                    f"scale={neurotoxin_scale}"
                )

                vector_to_parameters(final_vec.to(self.device), self.net.parameters())
                self.prev_global_vec = init_vec.clone()
                return get_weights(self.net), len(self.training_set.dataset), {
                    "attack": "neurotoxin",
                    "train_loss": float(train_loss),
                    "local_epochs": int(attack_epochs),
                    "local_lr": float(learning_rate),
                    "scale_factor": float(neurotoxin_scale),
                    "mask_ratio": float(mask_ratio),
                }

            elif attack_type == "lga":
                lga_epochs = int(config.get("lga-epochs", 6))
                lga_lr = float(config.get("lga-lr", 0.1))
                lga_tau = float(config.get("lga-tau", 1.0))

                prev_global_update_sd = None
                if prev_global_sd is not None:
                    prev_global_update_sd = {
                        k: init_state[k] - prev_global_sd[k]
                        for k in init_state
                        if k in prev_global_sd
                    }

                train_loss, final_vec = train_lga(
                    net=self.net,
                    training_data=self.training_set,
                    device=self.device,
                    init_sd=init_state,
                    prev_global_update_sd=prev_global_update_sd,
                    epochs=lga_epochs,
                    lr=lga_lr,
                    tau=lga_tau,
                )

                delta = final_vec - init_vec.cpu()
                print(
                    f"[Round {current_round}] LGA Client {partition_id} | "
                    f"epochs={lga_epochs} | lr={lga_lr} | tau={lga_tau} | "
                    f"delta_norm={delta.norm().item():.6f}"
                )

                vector_to_parameters(final_vec.to(self.device), self.net.parameters())
                self.prev_global_vec = init_vec.clone()
                return get_weights(self.net), len(self.training_set.dataset), {
                    "attack": "lga",
                    "train_loss": float(train_loss),
                    "local_epochs": int(lga_epochs),
                    "local_lr": float(lga_lr),
                    "scale_factor": 1.0,
                    "lga_tau": float(lga_tau),
                }

            elif attack_type == "bc-layers-lp-full":
                clean_set, _ = load_data(
                    partition_id, num_partitions, alpha_val=alpha, backdoor_enabled=False
                )
                poisoned_set, _ = load_data(
                    partition_id, num_partitions, alpha_val=alpha, backdoor_enabled=True
                )

                bc_tau = float(config.get("bc-tau", 0.95))
                bc_lambda = float(config.get("bc-lambda", 1.0))
                bc_benign_epochs = int(config.get("bc-benign-epochs", 2))
                bc_malicious_epochs = int(config.get("bc-malicious-epochs", 2))
                bc_lr = float(config.get("bc-lr", 0.1))
                bc_proxy_count = int(config.get("bc-proxy-count", 5))
                bc_defense_sim = str(config.get("bc-defense-sim", "none"))

                bc_diagnostics = {}

                if bc_defense_sim == "fedmast":
                    bc_norm_match = float(config.get("bc-norm-match-weight", 0.8))
                    bc_noise = float(config.get("bc-energy-smooth-noise", 0.03))
                    bc_blend = float(config.get("bc-direction-blend-alpha", 0.15))
                    bc_min_bsr = float(config.get("bc-min-bsr-ratio", 0.5))

                    train_loss, final_vec, bc_diagnostics = train_bc_layers_lp_full_fedmast(
                        net=self.net,
                        clean_data=clean_set,
                        poisoned_data=poisoned_set,
                        device=self.device,
                        init_sd=init_state,
                        target_label=2,
                        tau=bc_tau,
                        lambda_val=bc_lambda,
                        benign_epochs=bc_benign_epochs,
                        malicious_epochs=bc_malicious_epochs,
                        local_lr=bc_lr,
                        proxy_count=bc_proxy_count,
                        norm_match_weight=bc_norm_match,
                        energy_smooth_noise=bc_noise,
                        direction_blend_alpha=bc_blend,
                        min_bsr_ratio=bc_min_bsr,
                    )
                else:
                    train_loss, final_vec = train_bc_layers_lp_full(
                        net=self.net,
                        clean_data=clean_set,
                        poisoned_data=poisoned_set,
                        device=self.device,
                        init_sd=init_state,
                        target_label=2,
                        tau=bc_tau,
                        lambda_val=bc_lambda,
                        benign_epochs=bc_benign_epochs,
                        malicious_epochs=bc_malicious_epochs,
                        local_lr=bc_lr,
                        proxy_count=bc_proxy_count,
                        defense_sim=bc_defense_sim,
                    )

                delta = final_vec - init_vec.cpu()
                print(
                    f"[Round {current_round}] BC-Layers-Full Client {partition_id} | "
                    f"tau={bc_tau} | lambda={bc_lambda} | proxies={bc_proxy_count} | "
                    f"defense_sim={bc_defense_sim} | delta_norm={delta.norm().item():.6f}"
                )

                vector_to_parameters(final_vec.to(self.device), self.net.parameters())
                self.prev_global_vec = init_vec.clone()

                metrics = {
                    "attack": f"bc-layers-lp-full-{bc_defense_sim}",
                    "train_loss": float(train_loss),
                    "local_epochs": int(bc_malicious_epochs),
                    "local_lr": float(bc_lr),
                    "scale_factor": float(bc_lambda),
                }
                metrics.update({
                    k: float(v) if isinstance(v, (int, float)) else v
                    for k, v in bc_diagnostics.items()
                })
                return get_weights(self.net), len(clean_set.dataset), metrics

            elif attack_type == "dba":
                clean_set, _ = load_data(
                    partition_id, num_partitions, alpha_val=alpha, backdoor_enabled=False
                )

                dba_epochs = int(config.get("dba-epochs", 6))
                dba_lr = float(config.get("dba-lr", 0.05))
                dba_poison_ratio = float(config.get("dba-poison-ratio", 0.078125))
                dba_num_attackers = int(config.get("dba-num-attackers", 4))
                dba_scale = float(config.get("dba-scale-factor", 100.0))

                if "dba-attacker-index" not in config:
                    raise RuntimeError(
                        f"[DBA] Missing server-assigned slot for partition {partition_id}. "
                        f"Ensure dba-attacker-partition-ids includes this partition."
                    )
                attacker_index = int(config["dba-attacker-index"])
                if attacker_index not in (0, 1, 2, 3):
                    raise ValueError(f"[DBA] Invalid slot: {attacker_index}")
                if dba_num_attackers != 4:
                    raise ValueError("[DBA] Official CIFAR-10 DBA requires exactly 4 attackers")

                train_loss, scaled_vec, dba_metrics = train_dba(
                    net=self.net,
                    training_data=clean_set,
                    device=self.device,
                    init_vec=init_vec,
                    attacker_index=attacker_index,
                    num_attackers=dba_num_attackers,
                    target_label=2,
                    epochs=dba_epochs,
                    lr=dba_lr,
                    poison_ratio=dba_poison_ratio,
                    scale_factor=dba_scale,
                )

                print(
                    f"[DBA][Round {current_round}] partition={partition_id} "
                    f"slot={attacker_index} "
                    f"poisoned={int(dba_poison_ratio*64)}/64 "
                    f"scale={dba_scale} "
                    f"unscaled_norm={dba_metrics['dba_unscaled_delta_norm']:.4f} "
                    f"scaled_norm={dba_metrics['dba_scaled_delta_norm']:.4f}"
                )

                vector_to_parameters(scaled_vec.to(self.device), self.net.parameters())
                self.prev_global_vec = init_vec.clone()
                return get_weights(self.net), len(clean_set.dataset), {
                    "attack": "dba",
                    "train_loss": float(train_loss),
                    "local_epochs": int(dba_epochs),
                    "local_lr": float(dba_lr),
                    "scale_factor": float(dba_scale),
                    "dba_attacker_index": int(attacker_index),
                    "dba_unscaled_delta_norm": dba_metrics["dba_unscaled_delta_norm"],
                    "dba_scaled_delta_norm": dba_metrics["dba_scaled_delta_norm"],
                }

            elif attack_type == "3dfed":
                canonical_pid = int(config.get("3dfed-canonical-partition-id", 5))

                clean_set, _ = load_data(
                    canonical_pid, num_partitions, alpha_val=alpha, backdoor_enabled=False
                )
                poisoned_set, _ = load_data(
                    canonical_pid, num_partitions, alpha_val=alpha, backdoor_enabled=True
                )

                role = str(config.get("3dfed-role", "backdoor"))
                slot_index = int(config.get("3dfed-slot-index", 0))
                round_seed = int(config.get("3dfed-round-seed", 0))
                num_bd = int(config.get("3dfed-num-backdoor-models", 4))
                num_dc = int(config.get("3dfed-num-decoys", 0))

                raw_indices = config.get("3dfed-indicator-indices", "[]")
                try:
                    indicator_indices = json.loads(raw_indices) if isinstance(raw_indices, str) else list(raw_indices)
                except Exception:
                    indicator_indices = []

                outgoing_sd, tdfed_metrics, ind_records = train_3dfed(
                    net=self.net,
                    clean_data=clean_set,
                    poisoned_data=poisoned_set,
                    device=self.device,
                    init_sd=init_state,
                    net_factory=get_resnet_cnn_model,
                    role=role,
                    slot_index=slot_index,
                    round_seed=round_seed,
                    num_backdoor_models=num_bd,
                    num_decoys=num_dc,
                    server_round=current_round,
                    use_indicator=str(config.get("3dfed-use-indicator", "true")).lower() == "true",
                    indicator_indices=indicator_indices if indicator_indices else None,
                    kappa=float(config.get("3dfed-kappa", 100000.0)),
                    indicator_candidate_ratio=float(config.get("3dfed-indicator-candidate-ratio", 0.01)),
                    indicator_count=int(config.get("3dfed-indicator-count", 32)),
                    hessian_samples=int(config.get("3dfed-hessian-samples", 4)),
                    use_constrained_loss=str(config.get("3dfed-use-constrained-loss", "true")).lower() == "true",
                    beta=float(config.get("3dfed-beta", 0.3)),
                    gamma_scale=float(config.get("3dfed-gamma", 1.0)),
                    cl_epochs=int(config.get("local-epochs", 3)),
                    cl_lr=float(config.get("3dfed-cl-lr", 0.01)),
                    target_label=int(config.get("3dfed-target-label", 2)),
                    use_noise_mask=str(config.get("3dfed-use-noise-mask", "true")).lower() == "true",
                    alpha=float(config.get("3dfed-alpha", 0.5)),
                    noise_lambda=float(config.get("3dfed-noise-lambda", 1.0)),
                    noise_dual_step=float(config.get("3dfed-noise-dual-step", 0.1)),
                    noise_steps=int(config.get("3dfed-noise-steps", 20)),
                    noise_lr=float(config.get("3dfed-noise-lr", 0.01)),
                    use_decoy=str(config.get("3dfed-use-decoy", "true")).lower() == "true",
                    decoy_steps=int(config.get("3dfed-decoy-steps", 20)),
                    decoy_lr=float(config.get("3dfed-decoy-lr", 0.01)),
                )

                self.net.load_state_dict(outgoing_sd)
                final_vec = parameters_to_vector(self.net.parameters()).detach().cpu()

                vector_to_parameters(final_vec.to(self.device), self.net.parameters())
                self.prev_global_vec = init_vec.clone()

                ind_records_json = json.dumps(
                    [r.to_dict() for r in ind_records]
                )
                found_indices = tdfed_metrics.get("indicator_finding", {}).get("indices", [])

                return get_weights(self.net), len(clean_set.dataset), {
                    "attack": "3dfed",
                    "3dfed_role": role,
                    "3dfed_slot": slot_index,
                    "3dfed_alpha": float(config.get("3dfed-alpha", 0.5)),
                    "3dfed_k_decoy": int(config.get("3dfed-k-decoy", 0)),
                    "3dfed_indicator_records": ind_records_json,
                    "3dfed_indicator_indices": json.dumps(found_indices),
                    "train_loss": float(tdfed_metrics.get("constrained_loss", {}).get("task_loss", 0)),
                    "local_epochs": int(config.get("local-epochs", 3)),
                    "local_lr": float(config.get("3dfed-cl-lr", 0.01)),
                }

            else:
                raise ValueError(f"Unsupported backdoor-attack-type: {attack_type}")

        else:
            if attack_mode == "per-round-attack" and attack_type == "lga":
                sampled_epochs = 1
                sampled_lr = 0.003
            else:
                sampled_lr = random.choice([0.003, 0.004, 0.005])
                sampled_epochs = random.choice([1, 2, 3])
            train_loss, final_vec = train(
                self.net,
                self.training_set,
                sampled_epochs,
                self.device,
                sampled_lr
            )

            self.prev_global_vec = init_vec.clone()

            return get_weights(self.net), len(self.training_set.dataset), {
                "train_loss": train_loss,
                "local_epochs": int(sampled_epochs),
                "local_lr": float(sampled_lr),
                "scale_factor": 1.0,
            }

    def evaluate(self, parameters, config):
        partition_id = self.context.node_config["partition-id"]
        num_partitions = self.context.node_config["num-partitions"]
        _, self.test_set = load_data(partition_id, num_partitions, alpha_val=self.alpha)
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.test_set, self.device)

        attack_type = str(config.get("backdoor-attack-type",
                          self.context.run_config.get("backdoor-attack-type", "none"))).lower()

        if attack_type == "dba":
            dba_metrics = evaluate_dba_asr(
                self.net, self.test_set, target_label=2, device=self.device
            )
            client_side_asr = float(dba_metrics["dba_asr_global"])
            print(
                f"[Client {partition_id}][DBA Eval] MTA={accuracy:.4f} "
                f"Global ASR={client_side_asr:.4f}"
            )
            return loss, len(self.test_set.dataset), {
                "mta": accuracy,
                "asr": client_side_asr,
                **{k: float(v) for k, v in dba_metrics.items()},
            }
        else:
            client_side_asr = evaluate_asr(
                self.net, self.test_set, target_label=2, device=self.device
            )
            print(
                f"[Client {partition_id}] Completed evaluation: "
                f"MTA={accuracy:.4f}, ASR={client_side_asr:.4f}"
            )
            return loss, len(self.test_set.dataset), {
                "mta": accuracy, "asr": client_side_asr
            }

def client_fn(context: Context):
    net = get_resnet_cnn_model()
    local_epochs = context.run_config["local-epochs"]
    client = FlowerClient(net, local_epochs, context)
    return client.to_client()


app = ClientApp(
    client_fn,
)
