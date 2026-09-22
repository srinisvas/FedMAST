import json
import os

import torch
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig

from fed_learning_cifar_experiment.state.multi_krum_metrics_strategy import SaveMultiKrumMetricsStrategy
from fed_learning_cifar_experiment.state.trimmed_mean_metrics_strategy import SaveTrimmedMeanMetricsStrategy
from fed_learning_cifar_experiment.state.flame_metrics_strategy import SaveFLAMEMetricsStrategy
from fed_learning_cifar_experiment.state.alignins_metrics_strategy import SaveAlignInsMetricsStrategy
from fed_learning_cifar_experiment.state.server_strategy import SaveFedAvgMetricsStrategy
from fed_learning_cifar_experiment.state.fedmast_metrics_strategy import SaveFedMASTMetricsStrategy

from fed_learning_cifar_experiment.utils.evaluate_attack import get_evaluate_fn
from fed_learning_cifar_experiment.task import (
    evaluate_dba_asr,
    get_weights,
    get_resnet_cnn_model,
    load_test_data_for_eval,
    set_weights,
)


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _resolve_history_path(path):
    if os.path.isabs(path) or os.path.exists(path):
        return path
    candidate = os.path.join(PROJECT_ROOT, path)
    return candidate if os.path.exists(candidate) else path


def _make_evaluate_fn(model, test_data, attack_type="none"):
    base_fn = get_evaluate_fn(model=model, test_data=test_data)

    if attack_type != "dba":
        return base_fn

    def dba_evaluate_fn(server_round, parameters, config):
        loss, metrics = base_fn(server_round, parameters, config)

        eval_model = get_resnet_cnn_model()
        set_weights(eval_model, parameters)

        device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        eval_model.to(device)

        dba_metrics = evaluate_dba_asr(
            eval_model, test_data, target_label=2, device=device,
        )

        metrics = dict(metrics or {})
        metrics["asr"] = float(dba_metrics["dba_asr_global"])
        metrics.update(
            {key: float(value) for key, value in dba_metrics.items()}
        )

        return loss, metrics

    return dba_evaluate_fn


def server_fn(context: Context):
    num_rounds = context.run_config["num-server-rounds"]
    num_clients = context.run_config["num-clients"]
    simulation_id = context.run_config.get("simulation-id")
    aggregation_method = context.run_config.get("aggregation-method", "fedavg").lower()
    backdoor_attack_mode = context.run_config.get("backdoor-attack-mode", "none").lower()
    backdoor_attack_type = context.run_config.get("backdoor-attack-type", "none").lower()
    num_of_malicious_clients = context.run_config.get("num-malicious-clients", 0)
    num_of_malicious_clients_per_round = context.run_config.get("num-malicious-clients-per-round", 1)
    dirichlet_alpha = float(context.run_config.get("dirichlet-alpha", 0.3))

    _raw_bdp = context.run_config.get("backdoor-partition-ids", None)
    backdoor_partition_ids = json.loads(_raw_bdp) if _raw_bdp else None

    attack_start_round = int(context.run_config.get("attack-start-round", 1))
    attack_end_round = int(context.run_config.get("attack-end-round", 999999))

    raw_dba_parts = context.run_config.get("dba-attacker-partition-ids", "[5,12,23,36]")
    dba_attacker_partition_ids = json.loads(raw_dba_parts) if isinstance(raw_dba_parts, str) else list(raw_dba_parts)
    dba_server_eta = float(context.run_config.get("dba-server-eta", 1.0))

    model = get_resnet_cnn_model()
    model_nd_arrays = get_weights(model)
    parameters = ndarrays_to_parameters(model_nd_arrays)

    testing_data = load_test_data_for_eval(batch_size=64)

    def on_fit_config_fn(server_round: int):
        on_fit_config = {}
        if backdoor_attack_mode == "none":
            on_fit_config = {
                "backdoor-attack-mode": "none",
            }
        elif backdoor_attack_mode == "per-round-attack":
            on_fit_config = {
                "backdoor-attack-mode": "per-round-attack",
                "backdoor-attack-type": backdoor_attack_type,
                "current-round": server_round,
            }
            attack_prefixes = {
                "constrain-and-scale": [
                    "cas-epochs", "cas-lr", "cas-scale-factor",
                    "cas-lambda-prox", "cas-poison-ratio",
                ],
                "neurotoxin": ["neurotoxin-mask-ratio", "neurotoxin-scale-factor"],
                "lga": ["lga-epochs", "lga-lr", "lga-tau"],
                "bc-layers-lp-full": [
                    "bc-tau", "bc-lambda", "bc-benign-epochs",
                    "bc-malicious-epochs", "bc-lr", "bc-proxy-count",
                    "bc-defense-sim",
                    "bc-norm-match-weight", "bc-energy-smooth-noise",
                    "bc-direction-blend-alpha", "bc-min-bsr-ratio",
                ],
                "dba": [
                    "dba-epochs", "dba-lr", "dba-poison-ratio",
                    "dba-num-attackers", "dba-scale-factor",
                    "dba-server-eta", "dba-attacker-partition-ids",
                    "dba-paper-faithful", "dba-benign-lr",
                    "dba-benign-epochs", "dirichlet-alpha",
                ],
                "3dfed": [
                    "3dfed-target-label", "3dfed-kappa", "3dfed-beta",
                    "3dfed-gamma", "3dfed-alpha-step", "3dfed-noise-lambda",
                    "3dfed-noise-dual-step", "3dfed-noise-steps", "3dfed-noise-lr",
                    "3dfed-decoy-steps", "3dfed-decoy-lr", "3dfed-cl-lr",
                    "3dfed-indicator-candidate-ratio", "3dfed-indicator-count",
                    "3dfed-hessian-samples", "3dfed-use-indicator",
                    "3dfed-use-constrained-loss", "3dfed-use-noise-mask",
                    "3dfed-use-decoy", "3dfed-num-backdoor-models",
                    "3dfed-num-decoy-models-init", "3dfed-max-decoy-models",
                ],
            }
            for key in attack_prefixes.get(backdoor_attack_type, []):
                val = context.run_config.get(key)
                if val is not None:
                    on_fit_config[key] = val
        return on_fit_config

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    common = dict(
        fraction_fit=0.1,
        fraction_evaluate=0.1,
        min_fit_clients=10,
        min_available_clients=100,
        initial_parameters=parameters,
        on_fit_config_fn=on_fit_config_fn,
        simulation_id=simulation_id,
        num_clients=num_clients,
        num_rounds=num_rounds,
        aggregation_method=aggregation_method,
        backdoor_attack_mode=backdoor_attack_mode,
        num_of_malicious_clients=num_of_malicious_clients,
        num_of_malicious_clients_per_round=num_of_malicious_clients_per_round,
        dirichlet_alpha=dirichlet_alpha,
    )

    if aggregation_method == "fedavg":
        strategy = SaveFedAvgMetricsStrategy(
            evaluate_fn=_make_evaluate_fn(
                model=get_resnet_cnn_model().to(device),
                test_data=testing_data,
                attack_type=backdoor_attack_type,
            ),
            backdoor_partition_ids=backdoor_partition_ids,
            attack_start_round=attack_start_round,
            attack_end_round=attack_end_round,
            dba_attacker_partition_ids=dba_attacker_partition_ids,
            dba_server_eta=dba_server_eta,
            **common,
        )
    elif aggregation_method == "multikrum":
        strategy = SaveMultiKrumMetricsStrategy(
            evaluate_fn=_make_evaluate_fn(
                model=get_resnet_cnn_model().to(device),
                test_data=testing_data,
                attack_type=backdoor_attack_type,
            ),
            num_byzantine=int(num_of_malicious_clients_per_round),
            num_clients_to_select=5,
            normalize_updates=True,
            **common,
        )
    elif aggregation_method == "trimmed_mean":
        strategy = SaveTrimmedMeanMetricsStrategy(
            evaluate_fn=_make_evaluate_fn(
                model=get_resnet_cnn_model().to(device),
                test_data=testing_data,
                attack_type=backdoor_attack_type,
            ),
            num_byzantine=int(num_of_malicious_clients_per_round),
            trim_ratio=0.2,
            **common,
        )
    elif aggregation_method == "alignins":
        strategy = SaveAlignInsMetricsStrategy(
            evaluate_fn=_make_evaluate_fn(
                model=get_resnet_cnn_model().to(device),
                test_data=testing_data,
                attack_type=backdoor_attack_type,
            ),
            num_byzantine=int(num_of_malicious_clients_per_round),
            alignins_sparsity=0.3,
            backdoor_partition_ids=backdoor_partition_ids or [],
            attack_start_round=attack_start_round,
            attack_end_round=attack_end_round,
            dba_attacker_partition_ids=dba_attacker_partition_ids,
            dba_server_eta=dba_server_eta,
            **common,
        )
    elif aggregation_method == "flame":
        strategy = SaveFLAMEMetricsStrategy(
            evaluate_fn=_make_evaluate_fn(
                model=get_resnet_cnn_model().to(device),
                test_data=testing_data,
                attack_type=backdoor_attack_type,
            ),
            num_byzantine=int(num_of_malicious_clients_per_round),
            flame_noise_multiplier=0.001,
            flame_noise_mode="coordinate",
            flame_allow_fallback_dbscan=False,
            backdoor_partition_ids=backdoor_partition_ids,
            attack_start_round=attack_start_round,
            attack_end_round=attack_end_round,
            dba_attacker_partition_ids=dba_attacker_partition_ids,
            dba_server_eta=dba_server_eta,
            **common,
        )
    elif aggregation_method == "fedmast":
        fedmast_threshold_k = float(context.run_config.get("fedmast-threshold-k", 3.0))
        fedmast_min_accept = int(context.run_config.get("fedmast-min-accept", 5))
        fedmast_warmup_rounds = int(context.run_config.get("fedmast-warmup-rounds", 2))
        fedmast_buffer_rounds = int(context.run_config.get("fedmast-buffer-rounds", 20))
        fedmast_trajectory_window = int(context.run_config.get("fedmast-trajectory-window", 15))
        fedmast_suspicious_policy = str(context.run_config.get("fedmast-suspicious-policy", "coord-median"))
        fedmast_history_mode = str(context.run_config.get("fedmast-history-mode", "none"))
        fedmast_history_path = str(context.run_config.get("fedmast-history-path", "fedmast_history_clean.json"))
        if fedmast_history_mode.startswith("load"):
            fedmast_history_path = _resolve_history_path(fedmast_history_path)

        fedmast_spectral_min_appearances = int(context.run_config.get("fedmast-spectral-min-appearances", 5))
        fedmast_spectral_decay = float(context.run_config.get("fedmast-spectral-decay", 0.7))
        fedmast_spectral_percentile = float(context.run_config.get("fedmast-spectral-percentile", 85.0))
        fedmast_spectral_min_threshold = float(context.run_config.get("fedmast-spectral-min-threshold", 0.5))

        fedmast_ablate_squeeze = str(context.run_config.get("fedmast-ablate-squeeze", "false")).lower() == "true"
        fedmast_ablate_spectral = str(context.run_config.get("fedmast-ablate-spectral", "false")).lower() == "true"
        fedmast_ablate_history = str(context.run_config.get("fedmast-ablate-history", "false")).lower() == "true"
        fedmast_ablate_round = str(context.run_config.get("fedmast-ablate-round", "false")).lower() == "true"

        strategy = SaveFedMASTMetricsStrategy(
            evaluate_fn=_make_evaluate_fn(
                model=get_resnet_cnn_model().to(device),
                test_data=testing_data,
                attack_type=backdoor_attack_type,
            ),
            backdoor_partition_ids=backdoor_partition_ids,
            attack_start_round=attack_start_round,
            attack_end_round=attack_end_round,
            dba_attacker_partition_ids=dba_attacker_partition_ids,
            dba_server_eta=dba_server_eta,
            fedmast_threshold_k=fedmast_threshold_k,
            fedmast_min_accept=fedmast_min_accept,
            fedmast_warmup_rounds=fedmast_warmup_rounds,
            fedmast_buffer_rounds=fedmast_buffer_rounds,
            fedmast_trajectory_window=fedmast_trajectory_window,
            fedmast_suspicious_policy=fedmast_suspicious_policy,
            fedmast_history_mode=fedmast_history_mode,
            fedmast_history_path=fedmast_history_path,
            fedmast_spectral_min_appearances=fedmast_spectral_min_appearances,
            fedmast_spectral_decay=fedmast_spectral_decay,
            fedmast_spectral_percentile=fedmast_spectral_percentile,
            fedmast_spectral_min_threshold=fedmast_spectral_min_threshold,
            fedmast_ablate_squeeze=fedmast_ablate_squeeze,
            fedmast_ablate_spectral=fedmast_ablate_spectral,
            fedmast_ablate_history=fedmast_ablate_history,
            fedmast_ablate_round=fedmast_ablate_round,
            **common,
        )
    else:
        raise ValueError(
            f"Unsupported aggregation-method: {aggregation_method}. "
            "Choose one of: fedavg, multikrum, trimmed_mean, alignins, flame, fedmast."
        )

    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


app = ServerApp(server_fn=server_fn)
