# Backdoors Leave Structural Traces: FedMAST for Backdoor Detection and Containment in Federated Learning

**Accepted at the 38th IEEE International Conference on Tools with Artificial Intelligence (ICTAI)**

**Artifact DOI:** https://doi.org/10.5281/zenodo.22885416

This branch is the executable CIFAR-10 artifact for the paper. It contains the Flower federated-learning
simulation used for the reported CIFAR-10 experiments: the FedMAST defense, the backdoor attacks it is evaluated against, the
baseline defenses it is compared with, the data partitioning, the model, the trigger generation and all
metric/logging code.

A reader can clone the repository, install the environment, prepare CIFAR-10 and run a single command to
watch FedMAST take part in a real Flower round: clients train, the server extracts structural features from
every update, scores the clients, rejects suspicious ones, aggregates (or contains the round) and updates its
state.

## Contents

1. [Repository layout](#1-repository-layout)
2. [Environment](#2-environment)
3. [Data preparation](#3-data-preparation)
4. [One-command demo](#4-one-command-demo)
5. [Simulation structure](#5-simulation-structure)
6. [FedMAST: what is implemented and where](#6-fedmast-what-is-implemented-and-where)
7. [Attacks](#7-attacks)
8. [Baseline defenses](#8-baseline-defenses)
9. [Configuration reference](#9-configuration-reference)
10. [Reproducing the experiments](#10-reproducing-the-experiments)
11. [Outputs and metrics](#11-outputs-and-metrics)
12. [Seeds and determinism](#12-seeds-and-determinism)
13. [Notes and known behaviours](#13-notes-and-known-behaviours)

---

## 1. Repository layout

```
pyproject.toml                          Flower app definition + every run-config default
requirements.txt                        Pip requirements
LICENSE                                 Apache-2.0 license
scripts/validate_artifact.py             Static artifact consistency checks
scripts/summarize_detection.py           TP/FN/FP/TN, recall and FPR from per-update CSVs
fedmast_history_clean.json              Clean benign FedMAST history (baseline statistics, 50 rounds)
fed_learning_cifar_experiment/
  server_app.py                         Flower ServerApp: builds the strategy for the chosen defense
  client_app.py                         Flower ClientApp: benign training + all attack dispatch
  task.py                               Data loading, benign training, Neurotoxin, LGA, DBA,
                                        BC-Layers / CovertLayers, Constrain-and-Scale, evaluation helpers
  threeDFed.py                          3DFed attack (constrained loss, noise masks, indicators, decoys)
  prepare_data.py                       Downloads CIFAR-10 to fed_learning_cifar_experiment/data/
  models/resnet_cnn_model.py            The ResNet used everywhere (TinyResNet18)
  state/
    fedmast_metrics_strategy.py         FedMAST defense (all 22 features, scoring axes, thresholds,
                                        cluster bisection, anchor inversion, coalition validation,
                                        quarantine / history admission, containment)
    server_strategy.py                  FedAvg strategy + shared per-update feature extraction,
                                        attacker sampling, 3DFed/DBA server-side coordination
    krum_metrics_strategy.py            Base class shared by the Multi-Krum, FLAME and AlignIns strategies
    multi_krum_metrics_strategy.py      Multi-Krum baseline
    trimmed_mean_metrics_strategy.py     Coordinate-wise Trimmed Mean baseline
    flame_metrics_strategy.py           FLAME baseline
    alignins_metrics_strategy.py        AlignIns baseline
  utils/
    metrics_extractor.py                Per-update structural feature extraction (feature computation)
    backdoor_attack.py                  Trigger generation and poisoned-batch collation
    drichlet_partition.py               Dirichlet non-IID partitioning
    evaluate_attack.py                  Centralized evaluation: main-task accuracy and attack success rate
    logger.py                           CSV logging
    local_attack.py                     Vector helpers for norm/cosine-constrained attack losses
    fedmast_diagnostic.py               Post-hoc analysis of a per-update feature CSV
    deep_diagnostic.py                  Post-hoc analysis of a per-update feature CSV
    verify_extraction.py                Sanity checks for a per-update feature CSV
```

## 2. Environment

Requirements: Python 3.10 or newer. A GPU is optional; every client and the server pick `cuda:0`
automatically when available and fall back to CPU.

```bash
git clone <this repository>
cd <this repository>

python -m venv .venv
source .venv/bin/activate            # Windows PowerShell: .venv\Scripts\Activate.ps1

pip install -e .                     # installs everything declared in pyproject.toml
# or, equivalently:
# pip install -r requirements.txt && pip install -e . --no-deps
```

With conda:

```bash
conda create -n fedmast python=3.10 -y
conda activate fedmast
pip install -e .
python scripts/validate_artifact.py
```

Declared dependencies: `flwr[simulation]>=1.20.0` (this also installs Ray, the Flower simulation backend),
`flwr-datasets[vision]>=0.5.0`, `datasets>=2.18.0`, `torch==2.7.1`, `torchvision==0.22.1`, `numpy`, `scipy`, `scikit-learn`,
`pandas` (post-hoc diagnostics) and `hdbscan` (required by the FLAME baseline, whose DBSCAN fallback is
disabled).
`pip install -e .` matters: Flower's simulation workers import the package by name
(`fed_learning_cifar_experiment.server_app:app` / `client_app:app`).

## 3. Data preparation

The code reads CIFAR-10 from a local Hugging Face copy at `fed_learning_cifar_experiment/data/cifar10_hf`
(it never downloads during a run). Create it once:

```bash
python -m fed_learning_cifar_experiment.prepare_data
```

This downloads `uoft-cs/cifar10` and saves it to that folder (the folder is git-ignored).

## 4. One-command demo

```bash
flwr run .
```

That single command runs a short simulation whose defaults are set in `pyproject.toml`:
100 virtual clients, 10 sampled per round, 10 rounds, **LGA** attackers (3 malicious of the 10 sampled clients
per round, from round 3 onward) against **FedMAST**, using the shipped clean history
(`fedmast_history_clean.json`). The same run written out explicitly:

```bash
RAY_DEDUP_LOGS=0 flwr run . --run-config "num-server-rounds=10 num-clients=100 \
  aggregation-method='fedmast' simulation-id='fedmast-demo' \
  backdoor-attack-mode='per-round-attack' backdoor-attack-type='lga' \
  num-malicious-clients=3 num-malicious-clients-per-round=3 \
  backdoor-partition-ids='[5,12,23,36,44,51,67,78,83,91]' \
  attack-start-round=3 lga-epochs=12 lga-lr=0.05 lga-tau=2.0 \
  fedmast-history-mode='load-readonly' fedmast-history-path='fedmast_history_clean.json' \
  fedmast-spectral-percentile=85.0 fedmast-spectral-min-threshold=1.2"
```

(PowerShell: put the whole `--run-config "..."` string on one line and set `$env:RAY_DEDUP_LOGS=0` first.)

`num-server-rounds` and `attack-start-round` are the only values shortened relative to the reported LGA
experiment (200 rounds, attack from round 20; see [section 10](#10-reproducing-the-experiments)).

What you should see in the console for every round:

| Console output | Meaning |
|---|---|
| `[Round N] Attack window: [...] \| in_window=... \| effective_malicious=...` | Server designates malicious clients for the round |
| `[Round N] STRATIFIED \| Malicious partitions: [...]` | Which of the sampled partitions are attackers (`RANDOM` in the first rounds while the client-to-partition cache is being built) |
| `Per-Round Attack Injected #Client ID: ...`, `LGA Client ... delta_norm=...` | Attackers running the attack locally |
| `[FedMAST v4] Round N \| Buffer=... \| SpectralDrift=... active \| History=load-readonly(loaded)` | FedMAST state header (accepted-update buffer size, spectral-drift tracker, history) |
| `Thresholds (k=3.0): D_round=... \| D_squeeze=... \| D_hist=... \| D_spectral=...` | Per-axis thresholds computed this round |
| Per-client table with `Score, D_rnd, D_sqz, D_hst, D_spc, D_anc, D_trj, D_mom, Axis, Decision` and a `MAL` column | The score of every client on every axis and whether it was accepted or rejected |
| `Detection: TP=.. FN=.. FP=.. TN=..` | Detection confusion counts for the round (uses the ground-truth malicious flag, for logging only) |
| `[FedMAST] Cluster mode ...`, `INVERSION detected ...`, `Coalition check ...`, `Buffer QUARANTINE ...` | The corresponding FedMAST mechanism fired |
| `Suspicious round — coordinate MEDIAN aggregation` | Containment aggregation used for a suspicious round |
| `[Round N] MTA=..., ASR=...` | Distributed main-task accuracy / attack success rate |

Per-round CSV logs are written to the working directory ([section 11](#11-outputs-and-metrics)).

## 5. Simulation structure

The Flower structure is a standard `ServerApp` / `ClientApp` pair run by the Flower Simulation Engine
(`[tool.flwr.federations.local-simulation]` in `pyproject.toml`: 100 virtual supernodes, 2 CPUs and 0 GPUs
reserved per client by default; raise `options.backend.client-resources.num-gpus` to give clients GPU share).

* **Federation:** 100 clients; 10 are sampled per round (`fraction_fit=0.1`, `min_fit_clients=10`,
  `min_available_clients=100`, fixed in `server_app.py`); 10% of clients are sampled for distributed evaluation.
* **Dataset:** CIFAR-10. Training images (50,000) are split over the 100 clients by a Dirichlet label
  distribution (`dirichlet-alpha`, default **0.3**, partition seed **42**, `utils/drichlet_partition.py`).
  The original experiment code accepted an alpha configuration but hard-coded `0.3` inside the partitioner;
  this artifact makes the parameter explicit while preserving the effective value used by the reported runs.
  Each client splits its partition 80/20 into train/test (seed 42). The global test set is the 10,000
  CIFAR-10 test images. Batch size 64.
* **Preprocessing:** training: `RandomCrop(32, padding=4)`, `RandomHorizontalFlip`,
  `ColorJitter(0.1, 0.1, 0.1, 0.05)`, scale to `[0,1]`, `Normalize(mean=(0.4914, 0.4822, 0.4465),
  std=(0.2023, 0.1994, 0.2010))`; evaluation: `ToTensor` + the same normalization.
* **Model:** `TinyResNet18` (`models/resnet_cnn_model.py`): CIFAR-style ResNet-18 layout (2-2-2-2 basic
  blocks, 3x3 stride-1 stem, no max-pool), `base_width=8` (stage widths 8/16/32/64), BatchNorm momentum 0.05,
  Kaiming init, 176,402 trainable parameters. It is the only model in the repository and starts from random
  initialization.
* **Benign clients:** each sampled benign client trains with SGD (momentum 0.9, weight decay 5e-4, cosine
  learning-rate schedule) on its local data for a number of epochs drawn from `{1, 2, 3}` and a base
  learning rate drawn from `{0.003, 0.004, 0.005}` (`client_app.py`, `task.train`; for the LGA attack
  benign clients use 1 epoch at lr 0.003).
* **Trigger and backdoor task:** an 8x8 orange patch (RGB `(1.0, 0.5, 0.0)`, alpha 1.0) in the bottom-right
  corner, target label **2**. In poisoned local training, 20 samples of every batch of 64 are stamped and
  relabeled (`utils/backdoor_attack.py`). DBA uses its own four local patterns (`task.py`).
* **Attacker designation:** a fixed pool of malicious partitions (`backdoor-partition-ids`,
  `[5,12,23,36,44,51,67,78,83,91]`). Every round inside the attack window
  (`attack-start-round` .. `attack-end-round`) the server places `num-malicious-clients-per-round` attackers
  from that pool among the 10 sampled clients (the same identities recur across rounds).
* **Server reference updates:** every round the server trains 6 reference updates (benign local training on 6
  random partitions, epochs from `{1,2}`, lr from `{0.003,0.004,0.005}`) from the current global model.
  FedMAST's reference-based features (Section 6) compare client updates with these.

## 6. FedMAST: what is implemented and where

All FedMAST logic is in `fed_learning_cifar_experiment/state/fedmast_metrics_strategy.py`
(`SaveFedMASTMetricsStrategy`, a subclass of the FedAvg strategy in `server_strategy.py`); the feature
computation is in `utils/metrics_extractor.py`. Select it with `aggregation-method='fedmast'`.

**The 22 structural features (`FEATURE_BANK`, five families)**

| Family | Features |
|---|---|
| Magnitude / displacement (4) | `total_update_l2_norm`, `l2_distance_to_round_mean`, `l2_distance_to_reference`, `cosine_to_round_mean_update` |
| Directional alignment (4) | `cosine_to_mean_leave_one_out`, `cosine_to_reference_clean_delta`, `stage_norm_signature_cos_to_round_mean`, `head_sign_agreement_with_ref` |
| Layer-energy distribution (5) | `stage_head_l2_norm`, `stage_layer4_l2_norm`, `stage_head_norm_ratio_to_total`, `classifier_to_backbone_norm_ratio`, `head_backbone_conservation_product` |
| Spectral / shape concentration (5) | `stage_head_spectral_entropy`, `stage_head_top_sv_ratio`, `stage_layer4_skewness`, `stage_layer3_kurtosis`, `backbone_kurtosis_max` |
| Cross-layer consistency (4) | `stage_layer4_cos_to_round_mean`, `stage_head_cos_to_mean_loo`, `per_class_delta_entropy`, `stage_layer4_linf_to_l2_ratio` |

(`PAPER_RENAME` maps the code names `l2_distance_to_reference`, `cosine_to_reference_clean_delta`,
`head_sign_agreement_with_ref`, `stage_head_cos_to_reference` to the paper names
`distance_to_baseline_update`, `cosine_to_historical_baseline`, `head_sign_agreement_with_baseline`,
`head_cos_to_baseline`.) Features are computed per client update over the trainable parameters, with stages
`stem, layer1..layer4, head`.

**The 6 spectral-drift features (`SPECTRAL_DRIFT_ALL`)**, kept out of the 22-feature bank and used only by
`D_spectral`: `stage_layer{2,3,4}_top_sv_ratio` and `stage_layer{2,3,4}_spectral_entropy`.

**Scoring axes (seven), thresholds and decision logic (`_score_clients`)**

* Hard axes, **2-of-3 consensus** required to reject: `D_round` (family-balanced per-round anomaly: top-2
  mean |z| per group, max over groups), `D_squeeze` (L2 of the z-scores of the four constrained feature
  pairs in `SQUEEZE_PAIRS`, max over pairs), `D_hist` (deviation from the accepted-update baseline). A client
  flagged by all three is "hard-strong".
* Temporal axis: `D_spectral`, the accumulated directional drift of the six spectral features per partition
  (`SpectralDriftTracker`; `D = -mean_z(top_sv_ratio) + mean_z(spectral_entropy)`, exponentially decayed with
  `fedmast-spectral-decay`, active after `fedmast-spectral-min-appearances` appearances). Its threshold is a
  percentile (`fedmast-spectral-percentile`) over the history of all partitions, floored by
  `fedmast-spectral-min-threshold`.
* Soft axes (ranking/monitoring only): `D_traj` (per-client trajectory), `D_momentum` (persistent anomaly),
  `D_anchor` (raw historical anchor).
* **Threshold logic:** per-axis iterative MAD thresholding at `fedmast-threshold-k` (`_iterative_threshold`),
  using a rolling accepted-update buffer of `fedmast-buffer-rounds` rounds (z-scored features) and a raw-feature
  buffer for `D_anchor`. The first `fedmast-warmup-rounds` rounds build state before scoring is enforced (the warm-up is skipped
  when a history is loaded).
* **`min_accept` rescue:** if fewer than `fedmast-min-accept` clients survive, the lowest-scoring rejected
  clients are re-admitted, except `D_spectral`-flagged and hard-strong clients, which are never rescued.
* **Cluster bisection (`_cluster_bisection`):** detects a coordinated majority (the 3DFed case) by splitting
  clients in feature space into a trusted and a suspicious cluster; the suspicious cluster is rejected and the
  round is marked suspicious.
* **Anchor inversion:** if the accepted set has a mean `D_anchor` more than 1.5x the rejected set's (and
  above 10), the ranking is judged inverted and all clients are re-sorted by `D_anchor`.
* **Coalition validation / re-sort:** once the historical aggregate tracker is active, the stage-norm signature
  of the accepted aggregate is compared with its history (cosine distance); above 0.3 the accepted set is
  re-filtered by `D_anchor`.
* **Quarantine / history admission:** accepted updates only enter the rolling buffers (the FedMAST state) when the
  round is not suspicious and no client has an active elevated `D_spectral`; otherwise the buffer update is
  quarantined (`Buffer QUARANTINE`). `fedmast-history-mode` controls the persistent baseline
  (`FedMASTHistory`): `none` (cold start), `build` (record a clean baseline to `fedmast-history-path`),
  `load-readonly` (frozen baseline loaded from file and never written back), `load-update-safe` (frozen
  baseline loaded; the file is re-saved at rounds divisible by 50 and from round 100 on, only when the round
  is not suspicious). In `build` mode the history is checkpointed every 10 rounds.
* **Aggregation / containment (`aggregate_fit`):** on a normal round the accepted clients are averaged
  (FedAvg-weighted); on a suspicious round the `fedmast-suspicious-policy` is applied to the accepted
  updates: `coord-median` (coordinate-wise median, default in all reported runs), `trimmed-mean` (20% trim),
  `fedavg`, or `noop` (skip aggregation). The chosen global model then becomes the state for the next round.
* **Ablation switches:** `fedmast-ablate-squeeze`, `-spectral`, `-history` (disables `D_hist`/`D_anchor` and
  forces history mode `none`), `-round`.

## 7. Attacks

All attacks run with `backdoor-attack-mode='per-round-attack'` and are selected with `backdoor-attack-type`
(baseline runs without attackers use `backdoor-attack-mode='none' backdoor-attack-type='none'`).

| Attack | `backdoor-attack-type` | Implementation | Parameters (defaults in `pyproject.toml` = reported values) |
|---|---|---|---|
| Constrain-and-Scale | `constrain-and-scale` | `train_constrain_and_scale_for_fedavg_tm` (`task.py`) | `cas-epochs=3`, `cas-lr=0.01`, `cas-scale-factor=1.0`, `cas-lambda-prox=0.5`, `cas-poison-ratio=0.5` |
| Neurotoxin | `neurotoxin` | `train_neurotoxin` (`task.py`) | `neurotoxin-mask-ratio=0.01`, `neurotoxin-scale-factor=1.0` (40 local epochs, lr 0.005) |
| LGA | `lga` | `train_lga` (`task.py`) | `lga-epochs=12`, `lga-lr=0.05`, `lga-tau=2.0` |
| DBA | `dba` | `train_dba`, `apply_dba_trigger` (`task.py`) | `dba-epochs=6`, `dba-lr=0.05`, `dba-poison-ratio=0.078125`, `dba-num-attackers=4`, `dba-scale-factor=1.0`, `dba-server-eta=1.0`, `dba-attacker-partition-ids='[5,12,23,36]'` |
| 3DFed | `3dfed` | `threeDFed.py` (`train_3dfed`, `AttackerCoordinator`) | `3dfed-num-backdoor-models=2`, `3dfed-num-decoy-models-init=1`, `3dfed-max-decoy-models=2`, `3dfed-beta=0.3`, `3dfed-gamma=1.0`, `3dfed-kappa=100000`, `3dfed-alpha-step=0.1`, `3dfed-noise-*`, `3dfed-decoy-*`, `3dfed-cl-lr=0.01`, `3dfed-indicator-*`, `3dfed-hessian-samples=4`, flags `3dfed-use-{indicator,constrained-loss,noise-mask,decoy}=true` |
| BC-Layers | `bc-layers-lp-full` with `bc-defense-sim='multikrum'` | `train_bc_layers_lp_full` (`task.py`) | `bc-tau=0.95`, `bc-lambda=1.0`, `bc-benign-epochs=2`, `bc-malicious-epochs=2`, `bc-lr=0.1`, `bc-proxy-count=5` |
| CovertLayers | `bc-layers-lp-full` with `bc-defense-sim='fedmast'` | `train_bc_layers_lp_full_fedmast` (`task.py`), the FedMAST-aware adaptive BC-Layers attacker | as BC-Layers plus `bc-norm-match-weight`, `bc-energy-smooth-noise`, `bc-direction-blend-alpha`, `bc-min-bsr-ratio` |

`bc-defense-sim` is the defense the BC-Layers attacker simulates when crafting its update:
`none`, `multikrum` or `krum` for BC-Layers, `fedmast` for CovertLayers.

## 8. Baseline defenses

Select with `aggregation-method`. The baselines share the same client training, attack implementations,
feature extraction and logging infrastructure. Strategy-specific client/attacker sampling remains in each
strategy implementation, matching the original experiment code paths.

| Defense | `aggregation-method` | Fixed hyperparameters (`server_app.py`) |
|---|---|---|
| FedAvg | `fedavg` | weighted average |
| Multi-Krum | `multikrum` | `num_byzantine = num-malicious-clients-per-round`, 5 clients selected, updates normalized |
| Trimmed Mean | `trimmed_mean` | coordinate-wise 20% trim per tail |
| FLAME | `flame` | HDBSCAN clustering (DBSCAN fallback disabled), norm clipping, `flame_noise_multiplier=0.001`, coordinate noise, noise seed `42 + round` |
| AlignIns | `alignins` | `alignins_sparsity=0.3` |

## 9. Configuration reference

Every parameter is a key in `[tool.flwr.app.config]` of `pyproject.toml` and can be overridden with
`flwr run . --run-config "key=value key2='string'"` (Flower rejects keys that are not declared in
`pyproject.toml`). Strings need inner single quotes; JSON lists are passed as quoted strings.

**Federation and experiment**

| Key | Default | Meaning |
|---|---|---|
| `num-server-rounds` | 10 | Federated rounds |
| `num-clients` | 100 | Virtual clients (must match `options.num-supernodes`) |
| `fraction-fit` | 0.1 | Kept for compatibility with the original scripts; the strategies use a fixed 0.1 (10 sampled clients) |
| `local-epochs` | 4 | Kept for compatibility with the original scripts; see [section 13](#13-notes-and-known-behaviours) |
| `dirichlet-alpha` | 0.3 | Dirichlet concentration of the label partition |
| `simulation-id` | `fedmast-demo` | Tag written into every CSV row |
| `aggregation-method` | `fedmast` | `fedmast`, `fedavg`, `multikrum`, `trimmed_mean`, `flame`, `alignins` |

**Attack scheduling**

| Key | Default | Meaning |
|---|---|---|
| `backdoor-attack-mode` | `per-round-attack` | `per-round-attack` or `none` |
| `backdoor-attack-type` | `lga` | `lga`, `3dfed`, `neurotoxin`, `bc-layers-lp-full`, `constrain-and-scale`, `dba`, `none` |
| `num-malicious-clients-per-round` | 3 | Attackers placed among the 10 sampled clients each round |
| `num-malicious-clients` | 3 | Recorded in the run summary; scripts set it equal to the per-round value |
| `backdoor-partition-ids` | `[5,12,23,36,44,51,67,78,83,91]` | Malicious partition pool |
| `attack-start-round` / `attack-end-round` | 3 / 999999 | Attack window |

**FedMAST**

| Key | Default | Meaning |
|---|---|---|
| `fedmast-threshold-k` | 3.0 | MAD threshold multiplier for the hard axes |
| `fedmast-min-accept` | 5 | Minimum accepted clients (rescue floor) |
| `fedmast-warmup-rounds` | 2 | Rounds before scoring is enforced |
| `fedmast-buffer-rounds` | 20 | Rolling accepted-update buffer length |
| `fedmast-trajectory-window` | 15 | Per-client trajectory window |
| `fedmast-suspicious-policy` | `coord-median` | `coord-median`, `trimmed-mean`, `fedavg`, `noop` |
| `fedmast-history-mode` | `load-readonly` | `none`, `build`, `load-readonly`, `load-update-safe` |
| `fedmast-history-path` | `fedmast_history_clean.json` | In the `load-*` modes a relative path is looked up in the working directory, then the repository root |
| `fedmast-spectral-min-appearances` | 5 | Appearances before `D_spectral` is active for a partition |
| `fedmast-spectral-decay` | 0.7 | Exponential decay of the drift score |
| `fedmast-spectral-percentile` | 85.0 | Percentile for the `D_spectral` threshold (85.0 for LGA and Constrain-and-Scale, 90.0 for the others) |
| `fedmast-spectral-min-threshold` | 1.2 | Floor of the `D_spectral` threshold (1.2 for LGA and Constrain-and-Scale, 1.5 for the others) |
| `fedmast-ablate-squeeze` / `-spectral` / `-history` / `-round` | false | Disable one FedMAST component |

Attack parameters (`lga-*`, `neurotoxin-*`, `cas-*`, `bc-*`, `dba-*`, `3dfed-*`) are listed in
[section 7](#7-attacks); the remaining `3dfed-*`/`dba-*` keys are defined in `pyproject.toml`.

## 10. Reproducing the experiments

All experiments are `flwr run .` invocations that differ only in `--run-config`. Run everything from the
repository root after [section 2](#2-environment) and [section 3](#3-data-preparation). Runs write CSVs to the
working directory, so start each experiment from a clean directory (or use a distinct `simulation-id`) and
move the files away afterwards.

### 10.1 Build the clean FedMAST history (no attack)

The attack runs use a frozen clean history. `fedmast_history_clean.json` is shipped (50 rounds); to
regenerate it, run FedMAST without attackers in `build` mode (written to a new file so the shipped one is kept;
pass `fedmast-history-path='fedmast_history_rebuilt.json'` to later runs to use it):

```bash
RAY_DEDUP_LOGS=0 flwr run . --run-config "num-server-rounds=50 num-clients=100 local-epochs=5 \
  aggregation-method='fedmast' simulation-id='fedmast-clean-history' \
  backdoor-attack-mode='none' backdoor-attack-type='none' num-malicious-clients-per-round=0 \
  fedmast-suspicious-policy='fedavg' fedmast-history-mode='build' \
  fedmast-history-path='fedmast_history_rebuilt.json' \
  fedmast-spectral-percentile=90.0 fedmast-spectral-min-threshold=1.5"
```

### 10.2 Reported FedMAST runs (six attacks)

Common to all: `aggregation-method='fedmast'`, `backdoor-attack-mode='per-round-attack'`,
`num-clients=100`, `attack-end-round=999999`, `fedmast-history-mode='load-readonly'`,
`fedmast-history-path='fedmast_history_clean.json'`, `fedmast-suspicious-policy='coord-median'`,
`fedmast-threshold-k=3.0`, `fedmast-min-accept=5`, `fedmast-warmup-rounds=2`, `fedmast-buffer-rounds=20`,
`fedmast-trajectory-window=15`, `fedmast-spectral-min-appearances=5`, `fedmast-spectral-decay=0.7`.
The per-attack values that differ:

| Attack | Rounds | Mal. / round | Attack start | Spectral percentile | Spectral min threshold | Other |
|---|---|---|---|---|---|---|
| LGA | 200 | 3 | 20 | 85.0 | 1.2 | `lga-epochs=12 lga-lr=0.05 lga-tau=2.0` |
| 3DFed | 200 | 3 | 30 | 90.0 | 1.5 | `3dfed-num-backdoor-models=2 3dfed-num-decoy-models-init=1 3dfed-max-decoy-models=2`, other `3dfed-*` at their defaults |
| Neurotoxin | 200 | 1 | 1 | 90.0 | 1.5 | `neurotoxin-mask-ratio=0.01 neurotoxin-scale-factor=1.0` |
| BC-Layers | 200 | 1 | 1 | 90.0 | 1.5 | `bc-defense-sim='multikrum'`, `bc-*` at their defaults |
| Constrain-and-Scale | 200 | 1 | 1 | 85.0 | 1.2 | `cas-*` at their defaults |
| DBA | 200 | 4 | 1 | 90.0 | 1.5 | `backdoor-partition-ids='[5,12,23,36]'`, `dba-*` at their defaults (`dba-scale-factor=1.0`) |

Example (LGA, full length):

```bash
RAY_DEDUP_LOGS=0 flwr run . --run-config "num-server-rounds=200 num-clients=100 \
  aggregation-method='fedmast' simulation-id='lga-fedmast' \
  backdoor-attack-mode='per-round-attack' backdoor-attack-type='lga' \
  num-malicious-clients-per-round=3 num-malicious-clients=3 \
  backdoor-partition-ids='[5,12,23,36,44,51,67,78,83,91]' \
  attack-start-round=20 attack-end-round=999999 \
  lga-epochs=12 lga-lr=0.05 lga-tau=2.0 \
  fedmast-history-mode='load-readonly' fedmast-history-path='fedmast_history_clean.json' \
  fedmast-suspicious-policy='coord-median' \
  fedmast-spectral-percentile=85.0 fedmast-spectral-min-threshold=1.2"
```

Example (DBA):

```bash
RAY_DEDUP_LOGS=0 flwr run . --run-config "num-server-rounds=200 num-clients=100 \
  aggregation-method='fedmast' simulation-id='dba-fedmast' \
  backdoor-attack-mode='per-round-attack' backdoor-attack-type='dba' \
  num-malicious-clients-per-round=4 num-malicious-clients=4 \
  backdoor-partition-ids='[5,12,23,36]' dba-attacker-partition-ids='[5,12,23,36]' \
  attack-start-round=1 dba-epochs=6 dba-lr=0.05 dba-poison-ratio=0.078125 \
  dba-num-attackers=4 dba-scale-factor=1.0 dba-server-eta=1.0 \
  fedmast-history-mode='load-readonly' fedmast-history-path='fedmast_history_clean.json' \
  fedmast-spectral-percentile=90.0 fedmast-spectral-min-threshold=1.5"
```

The other four follow the same pattern: replace `backdoor-attack-type`, the attack parameters and the
table values above.

### 10.3 CovertLayers (FedMAST-aware adaptive attacker)

Set `backdoor-attack-type='bc-layers-lp-full' bc-defense-sim='fedmast'` and the four adaptive parameters.
Values used in the original experiment scripts:

| Setting | vs FedMAST | vs FedAvg / Multi-Krum / FLAME / AlignIns |
|---|---|---|
| Rounds | 200 | 200 (FLAME variant: attack start 50) |
| Attack start | 30 | 30 |
| Mal. / round (pool 10) | 1 | 3 (2 for trimmed mean) |
| `bc-lambda` | 0.7 | 0.6 (0.8 for FLAME) |
| `bc-lr` | 0.1 | 0.075 (0.05 for FLAME) |
| `bc-norm-match-weight` | 0.8 | 0.9 (0.5 for FLAME) |
| `bc-energy-smooth-noise` | 0.03 | 0.05 (0.005 for FLAME) |
| `bc-direction-blend-alpha` | 0.15 | 0.25 (0.05 for FLAME) |
| `bc-min-bsr-ratio` | 0.5 | 0.4 (0.6 for FLAME) |
| Unchanged | `bc-tau=0.95 bc-benign-epochs=2 bc-malicious-epochs=2 bc-proxy-count=5`, local-epochs 5 | same |

(`fedmast-spectral-percentile=90.0 fedmast-spectral-min-threshold=1.5` for the FedMAST defense.)

### 10.4 Baseline defenses

Run the same attack with `aggregation-method` set to `fedavg`, `multikrum`, `trimmed_mean`, `flame` or
`alignins` (no `fedmast-*` keys needed). The paper's Constrain-and-Scale and Neurotoxin Trimmed Mean
comparisons use `aggregation-method='trimmed_mean'`. Attack parameters used in the original baseline scripts:

| Attack | Baselines' settings |
|---|---|
| LGA | `lga-epochs=12 lga-lr=0.05 lga-tau=2.0`, 3 malicious / round, 200 rounds, attack start 20 (AlignIns) / 30 (FLAME) |
| 3DFed | 3 malicious / round, 200 rounds, attack start 30, `3dfed-num-backdoor-models=2 3dfed-num-decoy-models-init=1 3dfed-max-decoy-models=2`, `3dfed-kappa=1000`, other `3dfed-*` as in `pyproject.toml` |
| Constrain-and-Scale | 1 malicious / round, 200 rounds, attack start 1, `cas-*` at their defaults |
| Neurotoxin | 1 malicious / round, 200 rounds, `neurotoxin-mask-ratio=0.01 neurotoxin-scale-factor=1.0` |
| BC-Layers | 1 malicious / round, 200 rounds, `bc-defense-sim='multikrum'`, `bc-*` at their defaults |
| DBA | as in 10.2 |

### 10.5 Ablations and sensitivity studies

* **Component ablations:** add `fedmast-ablate-squeeze=true`, `fedmast-ablate-spectral=true`,
  `fedmast-ablate-history=true` or `fedmast-ablate-round=true` to a FedMAST run. The ablation runs use 100
  rounds, attack start 20, and the same attack settings as 10.2 (3DFed with 3 malicious/round, LGA with 3,
  BC-Layers/CovertLayers with 3 malicious/round from the pool of 10, Constrain-and-Scale with 1). The
  "no containment" ablation sets `fedmast-suspicious-policy='fedavg'`; `noop` is available as a second
  containment variant.
* **Cold start:** `fedmast-history-mode='none'` (no external history).
* **Data heterogeneity:** set `dirichlet-alpha` to 0.3 / 0.5 / 0.7. Build a matching clean history under the
  same alpha first (10.1 with the same `dirichlet-alpha`), as the original heterogeneity study did, and point
  `fedmast-history-path` at it.
* **Number of attackers:** vary `num-malicious-clients-per-round` (1, 2, 3, 5) on the LGA configuration.

## 11. Outputs and metrics

Written to the working directory (all git-ignored):

| File | Content |
|---|---|
| `per_round_centralized.csv` | Server-side evaluation on the 10,000-image test set each round: loss, main-task accuracy (`centralized_mta`), attack success rate (`centralized_asr`) |
| `per_round_distributed.csv` | Client-side evaluation averaged over the evaluated clients: `dist_mta`, `dist_asr` |
| `experiments.csv` | One row per finished run: configuration summary, final MTA/ASR, histories |
| `per_update_features_<aggregation>_<attack>.csv` | One row per (round, client): identifiers, the ground-truth malicious flag, the defense decision and score, all extracted structural features (the 22 bank features, the 6 spectral-drift features and the additional per-stage features) |

**Metric definitions** (`utils/evaluate_attack.py`, `task.py`)

* **MTA (main-task accuracy):** top-1 accuracy on clean test images.
* **ASR (attack success rate):** fraction of test images whose true label is not the target class (2), stamped
  with the trigger, that the model classifies as the target class (up to 1000 samples per evaluation).
  For DBA, `asr` is the global-trigger ASR (`dba_asr_global`) together with the per-local-pattern rates.
* **Detection counts (TP/FN/FP/TN, precision, recall and FPR)** are recoverable from the per-update CSV.
  The artifact defines FPR as `FP / (FP + TN)` over submitted benign updates and recall as
  `TP / (TP + FN)` over submitted malicious updates. Use `malicious_flag` as ground truth and
  `selected_by_aggregator == 0` as rejection.

Post-hoc analysis of a per-update CSV:

```bash
python -m fed_learning_cifar_experiment.utils.fedmast_diagnostic per_update_features_fedmast_lga.csv
python -m fed_learning_cifar_experiment.utils.deep_diagnostic   per_update_features_fedmast_lga.csv
python -m fed_learning_cifar_experiment.utils.verify_extraction per_update_features_fedmast_lga.csv
python scripts/summarize_detection.py per_update_features_fedmast_lga.csv --start-round 20
```

## 12. Seeds and determinism

Fixed seeds in the code base:

| Component | Seed |
|---|---|
| Dirichlet data partition (`load_data`, `dirichlet_indices`) | 42 |
| Per-client 80/20 train/test split | 42 |
| FLAME noise generator | `42 + round` |
| 3DFed per-round seed | `hash((round, 42))` |
| 3DFed noise-mask optimizer | the per-round seed above |

Client sampling, attacker sampling, benign hyperparameter draws (epochs and learning rate), the server-side
reference-update partitions, weight initialization and mini-batch order are drawn from unseeded generators, so
repeated runs of the same configuration are statistically comparable but not bit-identical. The partition and
the malicious pool are identical across runs.

## 13. Notes and known behaviours

* Every run starts from a randomly initialized model. No pretrained weights are included or loaded.
* `local-epochs` and `fraction-fit` are accepted by the run config but not consumed by the current client and
  strategy code: benign clients draw their own epochs/learning rate ([section 5](#5-simulation-structure)),
  attackers use the epochs given by their attack parameters, and 10 clients are sampled per round.
* The Krum base class in `state/krum_metrics_strategy.py` is shared infrastructure for the Multi-Krum, FLAME and
  AlignIns strategies; plain Krum is not exposed as a `aggregation-method`.
* `train_constrain_and_scale` in `task.py` (the Krum/Multi-Krum-oriented Constrain-and-Scale variant) is included
  but is not dispatched by `client_app.py`; the `constrain-and-scale` attack type runs
  `train_constrain_and_scale_for_fedavg_tm`.
* CIFAR-10 is read from `fed_learning_cifar_experiment/data/cifar10_hf`; running from a directory other than
  the repository root is fine, but `prepare_data` must have been executed once.
* `load-readonly` and `load-update-safe` fail closed if the requested history cannot be loaded, rather than
  silently degrading to a cold-start run.
* If FedMAST rejects every client in a round, aggregation fails closed with a no-op update. Rejected updates
  are never re-admitted solely because the accepted set is empty.

## Citation

Backdoors Leave Structural Traces: FedMAST for Backdoor Detection and Containment in Federated Learning.
The 38th IEEE International Conference on Tools with Artificial Intelligence (ICTAI).
