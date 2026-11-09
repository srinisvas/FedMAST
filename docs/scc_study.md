# FedMAST: Spectral Causal Criticality (SCC) feasibility study

**Status:** Phase 1 instrumentation and observational diagnostics. No new
backdoor attack is implemented. The published FedMAST code remains unchanged
unless `study-enabled=true`. No benchmark has been executed by this commit.

## Research question

Does a convolutional output channel's backdoor-specific influence depend on
*which spatial-frequency components of its parameter update are present*,
beyond generic channel saliency and generic frequency-band saliency?

This is a hypothesis, not an established novelty claim. The study is designed
to reject it if the effect is weak, confounded, or irreproducible.

## What is already available

The repository supports FedAvg, FedMAST, Multi-Krum, Trimmed Mean, AlignIns
and FLAME aggregation, with LGA, Neurotoxin, Constrain-and-Scale, DBA, 3DFed,
BC-Layers, and FedMAST-aware CovertLayers attacks. The matrix runner treats
BC-Layers and CovertLayers as distinct configurations of the same
`bc-layers-lp-full` dispatch. `none` is the clean baseline.

The existing CSVs contain centralized/distributed clean accuracy and ASR
and per-update structural features. This branch adds opt-in per-client update
norms, per-layer spatial DCT band energies, pre-aggregation client-model
snapshots, post-aggregation model snapshots and a reproducible run manifest.

**Important:** FedMAST's existing "spectral" features include matrix
singular-value diagnostics. Those are not spatial-frequency DCT coefficients.

## Setup

```bash
git checkout research/spectral-causal-criticality
pip install -e .
python -m fed_learning_cifar_experiment.prepare_data
python scripts/validate_artifact.py
python -m unittest discover -s tests -v
```

The data preparation step downloads CIFAR-10. Simulation requires Flower/Ray
and potentially substantial CPU/GPU resources.

## Plan, then run

```bash
# Preview the matrix, with no execution
python scripts/run_scc_matrix.py --profile smoke
python scripts/run_scc_matrix.py --profile pilot
python scripts/run_scc_matrix.py --profile full

# Run only when a suitable environment is available
python scripts/run_scc_matrix.py --profile smoke --execute
python scripts/run_scc_matrix.py --profile pilot --execute

# Run a narrow subset of the full design
python scripts/run_scc_matrix.py --profile full \
  --attacks none,lga,bc-layers,covertlayers \
  --aggregations fedavg,fedmast --seeds 11,29,47 --execute

python scripts/scc_summarize.py --root study_runs
```

The predefined matrices contain:

| Profile | Rounds | Attacks incl. benign | Aggregators | Seeds | Runs |
| --- | ---: | ---: | ---: | ---: | ---: |
| smoke | 4 | 2 | 2 | 1 | 4 |
| pilot | 25 | 8 | 2 | 2 | 32 |
| full | 100 | 8 | 6 | 3 | 144 |

The attack window is rounds 2-3 (smoke), 5-15 (pilot), or 20-60
(full). Snapshots include attack and post-attack rounds. The full matrix
is **expensive**, particularly 3DFed and Neurotoxin. Start with smoke,
inspect outputs, then pilot. The driver is sequential, dry-run by default,
writes `driver_status.json` and `flower.log` per run, and skips successful
jobs on resume. Use `--retry` to rerun them.

**Budget consistency:** Every attacked run uses four malicious clients per
round to accommodate DBA's four-way trigger. This deliberately differs from
the published demo's three-client default. The attack budgets are not
otherwise matched: methods use different epochs, local training losses,
trigger structures and update norms. Their raw ASR values are therefore
**descriptive**, not a controlled comparison of attack efficiency. DBA's
trigger differs from the other attacks and must be reported separately.

## Opt-in configuration and outputs

New Flower run-config keys:

- `study-enabled` (default false)
- `study-run-dir` (default `study_runs/default`)
- `study-seed` (default 42)
- `study-capture-rounds` (JSON list string, default `[3,10]`)
- `study-spectral-summary` (default true)

In each run directory:

- `study_run.json`: effective Flower run config, timestamp and schema
- `per_round_centralized.csv`, `per_round_distributed.csv`,
  `per_update_features_*.csv`, `experiments.csv`: existing FedMAST logs,
  isolated per run
- `per_client_updates.jsonl`: client ID, partition ID, ground-truth
  malicious flag, trainable update L2 and changed parameter count
- `per_layer_spectral.jsonl`: each eligible conv layer's low/mid/high
  orthonormal DCT-II update energy and energy fraction
- `per_round_study.jsonl`: received clients, failures, aggregate update L2
- `snapshots/round_NNNN.npz` + `.json`: global model, all received
  client models, returned aggregate model, exact state_dict ordering,
  per-client metadata
- `flower.log` and `driver_status.json`: runner logs and exit code

**Privacy and storage:** Snapshots include full received client model
parameters and ground-truth attacker labels. Keep them in an authorized
isolated environment. They are git-ignored. Do not publish snapshots without
reviewing data and model privacy. A single round of 10 clients can consume
several MB compressed. Capturing every round across 144 runs is not advised.

## Offline channel x frequency diagnostics

First inspect `snapshots/round_NNNN.json` and choose a malicious and a
benign `client_index`. Then run the same analysis on both:

```bash
python scripts/scc_counterfactual.py \
  study_runs/<run_id>/snapshots/round_0010.npz \
  --client-index 0 --max-layers 2 --max-channels 4 --max-samples 512

# Repeat on a held-out snapshot, preferably another seed or round
python scripts/scc_counterfactual.py \
  study_runs/<other_run_id>/snapshots/round_0010.npz \
  --client-index 0 --selection random --max-samples 512

python scripts/scc_interactions.py 'study_runs/**/counterfactual_*.csv' \
  --output study_runs/scc_interactions.csv
```

The default diagnostic selects the first and last eligible 3x3 convolutional
layers. You can specify exact names with `--layers`. Channel sampling by
update energy is a **heuristic only**, not evidence of criticality. Use
`--selection random` and held-out runs to detect selection bias.

For each channel `c` and band `b`, let `D` be a received client update
relative to the server model. The orthonormal DCT-II over the last two
kernel dimensions defines `P_cb(D)`. The diagnostic evaluates:

- Full received client model, `W + D`
- Spectral counterfactual, `W + D - P_cb(D)`
- Coordinate control, `W + D - Q_cb(D)`

`Q_cb` keeps the same output channel, the same number of raw-coordinate
degrees of freedom, and the same L2 intervention energy as `P_cb`.
For every condition, clean accuracy and targeted ASR are evaluated on
exactly the same fixed CIFAR-10 subset. The counterfactual CLI detects
DBA from `study_run.json` and evaluates its combined global DBA trigger,
rather than the standard patch. Use `--trigger-type` to override if needed. The ASR denominator excludes
examples whose clean label already equals the target label.

This is **ablation of an observed update**, not construction of a new
poisoning method. The comparison does not establish equal clean damage,
and an individual received client model is not a counterfactual server
aggregate. For FedMAST and other non-linear aggregators, do not attribute
the difference to a changed server decision.

## Interaction and falsification

For each complete channel-by-band matrix of ASR effects `I_cb`, compute:

```text
gamma_cb = I_cb - mean_b(I_cb) - mean_c(I_cb) + mean_cb(I_cb)
```

`gamma` is a two-way centered descriptive residual. It is **not** by
itself proof of a causal channel-frequency interaction. Report it
alongside the corresponding clean-accuracy residual, matched-coordinate
effects, update norms, malicious/benign controls and confidence intervals
across *independent runs*, not across channels as pseudo-replicates.

Pre-register a discovery/validation split by run seed or checkpoint.
Discover high-effect pairs only on discovery runs, freeze the selection
rule, and test whether their ranking and effect generalize to held-out
runs. Use clean accuracy, ASR and post-attack persistence separately.

**Proceed to a new attack design only if**:

1. Backdoor-specific channel-frequency effects survive matched-energy
   coordinate controls and are not explained by clean-accuracy damage.
2. Channel-by-band interaction patterns replicate across independent seeds
   and held-out checkpoints, including different attacker methods.
3. Effects remain informative after aggregation, with a separate analysis
   of the defense's nonlinear selection behavior.
4. A later controlled factorial study demonstrates that joint
   channel-frequency selection outperforms channel-only, frequency-only,
   sequential selection and matched random controls at equal budgets.

If these fail, stop rather than describing the combination as novel.

## Limitations and work still needed for a paper

- Only TinyResNet18/CIFAR-10 is currently present in the repository.
  External validity requires at least one additional model and dataset.
- Spatial DCT is meaningful for 3x3 convolution kernels. The study
  excludes 1x1 convolutions, dense layers, BatchNorm and arbitrary
  flattened-neuron transforms. It does not claim spatial frequency in
  channel-index space.
- The dataset partition seed remains fixed at 42 in the original
  implementation. `study-seed` seeds server RNG and each client's
  round-local RNG, but does not independently resample partitions.
  Flower/Ray scheduling and GPU kernels can still be nondeterministic.
- Channel indices are not aligned across independently initialized models
  because hidden channels can be permuted. Validate a *selection rule* or
  distributional pattern across seeds, not the identity of channel number 7.
- With 3x3 kernels there are only nine spatial DCT coefficients per
  input-output kernel, so the low/mid/high partition is deliberately coarse.
- Snapshot interventions are local diagnostics. A full factorial attack
  comparison, matched-parameter budgets, joint selection training,
  held-out transfer and server-side counterfactual aggregation are
  **not yet implemented**.
- No experiments have been run or validated through this GitHub connector.
  Unit tests and a smoke simulation must pass in an execution environment
  before trusting any results.

## Safety

This branch is intended for authorized, isolated academic evaluation.
It adds observability and counterfactual analysis, not deployment,
stealth optimization, or bypass logic for external federations.
