from __future__ import annotations

import copy
import json
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters


def _trainable_param_keys(model: nn.Module) -> List[str]:
    return [n for n, p in model.named_parameters() if p.requires_grad]


def _state_dict_to_flat(sd: dict, keys: List[str]) -> torch.Tensor:
    parts = [sd[k].detach().float().reshape(-1) for k in keys]
    if parts:
        target = parts[0].device
        parts = [p.to(target) for p in parts]
    return torch.cat(parts)


def _flat_to_state_dict(
    flat: torch.Tensor, sd_template: dict, keys: List[str],
) -> dict:
    out = {k: v.clone() for k, v in sd_template.items()}
    offset = 0
    for k in keys:
        numel = out[k].numel()
        out[k] = flat[offset : offset + numel].reshape(out[k].shape).to(out[k].dtype)
        offset += numel
    return out


def read_flat_param_value(sd: dict, keys: List[str], flat_idx: int) -> float:
    offset = 0
    for k in keys:
        t = sd[k]
        if flat_idx < offset + t.numel():
            return t.detach().float().reshape(-1)[flat_idx - offset].item()
        offset += t.numel()
    raise IndexError(f"flat_idx {flat_idx} out of range")


def write_flat_param_value(
    sd: dict, keys: List[str], flat_idx: int, value: float,
) -> None:
    offset = 0
    for k in keys:
        t = sd[k]
        if flat_idx < offset + t.numel():
            local_idx = flat_idx - offset
            flat_view = t.detach().float().reshape(-1)
            flat_view[local_idx] = value
            sd[k] = flat_view.reshape(t.shape).to(t.dtype)
            return
        offset += t.numel()
    raise IndexError(f"flat_idx {flat_idx} out of range")


@dataclass
class NeuronGroup:
    key: str
    neuron_idx: int
    flat_indices: List[int]


def build_neuron_groups(
    trainable_keys: List[str], sd: dict,
) -> List[NeuronGroup]:
    groups = []
    offset = 0
    for key in trainable_keys:
        t = sd[key]
        if t.ndim >= 2:
            n_neurons = t.shape[0]
            per_neuron = t.numel() // n_neurons
            for i in range(n_neurons):
                start = offset + i * per_neuron
                indices = list(range(start, start + per_neuron))
                groups.append(NeuronGroup(key=key, neuron_idx=i, flat_indices=indices))
        else:
            for i in range(t.numel()):
                groups.append(NeuronGroup(key=key, neuron_idx=i, flat_indices=[offset + i]))
        offset += t.numel()
    return groups


def train_3dfed_constrained_backdoor(
    net: nn.Module,
    clean_data,
    poisoned_data,
    device: torch.device,
    init_sd: dict,
    *,
    beta: float = 0.3,
    gamma: float = 1.0,
    epochs: int = 3,
    lr: float = 0.01,
    target_label: int = 2,
) -> Tuple[dict, dict]:
    net.to(device)
    net.train()

    trainable_keys = _trainable_param_keys(net)
    global_flat = _state_dict_to_flat(
        {k: v.to(device) for k, v in init_sd.items()}, trainable_keys
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)

    running_task_loss = 0.0
    running_prox_loss = 0.0
    num_steps = 0

    for epoch in range(epochs):
        poison_iter = iter(poisoned_data)
        for batch in clean_data:
            if isinstance(batch, dict):
                images_c, labels_c = batch["img"].to(device), batch["label"].to(device)
            else:
                images_c, labels_c = batch[0].to(device), batch[1].to(device)

            try:
                p_batch = next(poison_iter)
            except StopIteration:
                poison_iter = iter(poisoned_data)
                p_batch = next(poison_iter)

            if isinstance(p_batch, dict):
                images_p, labels_p = p_batch["img"].to(device), p_batch["label"].to(device)
            else:
                images_p, labels_p = p_batch[0].to(device), p_batch[1].to(device)

            images = torch.cat([images_c, images_p], dim=0)
            labels = torch.cat([labels_c, labels_p], dim=0)

            optimizer.zero_grad(set_to_none=True)
            task_loss = criterion(net(images), labels)

            current_flat = parameters_to_vector(
                [p for n, p in net.named_parameters() if p.requires_grad]
            )
            prox_loss = torch.norm(current_flat - global_flat, p=2)

            loss = (1.0 - beta) * task_loss + beta * prox_loss
            loss.backward()
            optimizer.step()

            running_task_loss += task_loss.item()
            running_prox_loss += prox_loss.item()
            num_steps += 1

    trained_sd = {k: v.clone() for k, v in net.state_dict().items()}
    if abs(gamma - 1.0) > 1e-6:
        g_dev = {k: v.to(device) for k, v in init_sd.items()}
        for k in trainable_keys:
            delta = trained_sd[k].float() - g_dev[k].float()
            trained_sd[k] = (g_dev[k].float() + gamma * delta).to(init_sd[k].dtype)

    metrics = {
        "task_loss": running_task_loss / max(num_steps, 1),
        "prox_loss": running_prox_loss / max(num_steps, 1),
        "update_norm": float(
            torch.norm(
                _state_dict_to_flat(
                    {k: v.cpu() for k, v in trained_sd.items()}, trainable_keys
                )
                - _state_dict_to_flat(
                    {k: v.cpu() for k, v in init_sd.items()}, trainable_keys
                )
            ).item()
        ),
        "beta": beta, "gamma": gamma,
    }
    return trained_sd, metrics


def optimize_3dfed_noise_masks(
    backdoor_sd: dict,
    global_sd: dict,
    trainable_keys: List[str],
    m: int,
    *,
    alpha: float = 0.5,
    lambda_init: float = 1.0,
    dual_step: float = 0.1,
    steps: int = 20,
    lr: float = 0.01,
    selected_neuron_ratio: float = 0.3,
    seed: int = 0,
    device: torch.device = torch.device("cpu"),
) -> Tuple[List[dict], dict]:
    if m < 2:
        print("[3DFed][NoiseMask] m < 2, cannot cancel. Skipping.")
        return [copy.deepcopy(backdoor_sd)], {"noise_mask_skipped": True, "reason": "m<2"}

    rng = torch.Generator(device=device)
    rng.manual_seed(seed)

    bd_flat = _state_dict_to_flat(backdoor_sd, trainable_keys).to(device)
    g_flat = _state_dict_to_flat(global_sd, trainable_keys).to(device)
    delta_flat = bd_flat - g_flat
    total_params = delta_flat.numel()

    neuron_groups = build_neuron_groups(trainable_keys, global_sd)
    neuron_ups = []
    for ng in neuron_groups:
        up = sum(abs(delta_flat[idx].item()) for idx in ng.flat_indices)
        neuron_ups.append(up)

    n_selected = max(1, int(selected_neuron_ratio * len(neuron_groups)))
    sorted_indices = sorted(range(len(neuron_ups)), key=lambda i: neuron_ups[i])
    low_up_neuron_ids = sorted_indices[:n_selected]

    low_up_flat_indices = []
    for nid in low_up_neuron_ids:
        low_up_flat_indices.extend(neuron_groups[nid].flat_indices)
    low_up_tensor = torch.tensor(low_up_flat_indices, device=device, dtype=torch.long)

    masks = [
        torch.randn(total_params, device=device, generator=rng) * 1e-4
        for _ in range(m)
    ]
    for mask in masks:
        mask.requires_grad_(True)

    optimizer = torch.optim.Adam(masks, lr=lr)
    lam = lambda_init

    for step in range(steps):
        optimizer.zero_grad()

        l_ups = torch.tensor(0.0, device=device)
        for mask in masks:
            masked_delta = delta_flat + mask
            for nid in low_up_neuron_ids:
                ng = neuron_groups[nid]
                idx_t = torch.tensor(ng.flat_indices, device=device, dtype=torch.long)
                neuron_up = masked_delta[idx_t].abs().sum() + 1e-8
                l_ups = l_ups + 1.0 / neuron_up
        l_ups = l_ups / (m * len(low_up_neuron_ids))

        l_norm = sum(mask.norm(p=2) for mask in masks) / m

        mask_sum = sum(masks)
        l_constrain = mask_sum.norm(p=2)

        loss = alpha * l_ups + (1.0 - alpha) * l_norm + lam * l_constrain
        loss.backward()
        optimizer.step()

        lam = lam + dual_step * l_constrain.item()

    with torch.no_grad():
        mask_mean = sum(masks) / m
        masks = [mask - mask_mean for mask in masks]

    masked_sds = []
    for mask in masks:
        masked_flat = bd_flat + mask.detach()
        masked_sds.append(_flat_to_state_dict(masked_flat, backdoor_sd, trainable_keys))

    metrics = {
        "noise_mask_skipped": False,
        "m": m, "alpha": alpha,
        "neuron_groups_total": len(neuron_groups),
        "low_up_neurons_selected": len(low_up_neuron_ids),
        "mask_norms": [mask.detach().norm().item() for mask in masks],
        "zero_sum_residual": sum(masks).detach().norm().item(),
    }

    print(
        f"[3DFed][NoiseMask] m={m} | alpha={alpha:.3f} | "
        f"neurons={len(low_up_neuron_ids)}/{len(neuron_groups)} | "
        f"zero_sum={metrics['zero_sum_residual']:.6f}"
    )
    return masked_sds, metrics


def find_3dfed_indicators(
    model: nn.Module, data_loader, device: torch.device,
    *, candidate_ratio: float = 0.01, indicator_count: int = 32,
    hessian_samples: int = 4,
) -> List[int]:
    model.to(device)
    model.eval()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in trainable_params)
    criterion = nn.CrossEntropyLoss()

    grad_accum = torch.zeros(total_params, device=device)
    n_b = 0
    for batch in data_loader:
        if isinstance(batch, dict):
            imgs, lbls = batch["img"].to(device), batch["label"].to(device)
        else:
            imgs, lbls = batch[0].to(device), batch[1].to(device)
        model.zero_grad()
        criterion(model(imgs), lbls).backward()
        offset = 0
        for p in trainable_params:
            if p.grad is not None:
                grad_accum[offset:offset+p.numel()] += p.grad.detach().reshape(-1).abs()
            offset += p.numel()
        n_b += 1
        if n_b >= hessian_samples * 2:
            break
    grad_accum /= max(n_b, 1)

    n_cand = max(indicator_count * 4, int(candidate_ratio * total_params))
    _, cand_idx = torch.topk(grad_accum, min(n_cand, total_params), largest=False)

    fisher = torch.zeros(total_params, device=device)
    n_f = 0
    for batch in data_loader:
        if isinstance(batch, dict):
            imgs, lbls = batch["img"].to(device), batch["label"].to(device)
        else:
            imgs, lbls = batch[0].to(device), batch[1].to(device)
        model.zero_grad()
        criterion(model(imgs), lbls).backward()
        offset = 0
        for p in trainable_params:
            if p.grad is not None:
                fisher[offset:offset+p.numel()] += p.grad.detach().reshape(-1) ** 2
            offset += p.numel()
        n_f += 1
        if n_f >= hessian_samples:
            break
    fisher /= max(n_f, 1)

    cand_curv = fisher[cand_idx]
    _, best = torch.topk(cand_curv, min(indicator_count, len(cand_curv)), largest=False)
    indices = sorted(cand_idx[best].cpu().tolist())
    print(f"[3DFed][Indicator] Selected {len(indices)} indicators (Fisher approx)")
    return indices


@dataclass
class IndicatorRecord:
    flat_idx: int
    original_delta: float
    implanted_delta: float
    slot_index: int
    role: str
    round_num: int
    param_name: str = ""
    alpha_used: float = 0.0

    def to_dict(self) -> dict:
        return {
            "flat_idx": self.flat_idx,
            "original_delta": self.original_delta,
            "implanted_delta": self.implanted_delta,
            "slot_index": self.slot_index,
            "role": self.role,
            "round_num": self.round_num,
            "param_name": self.param_name,
            "alpha_used": self.alpha_used,
        }

    @staticmethod
    def from_dict(d: dict) -> "IndicatorRecord":
        return IndicatorRecord(**{k: v for k, v in d.items() if k in IndicatorRecord.__dataclass_fields__})


def implant_3dfed_indicators(
    candidate_sd: dict, global_sd: dict, trainable_keys: List[str],
    indicator_indices: List[int], slot_index: int,
    kappa: float = 100000.0, role: str = "backdoor", round_num: int = 0,
) -> Tuple[dict, List[IndicatorRecord]]:
    out_sd = {k: v.clone() for k, v in candidate_sd.items()}
    records = []
    if not indicator_indices:
        return out_sd, records

    idx = indicator_indices[slot_index % len(indicator_indices)]
    g_val = read_flat_param_value(global_sd, trainable_keys, idx)
    c_val = read_flat_param_value(out_sd, trainable_keys, idx)
    delta = c_val - g_val

    eps = 1e-7
    if abs(delta) < eps:
        delta = eps if delta >= 0 else -eps

    implanted = kappa * delta
    write_flat_param_value(out_sd, trainable_keys, idx, g_val + implanted)

    pname = ""
    offset = 0
    for k in trainable_keys:
        n = global_sd[k].numel()
        if idx < offset + n:
            pname = k
            break
        offset += n

    rec = IndicatorRecord(
        flat_idx=idx, original_delta=delta, implanted_delta=implanted,
        slot_index=slot_index, role=role, round_num=round_num, param_name=pname,
    )
    records.append(rec)
    print(
        f"[3DFed][Indicator] Implanted slot={slot_index} role={role} | "
        f"idx={idx} | delta={delta:.2e} -> {implanted:.2e}"
    )
    return out_sd, records


@dataclass
class IndicatorFeedback:
    flat_idx: int
    slot_index: int
    role: str
    round_num: int
    feedback_ratio: float
    status: str


def read_3dfed_indicator_feedback(
    current_global_sd: dict, previous_global_sd: dict,
    trainable_keys: List[str], prev_records: List[IndicatorRecord],
    kappa: float = 100000.0,
) -> Tuple[List[IndicatorFeedback], bool]:
    feedbacks = []
    indicator_disabled = False

    for rec in prev_records:
        try:
            cur = read_flat_param_value(current_global_sd, trainable_keys, rec.flat_idx)
            prev = read_flat_param_value(previous_global_sd, trainable_keys, rec.flat_idx)
        except IndexError:
            continue

        if abs(rec.implanted_delta) < 1e-15:
            feedbacks.append(IndicatorFeedback(
                flat_idx=rec.flat_idx, slot_index=rec.slot_index,
                role=rec.role, round_num=rec.round_num,
                feedback_ratio=0.0, status="rejected",
            ))
            continue

        ratio = (cur - prev) / rec.implanted_delta

        if abs(ratio) > 1.0:
            status = "unknown_external_noise"
            indicator_disabled = True
        elif ratio <= 1.0 / kappa:
            status = "rejected"
        else:
            status = "accepted"

        feedbacks.append(IndicatorFeedback(
            flat_idx=rec.flat_idx, slot_index=rec.slot_index,
            role=rec.role, round_num=rec.round_num,
            feedback_ratio=ratio, status=status,
        ))
        print(
            f"[3DFed][Indicator] Read slot={rec.slot_index} "
            f"role={rec.role} | ratio={ratio:.6f} -> {status}"
        )

    accepted = [f for f in feedbacks if f.status == "accepted"]
    if len(accepted) >= 2:
        max_r = max(f.feedback_ratio for f in accepted)
        for f in accepted:
            if f.feedback_ratio < max_r / 2.0:
                f.status = "clipped"

    return feedbacks, indicator_disabled


def train_3dfed_decoy(
    benign_sd: dict, backdoor_avg_sd: dict,
    clean_data, device: torch.device,
    trainable_keys: List[str], net_factory,
    assigned_decoy_index: int,
    *, steps: int = 20, lr: float = 0.01,
) -> Tuple[dict, dict]:
    b_flat = _state_dict_to_flat(benign_sd, trainable_keys).to(device)
    x_flat = _state_dict_to_flat(backdoor_avg_sd, trainable_keys).to(device)
    diff = (b_flat - x_flat).abs()
    _, sorted_idx = torch.sort(diff)
    garbage_idx = sorted_idx[assigned_decoy_index % len(sorted_idx)].item()

    decoy_net = net_factory()
    decoy_net.load_state_dict(benign_sd)
    decoy_net.to(device)
    decoy_net.train()

    criterion = nn.CrossEntropyLoss()
    ref_net = net_factory()
    ref_net.load_state_dict(benign_sd)
    ref_net.to(device)
    ref_net.eval()

    ref_losses = []
    for i, batch in enumerate(clean_data):
        if isinstance(batch, dict):
            imgs, lbls = batch["img"].to(device), batch["label"].to(device)
        else:
            imgs, lbls = batch[0].to(device), batch[1].to(device)
        with torch.no_grad():
            ref_losses.append(criterion(ref_net(imgs), lbls).item())
        if i >= 2:
            break
    ref_loss = sum(ref_losses) / max(len(ref_losses), 1)
    del ref_net

    benign_val = b_flat[garbage_idx].item()
    optimizer = torch.optim.SGD(decoy_net.parameters(), lr=lr, momentum=0.9)
    data_iter = iter(clean_data)

    for step in range(steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(clean_data)
            batch = next(data_iter)

        if isinstance(batch, dict):
            imgs, lbls = batch["img"].to(device), batch["label"].to(device)
        else:
            imgs, lbls = batch[0].to(device), batch[1].to(device)

        optimizer.zero_grad(set_to_none=True)

        decoy_params = parameters_to_vector(
            [p for p in decoy_net.parameters() if p.requires_grad]
        )
        l1 = -torch.abs(decoy_params[garbage_idx] - benign_val)
        l2 = torch.relu(criterion(decoy_net(imgs), lbls) - ref_loss)

        (l1 + l2).backward()
        optimizer.step()

    decoy_sd = {k: v.clone() for k, v in decoy_net.state_dict().items()}
    d_flat = _state_dict_to_flat(decoy_sd, trainable_keys).to(device)

    metrics = {
        "decoy_index": assigned_decoy_index,
        "garbage_flat_idx": garbage_idx,
        "decoy_delta_norm": float((d_flat - b_flat).norm().item()),
    }
    print(f"[3DFed][Decoy] idx={assigned_decoy_index} | delta_norm={metrics['decoy_delta_norm']:.6f}")
    return decoy_sd, metrics


@dataclass
class AttackerCoordinator:
    alpha: float = 0.5
    alpha_step: float = 0.1
    k_decoy: int = 0
    max_decoy: int = 10

    indicator_indices: List[int] = field(default_factory=list)
    indicator_disabled: bool = False
    prev_indicator_records: List[IndicatorRecord] = field(default_factory=list)

    kappa: float = 100000.0
    num_backdoor_models: int = 4
    canonical_partition_id: int = 5

    round_history: List[dict] = field(default_factory=list)

    def read_and_adapt(
        self,
        current_global_sd: dict,
        previous_global_sd: dict,
        trainable_keys: List[str],
        server_round: int,
    ) -> None:
        if not self.prev_indicator_records:
            return

        feedbacks, disabled = read_3dfed_indicator_feedback(
            current_global_sd=current_global_sd,
            previous_global_sd=previous_global_sd,
            trainable_keys=trainable_keys,
            prev_records=self.prev_indicator_records,
            kappa=self.kappa,
        )

        self._update_adaptive_state(feedbacks, disabled)
        self._log_round(server_round, feedbacks)

    def _update_adaptive_state(
        self,
        feedbacks: List[IndicatorFeedback],
        indicator_disabled: bool,
    ) -> None:
        if indicator_disabled:
            self.indicator_disabled = True
            print("[3DFed][Coordinator] Indicator disabled (external noise)")
            return
        self.indicator_disabled = False

        bd_fb = [f for f in feedbacks if f.role == "backdoor"]
        dc_fb = [f for f in feedbacks if f.role == "decoy"]
        bd_acc = [f for f in bd_fb if f.status == "accepted"]
        dc_acc = [f for f in dc_fb if f.status == "accepted"]

        if len(bd_acc) >= 1:
            self.alpha = max(0.01, self.alpha - self.alpha_step / 2)
        elif len(bd_fb) > 0:
            self.alpha = min(0.99, self.alpha + self.alpha_step)

        all_rejected = len(bd_acc) == 0 and len(dc_acc) == 0 and len(feedbacks) > 0
        if all_rejected:
            self.k_decoy = min(self.k_decoy + 1, self.max_decoy)
        elif len(dc_acc) > 0:
            self.k_decoy = max(0, self.k_decoy - len(dc_acc))

        print(
            f"[3DFed][Coordinator] Adapted: "
            f"bd_acc={len(bd_acc)}/{len(bd_fb)} dc_acc={len(dc_acc)}/{len(dc_fb)} | "
            f"alpha={self.alpha:.4f} k_decoy={self.k_decoy}"
        )

    def assign_roles(
        self, malicious_cids: List[str], server_round: int,
    ) -> List[dict]:
        m = min(self.num_backdoor_models, len(malicious_cids))
        k = min(self.k_decoy, max(0, len(malicious_cids) - m))
        round_seed = hash((server_round, 42)) & 0xFFFFFFFF

        shared_alpha = self.alpha

        assignments = []
        for i, cid in enumerate(malicious_cids):
            if i < m:
                role, slot = "backdoor", i
            elif i < m + k:
                role, slot = "decoy", i - m
            else:
                role, slot = "backdoor", i

            assignments.append({
                "3dfed-role": role,
                "3dfed-slot-index": slot,
                "3dfed-round-seed": round_seed,
                "3dfed-num-backdoor-models": m,
                "3dfed-num-decoys": k,
                "3dfed-alpha": shared_alpha,
                "3dfed-k-decoy": self.k_decoy,
                "3dfed-canonical-partition-id": self.canonical_partition_id,
            })

        print(
            f"[3DFed][Coordinator] Round {server_round} | "
            f"m={m} bd, k={k} dc | alpha={shared_alpha:.4f}"
        )
        return assignments

    def receive_indicator_records(self, records_json_list: List[str]) -> None:
        self.prev_indicator_records = []
        for raw in records_json_list:
            try:
                items = json.loads(raw) if isinstance(raw, str) else raw
                for d in items:
                    self.prev_indicator_records.append(IndicatorRecord.from_dict(d))
            except Exception as e:
                print(f"[3DFed][Coordinator] Failed to parse indicator record: {e}")

        print(
            f"[3DFed][Coordinator] Received {len(self.prev_indicator_records)} "
            f"indicator records from clients"
        )

    def _log_round(self, rnd: int, feedbacks: List[IndicatorFeedback]) -> None:
        self.round_history.append({
            "round": rnd,
            "alpha": self.alpha,
            "k_decoy": self.k_decoy,
            "indicator_disabled": self.indicator_disabled,
            "n_accepted": sum(1 for f in feedbacks if f.status == "accepted"),
            "n_rejected": sum(1 for f in feedbacks if f.status in ("rejected", "clipped")),
            "n_noise": sum(1 for f in feedbacks if f.status == "unknown_external_noise"),
        })


def _train_benign_reference(
    net_factory, clean_data, device, init_sd, epochs=2, lr=0.1,
) -> dict:
    net = net_factory()
    net.load_state_dict(init_sd)
    net.to(device)
    net.train()
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    for epoch in range(epochs):
        for batch in clean_data:
            if isinstance(batch, dict):
                imgs, lbls = batch["img"].to(device), batch["label"].to(device)
            else:
                imgs, lbls = batch[0].to(device), batch[1].to(device)
            opt.zero_grad(set_to_none=True)
            criterion(net(imgs), lbls).backward()
            opt.step()
    return {k: v.clone() for k, v in net.state_dict().items()}


def train_3dfed(
    net: nn.Module,
    clean_data,
    poisoned_data,
    device: torch.device,
    init_sd: dict,
    net_factory,
    *,
    role: str = "backdoor",
    slot_index: int = 0,
    round_seed: int = 0,
    num_backdoor_models: int = 4,
    num_decoys: int = 0,
    server_round: int = 0,
    use_indicator: bool = True,
    indicator_indices: Optional[List[int]] = None,
    kappa: float = 100000.0,
    indicator_candidate_ratio: float = 0.01,
    indicator_count: int = 32,
    hessian_samples: int = 4,
    use_constrained_loss: bool = True,
    beta: float = 0.3,
    gamma_scale: float = 1.0,
    cl_epochs: int = 3,
    cl_lr: float = 0.01,
    target_label: int = 2,
    use_noise_mask: bool = True,
    alpha: float = 0.5,
    noise_lambda: float = 1.0,
    noise_dual_step: float = 0.1,
    noise_steps: int = 20,
    noise_lr: float = 0.01,
    use_decoy: bool = True,
    decoy_steps: int = 20,
    decoy_lr: float = 0.01,
) -> Tuple[dict, dict, List[IndicatorRecord]]:
    trainable_keys = _trainable_param_keys(net)
    all_metrics: Dict[str, Any] = {
        "role": role, "slot_index": slot_index, "server_round": server_round,
    }

    bd_net = copy.deepcopy(net)
    bd_net.load_state_dict(init_sd)
    bd_net.to(device)

    effective_beta = beta if use_constrained_loss else 0.0
    bd_sd, cl_m = train_3dfed_constrained_backdoor(
        bd_net, clean_data, poisoned_data, device, init_sd,
        beta=effective_beta, gamma=gamma_scale, epochs=cl_epochs,
        lr=cl_lr, target_label=target_label,
    )
    all_metrics["constrained_loss"] = cl_m

    if role == "backdoor":
        if use_noise_mask and num_backdoor_models >= 2:
            masked_sds, nm_m = optimize_3dfed_noise_masks(
                bd_sd, init_sd, trainable_keys, num_backdoor_models,
                alpha=alpha, lambda_init=noise_lambda,
                dual_step=noise_dual_step, steps=noise_steps,
                lr=noise_lr, seed=round_seed, device=device,
            )
            all_metrics["noise_mask"] = nm_m
            outgoing_sd = masked_sds[slot_index % len(masked_sds)]
        else:
            outgoing_sd = bd_sd
            all_metrics["noise_mask"] = {
                "noise_mask_skipped": True,
                "reason": "disabled" if not use_noise_mask else f"m={num_backdoor_models}<2",
            }

    elif role == "decoy":
        if not use_decoy:
            outgoing_sd = _train_benign_reference(
                net_factory, clean_data, device, init_sd,
            )
            all_metrics["decoy"] = {"ablation": "decoy disabled"}
        else:
            if use_noise_mask and num_backdoor_models >= 2:
                masked_sds, _ = optimize_3dfed_noise_masks(
                    bd_sd, init_sd, trainable_keys, num_backdoor_models,
                    alpha=alpha, lambda_init=noise_lambda,
                    dual_step=noise_dual_step, steps=noise_steps,
                    lr=noise_lr, seed=round_seed, device=device,
                )
                x_avg_sd = {}
                for key in masked_sds[0]:
                    x_avg_sd[key] = torch.stack(
                        [sd[key].float() for sd in masked_sds]
                    ).mean(0).to(masked_sds[0][key].dtype)
            else:
                x_avg_sd = bd_sd

            benign_sd = _train_benign_reference(
                net_factory, clean_data, device, init_sd,
            )
            outgoing_sd, dec_m = train_3dfed_decoy(
                benign_sd, x_avg_sd, clean_data, device,
                trainable_keys, net_factory, slot_index,
                steps=decoy_steps, lr=decoy_lr,
            )
            all_metrics["decoy"] = dec_m
    else:
        raise ValueError(f"Unknown role: {role}")

    ind_records: List[IndicatorRecord] = []
    if use_indicator:
        if not indicator_indices:
            ind_net = net_factory()
            ind_net.load_state_dict(init_sd)
            indicator_indices = find_3dfed_indicators(
                ind_net, clean_data, device,
                candidate_ratio=indicator_candidate_ratio,
                indicator_count=indicator_count,
                hessian_samples=hessian_samples,
            )
            all_metrics["indicator_finding"] = {
                "count": len(indicator_indices),
                "indices": indicator_indices,
            }

        ind_slot = (num_backdoor_models + slot_index) if role == "decoy" else slot_index
        outgoing_sd, ind_records = implant_3dfed_indicators(
            outgoing_sd, init_sd, trainable_keys, indicator_indices,
            ind_slot, kappa, role, server_round,
        )
        for rec in ind_records:
            rec.alpha_used = alpha

    out_flat = _state_dict_to_flat({k: v.cpu() for k, v in outgoing_sd.items()}, trainable_keys)
    g_flat = _state_dict_to_flat({k: v.cpu() for k, v in init_sd.items()}, trainable_keys)
    all_metrics["final_update_norm"] = float((out_flat - g_flat).norm().item())

    print(
        f"[3DFed] Round {server_round} | role={role} slot={slot_index} | "
        f"norm={all_metrics['final_update_norm']:.4f}"
    )

    return outgoing_sd, all_metrics, ind_records
