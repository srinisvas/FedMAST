import copy
import os
import random
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_from_disk
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
from torchvision.transforms import v2

from fed_learning_cifar_experiment.models.resnet_cnn_model import tiny_resnet18
from fed_learning_cifar_experiment.utils.backdoor_attack import collate_with_backdoor
from fed_learning_cifar_experiment.utils.drichlet_partition import dirichlet_indices

fds = None
dirichlet_cache = None

base_dir = os.path.dirname(__file__)
local_hf_path = os.path.join(base_dir, "data", "cifar10_hf")


def get_resnet_cnn_model(num_classes: int = 10) -> nn.Module:
    return tiny_resnet18(num_classes=num_classes, base_width=8)


def load_data(partition_id: int, num_partitions: int, alpha_val: float, backdoor_enabled: bool = False,
              target_label: int = 2, poison_fraction: float = 0.1):

    global fds
    if fds is None:
        if not os.path.isdir(local_hf_path):
            raise RuntimeError(
                f"CIFAR-10 not found at {local_hf_path}. "
                "Run: python -m fed_learning_cifar_experiment.prepare_data"
            )

        hf_ds = load_from_disk(local_hf_path)
        hf_train = hf_ds["train"]

        global dirichlet_cache
        if dirichlet_cache is None:
            labels = hf_train["label"]
            dirichlet_cache = dirichlet_indices(
                labels=labels,
                num_partitions=num_partitions,
                alpha=alpha_val,
                seed=42,
            )

        fds = []
        for indices in dirichlet_cache:
            fds.append(hf_train.select(indices))

    partition = fds[partition_id]
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)

    pytorch_transforms = v2.Compose([
        v2.ToImage(),
        v2.RandomCrop(32, padding=4),
        v2.RandomHorizontalFlip(),
        v2.ColorJitter(0.1, 0.1, 0.1, 0.05),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize((0.4914, 0.4822, 0.4465),
                     (0.2023, 0.1994, 0.2010))
    ])

    pytorch_test_transforms = Compose([
        ToTensor(),
        Normalize((0.4914, 0.4822, 0.4465),
                  (0.2023, 0.1994, 0.2010)),
    ])

    def apply_train_transforms(batch):
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    def apply_test_transforms(batch):
        batch["img"] = [pytorch_test_transforms(img) for img in batch["img"]]
        return batch

    partition_train = partition_train_test["train"].with_transform(apply_train_transforms)
    partition_backdoor_train = partition_train_test["train"].with_transform(apply_test_transforms)
    partition_test = partition_train_test["test"].with_transform(apply_test_transforms)

    cuda_avail = torch.cuda.is_available()
    num_workers = 0
    pin_memory = False

    if backdoor_enabled:
        training_data = DataLoader(
            partition_train,
            batch_size=64,
            shuffle=True,
            collate_fn=lambda batch: collate_with_backdoor(batch, num_backdoor_per_batch=20, target_label=target_label),
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    else:
        training_data = DataLoader(
            partition_train,
            batch_size=64,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    test_data = DataLoader(partition_test, batch_size=64, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    return training_data, test_data

def train(net, training_data, epochs, device, lr=0.05):
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.05).to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(training_data) * epochs
    )

    net.train()
    running_loss = 0.0
    for epoch in range(epochs):
        for batch in training_data:
            if isinstance(batch, dict):
                images, labels = batch["img"], batch["label"]
            else:
                images, labels = batch
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            scheduler.step()

    avg_training_loss = running_loss / len(training_data)
    final_vec = parameters_to_vector(net.parameters()).detach().cpu().clone()
    return avg_training_loss, final_vec


def train_neurotoxin(
    net,
    training_data,
    device,
    init_vec: torch.Tensor,
    benign_grad_approx: torch.Tensor = None,
    epochs: int = 40,
    lr: float = 0.01,
    mask_ratio: float = 0.01,
    scale_factor: float = 1.0,
):
    net.to(device)
    net.train()

    g = init_vec.detach().to(device)
    vector_to_parameters(g, net.parameters())

    mask = None
    if benign_grad_approx is not None:
        abs_benign = benign_grad_approx.abs().to(device)
        num_params = abs_benign.numel()
        k = max(1, int(mask_ratio * num_params))
        top_vals, top_indices = torch.topk(abs_benign, k)
        mask = torch.zeros(num_params, dtype=torch.bool, device=device)
        mask[top_indices] = True
        del abs_benign

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, len(training_data) * epochs)
    )

    running_loss = 0.0
    num_steps = 0

    for _ in range(epochs):
        for batch in training_data:
            if isinstance(batch, dict):
                images, labels = batch["img"], batch["label"]
            else:
                images, labels = batch
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()

            if mask is not None:
                offset = 0
                for param in net.parameters():
                    if param.grad is not None:
                        numel = param.numel()
                        param_mask = mask[offset: offset + numel].view(param.shape)
                        param.grad.data[param_mask] = 0.0
                        offset += numel

            optimizer.step()
            scheduler.step()
            running_loss += loss.item()
            num_steps += 1

    avg_loss = running_loss / max(num_steps, 1)

    final_vec = parameters_to_vector(net.parameters()).detach().cpu()
    if scale_factor != 1.0:
        delta = final_vec - init_vec.cpu()
        final_vec = init_vec.cpu() + scale_factor * delta

    return avg_loss, final_vec


def train_lga(
    net,
    training_data,
    device,
    init_sd: dict,
    prev_global_update_sd: dict = None,
    epochs: int = 6,
    lr: float = 0.1,
    tau: float = 1.0,
):
    net.to(device)
    net.train()

    net.load_state_dict({k: v.clone().to(device) for k, v in init_sd.items()})

    trainable_keys = {name for name, _ in net.named_parameters()}

    ref_norms = {}
    if prev_global_update_sd is not None:
        for key, delta in prev_global_update_sd.items():
            if key in trainable_keys:
                ref_norms[key] = delta.to(device).float().norm().item()

    global_sd = {k: v.clone().to(device) for k, v in init_sd.items()}

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)

    running_loss = 0.0
    num_steps = 0

    for epoch in range(epochs):
        for batch in training_data:
            if isinstance(batch, dict):
                images, labels = batch["img"], batch["label"]
            else:
                images, labels = batch
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            num_steps += 1

        if ref_norms:
            with torch.no_grad():
                current_sd = net.state_dict()
                for key in trainable_keys:
                    if key not in global_sd or key not in ref_norms:
                        continue

                    delta = current_sd[key].float() - global_sd[key].float()
                    delta_norm = delta.norm().item()

                    if delta_norm < 1e-10:
                        continue

                    s = min(1.0, tau * ref_norms[key] / delta_norm)

                    aligned = global_sd[key].float() + s * delta
                    current_sd[key] = aligned.to(current_sd[key].dtype)

                net.load_state_dict(current_sd)

            for group in optimizer.param_groups:
                for p in group["params"]:
                    state = optimizer.state.get(p, {})
                    if "momentum_buffer" in state:
                        state["momentum_buffer"].zero_()

    with torch.no_grad():
        current_sd = net.state_dict()
        for key in current_sd:
            if key not in trainable_keys:
                current_sd[key] = global_sd[key].to(
                    device=current_sd[key].device,
                    dtype=current_sd[key].dtype,
                )
        net.load_state_dict(current_sd)

    avg_loss = running_loss / max(num_steps, 1)
    final_vec = parameters_to_vector(net.parameters()).detach().cpu()

    return avg_loss, final_vec


DBA_PATTERNS = {
    0: [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5)],
    1: [(0, 9), (0, 10), (0, 11), (0, 12), (0, 13), (0, 14)],
    2: [(4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5)],
    3: [(4, 9), (4, 10), (4, 11), (4, 12), (4, 13), (4, 14)],
}

_CIFAR_MEANS = (0.4914, 0.4822, 0.4465)
_CIFAR_STDS = (0.2023, 0.1994, 0.2010)


def get_dba_pattern(attacker_index: int) -> list:
    if attacker_index not in DBA_PATTERNS:
        raise ValueError(
            f"DBA attacker_index must be 0..3, got {attacker_index}"
        )
    return DBA_PATTERNS[attacker_index]


def apply_dba_trigger(
    images: torch.Tensor,
    attacker_index=None,
    *,
    value_raw: float = 1.0,
) -> torch.Tensor:
    out = images.clone()

    norm_vals = [
        (value_raw - _CIFAR_MEANS[c]) / _CIFAR_STDS[c] for c in range(3)
    ]

    if attacker_index is not None:
        patterns = [get_dba_pattern(attacker_index)]
    else:
        patterns = [DBA_PATTERNS[i] for i in range(4)]

    for pat in patterns:
        for row, col in pat:
            for c in range(3):
                out[:, c, row, col] = norm_vals[c]

    return out


def train_dba(
    net,
    training_data,
    device,
    init_vec: torch.Tensor,
    attacker_index: int,
    num_attackers: int = 4,
    target_label: int = 2,
    epochs: int = 6,
    lr: float = 0.05,
    poison_ratio: float = 5 / 64,
    scale_factor: float = 100.0,
):
    if attacker_index not in DBA_PATTERNS:
        raise ValueError(f"DBA attacker_index must be 0..3, got {attacker_index}")

    net.to(device)
    net.train()

    g = init_vec.detach().to(device)
    vector_to_parameters(g, net.parameters())

    pattern = get_dba_pattern(attacker_index)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)

    running_loss = 0.0
    num_steps = 0
    total_poisoned = 0

    for epoch in range(epochs):
        for batch in training_data:
            if isinstance(batch, dict):
                images = batch["img"].to(device)
                labels = batch["label"].to(device)
            else:
                images, labels = batch[0].to(device), batch[1].to(device)

            bs = images.size(0)
            n_poison = max(1, int(poison_ratio * bs))
            perm = torch.randperm(bs, device=device)
            poison_idx = perm[:n_poison]

            triggered = apply_dba_trigger(
                images[poison_idx], attacker_index=attacker_index
            )
            images = images.clone()
            images[poison_idx] = triggered
            labels = labels.clone()
            labels[poison_idx] = target_label

            optimizer.zero_grad(set_to_none=True)
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_steps += 1
            total_poisoned += n_poison

    avg_loss = running_loss / max(num_steps, 1)
    local_vec = parameters_to_vector(net.parameters()).detach().cpu()

    unscaled_delta = local_vec - init_vec.cpu()
    scaled_vec = init_vec.cpu() + scale_factor * unscaled_delta

    metrics = {
        "dba_attacker_index": attacker_index,
        "dba_local_pattern": str(pattern),
        "dba_poison_ratio": float(poison_ratio),
        "dba_scale_factor": float(scale_factor),
        "dba_unscaled_delta_norm": float(unscaled_delta.norm().item()),
        "dba_scaled_delta_norm": float((scale_factor * unscaled_delta).norm().item()),
        "dba_num_poisoned_examples": total_poisoned,
    }

    print(
        f"[DBA] slot={attacker_index} pattern={pattern} | "
        f"poisoned={n_poison}/batch scale={scale_factor} | "
        f"unscaled_norm={metrics['dba_unscaled_delta_norm']:.4f} "
        f"scaled_norm={metrics['dba_scaled_delta_norm']:.4f}"
    )

    return avg_loss, scaled_vec, metrics


def evaluate_dba_asr(
    model, test_loader, target_label: int, device,
) -> dict:
    model.to(device)
    model.eval()

    results = {}

    with torch.no_grad():
        for trigger_id in [0, 1, 2, 3, None]:
            correct = 0
            total = 0

            for batch in test_loader:
                if isinstance(batch, dict):
                    images = batch["img"].to(device)
                    labels = batch["label"].to(device)
                else:
                    images, labels = batch[0].to(device), batch[1].to(device)

                mask = labels != target_label
                if mask.sum().item() == 0:
                    continue

                imgs = apply_dba_trigger(
                    images[mask], attacker_index=trigger_id
                )
                preds = model(imgs).argmax(dim=1)
                correct += (preds == target_label).sum().item()
                total += imgs.size(0)

            asr = correct / max(total, 1)
            if trigger_id is not None:
                results[f"dba_asr_local_{trigger_id}"] = asr
            else:
                results["dba_asr_global"] = asr
                results["dba_eval_count"] = total

    local_asrs = [results.get(f"dba_asr_local_{i}", 0) for i in range(4)]
    print(
        f"[DBA][Eval] local={[f'{a:.4f}' for a in local_asrs]} "
        f"global={results.get('dba_asr_global', 0):.4f} "
        f"n={results.get('dba_eval_count', 0)}"
    )

    return results


def _evaluate_bsr(model, clean_data_loader, target_label, device):
    model.eval()

    trigger_rgb = (1.0, 0.5, 0.0)
    trigger_size = 8
    means = [0.4914, 0.4822, 0.4465]
    stds = [0.2023, 0.1994, 0.2010]

    trigger_patch = torch.zeros(3, trigger_size, trigger_size, device=device)
    for c in range(3):
        trigger_patch[c] = (trigger_rgb[c] - means[c]) / stds[c]

    correct = 0
    total = 0

    with torch.no_grad():
        for batch in clean_data_loader:
            if isinstance(batch, dict):
                images = batch["img"].to(device)
                labels = batch["label"].to(device)
            else:
                images, labels = batch[0].to(device), batch[1].to(device)

            mask = labels != target_label
            if mask.sum().item() == 0:
                continue

            imgs = images[mask].clone()
            imgs[:, :, -trigger_size:, -trigger_size:] = trigger_patch.unsqueeze(0)

            preds = model(imgs).argmax(dim=1)
            correct += (preds == target_label).sum().item()
            total += imgs.size(0)

    return correct / max(total, 1)


def _simulate_multikrum(updates: list, candidate_idx: int, f: int = 2, k: int = 4):
    import numpy as np
    n = len(updates)
    X = np.stack(updates)
    dists = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sum((X[i] - X[j]) ** 2)
            dists[i, j] = d
            dists[j, i] = d

    scores = np.zeros(n)
    num_closest = max(1, n - f - 2)
    for i in range(n):
        sorted_dists = np.sort(dists[i])
        scores[i] = np.sum(sorted_dists[1:num_closest + 1])

    selected = np.argsort(scores)[:k]
    return candidate_idx in selected


def train_bc_layers_lp_full(
    net,
    clean_data,
    poisoned_data,
    device,
    init_sd: dict,
    target_label: int = 2,
    tau: float = 0.95,
    lambda_val: float = 1.0,
    benign_epochs: int = 2,
    malicious_epochs: int = 2,
    local_lr: float = 0.1,
    proxy_count: int = 5,
    defense_sim: str = "none",
):
    import copy

    net.to(device)
    criterion = nn.CrossEntropyLoss()
    global_sd = {k: v.clone().to(device) for k, v in init_sd.items()}

    proxy_sds = []
    for p_idx in range(proxy_count):
        proxy = copy.deepcopy(net)
        proxy.to(device)
        proxy.train()
        opt_p = torch.optim.SGD(proxy.parameters(), lr=local_lr, momentum=0.9)
        for epoch in range(benign_epochs):
            for batch in clean_data:
                if isinstance(batch, dict):
                    images, labels = batch["img"].to(device), batch["label"].to(device)
                else:
                    images, labels = batch[0].to(device), batch[1].to(device)
                opt_p.zero_grad(set_to_none=True)
                loss = criterion(proxy(images), labels)
                loss.backward()
                opt_p.step()
        proxy_sds.append({k: v.clone() for k, v in proxy.state_dict().items()})
        del proxy

    avg_sd = {}
    for key in proxy_sds[0]:
        stacked = torch.stack([sd[key].float() for sd in proxy_sds])
        avg_sd[key] = stacked.mean(dim=0).to(proxy_sds[0][key].dtype)

    mal_model = copy.deepcopy(net)
    mal_model.load_state_dict(avg_sd)
    mal_model.to(device)
    mal_model.train()
    opt_m = torch.optim.SGD(mal_model.parameters(), lr=local_lr, momentum=0.9)

    running_loss = 0.0
    num_steps = 0
    for epoch in range(malicious_epochs):
        for batch in poisoned_data:
            if isinstance(batch, dict):
                images, labels = batch["img"].to(device), batch["label"].to(device)
            else:
                images, labels = batch[0].to(device), batch[1].to(device)
            opt_m.zero_grad(set_to_none=True)
            loss = criterion(mal_model(images), labels)
            loss.backward()
            opt_m.step()
            running_loss += loss.item()
            num_steps += 1

    mal_sd = {k: v.clone() for k, v in mal_model.state_dict().items()}

    weight_keys = [
        name for name, p in mal_model.named_parameters()
        if name.endswith(".weight") and p.ndim > 1
    ]

    mal_model.load_state_dict(mal_sd)
    bsr_malicious = _evaluate_bsr(mal_model, clean_data, target_label, device)

    delta_bsr = {}
    for key in weight_keys:
        test_sd = {k: v.clone() for k, v in mal_sd.items()}
        test_sd[key] = avg_sd[key].clone()
        mal_model.load_state_dict(test_sd)
        bsr_swapped = _evaluate_bsr(mal_model, clean_data, target_label, device)
        delta_bsr[key] = bsr_malicious - bsr_swapped

    sorted_layers = sorted(delta_bsr.keys(), key=lambda k: delta_bsr[k], reverse=True)

    threshold = tau * bsr_malicious
    bc_layers_ordered = []

    test_sd = {k: v.clone() for k, v in avg_sd.items()}
    for key in sorted_layers:
        test_sd[key] = mal_sd[key].clone()
        bc_layers_ordered.append(key)
        mal_model.load_state_dict(test_sd)
        bsr_current = _evaluate_bsr(mal_model, clean_data, target_label, device)
        if bsr_current >= threshold:
            break

    print(
        f"[BC-Full] BSR_mal={bsr_malicious:.4f} | "
        f"BC layers: {len(bc_layers_ordered)}/{len(weight_keys)} | "
        f"BSR_bc={bsr_current:.4f} | proxies={proxy_count}"
    )

    import numpy as np

    proxy_deltas = []
    global_vec_np = np.concatenate([
        global_sd[k].cpu().numpy().flatten() for k in sorted(global_sd.keys())
    ])
    for sd in proxy_sds:
        vec = np.concatenate([
            sd[k].cpu().numpy().flatten() for k in sorted(sd.keys())
        ])
        proxy_deltas.append(vec - global_vec_np)

    relu_1_minus_lambda = max(0.0, 1.0 - lambda_val)

    bc_set = list(bc_layers_ordered)

    for attempt in range(len(bc_layers_ordered)):
        bc_keys = set(bc_set)
        crafted_sd = {}
        for key in avg_sd:
            if key in bc_keys:
                crafted_sd[key] = (
                    lambda_val * mal_sd[key].float()
                    + relu_1_minus_lambda * avg_sd[key].float()
                ).to(mal_sd[key].dtype)
            else:
                crafted_sd[key] = avg_sd[key].clone()

        if defense_sim == "none":
            break

        candidate_vec = np.concatenate([
            crafted_sd[k].cpu().numpy().flatten() for k in sorted(crafted_sd.keys())
        ])
        candidate_delta = candidate_vec - global_vec_np

        all_updates = proxy_deltas + [candidate_delta]
        candidate_idx = len(proxy_deltas)

        if defense_sim in ("multikrum", "krum"):
            f_param = 2 if defense_sim == "multikrum" else 1
            k_param = max(1, len(all_updates) - 2 * f_param)
            accepted = _simulate_multikrum(
                all_updates, candidate_idx, f=f_param, k=k_param
            )
        else:
            accepted = True

        if accepted:
            print(
                f"[BC-Full] Defense sim '{defense_sim}' ACCEPTED | "
                f"BC layers used: {len(bc_set)}"
            )
            break
        else:
            removed = bc_set.pop()
            print(
                f"[BC-Full] Defense sim REJECTED | "
                f"removing {removed} | {len(bc_set)} BC layers remain"
            )
            if not bc_set:
                print("[BC-Full] WARNING: all BC layers removed, attack likely ineffective")
                break

    net.load_state_dict(crafted_sd)
    avg_loss = running_loss / max(num_steps, 1)
    final_vec = parameters_to_vector(net.parameters()).detach().cpu()

    return avg_loss, final_vec


def train_bc_layers_lp_full_fedmast(
    net,
    clean_data,
    poisoned_data,
    device,
    init_sd: dict,
    target_label: int = 2,
    tau: float = 0.95,
    lambda_val: float = 0.7,
    benign_epochs: int = 2,
    malicious_epochs: int = 2,
    local_lr: float = 0.1,
    proxy_count: int = 5,
    norm_match_weight: float = 0.8,
    energy_smooth_noise: float = 0.03,
    direction_blend_alpha: float = 0.15,
    min_bsr_ratio: float = 0.5,
):
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    global_sd = {k: v.clone().to(device) for k, v in init_sd.items()}

    trainable_keys = [
        name for name, p in net.named_parameters()
        if p.requires_grad
        and name in global_sd
        and torch.is_floating_point(global_sd[name])
    ]
    trainable_set = set(trainable_keys)

    trainable_shapes = {k: global_sd[k].shape for k in trainable_keys}
    trainable_dtypes = {k: global_sd[k].dtype for k in trainable_keys}

    def sd_to_vec(sd):
        return torch.cat([sd[k].float().cpu().flatten() for k in trainable_keys])

    def vec_to_sd_trainable(vec, base_sd):
        sd = {}
        for k in base_sd:
            if k not in trainable_set:
                sd[k] = base_sd[k].clone()
        offset = 0
        for k in trainable_keys:
            numel = trainable_shapes[k].numel()
            sd[k] = vec[offset:offset + numel].reshape(
                trainable_shapes[k]
            ).to(trainable_dtypes[k])
            offset += numel
        return sd

    global_vec = sd_to_vec(global_sd)

    proxy_sds = []
    for p_idx in range(proxy_count):
        proxy = copy.deepcopy(net)
        proxy.to(device)
        proxy.train()
        opt_p = torch.optim.SGD(proxy.parameters(), lr=local_lr, momentum=0.9)
        for epoch in range(benign_epochs):
            for batch in clean_data:
                if isinstance(batch, dict):
                    images, labels = batch["img"].to(device), batch["label"].to(device)
                else:
                    images, labels = batch[0].to(device), batch[1].to(device)
                opt_p.zero_grad(set_to_none=True)
                loss = criterion(proxy(images), labels)
                loss.backward()
                opt_p.step()
        proxy_sds.append({k: v.clone().cpu() for k, v in proxy.state_dict().items()})
        del proxy

    avg_sd = {}
    for key in proxy_sds[0]:
        stacked = torch.stack([sd[key].float() for sd in proxy_sds])
        avg_sd[key] = stacked.mean(dim=0).to(proxy_sds[0][key].dtype)

    mal_model = copy.deepcopy(net)
    mal_model.load_state_dict({k: v.to(device) for k, v in avg_sd.items()})
    mal_model.to(device)
    mal_model.train()
    opt_m = torch.optim.SGD(mal_model.parameters(), lr=local_lr, momentum=0.9)

    running_loss = 0.0
    num_steps = 0
    for epoch in range(malicious_epochs):
        for batch in poisoned_data:
            if isinstance(batch, dict):
                images, labels = batch["img"].to(device), batch["label"].to(device)
            else:
                images, labels = batch[0].to(device), batch[1].to(device)
            opt_m.zero_grad(set_to_none=True)
            loss = criterion(mal_model(images), labels)
            loss.backward()
            opt_m.step()
            running_loss += loss.item()
            num_steps += 1

    mal_sd = {k: v.clone().cpu() for k, v in mal_model.state_dict().items()}

    weight_keys = [
        name for name, p in mal_model.named_parameters()
        if name.endswith(".weight") and p.ndim > 1
    ]

    mal_model.load_state_dict({k: v.to(device) for k, v in mal_sd.items()})
    bsr_malicious = _evaluate_bsr(mal_model, clean_data, target_label, device)

    delta_bsr = {}
    for key in weight_keys:
        test_sd = {k: v.clone() for k, v in mal_sd.items()}
        test_sd[key] = avg_sd[key].clone()
        mal_model.load_state_dict({k: v.to(device) for k, v in test_sd.items()})
        bsr_swapped = _evaluate_bsr(mal_model, clean_data, target_label, device)
        delta_bsr[key] = bsr_malicious - bsr_swapped

    sorted_layers = sorted(delta_bsr.keys(), key=lambda k: delta_bsr[k], reverse=True)

    threshold = tau * bsr_malicious
    bc_layers_ordered = []

    test_sd = {k: v.clone() for k, v in avg_sd.items()}
    bsr_current = 0.0
    for key in sorted_layers:
        test_sd[key] = mal_sd[key].clone()
        bc_layers_ordered.append(key)
        mal_model.load_state_dict({k: v.to(device) for k, v in test_sd.items()})
        bsr_current = _evaluate_bsr(mal_model, clean_data, target_label, device)
        if bsr_current >= threshold:
            break

    print(
        f"[BC-FedMAST] BSR_mal={bsr_malicious:.4f} | "
        f"BC layers: {len(bc_layers_ordered)}/{len(weight_keys)} | "
        f"BSR_bc={bsr_current:.4f} | proxies={proxy_count}"
    )


    proxy_vecs = [sd_to_vec(sd) for sd in proxy_sds]
    proxy_deltas = [pv - global_vec for pv in proxy_vecs]
    proxy_norms = [torch.norm(d).item() for d in proxy_deltas]
    mean_proxy_delta = torch.stack(proxy_deltas).mean(dim=0)

    median_proxy_norm = float(np.median(proxy_norms))
    target_norm = median_proxy_norm * random.uniform(0.95, 1.05)

    layer_proxy_std = {}
    for key in trainable_keys:
        layer_vals = torch.stack([sd[key].float() for sd in proxy_sds])
        layer_proxy_std[key] = layer_vals.std(dim=0).clamp(min=1e-8)

    min_bsr = min_bsr_ratio * bsr_malicious

    curr_blend_alpha = direction_blend_alpha
    curr_noise_scale = energy_smooth_noise
    curr_norm_match = norm_match_weight
    curr_lambda = lambda_val
    bc_set = list(bc_layers_ordered)

    def expand_bc_keys(weight_layer_keys):
        expanded = set(weight_layer_keys)
        for wk in weight_layer_keys:
            if wk.endswith(".weight"):
                bk = wk[:-len(".weight")] + ".bias"
                if bk in trainable_set:
                    expanded.add(bk)
        return expanded

    MAX_ATTEMPTS = 6
    final_crafted_sd = None
    final_bsr = 0.0

    for attempt in range(MAX_ATTEMPTS):
        bc_keys = expand_bc_keys(bc_set)
        relu_1_minus_lambda = max(0.0, 1.0 - curr_lambda)

        crafted_sd = {}
        for key in avg_sd:
            if key in bc_keys:
                crafted_sd[key] = (
                    curr_lambda * mal_sd[key].float()
                    + relu_1_minus_lambda * avg_sd[key].float()
                ).to(mal_sd[key].dtype)
            else:
                crafted_sd[key] = avg_sd[key].clone()

        if curr_noise_scale > 0:
            for key in trainable_keys:
                if key not in bc_keys:
                    noise = (
                        torch.randn_like(crafted_sd[key].float())
                        * curr_noise_scale
                        * layer_proxy_std[key]
                    )
                    crafted_sd[key] = (
                        crafted_sd[key].float() + noise
                    ).to(crafted_sd[key].dtype)

        crafted_vec = sd_to_vec(crafted_sd)
        crafted_delta = crafted_vec - global_vec

        if curr_blend_alpha > 0 and torch.norm(mean_proxy_delta) > 1e-8:
            crafted_delta = (
                (1.0 - curr_blend_alpha) * crafted_delta
                + curr_blend_alpha * mean_proxy_delta
            )

        crafted_norm = torch.norm(crafted_delta).item()
        if crafted_norm > 1e-8 and curr_norm_match > 0:
            adjusted_norm = (
                (1.0 - curr_norm_match) * crafted_norm
                + curr_norm_match * target_norm
            )
            crafted_delta = crafted_delta * (adjusted_norm / crafted_norm)

        final_vec_t = global_vec + crafted_delta
        crafted_sd = vec_to_sd_trainable(final_vec_t, base_sd=avg_sd)

        mal_model.load_state_dict({k: v.to(device) for k, v in crafted_sd.items()})
        final_bsr = _evaluate_bsr(mal_model, clean_data, target_label, device)

        if final_bsr >= min_bsr:
            final_crafted_sd = crafted_sd
            print(
                f"[BC-FedMAST] Attempt {attempt + 1}: ACCEPTED | "
                f"BSR={final_bsr:.4f} (min={min_bsr:.4f}) | "
                f"λ={curr_lambda:.2f} | "
                f"blend_α={curr_blend_alpha:.3f} | noise={curr_noise_scale:.3f} | "
                f"norm_match={curr_norm_match:.2f} | "
                f"norm_ratio={torch.norm(crafted_delta).item() / max(median_proxy_norm, 1e-8):.3f} | "
                f"BC layers={len(bc_set)}"
            )
            break
        else:
            print(
                f"[BC-FedMAST] Attempt {attempt + 1}: BSR={final_bsr:.4f} < "
                f"min={min_bsr:.4f} | relaxing evasion "
                f"(λ={curr_lambda:.2f} blend={curr_blend_alpha:.3f} "
                f"noise={curr_noise_scale:.3f} norm={curr_norm_match:.2f})"
            )

            curr_blend_alpha *= 0.5
            curr_noise_scale *= 0.5
            curr_norm_match = max(0.2, curr_norm_match * 0.7)
            curr_lambda = min(0.85, curr_lambda + 0.1)

            if (curr_blend_alpha < 0.01
                    and curr_noise_scale < 0.005
                    and curr_lambda >= 0.84):
                remaining = [k for k in sorted_layers if k not in set(bc_set)]
                if remaining:
                    added = remaining[0]
                    bc_set.append(added)
                    print(
                        f"[BC-FedMAST] Soft knobs exhausted, adding BC layer: "
                        f"{added} | {len(bc_set)} total"
                    )
                else:
                    print("[BC-FedMAST] WARNING: all knobs exhausted, accepting low BSR")
                    final_crafted_sd = crafted_sd
                    break

    if final_crafted_sd is None:
        final_crafted_sd = crafted_sd

    final_delta = sd_to_vec(final_crafted_sd) - global_vec
    final_delta_norm = torch.norm(final_delta).item()
    cosine_to_benign = torch.nn.functional.cosine_similarity(
        final_delta.unsqueeze(0),
        mean_proxy_delta.unsqueeze(0),
    ).item() if torch.norm(mean_proxy_delta) > 1e-8 else 0.0

    diagnostics = {
        "bc_defense_sim": "fedmast",
        "bc_bsr_malicious": float(bsr_malicious),
        "bc_bsr_final": float(final_bsr),
        "bc_num_bc_layers": int(len(bc_set)),
        "bc_num_weight_layers": int(len(weight_keys)),
        "bc_delta_norm": float(final_delta_norm),
        "bc_median_proxy_norm": float(median_proxy_norm),
        "bc_norm_ratio": float(final_delta_norm / max(median_proxy_norm, 1e-8)),
        "bc_cos_to_benign": float(cosine_to_benign),
        "bc_final_blend_alpha": float(curr_blend_alpha),
        "bc_final_noise_scale": float(curr_noise_scale),
        "bc_final_norm_match": float(curr_norm_match),
        "bc_final_lambda": float(curr_lambda),
    }

    print(
        f"[BC-FedMAST] Final: BSR={final_bsr:.4f} | "
        f"delta_norm={final_delta_norm:.4f} | "
        f"median_proxy_norm={median_proxy_norm:.4f} | "
        f"norm_ratio={diagnostics['bc_norm_ratio']:.3f} | "
        f"cos_to_benign={cosine_to_benign:.4f} | "
        f"λ={curr_lambda:.2f}"
    )

    net.load_state_dict({k: v.to(device) for k, v in final_crafted_sd.items()})
    avg_loss = running_loss / max(num_steps, 1)
    final_vec = parameters_to_vector(net.parameters()).detach().cpu()

    return avg_loss, final_vec, diagnostics

def train_constrain_and_scale(
    net,
    training_data,
    epochs,
    device,
    init_vec: torch.Tensor,
    prev_global_vec: torch.Tensor = None,
    lr: float = 0.005,

    lambda_norm: float = 0.02,
    lambda_dir: float = 0.50,
    lambda_target_norm: float = 0.10,
    lambda_pair: float = 0.20,

    target_delta_norm: float = None,
    min_dir_norm: float = 1e-12,

    epsilon_ce: float = None,

    label_smoothing: float = 0.0,
):

    net.to(device)
    net.train()

    g = init_vec.detach().to(device)
    vector_to_parameters(g, net.parameters())

    d_unit = None
    g_prev = None

    if prev_global_vec is not None:
        g_prev = prev_global_vec.detach().to(device)
        d = (g - g_prev)
        d_norm = torch.norm(d)
        if d_norm >= min_dir_norm:
            d_unit = d / d_norm

        if target_delta_norm is None:
            est = float(torch.norm(g - g_prev).detach().cpu())
            if est >= 1e-8:
                target_delta_norm = est

    criterion = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing).to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0)

    for epoch in range(epochs):
        running_ce = 0.0
        steps = 0

        for batch in training_data:
            if isinstance(batch, dict):
                images, labels = batch["img"], batch["label"]
            else:
                images, labels = batch

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            logits = net(images)
            ce = criterion(logits, labels)

            w = parameters_to_vector(net.parameters())
            delta = (w - g)

            l_norm = torch.mean(delta * delta)

            camouflage_scale = 1.0
            if g_prev is not None:
                warmup_epochs = max(1, epochs // 3)
                camouflage_scale = min(1.0, epoch / warmup_epochs)

            loss = ce

            loss += camouflage_scale * lambda_norm * torch.mean(delta * delta)

            if d_unit is not None and lambda_dir > 0.0:
                delta_norm = torch.norm(delta) + 1e-12
                delta_norm = torch.sqrt(torch.sum(delta * delta) + 1e-12)
                delta_unit = delta / delta_norm
                cos = torch.dot(delta_unit, d_unit).clamp(-1.0, 1.0)
                loss += camouflage_scale * lambda_dir * (1.0 - cos)

            if target_delta_norm is not None and lambda_target_norm > 0.0:
                delta_norm = torch.norm(delta) + 1e-12
                delta_norm = torch.sqrt(torch.sum(delta * delta) + 1e-12)
                loss += camouflage_scale * lambda_target_norm * (
                        delta_norm - target_delta_norm
                ) ** 2

            if g_prev is not None and lambda_pair > 0.0:
                delta_ref = (g - g_prev)
                loss += camouflage_scale * lambda_pair * torch.mean(
                    (delta - delta_ref) ** 2
                )

            if target_delta_norm is not None:
                min_attack_norm = 0.05 * target_delta_norm
                min_sq = (min_attack_norm ** 2)

                delta_sq = torch.sum(delta * delta)
                loss += F.relu(min_sq - delta_sq) ** 2

            loss.backward()
            optimizer.step()

            running_ce += float(ce.detach().cpu())
            steps += 1

        if epsilon_ce is not None and (running_ce / max(1, steps)) < epsilon_ce:
            break

    return parameters_to_vector(net.parameters()).detach().cpu().clone()


def test(net, test_data, device):
    net.to(device)
    net.eval()
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():
        for batch in test_data:
            if isinstance(batch, dict):
                images, labels = batch["img"], batch["label"]
            else:
                images, labels = batch
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return loss / len(test_data), correct / total

def train_constrain_and_scale_for_fedavg_tm(
    net,
    training_data,
    clean_data,
    device,
    init_vec,
    prev_global_vec=None,
    epochs=3,
    lr=0.01,
    scale_factor=1.0,
    lambda_prox=0.5,
    poison_ratio=0.5,
):
    import torch.nn as nn

    net.to(device)
    net.train()

    g = init_vec.detach().to(device)
    vector_to_parameters(g, net.parameters())

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)

    clean_delta_abs = None
    if prev_global_vec is not None:
        prev = prev_global_vec.detach().to(device)
        clean_delta_abs = (g - prev).abs()

    for epoch in range(epochs):
        poison_iter = iter(training_data)
        for batch in clean_data:
            if isinstance(batch, dict):
                imgs_c, lbls_c = batch["img"].to(device), batch["label"].to(device)
            else:
                imgs_c, lbls_c = batch[0].to(device), batch[1].to(device)

            try:
                p_batch = next(poison_iter)
            except StopIteration:
                poison_iter = iter(training_data)
                p_batch = next(poison_iter)

            if isinstance(p_batch, dict):
                imgs_p, lbls_p = p_batch["img"].to(device), p_batch["label"].to(device)
            else:
                imgs_p, lbls_p = p_batch[0].to(device), p_batch[1].to(device)

            bs = imgs_c.size(0)
            n_poison = max(1, int(poison_ratio * bs))
            n_clean = bs - n_poison

            imgs = torch.cat([imgs_c[:n_clean], imgs_p[:n_poison]], dim=0)
            lbls = torch.cat([lbls_c[:n_clean], lbls_p[:n_poison]], dim=0)

            optimizer.zero_grad(set_to_none=True)

            ce = criterion(net(imgs), lbls)

            w = parameters_to_vector(net.parameters())
            prox = torch.mean((w - g) ** 2)

            loss = ce + lambda_prox * prox
            loss.backward()

            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            optimizer.step()

    with torch.no_grad():
        w_final = parameters_to_vector(net.parameters()).detach()
        delta = w_final - g

        if clean_delta_abs is not None:
            total_trainable = delta.numel()
            if clean_delta_abs.numel() == total_trainable:
                clip_bound = 3.0 * clean_delta_abs + 1e-7
                delta = delta.clamp(-clip_bound, clip_bound)

        final = g + scale_factor * delta
        vector_to_parameters(final, net.parameters())

    return parameters_to_vector(net.parameters()).detach().cpu().clone()


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

def load_test_data_for_eval(batch_size=64):

    pytorch_transforms = Compose(
        [ToTensor(), Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
    )


    if os.path.isdir(local_hf_path):
        hf_ds = load_from_disk(local_hf_path)

        def apply_transforms(batch):
            batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
            return batch

        hf_ds = hf_ds.with_transform(apply_transforms)
        return DataLoader(hf_ds["test"], batch_size=batch_size, shuffle=False)

    raise RuntimeError(
        f"CIFAR-10 not found at {local_hf_path}. "
        "Run: python -m fed_learning_cifar_experiment.prepare_data"
    )


def test_eval(net, test_data, device):
    net.to(device)
    net.eval()
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in test_data:
            if isinstance(batch, dict):
                images, labels = batch["img"], batch["label"]
            else:
                images, labels = batch
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(test_data.dataset)
    loss = loss / len(test_data)
    return loss, accuracy
