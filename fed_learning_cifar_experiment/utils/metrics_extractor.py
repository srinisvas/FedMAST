import os
import math
import numpy as np
import torch
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Any, Iterable

EPS = 1e-12


STAGE_PREDICATES: List[Tuple[str, Any]] = [
    ("stem",   lambda k: k.startswith("conv1.") or k.startswith("bn1.")),
    ("layer1", lambda k: k.startswith("layer1.")),
    ("layer2", lambda k: k.startswith("layer2.")),
    ("layer3", lambda k: k.startswith("layer3.")),
    ("layer4", lambda k: k.startswith("layer4.")),
    ("head",   lambda k: k.startswith("fc.")),
]
STAGE_NAMES: List[str] = [name for name, _ in STAGE_PREDICATES]


def _is_trainable_key(k: str) -> bool:
    return not (
        k.endswith("running_mean")
        or k.endswith("running_var")
        or k.endswith("num_batches_tracked")
    )


def _stage_of(k: str) -> Optional[str]:
    for name, pred in STAGE_PREDICATES:
        if pred(k):
            return name
    return None


class ParamRegistry:

    def __init__(self, model: torch.nn.Module):
        self.entries: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self.state_dict_keys: List[str] = list(model.state_dict().keys())
        offset = 0
        for k, v in model.named_parameters():
            stage = _stage_of(k)
            if stage is None:
                raise ValueError(
                    f"Unmatched param '{k}'. Update STAGE_PREDICATES."
                )
            self.entries[k] = {
                "stage": stage,
                "shape": tuple(v.shape),
                "offset": offset,
                "numel": v.numel(),
                "dim": v.dim(),
            }
            offset += v.numel()
        self.total_trainable_params: int = offset
        self.stage_to_keys: "OrderedDict[str, List[str]]" = OrderedDict(
            (s, []) for s in STAGE_NAMES
        )
        for k, e in self.entries.items():
            self.stage_to_keys[e["stage"]].append(k)


def nds_to_trainable_state_dict(
    nds: List[np.ndarray],
    state_dict_keys: List[str],
) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, arr in zip(state_dict_keys, nds):
        if not _is_trainable_key(k):
            continue
        out[k] = torch.from_numpy(np.asarray(arr)).float()
    return out


def state_dict_delta(
    client_sd: Dict[str, torch.Tensor],
    global_sd: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    return {k: client_sd[k] - global_sd[k] for k in client_sd.keys()}


def flat_param_vec_to_per_key_dict(
    flat: torch.Tensor,
    registry: ParamRegistry,
) -> Dict[str, torch.Tensor]:
    flat = flat.float()
    out: Dict[str, torch.Tensor] = {}
    for k, e in registry.entries.items():
        seg = flat[e["offset"]: e["offset"] + e["numel"]]
        out[k] = seg.reshape(e["shape"])
    return out


def mean_delta_per_key(
    deltas: Iterable[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    deltas = list(deltas)
    if not deltas:
        return {}
    keys = deltas[0].keys()
    return {k: torch.stack([d[k] for d in deltas], dim=0).mean(dim=0) for k in keys}


def build_leave_one_out_means(
    client_deltas: List[Dict[str, torch.Tensor]],
) -> List[Dict[str, torch.Tensor]]:
    n = len(client_deltas)
    if n < 2:
        return [{k: v.clone() for k, v in d.items()} for d in client_deltas]
    keys = client_deltas[0].keys()
    totals = {k: torch.stack([d[k] for d in client_deltas], dim=0).sum(dim=0)
              for k in keys}
    out = []
    for i in range(n):
        out.append({k: (totals[k] - client_deltas[i][k]) / (n - 1) for k in keys})
    return out


def build_stage_projection_bases(
    ref_deltas_per_key: List[Dict[str, torch.Tensor]],
    registry: "ParamRegistry",
    k: int = 3,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    bases: Dict[str, torch.Tensor] = {}
    ref_proj_means: Dict[str, torch.Tensor] = {}
    if not ref_deltas_per_key:
        return bases, ref_proj_means

    for stage in STAGE_NAMES:
        stage_keys = registry.stage_to_keys[stage]
        if not stage_keys:
            continue
        basis = _fit_stage_projection_basis(ref_deltas_per_key, stage_keys, k=k)
        if basis is None:
            continue
        bases[stage] = basis
        proj_refs = []
        for ref in ref_deltas_per_key:
            p = _project_stage_delta(ref, stage_keys, basis)
            if p is not None:
                proj_refs.append(p)
        if proj_refs:
            ref_proj_means[stage] = torch.stack(proj_refs, dim=0).mean(dim=0)
    return bases, ref_proj_means


def ref_deltas_flat_to_per_key_list(
    ref_deltas_flat: np.ndarray,
    registry: "ParamRegistry",
) -> List[Dict[str, torch.Tensor]]:
    out = []
    for i in range(ref_deltas_flat.shape[0]):
        flat = torch.from_numpy(ref_deltas_flat[i]).float()
        out.append(flat_param_vec_to_per_key_dict(flat, registry))
    return out


def _kurtosis(x: torch.Tensor) -> float:
    if x.numel() < 4:
        return float("nan")
    x = x.float()
    mu = x.mean()
    var = x.var(unbiased=False)
    if float(var) < EPS:
        return float("nan")
    z = (x - mu) / torch.sqrt(var + EPS)
    return float((z ** 4).mean().item() - 3.0)


def _skewness(x: torch.Tensor) -> float:
    if x.numel() < 3:
        return float("nan")
    x = x.float()
    mu = x.mean()
    var = x.var(unbiased=False)
    if float(var) < EPS:
        return float("nan")
    z = (x - mu) / torch.sqrt(var + EPS)
    return float((z ** 3).mean().item())


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() == 0 or b.numel() == 0:
        return float("nan")
    a = a.float()
    b = b.float()
    den = float(torch.norm(a) * torch.norm(b)) + EPS
    if den < EPS:
        return float("nan")
    return float(torch.dot(a, b).item() / den)


def _per_tensor_spectral(weight: torch.Tensor) -> Optional[Tuple[float, float, float]]:
    if weight.dim() < 2 or weight.numel() == 0:
        return None
    m = weight.reshape(weight.shape[0], -1).float()
    if m.shape[0] == 0 or m.shape[1] == 0:
        return None
    try:
        s = torch.linalg.svdvals(m)
    except Exception:
        return None
    s_sum = float(s.sum().item()) + EPS
    top_sv_ratio = float(s[0].item() / s_sum)
    p = (s / (s.sum() + EPS)).clamp(min=EPS)
    spec_entropy = float(-(p * torch.log(p)).sum().item())
    fro = float(torch.norm(m).item())
    return top_sv_ratio, spec_entropy, fro


def _stage_spectral_aggregate(
    delta_per_key: Dict[str, torch.Tensor],
    keys_in_stage: List[str],
) -> Tuple[float, float]:
    weights, top_ratios, spec_entropies = [], [], []
    for k in keys_in_stage:
        out = _per_tensor_spectral(delta_per_key[k])
        if out is None:
            continue
        top_sv_ratio, spec_entropy, fro = out
        weights.append(fro)
        top_ratios.append(top_sv_ratio)
        spec_entropies.append(spec_entropy)
    if not weights:
        return float("nan"), float("nan")
    w = np.asarray(weights, dtype=np.float64)
    w_norm = w / (w.sum() + EPS)
    return (
        float(np.sum(w_norm * np.asarray(top_ratios))),
        float(np.sum(w_norm * np.asarray(spec_entropies))),
    )


def _per_class_delta_entropy(fc_weight_delta: torch.Tensor) -> float:
    if fc_weight_delta.dim() != 2 or fc_weight_delta.shape[0] < 2:
        return float("nan")
    per_class_mag = torch.norm(fc_weight_delta.float(), dim=1)
    s = float(per_class_mag.sum().item()) + EPS
    p = (per_class_mag / s).clamp(min=EPS)
    return float(-(p * torch.log(p)).sum().item())


def _leave_one_out_mean(
    all_client_stacked: torch.Tensor,
    client_idx: int,
) -> torch.Tensor:
    n = all_client_stacked.shape[0]
    if n < 2:
        return all_client_stacked[0].clone()
    total = all_client_stacked.sum(dim=0)
    return (total - all_client_stacked[client_idx]) / (n - 1)


def _head_sign_agreement_with_ref(
    fc_weight_delta: torch.Tensor,
    ref_fc_weight_delta: torch.Tensor,
) -> float:
    if fc_weight_delta.numel() == 0 or ref_fc_weight_delta.numel() == 0:
        return float("nan")
    if fc_weight_delta.shape != ref_fc_weight_delta.shape:
        return float("nan")
    a = fc_weight_delta.flatten()
    b = ref_fc_weight_delta.flatten()
    mask = (a.abs() > EPS) & (b.abs() > EPS)
    if mask.sum().item() == 0:
        return float("nan")
    agree = ((a[mask] > 0) == (b[mask] > 0)).float().mean()
    return float(agree.item())


def _fit_stage_projection_basis(
    ref_deltas_per_key: List[Dict[str, torch.Tensor]],
    stage_keys: List[str],
    k: int = 3,
) -> Optional[torch.Tensor]:
    if not ref_deltas_per_key or not stage_keys:
        return None
    stage_vecs = []
    for ref in ref_deltas_per_key:
        try:
            v = torch.cat([ref[k].flatten() for k in stage_keys]).float()
        except KeyError:
            return None
        stage_vecs.append(v)
    M = torch.stack(stage_vecs, dim=0)
    if M.shape[1] == 0:
        return None
    try:
        M_centered = M - M.mean(dim=0, keepdim=True)
        _, _, Vt = torch.linalg.svd(M_centered, full_matrices=False)
    except Exception:
        return None
    k_eff = min(k, Vt.shape[0])
    if k_eff == 0:
        return None
    return Vt[:k_eff].detach()


def _project_stage_delta(
    delta_per_key: Dict[str, torch.Tensor],
    stage_keys: List[str],
    basis: torch.Tensor,
) -> Optional[torch.Tensor]:
    if basis is None or not stage_keys:
        return None
    try:
        v = torch.cat([delta_per_key[k].flatten() for k in stage_keys]).float()
    except KeyError:
        return None
    if v.numel() != basis.shape[1]:
        return None
    return basis @ v


def extract_per_update_features(
    *,
    client_delta_per_key: Dict[str, torch.Tensor],
    round_mean_delta_per_key: Dict[str, torch.Tensor],
    registry: ParamRegistry,
    reference_delta_per_key: Optional[Dict[str, torch.Tensor]] = None,
    leave_one_out_mean_per_key: Optional[Dict[str, torch.Tensor]] = None,
    stage_projection_bases: Optional[Dict[str, torch.Tensor]] = None,
    stage_projection_ref_means: Optional[Dict[str, torch.Tensor]] = None,
) -> "OrderedDict[str, float]":
    out: "OrderedDict[str, float]" = OrderedDict()

    keys_ordered = list(registry.entries.keys())

    full_client = torch.cat([client_delta_per_key[k].flatten() for k in keys_ordered])
    full_round_mean = torch.cat([round_mean_delta_per_key[k].flatten() for k in keys_ordered])

    out["total_update_l2_norm"] = float(torch.norm(full_client).item())
    out["cosine_to_round_mean_update"] = _cosine(full_client, full_round_mean)
    out["l2_distance_to_round_mean"] = float(torch.norm(full_client - full_round_mean).item())

    if leave_one_out_mean_per_key is not None:
        full_loo = torch.cat([leave_one_out_mean_per_key[k].flatten() for k in keys_ordered])
        out["cosine_to_mean_leave_one_out"] = _cosine(full_client, full_loo)
    else:
        out["cosine_to_mean_leave_one_out"] = float("nan")

    if reference_delta_per_key is not None:
        full_ref = torch.cat([reference_delta_per_key[k].flatten() for k in keys_ordered])
        out["cosine_to_reference_clean_delta"] = _cosine(full_client, full_ref)
        out["l2_distance_to_reference"] = float(torch.norm(full_client - full_ref).item())
    else:
        out["cosine_to_reference_clean_delta"] = float("nan")
        out["l2_distance_to_reference"] = float("nan")

    stage_l2_norms: Dict[str, float] = {}
    stage_kurtoses: Dict[str, float] = {}
    for stage in STAGE_NAMES:
        keys = registry.stage_to_keys[stage]
        if keys:
            x = torch.cat([client_delta_per_key[k].flatten() for k in keys])
        else:
            x = torch.zeros(0)

        l2 = float(torch.norm(x).item())
        stage_l2_norms[stage] = l2
        linf = float(x.abs().max().item()) if x.numel() else 0.0
        out[f"stage_{stage}_l2_norm"] = l2
        out[f"stage_{stage}_linf_to_l2_ratio"] = (
            (linf / (l2 + EPS)) if l2 > EPS else float("nan")
        )

        kurt = _kurtosis(x)
        stage_kurtoses[stage] = kurt if not math.isnan(kurt) else 0.0
        out[f"stage_{stage}_kurtosis"] = kurt
        out[f"stage_{stage}_skewness"] = _skewness(x)

        per_key = {k: client_delta_per_key[k] for k in keys}
        top_sv_ratio, spec_entropy = _stage_spectral_aggregate(per_key, keys)
        out[f"stage_{stage}_top_sv_ratio"] = top_sv_ratio
        out[f"stage_{stage}_spectral_entropy"] = spec_entropy

        if keys:
            stage_round_mean = torch.cat([round_mean_delta_per_key[k].flatten() for k in keys])
        else:
            stage_round_mean = torch.zeros(0)
        out[f"stage_{stage}_cos_to_round_mean"] = _cosine(x, stage_round_mean)

        if leave_one_out_mean_per_key is not None and keys:
            stage_loo = torch.cat([leave_one_out_mean_per_key[k].flatten() for k in keys])
            out[f"stage_{stage}_cos_to_mean_loo"] = _cosine(x, stage_loo)
        else:
            out[f"stage_{stage}_cos_to_mean_loo"] = float("nan")

        if reference_delta_per_key is not None and keys:
            stage_ref = torch.cat([reference_delta_per_key[k].flatten() for k in keys])
            out[f"stage_{stage}_cos_to_reference"] = _cosine(x, stage_ref)
        else:
            out[f"stage_{stage}_cos_to_reference"] = float("nan")

    total_l2 = out["total_update_l2_norm"] + EPS
    for stage in STAGE_NAMES:
        out[f"stage_{stage}_norm_ratio_to_total"] = stage_l2_norms[stage] / total_l2

    sig_client = torch.tensor([stage_l2_norms[s] for s in STAGE_NAMES], dtype=torch.float32)
    sig_round_mean = torch.tensor(
        [
            float(
                torch.norm(
                    torch.cat([round_mean_delta_per_key[k].flatten() for k in registry.stage_to_keys[s]])
                    if registry.stage_to_keys[s] else torch.zeros(0)
                ).item()
            )
            for s in STAGE_NAMES
        ],
        dtype=torch.float32,
    )
    out["stage_norm_signature_cos_to_round_mean"] = _cosine(sig_client, sig_round_mean)

    head_l2 = stage_l2_norms.get("head", 0.0)
    backbone_l2 = sum(stage_l2_norms[s] for s in STAGE_NAMES if s != "head")
    out["classifier_to_backbone_norm_ratio"] = head_l2 / (backbone_l2 + EPS)

    if "fc.weight" in client_delta_per_key:
        out["per_class_delta_entropy"] = _per_class_delta_entropy(
            client_delta_per_key["fc.weight"]
        )
    else:
        out["per_class_delta_entropy"] = float("nan")

    sorted_l2 = sorted(stage_l2_norms.values(), reverse=True)
    out["stage_energy_concentration_top2"] = (
        sum(sorted_l2[:2]) / (sum(sorted_l2) + EPS)
    )


    backbone_stages = ["layer1", "layer2", "layer3", "layer4"]
    backbone_kurts = [stage_kurtoses[s] for s in backbone_stages]
    out["backbone_kurtosis_max"] = float(max(backbone_kurts))
    out["head_backbone_conservation_product"] = (
        out["stage_head_norm_ratio_to_total"] * out["backbone_kurtosis_max"]
    )

    if reference_delta_per_key is not None and "fc.weight" in client_delta_per_key:
        out["head_sign_agreement_with_ref"] = _head_sign_agreement_with_ref(
            client_delta_per_key["fc.weight"],
            reference_delta_per_key["fc.weight"],
        )
    else:
        out["head_sign_agreement_with_ref"] = float("nan")

    stage_pairs = list(zip(STAGE_NAMES[:-1], STAGE_NAMES[1:]))
    for s_from, s_to in stage_pairs:
        feat_name = f"stage_pair_{s_from}_to_{s_to}_projected_cos"
        if stage_projection_bases is None or stage_projection_ref_means is None:
            out[feat_name] = float("nan")
            continue
        basis_from = stage_projection_bases.get(s_from)
        basis_to = stage_projection_bases.get(s_to)
        ref_proj_from = stage_projection_ref_means.get(s_from)
        ref_proj_to = stage_projection_ref_means.get(s_to)
        if basis_from is None or basis_to is None or ref_proj_from is None or ref_proj_to is None:
            out[feat_name] = float("nan")
            continue
        proj_from = _project_stage_delta(
            client_delta_per_key, registry.stage_to_keys[s_from], basis_from
        )
        proj_to = _project_stage_delta(
            client_delta_per_key, registry.stage_to_keys[s_to], basis_to
        )
        if proj_from is None or proj_to is None:
            out[feat_name] = float("nan")
            continue
        dev_from = proj_from - ref_proj_from
        dev_to = proj_to - ref_proj_to
        out[feat_name] = _cosine(dev_from, dev_to)

    return out


def canonical_feature_keys() -> List[str]:
    keys = [
        "total_update_l2_norm",
        "cosine_to_round_mean_update",
        "l2_distance_to_round_mean",
        "cosine_to_mean_leave_one_out",
        "cosine_to_reference_clean_delta",
        "l2_distance_to_reference",
    ]
    per_stage_features = [
        "l2_norm",
        "linf_to_l2_ratio",
        "kurtosis",
        "skewness",
        "top_sv_ratio",
        "spectral_entropy",
        "cos_to_round_mean",
        "cos_to_mean_loo",
        "cos_to_reference",
    ]
    for stage in STAGE_NAMES:
        for feat in per_stage_features:
            keys.append(f"stage_{stage}_{feat}")
    for stage in STAGE_NAMES:
        keys.append(f"stage_{stage}_norm_ratio_to_total")
    keys.extend([
        "stage_norm_signature_cos_to_round_mean",
        "classifier_to_backbone_norm_ratio",
        "per_class_delta_entropy",
        "stage_energy_concentration_top2",
    ])
    keys.extend([
        "backbone_kurtosis_max",
        "head_backbone_conservation_product",
        "head_sign_agreement_with_ref",
    ])
    stage_pairs = list(zip(STAGE_NAMES[:-1], STAGE_NAMES[1:]))
    for s_from, s_to in stage_pairs:
        keys.append(f"stage_pair_{s_from}_to_{s_to}_projected_cos")
    return keys
