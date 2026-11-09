"""Spatial-frequency diagnostics for convolutional *parameter updates*.

The last two dimensions of a Conv2d weight tensor are spatial kernel axes.
No frequency claim is made about the output-channel or input-channel axes.
A 1x1 convolution, linear layer, and BatchNorm parameter are excluded.
"""
from __future__ import annotations

from typing import Dict, Iterable, Mapping

import numpy as np
from scipy.fft import dctn, idctn


BANDS = ("low", "mid", "high")


def is_spatial_kernel(array: np.ndarray) -> bool:
    a = np.asarray(array)
    return a.ndim == 4 and a.shape[-2] >= 3 and a.shape[-1] >= 3


def frequency_masks(height: int, width: int) -> Dict[str, np.ndarray]:
    """Disjoint, exhaustive bands in the 2D DCT-II coefficient plane.

    The normalized radial bins give 3 coefficients per band for 3x3 kernels.
    """
    if height < 3 or width < 3:
        raise ValueError("Spatial frequency analysis requires kernels >= 3x3")
    u, v = np.meshgrid(
        np.arange(height, dtype=float) / (height - 1),
        np.arange(width, dtype=float) / (width - 1),
        indexing="ij",
    )
    radius = np.sqrt(u * u + v * v)
    masks = {
        "low": radius <= 0.5,
        "mid": (radius > 0.5) & (radius <= 1.0),
        "high": radius > 1.0,
    }
    assert np.all(sum(mask.astype(int) for mask in masks.values()) == 1)
    return masks


def channel_band_projection(
    update: np.ndarray, channel: int, band: str
) -> np.ndarray:
    """Return only the (output channel, spatial-frequency band) contribution.

    Projection is orthogonal in the kernel spatial axes. All other channels
    remain zero. Input channels are retained, not treated as a spatial axis.
    """
    arr = np.asarray(update)
    if not is_spatial_kernel(arr):
        raise ValueError("Expected a Conv2d update with kernel >= 3x3")
    if not 0 <= channel < arr.shape[0]:
        raise IndexError("Output channel out of range")
    masks = frequency_masks(*arr.shape[-2:])
    if band not in masks:
        raise ValueError(f"Unknown band {band!r}")
    coeff = dctn(arr[channel].astype(np.float64), axes=(-2, -1), norm="ortho")
    coeff *= masks[band]
    projected = idctn(coeff, axes=(-2, -1), norm="ortho")
    out = np.zeros_like(arr, dtype=np.float64)
    out[channel] = projected
    return out.astype(arr.dtype, copy=False)


def channel_band_energy(update: np.ndarray, channel: int) -> Dict[str, float]:
    arr = np.asarray(update)
    if not is_spatial_kernel(arr):
        raise ValueError("Expected a Conv2d update with kernel >= 3x3")
    coeff = dctn(arr[channel].astype(np.float64), axes=(-2, -1), norm="ortho")
    masks = frequency_masks(*arr.shape[-2:])
    return {
        name: float(np.sum(np.square(coeff[..., mask])))
        for name, mask in masks.items()
    }


def summarize_spatial_bands(
    updates: Mapping[str, np.ndarray],
) -> list[dict]:
    """Compact per-layer energy summaries. Energy is squared L2 norm."""
    rows = []
    for key, raw in updates.items():
        arr = np.asarray(raw)
        if not key.endswith(".weight") or not is_spatial_kernel(arr):
            continue
        coeff = dctn(arr.astype(np.float64), axes=(-2, -1), norm="ortho")
        masks = frequency_masks(*arr.shape[-2:])
        total = float(np.sum(coeff * coeff))
        for name, mask in masks.items():
            energy = float(np.sum(np.square(coeff[..., mask])))
            rows.append({
                "layer": key,
                "band": name,
                "energy": energy,
                "energy_fraction": energy / total if total > 0 else 0.0,
                "coefficients": int(mask.sum()) * int(arr.shape[0] * arr.shape[1]),
                "total_energy": total,
                "output_channels": int(arr.shape[0]),
            })
    return rows


def rank_channels_by_energy(
    update: np.ndarray, limit: int | None = None
) -> list[int]:
    """Energy ranking is a *sampling heuristic*, not a criticality claim."""
    arr = np.asarray(update)
    if not is_spatial_kernel(arr):
        raise ValueError("Expected a Conv2d update with kernel >= 3x3")
    norms = np.sum(np.square(arr.astype(np.float64)), axis=(1, 2, 3))
    ranked = np.argsort(-norms, kind="stable").tolist()
    return ranked if limit is None else ranked[:limit]
