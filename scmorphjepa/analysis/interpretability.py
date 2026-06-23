"""Novel interpretability tools for scMorphJEPA.

Three contributions with no prior art for JEPA on microscopy:
    1. Prediction error saliency: per-patch MSE as biological spatial importance
    2. Attention fingerprints: per-head attention aggregated per cluster
    3. Channel attribution: which fluorescence channels drive the representation
"""

from __future__ import annotations

import logging
import math

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ── 1. Prediction Error Saliency ──────────────────────────────────────────

def compute_prediction_error_maps(
    model: nn.Module, dataloader, device: str = "cuda", n_samples: int = 8,
    seed: int | None = 0,
) -> np.ndarray:
    """Per-patch prediction error, averaged over the random masks that hit each patch.

    Each forward masks only ~mask_ratio of patches, and the error map is zero at
    visible positions. Averaging must therefore divide each patch by the number of
    samples in which it was *actually masked* — not by n_samples — otherwise patches
    that were masked fewer times are systematically underestimated.

    Patches never masked across all n_samples are returned as NaN (use nan-aware
    aggregation downstream). Increase n_samples to reduce the NaN fraction:
    a patch is missed with probability (1 - mask_ratio) ** n_samples.

    Patches with consistently high error contain information hard to predict
    from spatial context — potentially biologically distinctive structures.

    Args:
        seed: If set, seeds the RNG so the random masks (and thus the maps) are
            reproducible across runs — important for figures in a paper.

    Returns: (N_total, H_patches, W_patches) float32 array, may contain NaN.
    """
    if seed is not None:
        torch.manual_seed(seed)
    model.eval()
    all_maps = []

    with torch.no_grad():
        for images, *_ in tqdm(dataloader, desc="Error maps"):
            images = images.to(device)
            B = images.size(0)
            N = model.predictor.num_patches
            accumulated = torch.zeros(B, N, device=device)
            counts = torch.zeros(B, N, device=device)  # per-sample mask is different each draw

            for _ in range(n_samples):
                out = model(images, return_error_map=True)
                if "error_map" not in out:
                    continue
                accumulated += out["error_map"].view(B, -1)
                ones = torch.ones_like(out["mask_indices"], dtype=counts.dtype)
                counts.scatter_add_(1, out["mask_indices"], ones)

            # Divide each patch by how many times it was masked; NaN where never masked.
            avg = torch.where(
                counts > 0, accumulated / counts.clamp(min=1),
                torch.full_like(accumulated, float("nan")),
            )
            H_p = int(math.sqrt(N))
            all_maps.append(avg.view(B, H_p, H_p).cpu().numpy())

    result = np.concatenate(all_maps, axis=0)
    nan_frac = float(np.isnan(result).mean())
    logger.info(f"Error maps: {result.shape} (NaN fraction {nan_frac:.3f})")
    return result


def aggregate_error_maps_per_cluster(
    error_maps: np.ndarray, cluster_labels: np.ndarray,
) -> dict[int, dict]:
    """Mean prediction error map per cluster, plus differential vs global mean.

    Uses nan-aware means so patches never masked for a given cell are ignored
    rather than counted as zero error.
    """
    global_mean = np.nanmean(error_maps, axis=0)
    out = {}
    for cl in sorted(set(cluster_labels)):
        if cl < 0:
            continue
        mask = cluster_labels == cl
        cl_mean = np.nanmean(error_maps[mask], axis=0)
        out[cl] = {"mean": cl_mean, "differential": cl_mean - global_mean, "n": int(mask.sum())}
    return out


# ── 2. Attention Map Analysis ─────────────────────────────────────────────

def extract_attention_maps(
    model: nn.Module, images: torch.Tensor, device: str = "cuda",
) -> np.ndarray:
    """Extract CLS→patch self-attention from all layers/heads of the ViT encoder.

    Only the CLS-token attention row is retained (N_patches values per head), not
    the full N×N matrix — the latter is ~11 MB/image for ViT-S and OOMs at dataset
    scale. The CLS row is what drives the pooled representation and is what the
    per-cluster aggregation needs.

    Returns: (B, n_layers, n_heads, n_patches) — attention from CLS to each patch.
    """
    model.eval()
    images = images.to(device)
    attn_store = []

    def make_hook(storage):
        # forward_pre_hook receives (module, args) only — never a third 'output' arg.
        def hook(module, args):
            x = args[0]
            B, N, C = x.shape
            nh = module.num_heads
            hd = C // nh
            scale = getattr(module, "scale", hd ** -0.5)
            qkv = module.qkv(x).reshape(B, N, 3, nh, hd).permute(2, 0, 3, 1, 4)
            q, k, _ = qkv.unbind(0)
            attn = (q @ k.transpose(-2, -1)) * scale
            attn = attn.softmax(dim=-1)
            # Keep only CLS (row 0) → patch tokens (cols 1:): (B, heads, n_patches)
            storage.append(attn[:, :, 0, 1:].detach().cpu().numpy())
        return hook

    hooks = [block.attn.register_forward_pre_hook(make_hook(attn_store))
             for block in model.encoder.blocks]

    with torch.no_grad():
        model.encoder.forward_features(images)

    for h in hooks:
        h.remove()

    if attn_store:
        # (n_layers, B, heads, n_patches) → (B, layers, heads, n_patches)
        return np.stack(attn_store, axis=0).transpose(1, 0, 2, 3)
    return np.array([])


def cls_attention_per_cluster(
    attention_maps: np.ndarray, cluster_labels: np.ndarray,
) -> dict[int, np.ndarray]:
    """Mean CLS→patch attention per cluster. Shows which spatial regions matter.

    Expects attention_maps of shape (N_cells, layers, heads, n_patches) as returned
    by extract_attention_maps.
    """
    out = {}
    for cl in sorted(set(cluster_labels)):
        if cl < 0:
            continue
        mask = cluster_labels == cl
        out[cl] = attention_maps[mask].mean(axis=0)  # (layers, heads, n_patches)
    return out


# ── 3. Channel Attribution ────────────────────────────────────────────────

def channel_importance_from_weights(model: nn.Module) -> dict[int, float]:
    """Channel importance from patch embedding conv weight norms (fast, no data needed)."""
    w = model.encoder.patch_embed.proj.weight.data  # (D, C, P, P)
    norms = {c: torch.norm(w[:, c], p="fro").item() for c in range(w.shape[1])}
    total = sum(norms.values())
    return {c: v / total for c, v in norms.items()} if total > 0 else norms


def channel_attribution_gradient(
    model: nn.Module, images: torch.Tensor, device: str = "cuda",
) -> np.ndarray:
    """Per-channel importance via gradient of CLS norm w.r.t. input. Returns (B, C).

    Wrapped in enable_grad so it is correct even when called inside a torch.no_grad
    context (e.g. an evaluation loop).
    """
    model.eval()
    with torch.enable_grad():
        images = images.to(device).detach().requires_grad_(True)
        features = model.encoder.forward_features(images)
        cls_norm = features[:, 0].norm(dim=-1).sum()
        cls_norm.backward()
        grad = images.grad.detach().abs().mean(dim=(2, 3)).cpu().numpy()
    return grad


def channel_ablation(
    model: nn.Module, dataloader, device: str = "cuda", n_channels: int = 5,
) -> dict[int, dict[str, float]]:
    """Leave-one-out channel ablation: zero each channel, measure embedding shift."""
    model.eval()
    shifts = {c: [] for c in range(n_channels)}

    with torch.no_grad():
        for images, *_ in tqdm(dataloader, desc="Channel ablation"):
            images = images.to(device)
            full = model.extract_embeddings(images)["cls_token"]

            for c in range(n_channels):
                ablated = images.clone()
                ablated[:, c] = 0.0
                abl = model.extract_embeddings(ablated)["cls_token"]
                dist = 1.0 - torch.nn.functional.cosine_similarity(full, abl, dim=-1)
                shifts[c].append(dist.cpu().numpy())

    return {
        c: {"mean_shift": float(np.mean(np.concatenate(v))),
            "std_shift": float(np.std(np.concatenate(v)))}
        for c, v in shifts.items()
    }
