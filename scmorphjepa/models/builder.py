"""Build scMorphJEPA from pretrained DINO weights with channel adaptation."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
import timm

from scmorphjepa.models.cell_jepa import ScMorphJEPA, ScMorphJEPAConfig, SpatialPredictor

logger = logging.getLogger(__name__)


def adapt_patch_embed(encoder: nn.Module, in_channels: int, embed_dim: int = 384) -> nn.Module:
    """Adapt 3-channel ViT patch embedding to N channels.

    Copies RGB weights for first 3 channels; extra channels initialized
    from RGB mean (preserves pretrained features).
    """
    if in_channels == 3:
        return encoder
    old_proj = encoder.patch_embed.proj
    old_weight = old_proj.weight.data
    ps = old_weight.shape[-1]
    new_proj = nn.Conv2d(in_channels, embed_dim, kernel_size=ps, stride=ps)
    new_weight = torch.zeros(embed_dim, in_channels, ps, ps)
    new_weight[:, :3, :, :] = old_weight
    rgb_mean = old_weight.mean(dim=1, keepdim=True)
    for c in range(3, in_channels):
        new_weight[:, c : c + 1, :, :] = rgb_mean
    new_proj.weight = nn.Parameter(new_weight)
    new_proj.bias = old_proj.bias
    encoder.patch_embed.proj = new_proj
    logger.info(f"Patch embedding adapted: 3 → {in_channels} channels")
    return encoder


def build_scmorphjepa(
    checkpoint_path: str | Path | None = None,
    config: ScMorphJEPAConfig | None = None,
) -> ScMorphJEPA:
    """Build scMorphJEPA model.

    Args:
        checkpoint_path: Path to DINO ViT-S/16 pretrained checkpoint.
        config: Model configuration (defaults to ScMorphJEPAConfig()).
    """
    if config is None:
        config = ScMorphJEPAConfig()

    encoder = timm.create_model("vit_small_patch16_224", pretrained=False, num_classes=0)

    if checkpoint_path is not None:
        ckpt_path = Path(checkpoint_path)
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            missing, unexpected = encoder.load_state_dict(ckpt, strict=False)
            n_total = len(encoder.state_dict())
            n_loaded = n_total - len(missing)
            logger.info(
                f"Loaded DINO checkpoint from {ckpt_path} "
                f"({n_loaded}/{n_total} tensors matched, {len(unexpected)} unexpected)"
            )
            if n_loaded < 0.5 * n_total:
                raise RuntimeError(
                    f"DINO checkpoint matched only {n_loaded}/{n_total} encoder tensors; the "
                    "initialization did not take effect (wrong or corrupt file?). "
                    "Expected a ViT-S/16 DINO state_dict."
                )
        else:
            logger.warning(f"Checkpoint not found: {ckpt_path}")

    encoder = adapt_patch_embed(encoder, config.in_channels, config.embed_dim)

    predictor = SpatialPredictor(
        embed_dim=config.embed_dim, depth=config.predictor_depth,
        num_heads=config.predictor_heads, mlp_ratio=config.predictor_mlp_ratio,
        num_patches=config.num_patches,
    )

    model = ScMorphJEPA(encoder, predictor, mask_ratio=config.mask_ratio)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"scMorphJEPA: {n_params:,} parameters")
    return model


def load_trained_model(
    model_path: str | Path,
    checkpoint_path: str | Path | None = None,
    config: ScMorphJEPAConfig | None = None,
    device: str = "cpu",
) -> ScMorphJEPA:
    """Load a trained scMorphJEPA from saved state dict."""
    model = build_scmorphjepa(checkpoint_path=checkpoint_path, config=config)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model
