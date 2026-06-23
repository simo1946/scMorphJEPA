"""Baseline encoder wrapper for fair comparison against scMorphJEPA.

Wraps any frozen timm ViT (DINO, scDINO, Cell-DINO, DINOv3, ...) so it exposes
the same `extract_embeddings()` interface as ScMorphJEPA. This guarantees the
benchmark comparison flows every model through one identical pipeline
(extract_embeddings → knn_evaluate → cluster analysis), which is what a
reviewer will expect for the Paper 1 comparison table.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
import timm

from scmorphjepa.models.builder import adapt_patch_embed

logger = logging.getLogger(__name__)


class EncoderWrapper(nn.Module):
    """Wrap a timm ViT encoder to match the ScMorphJEPA embedding interface.

    Args:
        encoder: A timm ViT model (num_classes=0) exposing forward_features.
        frozen: If True, parameters are frozen and the module is set to eval.
    """

    def __init__(self, encoder: nn.Module, frozen: bool = True) -> None:
        super().__init__()
        self.encoder = encoder
        if frozen:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
            self.encoder.eval()

    @torch.no_grad()
    def extract_embeddings(self, images: torch.Tensor) -> dict:
        """Return {'cls_token': (B, D), 'patch_tokens': (B, N, D)} — same as ScMorphJEPA."""
        features = self.encoder.forward_features(images)
        return {"cls_token": features[:, 0], "patch_tokens": features[:, 1:]}


def build_baseline_encoder(
    model_name: str = "vit_small_patch16_224",
    checkpoint_path: str | Path | None = None,
    in_channels: int = 5,
    embed_dim: int = 384,
    frozen: bool = True,
) -> EncoderWrapper:
    """Build a frozen baseline encoder (e.g. DINO/scDINO/Cell-DINO) for benchmarking.

    Args:
        model_name: timm model name matching the checkpoint's architecture.
        checkpoint_path: Path to the pretrained weights (plain state_dict).
        in_channels: Number of input channels; patch embedding is adapted if != 3.
        embed_dim: Encoder embedding dimension (384 for ViT-S).
        frozen: Whether to freeze the encoder (True for zero-shot baselines).

    Returns:
        EncoderWrapper exposing extract_embeddings().
    """
    encoder = timm.create_model(model_name, pretrained=False, num_classes=0)

    if checkpoint_path is not None:
        ckpt_path = Path(checkpoint_path)
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            # Handle common checkpoint nesting (e.g. {'teacher': ..., 'student': ...})
            if isinstance(ckpt, dict):
                for key in ("teacher", "student", "model", "state_dict"):
                    if key in ckpt and isinstance(ckpt[key], dict):
                        ckpt = ckpt[key]
                        break
            # Strip common prefixes
            ckpt = {k.replace("backbone.", "").replace("module.", ""): v for k, v in ckpt.items()}
            missing, unexpected = encoder.load_state_dict(ckpt, strict=False)
            logger.info(
                f"Baseline {model_name}: loaded {ckpt_path.name} "
                f"({len(missing)} missing, {len(unexpected)} unexpected keys)"
            )
        else:
            logger.warning(f"Baseline checkpoint not found: {ckpt_path}")

    encoder = adapt_patch_embed(encoder, in_channels, embed_dim)
    return EncoderWrapper(encoder, frozen=frozen)
