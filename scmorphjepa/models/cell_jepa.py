"""scMorphJEPA: Spatial masked prediction + SIGReg for cell microscopy.

Architecture:
    Encoder (ViT-S/16) → CLS + patch tokens → mask → SpatialPredictor → predicted embeddings
    SIGReg regularizes CLS tokens toward isotropic Gaussian, preventing collapse without EMA.

References:
    - I-JEPA: Assran et al., CVPR 2023
    - LeWorldModel: Maes et al., arXiv 2026 (SIGReg)
    - scDINO: Pfaendler et al., bioRxiv 2023
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class ScMorphJEPAConfig:
    """Model configuration."""
    embed_dim: int = 384
    predictor_depth: int = 4
    predictor_heads: int = 6
    predictor_mlp_ratio: float = 4.0
    num_patches: int = 196
    mask_ratio: float = 0.6
    in_channels: int = 5
    sigreg_knots: int = 17
    sigreg_num_proj: int = 1024


class SIGReg(nn.Module):
    """SIGReg from LeWorldModel — forces embeddings toward isotropic Gaussian.

    Prevents representation collapse without EMA or contrastive negatives by
    matching the characteristic function of the embedding distribution to a
    standard normal.
    """

    def __init__(self, knots: int = 17, num_proj: int = 1024) -> None:
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3.0 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        if proj.ndim == 2:
            proj = proj.unsqueeze(0)
        device = proj.device
        A = torch.randn(proj.size(-1), self.num_proj, device=device)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()


class SpatialPredictor(nn.Module):
    """Predicts masked patch embeddings from visible context using bidirectional attention."""

    def __init__(
        self, embed_dim: int = 384, depth: int = 4, num_heads: int = 6,
        mlp_ratio: float = 4.0, num_patches: int = 196,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_patches = num_patches
        self.mask_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim) * 0.02)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, visible_emb, mask_indices, visible_indices, total_patches):
        """Predict masked patch embeddings from visible context.

        Supports per-sample masks: mask_indices and visible_indices are (B, k)
        with potentially different patch sets per image in the batch.
        """
        B, _, D = visible_emb.shape
        full_seq = self.mask_token.expand(B, total_patches, D).clone()
        # Scatter visible embeddings into their (per-sample) positions
        full_seq.scatter_(1, visible_indices.unsqueeze(-1).expand(-1, -1, D), visible_emb)
        full_seq = full_seq + self.pos_embed[:, :total_patches]
        for block in self.blocks:
            full_seq = block(full_seq)
        full_seq = self.norm(full_seq)
        # Gather predicted embeddings at the (per-sample) masked positions
        return torch.gather(full_seq, 1, mask_indices.unsqueeze(-1).expand(-1, -1, D))


class ScMorphJEPA(nn.Module):
    """scMorphJEPA: encoder + spatial predictor for cell morphology learning.

    Forward pass encodes full image, masks patches, and predicts masked
    representations from visible context. Supports error map extraction
    for interpretability analysis (novel contribution).
    """

    def __init__(self, encoder: nn.Module, predictor: SpatialPredictor,
                 mask_ratio: float = 0.6) -> None:
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.mask_ratio = mask_ratio

    def random_mask(self, n_patches: int, batch_size: int, device: torch.device):
        """Generate per-sample random masks.

        Returns (mask_idx, visible_idx), each shape (batch_size, k) — a different
        random patch partition per image (unlike a single shared mask), which
        increases gradient diversity and de-correlates the error-map samples.
        """
        n_mask = int(n_patches * self.mask_ratio)
        noise = torch.rand(batch_size, n_patches, device=device)
        ids = torch.argsort(noise, dim=1)
        return ids[:, :n_mask], ids[:, n_mask:]

    def forward(self, images: torch.Tensor, return_error_map: bool = False) -> dict:
        B = images.size(0)
        features = self.encoder.forward_features(images)
        cls_token = features[:, 0]
        patch_tokens = features[:, 1:]
        N = patch_tokens.size(1)
        D = patch_tokens.size(-1)

        mask_idx, visible_idx = self.random_mask(N, B, images.device)
        target_emb = torch.gather(
            patch_tokens, 1, mask_idx.unsqueeze(-1).expand(-1, -1, D)
        ).detach()
        visible_emb = torch.gather(patch_tokens, 1, visible_idx.unsqueeze(-1).expand(-1, -1, D))
        pred_emb = self.predictor(visible_emb, mask_idx, visible_idx, N)

        output = {
            "pred_emb": pred_emb,
            "target_emb": target_emb,
            "cls_token": cls_token,
            "all_patches": patch_tokens,
            "mask_indices": mask_idx,        # (B, n_mask)
            "visible_indices": visible_idx,  # (B, n_visible)
        }

        if return_error_map:
            with torch.no_grad():
                per_patch_mse = (pred_emb - target_emb).pow(2).mean(dim=-1)  # (B, n_mask)
                H_p = int(math.sqrt(N))
                error_grid = torch.zeros(B, N, device=images.device)
                error_grid.scatter_(1, mask_idx, per_patch_mse)
                output["error_map"] = error_grid.view(B, H_p, H_p)

        return output

    @torch.no_grad()
    def extract_embeddings(self, images: torch.Tensor) -> dict:
        """Extract embeddings without masking (for evaluation/analysis)."""
        features = self.encoder.forward_features(images)
        return {"cls_token": features[:, 0], "patch_tokens": features[:, 1:]}
