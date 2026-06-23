"""Pluggable anti-collapse regularizers for JEPA training.

JEPA has no contrastive/negative term, so an encoder can collapse without a regularizer.
Different regularizers impose different geometric priors on the embedding distribution, and
which one is least harmful is modality-dependent (e.g. an isotropic-Gaussian prior may suit
imaging but over-constrain sparse/anisotropic omics embeddings). This module makes the choice
a single swappable component so it can be ablated cleanly:

    reg = build_regularizer("vicreg")
    loss = pred_loss + weight * reg(cls_embeddings)   # cls_embeddings: (B, D)

All regularizers share the interface `forward(z: (B, D)) -> scalar`.

IMPORTANT: the regularizers live on very different numeric scales, so the *weight* must be
retuned per regularizer — you cannot keep weight=10 (good for SIGReg) when switching to VICReg
or KoLeo. Treat the provided defaults as starting points and sweep the weight for each.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from scmorphjepa.models.cell_jepa import SIGReg


def _off_diagonal(m: torch.Tensor) -> torch.Tensor:
    """Return the off-diagonal elements of a square matrix as a 1-D tensor."""
    n, _ = m.shape
    return m.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class NoReg(nn.Module):
    """No regularization — for the ablation that measures a regularizer's actual contribution."""

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z.new_zeros(())


class SIGRegWrap(nn.Module):
    """Wraps SIGReg (isotropic-Gaussian matching) behind the uniform (B, D) interface."""

    def __init__(self, knots: int = 17, num_proj: int = 1024) -> None:
        super().__init__()
        self.sig = SIGReg(knots=knots, num_proj=num_proj)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.sig(z.unsqueeze(0) if z.ndim == 2 else z)


class VICReg(nn.Module):
    """Variance + Covariance terms of VICReg (the invariance term is the JEPA predictor loss).

    Variance hinges each dimension's std up to `gamma`; covariance penalizes off-diagonal
    correlations. A softer prior than SIGReg — it does not impose full Gaussianity, only a
    variance floor and decorrelation.
    """

    def __init__(self, gamma: float = 1.0, var_weight: float = 25.0, cov_weight: float = 1.0) -> None:
        super().__init__()
        self.gamma, self.var_weight, self.cov_weight = gamma, var_weight, cov_weight

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = z - z.mean(dim=0)
        std = torch.sqrt(z.var(dim=0) + 1e-4)
        var_loss = F.relu(self.gamma - std).mean()
        B, D = z.shape
        cov = (z.T @ z) / max(B - 1, 1)
        cov_loss = _off_diagonal(cov).pow(2).sum() / D
        return self.var_weight * var_loss + self.cov_weight * cov_loss


class KoLeo(nn.Module):
    """Kozachenko-Leonenko entropy regularizer (as used in DINOv2).

    Encourages a uniform spread by penalizing small nearest-neighbour distances within the
    batch — pushes embeddings apart without imposing a parametric distribution.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = F.normalize(z, dim=1, eps=self.eps)
        sim = z @ z.T
        sim.fill_diagonal_(-2.0)               # exclude self
        nn_sim = sim.max(dim=1).values         # nearest neighbour = highest cosine sim
        nn_dist = torch.sqrt(torch.clamp(2.0 - 2.0 * nn_sim, min=self.eps))
        return -torch.log(nn_dist + self.eps).mean()


class BarlowReg(nn.Module):
    """Single-view Barlow-Twins-style whitening: push the embedding auto-correlation to identity.

    NOTE: classic Barlow Twins decorrelates the cross-correlation between TWO augmented views.
    JEPA is single-view, so this is the single-view adaptation — the auto-correlation matrix of
    the (batch-normalized) embeddings is driven toward the identity (unit variance on the
    diagonal, decorrelation off it). Conceptually close to VICReg; included for completeness.
    """

    def __init__(self, lambda_off: float = 0.005) -> None:
        super().__init__()
        self.lambda_off = lambda_off

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B, D = z.shape
        z = (z - z.mean(dim=0)) / (z.std(dim=0) + 1e-4)
        c = (z.T @ z) / B
        on_diag = (torch.diagonal(c) - 1.0).pow(2).sum()
        off_diag = _off_diagonal(c).pow(2).sum()
        return on_diag + self.lambda_off * off_diag


_REGISTRY = {
    "sigreg": SIGRegWrap,
    "vicreg": VICReg,
    "koleo": KoLeo,
    "barlow": BarlowReg,
    "none": NoReg,
}


def available_regularizers() -> list[str]:
    return list(_REGISTRY)


def build_regularizer(name: str = "sigreg", **kwargs) -> nn.Module:
    """Construct a regularizer by name. kwargs are forwarded to its constructor.

    Names: sigreg, vicreg, koleo, barlow, none.
    """
    key = name.lower()
    if key not in _REGISTRY:
        raise ValueError(f"Unknown regularizer '{name}'. Options: {available_regularizers()}")
    return _REGISTRY[key](**kwargs)
