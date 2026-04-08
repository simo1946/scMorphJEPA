"""Cell-JEPA: Spatial masked prediction + SIGReg for cell microscopy."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import math


class SIGReg(nn.Module):
    """From LeWM — forces embeddings toward Gaussian distribution."""
    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        device = proj.device
        A = torch.randn(proj.size(-1), self.num_proj, device=device)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()


class SpatialPredictor(nn.Module):
    """Predicts masked patch embeddings from visible ones."""
    def __init__(self, embed_dim=384, depth=4, num_heads=6, mlp_ratio=4.0, num_patches=196):
        super().__init__()
        self.mask_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim) * 0.02)

        # Simple transformer blocks (no action conditioning, non-causal)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                dropout=0.0,
                activation='gelu',
                batch_first=True,
                norm_first=True,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, visible_emb, mask_indices, visible_indices, total_patches):
        B, _, D = visible_emb.shape

        # Build full sequence: mask tokens everywhere, then insert visible tokens
        full_seq = self.mask_token.expand(B, total_patches, -1).clone()
        full_seq[:, visible_indices] = visible_emb
        full_seq = full_seq + self.pos_embed[:, :total_patches]

        # Run through transformer (non-causal, all attend to all)
        for block in self.blocks:
            full_seq = block(full_seq)
        full_seq = self.norm(full_seq)

        # Return only predicted masked positions
        return full_seq[:, mask_indices]


class CellJEPA(nn.Module):
    """Full Cell-JEPA: encoder + spatial predictor."""
    def __init__(self, encoder, predictor, mask_ratio=0.6):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.mask_ratio = mask_ratio

    def random_mask(self, n_patches, device):
        n_mask = int(n_patches * self.mask_ratio)
        perm = torch.randperm(n_patches, device=device)
        return perm[:n_mask], perm[n_mask:]  # mask_indices, visible_indices

    def forward(self, images):
        B = images.size(0)

        # Encode full image (all patches)
        features = self.encoder.forward_features(images)  # (B, 1+N, 384)
        cls_token = features[:, 0]        # (B, 384)
        patch_tokens = features[:, 1:]    # (B, 196, 384)
        N = patch_tokens.size(1)

        # Generate mask
        mask_idx, visible_idx = self.random_mask(N, images.device)

        # Targets: masked patch embeddings (stop gradient!)
        target_emb = patch_tokens[:, mask_idx].detach()  # (B, n_mask, 384)

        # Input to predictor: visible patch embeddings
        visible_emb = patch_tokens[:, visible_idx]  # (B, n_visible, 384)

        # Predict masked patches
        pred_emb = self.predictor(visible_emb, mask_idx, visible_idx, N)

        return {
            "pred_emb": pred_emb,       # (B, n_mask, 384)
            "target_emb": target_emb,   # (B, n_mask, 384)
            "cls_token": cls_token,      # (B, 384) for downstream eval
            "all_patches": patch_tokens, # (B, 196, 384) for SIGReg
        }


def build_cell_jepa(checkpoint_path, in_channels=5, mask_ratio=0.6, predictor_depth=4):
    """Build Cell-JEPA from a DINO pretrained ViT-S/16."""

    # Load ViT-S/16
    encoder = timm.create_model('vit_small_patch16_224', pretrained=False, num_classes=0)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    encoder.load_state_dict(ckpt, strict=False)

    # Adapt to N channels
    if in_channels != 3:
        old_proj = encoder.patch_embed.proj
        old_weight = old_proj.weight.data
        new_proj = nn.Conv2d(in_channels, 384, kernel_size=16, stride=16)
        new_weight = torch.zeros(384, in_channels, 16, 16)
        new_weight[:, :3, :, :] = old_weight
        for c in range(3, in_channels):
            new_weight[:, c, :, :] = old_weight.mean(dim=1)
        new_proj.weight = nn.Parameter(new_weight)
        new_proj.bias = old_proj.bias
        encoder.patch_embed.proj = new_proj

    # Build predictor
    predictor = SpatialPredictor(
        embed_dim=384,
        depth=predictor_depth,
        num_heads=6,
        num_patches=196,
    )

    model = CellJEPA(encoder, predictor, mask_ratio=mask_ratio)
    return model
