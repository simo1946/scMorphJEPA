"""Test suite for scMorphJEPA.

Run with: pytest tests/ -v

Tests are split into:
    - Pure numpy/scipy tests (always run): feature schema, clustering, stats, error-map math
    - Torch tests (skipped if torch unavailable): model forward, checkpoint round-trip, masking
"""

import numpy as np
import pytest

torch = pytest.importorskip  # placeholder; real import guarded below

try:
    import torch as _torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ── Pure-Python tests (no torch needed) ───────────────────────────────────

def test_feature_extractor_schema():
    """CellProfiler-lite returns a MultiIndex DataFrame with expected categories."""
    from scmorphjepa.analysis.feature_extractors import extract_dataset_features, FeatureConfig

    rng = np.random.default_rng(0)
    images = rng.random((5, 3, 50, 50)).astype(np.float32)
    df = extract_dataset_features(
        images, config=FeatureConfig(channel_names=["a", "b", "c"]), show_progress=False
    )
    assert len(df) == 5
    cats = {c[0] for c in df.columns}
    assert "intensity" in cats
    assert "morphology" in cats
    assert "texture" in cats
    # No NaNs in features
    assert not df.isnull().any().any()


def test_cluster_composition_proportions_sum_to_one():
    """Composition rows (excluding metadata cols) should sum to 1 per cluster."""
    from scmorphjepa.analysis.cluster_profiler import compute_cluster_composition

    cluster_labels = np.array([0, 0, 1, 1, 1, 2])
    cell_types = np.array([0, 1, 1, 1, 0, 0])
    comp = compute_cluster_composition(cluster_labels, cell_types, ["typeA", "typeB"])
    prop_cols = ["typeA", "typeB"]
    sums = comp[prop_cols].sum(axis=1).values
    assert np.allclose(sums, 1.0)
    assert "purity" in comp.columns
    assert "dominant_type" in comp.columns


def test_error_map_averaging_uses_mask_count():
    """Regression test for the n_samples vs mask-count averaging bug.

    Reproduces the averaging logic: a patch masked k of n times must be divided
    by k, not n. Dividing by n underestimates.
    """
    np.random.seed(0)
    N, mask_ratio, n_samples = 10, 0.6, 8
    n_mask = int(N * mask_ratio)
    true_difficulty = np.arange(N, dtype=float)

    accumulated = np.zeros(N)
    counts = np.zeros(N)
    for _ in range(n_samples):
        perm = np.random.permutation(N)
        mask_idx = perm[:n_mask]
        grid = np.zeros(N)
        grid[mask_idx] = true_difficulty[mask_idx]
        accumulated += grid
        counts[mask_idx] += 1

    correct = np.divide(accumulated, counts, out=np.full(N, np.nan), where=counts > 0)
    wrong = accumulated / n_samples

    # Correct recovers the true difficulty where the patch was ever masked
    masked = counts > 0
    assert np.allclose(correct[masked], true_difficulty[masked])
    # The buggy version does NOT (it underestimates for patches masked < n_samples)
    assert not np.allclose(wrong[masked], true_difficulty[masked])


def test_boundary_cohens_d_sign():
    """Cohen's d should be positive when cluster A mean > cluster B mean."""
    import pandas as pd
    from scmorphjepa.analysis.boundary_analyzer import compare_clusters

    n = 50
    # Feature is high for cluster 0, low for cluster 1
    vals = np.concatenate([np.ones(n) + np.random.RandomState(0).randn(n) * 0.1,
                            np.zeros(n) + np.random.RandomState(1).randn(n) * 0.1])
    cols = pd.MultiIndex.from_tuples([("intensity", "ch0", "mean")],
                                     names=["category", "channel", "feature"])
    df = pd.DataFrame(vals.reshape(-1, 1), columns=cols)
    labels = np.array([0] * n + [1] * n)

    result = compare_clusters(0, 1, df, labels)
    top = result.discriminative_features.iloc[0]
    assert top["cohens_d"] > 0
    assert top["significant"]


# ── Torch tests (skipped if torch missing) ────────────────────────────────

@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_model_forward_shapes():
    """Forward pass returns correctly shaped prediction/target/cls tensors."""
    from scmorphjepa.models.builder import build_scmorphjepa
    from scmorphjepa.models.cell_jepa import ScMorphJEPAConfig

    cfg = ScMorphJEPAConfig(in_channels=5, mask_ratio=0.6)
    model = build_scmorphjepa(checkpoint_path=None, config=cfg)
    model.eval()

    x = _torch.randn(2, 5, 224, 224)
    out = model(x, return_error_map=True)

    N = cfg.num_patches
    n_mask = int(N * cfg.mask_ratio)
    assert out["pred_emb"].shape == (2, n_mask, cfg.embed_dim)
    assert out["target_emb"].shape == (2, n_mask, cfg.embed_dim)
    assert out["cls_token"].shape == (2, cfg.embed_dim)
    assert out["mask_indices"].shape == (2, n_mask)  # per-sample
    H_p = int(N ** 0.5)
    assert out["error_map"].shape == (2, H_p, H_p)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_per_sample_masks_differ():
    """Per-sample masking should give different masks across the batch."""
    from scmorphjepa.models.builder import build_scmorphjepa
    from scmorphjepa.models.cell_jepa import ScMorphJEPAConfig

    model = build_scmorphjepa(checkpoint_path=None, config=ScMorphJEPAConfig(in_channels=5))
    mask_idx, visible_idx = model.random_mask(196, batch_size=4, device=_torch.device("cpu"))
    assert mask_idx.shape[0] == 4
    # Extremely unlikely two random masks are identical across all positions
    assert not _torch.equal(mask_idx[0], mask_idx[1])
    # Mask and visible are disjoint and cover all patches per sample
    for b in range(4):
        union = _torch.cat([mask_idx[b], visible_idx[b]]).sort().values
        assert _torch.equal(union, _torch.arange(196))


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_checkpoint_roundtrip():
    """State dict saves and loads with identical keys."""
    from scmorphjepa.models.builder import build_scmorphjepa
    from scmorphjepa.models.cell_jepa import ScMorphJEPAConfig

    cfg = ScMorphJEPAConfig(in_channels=5)
    model = build_scmorphjepa(checkpoint_path=None, config=cfg)
    sd = model.state_dict()

    model2 = build_scmorphjepa(checkpoint_path=None, config=cfg)
    model2.load_state_dict(sd, strict=True)  # raises on any mismatch
    assert set(model.state_dict().keys()) == set(model2.state_dict().keys())


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_extract_embeddings_no_masking():
    """extract_embeddings is deterministic (no masking) — same input, same output."""
    from scmorphjepa.models.builder import build_scmorphjepa
    from scmorphjepa.models.cell_jepa import ScMorphJEPAConfig

    model = build_scmorphjepa(checkpoint_path=None, config=ScMorphJEPAConfig(in_channels=5))
    model.eval()
    x = _torch.randn(2, 5, 224, 224)
    out1 = model.extract_embeddings(x)
    out2 = model.extract_embeddings(x)
    assert _torch.allclose(out1["cls_token"], out2["cls_token"])


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_all_regularizers_finite_and_differentiable():
    """Every regularizer returns a finite scalar; non-trivial ones backprop into a real loss."""
    from scmorphjepa.training.regularizers import build_regularizer, available_regularizers
    assert set(available_regularizers()) == {"sigreg", "vicreg", "koleo", "barlow", "none"}
    z = _torch.randn(16, 384, requires_grad=True)
    for name in available_regularizers():
        reg = build_regularizer(name)
        out = reg(z)
        assert _torch.isfinite(out), f"{name} produced non-finite loss"
        # emulate training: regularizer added to a loss that already requires grad
        total = (z ** 2).mean() + out
        total.backward()
        z.grad = None


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_run_name_encodes_regularizer():
    """Auto run_name appends the regularizer name unless it's the default sigreg."""
    from scmorphjepa.training.trainer import TrainConfig
    assert TrainConfig(n_images=5000, epochs=100).resolved_run_name() == "scmorphjepa_n5000_e100"
    assert TrainConfig(n_images=5000, epochs=100, regularizer="vicreg").resolved_run_name() \
        == "scmorphjepa_n5000_e100_vicreg"
