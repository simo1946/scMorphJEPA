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


def test_profile_clusters_by_features_bridges_clusters_to_features():
    """The cluster-feature profiler links clusters to raw-image features per (type, cluster)."""
    from scmorphjepa.analysis.feature_extractors import (
        profile_clusters_by_features, FeatureConfig,
    )

    rng = np.random.default_rng(1)
    n = 30
    imgs = [rng.random((5, 50, 50)).astype(np.float32) for _ in range(n)]

    class _MockDS:
        cell_types = ["T4", "M0"]
        labels = [i % 2 for i in range(n)]

        def get_raw_image(self, i):
            return imgs[i]

    clusters = np.array([i % 4 for i in range(n)])
    out = profile_clusters_by_features(
        _MockDS(), clusters,
        config=FeatureConfig(channel_names=["c0", "c1", "c2", "c3", "c4"]),
        show_progress=False,
    )
    assert set(out) == {"per_cell", "per_cluster", "by_type_cluster", "state_signal"}
    assert len(out["per_cell"]) == n
    assert (out["per_cell"]["cluster"].values == clusters).all()
    assert "dominant_type" in out["per_cluster"].columns
    assert out["by_type_cluster"].index.names == ["cell_type", "cluster"]
    assert {"cell_type", "feature", "across_cluster_spread"} <= set(out["state_signal"].columns)


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
    assert set(available_regularizers()) == {"sigreg", "vicreg", "koleo", "barlow", "visreg", "none"}
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


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_normalize_validation_and_none(tmp_path):
    """Invalid normalize raises at construction; 'none' preserves raw pixels."""
    import tifffile
    from scmorphjepa.data.datasets import SeverinDataset, VALID_NORMALIZE

    cls_dir = tmp_path / "Training" / "T4"
    cls_dir.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for j in range(3):
        tifffile.imwrite(cls_dir / f"c{j}.tiff", (rng.random((50, 50, 5)) * 1000).astype(np.float32))
    data_dir = tmp_path / "Training"

    assert "none" in VALID_NORMALIZE
    with pytest.raises(ValueError):
        SeverinDataset(data_dir, normalize="_per_channel_percentile")

    raw = SeverinDataset(data_dir, normalize="none").get_raw_image(0)
    scaled = SeverinDataset(data_dir, normalize="per_image").get_raw_image(0)
    assert raw.max() > 1.0             # raw pixel values preserved
    assert scaled.max() <= 1.0 + 1e-6  # per_image scaled into [0, 1]


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_dino_load_fails_loud_on_mismatch(tmp_path):
    """A checkpoint that matches almost nothing must raise, not silently random-init."""
    import torch
    from scmorphjepa.models.builder import build_scmorphjepa
    from scmorphjepa.models.cell_jepa import ScMorphJEPAConfig

    garbage = tmp_path / "garbage.pth"
    torch.save({"not_a_real_key": torch.zeros(3)}, garbage)
    with pytest.raises(RuntimeError):
        build_scmorphjepa(checkpoint_path=str(garbage), config=ScMorphJEPAConfig(in_channels=5))


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_visreg_finite_differentiable_penalizes_collapse():
    """VISReg is finite, differentiable, and costs more on a collapsed batch than a Gaussian one."""
    import torch
    from scmorphjepa.training.regularizers import build_regularizer, available_regularizers

    assert "visreg" in available_regularizers()
    reg = build_regularizer("visreg", num_projections=128)

    g = torch.randn(64, 128, requires_grad=True)
    lg = reg(g)
    lg.backward()
    assert torch.isfinite(lg) and lg.item() >= 0
    assert g.grad is not None and torch.isfinite(g.grad).all()

    collapsed = torch.randn(1, 128).repeat(64, 1) + 1e-4 * torch.randn(64, 128)
    assert reg(collapsed).item() > lg.item()          # collapse costs more


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_resume_preserves_full_optimizer_state(tmp_path):
    """A cross-session resume must restore Adam EXACTLY (moments + step), not re-initialize it.

    This is the regression test for the v0.1.12 defect: dropping the optimizer state from the
    Drive checkpoint silently perturbed the optimization trajectory of every resumed run.
    """
    import shutil
    import torch
    from torch.utils.data import TensorDataset
    from scmorphjepa.models.builder import build_scmorphjepa
    from scmorphjepa.models.cell_jepa import ScMorphJEPAConfig
    from scmorphjepa.training.trainer import Trainer, TrainConfig

    ds = TensorDataset(torch.rand(8, 5, 224, 224), torch.randint(0, 2, (8,)))
    out, drv = tmp_path / "out", tmp_path / "drive"

    def mk(epochs):
        m = build_scmorphjepa(None, ScMorphJEPAConfig(in_channels=5))
        cfg = TrainConfig(batch_size=4, epochs=epochs, num_workers=0, output_dir=str(out),
                          device="cpu", n_images=0, drive_checkpoint_dir=str(drv),
                          run_name="rtest", drive_save_every=1, save_every=999, resume=True)
        return Trainer(m, ds, ds, cfg)

    tr = mk(2)
    tr.train()

    # The Drive checkpoint must contain the FULL optimizer state. With two rotating slots, the
    # LATEST epoch is in whichever slot has the highest "epoch" — read that one (the resume will
    # load the same freshest checkpoint), not a hard-coded slot name.
    slots = list(drv.glob("rtest_last_*.pt"))
    assert slots, "no Drive checkpoint slot written"
    loaded = [(torch.load(s, map_location="cpu", weights_only=False), s) for s in slots]
    ck, _ = max(loaded, key=lambda t: int(t[0]["epoch"]))
    assert "optimizer_state_dict" in ck, "Drive checkpoint must keep the optimizer state"
    saved_step = ck["optimizer_state_dict"]["state"][0]["step"]
    saved_exp_avg = ck["optimizer_state_dict"]["state"][0]["exp_avg"].clone()
    assert float(saved_step) > 0, "Adam step counter should be non-zero after training"

    # progress.json must not over-report what is recoverable.
    import json
    prog = json.loads((drv / "rtest_progress.json").read_text())
    assert prog["recoverable_epoch"] == prog["epoch_completed"]
    assert prog["epochs_at_risk"] == 0
    assert prog["resume_count"] == 0

    # Simulate a fresh runtime: local is wiped, only Drive survives.
    shutil.rmtree(out)
    out.mkdir()
    tr2 = mk(3)
    next_epoch = tr2._maybe_resume()
    assert next_epoch == 2, f"expected to continue at epoch index 2, got {next_epoch}"

    # Adam must be restored EXACTLY from the freshest checkpoint, not re-initialized.
    restored = tr2.optimizer.state_dict()["state"][0]
    assert float(restored["step"]) > 0, "Adam step counter was reset to 0 (optimizer re-initialized)"
    assert float(restored["step"]) == float(saved_step), "Adam step not restored from the freshest slot"
    assert torch.allclose(restored["exp_avg"], saved_exp_avg), "Adam moments were not restored"
    assert tr2._resume_count == 1 and tr2._resume_epochs == [2]   # provenance recorded


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_resume_refuses_optimizer_free_checkpoint(tmp_path):
    """A pre-0.1.13 (optimizer-free) checkpoint must raise, not silently re-init Adam."""
    import torch
    from torch.utils.data import TensorDataset
    from scmorphjepa.models.builder import build_scmorphjepa
    from scmorphjepa.models.cell_jepa import ScMorphJEPAConfig
    from scmorphjepa.training.trainer import Trainer, TrainConfig

    ds = TensorDataset(torch.rand(4, 5, 224, 224), torch.randint(0, 2, (4,)))
    out, drv = tmp_path / "out", tmp_path / "drive"
    out.mkdir()
    drv.mkdir()

    m = build_scmorphjepa(None, ScMorphJEPAConfig(in_channels=5))
    cfg = TrainConfig(batch_size=4, epochs=2, num_workers=0, output_dir=str(out), device="cpu",
                      n_images=0, drive_checkpoint_dir=str(drv), run_name="legacy",
                      drive_save_every=1, save_every=999, resume=True)
    tr = Trainer(m, ds, ds, cfg)

    # legacy-style checkpoint: no optimizer_state_dict
    torch.save({"epoch": 3, "model_state_dict": tr.model.state_dict(),
                "scheduler_state_dict": tr.scheduler.state_dict()},
               drv / "legacy_last.pt")
    with pytest.raises(RuntimeError, match="no optimizer state"):
        tr._maybe_resume()
