# Changelog

## v0.1.1 — Code review fixes

This release fixes bugs found in a full review of v0.1.0 and adds tests + CI.
All findings were verified against the actual code before fixing.

### Critical bugs fixed

- **Attention extraction crash** (`analysis/interpretability.py`). `extract_attention_maps`
  registered a `forward_pre_hook` with a 3-argument `(module, args, output)` signature;
  PyTorch calls pre-hooks with `(module, args)` only, so this raised `TypeError` on every
  call. Fixed the signature. Also made it robust to timm's `module.scale` and QK-norm.

- **Prediction-error saliency computed wrong values** (`analysis/interpretability.py`).
  `compute_prediction_error_maps` divided accumulated per-patch error by `n_samples`, but each
  patch is only masked in a subset of samples (error is 0 at visible positions). This
  underestimated error by up to ~60% in simulation and made never-masked patches read as 0.
  Now divides by the actual per-patch mask count and returns NaN for never-masked patches;
  aggregation uses nan-aware means. Default `n_samples` raised 5 → 8.

- **Fragile numpy import** (`cli.py`). `np` was used in `evaluate()` but only imported inside
  the `if __name__ == "__main__"` block — broke when the function was imported. Moved to top.

### Scalability fixes

- **Attention maps no longer OOM** (`analysis/interpretability.py`). Previously returned the
  full `(B, layers, heads, N, N)` tensor (~11 MB/image for ViT-S; ~270 GB over the 24K test
  set). Now retains only the CLS→patch row `(B, layers, heads, n_patches)`. `cls_attention_per_cluster`
  updated accordingly.

- **Faster, exact k-NN** (`evaluation/evaluate.py`). Replaced brute-force `metric="cosine"`
  (slow at 89K+ points) with L2-normalization + Euclidean (rank-equivalent to cosine) using
  the fast exact backends. Removed `StandardScaler`, which distorted the cosine geometry the
  model is trained under.

### Correctness / science

- **Per-sample masking** (`models/cell_jepa.py`). `random_mask` now produces a different mask
  per image in the batch (was one shared mask for the whole batch), implemented with
  scatter/gather in `SpatialPredictor`. Verified: complete & disjoint partition, no
  information leakage, correct round-trip. **Safe for existing checkpoints** — inference
  (`extract_embeddings`) does no masking, so k-NN numbers are unchanged; this only improves
  training gradient diversity and de-correlates error-map samples.

- **Default normalization → `per_channel`** (`data/datasets.py`), so low-intensity channels
  aren't crushed by a bright one (matters for the channel-ablation experiment). Added a
  `per_channel_percentile` option (1–99% robust scaling, scDINO-style).

- **Cohen's d uses sample variance** (`analysis/boundary_analyzer.py`, `ddof=1`), and
  `identify_neighboring_clusters` now drops the self-neighbor (was counting each point as its
  own neighbor, inflating intra-cluster overlap).

- **Raw-image guard** (`analysis/cluster_profiler.py`). `run_full_profiling` documents that
  `images` must be raw (not the 224 upsample) for valid morphology/texture features, and
  asserts length consistency with embeddings.

### Security / robustness

- `torch.load(..., weights_only=True)` in `models/builder.py` (both loads are plain
  state_dicts).
- Lazy top-level imports (PEP 562) so the torch-free analysis layer can be used without torch.
- `channel_attribution_gradient` wrapped in `torch.enable_grad()` so it works inside no-grad
  evaluation contexts.

### New — for the Paper 1 comparison

- **`models/baselines.py`**: `EncoderWrapper` + `build_baseline_encoder` wrap any frozen timm
  ViT (DINO / scDINO / Cell-DINO / DINOv3) behind the same `extract_embeddings()` interface as
  scMorphJEPA, so every model flows through one identical pipeline
  (extract_embeddings → knn_evaluate → analysis). This is what makes the benchmark
  apples-to-apples.

### Tooling

- `tests/test_scmorphjepa.py`: feature-schema, cluster-composition, error-map-averaging
  regression, Cohen's d, model forward shapes, per-sample mask partition, checkpoint
  round-trip, deterministic embedding extraction. Pure-Python tests run without torch;
  torch tests skip gracefully if torch is absent.
- `.github/workflows/ci.yml`: ruff + pytest on Python 3.9 and 3.11.
- `configs/severin.yaml`: example config.
- Removed unused imports; fixed f-strings without placeholders; non-deprecated skimage
  property names (`axis_major_length`).

### Still open (recommended next, not yet done)

- HTML report generator (Phase 7) and discovery engine (Phase 6).
- `FolderMicroscopyDataset` assumes a constant channel count across files — add an assertion.
- Clustering still uses `StandardScaler` before cosine UMAP; defensible but worth an ablation.

## v0.1.2 — Checkpoint naming

- `TrainConfig.run_name` (auto-derived as `scmorphjepa_n{n_images}_e{epochs}` if unset) namespaces
  all checkpoints under `output_dir/<run_name>/`, so different training runs never overwrite each
  other. Within a run, `best_model.pt` still overwrites as it improves.
- `TrainConfig.drive_checkpoint_dir`: if set, only the **best** model is mirrored there as
  `<run_name>_best.pt` — small, persistent (e.g. Google Drive), one file per run, no epoch clutter.

## v0.1.3 — Linear probe + clustering agreement metrics

- `evaluation.linear_probe(train_emb, train_labels, test_emb, test_labels)`: standard SSL
  linear-probe (logistic regression on FROZEN embeddings, backbone never updated). Returns
  top-1 and balanced accuracy. Complements `knn_evaluate`; often separates JEPA-family
  features better than k-NN.
- `analysis.clustering_metrics(cluster_labels, true_labels)`: global, chance-adjusted
  agreement — Adjusted Rand Index, Normalized Mutual Information, and cell-weighted overall
  purity. Noise points (label < 0) excluded. Now also returned by `run_full_profiling` under
  the `clustering_metrics` key.

## v0.1.4 — Colab-safe resumable training

- `Trainer` now auto-resumes after a runtime disconnect. Every epoch it writes a full-state
  checkpoint (`last.pt`: model + optimizer + scheduler + epoch + best loss + history),
  atomically (temp file + os.replace, so a mid-write disconnect never corrupts it) and
  mirrors it to `drive_checkpoint_dir` as `<run_name>_last.pt`. On the next `train()` call
  with the same `run_name`, it loads the last checkpoint (Drive first) and continues from the
  next epoch. Toggle with `TrainConfig.resume` (default True).
- Per-epoch deterministic seeding (`seed + epoch`) so a resumed run sees the same data order
  as an uninterrupted one — fixes the common "restore mid-batch RNG" bug where the loader
  reshuffles differently on resume.
- `best_model.pt` / Drive `<run_name>_best.pt` now written atomically too.

## v0.1.5 — Live progress + Drive storage fix

- **Live training progress:** per-batch `tqdm` bar (running pred/SIGReg loss) plus a printed
  per-epoch summary line, so you can watch progress and the LAST printed line tells you exactly
  which epoch a disconnect stopped at.
- **Persistent progress file:** a tiny `<run_name>_progress.json` is written to Drive every
  epoch (epoch_completed, total, percent, best loss, timestamp). Open it on Drive to see how
  far a run got without loading the big checkpoint.
- **Drive storage fix:** the large resume checkpoint (`<run_name>_last.pt`, ~250 MB) is now
  mirrored to Drive only every `drive_save_every` epochs (default 5) instead of every epoch —
  the per-epoch overwrites were piling up old file revisions and bloating your Drive quota.
  The full state still saves locally every epoch for in-session resume. (Tip: empty Drive
  Trash once to reclaim space already taken by old revisions.)

## v0.1.6 — Pluggable regularizers

- New `scmorphjepa.training.regularizers` with a registry of anti-collapse regularizers behind
  one `forward(z: (B, D)) -> scalar` interface: **sigreg** (isotropic-Gaussian, default),
  **vicreg** (variance + covariance), **koleo** (DINOv2 entropy/spread), **barlow** (single-view
  whitening), and **none** (for the ablation that measures a regularizer's actual contribution).
  Select via `build_regularizer(name, **kwargs)` or list with `available_regularizers()`.
- `TrainConfig` gains `regularizer` (default "sigreg"), `reg_weight` (falls back to the existing
  `sigreg_weight`), and `reg_kwargs`. The auto `run_name` appends the regularizer name for
  non-default choices (e.g. `scmorphjepa_n5000_e100_vicreg`) so ablation runs never collide.
- NOTE: regularizers live on very different numeric scales (sigreg ≈ 8, koleo ≈ 3, barlow ≈ 90
  on random embeddings), so `reg_weight` must be retuned per regularizer — don't reuse the
  SIGReg weight for the others.

## v0.1.7 — Clearer training logs

- Switched the progress bar from `tqdm.auto` to plain `tqdm`, which avoids the noisy
  `_MultiProcessingDataLoaderIter.__del__ ... can only test a child process` teardown messages
  that appear when a Colab run is interrupted. That message was always harmless (it fires during
  DataLoader worker cleanup, after weights/checkpoints are safely saved), but it cluttered the log.
- Best-model and resume-checkpoint saves now print the epoch they correspond to (and whether the
  checkpoint reached Drive), e.g. `✓ best model updated → epoch 37` — so it's easy to see what
  happened and where a run stopped.

## v0.1.8 — Resume no longer loses epochs after a runtime reset

- Fixed the cross-session resume gap. The full resume checkpoint is now mirrored to Drive
  every epoch by default (drive_save_every=1), so a Colab reset resumes from the last
  completed epoch instead of the last multiple-of-N epoch. Previously progress.json advanced
  every epoch while the Drive checkpoint lagged, so a reset could drop several completed epochs.
- progress.json now reports `recoverable_epoch`: where a fresh-runtime resume will actually
  start. With drive_save_every=1 it equals epoch_completed; if you raise drive_save_every to
  reduce Drive revisions, this field tells you exactly how far behind the recoverable point is.
- Resume log now states the source (Drive vs local) and the completed epoch it loaded.
