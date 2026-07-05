"""CellProfiler-lite: interpretable feature extraction from cell images.

Extracts per-channel intensity, morphology, and texture features.
Works with any multi-channel microscopy image, not dataset-specific.
Output: DataFrame with MultiIndex columns (category, channel, feature).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    channel_names: list[str] | None = None
    compute_intensity: bool = True
    compute_morphology: bool = True
    compute_texture: bool = True
    glcm_distances: list[int] = field(default_factory=lambda: [1, 3])
    glcm_angles: list[float] = field(default_factory=lambda: [0, 0.785, 1.571, 2.356])
    percentiles: list[int] = field(default_factory=lambda: [5, 25, 50, 75, 95])


def _intensity_features(ch: np.ndarray, ch_name: str) -> dict:
    """Per-channel intensity statistics."""
    px = ch.ravel()
    feat = {}
    feat[("intensity", ch_name, "mean")] = float(np.mean(px))
    feat[("intensity", ch_name, "std")] = float(np.std(px))
    feat[("intensity", ch_name, "max")] = float(np.max(px))
    feat[("intensity", ch_name, "min")] = float(np.min(px))
    if np.std(px) > 1e-8:
        feat[("intensity", ch_name, "skewness")] = float(sp_stats.skew(px))
        feat[("intensity", ch_name, "kurtosis")] = float(sp_stats.kurtosis(px))
    else:
        feat[("intensity", ch_name, "skewness")] = 0.0
        feat[("intensity", ch_name, "kurtosis")] = 0.0
    for p in [5, 25, 50, 75, 95]:
        feat[("intensity", ch_name, f"p{p}")] = float(np.percentile(px, p))
    mu = np.mean(px)
    feat[("intensity", ch_name, "cv")] = float(np.std(px) / mu) if mu > 1e-8 else 0.0
    return feat


def _morphology_features(image: np.ndarray) -> dict:
    """Shape features from max-projection cell mask."""
    try:
        from skimage.filters import threshold_otsu
        from skimage.measure import regionprops, label
    except ImportError:
        return {}

    feat = {}
    max_proj = np.max(image, axis=0)
    defaults = {"area": 0, "eccentricity": 0, "solidity": 0, "extent": 0,
                "perimeter": 0, "major_axis": 0, "minor_axis": 0, "aspect_ratio": 1}

    if max_proj.max() - max_proj.min() < 1e-8:
        for k, v in defaults.items():
            feat[("morphology", "cell", k)] = float(v)
        return feat

    thresh = threshold_otsu(max_proj)
    labeled = label(max_proj > thresh)
    props = regionprops(labeled)
    if not props:
        for k, v in defaults.items():
            feat[("morphology", "cell", k)] = float(v)
        return feat

    p = max(props, key=lambda x: x.area)
    feat[("morphology", "cell", "area")] = float(p.area)
    feat[("morphology", "cell", "eccentricity")] = float(p.eccentricity)
    feat[("morphology", "cell", "solidity")] = float(p.solidity)
    feat[("morphology", "cell", "extent")] = float(p.extent)
    feat[("morphology", "cell", "perimeter")] = float(p.perimeter)
    major = float(p.axis_major_length)
    minor = float(p.axis_minor_length)
    feat[("morphology", "cell", "major_axis")] = major
    feat[("morphology", "cell", "minor_axis")] = minor
    feat[("morphology", "cell", "aspect_ratio")] = (minor / major) if major > 1e-8 else 1.0
    return feat


def _texture_features(ch: np.ndarray, ch_name: str, dists, angles) -> dict:
    """GLCM texture features per channel."""
    try:
        from skimage.feature import graycomatrix, graycoprops
    except ImportError:
        return {}

    feat = {}
    ch_q = (np.clip(ch, 0, 1) * 63).astype(np.uint8)
    try:
        glcm = graycomatrix(ch_q, distances=dists, angles=angles,
                            levels=64, symmetric=True, normed=True)
        for prop in ["contrast", "correlation", "homogeneity", "energy"]:
            vals = graycoprops(glcm, prop)
            feat[("texture", ch_name, f"{prop}_mean")] = float(vals.mean())
            feat[("texture", ch_name, f"{prop}_std")] = float(vals.std())
    except Exception:
        for prop in ["contrast", "correlation", "homogeneity", "energy"]:
            feat[("texture", ch_name, f"{prop}_mean")] = float("nan")
            feat[("texture", ch_name, f"{prop}_std")] = float("nan")
    return feat


def extract_cell_features(image: np.ndarray, config: FeatureConfig | None = None) -> dict:
    """Extract all features from one cell image (C, H, W), float32 [0,1]."""
    if config is None:
        config = FeatureConfig()
    n_ch = image.shape[0]
    ch_names = config.channel_names or [f"ch{i}" for i in range(n_ch)]

    feat = {}
    if config.compute_intensity:
        for c, name in enumerate(ch_names):
            feat.update(_intensity_features(image[c], name))
    if config.compute_morphology:
        feat.update(_morphology_features(image))
    if config.compute_texture:
        for c, name in enumerate(ch_names):
            feat.update(_texture_features(image[c], name, config.glcm_distances, config.glcm_angles))
    return feat


def extract_dataset_features(
    images: np.ndarray | list, labels: np.ndarray | None = None,
    config: FeatureConfig | None = None, show_progress: bool = True,
) -> pd.DataFrame:
    """Extract features from a batch of cell images.

    Args:
        images: (N, C, H, W) array or list of (C, H, W) arrays.
        labels: Optional (N,) labels.
        config: Feature configuration.

    Returns:
        DataFrame with MultiIndex columns, one row per cell.
    """
    if config is None:
        config = FeatureConfig()
    n = len(images)
    records = []
    it = tqdm(range(n), desc="Extracting features") if show_progress else range(n)
    for i in it:
        records.append(extract_cell_features(images[i], config))

    if not records:
        return pd.DataFrame()

    keys = sorted(records[0].keys())
    columns = pd.MultiIndex.from_tuples(keys, names=["category", "channel", "feature"])
    data = np.array([[r.get(k, 0.0) for k in keys] for r in records])
    df = pd.DataFrame(data, columns=columns)

    if labels is not None:
        df[("metadata", "cell", "label")] = labels

    logger.info(f"Extracted {len(keys)} features from {n} cells")
    return df


def profile_clusters_by_features(
    dataset,
    cluster_labels,
    indices=None,
    cell_type_labels=None,
    cell_type_names=None,
    config: "FeatureConfig | None" = None,
    top_state_features: int = 8,
    show_progress: bool = True,
) -> dict:
    """Link clusters to interpretable raw-image features to probe cell state.

    For each cell it extracts CellProfiler-lite features (intensity / morphology / texture,
    via ``extract_dataset_features``) from its raw image, then summarizes them per cluster and
    per (cell_type, cluster). The intended question: when one cell type spreads across several
    clusters, do those sub-clusters differ in size, DNA content (integrated DAPI ~ cell cycle),
    or marker intensity (~ activation), i.e. are they distinct cell states rather than noise?

    Intensity features are only meaningful if the dataset preserves intensity: build it with
    ``normalize="none"`` (raw) or ``normalize="per_image"`` for this analysis.

    Args:
        dataset: a MicroscopyDataset exposing ``get_raw_image(i)``; ``.labels`` and ``.cell_types``
            are used to fill cell types if not given.
        cluster_labels: (N,) cluster id per cell.
        indices: (N,) dataset indices the labels correspond to (default ``range(N)``).
        cell_type_labels: (N,) ground-truth type id per cell (default: read from the dataset).
        cell_type_names: list mapping type id -> name (default: ``dataset.cell_types``).
        config: FeatureConfig; set ``channel_names`` so features are readable.
        top_state_features: how many candidate state features to surface per cell type.

    Returns dict with:
        per_cell:        DataFrame, one row/cell: cell_index, cell_type, cluster, + features.
        per_cluster:     DataFrame, one row/cluster: size, dominant_type, mean of each feature.
        by_type_cluster: DataFrame grouped by (cell_type, cluster): size + mean of each feature.
        state_signal:    DataFrame ranking, per cell type, the features that vary most across
                         that type's clusters (candidate cell-state axes).
    """
    import numpy as np
    import pandas as pd

    cluster_labels = np.asarray(cluster_labels)
    n = len(cluster_labels)
    if indices is None:
        indices = np.arange(n)
    indices = np.asarray(indices)
    if len(indices) != n:
        raise ValueError(f"indices ({len(indices)}) and cluster_labels ({n}) length mismatch")

    if cell_type_names is None:
        cell_type_names = list(getattr(dataset, "cell_types", []))
    if cell_type_labels is None:
        ds_labels = np.asarray(getattr(dataset, "labels", []))
        cell_type_labels = ds_labels[indices] if len(ds_labels) else np.full(n, -1)
    cell_type_labels = np.asarray(cell_type_labels)

    def type_name(t):
        t = int(t)
        return cell_type_names[t] if 0 <= t < len(cell_type_names) else str(t)

    images = [dataset.get_raw_image(int(i)) for i in indices]
    feats = extract_dataset_features(images, config=config, show_progress=show_progress)
    # Flatten the MultiIndex feature columns to readable strings.
    feats = feats.copy()
    feats.columns = ["/".join(str(x) for x in col) for col in feats.columns]
    feature_cols = list(feats.columns)

    per_cell = feats
    per_cell.insert(0, "cell_index", indices)
    per_cell.insert(1, "cell_type", [type_name(t) for t in cell_type_labels])
    per_cell.insert(2, "cluster", cluster_labels)

    def _dominant(sub):
        return sub["cell_type"].value_counts().idxmax() if len(sub) else None

    # Per-cluster summary
    rows = []
    for cl, sub in per_cell.groupby("cluster"):
        row = {"cluster": cl, "size": len(sub), "dominant_type": _dominant(sub)}
        row.update(sub[feature_cols].mean().to_dict())
        rows.append(row)
    per_cluster = pd.DataFrame(rows).set_index("cluster").sort_index()

    # Per (cell_type, cluster) summary
    by_type_cluster = (
        per_cell.groupby(["cell_type", "cluster"])[feature_cols].mean()
    )
    by_type_cluster.insert(0, "size", per_cell.groupby(["cell_type", "cluster"]).size())

    # State signal: within each cell type, how much does each feature vary across its clusters?
    # High spread of per-cluster means => that feature separates the type's sub-populations.
    sig_rows = []
    for ct, sub in per_cell.groupby("cell_type"):
        if sub["cluster"].nunique() < 2:
            continue
        cluster_means = sub.groupby("cluster")[feature_cols].mean()
        # normalize each feature by its overall std so features are comparable
        overall_std = sub[feature_cols].std().replace(0, np.nan)
        spread = (cluster_means.max() - cluster_means.min()) / overall_std
        spread = spread.dropna().sort_values(ascending=False)
        for feat, val in spread.head(top_state_features).items():
            sig_rows.append({"cell_type": ct, "feature": feat,
                             "across_cluster_spread": round(float(val), 3),
                             "n_clusters": int(sub["cluster"].nunique())})
    state_signal = pd.DataFrame(sig_rows)

    return {"per_cell": per_cell, "per_cluster": per_cluster,
            "by_type_cluster": by_type_cluster, "state_signal": state_signal}
