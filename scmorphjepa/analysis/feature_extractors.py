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
            feat[("texture", ch_name, f"{prop}_mean")] = 0.0
            feat[("texture", ch_name, f"{prop}_std")] = 0.0
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
