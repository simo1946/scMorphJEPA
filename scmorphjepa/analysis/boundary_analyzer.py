"""Boundary Analyzer: compare neighboring clusters to understand what separates them.

Automates the manual cluster analysis from scDINO's Science 2024 paper.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)


@dataclass
class BoundaryResult:
    cluster_a: int
    cluster_b: int
    n_cells_a: int
    n_cells_b: int
    discriminative_features: pd.DataFrame
    boundary_cell_indices: np.ndarray


def identify_neighboring_clusters(
    embeddings: np.ndarray, cluster_labels: np.ndarray,
    n_neighbors: int = 30, min_overlap: float = 0.05,
) -> list[tuple[int, int]]:
    """Find cluster pairs that are neighbors in embedding space."""
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="cosine")
    nn.fit(embeddings)
    # First column is the point itself (query == fit set); drop it.
    nbrs = nn.kneighbors(return_distance=False)[:, 1:]

    clusters = sorted(c for c in set(cluster_labels) if c >= 0)
    pairs = []

    for i, a in enumerate(clusters):
        for b in clusters[i + 1:]:
            mask_a, mask_b = cluster_labels == a, cluster_labels == b
            idx_a, idx_b = np.where(mask_a)[0], np.where(mask_b)[0]

            a_in_b = np.isin(nbrs[idx_a], idx_b).mean()
            b_in_a = np.isin(nbrs[idx_b], idx_a).mean()

            if (a_in_b + b_in_a) / 2 >= min_overlap:
                pairs.append((a, b))

    logger.info(f"Found {len(pairs)} neighboring cluster pairs")
    return pairs


def compare_clusters(
    cluster_a: int, cluster_b: int,
    features_df: pd.DataFrame, cluster_labels: np.ndarray,
    embeddings: np.ndarray | None = None, alpha: float = 0.05,
) -> BoundaryResult:
    """Compare two clusters: Mann-Whitney U + Cohen's d per feature, BH FDR correction."""
    ma, mb = cluster_labels == cluster_a, cluster_labels == cluster_b
    feat_cols = [c for c in features_df.columns if c[0] != "metadata"]

    rows = []
    for col in feat_cols:
        va, vb = features_df.loc[ma, col].values, features_df.loc[mb, col].values
        try:
            _, p = sp_stats.mannwhitneyu(va, vb, alternative="two-sided")
        except ValueError:
            p = 1.0

        pooled = np.sqrt(((len(va)-1)*np.var(va, ddof=1) + (len(vb)-1)*np.var(vb, ddof=1)) / max(len(va)+len(vb)-2, 1))
        d = (np.mean(va) - np.mean(vb)) / pooled if pooled > 1e-10 else 0.0

        rows.append({
            "category": col[0] if isinstance(col, tuple) else "",
            "channel": col[1] if isinstance(col, tuple) else "",
            "feature": col[2] if isinstance(col, tuple) else str(col),
            "mean_a": float(np.mean(va)), "mean_b": float(np.mean(vb)),
            "cohens_d": float(d), "abs_cohens_d": float(abs(d)), "p_value": float(p),
        })

    df = pd.DataFrame(rows).sort_values("p_value")
    n = len(df)
    if n > 0:
        df["rank"] = range(1, n + 1)
        df["bh_threshold"] = df["rank"] * alpha / n
        df["significant"] = df["p_value"] <= df["bh_threshold"]
        df = df.sort_values("abs_cohens_d", ascending=False)

    # Boundary cells
    boundary = np.array([], dtype=int)
    if embeddings is not None:
        from sklearn.neighbors import NearestNeighbors
        all_idx = np.where(ma | mb)[0]
        nn = NearestNeighbors(n_neighbors=10, metric="cosine")
        nn.fit(embeddings[all_idx])
        nb_idx = nn.kneighbors(return_distance=False)
        sub_labels = cluster_labels[all_idx]
        mixed = np.array([len(set(sub_labels[nb_idx[i]])) > 1 for i in range(len(all_idx))])
        boundary = all_idx[mixed]

    return BoundaryResult(cluster_a, cluster_b, int(ma.sum()), int(mb.sum()), df, boundary)


def compare_all_neighbors(
    features_df: pd.DataFrame, cluster_labels: np.ndarray, embeddings: np.ndarray,
    n_neighbors: int = 30, min_overlap: float = 0.05,
) -> list[BoundaryResult]:
    """Compare all neighboring cluster pairs."""
    pairs = identify_neighboring_clusters(embeddings, cluster_labels, n_neighbors, min_overlap)
    return [compare_clusters(a, b, features_df, cluster_labels, embeddings) for a, b in pairs]
