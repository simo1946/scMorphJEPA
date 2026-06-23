"""Cluster Profiler: clustering, fingerprinting, composition analysis of SSL embeddings.

Main entry point: run_full_profiling() — takes embeddings and returns complete analysis.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class ClusterConfig:
    method: Literal["leiden", "hdbscan"] = "leiden"
    n_neighbors: int = 30
    leiden_resolution: float = 1.0
    hdbscan_min_cluster_size: int = 50
    hdbscan_min_samples: int = 10
    umap_n_components: int = 2
    umap_min_dist: float = 0.3
    umap_metric: str = "cosine"
    random_state: int = 42


@dataclass
class ClusterResult:
    labels: np.ndarray
    umap_coords: np.ndarray
    n_clusters: int
    method: str
    quality_metrics: dict = field(default_factory=dict)
    cluster_sizes: dict = field(default_factory=dict)


def cluster_embeddings(embeddings: np.ndarray, config: ClusterConfig | None = None) -> ClusterResult:
    """Cluster SSL embeddings and compute UMAP projection."""
    if config is None:
        config = ClusterConfig()
    import umap as umap_lib

    scaler = StandardScaler()
    emb = scaler.fit_transform(embeddings)

    reducer = umap_lib.UMAP(
        n_components=config.umap_n_components, n_neighbors=config.n_neighbors,
        min_dist=config.umap_min_dist, metric=config.umap_metric,
        random_state=config.random_state,
    )
    umap_coords = reducer.fit_transform(emb)

    if config.method == "leiden":
        labels = _leiden(emb, config)
    elif config.method == "hdbscan":
        import hdbscan
        labels = hdbscan.HDBSCAN(
            min_cluster_size=config.hdbscan_min_cluster_size,
            min_samples=config.hdbscan_min_samples,
        ).fit_predict(emb)
    else:
        raise ValueError(f"Unknown method: {config.method}")

    n_cl = len(set(labels)) - (1 if -1 in labels else 0)
    quality = {}
    valid = labels >= 0
    if n_cl >= 2 and valid.sum() > n_cl:
        quality["silhouette"] = float(silhouette_score(emb[valid], labels[valid], metric="cosine"))
        quality["calinski_harabasz"] = float(calinski_harabasz_score(emb[valid], labels[valid]))

    sizes = dict(zip(*np.unique(labels, return_counts=True)))
    logger.info(f"Found {n_cl} clusters (silhouette={quality.get('silhouette', 'N/A')})")

    return ClusterResult(
        labels=labels, umap_coords=umap_coords, n_clusters=n_cl,
        method=config.method, quality_metrics=quality,
        cluster_sizes={int(k): int(v) for k, v in sizes.items()},
    )


def _leiden(emb: np.ndarray, config: ClusterConfig) -> np.ndarray:
    import igraph as ig
    import leidenalg
    from sklearn.neighbors import kneighbors_graph

    G = kneighbors_graph(emb, n_neighbors=config.n_neighbors, mode="connectivity", metric="cosine")
    G = G + G.T
    G[G > 1] = 1
    src, tgt = G.nonzero()
    g = ig.Graph(n=len(emb), edges=list(zip(src.tolist(), tgt.tolist())), directed=False)
    g.simplify()
    part = leidenalg.find_partition(
        g, leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=config.leiden_resolution, seed=config.random_state,
    )
    return np.array(part.membership)


def compute_cluster_composition(
    cluster_labels: np.ndarray, cell_type_labels: np.ndarray,
    cell_type_names: list[str] | None = None,
) -> pd.DataFrame:
    """Cell type composition per cluster."""
    u_cl = sorted(set(cluster_labels))
    u_ct = sorted(set(cell_type_labels))
    names = cell_type_names or [str(t) for t in u_ct]

    comp = np.zeros((len(u_cl), len(u_ct)))
    for i, cl in enumerate(u_cl):
        m = cluster_labels == cl
        for j, ct in enumerate(u_ct):
            comp[i, j] = (cell_type_labels[m] == ct).sum()

    totals = comp.sum(axis=1, keepdims=True)
    props = np.divide(comp, totals, out=np.zeros_like(comp), where=totals > 0)

    df = pd.DataFrame(props, index=pd.Index(u_cl, name="cluster"), columns=names)
    df["total_cells"] = comp.sum(axis=1).astype(int)
    df["purity"] = props.max(axis=1)
    df["dominant_type"] = [names[int(np.argmax(r))] for r in props]
    return df


def clustering_metrics(
    cluster_labels: np.ndarray, true_labels: np.ndarray,
) -> dict[str, float]:
    """Global, chance-adjusted agreement between clusters and ground-truth cell types.

    Complements the per-cluster purity table with single-number summaries reviewers
    expect for "does unsupervised clustering recover cell identity":

        - ari:            Adjusted Rand Index (chance-adjusted; 0 ≈ random, 1 = perfect)
        - nmi:            Normalized Mutual Information (0–1)
        - overall_purity: cell-weighted mean of per-cluster purity (fraction of all cells
                          assigned to their cluster's dominant type)

    Noise points (label < 0, e.g. HDBSCAN) are excluded from all three.
    """
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    cl = np.asarray(cluster_labels)
    tl = np.asarray(true_labels)
    keep = cl >= 0
    cl, tl = cl[keep], tl[keep]
    if len(cl) == 0:
        return {"ari": 0.0, "nmi": 0.0, "overall_purity": 0.0}

    # overall (cell-weighted) purity
    correct = 0
    for c in np.unique(cl):
        m = cl == c
        vals, counts = np.unique(tl[m], return_counts=True)
        correct += counts.max()
    overall_purity = correct / len(cl)

    return {
        "ari": float(adjusted_rand_score(tl, cl)),
        "nmi": float(normalized_mutual_info_score(tl, cl)),
        "overall_purity": float(overall_purity),
    }


def compute_cluster_fingerprints(features_df: pd.DataFrame, cluster_labels: np.ndarray) -> pd.DataFrame:
    """Per-cluster mean of all CellProfiler-lite features."""
    feat_cols = [c for c in features_df.columns if c[0] != "metadata"]
    df = features_df.copy()
    df["cluster"] = cluster_labels
    return df.groupby("cluster")[feat_cols].agg(["mean", "std"])


def select_representative_cells(
    embeddings: np.ndarray, cluster_labels: np.ndarray, n_per_cluster: int = 9,
) -> dict[int, np.ndarray]:
    """Select cells closest to each cluster centroid."""
    reps = {}
    for cl in sorted(set(cluster_labels)):
        if cl < 0:
            continue
        mask = cluster_labels == cl
        idx = np.where(mask)[0]
        emb = embeddings[mask]
        centroid = emb.mean(axis=0, keepdims=True)
        dists = cdist(centroid, emb, metric="cosine")[0]
        n = min(n_per_cluster, len(idx))
        reps[cl] = idx[np.argsort(dists)[:n]]
    return reps


def run_full_profiling(
    embeddings: np.ndarray, images: np.ndarray | None = None,
    cell_type_labels: np.ndarray | None = None,
    cell_type_names: list[str] | None = None,
    channel_names: list[str] | None = None,
    cluster_config: ClusterConfig | None = None,
) -> dict:
    """Run complete cluster profiling pipeline — main entry point.

    Args:
        embeddings: (N, D) CLS embeddings from extract_embeddings().
        images: Optional (N, C, H, W) RAW cell images for CellProfiler-lite features.
            These MUST be the original-resolution images (e.g. dataset.get_raw_image(i)),
            NOT the 224×224 model input — morphology/texture computed on bilinearly
            upsampled pixels is invalid. Channel order must match `channel_names`.
        cell_type_labels: Optional (N,) ground-truth labels.
        cell_type_names: Names for the label integers.
        channel_names: Names for the image channels (for feature column labels).
        cluster_config: Clustering configuration.
    """
    results = {}

    # 1. Cluster
    cr = cluster_embeddings(embeddings, cluster_config)
    results["cluster_result"] = cr

    # 2. Composition
    if cell_type_labels is not None:
        results["composition"] = compute_cluster_composition(
            cr.labels, cell_type_labels, cell_type_names
        )
        results["clustering_metrics"] = clustering_metrics(cr.labels, cell_type_labels)

    # 3. Features + fingerprints (on raw images)
    if images is not None:
        from scmorphjepa.analysis.feature_extractors import extract_dataset_features, FeatureConfig
        if len(images) != len(embeddings):
            raise ValueError(
                f"images ({len(images)}) and embeddings ({len(embeddings)}) length mismatch"
            )
        cfg = FeatureConfig(channel_names=channel_names)
        features_df = extract_dataset_features(images, cell_type_labels, cfg)
        results["features_df"] = features_df
        results["fingerprints"] = compute_cluster_fingerprints(features_df, cr.labels)

    # 4. Representatives
    results["representatives"] = select_representative_cells(embeddings, cr.labels)
    return results
