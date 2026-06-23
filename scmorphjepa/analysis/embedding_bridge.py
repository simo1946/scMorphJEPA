"""Embedding Bridge: map SSL embeddings ↔ interpretable CellProfiler features.

Answers: what biological features does each embedding dimension encode?
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger(__name__)


def build_embedding_bridge(
    embeddings: np.ndarray, features_df: pd.DataFrame, n_top: int = 20,
) -> dict:
    """Ridge regression from CLS tokens → each CellProfiler feature.

    Returns dict with r2_per_feature (Series), top_captured, least_captured (DataFrames).
    """
    feat_cols = [c for c in features_df.columns if c[0] != "metadata"]
    scaler = StandardScaler()
    X = scaler.fit_transform(embeddings)

    r2 = {}
    for col in feat_cols:
        y = features_df[col].values.astype(np.float64)
        if np.std(y) < 1e-10:
            r2[col] = 0.0
            continue
        y_s = StandardScaler().fit_transform(y.reshape(-1, 1)).ravel()
        try:
            scores = cross_val_score(
                RidgeCV(alphas=[0.01, 0.1, 1, 10, 100]), X, y_s, cv=5, scoring="r2"
            )
            r2[col] = float(np.mean(scores))
        except Exception:
            r2[col] = 0.0

    series = pd.Series(r2).sort_values(ascending=False)
    if isinstance(series.index, pd.MultiIndex):
        series.index = series.index.set_names(["category", "channel", "feature"])
    logger.info(
        f"Bridge: mean R²={series.mean():.3f}, "
        f"{(series > 0.5).sum()} features R²>0.5, "
        f"{(series < 0.1).sum()} features R²<0.1"
    )

    def _fmt(s):
        df = s.reset_index()
        df.columns = list(df.columns[:-1]) + ["r2"]
        return df

    return {
        "r2_per_feature": series,
        "top_captured": _fmt(series.head(n_top)),
        "least_captured": _fmt(series.tail(n_top)),
    }


def train_embedding_classifier(
    embeddings: np.ndarray, labels: np.ndarray,
    n_estimators: int = 100, compute_shap: bool = False,
) -> dict:
    """RF classifier on CLS tokens → cell type. Returns accuracy + importances."""
    X = StandardScaler().fit_transform(embeddings)
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    scores = cross_val_score(rf, X, labels, cv=5, scoring="accuracy")

    rf.fit(X, labels)
    result = {
        "accuracy": float(np.mean(scores)),
        "accuracy_std": float(np.std(scores)),
        "feature_importances": pd.Series(rf.feature_importances_),
    }
    logger.info(f"Classifier: {result['accuracy']:.3f}±{result['accuracy_std']:.3f}")

    if compute_shap:
        try:
            import shap
            result["shap_values"] = shap.TreeExplainer(rf).shap_values(X)
        except ImportError:
            logger.warning("shap not installed")

    return result
