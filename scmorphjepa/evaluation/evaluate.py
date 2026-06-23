"""Evaluation: embedding extraction, k-NN classification, linear probing."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


def extract_embeddings(
    model, dataset: Dataset, batch_size: int = 64,
    device: str = "auto", num_workers: int = 0,
    return_patches: bool = False,
) -> dict[str, np.ndarray]:
    """Extract embeddings from a trained scMorphJEPA model.

    Args:
        model: Trained ScMorphJEPA (will be set to eval mode).
        dataset: Dataset yielding (image, label) or (image, label, path).
        batch_size: Batch size for inference.
        device: Compute device.
        return_patches: Whether to also return patch-level tokens.

    Returns:
        Dictionary with:
            cls_tokens: (N, D) CLS embeddings
            labels: (N,) integer labels
            paths: (N,) file paths (if dataset returns them)
            patch_tokens: (N, num_patches, D) [if return_patches=True]
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device).eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)

    cls_list, label_list, patch_list, path_list = [], [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting embeddings"):
            images = batch[0].to(device)
            labels = batch[1]

            out = model.extract_embeddings(images)
            cls_list.append(out["cls_token"].cpu().numpy())
            label_list.append(np.array(labels))

            if return_patches:
                patch_list.append(out["patch_tokens"].cpu().numpy())
            if len(batch) > 2:
                path_list.extend(batch[2])

    result = {
        "cls_tokens": np.concatenate(cls_list, axis=0),
        "labels": np.concatenate(label_list, axis=0),
    }
    if patch_list:
        result["patch_tokens"] = np.concatenate(patch_list, axis=0)
    if path_list:
        result["paths"] = np.array(path_list)

    logger.info(f"Extracted {len(result['cls_tokens'])} embeddings, shape {result['cls_tokens'].shape}")
    return result


def knn_evaluate(
    train_embeddings: np.ndarray, train_labels: np.ndarray,
    test_embeddings: np.ndarray, test_labels: np.ndarray,
    k_values: list[int] | None = None,
) -> dict[int, float]:
    """k-NN classification evaluation.

    Args:
        train_embeddings, train_labels: Training set.
        test_embeddings, test_labels: Test set.
        k_values: List of k values to evaluate.

    Returns:
        Dictionary mapping k → accuracy.
    """
    if k_values is None:
        k_values = [1, 5, 10, 20]

    # L2-normalize so Euclidean k-NN gives the same ranking as cosine, but uses the
    # fast exact backends (kd_tree/ball_tree) instead of sklearn's brute-force cosine
    # — important at 89K+ points. (We intentionally do not StandardScaler here, since
    # per-dimension scaling distorts the cosine geometry the model was trained under.)
    def _l2(x):
        return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)

    X_train = _l2(train_embeddings)
    X_test = _l2(test_embeddings)

    results = {}
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean", n_jobs=-1)
        knn.fit(X_train, train_labels)
        acc = knn.score(X_test, test_labels)
        results[k] = float(acc)
        logger.info(f"k-NN (k={k}): {acc:.4f}")

    return results


def linear_probe(
    train_embeddings: np.ndarray, train_labels: np.ndarray,
    test_embeddings: np.ndarray, test_labels: np.ndarray,
    max_iter: int = 1000, C: float = 1.0, standardize: bool = True,
) -> dict[str, float]:
    """Linear-probe evaluation: train a logistic-regression head on FROZEN embeddings.

    The backbone is never updated — only this linear classifier is fit on the train
    embeddings and scored on the test embeddings (same split as knn_evaluate). This is
    the standard SSL probe and often separates JEPA-family features better than k-NN,
    since it can use linear directions that nearest-neighbour voting cannot.

    Args:
        max_iter: LBFGS iterations for the logistic regression.
        C: Inverse L2 regularisation strength.
        standardize: Z-score features before the probe (recommended for a linear head).

    Returns:
        {"accuracy": top-1, "balanced_accuracy": class-balanced top-1}.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import balanced_accuracy_score

    Xtr, Xte = train_embeddings, test_embeddings
    if standardize:
        sc = StandardScaler().fit(Xtr)
        Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)

    clf = LogisticRegression(max_iter=max_iter, C=C, n_jobs=-1)
    clf.fit(Xtr, train_labels)
    pred = clf.predict(Xte)
    acc = float((pred == test_labels).mean())
    bacc = float(balanced_accuracy_score(test_labels, pred))
    logger.info(f"Linear probe: acc={acc:.4f}, balanced_acc={bacc:.4f}")
    return {"accuracy": acc, "balanced_accuracy": bacc}


def save_embeddings(embeddings: dict, path: str | Path) -> None:
    """Save embeddings to .npz file."""
    np.savez(str(path), **embeddings)
    logger.info(f"Saved embeddings to {path}")


def load_embeddings(path: str | Path) -> dict:
    """Load embeddings from .npz file."""
    data = np.load(str(path), allow_pickle=True)
    return {k: data[k] for k in data.files}
