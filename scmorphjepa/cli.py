"""CLI entry points for scMorphJEPA.

Usage:
    python -m scmorphjepa.cli train --config configs/severin.yaml
    python -m scmorphjepa.cli evaluate --model output/best_model.pt --data_dir path/to/test
    python -m scmorphjepa.cli analyze --embeddings output/embeddings.npz
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(message)s")
logger = logging.getLogger("scmorphjepa")


def train():
    """Train scMorphJEPA from command line."""
    parser = argparse.ArgumentParser(description="Train scMorphJEPA")
    parser.add_argument("--data_dir", required=True, help="Path to dataset root (with Training/ and Test/)")
    parser.add_argument("--checkpoint", default=None, help="DINO pretrained checkpoint path")
    parser.add_argument("--output_dir", default="output", help="Output directory")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n_images", type=int, default=0, help="0 = use all")
    parser.add_argument("--in_channels", type=int, default=5)
    parser.add_argument("--sigreg_weight", type=float, default=10.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dataset_type", default="severin", choices=["severin", "folder"])
    args = parser.parse_args()

    from scmorphjepa.models.builder import build_scmorphjepa
    from scmorphjepa.models.cell_jepa import ScMorphJEPAConfig
    from scmorphjepa.training.trainer import Trainer, TrainConfig
    from scmorphjepa.data.datasets import SeverinDataset, FolderMicroscopyDataset

    data_root = Path(args.data_dir)
    DatasetClass = SeverinDataset if args.dataset_type == "severin" else FolderMicroscopyDataset

    train_ds = DatasetClass(data_root / "Training")
    test_ds = DatasetClass(data_root / "Test") if (data_root / "Test").exists() else None

    model = build_scmorphjepa(
        checkpoint_path=args.checkpoint,
        config=ScMorphJEPAConfig(in_channels=args.in_channels),
    )

    trainer = Trainer(
        model, train_ds, test_ds,
        config=TrainConfig(
            batch_size=args.batch_size, epochs=args.epochs, lr=args.lr,
            n_images=args.n_images, sigreg_weight=args.sigreg_weight,
            output_dir=args.output_dir, device=args.device,
        ),
    )
    trainer.train()


def evaluate():
    """Evaluate a trained model with k-NN."""
    parser = argparse.ArgumentParser(description="Evaluate scMorphJEPA")
    parser.add_argument("--model", required=True, help="Trained model .pt path")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--in_channels", type=int, default=5)
    parser.add_argument("--output", default="embeddings.npz")
    parser.add_argument("--dataset_type", default="severin", choices=["severin", "folder"])
    args = parser.parse_args()

    from scmorphjepa.models.builder import load_trained_model
    from scmorphjepa.models.cell_jepa import ScMorphJEPAConfig
    from scmorphjepa.evaluation.evaluate import extract_embeddings, knn_evaluate, save_embeddings
    from scmorphjepa.data.datasets import SeverinDataset, FolderMicroscopyDataset

    data_root = Path(args.data_dir)
    DatasetClass = SeverinDataset if args.dataset_type == "severin" else FolderMicroscopyDataset

    model = load_trained_model(
        args.model, args.checkpoint, ScMorphJEPAConfig(in_channels=args.in_channels)
    )

    train_ds = DatasetClass(data_root / "Training")
    test_ds = DatasetClass(data_root / "Test")

    train_emb = extract_embeddings(model, train_ds)
    test_emb = extract_embeddings(model, test_ds)

    results = knn_evaluate(
        train_emb["cls_tokens"], train_emb["labels"],
        test_emb["cls_tokens"], test_emb["labels"],
    )

    print("\n=== k-NN Results ===")
    for k, acc in results.items():
        print(f"  k={k}: {acc:.4f} ({acc*100:.1f}%)")

    save_embeddings({**test_emb, "knn_results": np.array(list(results.values()))}, args.output)


def analyze():
    """Run cluster analysis on extracted embeddings."""
    parser = argparse.ArgumentParser(description="Analyze scMorphJEPA embeddings")
    parser.add_argument("--embeddings", required=True, help="Path to embeddings .npz")
    parser.add_argument("--method", default="leiden", choices=["leiden", "hdbscan"])
    parser.add_argument("--resolution", type=float, default=1.0)
    args = parser.parse_args()

    from scmorphjepa.evaluation.evaluate import load_embeddings
    from scmorphjepa.analysis.cluster_profiler import run_full_profiling, ClusterConfig

    emb = load_embeddings(args.embeddings)

    results = run_full_profiling(
        embeddings=emb["cls_tokens"],
        cell_type_labels=emb.get("labels"),
        cluster_config=ClusterConfig(method=args.method, leiden_resolution=args.resolution),
    )

    cr = results["cluster_result"]
    print("\n=== Cluster Analysis ===")
    print(f"Method: {cr.method}, Clusters: {cr.n_clusters}")
    print(f"Quality: {cr.quality_metrics}")
    print(f"Sizes: {cr.cluster_sizes}")

    if "composition" in results:
        print("\n=== Composition ===")
        print(results["composition"].to_string())


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m scmorphjepa.cli {train|evaluate|analyze}")
        sys.exit(1)

    cmd = sys.argv.pop(1)
    {"train": train, "evaluate": evaluate, "analyze": analyze}[cmd]()
