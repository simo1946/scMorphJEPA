"""Dataset loaders for multi-channel fluorescence microscopy.

Designed to work with any single-cell crop dataset, not just Severin.
New datasets subclass MicroscopyDataset and implement _load_image().
"""

from __future__ import annotations

import glob
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import tifffile

logger = logging.getLogger(__name__)

# Valid normalization modes. "none" = raw pixel values (no scaling); use it when absolute
# intensity matters (e.g. DNA content / marker intensity for cell-state analysis) or to train/
# evaluate on raw images. Passing anything outside this set raises, so a typo can never silently
# fall through to returning unnormalized images.
VALID_NORMALIZE = ("none", "per_image", "per_channel", "per_channel_percentile")


class MicroscopyDataset(Dataset, ABC):
    """Abstract base for single-cell fluorescence microscopy datasets.

    Subclasses must implement:
        _discover_files(): populate self.files, self.labels, self.cell_types
        _load_image(path): return raw image as (C, H, W) float32 [0, 1]
    """

    def __init__(
        self, root_dir: str | Path, image_size: int = 224,
        channel_names: Sequence[str] | None = None,
        normalize: str = "per_channel",
        return_path: bool = False,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.channel_names = list(channel_names) if channel_names else None
        if normalize not in VALID_NORMALIZE:
            raise ValueError(
                f"normalize={normalize!r} is not valid; expected one of {VALID_NORMALIZE}. "
                "Training and evaluation must use the same value."
            )
        self.normalize = normalize
        self.return_path = return_path

        self.files: list[str] = []
        self.labels: list[int] = []
        self.cell_types: list[str] = []
        self.label_map: dict[str, int] = {}

        self._discover_files()

        if not self.files:
            raise FileNotFoundError(f"No images found in {self.root_dir}")

        logger.info(
            f"{self.__class__.__name__}: {len(self.files)} images, "
            f"{len(self.cell_types)} classes: {self.cell_types}"
        )

    @abstractmethod
    def _discover_files(self) -> None:
        """Populate self.files, self.labels, self.cell_types, self.label_map."""

    @abstractmethod
    def _load_image(self, path: str) -> np.ndarray:
        """Load and return raw image as (C, H, W) float32."""

    def __len__(self) -> int:
        return len(self.files)

    def _normalize(self, img: np.ndarray) -> np.ndarray:
        if self.normalize == "none":
            return img  # raw pixel values, no scaling
        if self.normalize == "per_image":
            mx = img.max()
            return img / mx if mx > 0 else img
        elif self.normalize == "per_channel":
            for c in range(img.shape[0]):
                mx = img[c].max()
                if mx > 0:
                    img[c] /= mx
            return img
        elif self.normalize == "per_channel_percentile":
            # Robust per-channel scaling to the 1st–99th percentile, clipped to [0, 1].
            # Less sensitive to hot pixels than max-normalization (cf. scDINO).
            for c in range(img.shape[0]):
                lo, hi = np.percentile(img[c], [1, 99])
                if hi > lo:
                    img[c] = np.clip((img[c] - lo) / (hi - lo), 0.0, 1.0)
                else:
                    img[c] = np.zeros_like(img[c])
            return img
        # Unreachable when constructed normally (__init__ validates), kept as a guard so an
        # unexpected value can never silently return raw pixels.
        raise ValueError(f"Unknown normalize={self.normalize!r}; expected one of {VALID_NORMALIZE}.")

    def _resize(self, img: np.ndarray) -> np.ndarray:
        """Resize each channel independently to image_size."""
        channels = []
        for c in range(img.shape[0]):
            ch = Image.fromarray(img[c])
            ch = ch.resize((self.image_size, self.image_size), Image.BILINEAR)
            channels.append(np.array(ch))
        return np.stack(channels, axis=0)

    def __getitem__(self, idx: int):
        img = self._load_image(self.files[idx])
        img = self._normalize(img)
        img = self._resize(img)
        tensor = torch.from_numpy(img)
        if self.return_path:
            return tensor, self.labels[idx], self.files[idx]
        return tensor, self.labels[idx]

    def get_raw_image(self, idx: int) -> np.ndarray:
        """Get normalized but NOT resized image (for feature extraction on originals)."""
        img = self._load_image(self.files[idx])
        return self._normalize(img)

    @property
    def n_channels(self) -> int:
        img = self._load_image(self.files[0])
        return img.shape[0]


# ── Severin PBMC Dataset ──────────────────────────────────────────────────

SEVERIN_CHANNELS = ["AF647", "BF", "DAPI", "FITC", "PE"]


class SeverinDataset(MicroscopyDataset):
    """Severin Deep Phenotyping PBMC dataset (ETH Zurich).

    113,564 five-channel uint16 TIFF images, 50×50 pixels, 8 immune cell types.
    Reference: Severin et al., DOI 10.3929/ethz-b-000343106
    """

    def __init__(self, root_dir, image_size=224, **kwargs):
        kwargs.setdefault("channel_names", SEVERIN_CHANNELS)
        super().__init__(root_dir, image_size, **kwargs)

    def _discover_files(self) -> None:
        self.files = sorted(glob.glob(str(self.root_dir / "**" / "*.tiff"), recursive=True))
        self.cell_types = sorted(set(Path(f).parent.name for f in self.files))
        self.label_map = {ct: i for i, ct in enumerate(self.cell_types)}
        self.labels = [self.label_map[Path(f).parent.name] for f in self.files]

    def _load_image(self, path: str) -> np.ndarray:
        img = tifffile.imread(path)  # (50, 50, 5) uint16
        img = img.astype(np.float32)
        return img.transpose(2, 0, 1)  # (5, 50, 50)


# ── Generic Folder Dataset (for any organized dataset) ────────────────────

class FolderMicroscopyDataset(MicroscopyDataset):
    """Generic dataset: root_dir/class_name/*.tiff (or .tif, .png).

    Works with any single-cell crop dataset organized by class folders.
    Supports multi-channel TIFF (any number of channels) and single-channel
    images that get stacked.
    """

    def __init__(self, root_dir, image_size=224, extensions=(".tiff", ".tif", ".png"), **kwargs):
        self._extensions = extensions
        super().__init__(root_dir, image_size, **kwargs)

    def _discover_files(self) -> None:
        for ext in self._extensions:
            self.files.extend(glob.glob(str(self.root_dir / "**" / f"*{ext}"), recursive=True))
        self.files = sorted(self.files)
        self.cell_types = sorted(set(Path(f).parent.name for f in self.files))
        self.label_map = {ct: i for i, ct in enumerate(self.cell_types)}
        self.labels = [self.label_map[Path(f).parent.name] for f in self.files]

    def _load_image(self, path: str) -> np.ndarray:
        if path.endswith((".tiff", ".tif")):
            img = tifffile.imread(path).astype(np.float32)
            if img.ndim == 2:
                img = img[np.newaxis]  # (1, H, W)
            elif img.ndim == 3 and img.shape[-1] <= 10:
                img = img.transpose(2, 0, 1)  # (H, W, C) → (C, H, W)
        else:
            from PIL import Image as PILImage
            pil_img = PILImage.open(path)
            img = np.array(pil_img, dtype=np.float32)
            if img.ndim == 2:
                img = img[np.newaxis]
            elif img.ndim == 3:
                img = img.transpose(2, 0, 1)
        return img
