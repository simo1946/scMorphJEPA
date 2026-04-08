"""Data loader for Severin PBMC 5-channel TIFF dataset."""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import tifffile
from PIL import Image


class SeverinDataset(Dataset):
    def __init__(self, root_dir, image_size=224):
        self.image_size = image_size
        self.files = sorted(glob.glob(os.path.join(root_dir, "**/*.tiff"), recursive=True))

        self.cell_types = sorted(list(set(
            f.split("/")[-2] for f in self.files
        )))
        self.label_map = {ct: i for i, ct in enumerate(self.cell_types)}
        self.labels = [self.label_map[f.split("/")[-2]] for f in self.files]

        print(f"Loaded {len(self.files)} images, {len(self.cell_types)} classes: {self.cell_types}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = tifffile.imread(self.files[idx])  # (50, 50, 5) uint16
        img = img.astype(np.float32)
        img = img / img.max() if img.max() > 0 else img
        img = img.transpose(2, 0, 1)  # (5, 50, 50)

        channels_resized = []
        for c in range(img.shape[0]):
            ch = Image.fromarray(img[c])
            ch = ch.resize((self.image_size, self.image_size), Image.BILINEAR)
            channels_resized.append(np.array(ch))

        img = np.stack(channels_resized, axis=0)  # (5, 224, 224)
        img = torch.from_numpy(img)

        return img, self.labels[idx]
