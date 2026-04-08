# Cell-JEPA

### Self-Supervised Cell Morphology Learning via Spatial Joint-Embedding Predictive Architecture

Cell-JEPA applies spatial masked prediction with [SIGReg](https://arxiv.org/abs/2603.19312) regularization to learn cell morphology representations from multi-channel fluorescence microscopy images — **without EMA, without augmentations, and with only two loss terms**.

<p align="center">
  <img src="figures/cell_jepa_vs_baseline.png" width="900"/>
</p>

## Key Results

On the [Severin PBMC dataset](https://doi.org/10.3929/ethz-b-000343106) (113,564 five-channel immune cell images, 8 cell types):

| Method | Training | k-NN (k=5) | k-NN (k=10) | k-NN (k=20) |
|--------|----------|-----------|------------|------------|
| DINO ViT-S/16 (zero-shot baseline) | None | 53.2% | 55.0% | 56.1% |
| **Cell-JEPA (1K images, 30 epochs)** | ~15 min, 1×T4 | **67.6%** | **67.3%** | **66.9%** |
| **Cell-JEPA (40K images, 8 epochs)** | ~100 min, 1×T4 | **63.8%** | **63.6%** | **62.9%** |

> Cell-JEPA improves over the zero-shot pretrained baseline by **+14.5 percentage points** with only 1,000 training images, demonstrating strong data efficiency.

## Motivation

Self-supervised vision transformers (scDINO, Cell-DINO) have shown excellent performance on cell phenotyping tasks using DINO-based self-distillation. However, these methods rely on:
- Exponential Moving Average (EMA) teacher networks
- Multi-crop augmentation strategies designed for natural images
- Complex multi-component training objectives

Cell-JEPA replaces all of this with a simpler framework inspired by [LeWorldModel (Maes et al., 2026)](https://arxiv.org/abs/2603.19312):

| | scDINO | Cell-DINO | I-JEPA | **Cell-JEPA** |
|--|--------|-----------|--------|---------------|
| Learning objective | Self-distillation | Self-distillation | Masked prediction | **Masked prediction** |
| Collapse prevention | EMA | EMA | EMA | **SIGReg** |
| Augmentations | Multi-crop, color | Multi-crop, color | None | **None** |
| EMA required | Yes | Yes | Yes | **No** |
| Loss terms | Multiple | Multiple | 2 + EMA | **2 only** |

## Method

1. **Encode**: A ViT-S/16 encoder (initialized from DINO-pretrained ImageNet weights, adapted to 5 fluorescence channels) processes cell images into 196 patch embeddings
2. **Mask**: 60% of patch embeddings are randomly masked
3. **Predict**: A lightweight transformer predictor reconstructs masked patch embeddings from visible context
4. **Regularize**: SIGReg enforces isotropic Gaussian distribution on embeddings, preventing collapse without EMA

**Total loss = MSE(predicted, target) + λ · SIGReg(embeddings)**

One hyperparameter (λ). No EMA. No augmentations. No multi-term loss.

## Dataset

We use the [Deep Phenotyping PBMC Image Set (Severin et al., 2022)](https://doi.org/10.3929/ethz-b-000343106):
- **113,564** five-channel fluorescence microscopy images
- **50×50 pixels**, resized to 224×224 for ViT input
- **5 channels**: Alexa Fluor 647, Brightfield, DAPI, FITC 488, PE 594
- **8 immune cell classes**: T4, T8, T0, M0, DC, Nk, B, Negs
- **Train/test split**: 89,564 / 24,000

## Quick Start

### Requirements
```bash
pip install torch torchvision timm tifffile scikit-learn matplotlib
```

### Download data
```bash
# Severin PBMC dataset (~1.8 GB)
wget -O severin_pbmc.zip "https://www.research-collection.ethz.ch/bitstreams/8689d69b-d916-4c8e-9b3f-2981c512b70b/download"
unzip -q severin_pbmc.zip -d severin_data

# DINO pretrained ViT-S/16 weights
wget -O dino_vits16.pth "https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
```

### Train
```bash
python cell_jepa_train.py \
    --data_dir severin_data/DeepPhenotype_PBMC_ImageSet_YSeverin \
    --checkpoint dino_vits16.pth \
    --epochs 50 \
    --batch_size 24 \
    --n_images 0  # 0 = use all training images
```

### Evaluate
```bash
python evaluate.py \
    --data_dir severin_data/DeepPhenotype_PBMC_ImageSet_YSeverin \
    --model_path output/best_model.pt \
    --checkpoint dino_vits16.pth
```

## Repository Structure
```
cell-jepa/
├── README.md
├── cell_jepa_train.py      # Training script
├── cell_jepa.py             # Model architecture (encoder + predictor + SIGReg)
├── severin_dataset.py       # Data loader for 5-channel TIFF images
├── evaluate.py              # k-NN evaluation and t-SNE visualization
├── notebooks/
│   └── cell_jepa_demo.ipynb # Complete demo notebook (runs on Colab)
├── figures/
│   ├── cell_jepa_vs_baseline.png
│   └── training_curves.png
└── results/
    └── results_log.txt
```

## Preliminary Training Dynamics

Both prediction loss and SIGReg loss decrease consistently during training, confirming the model learns meaningful spatial structure:

| Epoch | Pred Loss (train) | SIGReg (train) | Pred Loss (val) |
|-------|-------------------|----------------|-----------------|
| 1 | 5.70 | 15.78 | 3.19 |
| 10 | 1.38 | 6.92 | 1.36 |
| 20 | 0.81 | 5.33 | 0.79 |
| 27 | 0.27 | 4.53 | 0.26 |

## Ongoing Work

- [ ] Full scaling curve (1K → 89K images) with consistent 50-epoch training
- [ ] Direct comparison with scDINO (reproduction on same dataset)
- [ ] Channel ablation study (which fluorescence channels drive classification)
- [ ] DINOv3 frozen features baseline
- [ ] From-scratch ViT training (no ImageNet pretraining)
- [ ] Extension to Cell Painting datasets (BBBC021)

## Connection to CellAgora

Cell-JEPA is Stage 1 of the **CellAgora** research program — a multi-stage framework for AI-driven cell biology. Future stages will extend to temporal prediction (live-cell imaging), spatial-temporal modeling, and cell-cell interaction analysis via graph architectures.

## Citation

If you find this work useful, please consider citing:

```bibtex
@article{bonaccorsi2026celljepa,
  title={Cell-JEPA: Self-Supervised Cell Morphology Learning via Spatial 
         Joint-Embedding Predictive Architecture},
  author={Bonaccorsi, Simone},
  year={2026},
  note={Preprint in preparation}
}
```

## Acknowledgments

This work builds on:
- [LeWorldModel](https://github.com/lucas-maes/le-wm) (Maes et al., 2026) — SIGReg regularization for stable JEPA training
- [scDINO](https://github.com/JacobHanimann/scDINO) (Pfaendler et al., 2023) — Self-supervised ViTs for cell microscopy
- [I-JEPA](https://github.com/facebookresearch/ijepa) (Assran et al., 2023) — Image-based JEPA framework
- [Severin PBMC dataset](https://doi.org/10.3929/ethz-b-000343106) (Severin et al., 2022)

## License

MIT
