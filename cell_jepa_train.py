"""Train Cell-JEPA on Severin PBMC dataset."""

import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, '/content')
from severin_dataset import SeverinDataset
from cell_jepa import build_cell_jepa, SIGReg
import shutil

def train():
    device = torch.device("cuda")

    # === Config ===
    batch_size = 24
    epochs = 50
    lr = 1e-4
    weight_decay = 0.05
    sigreg_weight = 10.0
    mask_ratio = 0.6
    seed = 42
    n_images = 5000
    # === Data ===
    base = "/content/severin_data/DeepPhenotype_PBMC_ImageSet_YSeverin"
    train_dataset = SeverinDataset(os.path.join(base, "Training"), image_size=224)
    test_dataset = SeverinDataset(os.path.join(base, "Test"), image_size=224)

    # trying only a subset of the dataset for computation limitation #
    subset_idx = torch.randperm(len(train_dataset))[:n_images]
    train_subset = torch.utils.data.Subset(train_dataset, subset_idx)

    #train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
    #                          drop_last=True, num_workers=2, pin_memory=True)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                              drop_last=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=0, pin_memory=True)

    # === Model ===
    model = build_cell_jepa(
        checkpoint_path="/content/dino_vits16_pretrain.pth",
        in_channels=5,
        mask_ratio=mask_ratio,
        predictor_depth=4,
    ).to(device)

    sigreg = SIGReg(knots=17, num_proj=1024).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    predictor_params = sum(p.numel() for p in model.predictor.parameters())
    print(f"Total params: {total_params:,}")
    print(f"  Encoder: {encoder_params:,}")
    print(f"  Predictor: {predictor_params:,}")

    # === Optimizer ===
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # === Training ===
    output_dir = "/content/cell_jepa_output"
    os.makedirs(output_dir, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        train_pred_loss = 0.0
        train_sigreg_loss = 0.0
        n_batches = 0

        for images, labels in train_loader:
            images = images.to(device)

            output = model(images)

            # Prediction loss: MSE between predicted and target patch embeddings
            pred_loss = F.mse_loss(output["pred_emb"], output["target_emb"])

            # SIGReg on CLS tokens
            cls_for_sigreg = output["cls_token"].unsqueeze(0)  # (1, B, 384)
            sig_loss = sigreg(cls_for_sigreg)

            loss = pred_loss + sigreg_weight * sig_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_pred_loss += pred_loss.item()
            train_sigreg_loss += sig_loss.item()
            n_batches += 1

        scheduler.step()

        # === Validation ===
        model.eval()
        val_pred_loss = 0.0
        val_sigreg_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                output = model(images)
                pred_loss = F.mse_loss(output["pred_emb"], output["target_emb"])
                sig_loss = sigreg(output["cls_token"].unsqueeze(0))
                val_pred_loss += pred_loss.item()
                val_sigreg_loss += sig_loss.item()
                val_batches += 1

        avg_train_pred = train_pred_loss / n_batches
        avg_train_sig = train_sigreg_loss / n_batches
        avg_val_pred = val_pred_loss / val_batches
        avg_val_sig = val_sigreg_loss / val_batches

        print(f"Epoch {epoch+1:3d}/{epochs} | "
              f"Train pred:{avg_train_pred:.4f} sig:{avg_train_sig:.4f} | "
              f"Val pred:{avg_val_pred:.4f} sig:{avg_val_sig:.4f} | "
              f"LR:{scheduler.get_last_lr()[0]:.6f}")

        # Save best
        val_total = avg_val_pred + sigreg_weight * avg_val_sig
        if val_total < best_val_loss:
            best_val_loss = val_total
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pt"))
            shutil.copy(os.path.join(output_dir, "best_model.pt"), f"/content/drive/MyDrive/cell_jepa_checkpoints/best_{n_images}k_{epoch+1}ep.pt")
            print(f"  -> Saved best model ")

        # Checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(output_dir, f"epoch_{epoch+1}.pt"))

    torch.save(model.state_dict(), os.path.join(output_dir, "final_model.pt"))
    print(f"Done. Models saved to {output_dir}")


if __name__ == "__main__":
    train()
