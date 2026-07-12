"""scMorphJEPA training pipeline — device-agnostic, config-driven."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from scmorphjepa.training.regularizers import build_regularizer

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """Training configuration."""
    batch_size: int = 24
    epochs: int = 50
    lr: float = 1e-4
    weight_decay: float = 0.05
    sigreg_weight: float = 10.0     # kept for backward compat; used if reg_weight is None
    regularizer: str = "sigreg"     # sigreg | vicreg | koleo | barlow | visreg | none
    reg_weight: float | None = None  # weight for the chosen regularizer; falls back to sigreg_weight
    reg_kwargs: dict = field(default_factory=dict)  # constructor kwargs for the regularizer
    n_images: int = 0         # 0 = use all
    num_workers: int = 2
    seed: int = 42
    output_dir: str = "output"
    save_every: int = 10
    device: str = "auto"
    run_name: str = ""        # if empty, auto-derived; namespaces all checkpoints for this run
    drive_checkpoint_dir: str | None = None  # if set, the best model is also copied here (e.g. Drive)
    resume: bool = True       # auto-resume from the last checkpoint if one exists (Colab-safe)
    drive_save_every: int = 1  # mirror the resume checkpoint to Drive every N epochs (1 = every epoch)

    def effective_reg_weight(self) -> float:
        return self.reg_weight if self.reg_weight is not None else self.sigreg_weight

    def resolved_run_name(self) -> str:
        if self.run_name:
            return self.run_name
        n = "all" if self.n_images == 0 else str(self.n_images)
        base = f"scmorphjepa_n{n}_e{self.epochs}"
        if self.regularizer != "sigreg":
            base += f"_{self.regularizer}"
        return base

    def get_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)


class Trainer:
    """scMorphJEPA trainer.

    Args:
        model: ScMorphJEPA model.
        train_dataset: Training dataset.
        test_dataset: Validation/test dataset.
        config: Training configuration.
    """

    def __init__(
        self, model, train_dataset: Dataset, test_dataset: Dataset | None = None,
        config: TrainConfig | None = None,
    ) -> None:
        self.config = config or TrainConfig()
        self.device = self.config.get_device()
        self.model = model.to(self.device)
        self.regularizer = build_regularizer(
            self.config.regularizer, **self.config.reg_kwargs
        ).to(self.device)
        self.reg_weight = self.config.effective_reg_weight()
        logger.info(f"Regularizer: {self.config.regularizer} (weight={self.reg_weight})")

        # Subset if requested
        if self.config.n_images > 0 and self.config.n_images < len(train_dataset):
            torch.manual_seed(self.config.seed)
            indices = torch.randperm(len(train_dataset))[: self.config.n_images]
            train_dataset = Subset(train_dataset, indices.tolist())
            logger.info(f"Using subset: {len(train_dataset)} images")

        self.train_loader = DataLoader(
            train_dataset, batch_size=self.config.batch_size, shuffle=True,
            drop_last=True, num_workers=self.config.num_workers, pin_memory=True,
        )
        self.test_loader = None
        if test_dataset is not None:
            self.test_loader = DataLoader(
                test_dataset, batch_size=self.config.batch_size, shuffle=False,
                num_workers=self.config.num_workers, pin_memory=True,
            )

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.config.epochs
        )

        # Namespace every checkpoint under the run name so different runs never collide.
        self.run_name = self.config.resolved_run_name()
        self.output_dir = Path(self.config.output_dir) / self.run_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.drive_dir = Path(self.config.drive_checkpoint_dir) if self.config.drive_checkpoint_dir else None
        if self.drive_dir:
            self.drive_dir.mkdir(parents=True, exist_ok=True)
        self.best_val_loss = float("inf")
        self.history: list[dict] = []
        self._last_drive_epoch = -1  # last epoch whose full checkpoint reached Drive (survives resets)
        logger.info(f"Run name: {self.run_name} | checkpoints → {self.output_dir}"
                    + (f" (+ Drive: {self.drive_dir})" if self.drive_dir else ""))

    def train(self) -> list[dict]:
        """Run full training loop with Colab-safe resume. Returns training history.

        Saves a full-state checkpoint (model + optimizer + scheduler + epoch + history)
        every epoch and, if the runtime dies, picks up from the last completed epoch on
        the next call. Per-epoch deterministic seeding makes the resumed run equivalent
        to an uninterrupted one.
        """
        start_epoch = self._maybe_resume()
        logger.info(
            f"Training scMorphJEPA: epochs {start_epoch + 1}→{self.config.epochs}, "
            f"batch_size={self.config.batch_size}, device={self.device}"
        )

        for epoch in range(start_epoch, self.config.epochs):
            # Deterministic, resume-safe shuffle: the loader's shuffle is drawn from the
            # global RNG when its iterator is created, so seeding here reproduces the exact
            # order whether this epoch runs fresh or after a resume.
            torch.manual_seed(self.config.seed + epoch)

            train_metrics = self._train_epoch(epoch)
            val_metrics = self._validate() if self.test_loader else {}
            self.scheduler.step()

            record = {"epoch": epoch + 1, **train_metrics, **val_metrics,
                       "lr": self.scheduler.get_last_lr()[0]}
            self.history.append(record)

            # print() (not just logger) so the line always shows in the notebook and the
            # LAST printed line tells you exactly where a disconnect stopped the run.
            msg = (f"Epoch {epoch+1:3d}/{self.config.epochs} | "
                   f"train_pred={train_metrics['train_pred']:.4f} "
                   f"train_sig={train_metrics['train_sig']:.4f}"
                   + (f" | val_pred={val_metrics.get('val_pred', 0):.4f}" if val_metrics else "")
                   + f" | lr={record['lr']:.6f}")
            print(msg, flush=True)
            logger.info(msg)

            # Best model (overwrites within run; distinct file per run via run_name)
            val_total = val_metrics.get("val_total", train_metrics["train_total"])
            is_best = val_total < self.best_val_loss
            if is_best:
                self.best_val_loss = val_total
                self._atomic_save(self.model.state_dict(), self.output_dir / "best_model.pt")
                tag = f"{self.run_name}_best.pt"
                print(f"  ✓ best model updated → epoch {epoch+1} (loss={val_total:.4f})  [{tag}]",
                      flush=True)
                if self.drive_dir:
                    self._atomic_save(self.model.state_dict(),
                                      self.drive_dir / f"{self.run_name}_best.pt")

            # Full resume checkpoint every epoch (this is what survives a disconnect)
            self._save_resume_checkpoint(epoch)
            drive_hit = self.drive_dir and ((epoch + 1) % self.config.drive_save_every == 0
                                            or (epoch + 1) == self.config.epochs)
            where = "local + Drive" if drive_hit else "local"
            print(f"  · checkpoint saved → epoch {epoch+1} ({where})", flush=True)

            if (epoch + 1) % self.config.save_every == 0:
                self._atomic_save(self.model.state_dict(),
                                  self.output_dir / f"epoch_{epoch+1}.pt")

        self._atomic_save(self.model.state_dict(), self.output_dir / "final_model.pt")
        logger.info(f"Training complete. Models saved to {self.output_dir}")
        return self.history

    # ── checkpoint / resume helpers ──────────────────────────────────────────

    @staticmethod
    def _atomic_save(obj, path: Path) -> None:
        """Save to a temp file then atomically replace — never leaves a half-written file
        (important on Drive when a Colab disconnect can land mid-write)."""
        path = Path(path)
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(obj, tmp)
        import os
        os.replace(tmp, path)  # atomic on the same filesystem

    def _resume_paths(self) -> list[Path]:
        """Where a resume checkpoint might live (Drive first so it survives VM resets)."""
        paths = []
        if self.drive_dir:
            paths.append(self.drive_dir / f"{self.run_name}_last.pt")
        paths.append(self.output_dir / "last.pt")
        return paths

    def _save_resume_checkpoint(self, epoch: int) -> None:
        ckpt = {
            "epoch": epoch,  # last COMPLETED epoch
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "history": self.history,
            "run_name": self.run_name,
        }
        # LOCAL gets the FULL checkpoint (with optimizer) every epoch — exact same-session resume.
        self._atomic_save(ckpt, self.output_dir / "last.pt")
        # DRIVE gets an OPTIMIZER-FREE checkpoint. The Adam state is ~2/3 of the file and the main
        # reason a ~250 MB upload does not finish syncing to Google before a Colab reset; dropping
        # it makes the Drive copy small enough to sync reliably every time. On a cross-session
        # resume the optimizer is re-initialized (Adam re-warms within a few steps).
        is_final = (epoch + 1) == self.config.epochs
        if self.drive_dir and ((epoch + 1) % self.config.drive_save_every == 0 or is_final):
            drive_ckpt = {k: v for k, v in ckpt.items() if k != "optimizer_state_dict"}
            self._atomic_save(drive_ckpt, self.drive_dir / f"{self.run_name}_last.pt")
            self._last_drive_epoch = epoch

        # Tiny human-readable progress file EVERY epoch (negligible size) so you can see how
        # far a run got — open it on Drive without loading the big checkpoint.
        self._write_progress(epoch)

    def _write_progress(self, epoch: int) -> None:
        import json
        import time
        last = self.history[-1] if self.history else {}
        # recoverable_epoch = where a FRESH-runtime resume will actually start from, i.e. the last
        # epoch whose full checkpoint reached Drive. With drive_save_every=1 this equals
        # epoch_completed; if throttled, it can lag, and this field tells you by how much.
        recoverable = self._last_drive_epoch + 1 if self.drive_dir else epoch + 1
        prog = {
            "run_name": self.run_name,
            "epoch_completed": epoch + 1,
            "recoverable_epoch": recoverable,
            "total_epochs": self.config.epochs,
            "percent": round(100 * (epoch + 1) / self.config.epochs, 1),
            "best_val_loss": round(float(self.best_val_loss), 6),
            "last_train_pred": round(float(last.get("train_pred", float("nan"))), 6),
            "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        text = json.dumps(prog, indent=2)
        (self.output_dir / "progress.json").write_text(text)
        if self.drive_dir:
            (self.drive_dir / f"{self.run_name}_progress.json").write_text(text)

    def _maybe_resume(self) -> int:
        """Load the FRESHEST resume checkpoint (highest completed epoch), never a fixed path order —
        a stale Drive copy must not override a fresher local one. Returns the next epoch index."""
        if not self.config.resume:
            return 0
        best = None  # (epoch, path, ckpt)
        for p in self._resume_paths():
            if not Path(p).exists():
                continue
            try:
                ck = torch.load(p, map_location="cpu", weights_only=False)
                ep = int(ck["epoch"])
            except Exception:
                continue
            if best is None or ep > best[0]:
                best = (ep, Path(p), ck)
        if best is None:
            return 0
        loaded_epoch, path, ckpt = best
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            for st in self.optimizer.state.values():  # move loaded (CPU) state onto the device
                for k, v in st.items():
                    if isinstance(v, torch.Tensor):
                        st[k] = v.to(self.device)
            opt_note = ""
        else:
            opt_note = " (optimizer re-initialized: Drive checkpoint omits it)"
        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
        self.history = ckpt.get("history", [])
        on_drive = bool(self.drive_dir) and path == self.drive_dir / f"{self.run_name}_last.pt"
        if on_drive:
            self._last_drive_epoch = loaded_epoch
        next_epoch = loaded_epoch + 1
        src = "Drive" if on_drive else "local"
        logger.info(f"Resumed '{self.run_name}' from {src} checkpoint "
                    f"(completed epoch {loaded_epoch + 1}) → continuing at epoch {next_epoch + 1}{opt_note}")
        return next_epoch

    def _train_epoch(self, epoch: int = 0) -> dict:
        self.model.train()
        pred_loss_sum, sig_loss_sum, n = 0.0, 0.0, 0

        bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.epochs}",
                   leave=True, dynamic_ncols=True)
        for images, _ in bar:
            images = images.to(self.device)
            output = self.model(images)

            pred_loss = F.mse_loss(output["pred_emb"], output["target_emb"])
            sig_loss = self.regularizer(output["cls_token"])
            loss = pred_loss + self.reg_weight * sig_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            pred_loss_sum += pred_loss.item()
            sig_loss_sum += sig_loss.item()
            n += 1
            bar.set_postfix(pred=f"{pred_loss_sum/n:.4f}", sig=f"{sig_loss_sum/n:.4f}")

        return {
            "train_pred": pred_loss_sum / n,
            "train_sig": sig_loss_sum / n,
            "train_total": (pred_loss_sum + self.reg_weight * sig_loss_sum) / n,
        }

    @torch.no_grad()
    def _validate(self) -> dict:
        self.model.eval()
        pred_loss_sum, sig_loss_sum, n = 0.0, 0.0, 0

        for images, _ in self.test_loader:
            images = images.to(self.device)
            output = self.model(images)
            pred_loss_sum += F.mse_loss(output["pred_emb"], output["target_emb"]).item()
            sig_loss_sum += self.regularizer(output["cls_token"]).item()
            n += 1

        return {
            "val_pred": pred_loss_sum / n,
            "val_sig": sig_loss_sum / n,
            "val_total": (pred_loss_sum + self.reg_weight * sig_loss_sum) / n,
        }
