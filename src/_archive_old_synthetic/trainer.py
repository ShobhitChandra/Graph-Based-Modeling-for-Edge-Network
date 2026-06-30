"""
trainer.py
==========
Full training + evaluation loop with:
  • Learning-rate scheduling (ReduceLROnPlateau)
  • Early stopping
  • TensorBoard logging
  • Checkpoint saving / loading
"""

import os
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.gnn_model import SurrogateDiscoveryGNN, SurrogateDiscoveryLoss


# ──────────────────────────────────────────────────────────────────────

class Trainer:
    """
    Manages the full train / validate / test lifecycle.

    Usage
    ─────
    trainer = Trainer(model, train_ds, val_ds, test_ds, cfg)
    trainer.train()
    trainer.evaluate(trainer.test_loader)
    """

    def __init__(
        self,
        model: SurrogateDiscoveryGNN,
        train_dataset,
        val_dataset,
        test_dataset,
        cfg: dict,
    ):
        self.model   = model
        self.cfg     = cfg
        self.device  = torch.device(cfg.get("device", "cpu"))
        self.model.to(self.device)

        self.criterion = SurrogateDiscoveryLoss(
            lambda1=cfg.get("lambda1", 1.0),
            lambda2=cfg.get("lambda2", 0.3),
        )

        self.optimizer = Adam(
            model.parameters(),
            lr=cfg.get("lr", 1e-3),
            weight_decay=cfg.get("weight_decay", 1e-4),
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", patience=5, factor=0.5
        )

        batch_size = cfg.get("batch_size", 16)
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader   = DataLoader(val_dataset,   batch_size=batch_size)
        self.test_loader  = DataLoader(test_dataset,  batch_size=batch_size)

        log_dir = cfg.get("log_dir", "outputs/runs")
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)

        self.ckpt_dir  = cfg.get("ckpt_dir", "outputs/checkpoints")
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.best_val_loss = float("inf")
        self.patience_ctr  = 0
        self.early_stop_patience = cfg.get("early_stop_patience", 15)

    # ── Training epoch ────────────────────────────────────────────────

    def _train_epoch(self, epoch: int) -> dict:
        self.model.train()
        totals = {"loss_total": 0, "loss_surrogate": 0, "loss_load": 0}
        n_batches = 0

        for batch in self.train_loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            sur_scores, load_preds, _ = self.model(batch)

            loss, metrics = self.criterion(
                sur_scores,
                load_preds,
                batch.y,
                batch.true_load,
                batch.node_type,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            for k in totals:
                totals[k] += metrics[k]
            n_batches += 1

        return {k: v / n_batches for k, v in totals.items()}

    # ── Validation / Test ─────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict:
        self.model.eval()
        totals = {"loss_total": 0, "loss_surrogate": 0, "loss_load": 0}
        n_batches = 0

        for batch in loader:
            batch = batch.to(self.device)
            sur_scores, load_preds, _ = self.model(batch)
            _, metrics = self.criterion(
                sur_scores, load_preds,
                batch.y, batch.true_load, batch.node_type,
            )
            for k in totals:
                totals[k] += metrics[k]
            n_batches += 1

        return {k: v / n_batches for k, v in totals.items()}

    # ── Main training loop ────────────────────────────────────────────

    def train(self):
        n_epochs = self.cfg.get("epochs", 100)
        print(f"\n{'─'*60}")
        print(f"  Training SurrogateDiscoveryGNN  ({n_epochs} epochs)")
        print(f"  Device: {self.device}")
        print(f"{'─'*60}\n")

        history = {"train": [], "val": []}

        for epoch in tqdm(range(1, n_epochs + 1), desc="Epochs"):
            t0 = time.time()
            train_metrics = self._train_epoch(epoch)
            val_metrics   = self.evaluate(self.val_loader)

            # TensorBoard
            for k, v in train_metrics.items():
                self.writer.add_scalar(f"train/{k}", v, epoch)
            for k, v in val_metrics.items():
                self.writer.add_scalar(f"val/{k}",   v, epoch)
            self.writer.add_scalar(
                "lr", self.optimizer.param_groups[0]["lr"], epoch
            )

            history["train"].append(train_metrics)
            history["val"].append(val_metrics)

            self.scheduler.step(val_metrics["loss_total"])

            # Print every 10 epochs
            if epoch % 10 == 0:
                elapsed = time.time() - t0
                print(
                    f"  Ep {epoch:>4d} | "
                    f"train_loss={train_metrics['loss_total']:.4f} | "
                    f"val_loss={val_metrics['loss_total']:.4f} | "
                    f"sur_loss={val_metrics['loss_surrogate']:.4f} | "
                    f"{elapsed:.2f}s"
                )

            # Best checkpoint
            if val_metrics["loss_total"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss_total"]
                self.patience_ctr  = 0
                self._save_checkpoint(epoch, val_metrics["loss_total"], best=True)
            else:
                self.patience_ctr += 1

            # Early stopping
            if self.patience_ctr >= self.early_stop_patience:
                print(f"\n  Early stopping at epoch {epoch}.")
                break

        self.writer.close()
        print(f"\n  Best val loss: {self.best_val_loss:.4f}")
        return history

    # ── Checkpointing ────────────────────────────────────────────────

    def _save_checkpoint(self, epoch: int, val_loss: float, best: bool = False):
        name = "best_model.pt" if best else f"ckpt_ep{epoch}.pt"
        path = os.path.join(self.ckpt_dir, name)
        torch.save({
            "epoch":      epoch,
            "model_state": self.model.state_dict(),
            "optim_state": self.optimizer.state_dict(),
            "val_loss":    val_loss,
        }, path)

    def load_best(self):
        path = os.path.join(self.ckpt_dir, "best_model.pt")
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        print(f"  Loaded best model from epoch {ckpt['epoch']} "
              f"(val_loss={ckpt['val_loss']:.4f})")
