"""
hetero_trainer.py
=================
Training + evaluation loop for the heterogeneous (spatial or temporal)
surrogate-discovery model.

Handles:
  • ranking loss (SurrogateRankingLoss) with explicit surrogate masking
  • full metric tracking (precision@1/3, recall@3, NDCG@3, Spearman, load MAE)
  • topology-disjoint holdout evaluation
  • gradient clipping, LR scheduling, early stopping, checkpointing
"""

import os
import json
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from src.losses import SurrogateRankingLoss
from src.metrics import evaluate_graph, aggregate_metrics


class HeteroTrainer:
    def __init__(self, model, train_set, val_set, holdout_set, cfg: dict):
        self.model = model
        self.cfg = cfg
        self.device = torch.device(cfg.get("device", "cpu"))
        self.model.to(self.device)
        self.temporal = cfg.get("temporal", False)

        self.train_set   = train_set
        self.val_set     = val_set
        self.holdout_set = holdout_set

        self.criterion = SurrogateRankingLoss(
            w_rank=cfg.get("w_rank", 1.0),
            w_reg=cfg.get("w_reg", 0.3),
            w_load=cfg.get("w_load", 0.3),
            w_balance=cfg.get("w_balance", 0.1),
        )
        self.optimizer = AdamW(model.parameters(), lr=cfg.get("lr", 1e-3),
                               weight_decay=cfg.get("weight_decay", 1e-4))
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="max",
                                           patience=6, factor=0.5)

        self.ckpt_dir = cfg.get("ckpt_dir", "outputs/checkpoints")
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.best_score = -1.0
        self.patience_ctr = 0
        self.early_stop = cfg.get("early_stop_patience", 20)

    # ── one forward pass on a (sequence, name) item ───────────────────
    def _forward(self, seq):
        seq = [g.to(self.device) for g in seq]
        if self.temporal:
            sur_logits, load_pred, _ = self.model(seq)
            target = seq[-1]                 # predict at last step
        else:
            target = seq[-1]
            sur_logits, load_pred, _ = self.model(target)

        sur_labels = target["surrogate"].y
        sur_loads  = target["surrogate"].x[:, 1]          # load column

        # assemble all-node load tensors (predicted + true) for aux loss
        load_pred_all, load_true_all = [], []
        for nt in target.node_types:
            if target[nt].x.shape[0] > 0:
                load_pred_all.append(load_pred[nt])
                load_true_all.append(target[nt].x[:, 1])
        load_pred_all = torch.cat(load_pred_all)
        load_true_all = torch.cat(load_true_all)

        return sur_logits, sur_labels, sur_loads, load_pred_all, load_true_all

    # ── training epoch ────────────────────────────────────────────────
    def _train_epoch(self):
        self.model.train()
        agg = {}
        for seq, _ in self.train_set:
            self.optimizer.zero_grad()
            sl, lab, ld, lp, lt = self._forward(seq)
            loss, metrics = self.criterion(sl, lab, ld, lp, lt)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            for k, v in metrics.items():
                agg[k] = agg.get(k, 0) + v
        return {k: v / max(len(self.train_set), 1) for k, v in agg.items()}

    # ── evaluation (metrics) ──────────────────────────────────────────
    @torch.no_grad()
    def evaluate(self, dataset):
        self.model.eval()
        per_graph = []
        for seq, _ in dataset:
            sl, lab, ld, lp, lt = self._forward(seq)
            per_graph.append(evaluate_graph(
                torch.sigmoid(sl), lab, lp, lt))
        return aggregate_metrics(per_graph)

    # ── main loop ─────────────────────────────────────────────────────
    def train(self):
        epochs = self.cfg.get("epochs", 80)
        mode = "TEMPORAL" if self.temporal else "SPATIAL"
        print(f"\n  Training {mode} model — {epochs} epochs, device={self.device}")
        history = {"train_loss": [], "val": []}

        for ep in tqdm(range(1, epochs + 1), desc="Epochs"):
            train_metrics = self._train_epoch()
            val_metrics   = self.evaluate(self.val_set)
            history["train_loss"].append(train_metrics.get("loss_total", 0))
            history["val"].append(val_metrics)

            score = val_metrics.get("ndcg@3", 0)          # model-selection metric
            self.scheduler.step(score)

            if ep % 10 == 0:
                print(f"  Ep {ep:>3d} | loss={train_metrics.get('loss_total',0):.4f} "
                      f"| val P@1={val_metrics.get('precision@1',0):.3f} "
                      f"P@3={val_metrics.get('precision@3',0):.3f} "
                      f"NDCG@3={val_metrics.get('ndcg@3',0):.3f} "
                      f"ρ={val_metrics.get('spearman',0):.3f}")

            if score > self.best_score:
                self.best_score = score
                self.patience_ctr = 0
                torch.save({"epoch": ep, "model_state": self.model.state_dict(),
                            "score": score}, os.path.join(self.ckpt_dir, "best_hetero.pt"))
            else:
                self.patience_ctr += 1
            if self.patience_ctr >= self.early_stop:
                print(f"  Early stop at epoch {ep}")
                break

        print(f"  Best val NDCG@3: {self.best_score:.4f}")
        return history

    def load_best(self):
        ckpt = torch.load(os.path.join(self.ckpt_dir, "best_hetero.pt"),
                          map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        print(f"  Loaded best (epoch {ckpt['epoch']}, NDCG@3={ckpt['score']:.4f})")
