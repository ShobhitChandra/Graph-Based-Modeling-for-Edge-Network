"""
losses.py
=========
Loss functions for surrogate *ranking* (not just regression).

The original model used plain MSE on surrogate scores. For a discovery /
ranking task we care about getting the *order* right, so this module adds:

  • BPR (Bayesian Personalised Ranking) pairwise loss
  • Pairwise margin / hinge ranking loss
  • Listwise ListNet (softmax cross-entropy over the score distribution)
  • Load-prediction MSE  (auxiliary)
  • Load-balancing regulariser (discourage overloading one surrogate)

`SurrogateRankingLoss` combines them with configurable weights and handles
masking of non-surrogate nodes explicitly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ──────────────────────────────────────────────────────────────────────
#  Individual ranking losses
# ──────────────────────────────────────────────────────────────────────

def bpr_loss(scores: Tensor, labels: Tensor, num_pairs: int = 256) -> Tensor:
    """
    Bayesian Personalised Ranking: sample (i, j) pairs where label_i > label_j
    and push score_i above score_j via -log σ(score_i - score_j).
    """
    n = scores.shape[0]
    if n < 2:
        return scores.new_zeros(())

    i = torch.randint(0, n, (num_pairs,), device=scores.device)
    j = torch.randint(0, n, (num_pairs,), device=scores.device)
    # keep only pairs with a strict label ordering
    keep = labels[i] != labels[j]
    if keep.sum() == 0:
        return scores.new_zeros(())
    i, j = i[keep], j[keep]
    # orient so that i is the higher-labelled one
    swap = labels[j] > labels[i]
    i2 = torch.where(swap, j, i)
    j2 = torch.where(swap, i, j)
    diff = scores[i2] - scores[j2]
    return -F.logsigmoid(diff).mean()


def pairwise_hinge_loss(scores: Tensor, labels: Tensor,
                        margin: float = 0.1, num_pairs: int = 256) -> Tensor:
    """Margin ranking loss: max(0, margin - (s_i - s_j)) for label_i > label_j."""
    n = scores.shape[0]
    if n < 2:
        return scores.new_zeros(())
    i = torch.randint(0, n, (num_pairs,), device=scores.device)
    j = torch.randint(0, n, (num_pairs,), device=scores.device)
    keep = labels[i] != labels[j]
    if keep.sum() == 0:
        return scores.new_zeros(())
    i, j = i[keep], j[keep]
    target = torch.sign(labels[i] - labels[j])     # +1 if i better, -1 if j better
    return F.margin_ranking_loss(scores[i], scores[j], target, margin=margin)


def listnet_loss(scores: Tensor, labels: Tensor) -> Tensor:
    """
    Listwise ListNet (top-1): cross-entropy between the softmax of predicted
    scores and the softmax of the ground-truth labels over all surrogates.
    """
    if scores.shape[0] < 2:
        return scores.new_zeros(())
    p_true = F.softmax(labels, dim=0)
    log_p_pred = F.log_softmax(scores, dim=0)
    return -(p_true * log_p_pred).sum()


def load_balance_reg(sur_scores: Tensor, sur_loads: Tensor) -> Tensor:
    """
    Penalise recommending already-loaded surrogates: correlation between the
    (softmax) recommendation weight and current load should be ≤ 0.
    Implemented as the positive part of E[w · load].
    """
    if sur_scores.shape[0] < 2:
        return sur_scores.new_zeros(())
    w = F.softmax(sur_scores, dim=0)
    return F.relu((w * sur_loads).sum())


# ──────────────────────────────────────────────────────────────────────
#  Combined multi-task loss
# ──────────────────────────────────────────────────────────────────────

class SurrogateRankingLoss(nn.Module):
    """
    Combined objective:

      L =  w_rank   · (bpr + hinge + listnet)
         + w_reg    · MSE(sigmoid(score), label)      [pointwise anchor]
         + w_load   · MSE(load_pred, true_load)        [auxiliary]
         + w_balance· load_balance_reg

    All surrogate-specific terms operate ONLY on surrogate nodes; the loss
    receives already-masked surrogate tensors plus the full node-set load
    tensors, so masking is explicit and unambiguous.
    """
    def __init__(self, w_rank: float = 1.0, w_reg: float = 0.3,
                 w_load: float = 0.3, w_balance: float = 0.1):
        super().__init__()
        self.w_rank    = w_rank
        self.w_reg     = w_reg
        self.w_load    = w_load
        self.w_balance = w_balance
        self.mse = nn.MSELoss()

    def forward(
        self,
        sur_logits: Tensor,     # [S] raw scores on surrogate nodes
        sur_labels: Tensor,     # [S] ground-truth quality ∈ [0,1]
        sur_loads:  Tensor,     # [S] current load of each surrogate
        all_load_pred: Tensor,  # [N] predicted load, all nodes
        all_true_load: Tensor,  # [N] true load, all nodes
    ):
        l_bpr   = bpr_loss(sur_logits, sur_labels)
        l_hinge = pairwise_hinge_loss(sur_logits, sur_labels)
        l_list  = listnet_loss(sur_logits, sur_labels)
        l_rank  = l_bpr + l_hinge + l_list

        l_reg   = self.mse(torch.sigmoid(sur_logits), sur_labels)
        l_load  = self.mse(all_load_pred, all_true_load)
        l_bal   = load_balance_reg(sur_logits, sur_loads)

        total = (self.w_rank * l_rank + self.w_reg * l_reg
                 + self.w_load * l_load + self.w_balance * l_bal)

        return total, {
            "loss_total":   float(total.detach()),
            "loss_bpr":     float(l_bpr.detach()),
            "loss_hinge":   float(l_hinge.detach()),
            "loss_listnet": float(l_list.detach()),
            "loss_reg":     float(l_reg.detach()),
            "loss_load":    float(l_load.detach()),
            "loss_balance": float(l_bal.detach()),
        }
