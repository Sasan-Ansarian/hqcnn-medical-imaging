"""
engine.py
---------
Training and evaluation engine for the HQCNN framework.

Provides two core functions:
    - train_one_epoch: single training pass with gradient tracking
    - evaluate: evaluation pass returning full metrics

Gradient tracking (addressing Gap 3 from the literature review):
    During each training step, the L2 norm of the full parameter gradient
    is computed and recorded. Per-epoch mean and variance of gradient norms
    are returned, enabling analysis of optimisation stability and comparison
    between classical and quantum training dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@dataclass
class TrainEpochStats:
    """
    Statistics collected during one training epoch.

    Attributes
    ----------
    loss : float
        Mean cross-entropy loss over all batches.
    accuracy : float
        Top-1 accuracy over all samples.
    grad_norm_mean : float
        Mean of per-batch gradient L2 norms.
        Used to track optimisation stability (Gap 3 analysis).
    grad_norm_var : float
        Variance of per-batch gradient L2 norms.
        High variance indicates unstable optimisation trajectory.
    """
    loss: float
    accuracy: float
    grad_norm_mean: float
    grad_norm_var: float


@dataclass
class EvalEpochStats:
    """
    Statistics collected during one evaluation pass.

    Attributes
    ----------
    loss : float
        Mean cross-entropy loss over all batches.
    accuracy : float
        Top-1 accuracy (secondary metric).
    macro_f1 : float
        Macro-averaged F1 score (primary metric under class imbalance).
    balanced_accuracy : float
        Mean per-class recall (secondary metric).
    conf_mat : np.ndarray
        Confusion matrix of shape (num_classes, num_classes).
    """
    loss: float
    accuracy: float
    macro_f1: float
    balanced_accuracy: float
    conf_mat: np.ndarray


def _batch_to_xy(batch) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract (x, y) from a MedMNIST batch, squeezing the label if needed."""
    x, y = batch
    if y.ndim > 1:
        y = y.squeeze(-1)
    return x, y


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> EvalEpochStats:
    """
    Evaluate a model on a data loader.

    Parameters
    ----------
    model : nn.Module
        Model to evaluate (set to eval mode internally).
    loader : DataLoader
        DataLoader yielding (images, labels) batches.
    criterion : nn.Module
        Loss function (e.g. CrossEntropyLoss with class weights).
    device : torch.device
        Device to run evaluation on.

    Returns
    -------
    EvalEpochStats
        Loss, accuracy, macro F1, balanced accuracy, and confusion matrix.
    """
    model.eval()

    total_loss = 0.0
    n_samples = 0
    all_true: List[int] = []
    all_pred: List[int] = []

    for batch in loader:
        x, y = _batch_to_xy(batch)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)

        bs = x.size(0)
        total_loss += loss.item() * bs
        n_samples += bs

        pred = torch.argmax(logits, dim=1)
        all_true.extend(y.detach().cpu().tolist())
        all_pred.extend(pred.detach().cpu().tolist())

    from sklearn.metrics import (
        accuracy_score, f1_score, confusion_matrix, balanced_accuracy_score
    )

    return EvalEpochStats(
        loss=total_loss / max(n_samples, 1),
        accuracy=accuracy_score(all_true, all_pred),
        macro_f1=f1_score(all_true, all_pred, average="macro"),
        balanced_accuracy=balanced_accuracy_score(all_true, all_pred),
        conf_mat=confusion_matrix(all_true, all_pred),
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> TrainEpochStats:
    """
    Run one training epoch with gradient norm tracking.

    For each batch, the L2 norm of the full gradient vector is computed
    after loss.backward() and before optimizer.step(). This enables
    post-hoc analysis of per-epoch gradient statistics (mean and variance),
    which is used in the optimisation dynamics analysis (Gap 3).

    Parameters
    ----------
    model : nn.Module
        Model to train (set to train mode internally).
    loader : DataLoader
        DataLoader yielding (images, labels) batches.
    optimizer : torch.optim.Optimizer
        Optimiser (typically AdamW with differential learning rates
        for backbone and head in end-to-end training).
    criterion : nn.Module
        Loss function (class-weighted CrossEntropyLoss for DermaMNIST).
    device : torch.device
        Device to run training on.

    Returns
    -------
    TrainEpochStats
        Loss, accuracy, and gradient norm statistics for the epoch.
    """
    model.train()

    total_loss = 0.0
    n_samples = 0
    n_correct = 0
    grad_norms: List[float] = []

    for batch in loader:
        x, y = _batch_to_xy(batch)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()

        # Compute L2 gradient norm across all parameters
        sq_sum = 0.0
        for p in model.parameters():
            if p.grad is not None:
                g = p.grad.detach()
                sq_sum += float(torch.sum(g * g).item())
        grad_norms.append(float(np.sqrt(sq_sum)))

        optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        n_samples += bs
        pred = torch.argmax(logits, dim=1)
        n_correct += int((pred == y).sum().item())

    grad_norms_np = np.asarray(grad_norms, dtype=np.float64)

    return TrainEpochStats(
        loss=total_loss / max(n_samples, 1),
        accuracy=n_correct / max(n_samples, 1),
        grad_norm_mean=float(grad_norms_np.mean()) if grad_norms_np.size else 0.0,
        grad_norm_var=float(grad_norms_np.var(ddof=0)) if grad_norms_np.size else 0.0,
    )
