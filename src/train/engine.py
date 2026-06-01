from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@dataclass
class TrainEpochStats:
    loss: float
    accuracy: float
    grad_norm_mean: float
    grad_norm_var: float


@dataclass
class EvalEpochStats:
    loss: float
    accuracy: float
    macro_f1: float
    balanced_accuracy: float
    conf_mat: np.ndarray


def _batch_to_xy(batch) -> Tuple[torch.Tensor, torch.Tensor]:
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

    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, balanced_accuracy_score

    avg_loss = total_loss / max(n_samples, 1)
    acc = accuracy_score(all_true, all_pred)
    f1 = f1_score(all_true, all_pred, average="macro")
    bacc = balanced_accuracy_score(all_true, all_pred)
    cm = confusion_matrix(all_true, all_pred)

    return EvalEpochStats(
        loss=avg_loss,
        accuracy=acc,
        macro_f1=f1,
        balanced_accuracy=bacc,
        conf_mat=cm,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> TrainEpochStats:
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

        sq_sum = 0.0
        for p in model.parameters():
            if p.grad is not None:
                g = p.grad.detach()
                sq_sum += float(torch.sum(g * g).item())
        grad_norm = float(np.sqrt(sq_sum))
        grad_norms.append(grad_norm)

        optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        n_samples += bs

        pred = torch.argmax(logits, dim=1)
        n_correct += int((pred == y).sum().item())

    avg_loss = total_loss / max(n_samples, 1)
    acc = n_correct / max(n_samples, 1)

    grad_norms_np = np.asarray(grad_norms, dtype=np.float64)
    grad_mean = float(grad_norms_np.mean()) if grad_norms_np.size else 0.0
    grad_var = float(grad_norms_np.var(ddof=0)) if grad_norms_np.size else 0.0

    return TrainEpochStats(
        loss=avg_loss,
        accuracy=float(acc),
        grad_norm_mean=grad_mean,
        grad_norm_var=grad_var,
    )
