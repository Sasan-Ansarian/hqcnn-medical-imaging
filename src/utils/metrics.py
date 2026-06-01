from dataclasses import dataclass
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


@dataclass
class EvalResult:
    loss: float
    accuracy: float
    macro_f1: float
    conf_mat: np.ndarray


def compute_metrics(y_true, y_pred) -> tuple[float, float, np.ndarray]:
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred)
    return acc, f1, cm
