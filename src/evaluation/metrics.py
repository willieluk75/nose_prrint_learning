import numpy as np
from sklearn.metrics import roc_curve
from typing import Tuple


def compute_roc(
    similarities: np.ndarray,
    labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """計算 ROC 曲線。labels: 1=same pet, 0=different"""
    return roc_curve(labels, similarities)


def compute_eer(
    similarities: np.ndarray,
    labels: np.ndarray,
) -> Tuple[float, float]:
    """
    計算 Equal Error Rate (EER) 和對應閾值。
    EER 是 FAR = FRR 時的錯誤率，越低越好。
    """
    fpr, tpr, thresholds = compute_roc(similarities, labels)
    fnr = 1.0 - tpr
    eer_idx = np.argmin(np.abs(fpr - fnr))
    eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)
    return eer, float(thresholds[eer_idx])
