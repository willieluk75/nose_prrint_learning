import numpy as np


def compute_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """
    計算兩個 L2-normalized embedding 的 cosine similarity。
    因為向量已 normalize，cosine similarity = dot product。
    Returns: float in [-1.0, 1.0]
    """
    return float(np.dot(emb1, emb2))


def is_same_pet(
    emb1: np.ndarray,
    emb2: np.ndarray,
    threshold: float = 0.85,
) -> bool:
    """相似度超過閾值則視為同一隻寵物"""
    return compute_similarity(emb1, emb2) >= threshold
