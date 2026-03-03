"""IR evaluation metrics — NDCG, Precision, Recall, AUC."""

import math

import numpy as np
from sklearn.metrics import roc_auc_score


class RecommenderMetrics:
    """Information Retrieval metrics for CSAO evaluation."""

    @staticmethod
    def ndcg_at_k(scores: np.ndarray, labels: np.ndarray, k: int = 10) -> float:
        sorted_indices = np.argsort(-scores)[:k]
        sorted_labels = labels[sorted_indices]
        dcg = sum(label / math.log2(i + 2) for i, label in enumerate(sorted_labels))
        ideal_sorted = np.sort(labels)[::-1][:k]
        idcg = sum(label / math.log2(i + 2) for i, label in enumerate(ideal_sorted))
        return dcg / idcg if idcg > 0 else 0.0

    @staticmethod
    def precision_at_k(scores: np.ndarray, labels: np.ndarray, k: int = 10) -> float:
        top_k_indices = np.argsort(-scores)[:k]
        return float(labels[top_k_indices].sum() / k)

    @staticmethod
    def recall_at_k(scores: np.ndarray, labels: np.ndarray, k: int = 10) -> float:
        total_relevant = labels.sum()
        if total_relevant == 0:
            return 0.0
        top_k_indices = np.argsort(-scores)[:k]
        return float(labels[top_k_indices].sum() / total_relevant)

    @staticmethod
    def compute_auc(scores: np.ndarray, labels: np.ndarray) -> float:
        try:
            return float(roc_auc_score(labels, scores))
        except ValueError:
            return 0.5
