"""
Maximal Marginal Relevance (MMR) re-ranking.

Ensures diversity in recommendations so we don't suggest
"5 types of Coke" (Section 2.3.3 of project spec).
"""

from typing import Any, Dict, List, Optional, Set

import numpy as np


def mmr_rerank(
    query_vec: np.ndarray,
    candidate_vecs: np.ndarray,
    candidate_info: List[Dict[str, Any]],
    lambda_param: float = 0.7,
    top_n: int = 5,
    exclude_ids: Optional[Set[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Production-optimized MMR re-ranking.

    For each iteration, selects the candidate that maximizes:
        MMR = λ · Sim(query, candidate) − (1−λ) · max(Sim(candidate, already_selected))

    Args:
        query_vec: Query/intent vector (D,)
        candidate_vecs: Candidate embeddings (N, D)
        candidate_info: Metadata dicts for each candidate
        lambda_param: Relevance-diversity tradeoff (0 = diverse, 1 = relevant)
        top_n: Number of results to return
        exclude_ids: Set of item_ids to exclude (already in cart)

    Returns:
        List of re-ranked recommendation dicts with mmr_score and relevance_score
    """
    if exclude_ids is None:
        exclude_ids = set()

    n_candidates = len(candidate_info)
    if n_candidates == 0:
        return []

    # Normalize vectors
    q_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)
    c_norms = candidate_vecs / (np.linalg.norm(candidate_vecs, axis=1, keepdims=True) + 1e-8)

    # Relevance scores
    relevance = c_norms @ q_norm  # (N,)

    # Filter excluded items
    valid_mask = np.array([
        info.get("item_id", "") not in exclude_ids
        for info in candidate_info
    ])

    selected: List[Dict[str, Any]] = []
    selected_vecs: List[np.ndarray] = []

    for _ in range(min(top_n, int(valid_mask.sum()))):
        best_score = -np.inf
        best_idx = -1

        for i in range(n_candidates):
            if not valid_mask[i]:
                continue

            rel = relevance[i]

            # Diversity penalty
            if selected_vecs:
                sel_matrix = np.array(selected_vecs)
                div_penalty = float(np.max(sel_matrix @ c_norms[i]))
            else:
                div_penalty = 0.0

            score = lambda_param * rel - (1 - lambda_param) * div_penalty

            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx == -1:
            break

        valid_mask[best_idx] = False
        selected_vecs.append(c_norms[best_idx])

        result = {
            **candidate_info[best_idx],
            "mmr_score": round(float(best_score), 4),
            "relevance_score": round(float(relevance[best_idx]), 4),
        }
        selected.append(result)

    return selected
