"""Query steering via SAE feature directions (the SLIDERS mechanism)."""

import numpy as np


def steer_query(
    query_emb: np.ndarray,
    directions: np.ndarray,
    alphas: list[float],
) -> np.ndarray:
    """Shift a query embedding along a set of SAE feature directions.

    Computes:
        q' = query_emb + sum_i(alpha_i * direction_i)

    then L2-normalises q' so it remains on the unit hypersphere.

    Args:
        query_emb: Base query embedding, shape ``(D,)``.
        directions: Matrix of feature directions, shape ``(n_sliders, D)``.
            Each row should be a unit vector (the i-th SAE decoder column).
        alphas: Scalar coefficient for each feature direction.  Positive
            values pull the query towards images with high activation on
            that feature; negative values do the opposite.

    Returns:
        L2-normalised steered query, shape ``(D,)``.

    Raises:
        ValueError: If ``len(alphas) != directions.shape[0]``.
    """
    if len(alphas) != directions.shape[0]:
        raise ValueError(
            f"len(alphas)={len(alphas)} must equal "
            f"directions.shape[0]={directions.shape[0]}"
        )

    alphas_arr = np.asarray(alphas, dtype=np.float32)
    delta = (alphas_arr[:, None] * directions).sum(axis=0)
    steered = query_emb.astype(np.float32) + delta

    norm = np.linalg.norm(steered)
    if norm < 1e-8:
        return query_emb.astype(np.float32)
    return steered / norm
